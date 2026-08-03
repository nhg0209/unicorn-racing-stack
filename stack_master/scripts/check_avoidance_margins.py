#!/usr/bin/env python3
"""Consistency check for the coupled avoidance margins (double-avoidance prevention).

The re-optimized global line must clear each obstacle by MORE than the reactive planner's
keep-out, or the re-opt line sits inside the reactive keep-out and the reactive layer
re-avoids the already-handled obstacle every lap (hump on top of hump, OVERTAKE flip-flop).

    reactive keep-out  = width_car/2 + safety_margin   (static_avoidance_params.yaml)
    re-opt clearance   = reopt_obs_margin              (base_system.launch.xml -> static_reopt_node)

THIRD chain member — the state machine's GB free-check: for the swapped line to read FREE
(GB_TRACK holds, no phantom TRAILING), the clearance the re-opt line is ENFORCED to must exceed
the SM's static requirement, and the whole chain must be ordered:

    SM requirement        =  gb_ego_width_m/2 + lateral_width_static_gb_m   (state_machine_params)
    clear-gate stay       =  width_car/2 + clear_margin_m                   (static_avoidance_params)
    enforced laid floor   =  reopt_obs_margin                               (base_system.launch.xml)

    SM requirement  <=  clear-gate stay  <=  enforced floor - slack

Between the first two thresholds sits a DEAD BAND: an obstacle clearing by more than the planner
needs to idle but less than the SM needs to call the line free leaves both sides standing down,
i.e. TRAILING behind a line that already drives around the obstacle. The re-opt's drift triggers
(reopt_obs_change_tol, reopt_clearance_dirty) must fit inside the headroom between the floor and
the SM requirement, or an obstacle can drift the followed line under the requirement with nothing
left to react. See check_static_chain_ordering().

FOURTH chain — the DYNAMIC overtaking path (lane_change_planner <-> SM free-check). The
planner mirrors the SM's clearance inputs and derives its solver target and monitor abort line
from them; if a mirror drifts, the planner publishes a lane the SM rejects and the car flaps
OVERTAKE<->TRAILING with neither side giving up. Also checks the horizon/engage-window ordering
the two nodes depend on. See check_dynamic_chain().

Run after tuning any side:  python3 stack_master/scripts/check_avoidance_margins.py
Exit code 0 = consistent, 1 = violation.
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

SLACK = 0.03  # [m] margin the re-opt clearance must exceed the reactive keep-out by

STACK_MASTER = Path(__file__).resolve().parents[1]


def load_reactive_params():
    p = STACK_MASTER / "config" / "static_avoidance_params.yaml"
    cfg = yaml.safe_load(p.read_text())["static_avoidance_planner"]["ros__parameters"]
    return p, cfg


def load_launch_args():
    p = STACK_MASTER / "launch" / "base_system.launch.xml"
    root = ET.parse(p).getroot()
    args = {a.get("name"): a.get("default") for a in root.iter("arg")}
    return p, args


def check_launch_agreement() -> bool:
    """race.launch.xml and base_system.launch.xml must agree on every reopt_* default.

    This exists because of a real miss: reopt_wall_margin was 0.05 in base_system.launch.xml (and
    in the node) but 0.12 in race.launch.xml, which is the entry point the runbook and every sim
    run actually use. The rest of this script only reads base_system, so it reported the safe 0.05
    and everyone believed it. Measured cost of that 7 cm on ifac: the corridor fit shrank the
    avoidance hump, worst-case lap loss went +0.49 -> +1.47 s and the laid line exceeded curvlim.

    Also flags reopt_* args base_system declares but race.launch.xml never forwards — those are
    silently un-tunable from the documented entry point (reopt_obs_margin was one).
    """
    base_p = STACK_MASTER / "launch" / "base_system.launch.xml"
    race_p = STACK_MASTER / "launch" / "race.launch.xml"
    if not race_p.is_file():
        return True
    base_root, race_root = ET.parse(base_p).getroot(), ET.parse(race_p).getroot()
    base_args = {a.get("name"): a.get("default") for a in base_root.iter("arg")
                 if (a.get("name") or "").startswith("reopt")}
    # <arg name=.. default=..> declarations vs <arg name=.. value=..> forwards inside <include>
    race_args = {a.get("name"): a.get("default") for a in race_root.iter("arg")
                 if (a.get("name") or "").startswith("reopt") and a.get("default") is not None}
    race_fwd = {a.get("name") for a in race_root.iter("arg")
                if (a.get("name") or "").startswith("reopt") and a.get("value") is not None}

    print(f"\n--- launch agreement ({race_p.name} <-> {base_p.name}) ---")
    ok = True
    for name in sorted(set(base_args) & set(race_args)):
        b, r = base_args[name], race_args[name]
        try:
            same = abs(float(b) - float(r)) < 1e-9
        except (TypeError, ValueError):
            same = str(b) == str(r)
        if not same:
            ok = False
            print(f"FAIL: {name} default is {r} in {race_p.name} but {b} in {base_p.name}. "
                  f"{race_p.name} wins for every documented run, so {b} is a fiction.")
    missing = sorted(n for n in base_args if n not in race_fwd)
    if missing:
        ok = False
        print(f"FAIL: {race_p.name} never forwards {', '.join(missing)} — un-tunable from the "
              f"entry point the runbook uses. Add <arg name=.. value=..> to the include.")
    if ok:
        print(f"OK: all {len(base_args)} reopt_* defaults agree and are forwarded.")
    return ok


def load_sm_params():
    p = STACK_MASTER / "config" / "state_machine_params.yaml"
    cfg = yaml.safe_load(p.read_text())["state_machine"]["ros__parameters"]
    return p, cfg


def load_lane_change_params():
    p = STACK_MASTER / "config" / "lane_change_params.yaml"
    cfg = yaml.safe_load(p.read_text())["planner_change"]["ros__parameters"]
    return p, cfg


def load_dynamic_planner_params():
    # state_machine/config/planners/ lives in a sibling package
    p = STACK_MASTER.parent / "state_machine" / "config" / "planners" / "dynamic_avoidance_planner.yaml"
    return p, yaml.safe_load(p.read_text())


def load_reopt_node_defaults():
    """{param: default} from static_reopt_node's own declare_parameter calls (numeric only).

    base_system.launch.xml wins whenever the node is launched from it, but `ros2 run` and any
    launch file that forgets an arg get these instead — so a node default that violates the
    margin chain is a real trap, not a formality.
    """
    p = (STACK_MASTER.parent / "planner" / "gb_optimizer" / "gb_optimizer" / "static_reopt_node.py")
    if not p.is_file():
        return p, {}
    pat = re.compile(r'self\.declare_parameter\(\s*"([A-Za-z_0-9]+)"\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)')
    return p, {m.group(1): float(m.group(2)) for m in pat.finditer(p.read_text())}


def load_reopt_arg_bindings():
    """{node param name: launch arg name} from base_system's static_reopt_node <param> elements.

    Read from the launch file rather than hardcoded, so a renamed arg cannot silently unhook a
    parameter from the checks below.
    """
    p = STACK_MASTER / "launch" / "base_system.launch.xml"
    root = ET.parse(p).getroot()
    out = {}
    for node in root.iter("node"):
        if node.get("exec") != "static_reopt_node":
            continue
        for prm in node.iter("param"):
            m = re.fullmatch(r"\$\(var\s+([A-Za-z_0-9]+)\)", (prm.get("value") or "").strip())
            if m:
                out[prm.get("name")] = m.group(1)
    return out


def check_static_chain_ordering(sm_path, sm, cfg, yaml_path, args, launch_path) -> bool:
    """The STATIC chain's ordering + drift rules — the three that were only ever prose.

    Chain, all measured line-centre to obstacle EDGE:

        SM static GB requirement  <=  planner clear-gate STAY threshold  <=  enforced floor - slack

    (a) is checked against the enforced floor by the caller. Here:
    (b) ORDERING. The SM requirement must not exceed the threshold at which the reactive planner
        stops planning. Above it there is a band where the SM calls the line blocked while the
        planner, correctly, publishes nothing to overtake with -- TRAILING behind a line that
        already clears the obstacle, with no way out. This was 0.35 vs 0.25 on the car.
    (c) DRIFT. reopt_obs_change_tol is how far an obstacle may move before the re-opt is
        re-triggered, so in the worst case the followed line's clearance decays by that much
        before anything reacts. It therefore has to fit inside the headroom between the enforced
        floor and the SM requirement. clearance_dirty_m, the consequence trigger, must sit inside
        the same band: below the floor (or it fires on a line that is fine) and at or above the
        SM requirement (or it fires only after the SM has already given up).
    """
    node_path, node_def = load_reopt_node_defaults()
    bindings = load_reopt_arg_bindings()

    gb_ego_half = float(sm["gb_ego_width_m"]) / 2.0
    static_gb = float(sm.get("lateral_width_static_gb_m", sm["lateral_width_gb_m"]))
    required = gb_ego_half + static_gb
    width_car = float(cfg["width_car"])
    clear_stay = width_car / 2.0 + float(cfg.get("clear_margin_m", 0.10))
    floor = float(args["reopt_obs_margin"])
    headroom = floor - required

    print(f"\n--- static chain ordering ({sm_path.name} <-> {yaml_path.name} <-> {launch_path.name}) ---")
    ok = True

    # (b) ordering
    if required > clear_stay + 1e-9:
        ok = False
        print(f"FAIL: SM requirement ({required:.3f}) > clear-gate stay threshold ({clear_stay:.3f}). "
              f"In [{clear_stay:.3f}, {required:.3f}) the SM reads the line blocked while the "
              f"planner idles -> TRAILING deadlock behind a cleared line. Lower "
              f"lateral_width_static_gb_m or raise clear_margin_m.")
    else:
        print(f"OK: SM requirement ({required:.3f}) <= clear-gate stay ({clear_stay:.3f}) "
              f"<= enforced floor ({floor:.3f}); dead band {clear_stay - required:.3f} m.")

    # (c) drift budget, applied to BOTH the launch value and the node's own default
    print(f"  headroom for drift = floor - SM requirement = {floor:.3f} - {required:.3f} = {headroom:.3f} m")
    for src, tol in (("launch", float(args["reopt_obs_change_tol"])),
                     ("node default", node_def.get("obs_change_tol"))):
        if tol is None:
            continue
        if tol > headroom + 1e-9:
            ok = False
            print(f"FAIL: obs_change_tol ({src}) = {tol:.3f} > headroom {headroom:.3f}. An obstacle "
                  f"may drift that far without re-triggering the re-opt, taking the followed line "
                  f"under the SM requirement with nothing left to react.")
        else:
            print(f"OK: obs_change_tol ({src}) = {tol:.3f} <= headroom {headroom:.3f}.")

    dirty = args.get("reopt_clearance_dirty", node_def.get("clearance_dirty_m"))
    if dirty is not None:
        dirty = float(dirty)
        if not (required - 1e-9 <= dirty <= floor + 1e-9):
            ok = False
            print(f"FAIL: clearance_dirty_m ({dirty:.3f}) is outside [SM requirement {required:.3f}, "
                  f"floor {floor:.3f}]. Below the requirement it re-optimizes only after the SM has "
                  f"already dropped to TRAILING; above the floor it re-optimizes a line that is fine.")
        else:
            print(f"OK: clearance_dirty_m ({dirty:.3f}) inside [{required:.3f}, {floor:.3f}].")

    # launch vs node-default disagreement: reported, not fatal (the rules above are applied to
    # both), because a deliberate launch override is normal — a SILENT one is not.
    if node_def and bindings:
        diffs = []
        for prm, arg in sorted(bindings.items()):
            nd, la = node_def.get(prm), args.get(arg)
            if nd is None or la is None:
                continue
            try:
                if abs(nd - float(la)) > 1e-9:
                    diffs.append(f"{prm}: node {nd:g} vs {arg} {float(la):g}")
            except (TypeError, ValueError):
                continue
        if diffs:
            print(f"NOTE: {node_path.name} defaults differ from {launch_path.name} "
                  f"({len(diffs)}): {'; '.join(diffs)}. Launched from base_system the launch value "
                  f"wins; `ros2 run` gets the node value.")
        else:
            print(f"OK: every bound reopt_* arg matches the node's own default.")
    return ok


def check_dynamic_chain(sm_path, sm) -> bool:
    """The DYNAMIC overtaking chain: lane_change_planner <-> SM free-check.

    The planner derives its clearance numbers from MIRRORS of the SM's parameters
    (_apply_margins in change_avoidance_node.py). Mirrors drift the moment someone tunes the
    SM side alone, and the failure is silent: the planner keeps publishing a lane the SM
    rejects and the car flaps OVERTAKE<->TRAILING. This section is the offline guard.
    """
    lc_path, lc = load_lane_change_params()
    dyn_path, dyn = load_dynamic_planner_params()

    print(f"\n--- dynamic chain ({lc_path.name} <-> {dyn_path.name}) ---")
    ok = True

    # 1. mirrors must equal the real values
    mirrors = [
        ("sm_gb_ego_width_m", float(lc["sm_gb_ego_width_m"]), float(sm["gb_ego_width_m"]), sm_path.name),
        ("sm_lateral_width_m", float(lc["sm_lateral_width_m"]), float(dyn["lateral_width_m"]), dyn_path.name),
    ]
    for name, mirrored, actual, src in mirrors:
        if abs(mirrored - actual) > 1e-9:
            ok = False
            print(f"FAIL: {name} = {mirrored:.3f} in {lc_path.name} but the real value is "
                  f"{actual:.3f} in {src}.")
            print("      The planner is sizing its lane against a stale copy of the SM's rule.")
        else:
            print(f"OK: {name} mirrors {src} ({actual:.3f}).")

    # 2. derived margins must bracket the SM accept line correctly.
    #    Compare what the planner will ACTUALLY compute at runtime (from its mirrors, which is
    #    all _apply_margins has access to) against what the SM ACTUALLY requires (from the real
    #    yamls). When the mirrors are correct these coincide; when they have drifted, this is
    #    what makes the resulting inversion visible instead of hiding it behind corrected numbers.
    sm_required = float(sm["gb_ego_width_m"]) / 2.0 + float(dyn["lateral_width_m"])
    planner_assumes = float(lc["sm_gb_ego_width_m"]) / 2.0 + float(lc["sm_lateral_width_m"])
    sep_margin = planner_assumes + float(lc["sep_slack_m"])
    sep_monitor = planner_assumes + float(lc["sep_monitor_slack_m"])
    print(f"  SM accept line (actual)     = {sm_required:.3f} m")
    print(f"  planner assumes             = {planner_assumes:.3f} m")
    print(f"  -> solver target sep_margin = {sep_margin:.3f} | monitor abort = {sep_monitor:.3f}")
    if sep_margin < sm_required:
        ok = False
        print(f"FAIL: solver target ({sep_margin:.3f}) < SM accept line ({sm_required:.3f}); "
              f"every lane would be rejected downstream. Raise sep_slack_m.")
    if sep_monitor < sm_required:
        ok = False
        print(f"FAIL: monitor abort line ({sep_monitor:.3f}) < SM accept line ({sm_required:.3f}); "
              f"the planner would hold a lane the SM rejects -> OVERTAKE<->TRAILING flapping. "
              f"Raise sep_monitor_slack_m to >= 0.")
    if sep_monitor > sep_margin:
        ok = False
        print(f"FAIL: monitor abort ({sep_monitor:.3f}) > solver target ({sep_margin:.3f}); "
              f"the monitors would abort a lane the solver just produced.")
    if ok:
        print(f"OK: monitor ({sep_monitor:.3f}) >= SM accept line ({sm_required:.3f}) and "
              f"<= solver target ({sep_margin:.3f}) -- the planner bails before the SM does.")

    # 3. horizon ordering: the published path must outreach the SM's obstacle window, or the
    #    free-check's `gap > max_gap` branch marks every obstacle beyond the path as blocking.
    hold_horizon = float(lc["hold_horizon_m"])
    interest = float(sm["interest_horizon_m"])
    if hold_horizon <= interest:
        ok = False
        print(f"FAIL: hold_horizon_m ({hold_horizon:.1f}) <= interest_horizon_m ({interest:.1f}); "
              f"obstacles inside the SM's window but past the published path read NOT-free.")
    else:
        print(f"OK: hold_horizon_m ({hold_horizon:.1f}) > interest_horizon_m ({interest:.1f}).")

    # 4. the planner must engage INSIDE the SM's commit window, or it prepares a lane the SM
    #    will not act on (and _check_getting_closer keeps answering NO).
    engage_gap = float(lc["engage_gap_m"])
    if engage_gap >= 10.0:  # _check_overtaking_mode passes threshold_m=10.0
        ok = False
        print(f"FAIL: engage_gap_m ({engage_gap:.1f}) >= the SM's getting_closer window (10.0).")
    else:
        print(f"OK: engage_gap_m ({engage_gap:.1f}) < the SM's getting_closer window (10.0).")

    return ok


def main() -> int:
    yaml_path, cfg = load_reactive_params()
    launch_path, args = load_launch_args()

    width_car = float(cfg["width_car"])
    safety_margin = float(cfg["safety_margin"])
    apex_bulge = float(cfg.get("apex_bulge", 0.0))
    reactive_wall = float(cfg.get("wall_margin", 0.0))
    keepout = width_car / 2.0 + safety_margin

    obs_margin = float(args["reopt_obs_margin"])
    reopt_wall = float(args.get("reopt_wall_margin", 0.0))
    qp_veh_width = float(args.get("reopt_qp_veh_width", 0.0))
    reopt_safety_width = float(args.get("reopt_safety_width", 0.0))

    print(f"reactive ({yaml_path.name}):")
    print(f"  width_car/2 + safety_margin = {width_car/2:.3f} + {safety_margin:.3f} = {keepout:.3f} m (keep-out)")
    print(f"  apex_bulge = {apex_bulge:.3f} m, wall_margin = {reactive_wall:.3f} m")
    print(f"re-opt ({launch_path.name} defaults; node defaults apply only if launched without these args):")
    print(f"  reopt_obs_margin = {obs_margin:.3f} m, reopt_wall_margin = {reopt_wall:.3f} m")
    print(f"  reopt_qp_veh_width = {qp_veh_width:.3f} m, reopt_safety_width = {reopt_safety_width:.3f} m")

    ok = True
    if obs_margin < keepout + SLACK:
        ok = False
        print(f"\nFAIL: reopt_obs_margin ({obs_margin:.3f}) < reactive keep-out + slack "
              f"({keepout:.3f} + {SLACK:.2f} = {keepout + SLACK:.3f}).")
        print("      The re-optimized line will be re-avoided by the reactive planner every lap.")
        print(f"      Raise reopt_obs_margin in {launch_path.name} or lower the reactive keep-out.")
    else:
        print(f"\nOK: reopt_obs_margin ({obs_margin:.3f}) >= reactive keep-out + slack ({keepout + SLACK:.3f}).")

    # Wall reserves compared like-for-like: reopt reserve = qp_veh_width/2 + wall_margin vs the
    # reactive planner's corridor reserve = width_car/2 (its bound_ok check). The reopt reserve
    # must not be LARGER than what the reactive apex was driven at by more than the slack, or the
    # corridor fit rejects reactive-proven apexes (all-or-nothing) and those obstacles never make
    # it into the re-opt line.
    reopt_reserve = qp_veh_width / 2.0 + reopt_wall
    react_reserve = width_car / 2.0 + reactive_wall
    if reopt_reserve < width_car / 2.0:
        print(f"NOTE: reopt wall reserve ({reopt_reserve:.3f}) < half car ({width_car/2:.3f}) — "
              f"the re-opt line may hug walls closer than the car physically fits.")
    if reopt_reserve > react_reserve + SLACK:
        ok = False
        print(f"FAIL: reopt wall reserve ({reopt_reserve:.3f}) > reactive terminal reserve + slack "
              f"({react_reserve:.3f} + {SLACK:.2f}) — reactive-proven apexes will be corridor-"
              f"rejected; lower reopt_wall_margin in {launch_path.name}.")
    else:
        print(f"OK: reopt wall reserve ({reopt_reserve:.3f}) <= reactive terminal reserve + slack "
              f"({react_reserve + SLACK:.3f}).")

    # reopt_fit_tol is spent OUT OF reopt_wall_margin: the corridor fit is allowed to overshoot the
    # bound by fit_tol, so the effective wall reserve is reopt_wall_margin - fit_tol. Without a
    # tolerance the fit collapses the hump over sub-mm ripples in the smoothed bound (that was the
    # bug); with too much, it eats the wall reserve it is measured against.
    fit_tol = float(args.get("reopt_fit_tol", 0.0))
    print(f"  reopt_fit_tol = {fit_tol:.4f} m -> effective wall reserve "
          f"{reopt_reserve - fit_tol:.3f} m")
    if fit_tol <= 0.0:
        ok = False
        print(f"FAIL: reopt_fit_tol is {fit_tol:.4f}. A zero-tolerance corridor fit rejects every "
              f"wide reach over a sub-millimetre violation and collapses the avoidance hump "
              f"(ifac: reach 5.00 -> 1.24 m, +1.62 s/lap, curvlim exceeded). Use ~0.005.")
    elif fit_tol > 0.5 * max(reopt_wall, 1e-9):
        ok = False
        print(f"FAIL: reopt_fit_tol ({fit_tol:.4f}) > half of reopt_wall_margin ({reopt_wall:.3f}); "
              f"the fit may spend most of the wall reserve. Lower it or raise reopt_wall_margin.")

    # --- the raceline-CLEAR gate vs the line static_reopt actually builds -------------------
    # The gate decides whether the current global line already avoids everything ahead. Its
    # threshold must sit BELOW the clearance the re-opt line is built to, with room for the
    # detection/frame error, or a correctly-cleared obstacle reads as on-path: the planner then
    # tries to re-clear it at the full obs_margin inside a corridor the line already used, rejects
    # every candidate, publishes feasible=False and the SM falls to TRAILING behind an obstacle it
    # had already solved. It used obs_margin (0.30) + clear_hyst_m, leaving 2 cm against the 0.35 m
    # re-opt build -- which is what made this fail on the car.
    clear_margin = float(cfg.get("clear_margin_m", 0.10))
    clear_hyst = float(cfg.get("clear_hyst_m", 0.0))
    clear_need = width_car / 2.0 + clear_margin + clear_hyst
    print(f"\nraceline-clear gate ({yaml_path.name}):")
    print(f"  need = width_car/2 + clear_margin_m + clear_hyst_m = {width_car/2:.3f} + "
          f"{clear_margin:.3f} + {clear_hyst:.3f} = {clear_need:.3f} m")
    print(f"  re-opt line is built to reopt_obs_margin = {obs_margin:.3f} m")
    if clear_need >= obs_margin - SLACK:
        ok = False
        print(f"FAIL: clear-gate need ({clear_need:.3f}) is not at least {SLACK:.2f} m below the "
              f"re-opt build ({obs_margin:.3f}). The obstacle-aware line will re-trigger the "
              f"reactive planner on obstacles it already clears -> all candidates rejected -> "
              f"TRAILING. Lower clear_margin_m or raise reopt_obs_margin.")
    elif clear_need < width_car / 2.0:
        ok = False
        print(f"FAIL: clear-gate need ({clear_need:.3f}) is below half the car ({width_car/2:.3f}); "
              f"the gate would call a line clear that the car does not physically fit through.")
    else:
        print(f"OK: clear-gate need ({clear_need:.3f}) sits {obs_margin - clear_need:.3f} m below "
              f"the re-opt build and >= half car ({width_car/2:.3f}).")

    # --- chain member 3: SM GB free-check vs the clearance the re-opt line is ENFORCED to ---
    sm_path, sm = load_sm_params()
    gb_ego_half = float(sm["gb_ego_width_m"]) / 2.0
    static_gb = float(sm.get("lateral_width_static_gb_m", sm["lateral_width_gb_m"]))
    required = gb_ego_half + static_gb
    # The floor is now ENFORCED, not aspirational: static_reopt_core accepts a hump only if the
    # geometry it lays clears the box edge by obs_margin, retries it once wider, and otherwise
    # drops it for the reactive layer (reason "clearance"); static_reopt_node then re-measures the
    # published line and refuses to publish one that misses an obstacle it claims to have
    # reshaped. So the guaranteed worst case of any PUBLISHED hump is obs_margin itself.
    #
    # This used to compare against keep-out + apex_bulge -- what the reactive apex was DRIVEN at,
    # which the old amplitude-ratio acceptance then let through at 0.90x. That number was never a
    # floor: on a 0.55 m apex it admitted 0.345 m, below the 0.35 the rest of this script assumes,
    # and the check passed anyway because it compared the intent rather than the acceptance rule.
    laid_floor = obs_margin
    print(f"\nstate machine ({sm_path.name}):")
    print(f"  gb_ego_width/2 + lateral_width_static_gb_m = {gb_ego_half:.3f} + {static_gb:.3f} = {required:.3f} m")
    print(f"  ENFORCED worst-case clearance of a published hump = reopt_obs_margin = {laid_floor:.3f} m")
    if laid_floor < required + SLACK:
        ok = False
        print(f"FAIL: enforced laid floor ({laid_floor:.3f}) < SM static GB requirement + slack "
              f"({required:.3f} + {SLACK:.2f} = {required + SLACK:.3f}).")
        print("      The swapped line reads BLOCKED to the SM -> phantom TRAILING + re-avoidance.")
        print(f"      Lower lateral_width_static_gb_m in {sm_path.name} or raise reopt_obs_margin.")
    else:
        print(f"OK: enforced laid floor ({laid_floor:.3f}) >= SM static GB requirement + slack "
              f"({required + SLACK:.3f}).")
    # The RECORDED reactive apex is the hump's starting amplitude. If the reactive layer drives
    # narrower than the floor, no hump reaches it on its own and every one depends on the core's
    # widen-once retry finding corridor room -- which a tight track may not have.
    reactive_reach = keepout + apex_bulge
    if reactive_reach < laid_floor:
        print(f"NOTE: the recorded reactive apex only reaches keep-out + apex_bulge = "
              f"{reactive_reach:.3f} m, under the {laid_floor:.3f} m floor. Every hump then relies "
              f"on the core widening it; on a tight corridor those apexes get dropped instead. "
              f"Raise apex_bulge or lower reopt_obs_margin.")

    if not check_static_chain_ordering(sm_path, sm, cfg, yaml_path, args, launch_path):
        ok = False
    if not check_dynamic_chain(sm_path, sm):
        ok = False
    if not check_launch_agreement():
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
