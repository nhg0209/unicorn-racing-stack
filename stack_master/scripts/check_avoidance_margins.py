#!/usr/bin/env python3
"""Consistency check for the coupled avoidance margins (double-avoidance prevention).

The re-optimized global line must clear each obstacle by MORE than the reactive planner's
keep-out, or the re-opt line sits inside the reactive keep-out and the reactive layer
re-avoids the already-handled obstacle every lap (hump on top of hump, OVERTAKE flip-flop).

    reactive keep-out  = width_car/2 + safety_margin   (static_avoidance_params.yaml)
    re-opt clearance   = reopt_obs_margin              (base_system.launch.xml -> static_reopt_node)

THIRD chain member — the state machine's GB free-check: for the swapped line to read FREE
(GB_TRACK holds, no phantom TRAILING), the line's actual box-edge clearance (keep-out +
apex_bulge, the recorded reactive apex the re-opt line passes through) must also exceed the
SM's static requirement:

    keep-out + apex_bulge  >=  gb_ego_width_m/2 + lateral_width_static_gb_m + slack
                               (state_machine_params.yaml)

FOURTH chain — the DYNAMIC overtaking path (lane_change_planner <-> SM free-check). The
planner mirrors the SM's clearance inputs and derives its solver target and monitor abort line
from them; if a mirror drifts, the planner publishes a lane the SM rejects and the car flaps
OVERTAKE<->TRAILING with neither side giving up. Also checks the horizon/engage-window ordering
the two nodes depend on. See check_dynamic_chain().

Run after tuning any side:  python3 stack_master/scripts/check_avoidance_margins.py
Exit code 0 = consistent, 1 = violation.
"""
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

    # --- chain member 3: SM GB free-check vs the swapped line's actual clearance -----------
    sm_path, sm = load_sm_params()
    gb_ego_half = float(sm["gb_ego_width_m"]) / 2.0
    static_gb = float(sm.get("lateral_width_static_gb_m", sm["lateral_width_gb_m"]))
    line_clearance = keepout + apex_bulge          # box-edge clearance of the obstacle-aware line
    required = gb_ego_half + static_gb
    print(f"\nstate machine ({sm_path.name}):")
    print(f"  gb_ego_width/2 + lateral_width_static_gb_m = {gb_ego_half:.3f} + {static_gb:.3f} = {required:.3f} m")
    print(f"  obstacle-aware line box-edge clearance (keep-out + apex_bulge) = {line_clearance:.3f} m")
    if line_clearance < required + SLACK:
        ok = False
        print(f"FAIL: line clearance ({line_clearance:.3f}) < SM static GB requirement + slack "
              f"({required:.3f} + {SLACK:.2f} = {required + SLACK:.3f}).")
        print("      The swapped line reads BLOCKED to the SM -> phantom TRAILING + re-avoidance.")
        print(f"      Lower lateral_width_static_gb_m in {sm_path.name} or raise apex_bulge.")
    else:
        print(f"OK: line clearance ({line_clearance:.3f}) >= SM static GB requirement + slack ({required + SLACK:.3f}).")

    if not check_dynamic_chain(sm_path, sm):
        ok = False
    if not check_launch_agreement():
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
