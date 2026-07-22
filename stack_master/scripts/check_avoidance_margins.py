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


def load_sm_params():
    p = STACK_MASTER / "config" / "state_machine_params.yaml"
    cfg = yaml.safe_load(p.read_text())["state_machine"]["ros__parameters"]
    return p, cfg


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

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
