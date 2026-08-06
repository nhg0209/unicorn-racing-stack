#!/usr/bin/env python3
"""Harness test of which AEB threshold the controller picks.

AEB_for_weird_local_wpnt clamps the speed command to 2.0 m/s when the nearest local waypoint is
implausibly far from the car -- a dead/garbage-planner net. An avoidance line is legitimately far
from the car, so it gets a looser threshold, and the question is how "an avoidance line" is
recognised.

It used to be `state == "OVERTAKE"`, which is the wrong question: the state machine hands the SAME
avoidance geometry to the controller under TRAILING too (holding the avoidance reference through a
drop, and on the trailing approach). There the 0.5 m bar fired against a path that was never meant
to sit under the car, and fired and released repeatedly -- a sawtooth on the speed command.

Run:
  python3 controller/test/test_aeb_threshold.py
"""
import types
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parents[1] / "controller" / "combined" / "src" / "Controller.py"
ctl = types.ModuleType("ctl")
ctl.__dict__["__file__"] = str(MOD)
exec(compile(MOD.read_text(), str(MOD), "exec"), ctl.__dict__)


def controller():
    n = ctl.Controller.__new__(ctl.Controller)
    n.AEB_thres, n.AEB_thres_overtake, n.AEB_offline_d_thres = 0.5, 0.9, 0.1
    # the clamp is LATCHED: hysteresis on release, plus a minimum hold counted in cycles
    n.AEB_release_hyst_m, n.AEB_min_hold_s, n.loop_rate = 0.25, 0.2, 50.0
    n._aeb_engaged, n._aeb_cycles = False, 0
    n.logger_warn = lambda *a, **k: None
    n.position_in_map = np.array([[0.0, 0.0, 0.0]])
    n.idx_nearest_waypoint = 0
    return n


def fires(state, max_d, nearest_dist, speed=6.0):
    """True when the AEB clamps."""
    n = controller()
    n.state = state
    arr = np.zeros((5, 9))
    arr[:, 8] = max_d                       # column 8 is d_m
    arr[0, 0] = nearest_dist                # nearest waypoint this far from the car at the origin
    n.waypoint_array_in_map = arr
    return n.AEB_for_weird_local_wpnt(speed) < speed


def test_offset_geometry_is_recognised_in_any_state():
    # 0.70 m: past AEB_thres (0.5), inside AEB_thres_overtake (0.9)
    assert not fires("OVERTAKE", max_d=0.55, nearest_dist=0.70)
    assert not fires("TRAILING", max_d=0.55, nearest_dist=0.70), \
        "the same avoidance geometry must get the same threshold under TRAILING"
    assert not fires("GB_TRACK", max_d=0.55, nearest_dist=0.70), \
        "...and under any other state the SM may be in while handing it over"
    print("PASS an offset local window gets the loose threshold whatever the state is called")


def test_the_tight_threshold_still_guards_the_raceline():
    assert fires("GB_TRACK", max_d=0.00, nearest_dist=0.70), \
        "a raceline window 0.70 m from the car is still a garbage path"
    assert fires("OVERTAKE", max_d=0.02, nearest_dist=0.70), \
        "...including when the state says OVERTAKE but the geometry is on the raceline"
    print("PASS a raceline window keeps the tight threshold, state notwithstanding")


def test_a_truly_dead_planner_still_clamps():
    assert fires("TRAILING", max_d=0.55, nearest_dist=1.00), \
        "the loose threshold must still catch a genuinely dead planner"
    assert fires("OVERTAKE", max_d=0.55, nearest_dist=1.00)
    print("PASS the loose threshold still catches a genuinely dead planner")


def drive(dists, max_d=0.0, speed=6.0):
    """One controller through a sequence of nearest-waypoint distances; returns the commands."""
    n = controller()
    n.state = "GB_TRACK"
    out = []
    for dist in dists:
        arr = np.zeros((5, 9))
        arr[:, 8] = max_d
        arr[0, 0] = dist
        n.waypoint_array_in_map = arr
        out.append(n.AEB_for_weird_local_wpnt(speed))
    return out


def test_the_clamp_is_latched_against_jitter_around_the_bar():
    # THE reason for the latch. A bare threshold turns any wobble around the bar into a per-cycle
    # engage/release: the docstring records 2.87 -> 2.00 -> 3.24 in two cycles on the real car,
    # and 86.8 m/s^2 of published acceleration. Here the measured distance dithers by 1 cm across
    # the 0.5 m bar.
    dists = [0.49, 0.51, 0.49, 0.51, 0.49, 0.51, 0.49, 0.51]
    out = drive(dists)
    edges = sum(1 for a, b in zip(out, out[1:]) if (a < 6.0) != (b < 6.0))
    assert edges <= 1, f"the clamp toggled {edges} times on 1 cm of jitter: {out}"
    print(f"PASS 1 cm of jitter across the bar produces {edges} clamp edge(s), not one per cycle")


def test_release_needs_the_hysteresis_and_the_hold():
    n_cycles_hold = int(round(0.2 * 50.0))
    # engage, then sit just under the bar but inside the hysteresis band -> stays engaged
    out = drive([0.60] + [0.40] * 10)
    assert all(v < 6.0 for v in out), f"released inside the hysteresis band: {out}"
    # ...and below the band it releases, but only after the minimum hold
    out = drive([0.60] + [0.20] * (n_cycles_hold + 2))
    assert out[1] < 6.0, "released before the minimum hold"
    assert out[-1] == 6.0, f"never released below the band: {out[-3:]}"
    print(f"PASS release needs both 0.25 m of hysteresis and {n_cycles_hold} cycles of hold")


def test_the_clamp_never_raises_the_command():
    # `return 2.0` was an assignment, so an emergency brake ACCELERATED a car already below 2 m/s.
    out = drive([0.60], speed=1.2)
    assert out[0] == 1.2, f"the clamp raised the command from 1.2 to {out[0]}"
    print("PASS the clamp is an upper bound, not an assignment")


def test_the_offset_test_is_the_documented_one():
    # exactly at the boundary the tight threshold still applies (strict >)
    assert fires("TRAILING", max_d=0.10, nearest_dist=0.70)
    assert not fires("TRAILING", max_d=0.11, nearest_dist=0.70)
    print("PASS the offset test is max|d_m| > AEB_offline_d_thres")


if __name__ == "__main__":
    test_offset_geometry_is_recognised_in_any_state()
    test_the_tight_threshold_still_guards_the_raceline()
    test_a_truly_dead_planner_still_clamps()
    test_the_clamp_is_latched_against_jitter_around_the_bar()
    test_release_needs_the_hysteresis_and_the_hold()
    test_the_clamp_never_raises_the_command()
    test_the_offset_test_is_the_documented_one()
    print("ALL PASS")
