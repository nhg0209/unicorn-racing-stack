#!/usr/bin/env python3
"""Harness test of WHEN the state machine adopts a freshly published avoidance path.

There used to be two writers of the same cache with different policies: update_waypoints, which
deliberately KEEPS a committed path through small changes, and _check_latest_wpnts, which adopted
whatever arrived as a side effect of a freshness check. Which one won depended on the transition
path taken that cycle, so the followed geometry could change without any of the rules that govern
it firing.

Adoption is now update_waypoints alone, under three rules:
  (i)   a genuinely NEW requirement -- more offset, or the opposite side,
  (ii)  a forward RE-SLICE of the same world-fixed geometry, adopted every cycle so the window's
        start follows the car,
  (iii) the cached path reads NOT-free and the fresh one is free.

Run (after sourcing the workspace):
  python3 state_machine/test/test_path_adoption.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_machine.state_machine_node import StateMachine, WaypointData   # noqa: E402

TRACK_LEN = 40.0
WPNT_DIST = 0.1


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def error(self, *a, **k): pass


def path_msg(s0, s1, apex_s, apex_d, stamp=100.0):
    """A hump: d rises to apex_d at apex_s and back, over [s0, s1]."""
    s = np.arange(s0, s1, WPNT_DIST)
    d = apex_d * np.exp(-((s - apex_s) ** 2) / 2.0)
    wp = [types.SimpleNamespace(x_m=float(v), y_m=float(dd), s_m=float(v), d_m=float(dd))
          for v, dd in zip(s, d)]
    return types.SimpleNamespace(
        wpnts=wp,
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(
            sec=int(stamp), nanosec=int((stamp % 1) * 1e9))))


def cache_from(msg, name="static_avoidance_planner"):
    wd = WaypointData.__new__(WaypointData)
    wd.name = name
    wd.list, wd.array, wd.stamp, wd.is_init = [], None, None, False
    wd.latest_threshold = 1.0
    wd.closest_target, wd.closest_gap = None, None
    wd.last_used_sec = None
    if msg is not None:
        wd.initialize_traj(msg)
    return wd


def sm(now=100.0):
    n = StateMachine.__new__(StateMachine)
    n.name = "state_machine"
    n.get_logger = lambda: _Log()
    n._t = now
    n.now_sec = lambda: n._t
    n.track_length = TRACK_LEN
    n.max_s = TRACK_LEN
    n.wpnt_dist = WPNT_DIST
    n.reslice_d_tol_m = 0.03
    n.reslice_end_s_tol_m = 0.20
    n.obstacles_in_interest = []
    n.cur_obstacles_in_interest = []
    n.recovery_wpnts = None
    n.cur_recovery_wpnts = cache_from(None, "recovery_planner")
    # global line: untouched by these tests
    gb = path_msg(0.0, TRACK_LEN, 0.0, 0.0)
    n.gb_wpnts = gb
    n.cur_gb_wpnts = cache_from(gb, "global_tracking")
    n.avoidance_wpnts = None
    n.static_avoidance_wpnts = None
    n.cur_static_avoidance_wpnts = cache_from(None)
    n.cur_avoidance_wpnts = cache_from(None, "dynamic_avoidance_planner")
    n._check_free_frenet = lambda wd: True         # overridden per test
    return n


def test_check_latest_is_pure():
    n = sm()
    fresh = path_msg(5.0, 15.0, 10.0, 0.5)
    cur = cache_from(path_msg(5.0, 15.0, 10.0, 0.5))
    before = cur.array.copy()
    n._check_on_spline = lambda wd: True
    assert n._check_latest_wpnts(fresh, cur) is True
    assert np.array_equal(cur.array, before), "the freshness check must not adopt anything"
    print("PASS _check_latest_wpnts is a pure predicate")


def test_rule_i_new_requirement():
    n = sm()
    n.cur_static_avoidance_wpnts = cache_from(path_msg(5.0, 15.0, 10.0, 0.40))
    # a SMALLER same-side peak is the planner easing back off a box already passed -> keep
    n.static_avoidance_wpnts = path_msg(5.0, 15.0, 10.0, 0.20)
    n.update_waypoints()
    assert abs(max(w.d_m for w in n.cur_static_avoidance_wpnts.list) - 0.40) < 1e-6, "must keep"
    # a BIGGER one is a new requirement -> adopt
    n.static_avoidance_wpnts = path_msg(5.0, 15.0, 10.0, 0.60)
    n.update_waypoints()
    assert abs(max(w.d_m for w in n.cur_static_avoidance_wpnts.list) - 0.60) < 1e-6, "must adopt"
    print("PASS rule (i): a bigger offset is adopted, a smaller same-side one is not")


def test_rule_ii_forward_reslice_follows_the_car():
    # THE freeze: the planner republishes the same world-fixed path, trimmed to what is still ahead.
    # Nothing about the requirement changed, so rule (i) keeps the cache -- and the cached START
    # stays where the car was seconds ago.
    n = sm()
    full = path_msg(5.0, 15.0, 10.0, 0.5)
    n.cur_static_avoidance_wpnts = cache_from(full)
    starts = []
    for car_s in (6.0, 8.0, 11.0, 13.0):       # note 11.0 and 13.0 are PAST the apex
        n.static_avoidance_wpnts = path_msg(car_s, 15.0, 10.0, 0.5)
        n.update_waypoints()
        starts.append(n.cur_static_avoidance_wpnts.list[0].s_m)
    assert starts == [6.0, 8.0, 11.0, 13.0], f"the cached start must follow the car: {starts}"
    print(f"PASS rule (ii): the cached start follows the car ({starts}), apex or not")


def test_rule_ii_rejects_a_genuinely_different_path():
    n = sm()
    n.cur_static_avoidance_wpnts = cache_from(path_msg(5.0, 15.0, 10.0, 0.5))
    # same peak and same start, but the path now ENDS somewhere else -> a new plan, not a re-slice
    n.static_avoidance_wpnts = path_msg(5.0, 18.0, 10.0, 0.5)
    assert n._is_forward_reslice(n.static_avoidance_wpnts, n.cur_static_avoidance_wpnts) is False
    # same extent, but the geometry between them differs by more than the tolerance
    n.static_avoidance_wpnts = path_msg(5.0, 15.0, 10.0, 0.45)
    assert n._is_forward_reslice(n.static_avoidance_wpnts, n.cur_static_avoidance_wpnts) is False
    print("PASS rule (ii) rejects a moved end station and a changed shape")


def test_rule_iii_blocked_cache_yields_to_a_free_path():
    n = sm()
    cached = path_msg(5.0, 15.0, 10.0, 0.50)
    fresh = path_msg(5.0, 15.0, 10.0, 0.45)     # smaller peak: rules (i) and (ii) both say KEEP
    n.cur_static_avoidance_wpnts = cache_from(cached)
    n.static_avoidance_wpnts = fresh
    # cached reads blocked, fresh reads free
    n._check_free_frenet = lambda wd: abs(max(w.d_m for w in wd.list)) < 0.48
    n.update_waypoints()
    assert abs(max(w.d_m for w in n.cur_static_avoidance_wpnts.list) - 0.45) < 1e-6, \
        "a free path must win over a held path the free-check rejects"
    # ...and when the cached one is fine, the same publish is still kept
    n2 = sm()
    n2.cur_static_avoidance_wpnts = cache_from(cached)
    n2.static_avoidance_wpnts = fresh
    n2._check_free_frenet = lambda wd: True
    n2.update_waypoints()
    assert abs(max(w.d_m for w in n2.cur_static_avoidance_wpnts.list) - 0.50) < 1e-6, \
        "a free cached path must still be kept"
    print("PASS rule (iii): a blocked cache yields to a free fresh path, otherwise it is kept")


def test_recovery_still_gets_adopted():
    # recovery had no other writer than the (now pure) freshness check
    n = sm()
    n.cur_static_avoidance_wpnts = cache_from(None)
    n.recovery_wpnts = path_msg(5.0, 15.0, 10.0, 0.3)
    n.update_waypoints()
    assert n.cur_recovery_wpnts.is_init, "recovery must still be adopted somewhere"
    print("PASS recovery is adopted by the single writer too")


if __name__ == "__main__":
    test_check_latest_is_pure()
    test_rule_i_new_requirement()
    test_rule_ii_forward_reslice_follows_the_car()
    test_rule_ii_rejects_a_genuinely_different_path()
    test_rule_iii_blocked_cache_yields_to_a_free_path()
    test_recovery_still_gets_adopted()
    print("ALL PASS")
