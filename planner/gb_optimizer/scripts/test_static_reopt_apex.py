#!/usr/bin/env python3
"""Unit tests for static_reopt_node's apex bookkeeping (no sim, no map files, no rclpy init).

Covers the lap-drift / late-swap fixes:
  1. a NEW apex sets the dirty flag (rebuild trigger no longer depends on the obstacle set
     changing after the apex exists),
  2. sub-5cm apex growth updates the record but does NOT re-trigger a solve,
  3. retro association: an obstacle confirmed AFTER the pass recovers its apex from the
     recent-path buffer,
  4. the commit gate demands agreement over the whole look-ahead horizon (wrap-aware), not
     just at the car's current station.

Run:  python3 planner/gb_optimizer/scripts/test_static_reopt_apex.py
(needs the workspace sourced OR PYTHONPATH pointing at src gb_optimizer + f110 deps)
"""
import sys
import types
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gb_optimizer.static_reopt_node import StaticReoptNode  # noqa: E402
from gb_optimizer import static_reopt_core as core          # noqa: E402


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    warn = warning


class _Clock:
    def __init__(self): self.t = 0.0
    def now(self): return types.SimpleNamespace(nanoseconds=int(self.t * 1e9))


def make_node():
    """Bare instance: only the state the apex/commit code paths touch."""
    n = StaticReoptNode.__new__(StaticReoptNode)
    n._clock = _Clock()
    n.get_logger = lambda: _Logger()
    n.get_clock = lambda: n._clock
    n._obstacles = []
    n._obs_ids = []
    n._apex_by_obs = {}
    n._apex_miss = {}
    n._apex_assoc_tol = 2.0
    n._apex_min_d = 0.05
    n._obstacles_dirty = False
    n._dirty_since = 0.0
    n._reactive_active = False
    n._reactive_idle_t = 0.0
    n.swap_idle_s = 0.3
    n.swap_horizon_min_m = 3.0
    n.swap_horizon_time_s = 1.0
    n._last_vs = 5.0
    n.apex_buffer_sec = 3.0
    from collections import deque
    n._path_buffer = deque()
    n._track_len = 40.0
    n._pending = None
    n._pending_dev = None
    n._pending_since = 0.0
    n.swap_deadlock_s = 5.0
    n.swap_deadlock_max_vs = 2.0
    n.swap_deadlock_max_dev = 0.6
    n.clean_bundle = object()
    n._notify_scaler_ticks = 0
    n.notify_ticks = 0
    # straight clean line along x with a 0.7 m corridor each side (apex plausibility check)
    n._clean_xy = np.column_stack([np.arange(0.0, 40.0, 0.1), np.zeros(400)])
    n._clean_dr = np.full(400, 0.7)
    n._clean_dl = np.full(400, 0.7)
    return n


def path_msg(points):
    """OTWpntArray stand-in: list of (x, y, d)."""
    wps = [types.SimpleNamespace(x_m=x, y_m=y, d_m=d) for x, y, d in points]
    return types.SimpleNamespace(wpnts=wps)


HUMP = [(x * 0.5, 0.4 * np.exp(-((x * 0.5 - 5.0) ** 2)), 0.4 * np.exp(-((x * 0.5 - 5.0) ** 2)))
        for x in range(20)]   # apex d=0.4 at x=5.0, y=0.4


def straight_bundle():
    """A straight active line along x (y=0), stations every 0.1 m — matches the clean line."""
    sa = np.arange(0.0, 40.0, 0.1)
    wp = [types.SimpleNamespace(s_m=s, x_m=s, y_m=0.0) for s in sa]
    return types.SimpleNamespace(glb_wpnts=types.SimpleNamespace(wpnts=wp))


def test_new_apex_sets_dirty():
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n.otwpnts_cb(path_msg(HUMP))
    assert ("id", 7) in n._apex_by_obs, "apex not recorded"
    assert n._obstacles_dirty, "new apex must arm the rebuild trigger"
    print("PASS new apex sets dirty")


def test_small_growth_no_retrigger():
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n.otwpnts_cb(path_msg(HUMP))
    n._obstacles_dirty = False   # pretend the solve consumed it
    grown = [(x, y * 1.05, d * 1.05) for x, y, d in HUMP]   # apex 0.40 -> 0.42 (<5cm)
    n.otwpnts_cb(path_msg(grown))
    # 0.42 exceeds the obstacle-required clearance (0.40) -> clamped back; either way the
    # change is sub-5cm, so no re-solve is triggered.
    assert abs(n._apex_by_obs[("id", 7)][2] - 0.40) < 0.02, "record must stay at the clamp"
    assert not n._obstacles_dirty, "sub-5cm change must not re-trigger a solve"
    print("PASS sub-5cm apex growth does not re-trigger")


def test_retro_association():
    n = make_node()
    # avoidance driven while NO obstacle was confirmed yet -> only the buffer keeps it
    n.otwpnts_cb(path_msg(HUMP))
    assert len(n._path_buffer) == 1 and not n._apex_by_obs
    # obstacle confirmed after the pass: obstacles_cb runs the retro replay
    from visualization_msgs.msg import Marker, MarkerArray
    m = Marker()
    m.action = Marker.ADD
    m.id = 9
    m.pose.position.x, m.pose.position.y = 5.0, -0.2
    m.scale.x = m.scale.y = 0.3
    msg = MarkerArray(); msg.markers = [m]
    n.default_obs_radius = 0.15
    n.obs_change_tol = 0.10
    n.apex_miss_frames = 20
    n.obstacles_cb(msg)
    assert ("id", 9) in n._apex_by_obs, "retro association failed"
    assert abs(n._apex_by_obs[("id", 9)][2] - 0.4) < 0.05
    assert n._obstacles_dirty
    print("PASS retro association from the path buffer")


def test_newest_wins_shrink():
    # keep-the-max RATCHETED (one outlier path permanently inflated the hump to 1.4 m on a
    # 1.39 m track); the record must follow the NEWEST qualifying path, shrink included.
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n.otwpnts_cb(path_msg(HUMP))                     # apex 0.40
    n._obstacles_dirty = False
    smaller = [(x, y * 0.5, d * 0.5) for x, y, d in HUMP]   # apex 0.20
    n.otwpnts_cb(path_msg(smaller))
    assert abs(n._apex_by_obs[("id", 7)][2] - 0.20) < 0.03, "newest path must replace the max"
    assert n._obstacles_dirty, ">5cm shrink is a rebuild-worthy change"
    print("PASS newest path wins (ratchet removed)")


def test_implausible_apex_rejected():
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    crazy = [(x * 0.5, 1.4 * np.exp(-((x * 0.5 - 5.0) ** 2)),
              1.4 * np.exp(-((x * 0.5 - 5.0) ** 2))) for x in range(20)]   # apex 1.4 > corridor
    n.otwpnts_cb(path_msg(crazy))
    assert ("id", 7) not in n._apex_by_obs, "apex outside the drivable band must be rejected"
    print("PASS implausible (out-of-corridor) apex rejected")


def test_overshoot_apex_clamped():
    # Steering slip while riding the hump anchors the replanned path at the DISPLACED car —
    # its widest point exceeds the required clearance. The record must clamp to what the
    # obstacle needs (d_obs + r + 0.45), not what the car happened to drive.
    n = make_node()
    n._clean_dr = np.full(400, 0.9)   # wide-side corridor (ifac reaches ~1.2): the overshoot
    n._clean_dl = np.full(400, 0.9)   # is PLAUSIBLE (inside the band) but beyond the need
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]   # need = -0.2 + (0.15+0.45) = +0.40
    n._obs_ids = [7]
    wide = [(x, y * 1.5, d * 1.5) for x, y, d in HUMP]   # driven apex 0.60 (overshoot)
    n.otwpnts_cb(path_msg(wide))
    rec = n._apex_by_obs[("id", 7)]
    assert abs(rec[2] - 0.40) < 0.02, f"overshoot must clamp to the required clearance, got {rec[2]:.2f}"
    print("PASS overshoot apex clamped to obstacle-required clearance")


def test_neighbor_ramp_does_not_overwrite():
    # The scenario that "forgot obstacle 1": after o1's avoidance, paths toward o2 sweep their
    # ramp within association range of o1 with a small decaying d — newest-wins then overwrote
    # o1's good apex. The ABEAM guard must keep o1's record; o2 records (and clamps) its own.
    n = make_node()
    n._clean_dr = np.full(400, 0.9); n._clean_dl = np.full(400, 0.9)
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n.otwpnts_cb(path_msg(HUMP))                     # o1 apex 0.40 recorded
    assert abs(n._apex_by_obs[("id", 7)][2] - 0.40) < 0.02
    # o2 appears 3 m past o1; the o2-avoidance path's entry ramp passes ~1.6 m from o1
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15), core.Obstacle(8.0, -0.2, 0.15)]
    n._obs_ids = [7, 8]
    o2_path = [(x, 0.55 * np.exp(-((x - 8.0) / 2.0) ** 2), 0.55 * np.exp(-((x - 8.0) / 2.0) ** 2))
               for x in np.arange(3.0, 13.0, 0.5)]
    n.otwpnts_cb(path_msg(o2_path))
    assert abs(n._apex_by_obs[("id", 7)][2] - 0.40) < 0.02, \
        "o1's apex must survive the neighbouring ramp (abeam guard)"
    assert ("id", 8) in n._apex_by_obs and abs(n._apex_by_obs[("id", 8)][2] - 0.40) < 0.02, \
        "o2 must record its own (clamped) apex"
    print("PASS neighbouring ramp does not overwrite a good apex")


def test_orphan_apex_adopted_on_id_reissue():
    # Layer unlatch->re-confirm re-issues marker ids; the record must follow the obstacle,
    # or the next rebuild silently drops that obstacle's hump.
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n.otwpnts_cb(path_msg(HUMP))
    assert ("id", 7) in n._apex_by_obs
    from visualization_msgs.msg import Marker, MarkerArray
    def mk(mid, x, y):
        m = Marker(); m.action = Marker.ADD; m.id = mid
        m.pose.position.x, m.pose.position.y = x, y
        m.scale.x = m.scale.y = 0.3
        return m
    msg = MarkerArray(); msg.markers = [mk(9, 5.0, -0.2), mk(10, 20.0, -0.2)]
    n.default_obs_radius = 0.15
    n.obs_change_tol = 0.10
    n.apex_miss_frames = 20
    n.obstacles_cb(msg)                              # id 7 -> re-issued as 9, plus a new obstacle
    assert ("id", 9) in n._apex_by_obs and abs(n._apex_by_obs[("id", 9)][2] - 0.40) < 0.02, \
        "orphaned record must be adopted by the re-issued id"
    assert ("id", 7) not in n._apex_by_obs
    print("PASS orphaned apex adopted on track-id re-issue")


def test_breaker_refuses_poisoned_pending():
    n = make_node()
    sa = np.arange(0.0, 40.0, 0.1)
    wp = [types.SimpleNamespace(s_m=s, x_m=s, y_m=0.0) for s in sa]
    n._pending = types.SimpleNamespace(glb_wpnts=types.SimpleNamespace(wpnts=wp))
    dev = np.zeros_like(sa); dev[(sa > 10.0) & (sa < 18.0)] = 0.9   # poisoned: 0.9 m at the car
    n._pending_dev = dev
    n._reactive_active = True
    n._last_vs = 0.5
    n._clock.t = 10.0
    sentinel = straight_bundle()
    n.active = sentinel
    n._publish_active = lambda b: None
    n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)
    n._commit_pending(12.0)
    assert n._pending is None, "poisoned pending must be discarded"
    assert n.active is sentinel, "and must NOT be committed"
    print("PASS breaker refuses a poisoned pending (discard, no commit)")


def test_set_change_drops_stale_pending():
    # A pending bundle built from the PREVIOUS obstacle state must not survive a set change:
    # it blocked the fresh rebuild (solve gate needs _pending None) and would commit an
    # outdated line (observed after a spurious unlatch->re-confirm flap).
    n = make_node()
    n._pending = object()
    n._pending_dev = np.zeros(3)
    from visualization_msgs.msg import Marker, MarkerArray
    m = Marker()
    m.action = Marker.ADD
    m.id = 3
    m.pose.position.x, m.pose.position.y = 5.0, -0.2
    m.scale.x = m.scale.y = 0.3
    msg = MarkerArray(); msg.markers = [m]
    n.default_obs_radius = 0.15
    n.obs_change_tol = 0.10
    n.apex_miss_frames = 20
    n.obstacles_cb(msg)
    assert n._pending is None and n._pending_dev is None, "stale pending must be discarded"
    assert n._obstacles_dirty
    print("PASS set change drops the stale pending bundle")


def test_commit_horizon():
    n = make_node()
    sa = np.arange(0.0, 40.0, 0.1)
    xa = sa.copy(); ya = np.zeros_like(sa)
    bundle = types.SimpleNamespace()
    wp = [types.SimpleNamespace(s_m=s, x_m=x, y_m=y) for s, x, y in zip(sa, xa, ya)]
    bundle.glb_wpnts = types.SimpleNamespace(wpnts=wp)
    n._pending = bundle
    n.active = straight_bundle()
    # deviation hump 2 m ahead of s=10 (inside the 5 m horizon at vs=5)
    dev = np.zeros_like(sa); dev[(sa > 12.0) & (sa < 16.0)] = 0.3
    n._pending_dev = dev
    n._reactive_active = False
    n._reactive_idle_t = -10.0
    n._publish_active = lambda b: None
    n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)
    n.notify_ticks = 0
    n._commit_pending(10.0)
    assert n._pending is not None, "must NOT commit with a hump inside the horizon"
    n._commit_pending(20.0)      # hump behind, horizon [20,25] clean
    assert n._pending is None, "must commit once the horizon is clean"
    print("PASS commit gate respects the look-ahead horizon")


def test_deadlock_breaker():
    # Car stuck trailing INSIDE the pending hump: horizon gate blocked (dev ahead), reactive
    # active (idle gate blocked). After swap_deadlock_s at low speed the commit must force.
    def stuck_node(vs):
        n = make_node()
        sa = np.arange(0.0, 40.0, 0.1)
        wp = [types.SimpleNamespace(s_m=s, x_m=s, y_m=0.0) for s in sa]
        n._pending = types.SimpleNamespace(glb_wpnts=types.SimpleNamespace(wpnts=wp))
        n.active = straight_bundle()
        dev = np.zeros_like(sa); dev[(sa > 10.0) & (sa < 18.0)] = 0.4   # hump around the car
        n._pending_dev = dev
        n._reactive_active = True                 # planner flickering on the obstacle ahead
        n._last_vs = vs
        n._pending_since = 0.0
        n._clock.t = 10.0                         # pending has waited 10 s > 5 s
        n._publish_active = lambda b: None
        n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)
        return n

    n = stuck_node(vs=0.5)
    n._commit_pending(12.0)                       # car inside the hump
    assert n._pending is None, "slow + long-waiting pending must force-commit (un-stick)"
    n = stuck_node(vs=5.0)
    n._commit_pending(12.0)
    assert n._pending is not None, "at speed the normal gates must still hold"
    print("PASS swap deadlock breaker (slow+stale commits, at-speed does not)")


def test_commit_horizon_wrap():
    n = make_node()
    sa = np.arange(0.0, 40.0, 0.1)
    wp = [types.SimpleNamespace(s_m=s, x_m=s, y_m=0.0) for s in sa]
    n._pending = types.SimpleNamespace(glb_wpnts=types.SimpleNamespace(wpnts=wp))
    n.active = straight_bundle()
    dev = np.zeros_like(sa); dev[sa < 2.0] = 0.3          # hump right past the seam
    n._pending_dev = dev
    n._reactive_active = False
    n._reactive_idle_t = -10.0
    n._publish_active = lambda b: None
    n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)
    n.notify_ticks = 0
    n._commit_pending(38.0)      # horizon [38, 40)+[0, 3) wraps into the hump
    assert n._pending is not None, "wrap-around horizon must see the seam hump"
    n._commit_pending(20.0)
    assert n._pending is None
    print("PASS commit horizon is wrap-aware")


if __name__ == "__main__":
    test_new_apex_sets_dirty()
    test_small_growth_no_retrigger()
    test_retro_association()
    test_newest_wins_shrink()
    test_implausible_apex_rejected()
    test_overshoot_apex_clamped()
    test_neighbor_ramp_does_not_overwrite()
    test_orphan_apex_adopted_on_id_reissue()
    test_set_change_drops_stale_pending()
    test_commit_horizon()
    test_commit_horizon_wrap()
    test_deadlock_breaker()
    test_breaker_refuses_poisoned_pending()
    print("ALL PASS")
