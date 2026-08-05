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
import threading
import time
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
    # the node's gates swallow their own exceptions into debug(); without it a bug inside a gate
    # surfaces as "no attribute 'debug'" and hides what actually failed
    def debug(self, *a, **k): pass
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
    n.swap_deadlock_max_dev_commit_m = 0.25
    n.clean_bundle = straight_bundle()   # a real bundle: _line_dev/_bundle_xy read its wpnts
    n._notify_scaler_ticks = 0
    n.notify_ticks = 0
    n.obs_margin = 0.35
    n.fit_tol = 0.005
    n.clearance_dirty_m = 0.30
    n._clearance_dirty_keys = set()
    n.apex_major_change_m = 0.10
    n.solve_min_interval_s = 1.0
    n.swap_block_trailing = True
    n.swap_min_obs_gap_m = 3.0
    n.swap_state_stale_s = 1.0
    n._sm_state = ""
    n._sm_state_t = -1e9
    n._last_solve_t = -1e9
    n._solve_backoff_until = 0.0
    n._last_s = None
    n._s_progressed = 0.0
    n.obs_change_tol = 0.05
    n.apex_span_margin_m = 0.5
    n.apex_undershoot_m = 0.12
    n.relax_floor = 0.30            # the bottom rung of the core's coverage ladder
    n.side_hint_margin_m = 0.15
    n.obs_forget_s = 1.5
    n.swap_min_gain_m = 0.05
    n._solve_epoch = 0
    n._solve_future = None
    n._solve_ctx = None
    n._last_seen = {}
    n.map_filter = None             # no map in the harness -> _grid_room falls back
    n._swap_block = {}
    n._swap_lap_t = 0.0
    n.apex_abeam_gap_m = 0.5
    n._apex_change_major = False
    n.active = straight_bundle()      # real node always has one (clean bundle at startup)
    n.pub_coverage = types.SimpleNamespace(publish=lambda m: None)
    # straight clean line along x with a 0.7 m corridor each side (apex plausibility check)
    n._clean_xy = np.column_stack([np.arange(0.0, 40.0, 0.1), np.zeros(400)])
    n._clean_dr = np.full(400, 0.7)
    n._clean_dl = np.full(400, 0.7)
    return n


def idle_msg():
    """The reactive layer publishing nothing = the maneuver is over. The apex-driven rebuild is
    armed on that transition (see otwpnts_cb), so a test that records an apex must also deliver
    the idle edge to see the trigger fire."""
    return types.SimpleNamespace(wpnts=[])


def drive(n, points):
    """One avoidance maneuver: publish the path, then go idle."""
    n.otwpnts_cb(path_msg(points))
    n.otwpnts_cb(idle_msg())


def squeeze_msg(points):
    """A path the planner could only solve at reduced margins (ot_line = 'squeeze')."""
    m = path_msg(points)
    m.ot_line = "squeeze"
    return m


def path_msg(points):
    """OTWpntArray stand-in: list of (x, y, d)."""
    wps = [types.SimpleNamespace(x_m=x, y_m=y, d_m=d) for x, y, d in points]
    return types.SimpleNamespace(wpnts=wps, ot_line="")


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
    drive(n, HUMP)
    assert ("id", 7) in n._apex_by_obs, "apex not recorded"
    assert n._obstacles_dirty, "new apex must arm the rebuild trigger once the maneuver ends"
    print("PASS new apex sets dirty")


def test_apex_does_not_arm_the_rebuild_mid_maneuver():
    # The reactive layer republishes at 20 Hz WHILE the car drives the avoidance. Arming the
    # rebuild on each of those discards the pending bundle over and over, so the swap the records
    # are being collected for can never land. The trigger waits for the maneuver to end.
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    for _ in range(5):
        n.otwpnts_cb(path_msg(HUMP))                   # still avoiding
    assert ("id", 7) in n._apex_by_obs, "the record still updates live"
    assert not n._obstacles_dirty, "no rebuild may be armed while the maneuver is running"
    n.otwpnts_cb(idle_msg())                           # maneuver over
    assert n._obstacles_dirty, "the deferred trigger must fire on the idle edge"
    print("PASS new apex sets dirty")


def test_small_growth_no_retrigger():
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    drive(n, HUMP)
    n._obstacles_dirty = False   # pretend the solve consumed it
    grown = [(x, y * 1.05, d * 1.05) for x, y, d in HUMP]   # apex 0.40 -> 0.42 (<5cm)
    drive(n, grown)
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


def test_newest_wins_shrink_within_the_undershoot_bound():
    # keep-the-max RATCHETED (one outlier path permanently inflated the hump to 1.4 m on a
    # 1.39 m track), so the record follows the NEWEST qualifying path, shrink included -- but
    # "qualifying" now bounds the shrink too. The obstacle here needs |d| = 0.40; a path offering
    # 0.32 is a real, slightly tighter avoidance, while one offering 0.20 is a decaying tail and
    # recording it would centre the next hump on a clearance the obstacle does not have.
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    drive(n, HUMP)                                   # apex 0.40 == the geometric need
    n._obstacles_dirty = False
    smaller = [(x, y * 0.8, d * 0.8) for x, y, d in HUMP]   # apex 0.32: 8 cm under, inside 0.12
    drive(n, smaller)
    assert abs(n._apex_by_obs[("id", 7)][2] - 0.32) < 0.03, "a real shrink must still be adopted"
    assert n._obstacles_dirty, ">5cm shrink is a rebuild-worthy change"
    tail = [(x, y * 0.5, d * 0.5) for x, y, d in HUMP]      # apex 0.20: 20 cm under -> a tail
    drive(n, tail)
    assert abs(n._apex_by_obs[("id", 7)][2] - 0.32) < 0.03, \
        "a record that undershoots the geometric need must be rejected, not adopted"
    print("PASS newest path wins, but only within the undershoot bound")


def test_apex_span_requirement():
    # The reactive planner republishes the slice still AHEAD of the car, so once the car is beside
    # a box the republished path is that box's EXIT RAMP -- widest point downstream of the box.
    # Under newest-wins those tails walked the record up to 1.0 m downstream and the hump built on
    # it missed by 3-12 cm. A path that genuinely passed the box starts before it and ends after it.
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n.otwpnts_cb(path_msg(HUMP))                       # spans x = 0 .. 9.5, obstacle at 5.0
    assert ("id", 7) in n._apex_by_obs, "a path that spans the obstacle must be accepted"
    n._apex_by_obs.clear()
    # the same geometry, sliced to start AT the obstacle: an exit ramp, nothing before the box
    tail = [(x, y, d) for x, y, d in HUMP if x >= 5.0]
    n.otwpnts_cb(path_msg(tail))
    assert ("id", 7) not in n._apex_by_obs, "an exit ramp must not set the record"
    # ...and sliced to END at the obstacle: an entry ramp, nothing after it
    n._apex_by_obs.clear()
    head = [(x, y, d) for x, y, d in HUMP if x <= 5.0]
    n.otwpnts_cb(path_msg(head))
    assert ("id", 7) not in n._apex_by_obs, "an entry ramp must not set the record either"
    print("PASS an apex is only believed from a path that spans the obstacle")


def test_apex_is_anchored_abeam_not_where_the_path_was_widest():
    # The record's STATION is what centres the hump. It is now taken from the obstacle, so a path
    # whose widest point drifted downstream can be wrong about the amplitude but never about where.
    n = make_node()
    o = core.Obstacle(5.0, -0.2, 0.15)
    n._obstacles = [o]
    n._obs_ids = [7]
    # a hump whose peak sits 0.30 m past the obstacle (inside the 0.5 m abeam gate)
    skew = [(x * 0.1, 0.4 * np.exp(-((x * 0.1 - 5.30) ** 2)),
             0.4 * np.exp(-((x * 0.1 - 5.30) ** 2))) for x in range(110)]
    n.otwpnts_cb(path_msg(skew))
    ax, ay, _amp = n._apex_by_obs[("id", 7)]
    assert abs(ax - o.x) < 1e-6, f"apex x must be the obstacle's station, got {ax:.3f}"
    print(f"PASS the apex is stored abeam the obstacle (x={ax:.2f}, obstacle x={o.x:.2f})")


def test_squeeze_apex_is_accepted_at_the_relax_floor():
    # The planner reaches for a squeeze precisely when it has NO candidate at its design margins,
    # so a squeeze apex is short of the geometric need BY CONSTRUCTION: on the shipped numbers
    # |d| = 0.45 against a need of 0.60, a 0.15 m shortfall against an undershoot bound of 0.12.
    # Judging it by the design bar rejected it, and about a fifth of squeezed obstacles never got a
    # global hump -- in the one place a global hump would relieve the reactive layer the most.
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, 0.0, 0.15)]         # need = 0 + (0.15 + 0.45) = 0.60
    n._obs_ids = [7]
    sq = [(x * 0.5, 0.45 * np.exp(-((x * 0.5 - 5.0) ** 2)),
           0.45 * np.exp(-((x * 0.5 - 5.0) ** 2))) for x in range(20)]
    n.otwpnts_cb(path_msg(sq))                              # ...as a NORMAL path: still a tail
    assert ("id", 7) not in n._apex_by_obs, "the design bar must still reject a plain short path"
    n.otwpnts_cb(squeeze_msg(sq))                           # ...tagged squeeze: usable evidence
    assert ("id", 7) in n._apex_by_obs, "a squeeze apex clearing the relax floor must be accepted"
    assert abs(n._apex_by_obs[("id", 7)][2] - 0.45) < 0.02
    # the floor is the one the CORE will lay a hump at: r + relax_floor = 0.15 + 0.30 = 0.45
    n._apex_by_obs.clear()
    thin = [(x, y * 0.8, d * 0.8) for x, y, d in sq]        # 0.36: under the reactive keep-out
    n.otwpnts_cb(squeeze_msg(thin))
    assert ("id", 7) not in n._apex_by_obs, \
        "below the floor the core would lay at, a squeeze apex is not usable evidence either"
    # ...and the OVERSHOOT clamp still applies to a squeeze path
    n._apex_by_obs.clear()
    n._clean_dr = np.full(400, 1.3); n._clean_dl = np.full(400, 1.3)   # room for the overshoot
    wide = [(x, y * 2.0, d * 2.0) for x, y, d in sq]        # 0.90, well past the need
    n.otwpnts_cb(squeeze_msg(wide))
    assert abs(n._apex_by_obs[("id", 7)][2] - 0.60) < 0.02, "a squeeze apex is still clamped"
    print("PASS a squeeze apex is accepted at the floor the core will lay at")


def test_apex_undershoot_rejected():
    # Symmetric with the long-standing overshoot clamp. The obstacle needs |d| = 0.40.
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n.otwpnts_cb(path_msg([(x, y * 0.72, d * 0.72) for x, y, d in HUMP]))   # 0.29: 11 cm under
    assert ("id", 7) in n._apex_by_obs, "a slightly tighter avoidance is still an avoidance"
    n._apex_by_obs.clear()
    n.otwpnts_cb(path_msg([(x, y * 0.4, d * 0.4) for x, y, d in HUMP]))     # 0.16: 24 cm under
    assert ("id", 7) not in n._apex_by_obs, "a decaying tail must not become the record"
    print("PASS an apex that undershoots the geometric need is rejected")


def test_apex_frozen_while_the_active_line_covers_it():
    # Once the ACTIVE line carries a hump that still clears the box, further reactive publishes --
    # made while the car is driving that very hump -- must not move the record.
    n = make_node()
    o = core.Obstacle(5.0, -0.2, 0.15)
    n._obstacles = [o]
    n._obs_ids = [7]
    n.otwpnts_cb(path_msg(HUMP))
    before = n._apex_by_obs[("id", 7)]
    # the harness's active line is straight at y=0, so it clears this box by 0.20-0.15 = 0.05 m;
    # a promise of 0.04 is one it keeps, i.e. the obstacle IS covered
    n.active.floor_by_key = {("id", 7): 0.04}
    n.otwpnts_cb(path_msg([(x, y * 0.8, d * 0.8) for x, y, d in HUMP]))
    assert n._apex_by_obs[("id", 7)] == before, "a covered obstacle's apex must be frozen"
    # ...and the freeze lifts once the line stops clearing it
    n.active.floor_by_key = {("id", 7): 0.35}          # a promise it does NOT keep (0.05 < 0.35)
    n.otwpnts_cb(path_msg([(x, y * 0.8, d * 0.8) for x, y, d in HUMP]))
    assert n._apex_by_obs[("id", 7)] != before, "the freeze must lift when coverage is lost"
    print("PASS the apex is frozen while the active line covers the obstacle")


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


def _straight_frame(n=400, step=0.1):
    """Straight clean line along +x with unit normals toward -y (the '+d_right' side)."""
    xy = np.column_stack([np.arange(n) * step, np.zeros(n)])
    s = np.arange(n) * step
    nvec = np.column_stack([np.zeros(n), -np.ones(n)])
    return xy, s, float(n * step), nvec


def test_clearance_floor_is_enforced_and_drops_honestly():
    # The acceptance rule is "the laid geometry clears the box EDGE by obs_margin", not "the
    # amplitude is >= 90% of the recorded apex". A 0.9 x 0.55 m apex leaves 0.345 m off a 0.15 m
    # box -- under the 0.35 m every downstream consumer assumes -- so the ratio proxy accepted
    # exactly the lines that then read as blocked to the SM.
    xy, s, L, nvec = _straight_frame()
    obs = (20.0, 0.0, 0.15)                       # box on the raceline at x=20
    apex = [(20.0, -0.55)]                        # reactive apex: 0.55 m to the -y side
    roomy = np.full(len(xy), 1.0)
    d_glob, n, _ek, dropped, laid = core.build_offset_profile(
        xy, s, L, nvec, apex, None, 0.0, 3.0, 3.0,
        hi_inc=roomy, lo_inc=-roomy, obstacles=[obs], obs_margin=0.35)
    assert n == 1 and laid, "a hump that CAN clear the box must be laid"
    assert laid[0]["clear"] >= 0.35 - 1e-9, f"laid line clears only {laid[0]['clear']:.3f}"

    # Same apex, corridor clamped to 0.40 m: the hump can never reach 0.35 m of edge clearance
    # (0.40 - 0.15 = 0.25), so it must be DROPPED for the reactive layer, not laid short.
    tight = np.full(len(xy), 0.40)
    _d, n_t, _e, dropped_t, laid_t = core.build_offset_profile(
        xy, s, L, nvec, apex, None, 0.0, 3.0, 3.0,
        hi_inc=tight, lo_inc=-tight, obstacles=[obs], obs_margin=0.35)
    assert n_t == 0 and not laid_t, "an unreachable clearance must not be laid"
    assert dropped_t and dropped_t[0]["reason"] == "clearance", dropped_t
    assert dropped_t[0]["clear"] < 0.35
    print("PASS clearance floor accepts what clears and drops what cannot")


def test_amplitude_comes_from_the_obstacle_not_the_apex():
    # The hump is sized by the BOX (d_obs + side*(r + obs_margin)), not by replaying the reactive
    # apex -- replaying it makes the global line a copy of the local one, with the same lap-time
    # cost and no reason to swap. Two very different recorded apexes on the same side of the same
    # obstacle must therefore produce the SAME line.
    xy, s, L, nvec = _straight_frame()
    obs = (20.0, 0.0, 0.15)
    roomy = np.full(len(xy), 1.0)

    def solve(apex_d):
        return core.build_offset_profile(
            xy, s, L, nvec, [(20.0, apex_d)], None, 0.0, 3.0, 3.0,
            hi_inc=roomy, lo_inc=-roomy, obstacles=[obs], obs_margin=0.35)

    _d, n_a, _e, drop_a, laid_a = solve(-0.45)     # reactive drove 0.30 m off the box edge
    _d, n_b, _e, drop_b, laid_b = solve(-0.75)     # ...or a very wide 0.60 m
    assert n_a == 1 and n_b == 1 and not drop_a and not drop_b, (drop_a, drop_b)
    assert abs(laid_a[0]["laid"] - laid_b[0]["laid"]) < 1e-6, \
        "the recorded apex must not set the amplitude"
    # ...and that amplitude is the obstacle's requirement, tighter than the reactive 0.55 m
    assert abs(laid_a[0]["want"] - 0.50) < 1e-6, laid_a[0]["want"]
    assert laid_a[0]["clear"] >= 0.35 - 1e-9
    assert abs(laid_a[0]["laid"]) < 0.55, "the global line must be tighter than the reactive one"
    print(f"PASS amplitude is obstacle-derived ({laid_a[0]['laid']:+.3f} m for both apexes)")


def test_raceline_already_clear_lays_nothing():
    # d_need is a CONSTRAINT, not a set-point: an obstacle the raceline already stands off by
    # r + obs_margin needs no hump, however far the reactive layer happened to swing around it.
    xy, s, L, nvec = _straight_frame()
    obs = (20.0, 0.9, 0.15)                        # 0.9 m to the -d side; raceline clears by 0.75
    roomy = np.full(len(xy), 1.5)
    _d, n, _e, dropped, laid = core.build_offset_profile(
        xy, s, L, nvec, [(20.0, -0.4)], None, 0.0, 3.0, 3.0,
        hi_inc=roomy, lo_inc=-roomy, obstacles=[obs], obs_margin=0.35)
    assert n == 0 and not laid and not dropped, (n, laid, dropped)
    print("PASS an obstacle the raceline already clears gets no hump")


def test_clearance_drift_retriggers_once():
    # A box that drifts 0.15 m -- under any sane obs_change_tol -- can still take the followed
    # line under the SM's static requirement. The consequence trigger must catch that, and must
    # fire ONCE per active line: _mark_dirty discards the pending, so an unlatched trigger would
    # stop the very rebuild it asked for from ever committing.
    n = make_node()
    n.active = straight_bundle()                       # line along y=0
    n.active.clearance_by_key = {("id", 7): 0.40}      # built with 0.40 m of edge clearance
    n._mark_dirty = lambda: setattr(n, "_obstacles_dirty", True)

    far = [core.Obstacle(5.0, 0.55, 0.15)]             # 0.55 - 0.15 = 0.40 m -> fine
    assert n._clearance_drifted(far, [7]) is False

    near = [core.Obstacle(5.0, 0.40, 0.15)]            # 0.25 m -> under the 0.30 threshold
    assert n._clearance_drifted(near, [7]) is True, "drift into the line must re-arm the solve"
    assert n._clearance_drifted(near, [7]) is False, "and must not re-fire on the same line"

    # An obstacle this line never cleared is the reactive layer's job -- it must never re-arm a
    # deterministic rebuild that would change nothing.
    n2 = make_node()
    n2.active = straight_bundle()
    n2.active.clearance_by_key = {("id", 7): 0.12}     # hump was dropped; line was always short
    assert n2._clearance_drifted([core.Obstacle(5.0, 0.20, 0.15)], [7]) is False
    print("PASS clearance drift re-arms the solve exactly once per active line")


def test_minor_apex_refinement_keeps_the_pending():
    # Discarding a queued bundle for a 5 cm apex refinement costs a whole swap opportunity: the
    # commit gates are hardest to satisfy exactly where the obstacles are, so the lap-2 swap slips
    # to lap 3 for an improvement nobody asked for. A NEW or relocated apex still discards it.
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    drive(n, HUMP)                                     # first apex -> major, no pending yet
    assert n._apex_change_major is True

    sentinel = object()
    n._pending, n._pending_dev = sentinel, np.zeros(3)
    n._obstacles_dirty = False
    # shrink, not grow: growth is capped by the overshoot clamp at the obstacle's requirement
    nudged = [(x, y * 0.83, d * 0.83) for x, y, d in HUMP]     # 0.40 -> 0.33: >5 cm, <10 cm
    drive(n, nudged)
    assert n._obstacles_dirty, "a >5cm change must still arm the rebuild"
    assert n._apex_change_major is False, "…but 6 cm is not a major change"
    assert n._pending is sentinel, "the queued line must survive a minor refinement"

    n._pending, n._pending_dev = sentinel, np.zeros(3)
    n._obstacles_dirty = False
    # a NEW apex -- the realistic major trigger, a newly confirmed obstacle -- must drop it. (A
    # large SHRINK cannot be used here any more: the undershoot predicate rejects a record that
    # falls more than apex_undershoot_m below what the obstacle geometrically requires.)
    n._apex_by_obs.clear()
    n._obstacles_dirty = False
    drive(n, HUMP)
    assert n._obstacles_dirty and n._apex_change_major, "a new apex must read as a major change"
    assert n._pending is None, "a major apex change must discard the queued line"
    print("PASS a minor apex refinement keeps the pending bundle, a major one drops it")


def drive_frenet(n, s, t):
    """One /car_state/frenet/odom sample at wall time `t`."""
    n._clock.t = t
    msg = types.SimpleNamespace(
        pose=types.SimpleNamespace(pose=types.SimpleNamespace(position=types.SimpleNamespace(x=s))),
        twist=types.SimpleNamespace(twist=types.SimpleNamespace(linear=types.SimpleNamespace(x=3.0))))
    n.frenet_cb(msg)


def test_swap_held_while_trailing_a_close_obstacle():
    # Changing the reference under a car that is already braking for a box a couple of metres ahead
    # is the worst moment there is: the controller's look-ahead lands on geometry it has never
    # tracked and the reactive commit is still anchored to the OLD line. The real run's collision
    # was 28 ms after a swap taken in exactly that state.
    n = make_node()
    n._clock.t = 100.0
    bundle = straight_bundle()
    n._pending, n._pending_dev = bundle, np.zeros(400)
    n._pending_since = 0.0                      # long enough that the deadlock breaker would fire
    n._last_vs = 0.5                            # ...and slow enough
    n._reactive_active = False
    n._reactive_idle_t = 0.0
    n._publish_active = lambda b: None
    n._publish_coverage = lambda b: None
    n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)
    n._obstacles = [core.Obstacle(12.0, 0.0, 0.15)]
    n.state_cb(types.SimpleNamespace(data="TRAILING"))
    n._commit_pending(10.0)                     # obstacle 2 m ahead
    assert n._pending is bundle, "a swap must not land while trailing a close obstacle"
    # the gate is the COMBINATION: trailing far from anything is the opponent's doing
    n._commit_pending(5.0)                      # obstacle 7 m ahead
    assert n._pending is None, "a distant obstacle must not hold the swap"
    # ...and a stale/absent state machine must not disable the feature (bag / headless)
    n._pending, n._pending_dev = bundle, np.zeros(400)
    n._clock.t = 110.0                          # the TRAILING message is now 10 s old
    n._commit_pending(10.0)
    assert n._pending is None, "a stale state must not block the swap forever"
    print("PASS the swap is held while TRAILING a close obstacle")


class _FakeMap:
    """Eroded-map stand-in: free everywhere except a wall band at a chosen lateral offset.

    The harness clean line runs along +x with the left normal +y, so a wall at y = `wall_y` blocks
    that side beyond it.
    """

    def __init__(self, wall_lo=-10.0, wall_hi=10.0, resolution=0.05):
        self.resolution = resolution
        self.origin = (-50.0, -50.0)
        self.wall_lo, self.wall_hi = wall_lo, wall_hi
        n = int(100.0 / resolution)
        self.eroded_image = np.full((n, n), 255, dtype=np.uint8)
        y = (np.arange(n) + 0.5) * resolution + self.origin[1]
        self.eroded_image[(y < wall_lo) | (y > wall_hi), :] = 0


def test_every_obstacle_gets_a_knot_even_with_no_reactive_apex():
    # A hump's amplitude and station were already derived from the obstacle; its EXISTENCE was not.
    # It needed a recorded reactive apex, so an obstacle the reactive layer never once avoided
    # successfully got no knot, stayed reactive-only forever, and did so quietly -- the thing that
    # would have fixed it was gated on the thing that was failing.
    n = make_node()
    o = core.Obstacle(5.0, -0.2, 0.15)
    n._obstacles = [o]
    n._obs_ids = [7]
    assert not n._apex_by_obs, "this obstacle has never been reactively avoided"
    pairs = n._apex_pairs([o])
    assert len(pairs) == 1, "every confirmed obstacle must get a knot"
    (kx, ky), po, from_apex = pairs[0]
    assert po is o and from_apex is False, "…and it must be marked as obstacle-derived"
    # the knot sits ABEAM the box, at the amplitude the box requires
    d_k, i_k, left = n._clean_offset(kx, ky)
    d_o, i_o, _ = n._clean_offset(o.x, o.y)
    assert i_k == i_o, "the knot's station must be the obstacle's own"
    assert abs(abs(d_k - d_o) - (o.r + n.obs_margin)) < 1e-6, \
        f"amplitude must be r + obs_margin off the box, got {abs(d_k - d_o):.3f}"
    print(f"PASS an obstacle with no reactive apex still gets a knot (d={d_k:+.2f} abeam)")


def test_knot_side_comes_from_the_map_not_the_waypoint_bounds():
    # d_left/d_right are labelled by one global decision in gb_optimizer and ship EXCHANGED on some
    # maps (map f: 125 of 128 sampled waypoints), so a side chosen from them puts the hump into a
    # wall. The eroded map cannot be mislabelled.
    o = core.Obstacle(5.0, 0.0, 0.15)
    for wall_lo, wall_hi, want, label in ((-0.35, 3.0, +1.0, "wall on the -d side"),
                                          (-3.0, 0.35, -1.0, "wall on the +d side")):
        n = make_node()
        n._obstacles, n._obs_ids = [o], [7]
        n.map_filter = _FakeMap(wall_lo, wall_hi)
        (kx, ky), _po, from_apex = n._apex_pairs([o])[0]
        d_k, _i, _l = n._clean_offset(kx, ky)
        assert np.sign(d_k) == want, f"{label}: expected side {want:+.0f}, knot at d={d_k:+.2f}"
        assert from_apex is False
    # ...and a recorded apex still decides when the two sides are within the hint margin
    n = make_node()
    n._obstacles, n._obs_ids = [o], [7]
    n.map_filter = _FakeMap(-3.0, 3.0)                  # symmetric: no measured preference
    n._apex_by_obs = {("id", 7): (5.0, -0.5, 0.5)}      # the reactive pass went to -d
    (kx, ky), _po, from_apex = n._apex_pairs([o])[0]
    assert from_apex is True and ky < 0, "a symmetric corridor must defer to the recorded apex"
    print("PASS the knot side is measured in the map, with the apex as the tie-break")


def test_a_stale_solve_cannot_undo_a_clean_swap():
    # THE terminal state. Three boxes are removed, the layer drops them, the set empties and the
    # line correctly swaps to CLEAN -- and then a solve submitted 0.95 s earlier, while the boxes
    # were still confirmed, lands and puts the obstacle-aware line back. With the set now empty and
    # unchanging there is nothing left to arm the trigger, so that line is permanent: the observed
    # run drove its last 37 s on it and pressing clear again did nothing.
    from concurrent.futures import ThreadPoolExecutor
    n = make_node()
    n._solve_pool = ThreadPoolExecutor(max_workers=1)
    n.reopt_method = "local_window"
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n._apex_by_obs = {("id", 7): (5.0, 0.4, 0.4)}
    n.active = straight_bundle()
    n._publish_active = lambda b: None
    n._publish_coverage = lambda b: None
    n._publish_clearance = lambda: None
    n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)
    gate = threading.Event()
    obstacle_line = straight_bundle()
    obstacle_line.n_apex, obstacle_line.clearance_ok, obstacle_line.coverage = 1, True, []

    def slow_build(obstacles, pairs=None):
        gate.wait(5.0)
        return obstacle_line
    n._build_obstacle_bundle = slow_build

    n._obstacles_dirty = True
    n._rebuild_and_swap("apex captured")            # submitted WITH the obstacle
    assert n._solve_future is not None

    # the boxes are removed: the set empties and the clean line goes in
    n._obstacles, n._obs_ids = [], []
    n._mark_dirty()
    n._obstacles_dirty = True
    n._rebuild_and_swap("obstacles cleared")
    assert n.active is n.clean_bundle or n._pending is n.clean_bundle, \
        "the empty set must produce the clean line"
    n.active = n.clean_bundle
    n._pending = None

    # ...and NOW the old solve finishes and is collected
    gate.set()
    for _ in range(50):
        n._collect_solve()
        if n._solve_future is None:
            break
        time.sleep(0.01)
    assert n.active is n.clean_bundle, "a stale solve must not undo the clean swap"
    assert n._pending is not obstacle_line, "…nor be queued for one"
    n._solve_pool.shutdown(wait=True)
    print("PASS a stale solve cannot undo a clean swap")


def test_an_empty_set_can_only_install_the_clean_line():
    # The backstop, independent of the epoch check: every gate below reasons about obstacles, so an
    # obstacle-aware bundle arriving with an empty set passes them all by default.
    n = make_node()
    n.active = straight_bundle()
    n._obstacles, n._obs_ids = [], []
    n._publish_active = lambda b: None
    n._publish_coverage = lambda b: None
    n._publish_clearance = lambda: None
    n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)
    rogue = straight_bundle()
    rogue.n_apex, rogue.clearance_ok, rogue.coverage = 1, True, []
    n._finish_rebuild(rogue, [], "stale", 100.0)
    assert n._pending is None and n.active is not rogue, \
        "an obstacle-aware line must not be installable with no obstacles"
    print("PASS an empty obstacle set can only install the clean line")


def test_a_finished_solve_is_collected_before_the_next_is_submitted():
    # A FINISHED but uncollected result was overwritten by the next submit and silently lost -- and
    # its trigger had already been burned, so the work vanished with nothing re-arming.
    from concurrent.futures import ThreadPoolExecutor
    n = make_node()
    n._solve_pool = ThreadPoolExecutor(max_workers=1)
    n.reopt_method = "local_window"
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n._apex_by_obs = {("id", 7): (5.0, 0.4, 0.4)}
    built = []
    n._build_obstacle_bundle = lambda obstacles, pairs=None: built.append(1) or straight_bundle()
    collected = []
    n._finish_rebuild = lambda b, o, r, t: collected.append(r)

    n._obstacles_dirty = True
    n._rebuild_and_swap("first")
    n._solve_future.result(timeout=5.0)              # finished, NOT collected
    n._obstacles_dirty = True
    n._rebuild_and_swap("second")
    assert "first" in collected, f"the finished result must be collected, got {collected}"
    n._solve_pool.shutdown(wait=True)
    print("PASS a finished solve is collected before the next is submitted")


def test_solve_runs_off_the_executor_and_is_collected_later():
    # The solve is 200-850 ms of pure function. Run inline on the 40 Hz frenet callback it stalled
    # the node for that long -- and the swap gates it feeds (ego-s freshness, reactive idleness)
    # are judged from callbacks that then could not run, so the solve blocked the very conditions
    # it was waiting for.
    from concurrent.futures import ThreadPoolExecutor
    n = make_node()
    n._solve_pool = ThreadPoolExecutor(max_workers=1)
    n._solve_future = None
    n._solve_ctx = None
    n.reopt_method = "local_window"
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n._apex_by_obs = {("id", 7): (5.0, 0.4, 0.4)}
    built = []
    gate = threading.Event()

    def slow_build(obstacles, pairs=None):
        gate.wait(5.0)                                # stands in for the 200-850 ms solve
        built.append((list(obstacles), pairs))
        return straight_bundle()
    n._build_obstacle_bundle = slow_build
    installed = []
    n._finish_rebuild = lambda b, o, r, t: installed.append((b, r))

    t0 = time.perf_counter()
    n._rebuild_and_swap("apex captured")
    submit_ms = (time.perf_counter() - t0) * 1e3
    assert submit_ms < 50.0, f"the trigger must return at once, took {submit_ms:.1f} ms"
    assert not installed, "nothing may be installed before the solve finishes"
    n._collect_solve()
    assert not installed, "a solve still running must not be collected"
    # a second trigger while one is in flight must not queue a second solve
    n._obstacles_dirty = True
    n._rebuild_and_swap("apex captured")
    gate.set()
    n._solve_future.result(timeout=5.0)
    assert len(built) == 1, f"one solve at a time, ran {len(built)}"
    n._collect_solve()
    assert len(installed) == 1 and installed[0][1] == "apex captured", installed
    assert n._solve_future is None
    # the apex pairs are resolved on the CALLER's thread -- otwpnts_cb rewrites them at 20 Hz
    assert built[0][1] is not None, "pairs must be snapshotted before the hand-off"
    # an EMPTY set needs no solve at all: the clean bundle is precomputed
    installed.clear()
    n._obstacles, n._obs_ids, n._apex_by_obs = [], [], {}
    n._obstacles_dirty = True
    n._rebuild_and_swap("obstacles cleared")
    assert len(installed) == 1 and installed[0][0] is n.clean_bundle, \
        "the clean revert must not wait on a worker"
    n._solve_pool.shutdown(wait=True)
    print("PASS the solve runs off the executor and is collected on the republish tick")


def test_swap_gate_tally_names_the_gate_that_held_the_swap():
    # The commit path is a chain of fail-closed gates and from outside they are indistinguishable:
    # the line simply never swaps. The tally turns that into one readable line per lap.
    n = make_node()
    n._clock.t = 100.0
    bundle = straight_bundle()
    n._pending, n._pending_dev = bundle, np.zeros(400)
    n._pending_since = 99.0
    n._last_vs = 5.0
    n._publish_active = lambda b: None
    n._publish_coverage = lambda b: None
    n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)

    n._reactive_active = True                       # the reactive layer is mid-maneuver
    for _ in range(3):
        n._commit_pending(10.0)
    assert n._swap_block.get("reactive_not_idle") == 3, n._swap_block

    n._reactive_active, n._reactive_idle_t = False, 0.0
    n._pending_dev = np.full(400, 0.5)              # ...and now the lines disagree ahead
    n._commit_pending(10.0)
    assert n._swap_block.get("lines_disagree_in_horizon") == 1, n._swap_block

    msgs = []
    n.get_logger = lambda: types.SimpleNamespace(
        info=lambda m, **k: msgs.append(m), warn=lambda *a, **k: None,
        warning=lambda *a, **k: None, error=lambda *a, **k: None, debug=lambda *a, **k: None)
    n._report_swap_blocks()
    assert msgs and "reactive_not_idle x3" in msgs[-1] and "lines_disagree_in_horizon x1" in msgs[-1], \
        msgs
    assert "a line is still queued" in msgs[-1]
    assert n._swap_block == {}, "the tally resets each lap"
    print("PASS the swap-gate tally names the gate that held the swap")


def test_solves_are_debounced():
    # The set-change and apex triggers can both fire several times a second while a track flaps or
    # an apex settles, and every solve REPLACES the queued bundle -- so the line kept being rebuilt
    # and the swap kept being pushed back past the obstacle it was built for.
    n = make_node()
    n._commit_pending = lambda s, force=False: None
    solves = []

    def fake_solve(reason):
        solves.append((n._clock.t, reason))
        n._last_solve_t = n._clock.t          # mirrors _rebuild_and_swap's stamp
        n._obstacles_dirty = False
    n._rebuild_and_swap = fake_solve
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n._apex_by_obs = {("id", 7): (5.0, 0.4, 0.4)}

    n._obstacles_dirty = True
    drive_frenet(n, 1.0, 100.0)
    assert len(solves) == 1, "the first trigger must solve immediately"
    n._obstacles_dirty = True                 # a flap re-arms 200 ms later
    drive_frenet(n, 1.6, 100.2)
    assert len(solves) == 1, "a re-trigger inside the debounce window must wait"
    n._obstacles_dirty = True
    drive_frenet(n, 4.0, 101.5)
    assert len(solves) == 2, "…and run once the window has passed"
    # an EMPTY set is exempt: reverting to the clean line is the safe direction and free
    n._obstacles, n._obs_ids, n._apex_by_obs = [], [], {}
    n._obstacles_dirty = True
    drive_frenet(n, 4.3, 101.6)
    assert len(solves) == 3 and solves[-1][1] == "obstacles cleared", \
        "the clean revert must not be debounced"
    print("PASS solves are debounced to one per solve_min_interval_s")


def test_a_rebuild_that_changes_nothing_is_not_queued():
    # 6 of 16 swaps in one run moved the line by at most 0.054 m -- rebuilds triggered by
    # re-measurement noise, producing a line the car cannot tell from the one it is on, each still
    # costing a /global_waypoints publish and a FrenetConverter rebuild in three consumers.
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, 0.60, 0.15)]     # the straight active line clears it by 0.45
    n._obs_ids = [7]
    n.active = straight_bundle()
    n.active.floor_by_key = {("id", 7): 0.35}
    n.active.coverage = []
    n._publish_active = lambda b: None
    n._publish_coverage = lambda b: None
    n._publish_clearance = lambda: None
    n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)

    same = straight_bundle()                            # identical geometry
    same.n_apex, same.clearance_ok, same.coverage = 1, True, []
    n._finish_rebuild(same, n._obstacles, "apex captured", 100.0)
    assert n._pending is None, "a rebuild identical to the active line must not be queued"

    # ...but one that MOVES the line is queued as before
    moved = straight_bundle()
    for w in moved.glb_wpnts.wpnts:
        w.y_m += 0.40
    moved.n_apex, moved.clearance_ok, moved.coverage = 1, True, []
    n._finish_rebuild(moved, n._obstacles, "apex captured", 100.0)
    assert n._pending is moved, "a materially different line must still be queued"

    # ...and so is an identical one when the ACTIVE line no longer clears the obstacle
    n._pending = None
    n.active.floor_by_key = {("id", 7): 0.90}           # a promise the active line does not keep
    same2 = straight_bundle()
    same2.n_apex, same2.clearance_ok, same2.coverage = 1, True, []
    n._finish_rebuild(same2, n._obstacles, "apex captured", 100.0)
    assert n._pending is same2, "coverage loss must override the no-change skip"
    print("PASS a rebuild with no material change and no coverage loss is not queued")


def test_a_brief_tracker_dropout_does_not_shrink_the_set():
    # The tracker deletes and recreates its tracks around every lap, and a recreated track is
    # published by nobody for a few frames. Taken at face value that shrinks the confirmed set,
    # re-solves, swaps, and does it again in reverse a fraction of a second later.
    from visualization_msgs.msg import Marker, MarkerArray

    def markers(*specs):
        msg = MarkerArray()
        for mid, x, y in specs:
            m = Marker(); m.action = Marker.ADD; m.id = mid
            m.pose.position.x, m.pose.position.y = float(x), float(y)
            m.scale.x = m.scale.y = 0.3
            msg.markers.append(m)
        return msg

    n = make_node()
    n.default_obs_radius, n.apex_miss_frames = 0.15, 20
    n._clock.t = 100.0
    n.obstacles_cb(markers((7, 5.0, -0.2), (8, 12.0, 0.2)))
    assert len(n._obstacles) == 2
    n._obstacles_dirty = False
    n._clock.t = 100.2                                  # 0.2 s later, id 8 is missing
    n.obstacles_cb(markers((7, 5.0, -0.2)))
    assert len(n._obstacles) == 2, "a 0.2 s dropout must not shrink the set"
    assert not n._obstacles_dirty, "…and must not arm a re-solve"
    # ...but a box gone for longer than obs_forget_s really is gone
    n._clock.t = 102.0
    n.obstacles_cb(markers((7, 5.0, -0.2)))
    assert len(n._obstacles) == 1, "past obs_forget_s the set must shrink"
    print("PASS a brief tracker dropout does not shrink the confirmed set")


def test_marker_id_zero_does_not_flip_the_key_scheme():
    # marker_id 0 is a perfectly good id -- the layer hands it out first. `any(i != 0)` meant the
    # key scheme flipped to position quantization the moment the only confirmed obstacle left was
    # the one holding id 0, and every lookup keyed on ("id", 0) missed at once: floor_by_key,
    # clearance_by_key, _apex_by_obs, and with them the apex freeze.
    n = make_node()
    o = core.Obstacle(5.0, -0.2, 0.15)
    assert n._keys_for([o], [0]) == [("id", 0)], n._keys_for([o], [0])
    assert n._keys_for([o, core.Obstacle(9.0, 0.1, 0.15)], [0, 1]) == [("id", 0), ("id", 1)]
    # duplicated ids are a real ambiguity and still fall back to position
    dup = n._keys_for([o, core.Obstacle(9.0, 0.1, 0.15)], [3, 3])
    assert all(k[0] == "q" for k in dup), dup
    print("PASS marker_id 0 keeps the id key scheme")


def test_obstacle_set_change_is_compared_by_id_not_by_order():
    # The old zip() paired the n-th new obstacle with the n-th old one, so a MarkerArray that
    # merely arrived in a different order read as every obstacle having jumped.
    n = make_node()
    a = core.Obstacle(5.0, -0.2, 0.15)
    b = core.Obstacle(12.0, 0.2, 0.15)
    n._obstacles, n._obs_ids = [a, b], [7, 8]
    assert n._obstacles_changed([b, a], [8, 7]) is False, "reordering is not a change"
    assert n._obstacles_changed([a, b], [7, 8]) is False
    moved = core.Obstacle(12.4, 0.2, 0.15)
    assert n._obstacles_changed([moved, a], [8, 7]) is True, "a real move still reads as one"
    print("PASS the obstacle set is compared by id, not by list order")


def test_shrinking_set_keeps_the_queued_line():
    # An unlatch flap (item: the layer demotes a track for ~0.5 s) used to throw the queued bundle
    # away, and the commit gates -- hardest to satisfy exactly where the obstacles are -- had to be
    # re-earned from scratch. A line built for MORE obstacles is conservative, not stale.
    from visualization_msgs.msg import Marker, MarkerArray

    def markers(*specs):
        msg = MarkerArray()
        for mid, x, y in specs:
            m = Marker(); m.action = Marker.ADD; m.id = mid
            m.pose.position.x, m.pose.position.y = float(x), float(y)
            m.scale.x = m.scale.y = 0.3
            msg.markers.append(m)
        return msg

    n = make_node()
    n.default_obs_radius = 0.15
    n.apex_miss_frames = 20
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15), core.Obstacle(12.0, 0.2, 0.15)]
    n._obs_ids = [7, 8]
    sentinel = object()
    n._pending, n._pending_dev = sentinel, np.zeros(3)
    n.obstacles_cb(markers((7, 5.0, -0.2)))            # obstacle 8 demoted away
    assert n._obstacles_dirty, "a shrink still arms the rebuild"
    assert n._pending is sentinel, "a line built for MORE obstacles must stay queued"
    # ...but a set that GREW or MOVED discards it: those humps are missing or in the wrong place
    n._pending, n._pending_dev = sentinel, np.zeros(3)
    n.obstacles_cb(markers((7, 5.0, -0.2), (9, 20.0, 0.3)))
    assert n._pending is None, "a new obstacle must discard the queued line"
    # a swap (one gone, one new) is NOT a shrink even though the count can fall
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15), core.Obstacle(12.0, 0.2, 0.15),
                    core.Obstacle(20.0, 0.3, 0.15)]
    n._obs_ids = [7, 8, 9]
    n._pending, n._pending_dev = sentinel, np.zeros(3)
    n.obstacles_cb(markers((7, 5.0, -0.2), (10, 30.0, 0.4)))
    assert n._pending is None, "a set that lost one obstacle and gained another is not a shrink"
    print("PASS a shrinking obstacle set keeps the queued line")


def test_cluster_curvature_budget_is_local():
    # The opposite-side "can we swing across at all?" test spends the curvature budget the RACELINE
    # leaves. Taking max|kappa| over the whole track meant one tight corner anywhere set the budget
    # everywhere, so a pair on a dead-flat straight was declared an impossible swing and both boxes
    # were dropped.
    s_loop = np.arange(0.0, 40.0, 0.1)
    kap = np.zeros_like(s_loop)
    kap[(s_loop > 30.0) & (s_loop < 33.0)] = 1.45     # one tight corner, far from the pair
    assert core._kappa_local_max(kap, s_loop, 5.0, 7.0, 40.0, 3.0) == 0.0, \
        "a straight must see none of a corner 20 m away"
    assert core._kappa_local_max(kap, s_loop, 29.0, 31.0, 40.0, 3.0) > 1.0, \
        "a pair ON the corner must still see it"
    # wrapped spans are measured the same way
    assert core._kappa_local_max(kap, s_loop, 38.0, 2.0, 40.0, 1.0) == 0.0
    assert core._kappa_local_max(kap, s_loop, 34.0, 38.0, 40.0, 2.0) > 1.0, \
        "the reach either side is part of the arc"
    # end to end: an opposite-side pair 2.5 m apart on the straight survives the cluster stage
    knots = [(5.0, -0.40, 3.0, 3.0, 5.0, -0.40, [(None, 0)], 0, -0.2),
             (7.5, 0.40, 3.0, 3.0, 7.5, 0.40, [(None, 1)], 1, 0.2)]
    dropped = []
    kept = core._cluster_knots(knots, 40.0, 2.0, 1.5, kap, s_loop, dropped)
    assert len(kept) == 2 and not dropped, \
        f"a swing on a flat straight must not be pre-dropped (kept {len(kept)}, dropped {dropped})"
    # ...while the same pair sitting ON the corner still is
    knots_c = [(30.2, -0.40, 3.0, 3.0, 30.2, -0.40, [(None, 0)], 0, -0.2),
               (30.7, 0.40, 3.0, 3.0, 30.7, 0.40, [(None, 1)], 1, 0.2)]
    dropped_c = []
    core._cluster_knots(knots_c, 40.0, 2.0, 1.5, kap, s_loop, dropped_c)
    assert dropped_c, "a swing where the raceline already spends the budget must still be dropped"
    print("PASS the cluster curvature budget is measured locally")


def test_coverage_outranks_lap_time_in_the_reach_search():
    # The reach search ranked candidates on estimated lap time alone, and NOT laying a hump is
    # always faster than laying it. The real run: reach 1.0 m fitted one of three humps and scored
    # 11.95 s, reach 2.0 m covered all three at 13.30 s -- so it shipped the line that left two
    # obstacles to the reactive layer, every lap.
    covers_all = core._rank_cost(13.30, 0.0, 0)
    drops_two = core._rank_cost(11.95, 0.0, 2)
    assert covers_all < drops_two, \
        f"a 1.35 s faster line that drops two obstacles must lose: {drops_two:.2f} vs {covers_all:.2f}"
    # one dropped obstacle is not outweighed by any plausible lap-time spread between reaches
    assert core._rank_cost(13.30, 0.0, 0) < core._rank_cost(11.95, 0.0, 1)
    # ...but at EQUAL coverage lap time still decides, and the span budget still outranks it
    assert core._rank_cost(11.95, 0.0, 1) < core._rank_cost(13.30, 0.0, 1)
    assert core._rank_cost(11.95, 3.0, 0) > core._rank_cost(13.30, 0.0, 0)
    # and when nothing covers everything, the least-dropping candidate still wins (no failure)
    assert core._rank_cost(20.0, 0.0, 1) < core._rank_cost(11.95, 0.0, 2)
    print("PASS coverage outranks lap time in the reach search")


def test_weave_failure_drops_only_the_implicated_humps():
    # A weave violation is an INTERACTION between overlapping humps. Discarding the whole profile
    # made an interaction at one end of the lap cost every other obstacle its coverage -- and the
    # fallback is not the clean line, it is whatever older, less-covered line is still active.
    n_st = 900
    xy = np.column_stack([np.arange(n_st) * 0.1, np.zeros(n_st)])
    s_l = np.arange(n_st) * 0.1
    nv = np.column_stack([np.zeros(n_st), -np.ones(n_st)])
    hi, lo = np.full(n_st, 1.2), np.full(n_st, -1.2)
    # a close OPPOSITE-side pair (1.5 m apart) that cannot be woven, plus one easy hump far away
    apex = [(19.0, -0.55), (20.5, 0.55), (60.0, -0.55)]
    obs = [(19.0, 0.0, 0.15), (20.5, 0.0, 0.15), (60.0, 0.0, 0.15)]
    _d, nn, _e, drp, lay = core.build_offset_profile(
        xy, s_l, float(n_st * 0.1), nv, apex, None, 0.0, 3.0, 3.0, hi_inc=hi, lo_inc=lo,
        obstacles=obs, obs_margin=0.35, relax_floor=0.30, curvlim=1.5,
        clean_kappa=np.zeros(n_st))
    assert nn == 1, f"the unimplicated hump must survive, got {nn} laid"
    assert [a["obs_i"] for a in lay] == [2], [a["obs_i"] for a in lay]
    assert sorted(d["obs_i"] for d in drp) == [0, 1]
    assert all(d["reason"] == "weave" for d in drp)
    print("PASS a weave failure drops only the humps that overlap the violation")


def _excursions(alpha, tol=0.02):
    """Contiguous off-line runs of a CYCLIC profile. One = the line left the raceline once and
    came back once; two over a same-side pair is the W."""
    off = np.abs(np.asarray(alpha, float)) > tol
    if not off.any():
        return 0
    if off.all():
        return 1
    off = np.roll(off, -int(np.argmin(off)))      # rotate so index 0 is ON the line
    return int(np.count_nonzero(off[1:] & ~off[:-1])) + int(off[0])


def test_wrap_normals_is_a_correct_lateral_basis():
    # The lateral basis is the coordinate the offset is laid in AND the coordinate every downstream
    # check measures in, so an error in it is indistinguishable from an error in the offset. Three
    # properties, on a closed line WITH the duplicated closing point every raceline ships with.
    n = 240
    R = 4.0
    th = np.arange(n) * (2.0 * np.pi / n)
    circle = R * np.column_stack([np.cos(th), np.sin(th)])
    closed = np.vstack([circle, circle[:1]])              # duplicated closing point
    nv = core._wrap_normals(closed)
    assert nv.shape == (n + 1, 2), nv.shape
    assert np.allclose(nv[-1], nv[0]), "the duplicated point must carry station 0's normal"
    assert np.allclose(np.hypot(nv[:, 0], nv[:, 1]), 1.0), "normals must be unit"

    # (1) PERPENDICULAR, and symmetrically so: a correct normal bisects the turn, so its angle to
    # the chord ahead and to the chord behind are equal and opposite. This is what the shipped
    # basis got wrong -- on ifac it is up to 33 deg asymmetric, i.e. simply pointing the wrong way.
    u = closed[:-1]
    fwd = np.roll(u, -1, axis=0) - u
    bwd = u - np.roll(u, 1, axis=0)
    fwd /= np.hypot(fwd[:, 0], fwd[:, 1])[:, None]
    bwd /= np.hypot(bwd[:, 0], bwd[:, 1])[:, None]
    asym = np.abs(np.einsum("ij,ij->i", nv[:n], fwd) + np.einsum("ij,ij->i", nv[:n], bwd))
    assert asym.max() < 1e-9, f"the normal must bisect the turn, asymmetry {asym.max():.2e}"

    # (2) it ROTATES at the rate the line's own curvature demands, at EVERY station including the
    # seam. On a circle of radius R that is exactly el/R per segment.
    a2, b2 = nv[:n], np.roll(nv[:n], -1, axis=0)
    rot = np.arctan2(a2[:, 0] * b2[:, 1] - a2[:, 1] * b2[:, 0], np.einsum("ij,ij->i", a2, b2))
    el = np.hypot(*(np.roll(u, -1, axis=0) - u).T)
    assert np.abs(np.abs(rot) - el / R).max() < 1e-6, "rotation must equal kappa * el everywhere"

    # (3) SIDE CONVENTION unchanged: it agrees with centerline_frame's +right normal everywhere.
    rl = np.column_stack([closed[:, 0], closed[:, 1], np.ones(n + 1), np.ones(n + 1)])
    _, nv_cf, _ = core.centerline_frame(rl)
    nv_cf[-1] = nv_cf[0]
    assert np.all(np.einsum("ij,ij->i", nv, nv_cf) > 0), "the side convention must not flip"
    print("PASS _wrap_normals bisects the turn, tracks kappa*el, and keeps the +right convention")


def test_wrap_normals_has_no_seam_artifact():
    # The failure the offset actually saw: the shipped basis arrives at s = 0 having rotated by the
    # wrong amount, so a smooth offset profile gets a one-station lateral step -- a kink in the
    # published line where d(s) is perfectly C2.
    n = 240
    R = 4.0
    th = np.arange(n) * (2.0 * np.pi / n)
    circle = R * np.column_stack([np.cos(th), np.sin(th)])
    closed = np.vstack([circle, circle[:1]])
    alpha = np.full(n + 1, 0.5)                          # a CONSTANT offset: a concentric circle
    stitch = closed + alpha[:, None] * core._wrap_normals(closed)
    k = np.abs(core._menger_kappa(stitch[:-1]))
    assert (k.max() - k.min()) < 1e-6, \
        f"a constant offset must give a constant curvature, spread {k.max() - k.min():.2e}"
    # ...and the shipped basis does NOT: the seam station stands out, which is the kink.
    rl = np.column_stack([closed[:, 0], closed[:, 1], np.ones(n + 1), np.ones(n + 1)])
    _, nv_cf, _ = core.centerline_frame(rl)
    nv_cf[-1] = nv_cf[0]
    k_cf = np.abs(core._menger_kappa((closed + alpha[:, None] * nv_cf)[:-1]))
    assert (k_cf.max() - k_cf.min()) > 10.0 * (k.max() - k.min()) + 1e-6, \
        "the harness must reproduce the artifact this basis was introduced to remove"
    print(f"PASS a constant offset is a constant curvature: spread {k.max() - k.min():.2e} "
          f"(shipped basis: {k_cf.max() - k_cf.min():.2e})")


def _hold_case(gap_m, kappa=0.0, hold_gap=8.0, hold_kappa=0.3, n_st=900, half=1.2, pinch=None):
    """One same-side pair `gap_m` apart on a synthetic straight; returns the offset profile.

    `pinch` = (x_lo, x_hi, limit) narrows the corridor over a band that lies BETWEEN the two humps'
    reaches, i.e. exactly where only a held plateau would go. Narrowing the whole corridor instead
    would shrink both humps and prove nothing about the bridge."""
    xy = np.column_stack([np.arange(n_st) * 0.1, np.zeros(n_st)])
    s_l = np.arange(n_st) * 0.1
    nv = np.column_stack([np.zeros(n_st), -np.ones(n_st)])
    hi, lo = np.full(n_st, half), np.full(n_st, -half)
    if pinch is not None:
        x_lo, x_hi, lim = pinch
        hi[(s_l >= x_lo) & (s_l <= x_hi)] = lim
    x0 = 30.0
    apex = [(x0, -0.55), (x0 + gap_m, -0.55)]                     # SAME side
    obs = [(x0, 0.0, 0.15), (x0 + gap_m, 0.0, 0.15)]
    d, nn, _e, drp, lay = core.build_offset_profile(
        xy, s_l, float(n_st * 0.1), nv, apex, None, 0.0, 2.0, 2.0, hi_inc=hi, lo_inc=lo,
        obstacles=obs, obs_margin=0.35, relax_floor=0.30, curvlim=1.5,
        clean_kappa=np.full(n_st, kappa), apex_merge_gap_m=0.0,
        hold_bridge=hold_gap > 0.0, hold_max_gap_m=hold_gap, hold_kappa_max=hold_kappa)
    return d, nn, lay


def _hold_case_arc(gap_m, radius, hold_gap=8.0, hold_kappa=0.3, half=1.2):
    """The same pair, on a CIRCLE of the given radius -- so |kappa_clean| = 1/radius is real
    GEOMETRY. The core judges curvature with Menger on the clean line (never an additive model),
    so a corner has to be built, not declared."""
    n_st = int(2.0 * np.pi * radius / 0.1)
    th = np.arange(n_st) * (2.0 * np.pi / n_st)
    xy = radius * np.column_stack([np.cos(th), np.sin(th)])
    nv = -np.column_stack([np.cos(th), np.sin(th)])               # +d points at the centre
    s_l = np.arange(n_st) * (2.0 * np.pi * radius / n_st)
    hi, lo = np.full(n_st, half), np.full(n_st, -half)
    i0 = n_st // 4
    i1 = (i0 + int(round(gap_m / (s_l[1] - s_l[0])))) % n_st
    apex, obs = [], []
    for i in (i0, i1):
        apex.append(tuple(xy[i] - 0.55 * nv[i]))                  # SAME side (outside), both
        obs.append((float(xy[i][0]), float(xy[i][1]), 0.15))
    d, nn, _e, _drp, lay = core.build_offset_profile(
        xy, s_l, float(s_l[-1] + (s_l[1] - s_l[0])), nv, apex, None, 0.0, 2.0, 2.0,
        hi_inc=hi, lo_inc=lo, obstacles=obs, obs_margin=0.35, relax_floor=0.30, curvlim=1.5,
        clean_kappa=None, apex_merge_gap_m=0.0,
        hold_bridge=hold_gap > 0.0, hold_max_gap_m=hold_gap, hold_kappa_max=hold_kappa)
    return d, nn, lay


def test_hold_bridge_holds_a_same_side_pair_on_a_straight():
    # Two boxes on the same side, 6 m apart on a straight, with a reach too short for the ramps to
    # overlap. Weaving back to the raceline between them costs two merge inflections and a speed
    # dip, and buys a stretch of racing line the car leaves again immediately.
    d_w, nn_w, lay_w = _hold_case(6.0, hold_gap=0.0)              # bridging OFF
    d_h, nn_h, lay_h = _hold_case(6.0, hold_gap=8.0)              # bridging ON
    assert nn_w == 2 and nn_h == 2, "both humps must be laid either way"
    assert _excursions(d_w) == 2, "without the bridge this is the W (two excursions)"
    assert _excursions(d_h) == 1, "with it the line must HOLD across the pair"
    # holding means the profile never returns to the raceline BETWEEN the two apexes
    mid = (np.arange(900) * 0.1 > 30.5) & (np.arange(900) * 0.1 < 35.5)
    assert np.min(np.abs(d_h[mid])) > 0.05, "the offset must be held, not merely smoothed"
    assert np.min(np.abs(d_w[mid])) < 0.02, "…and the un-bridged one does return to the line"
    # the plateau stays on ONE side and never exceeds the amplitudes it connects
    assert np.all(d_h[mid] * lay_h[0]["laid"] > 0), "the held plateau must stay on the apex side"
    amp = max(abs(a["laid"]) for a in lay_h)
    assert np.max(np.abs(d_h)) <= amp + 1e-6, "a bridge must not swing wider than its apexes"
    print(f"PASS a same-side pair 6 m apart is held (excursions {_excursions(d_w)} -> "
          f"{_excursions(d_h)})")


def test_hold_bridge_gates():
    # It is offered only where holding is the absence of a pointless detour, never a detour itself.
    # TOO FAR APART: the raceline stretch between the boxes is worth taking.
    d_far, nn_far, _l = _hold_case(6.0, hold_gap=4.0)
    assert nn_far == 2 and _excursions(d_far) == 2, "a pair wider than hold_max_gap_m must not hold"
    # IN A CORNER: the line between two boxes IS the apex; holding an offset gives it away.
    # Built as real geometry (a circle of radius 3 m -> |kappa| = 0.33), because the core judges
    # curvature with Menger on the clean line and rightly ignores a declared clean_kappa array.
    d_cor, nn_cor, _l = _hold_case_arc(6.0, radius=3.0, hold_kappa=0.3)
    assert nn_cor == 2 and _excursions(d_cor) == 2, "a bridge must not be laid through a corner"
    assert _excursions(_hold_case_arc(6.0, radius=3.0, hold_kappa=1.0)[0]) == 1, \
        "…and raising the bar past the corner's curvature re-enables it (the gate is what bit)"
    # a GENTLE curve (radius 30 m -> 0.033) is still a straight as far as this rule is concerned
    assert _excursions(_hold_case_arc(6.0, radius=30.0, hold_kappa=0.3)[0]) == 1
    # NO CORRIDOR for the plateau: the held offset has to fit between the walls the whole way. The
    # wall pinches in the 1 m band between the two humps' reaches -- each hump still fits at full
    # amplitude, only a plateau across the pair would hit it.
    d_pin, nn_pin, lay_pin = _hold_case(6.0, pinch=(32.5, 33.5, 0.30))
    assert nn_pin == 2 and abs(lay_pin[0]["laid"]) > 0.45, "the pinch must not shrink the humps"
    assert _excursions(d_pin) == 2, "a plateau that does not fit the corridor must not be laid"
    assert _excursions(_hold_case(6.0)[0]) == 1, "…without the pinch, the same pair holds"
    print("PASS the hold bridge is gated on gap, straightness and corridor room")


def _mixed_case(gap_m, hold_gap=8.0, half=1.2, pinch=None, pinch_neg=None, n_st=900, merge=0.0):
    """Two boxes `gap_m` apart, both ON the raceline, with OPPOSITE-side recorded apexes.

    `pinch` / `pinch_neg` = (x_lo, x_hi, limit) bring the + / - wall in over a band, so a test can
    say which sides are actually available where."""
    xy = np.column_stack([np.arange(n_st) * 0.1, np.zeros(n_st)])
    s_l = np.arange(n_st) * 0.1
    nv = np.column_stack([np.zeros(n_st), -np.ones(n_st)])
    hi, lo = np.full(n_st, half), np.full(n_st, -half)
    if pinch is not None:
        x_lo, x_hi, lim = pinch
        hi[(s_l >= x_lo) & (s_l <= x_hi)] = lim
    if pinch_neg is not None:
        x_lo, x_hi, lim = pinch_neg
        lo[(s_l >= x_lo) & (s_l <= x_hi)] = -lim
    x0 = 30.0
    apex = [(x0, -0.55), (x0 + gap_m, +0.55)]                     # OPPOSITE sides
    obs = [(x0, 0.0, 0.15), (x0 + gap_m, 0.0, 0.15)]
    d, nn, _e, drp, lay = core.build_offset_profile(
        xy, s_l, float(n_st * 0.1), nv, apex, None, 0.0, 2.0, 2.0, hi_inc=hi, lo_inc=lo,
        obstacles=obs, obs_margin=0.35, relax_floor=0.30, curvlim=1.5, clean_kappa=None,
        apex_merge_gap_m=merge, hold_bridge=hold_gap > 0.0, hold_max_gap_m=hold_gap,
        hold_kappa_max=0.3)
    return d, nn, lay, drp


def test_common_side_prepass_unifies_an_accidental_slalom():
    # Each side comes from that obstacle's own reactive apex -- evidence that a side is drivable,
    # chosen one obstacle at a time by a planner looking at one obstacle at a time. Two boxes ON
    # the raceline 6 m apart can therefore end up with opposite sides for no reason but which way
    # the car happened to be displaced, and the global line then crosses between them.
    d_off, nn_off, lay_off, _drp = _mixed_case(6.0, hold_gap=0.0)
    assert nn_off == 2 and _excursions(d_off) == 2, "without the pre-pass this is a slalom"
    assert lay_off[0]["laid"] * lay_off[1]["laid"] < 0, "…on opposite sides"
    d_on, nn_on, lay_on, _drp = _mixed_case(6.0)
    assert nn_on == 2, "unifying must not cost a hump"
    assert lay_on[0]["laid"] * lay_on[1]["laid"] > 0, "both humps must end up on ONE side"
    assert _excursions(d_on) == 1, "…and the line then leaves the raceline once"
    assert sum(bool(a.get("side_unified")) for a in lay_on) == 1, \
        "exactly the minority apex is flipped, and it is reported as such"
    # every laid hump still clears its own box by the floor -- the mirror is measured from the box
    assert all(a["clear"] >= 0.35 - 1e-6 for a in lay_on), [a["clear"] for a in lay_on]
    print(f"PASS an accidental slalom is unified onto one side "
          f"({[round(a['laid'], 2) for a in lay_off]} -> {[round(a['laid'], 2) for a in lay_on]})")


def test_common_side_prepass_follows_the_corridor_and_the_gates():
    # It picks the side that FITS, not the majority: pinching the wall beside the first box on the
    # + side moves the whole group to the - side instead.
    _d, nn, lay, _drp = _mixed_case(6.0, pinch=(29.0, 31.0, 0.30))
    assert nn == 2 and all(a["laid"] < 0 for a in lay), \
        f"a pinched + side must send the group to the - side, got {[a['laid'] for a in lay]}"
    # TOO FAR APART: outside the hold window the two obstacles are not a group at all.
    d_far, _nn, lay_far, _drp = _mixed_case(6.0, hold_gap=4.0)
    assert lay_far[0]["laid"] * lay_far[1]["laid"] < 0, "a pair wider than the window is left alone"
    assert _excursions(d_far) == 2
    # NEITHER SIDE FITS: a GENUINE slalom -- each box's own recorded side is the only one it has,
    # and they are opposite. Blocking the - wall beside box 1 and the + wall beside box 2 leaves no
    # common side, and the crossing between them is real geometry, not an accident.
    _d, nn_b, lay_b, _drp = _mixed_case(6.0, pinch=(34.0, 38.0, 0.30),
                                        pinch_neg=(28.0, 32.0, 0.30))
    assert nn_b == 2 and all(not a.get("side_unified") for a in lay_b), \
        "with no feasible common side the recorded sides must stand"
    assert lay_b[0]["laid"] * lay_b[1]["laid"] < 0, "…and the real slalom is still laid as one"
    print("PASS the common-side pre-pass follows the corridor and its gates")


def test_line_clearance_veto():
    # A line that does not clear an obstacle it CLAIMS to have reshaped must not be published;
    # one that misses an obstacle no hump was laid for is fine (reactive layer's job). Which
    # obstacles were claimed is decided by IDENTITY: `obs_i` indexes `apex_obs`.
    n = make_node()
    n.obs_margin = 0.35
    traj = np.column_stack([np.arange(400) * 0.1, np.arange(400) * 0.1, np.zeros(400)])
    close = core.Obstacle(20.0, 0.20, 0.15)       # line passes 0.05 m from the edge
    far = core.Obstacle(30.0, 1.00, 0.15)
    assert n._check_line_clearance(traj, [far], [{"xy": (30.0, 0.6), "obs_i": 0}],
                                   apex_obs=[far]) is True
    assert n._check_line_clearance(traj, [close], [{"xy": (20.0, 0.6), "obs_i": 0}],
                                   apex_obs=[close]) is False, \
        "a reshaped-but-not-cleared obstacle must veto the publish"
    assert n._check_line_clearance(traj, [close], [], apex_obs=[close]) is True, \
        "an obstacle with no hump laid for it is the reactive layer's job, not a veto"
    # a DROPPED neighbour close enough to be inside the old 1.5 m proximity radius must not veto
    laid_ok = core.Obstacle(20.0, 0.60, 0.15)
    assert n._check_line_clearance(traj, [laid_ok, close], [{"xy": (20.0, 0.9), "obs_i": 0}],
                                   apex_obs=[laid_ok, close]) is True, \
        "a dropped neighbour must not veto a line that is correct for what it claimed"
    # ...and the comparison carries the slack the FIT was allowed to spend. A hump accepted at
    # exactly its floor (the relax ladder lands on the floor by construction) reads back a fraction
    # of a millimetre short after weaving/resampling; vetoing on that leaves the car on a line that
    # clears LESS than the one refused.
    grazing = core.Obstacle(20.0, 0.4998, 0.15)   # 0.3498 m of edge clearance vs a 0.35 floor
    assert n._check_line_clearance(traj, [grazing], [{"xy": (20.0, 0.9), "obs_i": 0}],
                                   apex_obs=[grazing], floors={id(grazing): 0.35}) is True, \
        "a sub-fit_tol shortfall is representation noise, not a broken promise"
    real_short = core.Obstacle(20.0, 0.48, 0.15)  # 0.33 m: 2 cm short, well past fit_tol
    assert n._check_line_clearance(traj, [real_short], [{"xy": (20.0, 0.9), "obs_i": 0}],
                                   apex_obs=[real_short], floors={id(real_short): 0.35}) is False, \
        "a genuine shortfall must still veto"
    print("PASS line clearance vetoes only lines that break their own promise")


def test_breaker_holds_when_the_car_is_far_from_the_pending_line():
    # The breaker's only cap DISCARDED a pending beyond 0.6 m, so anything under that was forced
    # however far the car was from it. Observed: 0.59 m, forced mid-hump after a 5.3 s wait -- the
    # reference stepped half a metre sideways under a car already in an avoidance, the obstacle's
    # d jumped 0.58 m in one cycle as the frame changed, and the two replans that followed were at
    # g_near 0.81 and 0.56 m, inside the 0.9-2.0 m a fresh plan needs.
    n = make_node()
    sa = np.arange(0.0, 40.0, 0.1)
    wp = [types.SimpleNamespace(s_m=s, x_m=s, y_m=0.0) for s in sa]
    pending = types.SimpleNamespace(glb_wpnts=types.SimpleNamespace(wpnts=wp))
    n._pending = pending
    dev = np.zeros_like(sa); dev[(sa > 10.0) & (sa < 18.0)] = 0.59      # the observed deviation
    n._pending_dev = dev
    n._pending_since = 0.0
    n._reactive_active = False
    n._last_vs = 0.5                                   # slow + long wait = the breaker fires
    n._clock.t = 10.0
    n.active = straight_bundle()
    n._publish_active = lambda b: None
    n._publish_coverage = lambda b: None
    n._publish_clearance = lambda: None
    n.pub_update_map = types.SimpleNamespace(publish=lambda m: None)
    n._commit_pending(12.0)
    assert n._pending is pending, "0.59 m from the car must not be forced, and must not be thrown away"
    assert n.active is not pending
    assert n._swap_block.get("deadlock_dev_too_large") == 1, n._swap_block
    # ...and it still un-sticks once the car is near the new line
    n._pending_dev = np.full_like(sa, 0.10)
    n._commit_pending(12.0)
    assert n.active is pending, "within the cap the breaker must still commit"
    print("PASS the deadlock breaker holds when the car is far from the pending line")


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
        # 0.20 m of hump around the car -- inside swap_deadlock_max_dev_commit_m, so the breaker
        # may force it. Past that cap it holds instead; that case is its own test below.
        dev = np.zeros_like(sa); dev[(sa > 10.0) & (sa < 18.0)] = 0.20  # hump around the car
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
    test_apex_does_not_arm_the_rebuild_mid_maneuver()
    test_small_growth_no_retrigger()
    test_retro_association()
    test_newest_wins_shrink_within_the_undershoot_bound()
    test_apex_span_requirement()
    test_apex_is_anchored_abeam_not_where_the_path_was_widest()
    test_apex_undershoot_rejected()
    test_squeeze_apex_is_accepted_at_the_relax_floor()
    test_apex_frozen_while_the_active_line_covers_it()
    test_implausible_apex_rejected()
    test_overshoot_apex_clamped()
    test_neighbor_ramp_does_not_overwrite()
    test_orphan_apex_adopted_on_id_reissue()
    test_set_change_drops_stale_pending()
    test_commit_horizon()
    test_commit_horizon_wrap()
    test_deadlock_breaker()
    test_breaker_holds_when_the_car_is_far_from_the_pending_line()
    test_breaker_refuses_poisoned_pending()
    test_clearance_floor_is_enforced_and_drops_honestly()
    test_amplitude_comes_from_the_obstacle_not_the_apex()
    test_raceline_already_clear_lays_nothing()
    test_clearance_drift_retriggers_once()
    test_minor_apex_refinement_keeps_the_pending()
    test_swap_held_while_trailing_a_close_obstacle()
    test_every_obstacle_gets_a_knot_even_with_no_reactive_apex()
    test_knot_side_comes_from_the_map_not_the_waypoint_bounds()
    test_a_stale_solve_cannot_undo_a_clean_swap()
    test_an_empty_set_can_only_install_the_clean_line()
    test_a_finished_solve_is_collected_before_the_next_is_submitted()
    test_solve_runs_off_the_executor_and_is_collected_later()
    test_swap_gate_tally_names_the_gate_that_held_the_swap()
    test_solves_are_debounced()
    test_a_rebuild_that_changes_nothing_is_not_queued()
    test_a_brief_tracker_dropout_does_not_shrink_the_set()
    test_marker_id_zero_does_not_flip_the_key_scheme()
    test_obstacle_set_change_is_compared_by_id_not_by_order()
    test_shrinking_set_keeps_the_queued_line()
    test_cluster_curvature_budget_is_local()
    test_coverage_outranks_lap_time_in_the_reach_search()
    test_weave_failure_drops_only_the_implicated_humps()
    test_wrap_normals_is_a_correct_lateral_basis()
    test_wrap_normals_has_no_seam_artifact()
    test_hold_bridge_holds_a_same_side_pair_on_a_straight()
    test_hold_bridge_gates()
    test_common_side_prepass_unifies_an_accidental_slalom()
    test_common_side_prepass_follows_the_corridor_and_the_gates()
    test_line_clearance_veto()
    print("ALL PASS")
