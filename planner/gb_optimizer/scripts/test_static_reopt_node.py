#!/usr/bin/env python3
"""static_reopt_node's SAFETY MACHINERY -- swap, publish and concurrency. Not the solver.

Split out of test_static_reopt_apex.py when the hump pipeline was deleted. The name changed with
it, because "apex" described the 29 tests that went, not these: what is checked here is the node
around whichever solver is configured, and every one of them survives the hump because none of
them is about it. Naming it after the apex is how a file like this gets deleted by the next person
clearing out hump code.

WHAT EACH GROUP GUARDS -- this list was needed repeatedly during the rewrite, so it lives here:

  stale-solve epoch guard      test_a_stale_solve_cannot_undo_a_clean_swap
                               test_a_minor_refinement_does_not_discard_the_solve_in_flight
                               test_set_change_drops_stale_pending
  empty-set clean-only         test_an_empty_set_can_only_install_the_clean_line
  solve off the executor       test_solve_runs_off_the_executor_and_is_collected_later
                               test_a_finished_solve_is_collected_before_the_next_is_submitted
                               test_solves_are_debounced
  deadlock breaker             test_breaker_refuses_poisoned_pending
  publish deadband             test_a_rebuild_that_changes_nothing_is_not_queued
  swap gating + its accounting test_swap_held_while_trailing_a_close_obstacle
                               test_swap_gate_tally_names_the_gate_that_held_the_swap
  obstacle-set identity        test_marker_id_zero_does_not_flip_the_key_scheme
                               test_obstacle_set_change_is_compared_by_id_not_by_order
                               test_a_brief_tracker_dropout_does_not_shrink_the_set
  rebuild triggers             test_new_apex_sets_dirty, test_small_growth_no_retrigger,
                               test_apex_does_not_arm_the_rebuild_mid_maneuver,
                               test_minor_apex_refinement_keeps_the_pending,
                               test_the_drift_check_reads_the_LIVE_obstacle_position
  reactive-record association  test_retro_association, test_neighbor_ramp_does_not_overwrite,
                               test_apex_undershoot_rejected, test_implausible_apex_rejected
  lateral basis                test_wrap_normals_is_a_correct_lateral_basis,
                               test_wrap_normals_has_no_seam_artifact
  corridor measurement         test_common_side_prepass_follows_the_corridor_and_the_gates

Several still speak of an "apex": the node records what the REACTIVE layer drove past, and that
record is still how a rebuild is triggered and how an obstacle is associated -- it is no longer
what the solver consumes.

  ~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/test_static_reopt_node.py
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
    n._live_xy = {}                 # marker id -> the layer's UNHELD estimate (see obstacles_cb)
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

def test_implausible_apex_rejected():
    n = make_node()
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    crazy = [(x * 0.5, 1.4 * np.exp(-((x * 0.5 - 5.0) ** 2)),
              1.4 * np.exp(-((x * 0.5 - 5.0) ** 2))) for x in range(20)]   # apex 1.4 > corridor
    n.otwpnts_cb(path_msg(crazy))
    assert ("id", 7) not in n._apex_by_obs, "apex outside the drivable band must be rejected"
    print("PASS implausible (out-of-corridor) apex rejected")

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

def test_the_drift_check_reads_the_LIVE_obstacle_position():
    # THE reason the documented safety net could never fire. static_obstacle_layer HOLDS the pose
    # it publishes inside publish_deadband_m (0.12) to stop estimate noise re-arming a rebuild --
    # and this check was measuring the drift with that same held number, so the cause of a drift
    # and the measurement of it were one stale value. Arithmetic: the enforced floor is 0.35 and
    # the dead-band 0.12, so real clearance can be 0.23 m against the state machine's 0.25 m
    # static-GB requirement -- the line reads BLOCKED, the car trails, and nothing re-solves
    # because "the obstacle did not move". The layer now carries the live estimate in points[0].
    n = make_node()
    n.active = straight_bundle()                       # line along y=0
    n.active.clearance_by_key = {("id", 7): 0.40}
    n._mark_dirty = lambda: setattr(n, "_obstacles_dirty", True)
    held = core.Obstacle(5.0, 0.55, 0.15)              # what the layer PUBLISHES: 0.40 m clear
    assert n._clearance_drifted([held], [7]) is False
    # the same marker, whose live estimate has drifted 0.15 m in -- inside the dead-band, so the
    # published pose does not move at all
    n._live_xy = {7: (5.0, 0.40)}                      # 0.25 m clear: under the 0.30 threshold
    assert n._clearance_drifted([held], [7]) is True, (
        "a drift the publish dead-band is hiding must still re-arm the solve")
    print("PASS the drift check reads the live estimate, not the position the dead-band froze")

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

def test_a_minor_refinement_does_not_discard_the_solve_in_flight():
    # THE STARVATION. The epoch guard exists so a solve submitted for an old obstacle set cannot
    # install a line describing a world that no longer exists. Bumping it on EVERY _mark_dirty made
    # it fire for changes that invalidate nothing: the apex path arms _mark_dirty(keep_pending=True)
    # every time the reactive layer goes idle with a refined apex, a solve takes 200-850 ms, and a
    # refinement landing inside that window discarded it. The discard re-arms the trigger, the next
    # solve meets the next refinement, and no obstacle-aware line is EVER installed.
    #
    # The invariant: the epoch and the pending bundle are invalidated by the same events. If the
    # change is small enough to keep a queued LINE, it is small enough to keep a running SOLVE.
    from concurrent.futures import ThreadPoolExecutor
    n = make_node()
    n._solve_pool = ThreadPoolExecutor(max_workers=1)
    n.reopt_method = "local_window"
    n._obstacles = [core.Obstacle(5.0, -0.2, 0.15)]
    n._obs_ids = [7]
    n._apex_by_obs = {("id", 7): (5.0, 0.4, 0.4)}
    n._build_obstacle_bundle = lambda obstacles, pairs=None: straight_bundle()
    collected = []
    n._finish_rebuild = lambda b, o, r, t: collected.append(r)

    n._obstacles_dirty = True
    n._rebuild_and_swap("first")
    assert n._solve_future is not None, "the fixture must submit a solve"
    n._mark_dirty(keep_pending=True)                 # a MINOR apex refinement while it runs
    n._solve_future.result(timeout=5.0)
    n._collect_solve()
    assert collected == ["first"], (
        "a minor refinement discarded the solve in flight -- with the apex path arming one every "
        "time the reactive layer goes idle, that is a line that is never installed")

    # ...and a change that DOES invalidate still discards it
    collected.clear()
    n._obstacles_dirty = True
    n._rebuild_and_swap("second")
    n._mark_dirty()                                  # the set moved or grew
    n._solve_future.result(timeout=5.0)
    n._collect_solve()
    assert collected == [], "a real set change must still discard the solve it invalidated"
    assert n._obstacles_dirty, "...and must re-arm the trigger"
    n._solve_pool.shutdown(wait=True)
    print("PASS a minor refinement keeps the solve in flight; a real change still discards it")

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
