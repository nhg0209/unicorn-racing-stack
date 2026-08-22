#!/usr/bin/env python3
"""Standalone tests for static_obstacle_layer: confirm / lap-guard / sighting-based unlatch.

No ROS graph needed beyond rclpy init (no spin, no sim):
    source install/setup.bash && python3 planner/gb_optimizer/scripts/test_static_obstacle_layer.py
"""
import math
import sys
import types

import numpy as np
import rclpy
from f110_msgs.msg import Obstacle, ObstacleArray
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray, Wpnt
from std_msgs.msg import Int32MultiArray

from gb_optimizer.static_obstacle_layer import StaticObstacleLayer

TRACK_LEN = 40.0


# The tests construct a REAL StaticObstacleLayer, so rclpy must be initialised. main() does that
# for a standalone run; under pytest there is no main(), and every test failed with
# NotInitializedException -- i.e. the whole file counted as "collected" while gating nothing.
try:
    import pytest

    @pytest.fixture(scope="module", autouse=True)
    def _rclpy_context():
        if not rclpy.ok():
            rclpy.init()
            yield
            rclpy.shutdown()
        else:
            yield
except ImportError:                                   # standalone run without pytest installed
    pass


def make_node():
    node = StaticObstacleLayer()
    wp = WpntArray()
    for s in (0.0, TRACK_LEN):
        w = Wpnt()
        w.s_m = s
        wp.wpnts.append(w)
    node.glb_cb(wp)
    return node


def odom(s, d=0.0, vs=0.0):
    m = Odometry()
    m.pose.pose.position.x = s
    m.pose.pose.position.y = d
    m.twist.twist.linear.x = vs
    return m


def det(x, y, s, vs=0.0, visible=True, size=0.3):
    o = Obstacle()
    o.x_m, o.y_m, o.s_center = float(x), float(y), float(s)
    o.vs, o.vd = float(vs), 0.0
    o.is_static, o.is_visible = True, visible
    o.size = size
    return o


def arr(*obstacles):
    m = ObstacleArray()
    m.obstacles = list(obstacles)
    return m


def confirm_obstacle(node, x=3.0, y=0.0, s=10.0, settle=False):
    """`settle` also drives the track past publish_seed_hits, i.e. to where its published position
    is HELD. Confirmation alone no longer freezes it -- see the seeding note in publish_cb."""
    for _ in range(max(node.confirm_hits, node.publish_seed_hits if settle else 0)):
        node.obstacles_cb(arr(det(x, y, s)))
    assert node._tracks and node._tracks[0].confirmed, "obstacle should be confirmed"
    return node._tracks[0]


def test_confirm_and_unlatch():
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0))                       # gap = 3.0, inside [1.0, 4.0]
    for i in range(node.unlatch_clear_msgs):
        assert node._tracks[0].confirmed, f"unlatched too early at clear msg {i}"
        node.obstacles_cb(arr())                    # clear view, no detection
    assert not node._tracks[0].confirmed, \
        "track should be unlatched after unlatch_clear_msgs clear views"
    print("PASS confirm + sighting-based unlatch")


def test_unlatch_demotes_and_keeps_the_identity():
    # ~0.5 s of clear views is enough to stop PUBLISHING an obstacle and nowhere near enough to
    # forget it. Deleting the track threw away its marker_id, so a re-detection a moment later
    # arrived as a NEW obstacle with a NEW id -- which every consumer reads as a set change. On the
    # real run one detection gap produced a demote, a re-promote under a fresh id, two consecutive
    # /global_waypoints swaps and a discarded pending bundle, with the collision 28 ms after the
    # second swap.
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    mid = t.marker_id
    node.frenet_cb(odom(7.0))
    for _ in range(node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    assert node._tracks, "an unlatched track must be KEPT, not deleted"
    assert not node._tracks[0].confirmed, "…but must stop being published"
    assert node._tracks[0].marker_id == mid, "the identity must survive the demotion"
    assert node._tracks[0].hits == 0, "re-detection must earn confirm_hits again"
    # re-detection inside the match gate re-promotes THE SAME track under THE SAME id
    for _ in range(node.confirm_hits):
        node.obstacles_cb(arr(det(3.0, 0.0, 10.0)))
    assert len(node._tracks) == 1, "re-detection must not create a second track"
    assert node._tracks[0].confirmed and node._tracks[0].marker_id == mid, \
        "the re-confirmed obstacle must come back under its original id"
    print("PASS unlatch demotes and keeps the marker id")


def test_demoted_track_decays_at_the_lap_boundary():
    # Deletion belongs to the lap accounting, which is the evidence that can support it.
    node = make_node()
    confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0))
    for _ in range(node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    assert node._tracks and not node._tracks[0].confirmed
    # the lap it was demoted on is a lap it WAS seen on, so it survives that boundary
    node._on_lap_complete()
    assert node._tracks, "the demotion lap itself is not evidence of removal"
    node._tracks[0].opportunity_this_lap = True     # next lap: drove past its spot, saw nothing
    node._on_lap_complete()
    assert not node._tracks, "a demoted track never seen again must decay at the lap boundary"
    print("PASS a demoted track decays at the lap boundary")


def test_line_swap_suspends_the_streak():
    # After a swap, s is re-anchored here, ego_s comes from the republisher on the NEW line and the
    # tracker's own conversion follows a message later -- "no detection at that spot" is not a
    # judgment worth making from that snapshot.
    node = make_node()
    confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0))
    for _ in range(node.unlatch_clear_msgs - 1):
        node.obstacles_cb(arr())
    assert node._tracks[0].clear_streak > 0
    wp = WpntArray()                                 # a DIFFERENT line arrives
    for s_m, x_m in ((0.0, 0.0), (TRACK_LEN, 1.0)):
        w = Wpnt(); w.s_m = s_m; w.x_m = x_m
        wp.wpnts.append(w)
    node.glb_cb(wp)
    for _ in range(3 * node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    assert node._tracks[0].confirmed, "the streak must be suspended right after a swap"
    assert node._tracks[0].clear_streak == 0
    print("PASS a line swap suspends the unlatch streak")


def test_ego_offline_suspends_streak():
    # Mid-avoidance (ego off the raceline) the view of the very obstacle being avoided is
    # unreliable — a live obstacle was unlatched DURING its own avoidance (set flap 1->0->1).
    node = make_node()
    confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0, d=0.4))                # in window, but ego is OFF the line
    for _ in range(3 * node.unlatch_clear_msgs):
        node.obstacles_cb(arr())                    # no detection at all
    assert node._tracks[0].confirmed, "off-line ego must SUSPEND the unlatch streak"
    node.frenet_cb(odom(7.0, d=0.0))                # back on the raceline
    for _ in range(node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    assert not node._tracks[0].confirmed, "back on the line the streak must run to unlatch"
    print("PASS off-line ego suspends unlatch streak")


def test_sighting_resets_streak():
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0))
    for _ in range(node.unlatch_clear_msgs - 1):
        node.obstacles_cb(arr())
    node.obstacles_cb(arr(det(3.0, 0.0, 10.0)))     # visible sighting -> reset
    assert node._tracks and node._tracks[0].clear_streak == 0, "sighting must reset the streak"
    print("PASS sighting resets streak")


def test_memory_detection_does_not_reset():
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0))
    for _ in range(node.unlatch_clear_msgs):
        node.obstacles_cb(arr(det(3.0, 0.0, 10.0, visible=False)))  # tracker memory, not a view
    assert not node._tracks[0].confirmed, \
        "is_visible=False detections must not defeat the unlatch streak"
    print("PASS remembered (is_visible=False) detection does not reset streak")


def test_occlusion_suspends():
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0))
    opponent = det(1.0, 0.0, 8.5, vs=2.0)           # dynamic, gap 1.5 < track gap 3.0
    for _ in range(3 * node.unlatch_clear_msgs):
        node.obstacles_cb(arr(opponent))
    assert node._tracks[0].confirmed, "streak must be suspended while the opponent occludes the spot"
    assert node._tracks[0].clear_streak == 0
    print("PASS occlusion suspends streak")


def test_a_track_behind_the_car_never_accumulates_a_clear_streak():
    # The fast unlatch judges "I looked where it should be and it was not there", which is only
    # meaningful on the APPROACH. A track the car has already driven past is not being looked at,
    # so clear views of its spot mean nothing -- and if they counted, every confirmed obstacle
    # would be demoted once per lap on the way out, taking the obstacle SET with it and forcing a
    # /global_waypoints swap each lap.
    node = make_node()
    confirm_obstacle(node, s=10.0)
    for gap in (-0.5, -3.0, -8.0):                 # the box BEHIND the car, by |gap| metres
        node.frenet_cb(odom((10.0 - gap) % TRACK_LEN))
        for _ in range(5 * node.unlatch_clear_msgs):
            node.obstacles_cb(arr())               # no detection at all
        assert node._tracks[0].confirmed, f"a box {abs(gap):.1f} m BEHIND must not be unlatched"
        assert node._tracks[0].clear_streak == 0, "and must not even accumulate a streak"
    # ...and the same track still unlatches normally once it is ahead again, in the window
    node.frenet_cb(odom(7.0))                      # gap 3.0, inside [1.0, 5.0]
    for _ in range(node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    assert not node._tracks[0].confirmed, "the approach window must still work"
    print("PASS a track behind the car accumulates no clear streak")


def published_xy(node):
    """The positions the layer actually PUBLISHES (skipping the DELETEALL marker)."""
    pub = []
    node.pub = types.SimpleNamespace(publish=lambda arr: pub.append(arr))
    node.publish_cb()
    return [(round(m.pose.position.x, 3), round(m.pose.position.y, 3))
            for m in pub[-1].markers if m.action == 0]


def establish_hold(node, x, y, s, closest_gap=1.0, n_obs=8, t0_ns=5_000_000_000):
    """Confirm a track and drive the car PAST it, which is what seeds the hold now.

    The published position freezes on the closest-approach estimate, so a test that wants a held
    position has to supply an approach. Returns the track.
    """
    dt_ns = int(1e9 / DET_HZ)
    gaps = np.linspace(closest_gap + 3.0, closest_gap, n_obs)
    for k, g in enumerate(gaps):
        node.frenet_cb(odom(s - float(g)))
        msg = obs_msg(x, y, s, t0_ns + k * dt_ns)
        for _ in range(REPEATS):
            node.obstacles_cb(msg)
            node.publish_cb()
    node.frenet_cb(odom(s + 5.0))          # past it -> the hold is taken
    node.publish_cb()
    return node._tracks[0]

def test_publish_position_holds_still_under_estimate_noise():
    # The estimate never settles -- the same physical box wanders 0.04-0.06 m lap to lap because
    # each approach sees a different face of it. Every wander was an obstacle-set change
    # downstream: 12 of 25 set changes in one run were nothing else, and each cost a re-solve, a
    # swap and a FrenetConverter rebuild in three consumers.
    node = make_node()
    t = establish_hold(node, 3.0, 0.0, 10.0)
    held = published_xy(node)[0]
    assert abs(held[0] - 3.0) < 1e-6 and abs(held[1]) < 1e-6, held
    for dx, dy in ((0.05, 0.02), (-0.04, 0.03), (0.06, -0.05), (0.03, 0.06)):
        for _ in range(6):                      # let the EMA settle on the wandered position
            node.obstacles_cb(arr(det(3.0 + dx, 0.0 + dy, 10.0)))
        assert published_xy(node) == [held], \
            f"a {np.hypot(dx, dy):.3f} m wander must not move the published position"
    assert np.hypot(t.x - 3.0, t.y - 0.0) > 0.0, "…while the internal estimate did follow it"
    print(f"PASS the published position holds still through wanders up to "
          f"{max(np.hypot(*d) for d in ((0.05,0.02),(0.04,0.03),(0.06,0.05),(0.03,0.06))):.3f} m")


def test_the_hold_is_not_taken_while_the_estimate_is_still_biased():
    """The hold used to be seeded by SIGHTING COUNT -- originally at confirm_hits, then at
    publish_seed_hits. Both are "the first N frames", and on the car those frames are the ones
    taken at the range the box becomes visible from (~7 m) off the sparsest returns. With a = 0.3
    the EMA still carries 17% of the initial error at 5 updates, so what got frozen for good was
    the least-informed number the track would ever hold, and every refinement the approach bought
    was then thrown away inside the dead-band. The re-opt fits its humps to that number: floor
    0.35 - 0.12 = 0.23 m of real clearance, under the state machine's 0.25 m static-GB
    requirement, i.e. TRAILING behind a box the line actually clears.

    The count is gone as the trigger. The hold is taken when the car has PASSED, at the estimate
    from the closest point of that pass, and nothing freezes before then.
    """
    first, true = (3.00, 0.0), (3.10, 0.0)         # first sightings biased 0.10 m off
    node = make_node()
    dt_ns = int(1e9 / DET_HZ)
    t0 = 5_000_000_000
    k = 0
    for _ in range(node.confirm_hits):             # biased frames, taken far out
        node.frenet_cb(odom(10.0 - 7.0))
        msg = obs_msg(first[0], first[1], 10.0, t0 + k * dt_ns); k += 1
        for _ in range(REPEATS):
            node.obstacles_cb(msg)
            node.publish_cb()
    assert node._tracks[0].confirmed
    assert node._tracks[0].pub_x is None, \
        "the hold was taken while the car was still 7 m away, on the biased frames"
    for g in (5.0, 4.0, 3.0, 2.0, 1.0, 0.5):       # the estimate converges as the car closes
        node.frenet_cb(odom(10.0 - g))
        msg = obs_msg(true[0], true[1], 10.0, t0 + k * dt_ns); k += 1
        for _ in range(REPEATS):
            node.obstacles_cb(msg)
            node.publish_cb()
    node.frenet_cb(odom(15.0))                     # past it
    node.publish_cb()
    held = published_xy(node)[0]
    assert abs(held[0] - true[0]) < 0.02, (
        f"held {held}, which is nearer the biased first sightings {first} than the converged "
        f"estimate {node._tracks[0].x:.3f} the closest pass measured")
    # ...and once taken it IS held: sub-dead-band noise does not move it
    for _ in range(20):
        node.obstacles_cb(arr(det(true[0] + 0.05, true[1] + 0.03, 10.0)))
        node.publish_cb()
    assert published_xy(node)[0] == held, "after the pass the dead-band must still hold"
    print(f"PASS nothing freezes at 7 m; the hold is the closest-approach estimate {held}, "
          f"true {true}")


def test_the_live_estimate_is_published_alongside_the_held_pose():
    # The hold is a dead-band on the SET -- it stops estimate noise re-arming a rebuild. But
    # static_reopt_node._clearance_drifted asks "has this box drifted into the line I am
    # following?", and reading the held pose made the cause of a drift and the measurement of it
    # the same stale number: the safety net could not fire by construction. The live estimate now
    # travels in points[0], which a CYLINDER marker does not use and no existing consumer reads.
    node = make_node()
    establish_hold(node, 3.0, 0.0, 10.0)
    assert published_xy(node) == [(3.0, 0.0)]             # the pass seeded the hold here
    for _ in range(20):
        node.obstacles_cb(arr(det(3.08, 0.05, 10.0)))     # inside the dead-band
    pub = []
    node.pub = types.SimpleNamespace(publish=lambda a: pub.append(a))
    node.publish_cb()
    marks = [m for m in pub[-1].markers if m.action == 0]
    assert len(marks) == 1
    m = marks[0]
    assert (round(m.pose.position.x, 3), round(m.pose.position.y, 3)) == (3.0, 0.0), \
        "the POSE must still be the held one"
    assert m.points, "the live estimate must be published alongside it"
    live = (m.points[0].x, m.points[0].y)
    assert abs(live[0] - node._tracks[0].x) < 1e-9 and abs(live[1] - node._tracks[0].y) < 1e-9
    assert math.hypot(live[0] - m.pose.position.x, live[1] - m.pose.position.y) > 0.05, \
        "this fixture must actually separate the two"
    print(f"PASS the live estimate ({live[0]:.2f}, {live[1]:.2f}) rides with the held pose "
          f"({m.pose.position.x:.2f}, {m.pose.position.y:.2f})")


def test_publish_position_follows_a_real_move_at_once():
    # A box that has actually been moved crosses the dead band immediately -- reactivity is kept.
    node = make_node()
    confirm_obstacle(node, x=3.0, y=0.0, s=10.0, settle=True)
    published_xy(node)
    for _ in range(8):
        node.obstacles_cb(arr(det(3.51, -0.16, 10.0)))   # the 0.53 m move seen in the log
    got = published_xy(node)
    assert got and np.hypot(got[0][0] - 3.0, got[0][1] - 0.0) > 0.3, \
        f"a real move must be republished at once, got {got}"
    print(f"PASS a real move is republished immediately (now at {got[0]})")


def test_raw_detection_suppresses_the_streak_but_a_removed_box_still_unlatches():
    # A freshly (re-)created track has staticFlag = None for its first frames, so the tracker does
    # not put it on /tracking/obstacles at all -- and the clear streak, reading only that topic,
    # counted "there but unclassified" as "gone". The raw detections separate the two.
    node = make_node()
    confirm_obstacle(node, x=3.0, y=0.0, s=10.0)
    node.frenet_cb(odom(7.0))                        # gap 3.0, inside the window
    node.raw_obstacles_cb(arr(det(3.02, 0.01, 10.0)))   # detected, not yet classified
    for _ in range(4 * node.unlatch_clear_msgs):
        node.obstacles_cb(arr())                     # nothing on the CLASSIFIED topic
    assert node._tracks[0].confirmed, "a detected-but-unclassified box must not be demoted"
    assert node._tracks[0].clear_streak == 0, "and must not accumulate a streak"

    # ...and a box that has physically been taken away appears on NEITHER topic, so the fast
    # unlatch still does its job -- that is the whole purpose and it is untouched.
    node.raw_obstacles_cb(arr())
    for _ in range(node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    assert not node._tracks[0].confirmed, "a genuinely removed box must still unlatch"

    # a raw detection somewhere else entirely must not shield it either
    node2 = make_node()
    confirm_obstacle(node2, x=3.0, y=0.0, s=10.0)
    node2.frenet_cb(odom(7.0))
    node2.raw_obstacles_cb(arr(det(30.0, 30.0, 25.0)))
    for _ in range(node2.unlatch_clear_msgs):
        node2.obstacles_cb(arr())
    assert not node2._tracks[0].confirmed, "a detection elsewhere must not suppress the streak"
    print("PASS a raw detection suppresses the streak; a removed box still unlatches")


def test_reopt_line_active_suspends_the_streak_even_at_d_zero():
    """THE BUG. Reopt line active + ego reads d ~ 0 + no detection -> must NOT unlatch.

    /car_state/odom_frenet is computed against /global_waypoints, and on a reopt lap that IS the
    re-optimized line. The car passes the box half a metre wide while reading d ~ 0, so the
    ego_d guard -- whose whole job is "do not count a miss while avoiding" -- switches itself
    off in exactly the lap it is needed. The streak then ran free, demoted the box mid-avoidance,
    emptied the set, and the clean line came back; next lap the box was re-detected and the whole
    thing repeated. Observed as reopt/clean laps alternating indefinitely.

    static_reopt_node names the obstacles its ACTIVE line is shaped around on
    /static_reopt/active_cover; that, not d, is the evidence.
    """
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0, d=0.0))                # ON the published line -- but it is the
    node.reopt_cover_cb(Int32MultiArray(data=[t.marker_id]))   # ...line built for THIS box
    for _ in range(3 * node.unlatch_clear_msgs):
        node.obstacles_cb(arr())                    # no detection at all, for three streaks
    assert node._tracks[0].confirmed, (
        "the global line is shaped around this box and the car is driving it, but the streak "
        "ran anyway -- d ~ 0 was read as 'not avoiding'")
    assert node._tracks[0].clear_streak == 0, "the streak accumulated instead of suspending"
    print("PASS a box the ACTIVE reopt line covers does not unlatch at d ~ 0")


def test_the_suspension_lifts_when_the_clean_line_comes_back():
    """The suspension must not be a latch of its own: swapping back to clean re-enables removal.

    static_reopt_node publishes an EMPTY active_cover on the clean swap, in the same handler that
    swaps the line, so this arrives without waiting for anything else.
    """
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0, d=0.0))
    node.reopt_cover_cb(Int32MultiArray(data=[t.marker_id]))
    for _ in range(node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    assert node._tracks[0].confirmed
    node.reopt_cover_cb(Int32MultiArray(data=[]))   # swapped back to the CLEAN line
    for _ in range(node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    assert not node._tracks[0].confirmed, (
        "with the clean line driving, a box that is really gone must still unlatch")
    print("PASS the suspension lifts with the clean line, it is not a second latch")


def test_only_the_covered_box_is_suspended():
    """A cover set names ids; a box the line is NOT shaped around must still unlatch normally."""
    node = make_node()
    # both boxes confirmed together, so each gets its own marker_id (confirm_obstacle returns
    # _tracks[0] whichever one you meant, which is why this does not use it)
    for _ in range(node.confirm_hits):
        node.obstacles_cb(arr(det(3.0, 0.0, 10.0), det(3.0, 6.0, 16.0)))
    assert len(node._tracks) == 2 and all(t.confirmed for t in node._tracks), node._tracks
    covered = min(node._tracks, key=lambda t: t.s)
    other = max(node._tracks, key=lambda t: t.s)
    assert covered.marker_id != other.marker_id
    node.reopt_cover_cb(Int32MultiArray(data=[covered.marker_id]))
    node.frenet_cb(odom(13.0, d=0.0))               # window over `other`
    for _ in range(3 * node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    by_id = {t.marker_id: t for t in node._tracks}
    assert by_id[covered.marker_id].confirmed, "the covered box was unlatched"
    assert not by_id[other.marker_id].confirmed, (
        "an uncovered box stopped unlatching -- the suspension is too broad")
    print("PASS suspension applies per obstacle id, not to the whole set")


# ---------------------------------------------------------------------------------------
# 40 Hz republish of a 10 Hz detection -- the real car's chain
# ---------------------------------------------------------------------------------------

DET_HZ, PUB_HZ = 10.0, 40.0
REPEATS = int(PUB_HZ / DET_HZ)          # 4 timer messages carry each detection


def obs_msg(x, y, s, stamp_ns, vs=0.0, visible=True, size=0.3):
    """One /tracking/obstacles message. The stamp is the DETECTION's, which is what the real
    tracker publishes: multi_tracking stamps every timer message with self.current_stamp, and
    that only advances when /detect/raw_obstacles delivers a new frame."""
    m = arr(det(x, y, s, vs=vs, visible=visible, size=size))
    m.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    m.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    return m


def feed_approach(node, xs, ys, ss, ego_s, t0_ns=1_000_000_000,
                  det_hz=DET_HZ, pub_hz=PUB_HZ):
    """Drive `len(xs)` REAL observations through a pub_hz timer, repeating each one.

    Returns the wall-clock seconds consumed, so a test can say what a message count is worth.
    """
    reps = int(round(pub_hz / det_hz))
    dt_ns = int(1e9 / det_hz)
    for k, (x, y, s) in enumerate(zip(xs, ys, ss)):
        node.frenet_cb(odom(ego_s[k]))
        msg = obs_msg(x, y, s, t0_ns + k * dt_ns)
        for _ in range(reps):           # the timer republishes the SAME detection
            node.obstacles_cb(msg)
            node.publish_cb()           # ...and the publish timer runs alongside it
    return len(xs) / det_hz


def test_a_repeated_detection_is_not_new_evidence():
    """THE BUG. /tracking/obstacles runs on a 40 Hz timer and republishes the same detection four
    times per real 10 Hz frame. `hits` counted messages, so confirm_hits 5 was 1.25 observations
    and publish_seed_hits 15 was 3.75 -- the published position froze on under four looks, all of
    them taken at the range the box was first visible from (~7 m), which is the worst data the
    track will ever have. The estimate improved as the car closed in and the dead-band threw it
    away.
    """
    node = make_node()
    # eight REAL observations, ego closing from 7 m to 0.5 m away from a box at s = 10
    n = 8
    ego = [10.0 - g for g in (7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5)]
    feed_approach(node, [3.0] * n, [0.0] * n, [10.0] * n, ego)
    t = node._tracks[0]
    assert t.obs_count == n, (
        f"counted {t.obs_count} observations from {n} detections repeated "
        f"{REPEATS}x -- the timer repeats are being counted as evidence")
    print(f"PASS {n} real detections at {DET_HZ:.0f} Hz republished at {PUB_HZ:.0f} Hz "
          f"count as {t.obs_count} observations, not {n * REPEATS} hits")


def test_the_hold_freezes_on_the_closest_pass_not_the_first_frames():
    """The published position must be the estimate from the CLOSEST approach, and must not freeze
    before the car has actually passed the box."""
    node = make_node()
    # The measured centre walks IN as the car closes and back OUT as it leaves: the lidar sees a
    # different face of the box at every range, so the approach is not monotone and "the newest
    # estimate" is not the best one either. Truth 3.00, best seen at the closest pass.
    xs  = [3.40, 3.34, 3.26, 3.18, 3.08, 3.01, 3.00, 3.09, 3.20, 3.33]
    gaps = [7.0,  6.0,  5.0,  4.0,  3.0,  2.0,  1.0, -1.0, -2.5, -4.0]
    ego = [10.0 - g for g in gaps]
    feed_approach(node, xs, [0.0] * len(xs), [10.0] * len(xs), ego)
    node.frenet_cb(odom(30.0))                  # obstacle now well behind
    node.publish_cb()
    t = node._tracks[0]
    assert t.pub_x is not None, "never froze after the pass"
    truth = 3.00
    err_hold = abs(t.pub_x - truth)
    err_live = abs(t.x - truth)          # what freezing "now", at the end, would have kept
    err_first = abs(xs[3] - truth)       # what the first-N-frames rule reached (~4 observations)
    # The claim is not "the hold equals the truth" -- it cannot, because the estimate is an EMA
    # with a = 0.3 and it lags a walking measurement by design. The claim is that freezing on the
    # closest pass beats BOTH of the things it replaced: the first frames, and whatever the live
    # estimate happens to be when the timer next looks.
    assert err_hold < err_first, (
        f"hold {t.pub_x:.3f} (err {err_hold:.3f}) is no better than the first-frames estimate "
        f"{xs[3]:.3f} (err {err_first:.3f}) -- the freeze is still range-biased")
    assert err_hold < err_live, (
        f"hold {t.pub_x:.3f} (err {err_hold:.3f}) is worse than the live estimate "
        f"{t.x:.3f} (err {err_live:.3f}) -- freezing bought nothing")
    assert t.hold_gap is not None and t.hold_gap <= 1.5, (
        f"hold taken from {t.hold_gap} m; the closest pass was 1.0 m")
    print(f"PASS hold {t.pub_x:.3f} (err {err_hold:.3f}) from {t.hold_gap:.1f} m beats "
          f"first-frames {xs[3]:.3f} (err {err_first:.3f}) and live {t.x:.3f} "
          f"(err {err_live:.3f}); residual is EMA lag, not freeze point")


def test_a_closer_later_estimate_beats_the_deadband():
    """A hold taken far away must be cheap to correct; one taken close must not wobble.

    Same absolute correction, two different holds: the far one republishes, the near one does not.
    """
    node = make_node()
    xs = [3.40] * 6
    ego_far = [10.0 - g for g in (7.0, 6.8, 6.6, 6.4, 6.2, 6.0)]     # never gets close
    feed_approach(node, xs, [0.0] * 6, [10.0] * 6, ego_far)
    node.frenet_cb(odom(30.0))
    node.publish_cb()
    far = node._tracks[0]
    assert far.pub_x is not None
    far_band = node._deadband_for(far)
    assert far_band < node.publish_deadband_m, (
        f"a hold taken {far.hold_gap:.1f} m away got the full {node.publish_deadband_m} m "
        f"dead-band -- a poor estimate is being defended as hard as a good one")
    print(f"PASS dead-band scales with the range the hold was taken at "
          f"({far.hold_gap:.1f} m -> {far_band:.3f} m vs base {node.publish_deadband_m:.2f} m)")


def test_window_exit_resets():
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0))
    for _ in range(node.unlatch_clear_msgs - 1):
        node.obstacles_cb(arr())
    node.frenet_cb(odom(9.5))                       # gap 0.5 < unlatch_gap_min -> leave window
    node.obstacles_cb(arr())
    assert node._tracks and node._tracks[0].clear_streak == 0, "leaving the window must reset"
    print("PASS window exit resets streak")


def test_lap_guard():
    node = make_node()
    # seam jitter without progress: park at the seam and flicker across it
    for s in (39.9, 0.05, 39.9, 0.05, 39.9, 0.05):
        node.frenet_cb(odom(s))
    assert node._lap == 0, "seam flicker without progress must not count laps"
    # a genuine full lap of forward progress does count
    s = 0.1
    while s < TRACK_LEN:
        node.frenet_cb(odom(s))
        s += 0.5
    node.frenet_cb(odom(0.05))
    assert node._lap == 1, "a full lap of forward progress must count exactly once"
    print("PASS lap forward-progress guard")


def test_unlatch_streak_scales_with_speed():
    # The streak must COMPLETE within one approach, and the ego is only inside the 4 m window for
    # (gap_max-gap_min)/v seconds. At a flat 20 messages that is impossible above 8 m/s -- the
    # fast-unlatch path silently stopped existing at racing speed.
    node = make_node()
    node.frenet_cb(odom(0.0, vs=1.0))
    assert node._clear_msgs_needed() == node.unlatch_clear_msgs, "slow ego: full requirement"
    node.frenet_cb(odom(0.0, vs=8.0))
    assert node._clear_msgs_needed() == 20, "at 8 m/s exactly 20 messages still fit"
    node.frenet_cb(odom(0.0, vs=16.0))
    n_fast = node._clear_msgs_needed()
    assert n_fast < node.unlatch_clear_msgs, "fast ego: fewer messages fit in the window"
    assert n_fast >= node.unlatch_clear_msgs_min, "...but never fewer than the floor"
    node.frenet_cb(odom(0.0, vs=200.0))
    assert node._clear_msgs_needed() == node.unlatch_clear_msgs_min, "floor holds at any speed"
    # ...and the streak can now actually complete at speed
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0, vs=16.0))            # gap 3.0 m -> inside [1, 5]
    for _ in range(node._clear_msgs_needed()):
        node.obstacles_cb(arr())
    assert not node._tracks[0].confirmed, "a clear-view streak must be completable at racing speed"
    print("PASS unlatch streak requirement scales with speed")


def test_memory_frame_does_not_confirm_or_feed_a_track():
    # is_visible=False is tracker MEMORY. It used to reach _associate, so it ran the EMA update,
    # incremented hits, set seen_this_lap and reset miss_laps -- a track could be CONFIRMED, and
    # kept alive across laps, purely on the tracker replaying its own memory.
    node = make_node()
    for _ in range(node.confirm_hits * 2):
        node.obstacles_cb(arr(det(3.0, 0.0, 10.0, visible=False)))
    assert not node._tracks, "memory frames alone must never create or confirm a track"
    # a confirmed track must not have its lap accounting refreshed by memory either
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    t.seen_this_lap = False
    t.miss_laps = 1
    hits_before = t.hits
    node.obstacles_cb(arr(det(3.0, 0.0, 10.0, visible=False)))
    assert t.hits == hits_before and not t.seen_this_lap and t.miss_laps == 1, \
        "a memory frame must not count as an observation"
    print("PASS a remembered detection neither confirms nor feeds a track")


def test_track_s_reanchors_on_a_line_swap():
    # static_reopt swaps /global_waypoints for a line whose arc length differs, so an `s` captured
    # on the old line names a different place on the new one -- and every gap this node computes is
    # (t.s - ego_s) against an ego_s from the NEW line.
    node = StaticObstacleLayer()
    line_a = WpntArray()
    for k in range(41):
        w = Wpnt(); w.s_m = float(k); w.x_m = float(k); w.y_m = 0.0
        line_a.wpnts.append(w)
    node.glb_cb(line_a)
    t = confirm_obstacle(node, x=10.0, y=0.0, s=10.0)
    assert abs(t.s - 10.0) < 1e-6
    # same geometry republished -> no churn
    node.glb_cb(line_a)
    assert abs(t.s - 10.0) < 1e-6, "an unchanged line must not re-anchor anything"
    # swapped line: an avoidance hump between x=2 and x=8 bulges the line and LENGTHENS it, so
    # every station past the hump carries a larger s than it did on the clean line.
    import math
    line_b = WpntArray()
    xs = [float(k) for k in range(41)]
    ys = [0.6 * math.exp(-((x - 5.0) ** 2) / 2.0) for x in xs]
    acc = 0.0
    for k, (x, y) in enumerate(zip(xs, ys)):
        if k:
            acc += math.hypot(x - xs[k - 1], y - ys[k - 1])
        w = Wpnt(); w.s_m = acc; w.x_m = x; w.y_m = y
        line_b.wpnts.append(w)
    node.glb_cb(line_b)
    expected = line_b.wpnts[10].s_m            # nearest point on the new line to the track at x=10
    assert abs(t.s - expected) < 1e-6, f"s must follow the new parameterisation, got {t.s}"
    assert t.s > 10.0, "the swapped line is longer, so the station past the hump moved"
    print(f"PASS track s re-anchored on a line swap (10.00 -> {t.s:.3f} m)")


def main():
    rclpy.init()
    try:
        for fn in (test_raw_detection_suppresses_the_streak_but_a_removed_box_still_unlatches,
                   test_publish_position_holds_still_under_estimate_noise,
                   test_the_hold_is_not_taken_while_the_estimate_is_still_biased,
                   test_a_repeated_detection_is_not_new_evidence,
                   test_the_hold_freezes_on_the_closest_pass_not_the_first_frames,
                   test_a_closer_later_estimate_beats_the_deadband,
                   test_the_live_estimate_is_published_alongside_the_held_pose,
                   test_publish_position_follows_a_real_move_at_once,
                   test_confirm_and_unlatch, test_unlatch_demotes_and_keeps_the_identity,
                   test_demoted_track_decays_at_the_lap_boundary,
                   test_line_swap_suspends_the_streak, test_ego_offline_suspends_streak,
                   test_reopt_line_active_suspends_the_streak_even_at_d_zero,
                   test_the_suspension_lifts_when_the_clean_line_comes_back,
                   test_only_the_covered_box_is_suspended,
                   test_sighting_resets_streak, test_memory_detection_does_not_reset,
                   test_occlusion_suspends, test_window_exit_resets,
                   test_a_track_behind_the_car_never_accumulates_a_clear_streak,
                   test_lap_guard,
                   test_unlatch_streak_scales_with_speed,
                   test_memory_frame_does_not_confirm_or_feed_a_track,
                   test_track_s_reanchors_on_a_line_swap):
            fn()
    finally:
        rclpy.shutdown()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
