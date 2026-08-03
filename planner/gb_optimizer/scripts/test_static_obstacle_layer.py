#!/usr/bin/env python3
"""Standalone tests for static_obstacle_layer: confirm / lap-guard / sighting-based unlatch.

No ROS graph needed beyond rclpy init (no spin, no sim):
    source install/setup.bash && python3 planner/gb_optimizer/scripts/test_static_obstacle_layer.py
"""
import sys

import rclpy
from f110_msgs.msg import Obstacle, ObstacleArray
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray, Wpnt

from gb_optimizer.static_obstacle_layer import StaticObstacleLayer

TRACK_LEN = 40.0


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


def confirm_obstacle(node, x=3.0, y=0.0, s=10.0):
    for _ in range(node.confirm_hits):
        node.obstacles_cb(arr(det(x, y, s)))
    assert node._tracks and node._tracks[0].confirmed, "obstacle should be confirmed"
    return node._tracks[0]


def test_confirm_and_unlatch():
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0))                       # gap = 3.0, inside [1.0, 4.0]
    for i in range(node.unlatch_clear_msgs):
        assert node._tracks, f"unlatched too early at clear msg {i}"
        node.obstacles_cb(arr())                    # clear view, no detection
    assert not node._tracks, "track should be unlatched after unlatch_clear_msgs clear views"
    print("PASS confirm + sighting-based unlatch")


def test_ego_offline_suspends_streak():
    # Mid-avoidance (ego off the raceline) the view of the very obstacle being avoided is
    # unreliable — a live obstacle was unlatched DURING its own avoidance (set flap 1->0->1).
    node = make_node()
    confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0, d=0.4))                # in window, but ego is OFF the line
    for _ in range(3 * node.unlatch_clear_msgs):
        node.obstacles_cb(arr())                    # no detection at all
    assert node._tracks, "off-line ego must SUSPEND the unlatch streak"
    node.frenet_cb(odom(7.0, d=0.0))                # back on the raceline
    for _ in range(node.unlatch_clear_msgs):
        node.obstacles_cb(arr())
    assert not node._tracks, "back on the line the streak must run to unlatch"
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
    assert not node._tracks, "is_visible=False detections must not defeat the unlatch streak"
    print("PASS remembered (is_visible=False) detection does not reset streak")


def test_occlusion_suspends():
    node = make_node()
    t = confirm_obstacle(node, s=10.0)
    node.frenet_cb(odom(7.0))
    opponent = det(1.0, 0.0, 8.5, vs=2.0)           # dynamic, gap 1.5 < track gap 3.0
    for _ in range(3 * node.unlatch_clear_msgs):
        node.obstacles_cb(arr(opponent))
    assert node._tracks, "streak must be suspended while the opponent occludes the spot"
    assert node._tracks[0].clear_streak == 0
    print("PASS occlusion suspends streak")


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
    assert not node._tracks, "a clear-view streak must be completable at racing speed"
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
        for fn in (test_confirm_and_unlatch, test_ego_offline_suspends_streak,
                   test_sighting_resets_streak, test_memory_detection_does_not_reset,
                   test_occlusion_suspends, test_window_exit_resets, test_lap_guard,
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
