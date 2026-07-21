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


def odom(s, d=0.0):
    m = Odometry()
    m.pose.pose.position.x = s
    m.pose.pose.position.y = d
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


def main():
    rclpy.init()
    try:
        for fn in (test_confirm_and_unlatch, test_ego_offline_suspends_streak,
                   test_sighting_resets_streak, test_memory_detection_does_not_reset,
                   test_occlusion_suspends, test_window_exit_resets, test_lap_guard):
            fn()
    finally:
        rclpy.shutdown()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
