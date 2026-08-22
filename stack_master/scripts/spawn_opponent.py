#!/usr/bin/env python3
"""
Spawn and drive the virtual OPPONENT at a chosen Frenet (s, d) for the dynamic-overtaking
regression scenarios D0-D8. The dynamic analogue of spawn_static_obstacle.py.

Why this exists: the opponent is spawned by publishing a map-frame pose on /goal_pose (the
RViz "2D Goal Pose" tool). Clicking it by hand makes the initial gap different on every run,
which destroys A/B comparability -- and the dynamic-overtaking gates are all gap-dependent
(engage_gap_m 5 m, the SM's getting_closer window 10 m). This converts a gap in METRES along
the raceline into the right pose, sets the driving mode, and nudges the speed to a target.

  --gap G      spawn G metres AHEAD of the ego's current s (needs /car_state/odom_frenet)
  --s0 S       spawn at absolute s = S instead (no ego needed)
  --d D        lateral offset from the raceline [m], default 0 (on the line)
  --speed V    target speed [m/s]. NOTE the controller reads max_speed ONCE at startup and has
               no parameter callback, so `ros2 param set` does nothing; the only live control
               is the DELTA topic /sim/opp_speed_delta. This sends delta = V - assumed_current,
               where assumed_current defaults to the node's 2.0 m/s default (--current to
               override if you have already nudged it this session).
  --mode M     manual | path | ftg   (default path: pure-pursuit on the centerline at a
               constant speed -- deterministic, which is what a regression run wants)
  --inject I   overlay | merge       (which virtual_perception seam injects it)
  --remove     despawn and exit

Run (workspace built + sourced; no colcon rebuild needed):
  python3 stack_master/scripts/spawn_opponent.py --gap 20 --speed 1.5            # D1
  python3 stack_master/scripts/spawn_opponent.py --gap 3  --speed 1.5            # D2
  python3 stack_master/scripts/spawn_opponent.py --gap 12 --d 0.8 --mode ftg     # D6
  python3 stack_master/scripts/spawn_opponent.py --gap 20 --inject merge         # D7-b
  python3 stack_master/scripts/spawn_opponent.py --remove

Prereq: race.launch.xml with sim:=true (or virtual:=true) so virtual_perception is up. Since
the launch chain now forwards opp_spawn/opp_mode, `opp_mode:=path` alone is enough when you do
not care about the exact gap; use this script when you do.
"""
import argparse
import math
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32, Empty
from f110_msgs.msg import WpntArray

from frenet_conversion.frenet_converter import FrenetConverter

# opponent_controller.py declares max_speed with this default and reads it once at __init__.
OPP_DEFAULT_SPEED = 2.0


def _xy(conv, s, d):
    """(s, d) -> (x, y) floats. get_cartesian returns np.array([x, y]) of shape (2, 1)."""
    resp = np.asarray(conv.get_cartesian(np.array([float(s)]), np.array([float(d)])))
    return float(resp[0].item()), float(resp[1].item())


class Spawner(Node):
    def __init__(self, args):
        super().__init__("spawn_opponent")
        self.args = args
        self.gb = None          # (N,2) raceline xy
        self.gb_s = None        # (N,) s of each raceline point
        self.ego_s = None

        self.create_subscription(WpntArray, "/global_waypoints", self._gb_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self._frenet_cb, 10)

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.mode_pub = self.create_publisher(String, "/sim/opp_mode", 10)
        self.speed_pub = self.create_publisher(Float32, "/sim/opp_speed_delta", 10)
        self.inject_pub = self.create_publisher(String, "/vp/inject_mode", 10)
        self.remove_pub = self.create_publisher(Empty, "/sim/remove_opponent", 10)

    def _gb_cb(self, msg: WpntArray):
        self.gb = np.array([[w.x_m, w.y_m] for w in msg.wpnts])
        self.gb_s = np.array([w.s_m for w in msg.wpnts])

    def _frenet_cb(self, msg: Odometry):
        self.ego_s = float(msg.pose.pose.position.x)

    def _spin_until(self, pred, what, timeout=15.0):
        t0 = self.get_clock().now().nanoseconds * 1e-9
        while rclpy.ok() and not pred():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.get_clock().now().nanoseconds * 1e-9 - t0 > timeout:
                self.get_logger().error(f"timed out waiting for {what}")
                return False
        return True

    def run(self) -> int:
        a = self.args

        if a.remove:
            # publishers need a moment to match before the message would be dropped
            self._spin_until(lambda: self.remove_pub.get_subscription_count() > 0,
                             "/sim/remove_opponent subscriber", timeout=5.0)
            self.remove_pub.publish(Empty())
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.05)
            self.get_logger().info("opponent removed")
            return 0

        if not self._spin_until(lambda: self.gb is not None, "/global_waypoints"):
            return 1
        if a.s0 is None and not self._spin_until(lambda: self.ego_s is not None,
                                                 "/car_state/odom_frenet"):
            return 1

        track_len = float(self.gb_s[-1] + (self.gb_s[1] - self.gb_s[0]))
        s_target = (a.s0 if a.s0 is not None else self.ego_s + a.gap) % track_len

        conv = FrenetConverter(self.gb[:, 0], self.gb[:, 1])
        # get_cartesian returns np.array([x, y]) -> shape (2, 1) for a 1-element query
        x, y = _xy(conv, s_target, a.d)
        # heading = raceline tangent at s_target, so `path` mode starts pointing down the track
        x2, y2 = _xy(conv, (s_target + 0.5) % track_len, a.d)
        yaw = math.atan2(y2 - y, x2 - x)

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._spin_until(lambda: self.goal_pub.get_subscription_count() > 0,
                         "/goal_pose subscriber (is virtual_perception up?)", timeout=10.0)
        self.goal_pub.publish(pose)
        self.inject_pub.publish(String(data=a.inject))
        self.mode_pub.publish(String(data=a.mode))

        delta = a.speed - a.current
        if abs(delta) > 1e-6:
            self.speed_pub.publish(Float32(data=float(delta)))

        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)

        gap_txt = f"{a.gap:+.1f} m ahead of ego" if a.s0 is None else "absolute"
        self.get_logger().info(
            f"opponent spawned: s={s_target:.2f} ({gap_txt}) d={a.d:+.2f} "
            f"xy=({x:.2f}, {y:.2f}) yaw={math.degrees(yaw):.0f}deg | "
            f"mode={a.mode} inject={a.inject} | speed {a.current:.1f} -> {a.speed:.1f} "
            f"(delta {delta:+.1f})")
        if a.mode == "path":
            self.get_logger().info(
                "NOTE: `path` mode follows centerline.csv, which carries no speeds, so the "
                "opponent cruises at a CONSTANT speed and does not slow for corners.")
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gap", type=float, default=20.0, help="metres ahead of the ego [m]")
    g.add_argument("--s0", type=float, default=None, help="absolute Frenet s [m]")
    ap.add_argument("--d", type=float, default=0.0, help="lateral offset from the raceline [m]")
    ap.add_argument("--speed", type=float, default=1.5, help="target opponent speed [m/s]")
    ap.add_argument("--current", type=float, default=OPP_DEFAULT_SPEED,
                    help="opponent's speed right now [m/s]; only needed if you already nudged "
                         "it this session (max_speed is not live-readable)")
    ap.add_argument("--mode", choices=("manual", "path", "ftg"), default="path")
    ap.add_argument("--inject", choices=("overlay", "merge"), default="overlay")
    ap.add_argument("--remove", action="store_true", help="despawn the opponent and exit")
    args = ap.parse_args()

    rclpy.init()
    node = Spawner(args)
    try:
        rc = node.run()
    except KeyboardInterrupt:
        rc = 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
