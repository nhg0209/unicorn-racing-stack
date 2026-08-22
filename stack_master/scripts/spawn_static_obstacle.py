#!/usr/bin/env python3
"""
Spawn static obstacle(s) at chosen Frenet (s, d) positions in the in-repo simulator, for the
static-avoidance regression scenarios S1-S5.

There is no (s, d) spawner in the stack (static_obstacle_manager is click-only, Cartesian). This
converts each (s, d) to a map (x, y) using a FrenetConverter built from /global_waypoints and
publishes it on /clicked_point -- exactly as if you had used RViz "Publish Point" -- so
static_obstacle_manager stores it and the normal virtual_perception seam renders it. It also sets
/vp/inject_mode so you can choose the detection path:

  --inject overlay  (default) box is overlaid on /scan -> REAL detect -> multi_tracking pipeline
                    (exercises the position-persistence classifier / is_static / s_var,d_var).
  --inject merge    tracking_merger injects ground-truth straight onto /tracking/obstacles
                    (deterministic, bypasses detection -- use to test the planner/SM in isolation).

NOTE: obstacle SIZE is owned by static_obstacle_manager (a square `size` param, default 0.30).
For the regulation 0.35x0.32 m box run:  ros2 param set /static_obstacle_manager size 0.32

Run (workspace already built + sourced; no colcon rebuild needed):
  python3 stack_master/scripts/spawn_static_obstacle.py --obs "8,0.0"
  python3 stack_master/scripts/spawn_static_obstacle.py --scenario S2 --s0 8.0
  python3 stack_master/scripts/spawn_static_obstacle.py --clear      # remove all
"""
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Empty
from f110_msgs.msg import WpntArray
from frenet_conversion.frenet_converter import FrenetConverter

# (s_offset_from_s0 [m], d [m]).  d: +left / -right.  s offsets are map-relative; tune --s0 (and,
# for S3, pick an --s0 just after a corner) to the actual ifac track features.
SCENARIOS = {
    "S1": [(0.0, 0.0)],                                   # straight, single centered box
    "S2": [(0.0, 0.40), (3.0, -0.40), (6.0, 0.40)],      # slalom (3 m alternating) -- old failure case
    "S3": [(0.0, 0.0)],                                   # corner-exit: set --s0 just after a corner
    "S4": [(0.0, 0.35)],                                  # ~40 cm gap: offset box near one wall
    "S5": [],                                             # no obstacle (raceline regression)
}


class Spawner(Node):
    def __init__(self, obs_sd, inject_mode, clear):
        super().__init__("spawn_static_obstacle")
        self.obs_sd = obs_sd
        self.inject_mode = inject_mode
        self.clear = clear
        self.converter = None
        self.create_subscription(WpntArray, "/global_waypoints", self._gb_cb, 10)
        self.click_pub = self.create_publisher(PointStamped, "/clicked_point", 10)
        self.mode_pub = self.create_publisher(String, "/vp/inject_mode", 10)
        self.clear_pub = self.create_publisher(Empty, "/sim/clear_obstacles", 10)
        self._done = False

    def _gb_cb(self, data: WpntArray):
        if self.converter is not None:
            return
        x = np.array([w.x_m for w in data.wpnts])
        y = np.array([w.y_m for w in data.wpnts])
        psi = np.array([w.psi_rad for w in data.wpnts])
        s = np.array([w.s_m for w in data.wpnts])
        self.converter = FrenetConverter(x, y, psi)
        self._track_len = float(s[-1])
        self.get_logger().info(f"FrenetConverter ready (track length {self._track_len:.2f} m)")

    def tick(self):
        if self._done or self.converter is None:
            return
        if self.clear:
            self.clear_pub.publish(Empty())
            self.get_logger().info("published /sim/clear_obstacles")
            self._done = True
            return
        self.mode_pub.publish(String(data=self.inject_mode))
        for (s, d) in self.obs_sd:
            s_wrapped = s % self._track_len
            xy = self.converter.get_cartesian(s_wrapped, d)
            xy = np.asarray(xy).reshape(-1)
            msg = PointStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x, msg.point.y, msg.point.z = float(xy[0]), float(xy[1]), 0.0
            self.click_pub.publish(msg)
            self.get_logger().info(f"spawned obstacle at s={s_wrapped:.2f} d={d:+.2f} -> "
                                   f"x={xy[0]:.2f} y={xy[1]:.2f}  (inject={self.inject_mode})")
        self._done = True


def parse_obs(obs_str, scenario, s0):
    if obs_str:
        out = []
        for pair in obs_str.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            s, d = pair.split(",")
            out.append((float(s), float(d)))
        return out
    if scenario:
        return [(s0 + ds, d) for (ds, d) in SCENARIOS[scenario.upper()]]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default="", help='explicit "s,d; s,d" (absolute s, metres)')
    ap.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="S1..S5 preset")
    ap.add_argument("--s0", type=float, default=8.0, help="base s [m] for --scenario offsets")
    ap.add_argument("--inject", choices=["overlay", "merge"], default="overlay")
    ap.add_argument("--clear", action="store_true", help="clear all obstacles and exit")
    args = ap.parse_args()

    obs_sd = parse_obs(args.obs, args.scenario, args.s0)
    if not args.clear and not obs_sd:
        print("Nothing to spawn (S5 / empty). Use --obs or --scenario, or --clear.")
        # still publish inject mode below via a short spin so S5 sets a clean state
    rclpy.init()
    node = Spawner(obs_sd, args.inject, args.clear)
    # spin until the converter is up and one publish tick has run, then a few extra ticks so the
    # latched-ish click/mode messages are actually delivered.
    ticks = 0
    while rclpy.ok() and ticks < 200:
        rclpy.spin_once(node, timeout_sec=0.05)
        node.tick()
        if node._done:
            ticks += 1
        if ticks > 20:
            break
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
