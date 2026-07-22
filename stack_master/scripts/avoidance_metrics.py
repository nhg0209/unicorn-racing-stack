#!/usr/bin/env python3
"""
Record the static-avoidance validation metrics for one sim run and print a summary row.

Subscribes to the running-sim topics (no rebuild needed; run with python3 after sourcing):
  /lap_data                     f110_msgs/LapData     -> lap time, lateral error
  /planner/avoidance/latency    std_msgs/Float32      -> planner loop time (s) -> ms stats
  /state_machine                std_msgs/String       -> state transition count (chatter)
  /tracking/obstacles           f110_msgs/ObstacleArray
  /car_state/odom               nav_msgs/Odometry     -> ego pose for clearance

Metrics printed (matches the Phase-3 table): collision (y/n), lap time, planner latency
(mean/max ms), state-transition count, min obstacle clearance [m].

  # launch lap_analyser first (race.launch.xml does NOT):
  ros2 launch lap_analyser lap_analyser.launch.py
  python3 stack_master/scripts/avoidance_metrics.py --label S1 --collision-thresh 0.05
Ctrl-C to stop and print the final row.
"""
import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from nav_msgs.msg import Odometry
from f110_msgs.msg import LapData, ObstacleArray


class Metrics(Node):
    def __init__(self, label, car_half_width, collision_thresh):
        super().__init__("avoidance_metrics")
        self.label = label
        self.car_half_width = car_half_width
        self.collision_thresh = collision_thresh

        self.ego = None
        self.latencies_ms = []
        self.state = None
        self.transitions = 0
        self.state_hist = {}
        self.min_clearance = math.inf
        self.laps = []            # (lap_count, lap_time, avg_err, max_err)

        self.create_subscription(Odometry, "/car_state/odom", self._odom_cb, 10)
        self.create_subscription(Float32, "/planner/avoidance/latency", self._lat_cb, 10)
        self.create_subscription(String, "/state_machine", self._state_cb, 10)
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self._obs_cb, 10)
        self.create_subscription(LapData, "/lap_data", self._lap_cb, 10)
        self.create_timer(2.0, self._print_live)

    def _odom_cb(self, m):
        self.ego = np.array([m.pose.pose.position.x, m.pose.pose.position.y])

    def _lat_cb(self, m):
        self.latencies_ms.append(float(m.data) * 1e3)

    def _state_cb(self, m):
        if self.state is not None and m.data != self.state:
            self.transitions += 1
        self.state = m.data
        self.state_hist[m.data] = self.state_hist.get(m.data, 0) + 1

    def _obs_cb(self, m):
        if self.ego is None:
            return
        for o in m.obstacles:
            d = math.hypot(o.x_m - self.ego[0], o.y_m - self.ego[1])
            clearance = d - o.size / 2.0 - self.car_half_width
            self.min_clearance = min(self.min_clearance, clearance)

    def _lap_cb(self, m):
        row = (int(m.lap_count), float(m.lap_time),
               float(m.average_lateral_error_to_global_waypoints),
               float(m.max_lateral_error_to_global_waypoints))
        if not self.laps or self.laps[-1][0] != row[0]:
            self.laps.append(row)
            self.get_logger().info(f"lap {row[0]}: t={row[1]:.3f}s avg_err={row[2]:.3f} max_err={row[3]:.3f}")

    def _lat_stats(self):
        if not self.latencies_ms:
            return (float("nan"), float("nan"), float("nan"))
        a = np.array(self.latencies_ms)
        return (float(a.mean()), float(a.max()), float(np.percentile(a, 95)))

    def _print_live(self):
        mean, mx, p95 = self._lat_stats()
        clr = self.min_clearance if math.isfinite(self.min_clearance) else float("nan")
        self.get_logger().info(
            f"[{self.label}] state={self.state} transitions={self.transitions} "
            f"latency ms(mean/max/p95)={mean:.2f}/{mx:.2f}/{p95:.2f} "
            f"min_clearance={clr:.3f} laps={len(self.laps)}",
            throttle_duration_sec=0.0)

    def summary(self):
        mean, mx, p95 = self._lat_stats()
        clr = self.min_clearance if math.isfinite(self.min_clearance) else float("nan")
        collision = math.isfinite(self.min_clearance) and self.min_clearance < self.collision_thresh
        best_lap = min((l[1] for l in self.laps), default=float("nan"))
        print("\n================ AVOIDANCE METRICS SUMMARY ================")
        print(f" scenario                : {self.label}")
        print(f" collision (<{self.collision_thresh:.2f} m)   : {'YES' if collision else 'no'}")
        print(f" best lap time [s]       : {best_lap:.3f}   (laps recorded: {len(self.laps)})")
        print(f" planner latency [ms]    : mean {mean:.2f} / max {mx:.2f} / p95 {p95:.2f}"
              f"   ({'OK <10ms' if mx < 10 else 'OVER 10ms'})")
        print(f" state transitions       : {self.transitions}   (chatter indicator)")
        print(f" min obstacle clearance  : {clr:.3f} m")
        print(f" state histogram         : {self.state_hist}")
        print("==========================================================\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="run")
    ap.add_argument("--car-half-width", type=float, default=0.15)
    ap.add_argument("--collision-thresh", type=float, default=0.05, help="min clearance [m] below which = collision")
    args = ap.parse_args()
    rclpy.init()
    node = Metrics(args.label, args.car_half_width, args.collision_thresh)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
