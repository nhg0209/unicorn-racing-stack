#!/usr/bin/env python3
"""scan_augmentor — VIL injection seam (f110_msgs/ObstacleArray based).

Republishes the base lidar scan UNCHANGED (passthrough) until virtual obstacles
(opponent + static) are published as f110_msgs/ObstacleArray, then overlays each
as a small axis-aligned square of side `size` centred at (x_m, y_m) onto the
scan (min(real_range, box_range)). The opponent is intentionally a small box at
its base_link (rear axle), mirroring a real detection box rather than the whole
car.

Obstacles are carried as f110_msgs so the SAME source feeds two consumers:
  (a) this overlay (sensor-level VIL), and
  (b) a future concat with detection/tracking output (object-level VIL).

Ego pose comes from odometry (stable; no per-scan TF lookup). If no fresh
obstacles or no ego pose, it falls back to byte-for-byte passthrough.
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from f110_msgs.msg import ObstacleArray


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _overlay_box(scan, cx, cy, theta, half, px, py, dx, dy):
    """min(scan, distance to a `2*half` square centred at (cx,cy), rotated by
    theta) per beam (ray vs 4 edges)."""
    c, s = math.cos(theta), math.sin(theta)
    loc = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    V = loc @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
    E = np.roll(V, -1, 0) - V
    for a, e in zip(V, E):
        det = e[0] * dy - e[1] * dx
        safe = np.abs(det) > 1e-12
        den = np.where(safe, det, 1.0)
        wx, wy = a[0] - px, a[1] - py
        t = (e[0] * wy - e[1] * wx) / den
        u = (dx * wy - dy * wx) / den
        ok = safe & (t > 0) & (u >= 0) & (u <= 1) & (t < scan)
        scan = np.where(ok, t, scan)
    return scan


class ScanAugmentor(Node):
    def __init__(self):
        super().__init__('scan_augmentor')
        self.declare_parameter('in_topic', '/scan_raw')
        self.declare_parameter('out_topic', '/scan')
        self.declare_parameter('obstacle_topics',
                               ['/sim/dynamic_obstacles', '/sim/static_obstacles'])
        self.declare_parameter('ego_odom_topic', '/car_state/odom')
        self.declare_parameter('scan_distance_to_base_link', 0.275)
        self.declare_parameter('watchdog_sec', 0.5)

        in_topic = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value
        self.scan_dist = float(self.get_parameter('scan_distance_to_base_link').value)
        self.watchdog = float(self.get_parameter('watchdog_sec').value)

        self.enabled = True
        self.ego = None                  # (x, y, yaw) base_link in map
        self.obs = {}                    # topic -> (obstacles list, recv_time)

        for topic in self.get_parameter('obstacle_topics').value:
            self.create_subscription(
                ObstacleArray, topic,
                lambda msg, t=topic: self._obs_cb(t, msg), 10)
        self.create_subscription(
            Odometry, self.get_parameter('ego_odom_topic').value, self._ego_cb, 10)
        self.create_subscription(Bool, '/vil/enable', self._enable_cb, 10)

        self.sub = self.create_subscription(
            LaserScan, in_topic, self._scan_cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(LaserScan, out_topic, qos_profile_sensor_data)
        self.get_logger().info(
            f"[scan_augmentor] {in_topic} -> {out_topic} (f110_msgs obstacles; "
            "passthrough until obstacles arrive)")

    def _obs_cb(self, topic, msg):
        self.obs[topic] = (msg.obstacles, self.get_clock().now())

    def _ego_cb(self, msg):
        p = msg.pose.pose
        self.ego = (p.position.x, p.position.y, _yaw(p.orientation))

    def _enable_cb(self, msg):
        self.enabled = bool(msg.data)

    def _fresh_obstacles(self):
        now = self.get_clock().now()
        out = []
        for obstacles, t in self.obs.values():
            if (now - t).nanoseconds * 1e-9 < self.watchdog:
                out.extend(obstacles)
        return out

    def _scan_cb(self, msg):
        obstacles = self._fresh_obstacles() if self.enabled else []
        if not obstacles or self.ego is None:
            self.pub.publish(msg)                     # passthrough
            return

        ex, ey, eyaw = self.ego
        px = ex + self.scan_dist * math.cos(eyaw)     # laser origin in map
        py = ey + self.scan_dist * math.sin(eyaw)
        n = len(msg.ranges)
        ang = eyaw + msg.angle_min + np.arange(n) * msg.angle_increment
        dx, dy = np.cos(ang), np.sin(ang)

        orig = np.asarray(msg.ranges, dtype=np.float64)
        valid = np.isfinite(orig) & (orig > 0.0)
        eff = np.where(valid, orig, msg.range_max)    # no-return -> open for overlay
        work = eff.copy()
        for o in obstacles:
            half = max(float(o.size), 0.01) / 2.0
            work = _overlay_box(work, float(o.x_m), float(o.y_m), float(o.theta),
                                half, px, py, dx, dy)

        hit = work < eff
        out_ranges = np.where(hit, work, orig)        # preserve original elsewhere

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = out_ranges.astype(np.float32).tolist()
        out.intensities = msg.intensities
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ScanAugmentor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
