#!/usr/bin/env python3
"""
kiss_obstacle_bridge
====================
Bridges the kiss_icp_localization BEV detector output into the racing
perception pipeline so the Livox MID-360 (3D) can drive obstacle detection
WITHOUT flattening to a 2D /scan.

kiss (detect_en:true) publishes its detections as a viz MarkerArray on
/kiss_loc/detections: one CUBE marker per DBSCAN cluster (ns="objects", map
frame, identity orientation) whose pose.position is the bbox center and
scale.x/scale.y is the real bbox extent. The rest of the stack
(tracking -> state_machine) expects f110_msgs/ObstacleArray on
/detect/raw_obstacles with the Frenet fields filled.

This bridge reads each CUBE marker, converts its center (x, y map) to Frenet
(s, d), and — crucially — projects the axis-aligned bbox onto the track
tangent/normal (using psi_rad from the global path) so d_left/d_right and
s_start/s_end reflect the REAL obstacle width. The static-avoidance planner
sizes its keep-out box from d_left/d_right, so an accurate width is what lets
it find a feasible overtake corridor (a fixed oversized box blocks it).

Mirrors:
  - perception/src/detect.cpp:637-661  (ObstacleArray field population)
  - perception/scripts/multi_tracking.py:531-552 (lazy FrenetConverter init)
  - kiss localization_node.cpp runDetection() (CUBE marker = center + scale)
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import WpntArray, ObstacleArray, Obstacle
from frenet_conversion.frenet_converter import FrenetConverter


class KissObstacleBridge(Node):
    def __init__(self):
        super().__init__('kiss_obstacle_bridge')

        marker_topic = self.declare_parameter(
            'marker_topic', '/kiss_loc/detections').value
        # Minimum box half-extent [m] (kiss already floors scale at 0.1; guard anyway).
        self.min_half = self.declare_parameter('min_half_extent', 0.05).value

        self.converter = None
        self.track_length = None
        self.wpnt_s = None       # cumulative s of the global path [N]
        self.wpnt_psi = None     # track heading psi_rad at each waypoint [N]
        self._wx = None          # cached geometry for change detection (reopt swaps)
        self._wy = None

        self.create_subscription(
            WpntArray, '/global_waypoints_scaled', self.path_cb, 10)
        self.create_subscription(MarkerArray, marker_topic, self.markers_cb, 10)

        self.obstacle_pub = self.create_publisher(
            ObstacleArray, '/detect/raw_obstacles', 5)

        self.get_logger().info(
            f"[kiss_obstacle_bridge] {marker_topic} (CUBE) -> /detect/raw_obstacles; "
            "waiting for global path...")

    def path_cb(self, data: WpntArray):
        """Build the FrenetConverter + heading lookup from the scaled path. The global
        line can CHANGE at runtime (static re-optimization swaps in an obstacle-aware
        line), so rebuild whenever the geometry actually changes — a stale converter
        puts every detection at a wrong (s, d) relative to the line the car follows."""
        if not data.wpnts:
            return
        wx = np.array([w.x_m for w in data.wpnts])
        wy = np.array([w.y_m for w in data.wpnts])
        if self.converter is not None and self._wx is not None \
                and wx.shape == self._wx.shape \
                and np.allclose(wx, self._wx) and np.allclose(wy, self._wy):
            return                       # unchanged geometry -> keep the converter
        self._wx, self._wy = wx, wy
        self.converter = FrenetConverter(wx, wy)
        self.wpnt_s = np.array([w.s_m for w in data.wpnts])
        self.wpnt_psi = np.array([w.psi_rad for w in data.wpnts])
        self.track_length = data.wpnts[-1].s_m
        self.get_logger().info(
            "[kiss_obstacle_bridge] FrenetConverter (re)built "
            f"(track_length {self.track_length:.2f} m)")

    @staticmethod
    def _yaw(q) -> float:
        """Planar yaw from a quaternion (only z,w matter for a z-axis rotation)."""
        return 2.0 * math.atan2(q.z, q.w)

    def markers_cb(self, data: MarkerArray):
        # Publish every cycle (empty included) so tracking uses overwrite
        # semantics — an empty array clears stale detections.
        # MarkerArray has no header; stamp with ROS time (bridge == detection time).
        out = ObstacleArray()
        out.header.frame_id = 'map'
        out.header.stamp = self.get_clock().now().to_msg()

        if self.converter is None:
            self.obstacle_pub.publish(out)   # no path yet -> can't compute Frenet
            return

        cubes = [m for m in data.markers
                 if m.type == Marker.CUBE
                 and m.action != Marker.DELETEALL
                 and m.ns == 'objects']

        if cubes:
            xs = np.array([m.pose.position.x for m in cubes])
            ys = np.array([m.pose.position.y for m in cubes])
            s_arr, d_arr = self.converter.get_frenet(xs, ys)
            tl = self.track_length

            for i, m in enumerate(cubes):
                s = float(s_arr[i]) % tl
                d = float(d_arr[i])

                # Axis-aligned bbox half-extents in the marker frame (map, since
                # kiss uses identity orientation — general yaw handled anyway).
                hx = max(0.5 * m.scale.x, self.min_half)
                hy = max(0.5 * m.scale.y, self.min_half)
                # Box orientation relative to the track tangent at the obstacle.
                psi = float(self.wpnt_psi[int(np.argmin(np.abs(self.wpnt_s - s)))])
                alpha = psi - self._yaw(m.pose.orientation)
                ca, sa = abs(math.cos(alpha)), abs(math.sin(alpha))
                # Support half-widths: along tangent (s) and along normal (d).
                half_s = hx * ca + hy * sa
                half_d = hx * sa + hy * ca

                obs = Obstacle()
                obs.id = i
                obs.x_m = float(xs[i])
                obs.y_m = float(ys[i])
                obs.theta = 0.0
                obs.s_center = s
                obs.d_center = d
                obs.s_start = (s - half_s) % tl
                obs.s_end = (s + half_s) % tl
                obs.d_left = d + half_d
                obs.d_right = d - half_d
                obs.size = 2.0 * max(half_s, half_d)
                # RAW-stage placeholders, matching detect.cpp: static/dynamic classification is
                # owned by tracking's position-std voting downstream (a single frame has no
                # velocity), and is_visible=True is literally true here — every raw detection
                # is a live sighting. Tracking re-derives both on /tracking/obstacles.
                obs.is_static = False
                obs.is_visible = True
                out.obstacles.append(obs)

        self.obstacle_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = KissObstacleBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
