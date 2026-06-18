#!/usr/bin/env python3
"""
detect_ros.py  –  ROS 2 wrapper for LiDAR obstacle detection.

Responsibilities
----------------
* Subscribe to the 2-D LiDAR scan (sensor_msgs/LaserScan).
* Convert the scan to plain NumPy arrays and call the ROS-free detector.
* Convert the resulting DetectedObstacle list → f110_msgs/ObstacleArray.
* Publish the ObstacleArray and (optionally) visualisation markers.

All detection *logic* lives in detect.py — this file only handles ROS I/O.

Topics
------
Subscribed:
  /scan          (sensor_msgs/LaserScan)   – raw 2-D LiDAR data

Published:
  /detections    (f110_msgs/ObstacleArray) – detected obstacle list
  /detection_markers (visualization_msgs/MarkerArray) – RViz markers

Parameters
----------
  detector_type      (str,   default 'jump')  – 'jump' | 'dbscan'
  cluster_threshold  (float, default 0.3)  [m]  (jump detector)
  min_points         (int,   default 3)          (both detectors)
  max_range          (float, default 10.0) [m]
  min_range          (float, default 0.05) [m]
  dbscan_eps         (float, default 0.3)  [m]  (dbscan detector)
  publish_markers    (bool,  default true)
"""

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from f110_msgs.msg import Obstacle, ObstacleArray

# ROS-free detection logic
from perception.detect import LidarDetector, DBSCANDetector, DetectedObstacle


class DetectNode(Node):

    def __init__(self) -> None:
        super().__init__('detect_node')

        # ----------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------
        self.declare_parameter('detector_type', 'jump')
        self.declare_parameter('cluster_threshold', 0.3)
        self.declare_parameter('min_points', 3)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('min_range', 0.05)
        self.declare_parameter('dbscan_eps', 0.3)
        self.declare_parameter('publish_markers', True)

        det_type = self.get_parameter('detector_type').value
        cluster_th = self.get_parameter('cluster_threshold').value
        min_pts = self.get_parameter('min_points').value
        max_r = self.get_parameter('max_range').value
        min_r = self.get_parameter('min_range').value
        dbscan_eps = self.get_parameter('dbscan_eps').value
        self.publish_markers = self.get_parameter('publish_markers').value

        # ----------------------------------------------------------------
        # Detector selection
        # ----------------------------------------------------------------
        if det_type == 'dbscan':
            self.detector = DBSCANDetector(
                eps=dbscan_eps,
                min_samples=min_pts,
                max_range=max_r,
                min_range=min_r,
            )
        else:  # default: jump
            self.detector = LidarDetector(
                cluster_threshold=cluster_th,
                min_points=min_pts,
                max_range=max_r,
                min_range=min_r,
            )

        # ----------------------------------------------------------------
        # Subscriptions
        # ----------------------------------------------------------------
        self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)

        # ----------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------
        self.obs_pub = self.create_publisher(ObstacleArray, '/detections', 10)

        if self.publish_markers:
            self.marker_pub = self.create_publisher(
                MarkerArray, '/detection_markers', 10
            )
        else:
            self.marker_pub = None

        self.get_logger().info(
            f'DetectNode started (detector={det_type}, '
            f'cluster_th={cluster_th:.2f} m)'
        )

    # --------------------------------------------------------------------
    # Scan callback
    # --------------------------------------------------------------------

    def _scan_callback(self, msg: LaserScan) -> None:
        ranges = np.array(msg.ranges, dtype=np.float32)

        # --- Run ROS-free detector ---
        obstacles = self.detector.detect(
            ranges=ranges,
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
        )

        # --- Publish ObstacleArray ---
        obs_array = self._to_obstacle_array(msg, obstacles)
        self.obs_pub.publish(obs_array)

        # --- Publish RViz markers (optional) ---
        if self.marker_pub is not None:
            self.marker_pub.publish(
                self._to_marker_array(msg, obstacles)
            )

    # --------------------------------------------------------------------
    # Conversion helpers  (ROS types ↔ plain Python types)
    # --------------------------------------------------------------------

    def _to_obstacle_array(
        self,
        scan: LaserScan,
        obstacles: list[DetectedObstacle],
    ) -> ObstacleArray:
        """Convert list[DetectedObstacle] → f110_msgs/ObstacleArray."""
        arr = ObstacleArray()
        arr.header.stamp = scan.header.stamp
        arr.header.frame_id = scan.header.frame_id   # usually 'laser'

        for det in obstacles:
            obs = Obstacle()
            obs.id = det.id
            obs.x_m = det.cx
            obs.y_m = det.cy
            obs.size = max(det.width, det.height, 0.05)
            obs.is_visible = True
            obs.is_static = False
            arr.obstacles.append(obs)

        return arr

    def _to_marker_array(
        self,
        scan: LaserScan,
        obstacles: list[DetectedObstacle],
    ) -> MarkerArray:
        """Build RViz cube markers for each detected obstacle."""
        marker_array = MarkerArray()

        # Delete all previous markers
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for det in obstacles:
            m = Marker()
            m.header.stamp = scan.header.stamp
            m.header.frame_id = scan.header.frame_id
            m.ns = 'detections'
            m.id = det.id
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = det.cx
            m.pose.position.y = det.cy
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0

            m.scale.x = max(det.width, 0.1)
            m.scale.y = max(det.height, 0.1)
            m.scale.z = 0.3           # arbitrary height for visualization

            m.color.r = 1.0
            m.color.g = 0.3
            m.color.b = 0.0
            m.color.a = 0.7

            m.lifetime.sec = 0
            m.lifetime.nanosec = 200_000_000   # 0.2 s auto-expire

            marker_array.markers.append(m)

        return marker_array


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
