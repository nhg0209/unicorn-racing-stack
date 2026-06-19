#!/usr/bin/env python3
"""
tracking_ros.py  –  ROS 2 wrapper for the obstacle tracker.

Responsibilities
----------------
* Subscribe to the raw detections (f110_msgs/ObstacleArray) from detect_ros.
* Convert the ObstacleArray → list of DetectedObstacle (plain Python).
* Call the ROS-free ObstacleTracker to produce stable tracks.
* Convert TrackedObstacle list → f110_msgs/ObstacleArray (with stable IDs
  and velocity estimates) and publish.
* Optionally publish RViz markers for visualisation.

All tracking *logic* lives in tracking.py — this file only handles ROS I/O.

Topics
------
Subscribed:
  /detections     (f110_msgs/ObstacleArray) – raw detector output

Published:
  /tracked_obstacles  (f110_msgs/ObstacleArray)  – filtered & tracked
  /tracking_markers   (visualization_msgs/MarkerArray) – RViz arrows

Parameters
----------
  max_coasting          (int,   default 5)    – frames before track deletion
  association_threshold (float, default 1.0)  [m]
  min_hits              (int,   default 1)    – min matches before reporting
  publish_markers       (bool,  default true)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from f110_msgs.msg import Obstacle, ObstacleArray

# ROS-free tracking logic
from perception.tracking import ObstacleTracker, TrackedObstacle
from perception.detect import DetectedObstacle


class TrackingNode(Node):

    def __init__(self) -> None:
        super().__init__('tracking_node')

        # ----------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------
        self.declare_parameter('max_coasting', 5)
        self.declare_parameter('association_threshold', 1.0)
        self.declare_parameter('min_hits', 1)
        self.declare_parameter('publish_markers', True)
        # Tracks are output in the MAP frame (transformed from the sensor frame
        # via TF) so the obstacle_merger can fill the Frenet (s/d) fields.
        # Default topic is /tracked_obstacles (object-seam: merger injects the
        # virtual obstacles directly). Set output_topic:=/tracking/obstacles_raw
        # for the overlay seam, where the merger concatenates REAL perception
        # (this output) with the virtual obstacles overlaid on /scan.
        self.declare_parameter('output_topic', '/tracked_obstacles')
        self.declare_parameter('target_frame', 'map')

        max_coast = self.get_parameter('max_coasting').value
        assoc_th = self.get_parameter('association_threshold').value
        min_hits = self.get_parameter('min_hits').value
        self.publish_markers = self.get_parameter('publish_markers').value
        self.target_frame = self.get_parameter('target_frame').value

        # TF for sensor-frame -> map-frame obstacle transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ----------------------------------------------------------------
        # Tracker (ROS-free)
        # ----------------------------------------------------------------
        self.tracker = ObstacleTracker(
            max_coasting=max_coast,
            association_threshold=assoc_th,
            min_hits=min_hits,
        )

        self._last_stamp: float | None = None   # nanoseconds

        # ----------------------------------------------------------------
        # Subscriptions
        # ----------------------------------------------------------------
        self.create_subscription(
            ObstacleArray,
            '/detections',
            self._detection_callback,
            10,
        )

        # ----------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------
        self.tracked_pub = self.create_publisher(
            ObstacleArray,
            self.get_parameter('output_topic').value,
            10,
        )

        if self.publish_markers:
            self.marker_pub = self.create_publisher(
                MarkerArray,
                '/tracking_markers',
                10,
            )
        else:
            self.marker_pub = None

        self.get_logger().info(
            f'TrackingNode started (max_coast={max_coast}, '
            f'assoc_th={assoc_th:.2f} m, min_hits={min_hits})'
        )

    # --------------------------------------------------------------------
    # Detection callback
    # --------------------------------------------------------------------

    def _detection_callback(self, msg: ObstacleArray) -> None:
        # Compute dt
        now_ns = self.get_clock().now().nanoseconds
        if self._last_stamp is None:
            dt = 0.05                          # default 50 ms on first call
        else:
            dt = (now_ns - self._last_stamp) * 1e-9
        self._last_stamp = now_ns

        # --- Convert ROS msg → plain Python ---
        detections = self._from_obstacle_array(msg)

        # --- Run ROS-free tracker ---
        tracks = self.tracker.update(detections, dt=max(dt, 1e-3))

        # --- Publish tracked obstacles ---
        tracked_array = self._to_obstacle_array(msg.header, tracks)
        self.tracked_pub.publish(tracked_array)

        # --- Publish RViz markers ---
        if self.marker_pub is not None:
            self.marker_pub.publish(
                self._to_marker_array(msg.header, tracks)
            )

    # --------------------------------------------------------------------
    # Conversion helpers
    # --------------------------------------------------------------------

    def _from_obstacle_array(
        self,
        msg: ObstacleArray,
    ) -> list[DetectedObstacle]:
        """Convert f110_msgs/ObstacleArray → list[DetectedObstacle]."""
        dets = []
        for i, obs in enumerate(msg.obstacles):
            size = float(obs.size) if obs.size > 0.0 else 0.2
            dets.append(DetectedObstacle(
                cx=obs.x_m,
                cy=obs.y_m,
                width=size,
                height=size,
                num_points=0,
                id=i,
            ))
        return dets

    def _map_transform(self, frame_id: str, stamp):
        """Return (tx, ty, yaw) of target_frame <- frame_id, or None."""
        if not frame_id or frame_id == self.target_frame:
            return (0.0, 0.0, 0.0)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, stamp, timeout=Duration(seconds=0.05))
        except Exception:
            try:  # fall back to the latest available transform
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame, frame_id, rclpy.time.Time())
            except Exception:
                return None
        tr = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (tr.x, tr.y, yaw)

    def _to_obstacle_array(
        self,
        header: Header,
        tracks: list[TrackedObstacle],
    ) -> ObstacleArray:
        """Convert list[TrackedObstacle] → f110_msgs/ObstacleArray, transformed
        from the sensor frame into the map frame (so the merger can fill Frenet)."""
        arr = ObstacleArray()
        arr.header.stamp = header.stamp
        tf = self._map_transform(header.frame_id, header.stamp)
        if tf is not None:
            arr.header.frame_id = self.target_frame
            tx, ty, yaw = tf
            c, s = math.cos(yaw), math.sin(yaw)
        else:                                   # no TF yet -> keep sensor frame
            arr.header.frame_id = header.frame_id
            tx = ty = yaw = 0.0
            c, s = 1.0, 0.0

        for t in tracks:
            obs = Obstacle()
            obs.id = t.track_id
            obs.x_m = tx + c * t.cx - s * t.cy
            obs.y_m = ty + s * t.cx + c * t.cy
            obs.theta = yaw
            obs.size = max(t.width, t.height, 0.05)
            # Cartesian (map-frame) velocity; the merger/Frenet stage refines s/d.
            obs.vs = c * t.vx - s * t.vy
            obs.vd = s * t.vx + c * t.vy
            obs.is_static = math.hypot(t.vx, t.vy) < 0.1
            obs.is_visible = True

            arr.obstacles.append(obs)

        return arr

    def _to_marker_array(
        self,
        header: Header,
        tracks: list[TrackedObstacle],
    ) -> MarkerArray:
        """
        Build RViz markers for each tracked obstacle.
        * Cube  – bounding box
        * Arrow – velocity vector
        """
        marker_array = MarkerArray()

        # Delete all previous markers
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for t in tracks:
            # ---- Cube (position) ----
            cube = Marker()
            cube.header = header
            cube.ns = 'tracks_box'
            cube.id = t.track_id
            cube.type = Marker.CUBE
            cube.action = Marker.ADD

            cube.pose.position.x = t.cx
            cube.pose.position.y = t.cy
            cube.pose.position.z = 0.0
            cube.pose.orientation.w = 1.0

            cube.scale.x = max(t.width, 0.1)
            cube.scale.y = max(t.height, 0.1)
            cube.scale.z = 0.3

            cube.color.r = 0.0
            cube.color.g = 0.8
            cube.color.b = 1.0
            cube.color.a = 0.7

            cube.lifetime.nanosec = 300_000_000

            # ---- Arrow (velocity) ----
            speed = math.sqrt(t.vx ** 2 + t.vy ** 2)
            arrow = Marker()
            arrow.header = header
            arrow.ns = 'tracks_vel'
            arrow.id = t.track_id
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            # Arrow from centroid to centroid + velocity vector
            from geometry_msgs.msg import Point as RosPoint
            p_start = RosPoint()
            p_start.x, p_start.y, p_start.z = t.cx, t.cy, 0.15

            p_end = RosPoint()
            p_end.x = t.cx + t.vx * 0.3   # 0.3 s look-ahead scale
            p_end.y = t.cy + t.vy * 0.3
            p_end.z = 0.15

            arrow.points = [p_start, p_end]
            arrow.scale.x = 0.05   # shaft diameter
            arrow.scale.y = 0.10   # head diameter
            arrow.scale.z = 0.10   # head length

            arrow.color.r = 1.0
            arrow.color.g = 1.0
            arrow.color.b = 0.0
            arrow.color.a = 0.9 if speed > 0.1 else 0.0   # hide if static

            arrow.lifetime.nanosec = 300_000_000

            marker_array.markers.extend([cube, arrow])

        return marker_array


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
