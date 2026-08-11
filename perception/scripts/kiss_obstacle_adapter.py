#!/usr/bin/env python3
"""Feed KISS-ICP BEV detections into the racing perception pipeline.

Replaces the 2D-scan `detect` node when the car has no 2D LiDAR (car2). The
KISS localization node already clusters the 3D Livox scan and publishes obstacle
centres as a PoseArray (map frame) on `/kiss_loc/obstacle_poses`. This adapter
converts each centre to Frenet (s, d) against the global raceline and republishes
them as f110_msgs/ObstacleArray on `/detect/raw_obstacles` -- exactly what the
downstream `tracking` (multi_tracking) node consumes.

tracking only reads s_center / d_center / size from each measurement (it runs its
own KF for velocity + static/dynamic classification), so those are all we fill;
id/bounds are set for completeness. No 2D `/scan` is used anywhere in this path.

Parameters:
  in_topic        = /kiss_loc/obstacle_poses  KISS cluster centres (PoseArray, map)
  out_topic       = /detect/raw_obstacles     measurements for tracking (ObstacleArray)
  waypoints_topic = /global_waypoints         raceline for the Frenet converter (WpntArray)
  obstacle_size   = 0.5                        assumed obstacle footprint [m] (PoseArray has none)
"""
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from f110_msgs.msg import WpntArray, ObstacleArray, Obstacle
from frenet_conversion.frenet_converter import FrenetConverter


class KissObstacleAdapter(Node):
    def __init__(self):
        super().__init__("kiss_obstacle_adapter")
        self.in_topic = self.declare_parameter("in_topic", "/kiss_loc/obstacle_poses").value
        self.out_topic = self.declare_parameter("out_topic", "/detect/raw_obstacles").value
        self.wpnt_topic = self.declare_parameter("waypoints_topic", "/global_waypoints").value
        self.size = float(self.declare_parameter("obstacle_size", 0.5).value)

        self.converter = None
        self.track_length = None

        self.pub = self.create_publisher(ObstacleArray, self.out_topic, 5)
        self.create_subscription(WpntArray, self.wpnt_topic, self._on_wpnts, 10)
        self.create_subscription(PoseArray, self.in_topic, self._on_poses, 10)
        self.get_logger().info(
            f"kiss->obstacles: '{self.in_topic}' -> '{self.out_topic}' "
            f"(Frenet from '{self.wpnt_topic}')")

    def _on_wpnts(self, data):
        if self.converter is not None or not data.wpnts:
            return
        xs = np.array([w.x_m for w in data.wpnts])
        ys = np.array([w.y_m for w in data.wpnts])
        self.track_length = data.wpnts[-1].s_m
        self.converter = FrenetConverter(xs, ys)
        self.get_logger().info("kiss->obstacles: FrenetConverter ready")

    def _on_poses(self, data):
        if self.converter is None:
            return  # raceline not up yet -> can't do Frenet
        msg = ObstacleArray()
        msg.header.stamp = data.header.stamp
        msg.header.frame_id = "map"
        if data.poses:
            xs = np.array([p.position.x for p in data.poses])
            ys = np.array([p.position.y for p in data.poses])
            sd = self.converter.get_frenet(xs, ys)  # -> [s_array, d_array]
            s_arr, d_arr = np.atleast_1d(sd[0]), np.atleast_1d(sd[1])
            half = self.size / 2.0
            L = self.track_length
            for i in range(len(s_arr)):
                s, d = float(s_arr[i]), float(d_arr[i])
                o = Obstacle()
                o.id = i
                o.s_center = s % L
                o.d_center = d
                o.s_start = (s - half) % L
                o.s_end = (s + half) % L
                o.d_left = d + half
                o.d_right = d - half
                o.size = self.size
                o.is_static = False
                o.is_visible = True
                msg.obstacles.append(o)
        self.pub.publish(msg)  # publish even when empty (tracking clears)


def main():
    rclpy.init()
    node = KissObstacleAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
