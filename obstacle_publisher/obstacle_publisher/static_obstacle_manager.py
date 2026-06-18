#!/usr/bin/env python3
"""
Static Obstacle Manager for F1TENTH Simulator (ROS2)

Manages static obstacles placed via RViz publish_point.
Provides Clear button via Interactive Marker.
Publishes obstacle positions for map_publisher or obstacle_publisher_grid to use.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers import InteractiveMarkerServer
from std_msgs.msg import ColorRGBA


class StaticObstacleManager(Node):
    """Manages static obstacles from RViz publish_point"""

    def __init__(self):
        super().__init__('static_obstacle_manager')

        # ===== Parameters =====
        self.declare_parameter('obstacle_diameter_m', 0.5)
        self.obstacle_diameter_m = self.get_parameter('obstacle_diameter_m').value

        # ===== Static obstacles list =====
        self.static_obstacles = []  # List of (x, y) tuples

        # ===== ROS Publishers =====
        self.obstacle_marker_pub = self.create_publisher(MarkerArray, '/static_obstacle_markers', 1)
        self.obstacle_positions_pub = self.create_publisher(MarkerArray, '/static_obstacles', 10)

        # ===== ROS Subscribers =====
        self.point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_cb,
            10
        )

        # ===== Interactive Marker Server for Clear button =====
        self.marker_server = InteractiveMarkerServer(self, 'obstacle_controls')
        self.create_clear_button()

        # ===== Timer to publish obstacle positions =====
        self.timer = self.create_timer(0.1, self.publish_obstacles)  # 10 Hz

        self.get_logger().info('[StaticObstacleManager] Initialized')
        self.get_logger().info('  - Subscribe: /clicked_point')
        self.get_logger().info('  - Publish: /static_obstacles, /static_obstacle_markers')
        self.get_logger().info('  - Interactive Marker: Clear Obstacles button at (0, 0)')

    def clicked_point_cb(self, msg: PointStamped):
        """Callback for RViz publish_point - add static obstacle"""
        x_m = msg.point.x
        y_m = msg.point.y

        self.static_obstacles.append((x_m, y_m))
        self.get_logger().info(f'[StaticObstacleManager] Added static obstacle at ({x_m:.2f}, {y_m:.2f})')
        self.get_logger().info(f'[StaticObstacleManager] Total: {len(self.static_obstacles)} obstacles')

        # Publish immediately
        self.publish_obstacles()
        self.publish_markers()

    def create_clear_button(self):
        """Create Interactive Marker for clearing obstacles"""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = "clear_obstacles"
        int_marker.description = "Clear Obstacles\n(Left Click)"
        int_marker.pose.position.x = 0.0
        int_marker.pose.position.y = -5.0
        int_marker.pose.position.z = 0.0
        int_marker.scale = 1.0

        # Create a button control
        button_control = InteractiveMarkerControl()
        button_control.interaction_mode = InteractiveMarkerControl.BUTTON
        button_control.always_visible = True
        button_control.name = "clear_button"

        # Visual marker for the button (green cube)
        button_marker = Marker()
        button_marker.type = Marker.CUBE
        button_marker.scale.x = 0.45
        button_marker.scale.y = 0.65
        button_marker.scale.z = 0.45
        button_marker.color.r = 0.0
        button_marker.color.g = 1.0
        button_marker.color.b = 0.0
        button_marker.color.a = 1.0

        button_control.markers.append(button_marker)
        int_marker.controls.append(button_control)

        # Add text label
        text_marker = Marker()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.text = "Clear Obstacles"
        text_marker.scale.z = 0.2
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.pose.position.z = 0.3

        text_control = InteractiveMarkerControl()
        text_control.interaction_mode = InteractiveMarkerControl.NONE
        text_control.always_visible = True
        text_control.markers.append(text_marker)
        int_marker.controls.append(text_control)

        # Register callback
        self.marker_server.insert(int_marker, feedback_callback=self.clear_button_callback)
        self.marker_server.applyChanges()

        self.get_logger().info('[StaticObstacleManager] Clear button created at origin')

    def clear_button_callback(self, feedback: InteractiveMarkerFeedback):
        """Callback when Clear button is clicked"""
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            prev_count = len(self.static_obstacles)
            self.static_obstacles.clear()
            self.get_logger().info(f'[StaticObstacleManager] Cleared {prev_count} obstacles')

            # Publish empty lists
            self.publish_obstacles()
            self.publish_markers()

    def publish_obstacles(self):
        """Publish static obstacle positions as MarkerArray (for map publishers to use)"""
        marker_array = MarkerArray()

        for i, (x_m, y_m) in enumerate(self.static_obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = x_m
            marker.pose.position.y = y_m
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            # Size from parameter
            marker.scale.x = self.obstacle_diameter_m
            marker.scale.y = self.obstacle_diameter_m
            marker.scale.z = 0.01  # Flat marker for position info

            # Invisible color (just position data)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.0

            marker_array.markers.append(marker)

        # If empty, send delete-all marker
        if len(self.static_obstacles) == 0:
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)

        self.obstacle_positions_pub.publish(marker_array)

    def publish_markers(self):
        """Publish visualization markers for static obstacles (blue cylinders)"""
        marker_array = MarkerArray()

        for i, (x_m, y_m) in enumerate(self.static_obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = x_m
            marker.pose.position.y = y_m
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0

            marker.scale.x = self.obstacle_diameter_m
            marker.scale.y = self.obstacle_diameter_m
            marker.scale.z = self.obstacle_diameter_m

            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        # If empty, send delete-all
        if len(self.static_obstacles) == 0:
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)

        self.obstacle_marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = StaticObstacleManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
