#!/usr/bin/env python3
"""
Dynamic Obstacle State Publisher for F1TENTH Simulator (ROS2)

Publishes dynamic obstacle state (position, heading, size) for the F1TENTH simulator.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
import numpy as np
import csv
import os
from visualization_msgs.msg import Marker, MarkerArray

_DYNAMIC = ParameterDescriptor(dynamic_typing=True)


class DynamicObstaclePublisher(Node):
    """Publish dynamic obstacle state for the F1TENTH simulator"""

    def __init__(self):
        super().__init__('dynamic_obstacle_publisher')

        # ===== Parameters (dynamic_typing for float params to accept int from CLI) =====
        self.declare_parameter('update_rate', 20)
        self.declare_parameter('speed_scaler', 1.0, _DYNAMIC)
        self.declare_parameter('constant_speed', False)
        self.declare_parameter('starting_s', 0.0, _DYNAMIC)
        self.declare_parameter('obstacle_length_m', 0.65, _DYNAMIC)
        self.declare_parameter('obstacle_width_m', 0.35, _DYNAMIC)
        self.declare_parameter('map_name', 'f')
        self.declare_parameter('trajectory_csv', 'global_waypoints.csv')
        self.declare_parameter('reactive', False)
        self.declare_parameter('reactive_freq', 0.3, _DYNAMIC)

        self.update_rate = self.get_parameter('update_rate').value
        self.speed_scaler = float(self.get_parameter('speed_scaler').value)
        self.constant_speed = self.get_parameter('constant_speed').value
        self.starting_s = float(self.get_parameter('starting_s').value)
        self.obstacle_length_m = float(self.get_parameter('obstacle_length_m').value)
        self.obstacle_width_m = float(self.get_parameter('obstacle_width_m').value)
        self.map_name = self.get_parameter('map_name').value
        self.trajectory_csv = self.get_parameter('trajectory_csv').value
        self.reactive = self.get_parameter('reactive').value
        self.reactive_frequency = float(self.get_parameter('reactive_freq').value)

        self.looptime = 1.0 / self.update_rate

        # ===== Reactive mode state (sine-wave lateral oscillation) =====
        self.reactive_phase = 0.0
        self.reactive_amplitude = 0.0
        self.max_amplitude_limit = 0.5  # Max lateral offset in meters
        self.prev_d_perturbation = 0.0

        # ===== Map directory (source path via realpath) =====
        # __file__: .../creating_autonomous_car/simulator/obstacle_publisher/obstacle_publisher/dynamic_obstacle_publisher.py
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        self.map_dir = os.path.join(pkg_root, 'stack_master', 'maps', self.map_name)

        # ===== Obstacle state =====
        self.current_s = self.starting_s
        self.current_speed = 0.0

        # ===== Trajectory waypoints =====
        self.waypoints = []
        self.waypoints_s_array = None
        self.max_s = None

        # ===== ROS Publishers =====
        self.state_pub = self.create_publisher(Marker, '/dynamic_obstacle_state', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/dynamic_obstacle_marker', 1)

        # ===== Load trajectory =====
        if not self.load_trajectory_from_csv():
            self.get_logger().warn('[DynamicObstaclePublisher] Failed to load trajectory, using dummy')

        # ===== Timer =====
        self.timer = self.create_timer(self.looptime, self.timer_callback)

        self.get_logger().info('[DynamicObstaclePublisher] Initialized')
        self.get_logger().info(f'  - Map: {self.map_name}')
        self.get_logger().info(f'  - Speed scaler: {self.speed_scaler}')
        self.get_logger().info(f'  - Obstacle size: {self.obstacle_length_m}m x {self.obstacle_width_m}m')
        self.get_logger().info(f'  - Update rate: {self.update_rate}Hz')
        self.get_logger().info(f'  - Reactive mode: {self.reactive}')
        self.get_logger().info(f'  - Publishing state to /dynamic_obstacle_state')

    def load_trajectory_from_csv(self):
        """Load centerline from CSV"""
        csv_path = os.path.join(self.map_dir, self.trajectory_csv)

        if not os.path.exists(csv_path):
            self.get_logger().warn(f'Centerline not found: {csv_path}')
            self.create_dummy_centerline()
            return False

        self.get_logger().info(f'Loading centerline: {csv_path}')

        waypoints_data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                waypoints_data.append(row)

        if len(waypoints_data) == 0:
            self.get_logger().error('Empty centerline CSV')
            return False

        x_array = np.array([float(row['x_m']) for row in waypoints_data])
        y_array = np.array([float(row['y_m']) for row in waypoints_data])

        dx = np.diff(x_array, prepend=x_array[0])
        dy = np.diff(y_array, prepend=y_array[0])
        ds = np.sqrt(dx**2 + dy**2)
        s_array = np.cumsum(ds)

        if 'vx_mps' in waypoints_data[0]:
            speed_array = np.array([float(row['vx_mps']) for row in waypoints_data])
        else:
            speed_array = np.ones(len(waypoints_data)) * 1.0

        if self.constant_speed:
            speed_array = np.ones(len(waypoints_data)) * self.speed_scaler
        else:
            speed_array = speed_array * self.speed_scaler

        if 'heading' in waypoints_data[0]:
            heading_array = np.array([float(row['heading']) for row in waypoints_data])
        else:
            heading_array = np.arctan2(np.diff(y_array, append=y_array[0]),
                                       np.diff(x_array, append=x_array[0]))

        # Track widths for reactive mode
        if 'w_tr_right_m' in waypoints_data[0]:
            w_right_array = np.array([float(row['w_tr_right_m']) for row in waypoints_data])
        else:
            w_right_array = np.ones(len(waypoints_data)) * 0.5

        if 'w_tr_left_m' in waypoints_data[0]:
            w_left_array = np.array([float(row['w_tr_left_m']) for row in waypoints_data])
        else:
            w_left_array = np.ones(len(waypoints_data)) * 0.5

        self.waypoints = []
        for i in range(len(x_array)):
            wpnt = {
                'x_m': x_array[i],
                'y_m': y_array[i],
                's_m': s_array[i],
                'vx_mps': speed_array[i],
                'heading': heading_array[i],
                'w_right': w_right_array[i],
                'w_left': w_left_array[i]
            }
            self.waypoints.append(wpnt)

        self.waypoints_s_array = s_array
        self.max_s = s_array[-1]

        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints, max_s={self.max_s:.2f}m')
        return True

    def create_dummy_centerline(self):
        """Create circular centerline for testing"""
        num_points = 100
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        radius = 5.0

        x_array = radius * np.cos(theta)
        y_array = radius * np.sin(theta)

        dx = np.diff(x_array, prepend=x_array[-1])
        dy = np.diff(y_array, prepend=y_array[-1])
        ds = np.sqrt(dx**2 + dy**2)
        s_array = np.cumsum(ds)

        speed_array = np.ones(num_points) * 2.0 * self.speed_scaler
        heading_array = theta + np.pi/2

        self.waypoints = []
        for i in range(num_points):
            wpnt = {
                'x_m': x_array[i],
                'y_m': y_array[i],
                's_m': s_array[i],
                'vx_mps': speed_array[i],
                'heading': heading_array[i],
                'w_right': 0.5,
                'w_left': 0.5
            }
            self.waypoints.append(wpnt)

        self.waypoints_s_array = s_array
        self.max_s = s_array[-1]

        self.get_logger().info(f'Dummy centerline: {len(self.waypoints)} waypoints')

    def get_obstacle_state(self):
        """Get obstacle position, heading, speed (with reactive perturbation if enabled)"""
        if len(self.waypoints) == 0:
            return None, None, None, None

        idx = np.abs(self.waypoints_s_array - self.current_s).argmin()
        wpnt = self.waypoints[idx]

        x_m = wpnt['x_m']
        y_m = wpnt['y_m']
        heading = wpnt['heading']

        if self.reactive:
            w_left = wpnt['w_left']
            w_right = wpnt['w_right']

            # Update phase
            phase_increment = 2.0 * np.pi * self.reactive_frequency * self.looptime
            new_phase = self.reactive_phase + phase_increment

            # New cycle → random amplitude
            if new_phase >= 2.0 * np.pi:
                max_amp = min(w_left, w_right, self.max_amplitude_limit)
                self.reactive_amplitude = np.random.uniform(0.3 * max_amp, max_amp)
                new_phase = new_phase % (2.0 * np.pi)

            self.reactive_phase = new_phase

            # Lateral perturbation (sine wave)
            d_perturbation = self.reactive_amplitude * np.sin(self.reactive_phase)

            # Apply in normal direction (perpendicular to heading)
            normal_x = -np.sin(heading)
            normal_y = np.cos(heading)
            x_m += d_perturbation * normal_x
            y_m += d_perturbation * normal_y

            # Heading adjustment from lateral velocity
            d_dot = (d_perturbation - self.prev_d_perturbation) / self.looptime
            self.prev_d_perturbation = d_perturbation
            if wpnt['vx_mps'] > 0.1:
                heading += np.arctan2(d_dot, wpnt['vx_mps'])

        return x_m, y_m, heading, wpnt['vx_mps']

    def publish_state(self, x_m, y_m, heading):
        """Publish obstacle state as Marker"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = x_m
        marker.pose.position.y = y_m
        marker.pose.position.z = 0.0

        # Set orientation from heading
        marker.pose.orientation.z = np.sin(heading / 2.0)
        marker.pose.orientation.w = np.cos(heading / 2.0)

        # Set size
        marker.scale.x = self.obstacle_length_m
        marker.scale.y = self.obstacle_width_m
        marker.scale.z = 0.5

        # Color doesn't matter for state, but set for visibility
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.state_pub.publish(marker)

    def publish_visualization_marker(self, x_m, y_m, heading):
        """Publish visualization marker (red cube)"""
        marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = x_m
        marker.pose.position.y = y_m
        marker.pose.position.z = 0.0

        marker.pose.orientation.z = np.sin(heading / 2.0)
        marker.pose.orientation.w = np.cos(heading / 2.0)

        marker.scale.x = self.obstacle_length_m
        marker.scale.y = self.obstacle_width_m
        marker.scale.z = 0.5

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def timer_callback(self):
        """Main loop - publish obstacle state"""
        if len(self.waypoints) == 0:
            return

        x_m, y_m, heading, speed_mps = self.get_obstacle_state()

        if x_m is None:
            return

        # Publish obstacle state
        self.publish_state(x_m, y_m, heading)

        # Publish visualization marker
        self.publish_visualization_marker(x_m, y_m, heading)

        # Update position
        self.current_s = (self.current_s + speed_mps * self.looptime) % self.max_s


    def destroy_node(self):
        """Publish DELETE marker before shutdown so gym_bridge clears dynamic obstacle."""
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.id = 0
        delete_marker.action = Marker.DELETE
        self.state_pub.publish(delete_marker)
        self.get_logger().info('[DynamicObstaclePublisher] Sent DELETE marker on shutdown')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstaclePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
