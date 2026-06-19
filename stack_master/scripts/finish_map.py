#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cartographer_ros_msgs.srv import FinishTrajectory, WriteState
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import subprocess
import sys
import os


class FinishMapNode(Node):
    def __init__(self):
        super().__init__('finish_map_node')

        # Declare and get parameters
        self.declare_parameter('map_name', 'my_map')
        self.declare_parameter('maps_dir', '')

        self.map_name = self.get_parameter('map_name').value
        self.maps_dir = self.get_parameter('maps_dir').value

        if not self.maps_dir:
            self.get_logger().error('maps_dir parameter is required!')
            sys.exit(1)

        # Create maps directory if it doesn't exist
        os.makedirs(self.maps_dir, exist_ok=True)

        self.get_logger().info(f'Map name: {self.map_name}')
        self.get_logger().info(f'Maps directory: {self.maps_dir}')

        # Create service clients for Cartographer
        self.finish_trajectory_client = self.create_client(
            FinishTrajectory, '/finish_trajectory')
        self.write_state_client = self.create_client(
            WriteState, '/write_state')

        # Wait for services to be available
        self.get_logger().info('Waiting for Cartographer services...')
        if not self.finish_trajectory_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/finish_trajectory service not available')
        if not self.write_state_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/write_state service not available')

        # Subscribe to /goal_pose topic from RViz
        # Use RELIABLE QoS to prevent message loss over network
        goal_pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        self.goal_pose_subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            goal_pose_qos)

        # Flag to trigger map saving
        self.should_save = False

        # Timer to check if we should save
        self.timer = self.create_timer(0.1, self.check_and_save)

        self.get_logger().warning("\n" + "="*60)
        self.get_logger().warning("Click '2D Goal Pose' in RViz to finish mapping and save the map")
        self.get_logger().warning("="*60 + "\n")

    def goal_pose_callback(self, msg: PoseStamped):
        """Callback when goal pose is received from RViz"""
        self.get_logger().info('Goal pose received from RViz. Triggering map save...')
        self.should_save = True

    def check_and_save(self):
        """Check if we should save the map"""
        if self.should_save:
            self.timer.cancel()
            self.finish_and_save_map()

    def finish_and_save_map(self):
        """Finish trajectory and save map"""
        self.get_logger().info('Starting map save process...')

        # Step 1: Finish trajectory
        self.get_logger().info('Step 1/3: Finishing trajectory 0...')
        finish_req = FinishTrajectory.Request()
        finish_req.trajectory_id = 0

        finish_future = self.finish_trajectory_client.call_async(finish_req)
        finish_future.add_done_callback(self.finish_trajectory_callback)

    def finish_trajectory_callback(self, future):
        """Callback when finish trajectory is done"""
        try:
            result = future.result()
            if result is not None:
                self.get_logger().info('Trajectory finished successfully')
                self.save_pbstream()
            else:
                self.get_logger().error('Failed to finish trajectory')
        except KeyboardInterrupt:
            raise  # Re-raise KeyboardInterrupt to shutdown properly
        except Exception as e:
            self.get_logger().error(f'Error finishing trajectory: {str(e)}')

    def save_pbstream(self):
        """Save the pbstream file"""
        pbstream_path = os.path.join(self.maps_dir, f'{self.map_name}.pbstream')
        self.get_logger().info(f'Step 2/3: Saving pbstream to {pbstream_path}...')

        write_req = WriteState.Request()
        write_req.filename = pbstream_path
        write_req.include_unfinished_submaps = True

        write_future = self.write_state_client.call_async(write_req)
        write_future.add_done_callback(self.write_state_callback)

    def write_state_callback(self, future):
        """Callback when write state is done"""
        try:
            result = future.result()
            if result is not None:
                self.get_logger().info('Pbstream saved successfully')
                self.save_occupancy_grid()
            else:
                self.get_logger().error('Failed to save pbstream')
        except KeyboardInterrupt:
            raise  # Re-raise KeyboardInterrupt to shutdown properly
        except Exception as e:
            self.get_logger().error(f'Error saving pbstream: {str(e)}')

    def save_occupancy_grid(self):
        """Save occupancy grid as PNG and YAML"""
        # Step 3: Save occupancy grid as PNG and YAML using map_saver
        map_path = os.path.join(self.maps_dir, self.map_name)
        pbstream_path = os.path.join(self.maps_dir, f'{self.map_name}.pbstream')

        self.get_logger().info(f'Step 3/3: Saving occupancy grid to {map_path}.png and {map_path}.yaml...')

        try:
            # Use nav2_map_server's map_saver_cli to save the map
            cmd = [
                'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                '-f', map_path,
                '--ros-args', '-p', 'save_map_timeout:=10000.0',
                '-r', '/map:=/map_carto'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            if result.returncode == 0:
                self.get_logger().info(f'Map image saved to {map_path}.png')
                self.get_logger().info(f'Map YAML saved to {map_path}.yaml')
            else:
                self.get_logger().error(f'Failed to save map image: {result.stderr}')
        except subprocess.TimeoutExpired:
            self.get_logger().error('Map saver timed out')
        except Exception as e:
            self.get_logger().error(f'Error saving map image: {str(e)}')

        self.get_logger().info("\n" + "="*60)
        self.get_logger().info(f"Map saving complete!")
        self.get_logger().info(f"  - {pbstream_path}")
        self.get_logger().info(f"  - {map_path}.png")
        self.get_logger().info(f"  - {map_path}.yaml")
        self.get_logger().info("="*60 + "\n")

        self.rebuild_workspace()

    def rebuild_workspace(self):
        """Run colcon build with symlink-install so the new map is immediately available."""
        # maps_dir = .../ws/src/creating_autonomous_car/stack_master/maps/<map_name>
        # workspace root is 5 levels up
        ws_dir = os.path.abspath(os.path.join(self.maps_dir, '../../../../..'))
        self.get_logger().info(f'Rebuilding workspace at: {ws_dir}')

        cmd = (
            'colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'
            ' && source install/setup.bash'
        )
        try:
            result = subprocess.run(
                ['bash', '-c', cmd],
                cwd=ws_dir,
                timeout=300
            )
            if result.returncode == 0:
                self.get_logger().info('Workspace rebuilt successfully. New map is now available.')
            else:
                self.get_logger().error(f'colcon build failed (exit code {result.returncode})')
        except subprocess.TimeoutExpired:
            self.get_logger().error('colcon build timed out (>5 min)')
        except Exception as e:
            self.get_logger().error(f'Error during colcon build: {str(e)}')

        # Shutdown the node
        self.get_logger().info('Shutting down finish_map node...')
        raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = FinishMapNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
