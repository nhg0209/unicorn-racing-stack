#!/usr/bin/env python3
"""
Save the current cartographer map (.pbstream) when the user clicks
"Publish Point" in RViz (-> /clicked_point). Unicorn ROS1 save_carto_map port.

In ROS1 the map base directory was derived from the stack_master package path;
in ROS2 the stack_master package is ament_cmake (no python module), so the map
base directory is provided via the `maps_dir` parameter (typically the installed
stack_master/maps share directory). Filenames are auto-versioned to avoid
overwriting existing pbstreams.
"""
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

from cartographer_ros_msgs.srv import (
    FinishTrajectory, GetTrajectoryStates, WriteState)


class SaveCartographerMap(Node):
    def __init__(self):
        super().__init__('savemap')

        self.declare_parameter('config_dir', '')
        self.declare_parameter('config_base', 'localization.lua')
        self.declare_parameter('map', '')
        self.declare_parameter('maps_dir', '')

        self.CONFIG_DIR = self.get_parameter('config_dir').get_parameter_value().string_value
        self.CONFIG_BASE = self.get_parameter('config_base').get_parameter_value().string_value
        self.map = self.get_parameter('map').get_parameter_value().string_value
        self.maps_dir = self.get_parameter('maps_dir').get_parameter_value().string_value

        self.base_path = os.path.join(self.maps_dir, self.map)
        self.map_path = os.path.join(self.base_path, self.map + '.pbstream')

        self.initial_pose = PointStamped()
        self.ready = False

        self.get_logger().info("Click the Publish Point button in RViz to save the cartographer...")
        self.create_subscription(PointStamped, '/clicked_point', self.save_map, 10)

        self.finish_cli = self.create_client(FinishTrajectory, '/finish_trajectory')
        self.states_cli = self.create_client(GetTrajectoryStates, '/get_trajectory_states')
        self.write_cli = self.create_client(WriteState, '/write_state')

        self.trajectory_num = 1

    def get_next_trajectory_id(self):
        if not self.states_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/get_trajectory_states service unavailable')
            return 1
        future = self.states_cli.call_async(GetTrajectoryStates.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result()
        if response is None:
            self.get_logger().error('Service call failed: get_trajectory_states')
            return 1
        try:
            trajectory_ids = response.trajectory_states.trajectory_id
            if len(trajectory_ids):
                return trajectory_ids[-1]
            return 1
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return 1

    def save_map(self, msg):
        self.trajectory_num = self.get_next_trajectory_id()

        if self.finish_cli.wait_for_service(timeout_sec=5.0):
            finish_req = FinishTrajectory.Request()
            finish_req.trajectory_id = int(self.trajectory_num)
            future = self.finish_cli.call_async(finish_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        else:
            self.get_logger().error('/finish_trajectory service unavailable')

        os.makedirs(self.base_path, exist_ok=True)
        file_path = os.path.join(self.base_path, self.map + '.pbstream')
        version = 2
        while os.path.exists(file_path):
            file_path = os.path.join(self.base_path, f"{self.map}_v{version}.pbstream")
            version += 1

        if self.write_cli.wait_for_service(timeout_sec=5.0):
            write_req = WriteState.Request()
            write_req.filename = file_path
            write_req.include_unfinished_submaps = True
            self.get_logger().info(f"Calling /write_state -> {file_path}")
            self.write_cli.call_async(write_req)
        else:
            self.get_logger().error('/write_state service unavailable')


def main():
    rclpy.init()
    node = SaveCartographerMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping savemap...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
