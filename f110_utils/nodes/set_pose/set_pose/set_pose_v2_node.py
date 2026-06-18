#!/usr/bin/env python3
"""
set_pose_v2: like set_pose_node but queries the current trajectory id from
cartographer (GetTrajectoryStates) before finishing it (unicorn ROS1 port).
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

from cartographer_ros_msgs.srv import (
    FinishTrajectory, StartTrajectory, GetTrajectoryStates)


class SetInitialPose(Node):
    def __init__(self):
        super().__init__('setpose')

        self.declare_parameter('config_dir', '')
        self.declare_parameter('config_base', 'localization.lua')
        self.CONFIG_DIR = self.get_parameter('config_dir').get_parameter_value().string_value
        self.CONFIG_BASE = self.get_parameter('config_base').get_parameter_value().string_value

        self.initial_pose = PoseWithCovarianceStamped()
        self.ready = False

        self.get_logger().info("Click the 2D Pose Estimate button in RViz to set the robot's pose...")
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

        self.finish_cli = self.create_client(FinishTrajectory, '/finish_trajectory')
        self.start_cli = self.create_client(StartTrajectory, '/start_trajectory')
        self.states_cli = self.create_client(GetTrajectoryStates, '/get_trajectory_states')

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

    def update_initial_pose(self, msg):
        self.get_logger().info(f'initial pos {msg.pose.pose.position}')
        self.get_logger().info(f'initial orientation {msg.pose.pose.orientation}')
        self.initial_pose = msg

        self.trajectory_num = self.get_next_trajectory_id()

        if self.finish_cli.wait_for_service(timeout_sec=5.0):
            finish_req = FinishTrajectory.Request()
            finish_req.trajectory_id = int(self.trajectory_num)
            self.finish_cli.call_async(finish_req)
        else:
            self.get_logger().error('/finish_trajectory service unavailable')

        if self.start_cli.wait_for_service(timeout_sec=5.0):
            start_req = StartTrajectory.Request()
            start_req.configuration_directory = self.CONFIG_DIR
            start_req.configuration_basename = self.CONFIG_BASE
            start_req.use_initial_pose = True
            start_req.initial_pose = msg.pose.pose
            start_req.relative_to_trajectory_id = 0
            self.start_cli.call_async(start_req)
        else:
            self.get_logger().error('/start_trajectory service unavailable')

        self.trajectory_num += 1


def main():
    rclpy.init()
    node = SetInitialPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Localize finished.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
