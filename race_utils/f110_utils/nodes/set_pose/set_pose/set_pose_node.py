#!/usr/bin/env python3
"""
Set the initial pose for cartographer localization (unicorn ROS1 set_pose_node port).

When the user clicks "2D Pose Estimate" in RViz (-> /initialpose), the current
cartographer trajectory is finished and a new one is started from the chosen pose.
In ROS1 this used `rosservice call`; in ROS2 we use rclpy service clients on the
cartographer_ros FinishTrajectory / StartTrajectory services.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

from cartographer_ros_msgs.srv import FinishTrajectory, StartTrajectory


class SetInitialPose(Node):
    def __init__(self):
        super().__init__('setpose')

        # e.g. '/home/.../config/common/slam'
        self.declare_parameter('config_dir', '')
        self.declare_parameter('config_base', 'localization.lua')
        self.CONFIG_DIR = self.get_parameter('config_dir').get_parameter_value().string_value
        self.CONFIG_BASE = self.get_parameter('config_base').get_parameter_value().string_value

        self.initial_pose = PoseWithCovarianceStamped()
        self.ready = False

        self.get_logger().info("Click the 2D Pose Estimate button in RViz to set the robot's pose...")
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

        # service clients
        self.finish_cli = self.create_client(FinishTrajectory, '/finish_trajectory')
        self.start_cli = self.create_client(StartTrajectory, '/start_trajectory')

        # 0 is the one we saved during mapping, so in loc only we start trajectory 1
        self.trajectory_num = 1

    def update_initial_pose(self, msg):
        self.get_logger().info(f'initial pos {msg.pose.pose.position}')
        self.get_logger().info(f'initial orientation {msg.pose.pose.orientation}')
        self.initial_pose = msg

        # finish current trajectory
        if self.finish_cli.wait_for_service(timeout_sec=5.0):
            finish_req = FinishTrajectory.Request()
            finish_req.trajectory_id = int(self.trajectory_num)
            self.finish_cli.call_async(finish_req)
        else:
            self.get_logger().error('/finish_trajectory service unavailable')

        # start a new trajectory from the given pose, relative to trajectory 0
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
