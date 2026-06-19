#!/usr/bin/env python3
"""
controller_ros.py - main racing controller (ROS1 controller_manager equivalent).

Follows the state machine's BehaviorStrategy: a Pure-Pursuit lateral law over the
local waypoints, using the per-waypoint speed (already capped by the state
machine's velocity planner for trailing / sectors). Publishes the drive command
to the same mux input the rest of the stack uses.

Subscribes:
  /behavior_strategy  (f110_msgs/BehaviorStrategy) - local waypoints + state + targets
  <odom_topic>        (nav_msgs/Odometry)          - ego pose (map frame)

Publishes:
  /vesc/high_level/ackermann_cmd (ackermann_msgs/AckermannDriveStamped)
  /controller/lookahead          (visualization_msgs/Marker)
  /controller/state              (std_msgs/String)
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from f110_msgs.msg import BehaviorStrategy

PARAMS = {
    'control_rate_hz': 50.0,
    'pp_lookahead':     1.0,
    'pp_wheelbase':    0.33,
    'pp_max_steer':     0.4,
    'min_speed':        0.8,
    'odom_topic':      '/vesc/odom',
}


class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller')
        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value

        self.lookahead = p('pp_lookahead')
        self.wheelbase = p('pp_wheelbase')
        self.max_steer = p('pp_max_steer')
        self.min_speed = p('min_speed')

        self.odom = None
        self.wpnts = []
        self.state = ''

        self.create_subscription(Odometry, p('odom_topic'), self._odom_cb, 10)
        self.create_subscription(BehaviorStrategy, '/behavior_strategy', self._bs_cb, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.look_pub = self.create_publisher(Marker, '/controller/lookahead', 10)
        self.state_pub = self.create_publisher(String, '/controller/state', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)
        self.get_logger().info('ControllerNode ready (follows /behavior_strategy)')

    def _odom_cb(self, msg):
        self.odom = msg

    def _bs_cb(self, msg):
        self.wpnts = msg.local_wpnts
        self.state = msg.state

    def _loop(self):
        if self.odom is None or not self.wpnts:
            return
        steer, speed = self._pure_pursuit()
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.drive.steering_angle = steer
        out.drive.speed = speed
        self.drive_pub.publish(out)
        self.state_pub.publish(String(data=self.state))

    def _pure_pursuit(self):
        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cy, sy = math.cos(-yaw), math.sin(-yaw)

        v_now = math.hypot(self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y)
        Ld = min(3.0, max(self.lookahead, 0.5 + 0.30 * v_now))

        target = -1
        best = float('inf')
        tx = ty = 0.0
        for i, w in enumerate(self.wpnts):
            dx = float(w.x_m) - p.x
            dy = float(w.y_m) - p.y
            lx = cy * dx - sy * dy
            ly = sy * dx + cy * dy
            if lx <= 0.0:
                continue
            err = abs(math.hypot(lx, ly) - Ld)
            if err < best:
                best, target, tx, ty = err, i, lx, ly

        if target < 0:
            return 0.0, self.min_speed

        L2 = tx * tx + ty * ty
        if L2 < 1e-6:
            return 0.0, self.min_speed
        steer = math.atan(self.wheelbase * (2.0 * ty / L2))
        steer = max(-self.max_steer, min(self.max_steer, steer))

        self._publish_lookahead(self.wpnts[target])
        speed = float(self.wpnts[target].vx_mps)
        if speed <= 0.1:
            speed = self.min_speed
        return steer, speed

    def _publish_lookahead(self, wp):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = 'controller_lookahead'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(wp.x_m)
        m.pose.position.y = float(wp.y_m)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.g = 1.0
        m.color.a = 1.0
        self.look_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
