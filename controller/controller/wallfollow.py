import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

PARAMS = {
    'control_rate_hz': 50.0,
    'wf_target_dist':   0.8,
    'wf_kp':            0.8,
    'wf_kd':            0.5,
    'wf_ki':            2.0,
    'wf_speed':         1.5,
    'wf_max_steer':     0.4,
    'wf_lookahead':     0.5,
    'wf_target_dist':   0.8,
}


class WallFollowNode(Node):

    def __init__(self):
        super().__init__('wall_follow')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.target_dist = p('wf_target_dist')
        self.kp          = p('wf_kp')
        self.kd          = p('wf_kd')
        self.ki          = p('wf_ki')
        self.speed       = p('wf_speed')
        self.lookahead   = p('wf_lookahead')
        self.desired_d   = p('wf_target_dist')

        self._prev_error = 0.0
        self._integral_error = 0.0

        self.scan    = None
        self.odom    = None
        self._prev_t = None

        self.create_subscription(LaserScan, '/scan',      self._scan_cb, 10)
        self.create_subscription(Odometry,  '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('WallFollowNode ready')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg

    def _loop(self):
        if self.scan is None or self.odom is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = (now - self._prev_t) if self._prev_t else 0.02
        self._prev_t = now


        steer, speed = self._compute(dt)

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def _compute(self, dt):
        # TODO: Implement Wall Follow using PID control (follow the right wall)
        #
        # Goal:
        #   - Keep the vehicle at a desired distance from the right wall
        #   - Use LiDAR scan data to estimate right-wall distance/error
        #   - Compute steering using PID control
        #
        # You should return (steering, speed) from this function.
        #
        # Useful information:
        #   - self.scan.ranges             : LiDAR distance array [m]
        #   - self.scan.angle_min          : angle of first beam [rad]
        #   - self.scan.angle_increment    : angular step between beams [rad]
        #   - self.target_dist             : desired wall distance [m]
        #   - self.lookahead               : lookahead distance for projected wall error [m]
        #   - self.kp                      : proportional gain
        #   - self.ki                      : integral gain
        #   - self.kd                      : derivative gain
        #   - self._prev_error             : previous wall-distance error (for D term)
        #   - self._integral_error         : accumulated error (for I term)
        #   - dt                           : control timestep [s]
        #
        # Suggested approach (right-wall geometry):
        #   - Pick two LiDAR beams on the right side
        #   - Estimate wall angle (alpha) from the two ranges
        #   - Estimate current perpendicular distance to the wall
        #   - Project the distance error forward using lookahead
        #   - error = target_dist - projected_right_wall_distance
        #   - integral_error += error * dt
        #   - derivative = (error - prev_error) / dt
        #   - steering = kp * error + ki * integral_error + kd * derivative
        #   - Clamp steering to max steering angle if needed
        #
        # Output:
        #   - steering [rad]
        #   - speed [m/s]

        speed = 0.0 
        steer = 0.0

        return steer, speed


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
