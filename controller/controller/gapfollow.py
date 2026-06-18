import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from controller.estop import EStop

PARAMS = {
    'control_rate_hz':  50.0,
    'gf_bubble_radius':  0.3,
    'gf_speed':          2.0,
    'gf_max_steer':      0.4,
    'gf_max_range':     10.0,
}


class GapFollowNode(Node):

    def __init__(self):
        super().__init__('gap_follow')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.estop         = EStop(self)
        self.bubble_radius = p('gf_bubble_radius')
        self.speed         = p('gf_speed')
        self.max_steer     = p('gf_max_steer')
        self.max_range     = p('gf_max_range')

        self.scan = None
        self.odom = None

        self.create_subscription(LaserScan, '/scan',      self._scan_cb, 10)
        self.create_subscription(Odometry,  '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('GapFollowNode ready')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg

    def _loop(self):
        if self.scan is None or self.odom is None:
            return

        steer, speed = self._compute()

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def _compute(self):
        # TODO: Implement Follow-the-Gap (FTG) using LiDAR free-space selection
        #
        # Goal:
        #   - Use LiDAR scan data to find a collision-free gap
        #   - Select a safe target direction inside the best gap
        #   - Convert the target direction into a steering command
        #
        # You should return (steering, speed) from this function.
        #
        # Useful information:
        #   - self.scan.ranges             : LiDAR distance array [m]
        #   - self.scan.angle_min          : angle of first beam [rad]
        #   - self.scan.angle_increment    : angular step between beams [rad]
        #   - self.bubble_radius           : safety bubble radius around closest obstacle [m]
        #   - self.max_range               : cap LiDAR ranges to suppress outliers [m]
        #   - self.max_steer               : steering clamp [rad]
        #   - self.speed                   : nominal driving speed [m/s]
        #   - self.odom                    : current vehicle motion (optional for speed adjustment)
        #
        # Suggested approach (FTG pipeline):
        #   - Preprocess ranges:
        #       * replace NaN/Inf/invalid values
        #       * clip ranges to [0, self.max_range]
        #       * optionally focus on a front field-of-view
        #   - Find the closest obstacle beam
        #   - Create a safety bubble around that obstacle (zero-out nearby beams)
        #   - Find the longest contiguous non-zero gap
        #   - Choose the best target beam in the gap
        #       * e.g., farthest beam or weighted by distance and heading
        #   - Convert target beam index to steering angle
        #   - Clamp steering to +/- self.max_steer
        #   - Optionally reduce speed when |steering| is large or obstacle is close
        #
        # Output:
        #   - steering [rad]
        #   - speed [m/s]
        return 0.0, 0.0


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
