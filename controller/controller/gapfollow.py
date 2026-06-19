import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
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

        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
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
        scan = self.scan
        n = len(scan.ranges)
        rng = np.asarray(scan.ranges, dtype=np.float32)
        # 1. preprocess: invalid -> 0, clip to max_range
        rng = np.nan_to_num(rng, nan=0.0, posinf=self.max_range, neginf=0.0)
        rng = np.clip(rng, 0.0, self.max_range)

        # restrict to the front field of view (+/- 90 deg) so we never aim behind
        ang = scan.angle_min + np.arange(n) * scan.angle_increment
        front = np.abs(ang) <= math.radians(90.0)
        proc = np.where(front, rng, 0.0)

        # 2. safety bubble around the closest (valid) beam
        valid = proc[front]
        if valid.size == 0 or np.all(valid == 0.0):
            return 0.0, self.speed * 0.5
        closest = int(np.argmin(np.where(proc > 0.05, proc, np.inf)))
        if np.isfinite(proc[closest]) and proc[closest] > 0.05:
            bubble_beams = int(
                math.atan2(self.bubble_radius, max(proc[closest], 0.1)) /
                max(scan.angle_increment, 1e-4))
            lo = max(0, closest - bubble_beams)
            hi = min(n, closest + bubble_beams + 1)
            proc[lo:hi] = 0.0

        # 3. longest contiguous non-zero gap
        free = proc > 0.05
        best_start = best_len = cur_start = cur_len = 0
        for i in range(n):
            if free[i]:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        if best_len == 0:
            return 0.0, self.speed * 0.5

        # 4. best beam in the gap: farthest point (deepest free space)
        gap = proc[best_start:best_start + best_len]
        target = best_start + int(np.argmax(gap))

        # 5. beam -> steering, clamped
        steer = float(ang[target])
        steer = max(-self.max_steer, min(self.max_steer, steer))

        # 6. slow down for sharp turns / nearby obstacles
        speed = self.speed * (1.0 - 0.6 * abs(steer) / self.max_steer)
        speed = max(0.6, speed)
        return steer, speed


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
