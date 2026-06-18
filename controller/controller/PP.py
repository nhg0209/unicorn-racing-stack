import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from f110_msgs.msg import WpntArray

PARAMS = {
    'control_rate_hz': 50.0,
    'pp_lookahead':     1.0,
    'pp_wheelbase':    0.33,
    'pp_max_steer':     0.4,
}


class PPNode(Node):

    def __init__(self):
        super().__init__('pp')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.lookahead = p('pp_lookahead')
        self.wheelbase = p('pp_wheelbase')
        self.max_steer = p('pp_max_steer')

        self.odom      = None
        self.waypoints = []

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(Odometry,   '/vesc/odom',        self._odom_cb, 10)
        self.create_subscription(WpntArray,  '/global_waypoints', self._wp_cb, latched)
        self.drive_pub     = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/pp/lookahead', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('PPNode ready')

    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self, msg):   self.waypoints = msg.wpnts

    def _publish_lookahead(self, wp):
        """Publish the chosen lookahead waypoint as a green sphere in RViz (frame=map)."""
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = 'pp_lookahead'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(wp.x_m)
        m.pose.position.y = float(wp.y_m)
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 1.0
        self.lookahead_pub.publish(m)

    def _loop(self):
        if self.odom is None or not self.waypoints:
            return

        steer, speed = self._compute()

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed

        self.drive_pub.publish(msg)

    def _compute(self):
        # TODO: Pure Pursuit algorithm
        # inputs : self.odom, self.waypoints, self.lookahead, self.wheelbase
        # output : (steering [rad], speed [m/s])
        #
        # Pseudo-code
        # ┌─ Step 1. extract vehicle pose
        # │   - position p = self.odom.pose.pose.position           (map frame)
        # │   - orientation q → yaw  (quaternion → atan2)
        # │
        # ├─ Step 2. pick the lookahead point
        # │   - transform every waypoint into the vehicle frame
        # │     (translate by -p, rotate by -yaw)
        # │   - among the points in front (local_x > 0), pick the one whose
        # │     distance is closest to self.lookahead and use it as the target
        # │   - if no candidate exists, return a safe fallback (e.g. 0, 0)
        # │   - debug: after picking the target, call
        # │     self._publish_lookahead(self.waypoints[target_idx])
        # │     to mark it as a green sphere on the /pp/lookahead topic in RViz
        # │
        # ├─ Step 3. curvature → steering
        # │   - target in vehicle frame (lx, ly), L = sqrt(lx² + ly²)
        # │   - curvature κ = 2 · ly / L²
        # │   - steering δ = atan(self.wheelbase · κ)
        # │   - clip δ to ±self.max_steer
        # │
        # └─ Step 4. speed
        #     - use the target waypoint's vx_mps directly
        #       (or average a few waypoints around the lookahead)
        #     - if vx is 0 or negative, fall back to a sane default (e.g. 1.5 m/s)

        #TODO

        raise NotImplementedError('PPNode._compute is not implemented yet')

        #TODO


def main(args=None):
    rclpy.init(args=args)
    node = PPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
