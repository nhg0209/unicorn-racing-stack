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
    # which WpntArray to follow: '/global_waypoints' (time-trial) or
    # '/local_waypoints' (head-to-head: state-machine output w/ avoidance)
    'waypoint_topic':  '/global_waypoints',
    'odom_topic':      '/vesc/odom',
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

        self.create_subscription(Odometry,   p('odom_topic'),      self._odom_cb, 10)
        # Use a plain VOLATILE depth-10 QoS for the waypoint source: it is
        # compatible with BOTH the state machine's volatile /local_waypoints
        # (head-to-head) and waypoint_publisher's TRANSIENT_LOCAL
        # /global_waypoints (time-trial, republished at 1 Hz). A TRANSIENT_LOCAL
        # subscriber would silently reject the volatile /local_waypoints.
        self.create_subscription(WpntArray,  p('waypoint_topic'),  self._wp_cb, 10)
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

        # Step 1. vehicle pose (waypoints are in the map frame; sim /vesc/odom is GT in map)
        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cy, sy = math.cos(-yaw), math.sin(-yaw)

        # Speed-adaptive lookahead keeps PP stable across the speed profile
        v_now = math.hypot(self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y)
        Ld = min(3.0, max(self.lookahead, 0.5 + 0.30 * v_now))

        # Step 2. transform waypoints into the vehicle frame, pick the lookahead point
        target_idx = -1
        best_err = float('inf')
        tx = ty = 0.0
        for i, w in enumerate(self.waypoints):
            dx = float(w.x_m) - p.x
            dy = float(w.y_m) - p.y
            lx = cy * dx - sy * dy          # forward (+x ahead of the car)
            ly = sy * dx + cy * dy          # left
            if lx <= 0.0:
                continue                    # only consider points ahead
            err = abs(math.hypot(lx, ly) - Ld)
            if err < best_err:
                best_err = err
                target_idx = i
                tx, ty = lx, ly

        if target_idx < 0:
            # nothing ahead (e.g. just spawned facing away) → creep straight to recover
            return 0.0, 1.0

        # Step 3. curvature → steering (bicycle model)
        L2 = tx * tx + ty * ty
        if L2 < 1e-6:
            return 0.0, 1.0
        curvature = 2.0 * ty / L2
        steer = math.atan(self.wheelbase * curvature)
        steer = max(-self.max_steer, min(self.max_steer, steer))

        self._publish_lookahead(self.waypoints[target_idx])

        # Step 4. speed from the target waypoint's profile velocity
        speed = float(self.waypoints[target_idx].vx_mps)
        if speed <= 0.1:
            speed = 1.5
        return steer, speed


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
