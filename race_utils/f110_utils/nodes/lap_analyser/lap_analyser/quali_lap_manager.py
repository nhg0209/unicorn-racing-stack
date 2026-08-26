#!/usr/bin/env python3
"""quali_lap_manager — qualification lap manager + speed governor.

RViz "2D Goal Pose" (/goal_pose) sets the start/finish line (nearest global
waypoint, same convention as lap_analyser). From that line:

  laps 1..num_slow_laps : controller output speed is scaled by slow_speed_scale
  afterwards            : commands pass through untouched (full pace), and
                          /quali/obstacles_off is latched True so
                          tracking_quali stops publishing obstacles entirely
                          (the crew is clearing the track; no more avoidance).
  During the slow laps obstacles are visible (all static, via tracking_quali).

Until the goal pose is set AND the line is first crossed, the slow scale is
applied (safe default: the car never runs full pace before the quali start is
armed).

Wiring (quali.launch.xml): the controller's drive_topic param is pointed at
cmd_in_topic, this node republishes to cmd_out_topic (the topic simple_mux
actually consumes). Man-in-the-middle, no controller changes.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import Marker


class QualiLapManager(Node):
    def __init__(self):
        super().__init__('quali_lap_manager')
        self.declare_parameter('num_slow_laps', 8)
        self.declare_parameter('slow_speed_scale', 0.5)
        self.declare_parameter('cmd_in_topic', '/quali/ackermann_cmd_raw')
        self.declare_parameter('cmd_out_topic', '/vesc/high_level/ackermann_cmd')

        self.num_slow_laps = int(self.get_parameter('num_slow_laps').value)
        self.slow_scale = float(self.get_parameter('slow_speed_scale').value)
        cmd_in = str(self.get_parameter('cmd_in_topic').value)
        cmd_out = str(self.get_parameter('cmd_out_topic').value)

        self.gb_sxy = None            # [s, x, y] per waypoint
        self.start_s = None           # armed by RViz 2D Goal Pose
        self.start_xy = None
        self.last_s = None
        self.lap_count = -1           # -1 = line not yet crossed; N = laps completed

        self.create_subscription(WpntArray, '/global_waypoints', self.wpnts_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom_frenet', self.frenet_cb, 10)
        self.create_subscription(AckermannDriveStamped, cmd_in, self.cmd_cb, 10)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, cmd_out, 10)
        self.lap_pub = self.create_publisher(Int32, '/quali/lap_count', latched)
        self.obstacles_off_pub = self.create_publisher(Bool, '/quali/obstacles_off', latched)
        self.marker_pub = self.create_publisher(Marker, '/quali/marker', 5)

        self.obstacles_off_pub.publish(Bool(data=False))
        self.lap_pub.publish(Int32(data=self.lap_count))
        self.get_logger().info(
            f"[quali] governor {cmd_in} -> {cmd_out} | first {self.num_slow_laps} laps "
            f"x{self.slow_scale:.2f}; set the start line with RViz 2D Goal Pose")

    # -------------------------------------------------- phase
    @property
    def slow_active(self):
        return self.lap_count < self.num_slow_laps

    # -------------------------------------------------- callbacks
    def wpnts_cb(self, msg):
        if self.gb_sxy is None and msg.wpnts:
            self.gb_sxy = np.array([[w.s_m, w.x_m, w.y_m] for w in msg.wpnts])

    def goal_pose_cb(self, msg):
        if self.gb_sxy is None:
            self.get_logger().warn("[quali] goal pose ignored, waiting for /global_waypoints")
            return
        gx, gy = msg.pose.position.x, msg.pose.position.y
        idx = int(np.argmin((self.gb_sxy[:, 1] - gx) ** 2 + (self.gb_sxy[:, 2] - gy) ** 2))
        self.start_s = float(self.gb_sxy[idx, 0])
        self.start_xy = (float(self.gb_sxy[idx, 1]), float(self.gb_sxy[idx, 2]))
        self.lap_count = -1
        self.last_s = None
        self._announce()
        self.get_logger().info(
            f"[quali] start line armed at s={self.start_s:.2f} "
            f"(nearest wpnt to {gx:.2f},{gy:.2f}); lap count reset")

    def frenet_cb(self, msg):
        if self.start_s is None:
            return
        s = msg.pose.pose.position.x
        if self.last_s is not None and self._crossed(self.last_s, s):
            self.lap_count += 1
            self._announce()
        self.last_s = s

    def cmd_cb(self, msg):
        if self.slow_active:
            msg.drive.speed *= self.slow_scale
        self.cmd_pub.publish(msg)

    # -------------------------------------------------- helpers
    def _crossed(self, last_s, s):
        """Forward crossing of start_s, robust to the s wrap at 0."""
        if self.start_s <= 1e-6:
            return (last_s - s) > 1.0            # track wrap
        if (last_s - s) > 1.0:                   # wrapped this tick
            return s >= self.start_s or last_s < self.start_s
        return last_s < self.start_s <= s

    def _announce(self):
        self.lap_pub.publish(Int32(data=self.lap_count))
        self.obstacles_off_pub.publish(Bool(data=not self.slow_active))
        if self.slow_active:
            done = max(self.lap_count, 0)
            status = f"QUALI {done}/{self.num_slow_laps} SLOW x{self.slow_scale:.2f}"
        else:
            status = f"QUALI lap {self.lap_count} FULL PACE | obstacles IGNORED"
        self.get_logger().warn(f"[quali] {status}")
        if self.start_xy is not None:
            self._marker(status)

    def _marker(self, text):
        line = Marker()
        line.header.frame_id = 'map'
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns, line.id = 'quali_start', 0
        line.type, line.action = Marker.CYLINDER, Marker.ADD
        line.pose.position.x, line.pose.position.y = self.start_xy
        line.pose.orientation.w = 1.0
        line.scale.x = line.scale.y = 0.3
        line.scale.z = 1.0
        line.color.r, line.color.b, line.color.a = 1.0, 1.0, 0.9
        self.marker_pub.publish(line)

        txt = Marker()
        txt.header.frame_id = 'map'
        txt.header.stamp = line.header.stamp
        txt.ns, txt.id = 'quali_status', 1
        txt.type, txt.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        txt.pose.position.x, txt.pose.position.y = self.start_xy
        txt.pose.position.z = 1.3
        txt.scale.z = 0.4
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = text
        self.marker_pub.publish(txt)


def main():
    rclpy.init()
    node = QualiLapManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
