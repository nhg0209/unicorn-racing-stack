#!/usr/bin/env python3
"""opponent_vehicle — engine-agnostic virtual opponent (works in sim AND real).

A single virtual opponent driven by a kinematic bicycle model. It is NOT part of
any physics engine, so the exact same node spawns an opponent in the f1tenth_gym
simulation and on the real car.

Inputs:
  /opp_drive (AckermannDriveStamped) : speed + steering from opponent_controller
  /goal_pose (PoseStamped)           : RViz "2D Goal Pose" -> spawn / relocate
  /sim/remove_opponent (Empty)       : park off-map / hide
  <ego_odom_topic> (Odometry)        : ego pose, so the opponent's own lidar sees it
Outputs:
  /vil/opponents (MarkerArray)       : opponent box -> scan_augmentor overlays it
  /opp_scan (LaserScan)              : opponent's OWN lidar (map + ego box) for FTG
  <opp>/odom (Odometry), TF map-><opp>/base_link, /vil/opponents_viz (markers)
"""
import os
import sys
import math
import importlib.util

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Empty, Bool
from f110_msgs.msg import Obstacle, ObstacleArray
from tf2_ros import TransformBroadcaster
from transforms3d import euler


def _load_RaycastEngine(rc_dir):
    path = os.path.join(rc_dir, 'raycaster.py')
    spec = importlib.util.spec_from_file_location('raycaster', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['raycaster'] = mod
    spec.loader.exec_module(mod)
    return mod.RaycastEngine


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class OpponentVehicle(Node):
    def __init__(self):
        super().__init__('opponent_vehicle')

        self.declare_parameter('map_path', '')
        self.declare_parameter('raycaster_dir',
                               '/home/js/unicorn_racing_stack/src/unicorn-racing-stack/race_utils/raycaster')
        self.declare_parameter('ego_odom_topic', '/car_state/odom')
        self.declare_parameter('opp_drive_topic', '/opp_drive')
        self.declare_parameter('opp_scan_topic', '/opp_scan')
        self.declare_parameter('opp_odom_topic', '/opp_racecar/odom')
        self.declare_parameter('opp_frame', 'opp_racecar/base_link')
        self.declare_parameter('opp_laser_frame', 'opp_racecar/laser')
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('length', 0.58)
        self.declare_parameter('width', 0.31)
        self.declare_parameter('box_size', 0.2)            # lidar/detection box side (m) at base_link
        self.declare_parameter('dynamic_obstacles_topic', '/sim/dynamic_obstacles')
        self.declare_parameter('opp_lidar_default_on', False)  # opp self-lidar OFF by default
        self.declare_parameter('scan_beams', 1080)
        self.declare_parameter('scan_fov', 4.7)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('scan_distance_to_base_link', 0.27)
        self.declare_parameter('rate_hz', 40.0)
        self.declare_parameter('start_with_opp', False)

        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.length = float(self.get_parameter('length').value)
        self.width = float(self.get_parameter('width').value)
        self.box_size = float(self.get_parameter('box_size').value)
        self.opp_scan_on = bool(self.get_parameter('opp_lidar_default_on').value)
        self.beams = int(self.get_parameter('scan_beams').value)
        self.fov = float(self.get_parameter('scan_fov').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.scan_dist = float(self.get_parameter('scan_distance_to_base_link').value)
        self.opp_frame = self.get_parameter('opp_frame').value
        self.opp_ns = self.opp_frame.rsplit('/', 1)[0]   # 'opp_racecar/base_link' -> 'opp_racecar'
        self.opp_laser_frame = self.get_parameter('opp_laser_frame').value
        rate = float(self.get_parameter('rate_hz').value)

        # opponent state (kinematic bicycle): x, y, theta, v
        self.PARK = (1000.0, 1000.0, 0.0)
        self.has_opp = bool(self.get_parameter('start_with_opp').value)
        self.x, self.y, self.th = self.PARK
        self.v = 0.0
        self.cmd_speed = 0.0
        self.cmd_steer = 0.0
        self.ego_pose = None   # (x, y, yaw) for the opponent's own lidar

        # raycaster for the opponent's OWN lidar (map + ego box)
        self.eng = None
        map_path = self.get_parameter('map_path').value
        rc_dir = self.get_parameter('raycaster_dir').value
        if map_path:
            try:
                RaycastEngine = _load_RaycastEngine(rc_dir)
                occ, res, origin = RaycastEngine.load_map_yaml(map_path)
                self.eng = RaycastEngine('rm', max_range_m=self.max_range).set_map(occ, res, origin)
                self.get_logger().info(f'[opponent_vehicle] opp lidar map loaded from {map_path}')
            except Exception as e:
                self.get_logger().warn(f'[opponent_vehicle] opp lidar disabled (map load failed: {e})')

        # I/O
        self.create_subscription(AckermannDriveStamped,
                                 self.get_parameter('opp_drive_topic').value, self._drive_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self._spawn_cb, 10)
        self.create_subscription(Empty, '/sim/remove_opponent', self._remove_cb, 10)
        self.create_subscription(Odometry,
                                 self.get_parameter('ego_odom_topic').value, self._ego_cb, 10)
        # opp self-lidar enable (default OFF; FTG / panel turns it on)
        self.create_subscription(Bool, '/sim/opp_lidar_enable', self._lidar_cb, 10)

        # opponent as f110_msgs/Obstacle -> overlay (scan_augmentor) AND concat (obstacle_merger)
        self.obs_pub = self.create_publisher(
            ObstacleArray, self.get_parameter('dynamic_obstacles_topic').value, 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/vil/opponents_viz', 1)
        self.odom_pub = self.create_publisher(Odometry, self.get_parameter('opp_odom_topic').value, 10)
        self.scan_pub = self.create_publisher(LaserScan,
                                              self.get_parameter('opp_scan_topic').value,
                                              qos_profile_sensor_data)
        self.br = TransformBroadcaster(self)

        self.dt = 1.0 / rate
        self.create_timer(self.dt, self._loop)
        self.get_logger().info(
            f"[opponent_vehicle] up (has_opp={self.has_opp}). Spawn with RViz 2D Goal Pose.")

    # ---- callbacks ----
    def _drive_cb(self, msg):
        self.cmd_speed = float(msg.drive.speed)
        self.cmd_steer = float(msg.drive.steering_angle)

    def _ego_cb(self, msg):
        p = msg.pose.pose
        self.ego_pose = (p.position.x, p.position.y, _yaw(p.orientation))

    def _spawn_cb(self, msg):
        q = msg.pose.orientation
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.th = _yaw(q)
        self.v = 0.0
        self.cmd_speed = 0.0
        self.cmd_steer = 0.0
        if not self.has_opp:
            self.has_opp = True
            self.get_logger().info(f'[opponent_vehicle] SPAWNED at ({self.x:.2f}, {self.y:.2f})')
        else:
            self.get_logger().info(f'[opponent_vehicle] moved to ({self.x:.2f}, {self.y:.2f})')

    def _remove_cb(self, _msg):
        if not self.has_opp:
            return
        self.has_opp = False
        self.x, self.y, self.th = self.PARK
        self.v = 0.0
        self.get_logger().info('[opponent_vehicle] REMOVED (parked off-map)')

    def _lidar_cb(self, msg):
        self.opp_scan_on = bool(msg.data)
        self.get_logger().info(
            f'[opponent_vehicle] opp self-lidar {"ON" if self.opp_scan_on else "OFF"}')

    # ---- main loop ----
    def _loop(self):
        if self.has_opp:
            # kinematic bicycle integration
            self.v = self.cmd_speed
            self.x += self.v * math.cos(self.th) * self.dt
            self.y += self.v * math.sin(self.th) * self.dt
            self.th += self.v / self.wheelbase * math.tan(self.cmd_steer) * self.dt

        stamp = self.get_clock().now().to_msg()
        self._publish_opponent(stamp)   # /vil/opponents (+viz) for the augmentor
        self._publish_odom_tf(stamp)
        self._publish_wheel_tf(stamp)   # front steering wheels (else they vanish in RViz)
        self._publish_opp_scan(stamp)   # opponent's own lidar (FTG)

    def _publish_wheel_tf(self, stamp):
        """Front wheels attach via a non-fixed (continuous) joint, so robot_state_
        publisher won't place them without a transform -> publish the steering TF
        (front_*_hinge -> front_*_wheel) ourselves, like gym_bridge did."""
        if not self.has_opp:
            return
        q = euler.euler2quat(0.0, 0.0, self.cmd_steer, axes='sxyz')
        for side in ('left', 'right'):
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = f'{self.opp_ns}/front_{side}_hinge'
            t.child_frame_id = f'{self.opp_ns}/front_{side}_wheel'
            t.transform.rotation.w = q[0]
            t.transform.rotation.x = q[1]
            t.transform.rotation.y = q[2]
            t.transform.rotation.z = q[3]
            self.br.sendTransform(t)

    def _publish_opponent(self, stamp):
        # opponent as an f110_msgs Obstacle: a small box at base_link (rear axle),
        # like a real detection box. Consumed by scan_augmentor (overlay) and the
        # obstacle_merger (object concat).
        arr = ObstacleArray()
        arr.header.stamp = stamp
        arr.header.frame_id = 'map'
        if self.has_opp:
            o = Obstacle()
            o.id = 1
            o.x_m = self.x
            o.y_m = self.y
            o.theta = self.th
            o.size = self.box_size
            o.vs = self.v
            o.is_static = False
            o.is_visible = True
            arr.obstacles.append(o)
        self.obs_pub.publish(arr)

        # RViz visualization of the detection box (red cube of box_size)
        viz = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = stamp
        m.ns = 'opp_box'
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD if self.has_opp else Marker.DELETE
        m.pose.position.x = self.x
        m.pose.position.y = self.y
        m.pose.position.z = 0.1
        q = euler.euler2quat(0.0, 0.0, self.th, axes='sxyz')
        m.pose.orientation.w, m.pose.orientation.x = q[0], q[1]
        m.pose.orientation.y, m.pose.orientation.z = q[2], q[3]
        m.scale.x = m.scale.y = self.box_size
        m.scale.z = 0.2
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.1, 0.85
        viz.markers.append(m)
        self.viz_pub.publish(viz)

    def _publish_odom_tf(self, stamp):
        if not self.has_opp:
            return
        q = euler.euler2quat(0.0, 0.0, self.th, axes='sxyz')
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'map'
        odom.child_frame_id = self.opp_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.w = q[0]
        odom.pose.pose.orientation.x = q[1]
        odom.pose.pose.orientation.y = q[2]
        odom.pose.pose.orientation.z = q[3]
        odom.twist.twist.linear.x = self.v
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = self.opp_frame
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.w = q[0]
        t.transform.rotation.x = q[1]
        t.transform.rotation.y = q[2]
        t.transform.rotation.z = q[3]
        self.br.sendTransform(t)

    def _publish_opp_scan(self, stamp):
        if self.eng is None or not self.has_opp or not self.opp_scan_on:
            return
        lx = self.x + self.scan_dist * math.cos(self.th)
        ly = self.y + self.scan_dist * math.sin(self.th)
        opp_boxes = None
        if self.ego_pose is not None:   # opponent sees the ego as a moving box
            opp_boxes = [[self.ego_pose[0], self.ego_pose[1], self.ego_pose[2]]]
        ranges = self.eng.scan_with_dynamics(
            np.array([lx, ly, self.th]), self.beams, self.fov,
            opp_poses=opp_boxes, obstacles=None, miss=None)
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.opp_laser_frame
        scan.angle_min = -self.fov / 2.0
        scan.angle_max = self.fov / 2.0
        scan.angle_increment = self.fov / (self.beams - 1)
        scan.range_min = 0.0
        scan.range_max = self.max_range
        scan.ranges = np.asarray(ranges, np.float32).tolist()
        self.scan_pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentVehicle()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
