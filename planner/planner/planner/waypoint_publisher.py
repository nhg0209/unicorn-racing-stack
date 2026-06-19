#!/usr/bin/env python3
"""
Waypoint Publisher Node for F1TENTH

Reads CSV files from the map directory and publishes RViz markers + WpntArray.
Publishes only if the CSV file exists, silently skips otherwise.

Published topics (markers):
    /centerline_waypoints/markers - Blue line strip (from centerline.csv)
    /track_bounds/markers         - Green/Yellow boundaries (from boundary_right/left.csv)
    /global_waypoints/markers     - Red line strip (from global_waypoints.csv)

Published topics (waypoints):
    /centerline_waypoints/wpnts   - WpntArray (from centerline.csv)
    /global_waypoints             - WpntArray (from global_waypoints.csv)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from f110_msgs.msg import Wpnt, WpntArray

import numpy as np
import csv
import os

from planner.track_bounds import TrackBounds


class WaypointPublisher(Node):

    def __init__(self):
        super().__init__('waypoint_publisher')

        self.declare_parameter('map_name', '')
        self.map_name = self.get_parameter('map_name').value

        if not self.map_name:
            self.get_logger().error('[WaypointPublisher] map_name parameter is required!')
            return

        # Resolve source map directory from __file__ (realpath resolves symlinks)
        # __file__: .../creating_autonomous_car/planner/planner/waypoint_publisher.py
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.map_dir = os.path.join(pkg_root, 'stack_master', 'maps', self.map_name)
        self.get_logger().info(f'[WaypointPublisher] map_dir: {self.map_dir}')

        # Load track boundaries for d_right/d_left computation
        self.track_bounds = TrackBounds(self.map_dir)
        if self.track_bounds.is_valid():
            self.get_logger().info('[WaypointPublisher] Track boundaries loaded')
        else:
            self.get_logger().info('[WaypointPublisher] No boundary CSVs found, d_right/d_left will use CSV values or 0')

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Marker publishers
        self.centerline_marker_pub = self.create_publisher(
            MarkerArray, '/centerline_waypoints/markers', latched_qos)
        self.track_bounds_marker_pub = self.create_publisher(
            MarkerArray, '/track_bounds/markers', latched_qos)
        self.global_wp_marker_pub = self.create_publisher(
            MarkerArray, '/global_waypoints/markers', latched_qos)

        # WpntArray publishers
        self.centerline_wpnt_pub = self.create_publisher(
            WpntArray, '/centerline_waypoints', latched_qos)
        self.global_wp_wpnt_pub = self.create_publisher(
            WpntArray, '/global_waypoints', latched_qos)

        # Load and publish
        self._centerline_marker = None
        self._track_bounds_marker = None
        self._global_wp_marker = None
        self._centerline_wpnt = None
        self._global_wp_wpnt = None

        self._load_centerline()
        self._load_track_bounds()
        self._load_global_waypoints()

        # Republish at 1Hz for late subscribers
        self.timer = self.create_timer(1.0, self._republish)

        self.get_logger().info(f'[WaypointPublisher] Initialized for map: {self.map_name}')

    def _load_csv(self, filename):
        """Load CSV file, return list of dicts or None if not found."""
        csv_path = os.path.join(self.map_dir, filename)
        if not os.path.exists(csv_path):
            self.get_logger().info(f'[WaypointPublisher] {filename} not found, skipping')
            return None

        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if len(rows) == 0:
            return None

        self.get_logger().info(f'[WaypointPublisher] Loaded {filename}: {len(rows)} points')
        return rows

    def _csv_to_wpnt_array(self, data):
        """Convert CSV data to WpntArray message.

        Standard CSV column format (all waypoint CSVs should follow this):
            col 1: x_m          — x coordinate [m]
            col 2: y_m          — y coordinate [m]
            col 3: w_tr_right_m — right track width [m] (can be empty)
            col 4: w_tr_left_m  — left track width [m] (can be empty)
            col 5: psi_rad      — heading [rad]
            col 6: kappa_radpm  — curvature [rad/m]
            col 7: vx_mps       — velocity [m/s]

        Wpnt field sources:
            x_m, y_m   : CSV col 1-2
            d_right/left: TrackBounds (boundary CSVs), fallback → CSV col 3-4
            psi_rad    : CSV col 5, fallback → atan2 computed from x,y
            kappa_radpm: CSV col 6, fallback → 0.0
            vx_mps     : CSV col 7, fallback → 0.0
            ax_mps2    : 0.0 (trajectory optimizer will fill)
            TODO: s_m  — proper arc-length (frenet frame)
            TODO: d_m  — lateral offset from centerline (frenet frame)
        """
        cols = data[0].keys()

        x = np.array([float(r['x_m']) for r in data])
        y = np.array([float(r['y_m']) for r in data])

        # TODO: s_m — 현재는 단순 누적 유클리드 거리, frenet frame 구현 시 교체
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        s = np.cumsum(np.sqrt(dx**2 + dy**2))

        # col 5: psi_rad (heading)
        if 'psi_rad' in cols:
            psi = np.array([float(r['psi_rad']) for r in data])
        else:
            psi = np.arctan2(np.diff(y, append=y[0]), np.diff(x, append=x[0]))

        # col 6: kappa_radpm (curvature)
        if 'kappa_radpm' in cols:
            kappa = np.array([float(r['kappa_radpm']) for r in data])
        else:
            kappa = np.zeros(len(x))

        # col 7: vx_mps (velocity)
        if 'vx_mps' in cols:
            vx = np.array([float(r['vx_mps']) for r in data])
        else:
            vx = np.zeros(len(x))

        # d_right/d_left: TrackBounds first, CSV fallback
        if self.track_bounds.is_valid():
            d_right, d_left = self.track_bounds.compute_distances(x, y)
        elif 'w_tr_right_m' in cols:
            d_right = np.array([float(r['w_tr_right_m']) for r in data])
            d_left = np.array([float(r['w_tr_left_m']) for r in data])
        elif 'd_right' in cols:
            d_right = np.array([float(r['d_right']) for r in data])
            d_left = np.array([float(r['d_left']) for r in data])
        else:
            d_right = np.zeros(len(x))
            d_left = np.zeros(len(x))

        # Build WpntArray
        msg = WpntArray()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for i in range(len(x)):
            w = Wpnt()
            w.id = i
            w.x_m = float(x[i])
            w.y_m = float(y[i])
            w.s_m = float(s[i])
            w.d_m = 0.0  # TODO: frenet frame lateral offset
            w.psi_rad = float(psi[i])
            w.kappa_radpm = float(kappa[i])
            w.vx_mps = float(vx[i])
            w.ax_mps2 = 0.0
            w.d_right = float(d_right[i])
            w.d_left = float(d_left[i])
            msg.wpnts.append(w)

        return msg

    def _load_centerline(self):
        """Load centerline.csv → publish markers + WpntArray."""
        data = self._load_csv('centerline.csv')
        if data is None:
            return

        x = np.array([float(r['x_m']) for r in data])
        y = np.array([float(r['y_m']) for r in data])

        # Markers (blue line strip)
        self._centerline_marker = self._create_line_strip_markers(
            x, y, ns='centerline_waypoints', r=0.0, g=0.0, b=1.0, width=0.03)
        self.centerline_marker_pub.publish(self._centerline_marker)

        # WpntArray
        self._centerline_wpnt = self._csv_to_wpnt_array(data)
        self.centerline_wpnt_pub.publish(self._centerline_wpnt)

    def _load_track_bounds(self):
        """Load boundary_right.csv + boundary_left.csv → publish markers."""
        right_data = self._load_csv('boundary_right.csv')
        left_data = self._load_csv('boundary_left.csv')
        if right_data is None or left_data is None:
            return

        right_x = np.array([float(r['x_m']) for r in right_data])
        right_y = np.array([float(r['y_m']) for r in right_data])
        left_x = np.array([float(r['x_m']) for r in left_data])
        left_y = np.array([float(r['y_m']) for r in left_data])

        tb_msg = MarkerArray()
        tb_msg.markers.append(self._create_line_strip(
            right_x, right_y, ns='right_bound', marker_id=0,
            r=0.0, g=1.0, b=0.0, a=0.6, width=0.02))
        tb_msg.markers.append(self._create_line_strip(
            left_x, left_y, ns='left_bound', marker_id=1,
            r=1.0, g=1.0, b=0.0, a=0.6, width=0.02))
        self.track_bounds_marker_pub.publish(tb_msg)
        self._track_bounds_marker = tb_msg

    def _load_global_waypoints(self):
        """Load global_waypoints.csv → publish markers + WpntArray."""
        data = self._load_csv('global_waypoints.csv')
        if data is None:
            return

        x = np.array([float(r['x_m']) for r in data])
        y = np.array([float(r['y_m']) for r in data])

        # Markers (red line strip)
        self._global_wp_marker = self._create_line_strip_markers(
            x, y, ns='global_waypoints', r=1.0, g=0.0, b=0.0, width=0.03)
        self.global_wp_marker_pub.publish(self._global_wp_marker)

        # WpntArray
        self._global_wp_wpnt = self._csv_to_wpnt_array(data)
        self.global_wp_wpnt_pub.publish(self._global_wp_wpnt)

    def _create_line_strip(self, x, y, ns, marker_id, r, g, b, a=1.0, width=0.03):
        """Create a single LINE_STRIP marker."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a

        n = len(x)
        step = max(1, n // 2000)
        for i in range(0, n, step):
            p = Point()
            p.x = float(x[i])
            p.y = float(y[i])
            marker.points.append(p)

        # Close the loop
        p = Point()
        p.x = float(x[0])
        p.y = float(y[0])
        marker.points.append(p)

        return marker

    def _create_line_strip_markers(self, x, y, ns, r, g, b, width=0.03):
        """Create MarkerArray with a single LINE_STRIP."""
        msg = MarkerArray()
        msg.markers.append(self._create_line_strip(x, y, ns, 0, r, g, b, 1.0, width))
        return msg

    def _republish(self):
        """Republish for late subscribers."""
        if self._centerline_marker:
            self.centerline_marker_pub.publish(self._centerline_marker)
        if self._track_bounds_marker:
            self.track_bounds_marker_pub.publish(self._track_bounds_marker)
        if self._global_wp_marker:
            self.global_wp_marker_pub.publish(self._global_wp_marker)
        if self._centerline_wpnt:
            self._centerline_wpnt.header.stamp = self.get_clock().now().to_msg()
            self.centerline_wpnt_pub.publish(self._centerline_wpnt)
        if self._global_wp_wpnt:
            self._global_wp_wpnt.header.stamp = self.get_clock().now().to_msg()
            self.global_wp_wpnt_pub.publish(self._global_wp_wpnt)


    def cleanup_markers(self):
        """Publish DELETEALL markers to clear DDS cache before shutdown."""
        delete_msg = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        delete_msg.markers.append(m)

        self.centerline_marker_pub.publish(delete_msg)
        self.track_bounds_marker_pub.publish(delete_msg)
        self.global_wp_marker_pub.publish(delete_msg)
        self.get_logger().info('[WaypointPublisher] Markers cleaned up (DELETEALL)')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cleanup_markers()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
