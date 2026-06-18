#!/usr/bin/env python3
import time
from typing import List, Any, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    SetParametersResult,
    ParameterType,
)
from rclpy.parameter import Parameter

import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import BPoly
from scipy.signal import argrelextrema
from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray
from frenet_conversion.frenet_converter import FrenetConverter
from transforms3d.euler import quat2euler
from grid_filter.grid_filter import GridFilter
import trajectory_planning_helpers as tph


class ObstacleSpliner(Node):
    """
    This class implements a ROS node that performs splining around obstacles.

    It subscribes to the following topics:
        - `/tracking/obstacles`: Subscribes to the obstacle array.
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/global_waypoints`: Subscribes to global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.

    The node publishes the following topics:
        - `/planner/avoidance/markers`: Publishes spline markers.
        - `/planner/avoidance/otwpnts`: Publishes splined waypoints.
        - `/planner/avoidance/considered_OBS`: Publishes markers for the closest obstacle.
        - `/planner/avoidance/propagated_obs`: Publishes markers for the propagated obstacle.
        - `/planner/avoidance/latency`: Publishes the latency of the spliner node. (only if measuring is enabled)
    """

    def __init__(self):
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        # Initialize the node
        self.name = "recovery_spliner_node"
        super().__init__(self.name)

        # initialize the instance variable
        self.gb_wpnts = None
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = None
        self.cur_d = None
        self.cur_vs = None
        self.cur_x = None
        self.cur_y = None
        self.cur_yaw = None
        self.waypoints = None

        self.gb_scaled_wpnts = None
        self.inflection_points = None

        # static params
        self.declare_parameters(
            namespace='',
            parameters=[
                ('from_bag', False),
                ('measure', False),
                ('n_loc_wpnts', 80),
            ])
        self.from_bag = self.get_parameter(
            'from_bag').get_parameter_value().bool_value
        self.measuring = self.get_parameter(
            'measure').get_parameter_value().bool_value
        self.n_loc_wpnts = self.get_parameter(
            'n_loc_wpnts').get_parameter_value().integer_value

        # dynamic (tunable) params - folded from dynamic_reconfigure cfg
        self.save_params = False
        self.min_candidates_lookahead_n = 20
        self.num_kappas = 20
        self.spline_scale = 0.8
        self.kernel_size = 5
        self.smooth_len = 1.0

        int_lookahead_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            integer_range=[IntegerRange(from_value=10, to_value=100, step=1)]
        )
        int_num_kappas_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            integer_range=[IntegerRange(from_value=10, to_value=50, step=1)]
        )
        int_kernel_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            integer_range=[IntegerRange(from_value=1, to_value=20, step=1)]
        )
        double_spline_scale_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.5, to_value=2.0, step=0.001)]
        )
        double_smooth_len_pd = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.0, step=0.001)]
        )
        bool_pd = ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)

        param_dicts = [
            {'name': 'save_params', 'default': self.save_params, 'descriptor': bool_pd},
            {'name': 'min_candidates_lookahead_n', 'default': self.min_candidates_lookahead_n, 'descriptor': int_lookahead_pd},
            {'name': 'num_kappas', 'default': self.num_kappas, 'descriptor': int_num_kappas_pd},
            {'name': 'spline_scale', 'default': self.spline_scale, 'descriptor': double_spline_scale_pd},
            {'name': 'kernel_size', 'default': self.kernel_size, 'descriptor': int_kernel_pd},
            {'name': 'smooth_len', 'default': self.smooth_len, 'descriptor': double_smooth_len_pd},
        ]
        self.declare_all_parameters(param_dicts=param_dicts)

        # Subscribe to the topics
        self.create_subscription(
            Odometry, "/car_state/odom_frenet", self.state_frenet_cb, QoSProfile(depth=10))
        self.create_subscription(
            Odometry, "/car_state/odom", self.state_cb, QoSProfile(depth=10))
        self.create_subscription(
            WpntArray, "/global_waypoints", self.gb_cb, QoSProfile(depth=10))
        self.create_subscription(
            WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, QoSProfile(depth=10))

        self.mrks_pub = self.create_publisher(
            MarkerArray, "/planner/recovery/markers", QoSProfile(depth=10))
        self.recovery_wpnts_pub = self.create_publisher(
            WpntArray, "/planner/recovery/wpnts", QoSProfile(depth=10))
        self.recovery_lookahead_pub = self.create_publisher(
            Marker, "/planner/recovery/lookahead_point", QoSProfile(depth=10))

        if self.measuring:
            self.latency_pub = self.create_publisher(
                Float32, "/planner/recovery/latency", QoSProfile(depth=10))
            self.checkpoints_pub = self.create_publisher(
                MarkerArray, "/planner/recovery/checkpoints", QoSProfile(depth=10))

        # Register dynamic params callback after publishers exist (checkpoints may publish in gb_cb)
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        # Wait for critical messages
        self.wait_for_messages()

        self.converter = self.initialize_converter()
        self.map_filter = GridFilter(map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        # Main loop at 40 Hz
        self.create_timer(1.0 / 40.0, self.loop)
        self.get_logger().info(f"[{self.name}] Ready!")

    #################### DYNAMIC PARAMS ####################
    def declare_all_parameters(self, param_dicts: List[dict]):
        params = []
        for param_dict in param_dicts:
            param = self.declare_parameter(
                param_dict['name'], param_dict['default'], param_dict['descriptor'])
            params.append(param)
        return params

    #############
    # CALLBACKS #
    #############
    def state_frenet_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

    def state_cb(self, data: Odometry):
        self.cur_x = data.pose.pose.position.x
        self.cur_y = data.pose.pose.position.y
        quat = data.pose.pose.orientation
        # transforms3d expects (w, x, y, z)
        euler = quat2euler([quat.w, quat.x, quat.y, quat.z])
        self.cur_yaw = euler[2]

    # Callback for global waypoint topic
    def gb_cb(self, data: WpntArray):

        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m

        # psi_array = np.array([wpnt.psi_rad for wpnt in data.wpnts])
        kappas = np.array([wpnt.kappa_radpm for wpnt in data.wpnts])

        sign_changes = np.sign(kappas)

        self.inflection_points = np.where(np.diff(sign_changes) != 0)[0]

        # max_indices = argrelextrema(psi_array, np.greater)[0]
        # min_indices = argrelextrema(psi_array, np.less)[0]

        if self.measuring:
            mrks = MarkerArray()
            for idx in self.inflection_points:
                # print(idx)
                mrk = Marker()
                mrk.header.frame_id = "map"
                mrk.header.stamp = self.get_clock().now().to_msg()
                mrk.type = mrk.CYLINDER
                mrk.scale.x = 0.3
                mrk.scale.y = 0.3
                mrk.scale.z = 0.3
                mrk.color.a = 1.0
                mrk.color.b = 0.75
                mrk.color.r = 0.75
                mrk.id = int(idx)
                mrk.pose.position.x = data.wpnts[idx].x_m
                mrk.pose.position.y = data.wpnts[idx].y_m
                mrk.pose.position.z = 0.0
                mrk.pose.orientation.w = 1.0
                mrks.markers.append(mrk)
            self.checkpoints_pub.publish(mrks)

    # Callback for scaled global waypoint topic
    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: List[Parameter]):
        """
        Notices the change in the parameters and changes spline params
        """
        for param in params:
            param_name = param.name
            if param_name == 'save_params':
                self.save_params = param.value
            elif param_name == 'min_candidates_lookahead_n':
                self.min_candidates_lookahead_n = param.value
            elif param_name == 'num_kappas':
                self.num_kappas = param.value
            elif param_name == 'spline_scale':
                self.spline_scale = param.value
            elif param_name == 'kernel_size':
                self.kernel_size = param.value
            elif param_name == 'smooth_len':
                self.smooth_len = param.value

        if hasattr(self, 'map_filter'):
            self.map_filter.set_erosion_kernel_size(self.kernel_size)

        self.get_logger().info(
            f"[{self.name}] Dynamic reconf triggered new params: "
            f"min_candidates_lookahead_n: {self.min_candidates_lookahead_n}, "
            f"num_kappas: {self.num_kappas}, spline_scale: {self.spline_scale}, "
            f"kernel_size: {self.kernel_size}, smooth_len: {self.smooth_len}"
        )
        return SetParametersResult(successful=True)

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        if self.measuring:
            start = time.perf_counter()
        # Sample data
        gb_scaled_wpnts = self.gb_scaled_wpnts.wpnts
        wpnts = WpntArray()
        mrks = MarkerArray()

        del_mrk = Marker()
        del_mrk.header.stamp = self.get_clock().now().to_msg()
        del_mrk.action = Marker.DELETEALL
        mrks.markers.append(del_mrk)
        self.mrks_pub.publish(mrks)

        wpnts, mrks = self.do_spline(gb_wpnts=gb_scaled_wpnts)

        # Publish wpnts and markers
        if self.measuring:
            end = time.perf_counter()
            latency_msg = Float32()
            latency_msg.data = float(1 / (end - start))
            self.latency_pub.publish(latency_msg)
        self.recovery_wpnts_pub.publish(wpnts)
        self.mrks_pub.publish(mrks)

    #########
    # UTILS #
    #########
    def wait_for_messages(self):
        self.get_logger().info(f"[{self.name}] Waiting for messages and services...")
        waitlist = [self.gb_wpnts, self.gb_scaled_wpnts, self.cur_x, self.cur_s]
        while None in waitlist:
            rclpy.spin_once(self)
            waitlist = [self.gb_wpnts, self.gb_scaled_wpnts, self.cur_x, self.cur_s]
        self.get_logger().info(f"[{self.name}] All required messages received.")

    def initialize_converter(self) -> FrenetConverter:
        """
        Initialize the FrenetConverter object"""
        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")

        return converter

    def find_tangent_idx(self, xy_m, psi_rads):
        # Get current position
        cur_x, cur_y = self.cur_x, self.cur_y
        smooth = np.cos(self.cur_yaw), np.sin(self.cur_yaw) * self.smooth_len

        # Compute direction vectors from the current position to waypoints
        dx = xy_m[:, 0] - (cur_x + smooth[0])
        dy = xy_m[:, 1] - (cur_y + smooth[1])

        # Normalize vectors
        norm = np.sqrt(dx**2 + dy**2)  # Vector magnitude
        unit_vectors = np.vstack((dx / norm, dy / norm)).T  # (N, 2) unit vectors

        # Convert waypoint heading angles to unit vectors
        psi_unit_vectors = np.vstack((np.cos(psi_rads), np.sin(psi_rads))).T  # (N, 2)

        # Compute cosine similarity between vectors
        cos_theta = np.sum(unit_vectors * psi_unit_vectors, axis=1)  # Dot product
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevent numerical errors

        # Compute angular difference in radians
        angles = np.arccos(cos_theta)  # Range: 0 ~ π

        # Find the index with the smallest angle
        tangent_idx = np.argmin(angles)

        return tangent_idx

    def do_spline(self, gb_wpnts: WpntArray) -> Tuple[WpntArray, MarkerArray]:
        """
        Creates an evasion trajectory for the closest obstacle by splining between pre- and post-apex points.

        This function takes as input the obstacles to be evaded, and a list of global waypoints that describe a reference raceline.
        It only considers obstacles that are within a threshold of the raceline and generates an evasion trajectory for each of these obstacles.
        The evasion trajectory consists of a spline between pre- and post-apex points that are spaced apart from the obstacle.
        The spatial and velocity components of the spline are calculated using the `Spline` class, and the resulting waypoints and markers are returned.

        Args:
        - obstacles (ObstacleArray): An array of obstacle objects to be evaded.
        - gb_wpnts (WpntArray): A list of global waypoints that describe a reference raceline.
        - state (Odometry): The current state of the car.

        Returns:
        - wpnts (WpntArray): An array of waypoints that describe the evasion trajectory to the closest obstacle.
        - mrks (MarkerArray): An array of markers that represent the waypoints in a visualization format.

        """
        # Return wpnts and markers
        mrks = MarkerArray()
        wpnts = WpntArray()

        # Get spacing between wpnts for rough approximations
        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m

        cur_s = self.cur_s
        cur_d = self.cur_d
        cur_s_idx = int(cur_s / wpnt_dist)

        if len(self.inflection_points) != 0:
            infl_sector_idx = np.searchsorted(self.inflection_points, cur_s_idx)

            next_infl_sector_idx = self.inflection_points[(infl_sector_idx) % len(self.inflection_points)]

            candidate_len = next_infl_sector_idx - cur_s_idx + self.gb_max_idx if infl_sector_idx == len(self.inflection_points) \
                else next_infl_sector_idx - cur_s_idx
        else:
            candidate_len = int(self.gb_max_idx / 2)

        candidate_len = max(candidate_len, self.min_candidates_lookahead_n)

        gb_idxs = [(cur_s_idx + i) % self.gb_max_idx for i in range(candidate_len)]

        num_kappas_ = min(self.num_kappas, self.min_candidates_lookahead_n)
        kappas = np.array([gb_wpnts[gb_idx].kappa_radpm for gb_idx in gb_idxs[:num_kappas_]])

        outside = True if np.sum(kappas) * cur_d < 0 else False

        tangent_idx = 20
        # if outside: # TODO : need to consider outside and inside
        if True:
            xy_m = np.array([(gb_wpnts[gb_idx].x_m, gb_wpnts[gb_idx].y_m) for gb_idx in gb_idxs])
            psi_rads = np.array([gb_wpnts[gb_idx].psi_rad for gb_idx in gb_idxs])

            tangent_idx = self.find_tangent_idx(xy_m, psi_rads)

            if self.measuring:
                mrk = Marker()
                mrk.header.frame_id = "map"
                mrk.header.stamp = self.get_clock().now().to_msg()
                mrk.type = mrk.SPHERE
                mrk.scale.x = 0.5
                mrk.scale.y = 0.5
                mrk.scale.z = 0.5
                mrk.color.a = 1.0
                mrk.color.b = 1.0
                mrk.color.r = 0.0
                mrk.color.g = 0.65

                mrk.pose.position.x = float(xy_m[tangent_idx, 0])
                mrk.pose.position.y = float(xy_m[tangent_idx, 1])
                mrk.pose.position.z = 0.01
                mrk.pose.orientation.w = 1.0
                self.recovery_lookahead_pub.publish(mrk)

            if tangent_idx != 0:
                target_s = tangent_idx * wpnt_dist

        points = []
        tangents = []

        points.append([self.cur_x, self.cur_y])
        points.append([xy_m[tangent_idx, 0], xy_m[tangent_idx, 1]])

        tangents.append(np.array([np.cos(self.cur_yaw), np.sin(self.cur_yaw)]))
        tangents.append(np.array([np.cos(psi_rads[tangent_idx]), np.sin(psi_rads[tangent_idx])]))

        tangents = np.dot(tangents, self.spline_scale * np.eye(2))
        points = np.asarray(points)
        nPoints, dim = points.shape

        # Parametrization parameter s.
        dp = np.diff(points, axis=0)                 # difference between points
        dp = np.linalg.norm(dp, axis=1)              # distance between points
        d = np.cumsum(dp)                            # cumsum along the segments
        d = np.hstack([[0], d])                      # add distance from first point
        l = d[-1]                                    # length of point sequence
        nSamples = int(l / wpnt_dist)                # number of samples
        s, r = np.linspace(0, l, nSamples, retstep=True)  # sample parameter and step

        # Bring points and (optional) tangent information into correct format.
        assert(len(points) == len(tangents))
        spline_result = np.empty([nPoints, dim], dtype=object)
        for i, ref in enumerate(points):
            t = tangents[i]
            # Either tangent is None or has the same
            # number of dimensions as the point ref.
            assert(t is None or len(t) == dim)
            fuse = list(zip(ref, t) if t is not None else zip(ref,))
            spline_result[i, :] = fuse

        # Compute splines per dimension separately.
        samples = np.zeros([nSamples, dim])
        for i in range(dim):
            poly = BPoly.from_derivatives(d, spline_result[:, i])
            samples[:, i] = poly(s)

        # if samples.shape[0] < self.n_loc_wpnts:
        n_additional = 80
        xy_additional = np.array([
            (
                gb_wpnts[(tangent_idx + cur_s_idx + i + 1) % self.gb_max_idx].x_m,
                gb_wpnts[(tangent_idx + cur_s_idx + i + 1) % self.gb_max_idx].y_m
            )
            for i in range(n_additional)
        ])

        samples = np.vstack([samples, xy_additional])

        s_, d_ = self.converter.get_frenet(samples[:, 0], samples[:, 1])

        psi_, kappa_ = tph.calc_head_curv_num.\
            calc_head_curv_num(
                path=samples,
                el_lengths=0.1 * np.ones(len(samples) - 1),
                is_closed=False
            )

        danger_flag = False
        for i in range(samples.shape[0]):
            gb_wpnt_i = int((s_[i] / wpnt_dist) % self.gb_max_idx)

            inside = self.map_filter.is_point_inside(samples[i, 0], samples[i, 1])
            if not inside:
                self.get_logger().info(
                    f"[{self.name}]: Evasion trajectory too close to TRACKBOUNDS, aborting evasion",
                    throttle_duration_sec=2,
                )            # if abs(evasion_d[i]) > abs(tb_dist) - self.spline_bound_mindist:
                danger_flag = True
                break
            # Get V from gb wpnts and go slower if we are going through the inside
            vi = gb_wpnts[gb_wpnt_i].vx_mps if outside else gb_wpnts[gb_wpnt_i].vx_mps * 0.9  # TODO make speed scaling ros param
            wpnts.wpnts.append(
                self.xyv_to_wpnts(x=samples[i, 0], y=samples[i, 1], s=s_[i], d=d_[i], v=vi, psi=psi_[i] + np.pi / 2, kappa=kappa_[i], wpnts=wpnts)
            )
            mrks.markers.append(self.xyv_to_markers(x=samples[i, 0], y=samples[i, 1], v=vi, mrks=mrks))

        # Fill the rest of OTWpnts
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"

        if danger_flag:
            wpnts.wpnts = []
            mrks.markers = []

        return wpnts, mrks

    ######################
    # VIZ + MSG WRAPPING #
    ######################
    def xyv_to_markers(self, x: float, y: float, v: float, mrks: MarkerArray) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = float(v / self.gb_vmax)
        mrk.color.a = 1.0
        mrk.color.b = 0.75
        mrk.color.r = 0.75
        if self.from_bag:
            mrk.color.g = 0.75

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = float(v / self.gb_vmax / 2)
        mrk.pose.orientation.w = 1.0

        return mrk

    def xy_to_point(self, x: float, y: float, opponent=True) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.SPHERE
        mrk.scale.x = 0.5
        mrk.scale.y = 0.5
        mrk.scale.z = 0.5
        mrk.color.a = 0.8
        mrk.color.b = 0.65
        mrk.color.r = 1.0 if opponent else 0.0
        mrk.color.g = 0.65

        mrk.pose.position.x = float(x)
        mrk.pose.position.y = float(y)
        mrk.pose.position.z = 0.01
        mrk.pose.orientation.w = 1.0

        return mrk

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, psi: float, kappa: float, wpnts: WpntArray) -> Wpnt:
        wpnt = Wpnt()
        wpnt.id = len(wpnts.wpnts)
        wpnt.x_m = float(x)
        wpnt.y_m = float(y)
        wpnt.s_m = float(s)
        wpnt.d_m = float(d)
        wpnt.vx_mps = float(v)
        wpnt.psi_rad = float(psi)
        wpnt.kappa_radpm = float(kappa)

        return wpnt


def main(args=None):
    rclpy.init(args=args)
    spliner = ObstacleSpliner()
    try:
        rclpy.spin(spliner)
    except KeyboardInterrupt:
        pass
    spliner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
