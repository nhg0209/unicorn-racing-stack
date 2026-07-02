#!/usr/bin/env python3
import time
from typing import List, Any, Tuple
import copy

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    ParameterType,
    SetParametersResult,
)

import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import BPoly
from scipy.signal import argrelextrema, savgol_filter
from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray, BehaviorStrategy
from frenet_conversion.frenet_converter import FrenetConverter
from transforms3d.euler import quat2euler
from grid_filter.grid_filter import GridFilter
import trajectory_planning_helpers as tph

# --- Evasion path smoothing ---
SMOOTH_OTWPNTS = True
# Savitzky-Golay window for the kappa profile (odd, in waypoints; 0.1 m spacing -> 11 = ~1.1 m).
# Big enough to kill the point-to-point numeric-curvature noise, small enough to preserve the
# real ~2 m evasion bends (a 5 m window flattened them and over-smoothed the speed profile).
SMOOTH_OTWPNTS_WINDOW = 11
SMOOTH_OTWPNTS_POLYORDER = 2
GB_BLEND_LEN = 40                # waypoints over which to quad-ease back onto the GB line


class ObstacleSpliner(Node):
    """
    This class implements a ROS node that performs splining around static obstacles.

    It subscribes to the following topics:
        - `/behavior_strategy`: Subscribes to the behavior strategy (overtaking targets).
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/car_state/odom`: Subscribes to the car state in cartesian coordinates.
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
        self.name = "static_avoidance_planner"
        super().__init__('static_avoidance_planner')

        # initialize the instance variable
        self.obs_in_interest = None
        self._behavior_target = None  # obstacle from /behavior_strategy (primary while not in OVERTAKE)
        self.obstacles = []          # latest /tracking/obstacles (fallback source, like change_avoidance_node)
        # short obstacle memory: if /tracking/obstacles briefly drops the static obstacle (a 1-2 frame
        # detection gap), reuse the last passable candidate set for this window so the published spline
        # doesn't blank out and un-commit the OVERTAKE (the SM latest_threshold is only 0.1 s).
        self.obs_memory_sec = 0.3
        self._mem_cands_obs = []      # cached candidate Obstacle objects (absolute-s, frame-invariant)
        self._mem_cands_time = None   # rclpy Time of the last non-empty candidate set
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
        self.gb_scaled_wpnts = None
        self.waypoints = None
        self.lookahead = 10  # in meters [m]
        self.last_switch_time = self.get_clock().now().to_msg()
        self.last_ot_side = ""

        # Static parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('from_bag', False),
                ('measure', False),
            ])
        self.from_bag = self.get_parameter('from_bag').get_parameter_value().bool_value
        self.measuring = self.get_parameter('measure').get_parameter_value().bool_value

        # dyn params defaults
        self.save_params = False
        self.kernel_size = 3
        self.post_sampling_dist = 5.0
        self.sampling_dist = 5.0
        self.post_min_dist = 1.5
        self.post_max_dist = 5.0
        self.spline_scale = 0.8
        # cosine ease-in/out ramp lengths [m] for the s-monotonic avoidance profile
        self.back_to_raceline_before = 2.0   # start easing off the raceline this far before the obstacle
        self.back_to_raceline_after = 2.0    # ease back onto the raceline this far after it
        # Build the evasion path while the obstacle is this far ahead [m]. Kept ABOVE the state
        # machine's GB-track max_horizon (15 m, the overtake-intent trigger) so a fresh escape spline
        # is always available the moment the SM decides the raceline is blocked -> immediate OVERTAKE
        # commit instead of trailing/crawling up to the obstacle. Still capped at gb_max_s/2 to avoid
        # the path wrapping past half the lap.
        self.gen_horizon = 17.0
        self.tail_m = 6.0         # raceline tail after the ease-out so the path spans the SM horizon [m]
        self.evasion_dist = 0.35   # lateral gap [m] car near-edge <-> obstacle edge (main clearance knob)
        self.obs_traj_tresh = 0.3
        self.spline_bound_mindist = 0.2
        self.kd_obs_pred = 1.0
        self.fixed_pred_time = 0.15
        self.n_loc_wpnts = 80
        self.width_car = 0.30
        self.safety_margin = 0.1   # lateral safety margin for the evasion apex (matches change_avoidance_node)

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        self.declare_all_parameters()
        # Apply loaded params to working members at startup (callback only fires on later set).
        self.dyn_param_cb(self.get_parameters([
            'save_params', 'kernel_size', 'post_sampling_dist', 'post_min_dist',
            'post_max_dist', 'spline_scale', 'evasion_dist', 'obs_traj_tresh',
            'spline_bound_mindist', 'kd_obs_pred', 'fixed_pred_time',
        ]))
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        # Subscribe to the topics
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        # Direct obstacle source (like change_avoidance_node): the previous behavior_strategy-only
        # input starved this planner the moment the state machine entered OVERTAKE (it stops
        # publishing overtaking_targets there), which killed the very spline the car was following.
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacles_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)

        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers", 10)
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", 10)
        self.closest_obs_pub = self.create_publisher(Marker, "/planner/avoidance/considered_OBS", 10)
        self.pub_propagated = self.create_publisher(Marker, "/planner/avoidance/propagated_obs", 10)
        # Debug: per-sample spline bounds-check viz (green=pass, red=fail, blue=unchecked tail)
        self.spline_samples_pub = self.create_publisher(MarkerArray, "/planner/avoidance/spline_samples", 10)
        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/planner/avoidance/latency", 10)

        # Wait for critical messages
        self.wait_for_messages()

        self.converter = self.initialize_converter()

        # Set the rate at which the loop runs
        self.create_timer(1.0 / 20.0, self.loop)

    #####################
    # DYNAMIC PARAMETERS #
    #####################
    def declare_all_parameters(self):
        """
        Declare the dynamic-reconfigure tunables (from cfg/dyn_spliner_tuner.cfg) as ROS2
        parameters with proper descriptor ranges.
        """
        def dbl(min_v, max_v, desc=""):
            return ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description=desc,
                floating_point_range=[FloatingPointRange(from_value=float(min_v),
                                                         to_value=float(max_v),
                                                         step=0.001)],
            )

        def intd(min_v, max_v, desc=""):
            return ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description=desc,
                integer_range=[IntegerRange(from_value=int(min_v),
                                            to_value=int(max_v),
                                            step=1)],
            )

        self.declare_parameter('save_params', False,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Save params"))
        self.declare_parameter('kernel_size', 3, intd(1, 20))  # 8 eroded ~0.2m and rejected the raceline itself on the tight ifac track
        self.declare_parameter('post_sampling_dist', 5.0, dbl(0.5, 20.0))
        self.declare_parameter('post_min_dist', 1.5, dbl(0.5, 3.0))
        self.declare_parameter('post_max_dist', 5.0, dbl(3.0, 20.0))
        self.declare_parameter('spline_scale', 0.8, dbl(0.5, 2.0))
        self.declare_parameter('evasion_dist', 0.35, dbl(0.25, 1.25))
        self.declare_parameter('obs_traj_tresh', 1.0, dbl(0.1, 1.5))
        self.declare_parameter('spline_bound_mindist', 0.30, dbl(0.05, 1.0))
        self.declare_parameter('pre_apex_dist0', 4.0, dbl(0.5, 8.0))
        self.declare_parameter('pre_apex_dist1', 3.0, dbl(0.5, 8.0))
        self.declare_parameter('pre_apex_dist2', 2.0, dbl(0.5, 8.0))
        self.declare_parameter('post_apex_dist0', 4.5, dbl(0.5, 12.0))
        self.declare_parameter('post_apex_dist1', 5.0, dbl(0.5, 12.0))
        self.declare_parameter('post_apex_dist2', 5.5, dbl(0.5, 12.0))
        self.declare_parameter('kd_obs_pred', 1.0, dbl(0.1, 10.0))
        self.declare_parameter('fixed_pred_time', 0.15, dbl(0.0, 1.0))

    # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: List[Parameter]):
        """
        Notices the change in the parameters and changes spline params.
        """
        for param in params:
            if param.name == 'evasion_dist':
                self.evasion_dist = round(param.value * 20) / 20
            elif param.name == 'obs_traj_tresh':
                self.obs_traj_tresh = round(param.value * 20) / 20
            elif param.name == 'spline_bound_mindist':
                self.spline_bound_mindist = round(param.value * 20) / 20
            elif param.name == 'kd_obs_pred':
                self.kd_obs_pred = round(param.value * 20) / 20
            elif param.name == 'fixed_pred_time':
                self.fixed_pred_time = round(param.value * 100) / 100
            elif param.name == 'post_sampling_dist':
                self.sampling_dist = param.value
                self.post_sampling_dist = param.value
            elif param.name == 'spline_scale':
                self.spline_scale = param.value
            elif param.name == 'post_min_dist':
                self.post_min_dist = param.value
            elif param.name == 'post_max_dist':
                self.post_max_dist = param.value
            elif param.name == 'kernel_size':
                self.kernel_size = param.value
                self.map_filter.set_erosion_kernel_size(self.kernel_size)
            elif param.name == 'save_params':
                self.save_params = param.value

        self.get_logger().info(
            f"[{self.name}] evasion apex distance: {self.evasion_dist} [m],\n"
            f" obstacle trajectory treshold: {self.obs_traj_tresh} [m]\n"
            f" obstacle prediciton k_d: {self.kd_obs_pred},    obstacle prediciton constant time: {self.fixed_pred_time} [s] "
        )
        return SetParametersResult(successful=True)

    #############
    # CALLBACKS #
    #############
    def behavior_cb(self, data: BehaviorStrategy):
        # Primary target while the state machine still publishes it (GB_TRACK / TRAILING).
        # It goes empty once the SM enters OVERTAKE; the loop then falls back to selecting the
        # obstacle directly from /tracking/obstacles so the spline keeps regenerating.
        self._behavior_target = data.overtaking_targets[0] if len(data.overtaking_targets) != 0 else None

    def obstacles_cb(self, data: ObstacleArray):
        self.obstacles = data.obstacles

    def state_frenet_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

    def state_cb(self, data: Odometry):
        self.cur_x = data.pose.pose.position.x
        self.cur_y = data.pose.pose.position.y
        quat = data.pose.pose.orientation
        # transforms3d uses (w, x, y, z) quaternion ordering
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

    # Callback for scaled global waypoint topic
    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        if self.measuring:
            start = time.perf_counter()
        # do_spline now reads ALL static obstacles ahead directly and chains them into one path.
        wpnts, mrks = self.do_spline(gb_wpnts=self.gb_scaled_wpnts.wpnts)

        if self.measuring:
            end = time.perf_counter()
            self.latency_pub.publish(Float32(data=float(end - start)))
        self.evasion_pub.publish(wpnts)
        self.mrks_pub.publish(mrks)

    #########
    # UTILS #
    #########
    def wait_for_messages(self):
        self.get_logger().info(f"[{self.name}] Waiting for messages and services...")
        waitlist = [self.cur_s, self.cur_x, self.gb_wpnts, self.gb_scaled_wpnts]
        while None in waitlist:
            rclpy.spin_once(self)
            waitlist = [self.cur_s, self.cur_x, self.gb_wpnts, self.gb_scaled_wpnts]
        self.get_logger().info(f"[{self.name}] Ready!")

    def initialize_converter(self) -> FrenetConverter:
        """
        Initialize the FrenetConverter object"""
        waypoint_array = self.gb_wpnts.wpnts
        waypoints_x = np.array([wpnt.x_m for wpnt in waypoint_array])
        waypoints_y = np.array([wpnt.y_m for wpnt in waypoint_array])
        waypoints_psi = np.array([wpnt.psi_rad for wpnt in waypoint_array])
        converter = FrenetConverter(waypoints_x, waypoints_y, waypoints_psi)
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")
        return converter

    def _side_and_apex(self, obs: Obstacle, gb_wpnts, wpnt_dist: float) -> Tuple[bool, float]:
        """Per-obstacle go/no-go + apex offset. Returns (feasible, d_apex).
        Convention: +d = LEFT (gb_wp.d_left), -d = RIGHT (gb_wp.d_right); obs.d_left/d_right are
        the obstacle edges in the same signed frame."""
        obs_s_idx = int(obs.s_center / wpnt_dist) % self.gb_max_idx
        gb_wp = gb_wpnts[obs_s_idx]
        half_car = self.width_car / 2
        clearance = self.evasion_dist            # desired car-edge <-> obstacle-edge gap (dyn param)
        wall_margin = 0.15
        min_gap = half_car + self.spline_bound_mindist
        left_gap = gb_wp.d_left - obs.d_left      # obstacle left edge  -> left wall
        right_gap = gb_wp.d_right + obs.d_right   # obstacle right edge -> right wall
        left_feasible = left_gap >= min_gap
        right_feasible = right_gap >= min_gap
        if not (left_feasible or right_feasible):
            return False, 0.0
        d_apex_left = min(obs.d_left + half_car + clearance, gb_wp.d_left - wall_margin)      # +d
        d_apex_right = max(obs.d_right - half_car - clearance, -(gb_wp.d_right - wall_margin))  # -d
        # among the feasible sides take the one closest to the raceline (smaller |d_apex|)
        if left_feasible and right_feasible:
            d_apex = d_apex_left if abs(d_apex_left) <= abs(d_apex_right) else d_apex_right
        else:
            d_apex = d_apex_left if left_feasible else d_apex_right
        return True, d_apex

    def _gather_cands(self, obstacles, gen_h: float) -> List[Tuple[float, Obstacle]]:
        """Static/near-stationary obstacles ahead within [0.5, gen_h] as (gap, obs), unsorted."""
        cands = []
        for o in obstacles:
            if not (o.is_static or (abs(o.vs) < 0.5 and abs(o.vd) < 0.5)):
                continue
            gap = (o.s_center - self.cur_s) % self.gb_max_s
            if 0.5 <= gap <= gen_h:
                cands.append((gap, o))
        return cands

    def do_spline(self, gb_wpnts: WpntArray) -> Tuple[WpntArray, MarkerArray]:
        """
        Build ONE frame-stable evasion path that chains ALL static obstacles ahead within the
        generation horizon — or nothing, if the nearest one has no room to pass.

        Design:
        - GO/NO-GO per obstacle lives in `_side_and_apex`; if the nearest obstacle can't be passed
          we publish an empty path (state machine keeps TRAILING).
        - Each passable obstacle adds a cosine bump to a single d(s) profile anchored to the
          global-waypoint grid. Consecutive obstacles are therefore handled in ONE path (fixing
          "avoids #1 then drives the raceline into #2"); d returns to 0 in the clear stretches.
        - Grid-anchored s means d(s) is frame-invariant: as the car advances the window just slides,
          so the published path doesn't morph -> no controller jitter / wobble.
        """
        mrks = MarkerArray()
        wpnts = OTWpntArray()
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"
        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m

        def _empty():
            del_mrk = Marker()
            del_mrk.header.frame_id = "map"
            del_mrk.action = Marker.DELETEALL
            mrks.markers = [del_mrk]
            wpnts.wpnts = []
            return wpnts, mrks

        if self.cur_s is None or self.gb_max_s is None:
            return _empty()

        # --- gather ALL near-stationary obstacles ahead within the horizon (nearest first) ---
        gen_h = min(self.gen_horizon, self.gb_max_s / 2.0)
        cands = self._gather_cands(self.obstacles, gen_h)
        now = self.get_clock().now()
        if cands:
            # cache the raw obstacle objects (their absolute s is frame-invariant, so they stay
            # valid to re-gather next frame with the current cur_s until the car passes them).
            self._mem_cands_obs = [o for _, o in cands]
            self._mem_cands_time = now
        elif self._mem_cands_obs and self._mem_cands_time is not None and \
                (now - self._mem_cands_time).nanoseconds * 1e-9 < self.obs_memory_sec:
            # brief detection dropout -> reuse remembered obstacles so the spline stays put
            cands = self._gather_cands(self._mem_cands_obs, gen_h)
        if not cands:
            return _empty()
        cands.sort(key=lambda go: go[0])
        self.obs_in_interest = cands[0][1]   # nearest, kept for viz/debug

        # --- grid anchor (frame-stable) -----------------------------------------------------
        ramp_in = max(self.back_to_raceline_before, 1e-3)
        ramp_out = max(self.back_to_raceline_after, 1e-3)
        car_idx = int(self.cur_s / wpnt_dist) % self.gb_max_idx
        grid_start_s = gb_wpnts[car_idx].s_m

        # per-obstacle feasibility/side/apex; stop at the first obstacle we cannot pass so we never
        # plan a path through it (the state machine keeps TRAILING that one).
        bumps = []   # (pre, obs_half, d_apex) in grid-anchored absolute s
        for gap, o in cands:
            feasible, d_apex = self._side_and_apex(o, gb_wpnts, wpnt_dist)
            if not feasible:
                break
            obs_half = ((o.s_end - o.s_start) % self.gb_max_s) / 2.0
            pre = (o.s_center - grid_start_s) % self.gb_max_s
            bumps.append((pre, obs_half, d_apex))
        if not bumps:
            self.get_logger().info(
                f"[{self.name}]: nearest obstacle has no lateral gap -> no spline, TRAILING",
                throttle_duration_sec=1.0,
            )
            return _empty()

        # --- path span reaches past the LAST avoided obstacle -------------------------------
        last_pre, last_half, _ = bumps[-1]
        span = last_pre + last_half + ramp_out + self.tail_m
        n = max(int(span / wpnt_dist), 5)
        idxs = (car_idx + np.arange(n)) % self.gb_max_idx
        s_abs = grid_start_s + np.arange(n) * wpnt_dist

        # --- combined d-profile: each obstacle a cosine bump; on overlap the more-active (larger
        #     weight) bump wins, so obstacles chain and d returns to 0 between well-separated ones.
        d_arr = np.zeros(n)
        win_w = np.zeros(n)
        for pre, obs_half, d_apex in bumps:
            obs_start_abs = grid_start_s + pre - obs_half
            obs_end_abs = grid_start_s + pre + obs_half
            ease_in_start = obs_start_abs - ramp_in
            w = np.zeros(n)
            m_in = (s_abs >= ease_in_start) & (s_abs < obs_start_abs)
            m_hold = (s_abs >= obs_start_abs) & (s_abs <= obs_end_abs)
            m_out = (s_abs > obs_end_abs) & (s_abs <= obs_end_abs + ramp_out)
            w[m_in] = 0.5 * (1 - np.cos(np.pi * (s_abs[m_in] - ease_in_start) / ramp_in))
            w[m_hold] = 1.0
            w[m_out] = 0.5 * (1 + np.cos(np.pi * (s_abs[m_out] - obs_end_abs) / ramp_out))
            take = w > win_w
            d_arr[take] = (w * d_apex)[take]
            win_w[take] = w[take]

        # clamp every sample inside the local drivable corridor
        wall_margin = 0.15
        d_right_arr = np.array([gb_wpnts[j].d_right for j in idxs])
        d_left_arr = np.array([gb_wpnts[j].d_left for j in idxs])
        d_arr = np.clip(d_arr, -(d_right_arr - wall_margin), d_left_arr - wall_margin)

        s_mod = s_abs % self.gb_max_s
        resp = self.converter.get_cartesian(s_mod, d_arr)
        resp = resp.T if resp.ndim == 2 else resp
        xy = np.asarray(resp, dtype=float).reshape(-1, 2)

        # heading/curvature over the whole (continuous) path, then smooth kappa (the velocity
        # profile is kappa-driven and the controller uses kappa as a steering feed-forward).
        psi_, kappa_ = tph.calc_head_curv_num.calc_head_curv_num(
            path=xy, el_lengths=wpnt_dist * np.ones(len(xy) - 1), is_closed=False)
        if SMOOTH_OTWPNTS and kappa_.size > SMOOTH_OTWPNTS_POLYORDER + 2:
            win = min(SMOOTH_OTWPNTS_WINDOW, kappa_.size)
            if win % 2 == 0:
                win -= 1
            if win > SMOOTH_OTWPNTS_POLYORDER:
                kappa_ = savgol_filter(kappa_, win, SMOOTH_OTWPNTS_POLYORDER)

        for i in range(len(xy)):
            wpnts.wpnts.append(
                self.xyv_to_wpnts(x=xy[i, 0], y=xy[i, 1], s=s_mod[i], d=d_arr[i], v=2,
                                  psi=psi_[i] + np.pi / 2, kappa=kappa_[i], wpnts=wpnts)
            )

        del_mrk = Marker()
        del_mrk.header.frame_id = "map"
        del_mrk.action = Marker.DELETEALL
        mrks.markers = [del_mrk, self._spline_line_marker(wpnts)]
        return wpnts, mrks

    def _spline_line_marker(self, wpnts: OTWpntArray) -> Marker:
        """Single LINE_STRIP for the whole evasion path instead of one CYLINDER per
        sample. ~150 markers/frame at 20 Hz was the dominant RViz load; one polyline
        renders essentially for free and needs no DELETEALL churn (fixed id overwrites)."""
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.ns = "avoidance_spline"
        mrk.id = 0
        mrk.type = Marker.LINE_STRIP
        mrk.action = Marker.ADD
        mrk.scale.x = 0.08   # line width [m]
        mrk.color.a = 1.0
        mrk.color.b = 0.75
        mrk.color.r = 0.75
        if self.from_bag:
            mrk.color.g = 0.75
        mrk.pose.orientation.w = 1.0
        mrk.points = [Point(x=float(w.x_m), y=float(w.y_m), z=0.0) for w in wpnts.wpnts]
        return mrk

    def _publish_spline_samples_markers(self, samples: np.ndarray, bounds_check_results: List[bool]):
        """Debug viz: each spline sample as a sphere. green=passed bounds check,
        red=failed (the point that aborted evasion), blue=unchecked tail point."""
        markers = MarkerArray()
        del_mrk = Marker()
        del_mrk.header.frame_id = "map"
        del_mrk.action = Marker.DELETEALL
        markers.markers.append(del_mrk)
        for i in range(samples.shape[0]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "spline_samples"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(samples[i, 0])
            marker.pose.position.y = float(samples[i, 1])
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            if i < len(bounds_check_results):
                if bounds_check_results[i]:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 0.8  # green: passed
                else:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 1.0  # red: failed
            else:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.0, 1.0, 0.5  # blue: unchecked tail
            markers.markers.append(marker)
        self.spline_samples_pub.publish(markers)

    def _obs_filtering(self, obstacles: ObstacleArray) -> List[Obstacle]:
        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        obs_on_traj = [obs for obs in obstacles.obstacles if abs(obs.d_center) < self.obs_traj_tresh]

        # Only use obstacles that within self.lookahead in front of the car
        close_obs = []
        for obs in obs_on_traj:
            obs = self._predict_obs_movement(obs)
            # Handle wraparound
            dist_in_front = (obs.s_center - self.cur_s) % self.gb_max_s
            # dist_in_back = abs(dist_in_front % (-self.gb_max_s)) # distance from ego to obstacle in the back
            if dist_in_front < self.lookahead:
                close_obs.append(obs)
                # Not within lookahead
            else:
                pass
        return close_obs

    def _predict_obs_movement(self, obs: Obstacle, mode: str = "constant") -> Obstacle:
        """
        Predicts the movement of an obstacle based on the current state and mode.

        TODO: opponent prediction should be completely isolated for added modularity

        Args:
            obs (Obstacle): The obstacle to predict the movement for.
            mode (str, optional): The mode for predicting the movement. Defaults to "constant".

        Returns:
            Obstacle: The updated obstacle with the predicted movement.
        """
        # propagate opponent by time dependent on distance
        if (obs.s_center - self.cur_s) % self.gb_max_s < 10:  # TODO make param
            if mode == "adaptive":
                # distance in s coordinate
                cur_s = self.cur_s
                ot_distance = (obs.s_center - cur_s) % self.gb_max_s
                rel_speed = np.clip(self.gb_scaled_wpnts.wpnts[int(cur_s * 10)].vx_mps - obs.vs, 0.1, 10)
                ot_time_distance = np.clip(ot_distance / rel_speed, 0, 5) * 0.5

                delta_s = ot_time_distance * obs.vs
                delta_d = ot_time_distance * obs.vd
                delta_d = -(obs.d_center + delta_d) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            elif mode == "adaptive_velheuristic":
                opponent_scaler = 0.7
                cur_s = self.cur_s
                ego_speed = self.gb_scaled_wpnts.wpnts[int(cur_s * 10)].vx_mps

                # distance in s coordinate
                ot_distance = (obs.s_center - cur_s) % self.gb_max_s
                rel_speed = (1 - opponent_scaler) * ego_speed
                ot_time_distance = np.clip(ot_distance / rel_speed, 0, 5)

                delta_s = ot_time_distance * opponent_scaler * ego_speed
                delta_d = -(obs.d_center) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            # propagate opponent by constant time
            elif mode == "constant":
                delta_s = self.fixed_pred_time * obs.vs
                delta_d = self.fixed_pred_time * obs.vd
                # delta_d = -(obs.d_center+delta_d)*np.exp(-np.abs(self.kd_obs_pred*obs.d_center))

            elif mode == "heuristic":
                # distance in s coordinate
                ot_distance = (obs.s_center - self.cur_s) % self.gb_max_s
                rel_speed = 3
                ot_time_distance = ot_distance / rel_speed

                delta_d = ot_time_distance * obs.vd
                delta_d = -(obs.d_center + delta_d) * np.exp(-np.abs(self.kd_obs_pred * obs.d_center))

            # update
            obs.s_start += delta_s
            obs.s_center += delta_s
            obs.s_end += delta_s
            obs.s_start %= self.gb_max_s
            obs.s_center %= self.gb_max_s
            obs.s_end %= self.gb_max_s

            obs.d_left += delta_d
            obs.d_center += delta_d
            obs.d_right += delta_d

            resp = self.converter.get_cartesian([obs.s_center], [obs.d_center])

            marker = self.xy_to_point(resp[0], resp[1], opponent=True)
            self.pub_propagated.publish(marker)

        return obs

    def _check_ot_side_possible(self, more_space) -> bool:
        if abs(self.cur_d) > 0.25 and more_space != self.last_ot_side:  # TODO make rosparam for cur_d threshold
            self.get_logger().info(f"[{self.name}]: Can't switch sides, because we are not on the raceline")
            return False
        return True

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
