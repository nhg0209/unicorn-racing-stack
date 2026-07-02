#!/usr/bin/env python3
import time
from typing import List, Tuple

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
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.interpolate import BPoly
from scipy.signal import savgol_filter
from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray, BehaviorStrategy
from frenet_conversion.frenet_converter import FrenetConverter
from transforms3d.euler import quat2euler
from grid_filter.grid_filter import GridFilter
import trajectory_planning_helpers as tph

# --- Evasion path kappa smoothing ---
SMOOTH_OTWPNTS = True
# Savitzky-Golay window for the kappa profile (odd, in waypoints; 0.1 m spacing -> 11 = ~1.1 m).
# Big enough to kill the point-to-point numeric-curvature noise, small enough to preserve the
# real ~2 m evasion bends.
SMOOTH_OTWPNTS_WINDOW = 11
SMOOTH_OTWPNTS_POLYORDER = 2


class ObstacleSpliner(Node):
    """
    Frenet grid-sampling static-obstacle avoidance planner (node ``static_avoidance_planner``).

    Each cycle it samples N terminal lateral offsets across the drivable width at a
    speed-proportional lookahead, builds a quintic d(s) to each, rejects candidates that leave the
    corridor / hit the eroded map / collide with any obstacle box / exceed a curvature limit, and
    picks the minimum-cost survivor.

    Subscribes:
        - ``/behavior_strategy``            (BehaviorStrategy) target hint (not required)
        - ``/tracking/obstacles``           (ObstacleArray)    ALL obstacles for collision checks
        - ``/car_state/odom_frenet``        (Odometry)         cur_s, cur_d, cur_vs
        - ``/car_state/odom``               (Odometry)         cur_x, cur_y, cur_yaw
        - ``/global_waypoints``             (WpntArray)        geometry + FrenetConverter seed
        - ``/global_waypoints_scaled``      (WpntArray)        velocity + d_left/d_right corridor

    Publishes:
        - ``/planner/avoidance/otwpnts``    (OTWpntArray)      selected evasion path (may be empty)
        - ``/planner/avoidance/feasible``   (Bool)             False if 0 feasible candidates
        - ``/planner/avoidance/markers``    (MarkerArray)      grey=all, red=rejected, green=selected
        - ``/planner/avoidance/latency``    (Float32)          loop time (only if ``measure``)
    """

    def __init__(self):
        self.name = "static_avoidance_planner"
        super().__init__('static_avoidance_planner')

        # --- state ---
        self.obs_in_interest = None
        self._behavior_target = None
        self.obstacles = []          # latest /tracking/obstacles
        # short obstacle memory: if /tracking/obstacles briefly drops the static obstacle (a 1-2
        # frame gap) reuse the last set for this window so the published path doesn't blink out and
        # un-commit the OVERTAKE (the SM freshness gate is tight).
        self.obs_memory_sec = 0.3
        self._mem_cands_obs = []
        self._mem_cands_time = None
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
        self._d_end_prev = 0.0       # last selected terminal offset (chatter damping)
        self._last_feasible = False

        # --- sampling-planner param defaults (all overridable via ROS params / config yaml) ---
        self.kernel_size = 3         # GridFilter erosion (cells); 8 ate ~0.2 m and rejected the raceline
        self.lookahead_min = 8.0     # [m]
        self.lookahead_k = 1.5       # [s]  lookahead = max(lookahead_min, k * cur_vs)
        self.n_d_samples = 13        # terminal offsets sampled across the width
        self.kappa_max = 2.0         # [1/m] curvature reject limit (min turn radius)
        self.a_lat_max = 6.0         # [m/s^2] lateral-accel cap for the velocity profile
        self.safety_margin = 0.05    # [m] extra clearance around the obstacle box (beyond half car)
        self.wall_margin = 0.05      # [m] clearance to the wall the candidate may reach (corridor)
        self.shift_min = 1.0         # [m] min arc length over which the lateral maneuver completes
        self.shift_buffer = 0.5      # [m] finish the shift this far before the obstacle near-edge
        self.ramp_len = 4.0          # [m] length of the ramp onto the offset (stay on raceline before it)
        self.hold_after = 0.5        # [m] hold the offset this far past the obstacle far-edge
        self.return_len = 2.5        # [m] length of the ramp back to the raceline after the obstacle
        self.width_car = 0.30        # [m]
        self.tail_m = 1.0            # [m] short raceline (d=0) tail after the return
        self.w_d = 1.0               # cost: raceline deviation
        self.w_k = 0.1               # cost: curvature (smoothness)
        self.w_c = 5.0               # cost: consistency with previous choice
        self.w_obs = 2.0             # cost: soft obstacle proximity
        self.obs_sigma = 0.5         # [m] soft-penalty length scale
        self.use_grid_check = True   # reject candidates crossing the eroded map

        # Static params
        self.declare_parameters(namespace='', parameters=[('from_bag', False), ('measure', False)])
        self.from_bag = self.get_parameter('from_bag').get_parameter_value().bool_value
        self.measuring = self.get_parameter('measure').get_parameter_value().bool_value

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        self.declare_all_parameters()
        # Sync members from loaded params (yaml/defaults), then register live-reconfigure callback.
        self.dyn_param_cb(self.get_parameters([
            'kernel_size', 'lookahead_min', 'lookahead_k', 'n_d_samples', 'kappa_max', 'a_lat_max',
            'safety_margin', 'wall_margin', 'shift_min', 'shift_buffer', 'ramp_len', 'hold_after',
            'return_len', 'width_car', 'tail_m', 'w_d', 'w_k', 'w_c', 'w_obs', 'obs_sigma',
            'use_grid_check',
        ]))
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        # Subscribers
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacles_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)

        # Publishers (topic names unchanged; /planner/avoidance/feasible is the only new topic)
        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers", 10)
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", 10)
        self.feasible_pub = self.create_publisher(Bool, "/planner/avoidance/feasible", 10)
        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/planner/avoidance/latency", 10)

        self.wait_for_messages()
        self.converter = self.initialize_converter()
        self.create_timer(1.0 / 20.0, self.loop)   # 20 Hz

    #####################
    # DYNAMIC PARAMETERS #
    #####################
    def declare_all_parameters(self):
        def dbl(min_v, max_v, desc=""):
            return ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE, description=desc,
                floating_point_range=[FloatingPointRange(from_value=float(min_v),
                                                         to_value=float(max_v), step=0.001)])

        def intd(min_v, max_v, desc=""):
            return ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER, description=desc,
                integer_range=[IntegerRange(from_value=int(min_v), to_value=int(max_v), step=1)])

        self.declare_parameter('kernel_size', 3, intd(1, 20, "GridFilter erosion kernel [cells]"))
        self.declare_parameter('lookahead_min', 8.0, dbl(1.0, 20.0, "min planning lookahead [m]"))
        self.declare_parameter('lookahead_k', 1.5, dbl(0.0, 5.0, "lookahead = max(min, k*cur_vs) [s]"))
        self.declare_parameter('n_d_samples', 13, intd(3, 41, "terminal lateral offsets sampled"))
        self.declare_parameter('kappa_max', 2.0, dbl(0.1, 10.0, "curvature reject limit [1/m]"))
        self.declare_parameter('a_lat_max', 6.0, dbl(1.0, 20.0, "lateral-accel cap [m/s^2]"))
        self.declare_parameter('safety_margin', 0.05, dbl(0.0, 1.0, "clearance around obstacle box [m]"))
        self.declare_parameter('wall_margin', 0.05, dbl(0.0, 1.0, "clearance to wall a candidate may reach [m]"))
        self.declare_parameter('shift_min', 1.0, dbl(0.3, 10.0, "min arc length for the lateral maneuver [m]"))
        self.declare_parameter('shift_buffer', 0.5, dbl(0.0, 5.0, "finish the shift this far before the obstacle [m]"))
        self.declare_parameter('ramp_len', 4.0, dbl(0.5, 15.0, "ramp length onto the offset [m]"))
        self.declare_parameter('hold_after', 0.5, dbl(0.0, 5.0, "hold the offset past the obstacle far-edge [m]"))
        self.declare_parameter('return_len', 2.5, dbl(0.5, 10.0, "ramp length back to the raceline [m]"))
        self.declare_parameter('width_car', 0.30, dbl(0.1, 1.0, "car width [m]"))
        self.declare_parameter('tail_m', 1.0, dbl(0.0, 20.0, "short raceline tail after the return [m]"))
        self.declare_parameter('w_d', 1.0, dbl(0.0, 100.0, "cost weight: raceline deviation"))
        self.declare_parameter('w_k', 0.1, dbl(0.0, 100.0, "cost weight: curvature"))
        self.declare_parameter('w_c', 5.0, dbl(0.0, 100.0, "cost weight: choice consistency"))
        self.declare_parameter('w_obs', 2.0, dbl(0.0, 100.0, "cost weight: obstacle proximity"))
        self.declare_parameter('obs_sigma', 0.5, dbl(0.05, 5.0, "soft-penalty length scale [m]"))
        self.declare_parameter('use_grid_check', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Reject candidates crossing eroded map"))

    def dyn_param_cb(self, params: List[Parameter]):
        for p in params:
            n = p.name
            if n == 'kernel_size':
                self.kernel_size = int(p.value)
                self.map_filter.set_erosion_kernel_size(self.kernel_size)
            elif n == 'lookahead_min':
                self.lookahead_min = float(p.value)
            elif n == 'lookahead_k':
                self.lookahead_k = float(p.value)
            elif n == 'n_d_samples':
                self.n_d_samples = int(p.value)
            elif n == 'kappa_max':
                self.kappa_max = float(p.value)
            elif n == 'a_lat_max':
                self.a_lat_max = float(p.value)
            elif n == 'safety_margin':
                self.safety_margin = float(p.value)
            elif n == 'wall_margin':
                self.wall_margin = float(p.value)
            elif n == 'shift_min':
                self.shift_min = float(p.value)
            elif n == 'shift_buffer':
                self.shift_buffer = float(p.value)
            elif n == 'ramp_len':
                self.ramp_len = float(p.value)
            elif n == 'hold_after':
                self.hold_after = float(p.value)
            elif n == 'return_len':
                self.return_len = float(p.value)
            elif n == 'width_car':
                self.width_car = float(p.value)
            elif n == 'tail_m':
                self.tail_m = float(p.value)
            elif n == 'w_d':
                self.w_d = float(p.value)
            elif n == 'w_k':
                self.w_k = float(p.value)
            elif n == 'w_c':
                self.w_c = float(p.value)
            elif n == 'w_obs':
                self.w_obs = float(p.value)
            elif n == 'obs_sigma':
                self.obs_sigma = float(p.value)
            elif n == 'use_grid_check':
                self.use_grid_check = bool(p.value)
        return SetParametersResult(successful=True)

    #############
    # CALLBACKS #
    #############
    def behavior_cb(self, data: BehaviorStrategy):
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
        euler = quat2euler([quat.w, quat.x, quat.y, quat.z])  # transforms3d: (w, x, y, z)
        self.cur_yaw = euler[2]

    def gb_cb(self, data: WpntArray):
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m

    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        if self.measuring:
            start = time.perf_counter()
        wpnts, mrks = self.do_spline(gb_wpnts=self.gb_scaled_wpnts.wpnts)
        if self.measuring:
            self.latency_pub.publish(Float32(data=float(time.perf_counter() - start)))
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
        waypoint_array = self.gb_wpnts.wpnts
        waypoints_x = np.array([wpnt.x_m for wpnt in waypoint_array])
        waypoints_y = np.array([wpnt.y_m for wpnt in waypoint_array])
        waypoints_psi = np.array([wpnt.psi_rad for wpnt in waypoint_array])
        converter = FrenetConverter(waypoints_x, waypoints_y, waypoints_psi)
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")
        return converter

    def _gather_obstacles_ahead(self, obstacles, lookahead: float) -> List[Tuple[float, Obstacle]]:
        """Static / near-stationary obstacles ahead within [0, lookahead], as (gap, obs), sorted."""
        cands = []
        for o in obstacles:
            if not (o.is_static or (abs(o.vs) < 0.5 and abs(o.vd) < 0.5)):
                continue
            gap = (o.s_center - self.cur_s) % self.gb_max_s
            if gap <= lookahead:
                cands.append((gap, o))
        cands.sort(key=lambda go: go[0])
        return cands

    def do_spline(self, gb_wpnts) -> Tuple[OTWpntArray, MarkerArray]:
        wpnts = OTWpntArray()
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"

        def _empty():
            self._publish_feasible(False)
            del_mrk = Marker()
            del_mrk.header.frame_id = "map"
            del_mrk.action = Marker.DELETEALL
            m = MarkerArray()
            m.markers = [del_mrk]
            wpnts.wpnts = []
            return wpnts, m

        if self.cur_s is None or self.gb_max_s is None or self.cur_d is None:
            return _empty()

        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m
        half_car = self.width_car / 2.0
        obs_margin = half_car + self.safety_margin      # keep-out half-width around obstacle boxes
        sample_margin = half_car + self.wall_margin     # how close to the wall a candidate may reach

        # --- speed-proportional lookahead (capped at half the lap) ---
        cur_vs = self.cur_vs if self.cur_vs is not None else 0.0
        lookahead = max(self.lookahead_min, self.lookahead_k * cur_vs)
        lookahead = min(lookahead, self.gb_max_s / 2.0)

        # --- obstacles ahead (with brief-dropout memory) ---
        cands_obs = self._gather_obstacles_ahead(self.obstacles, lookahead)
        now = self.get_clock().now()
        if cands_obs:
            self._mem_cands_obs = [o for _, o in cands_obs]
            self._mem_cands_time = now
        elif self._mem_cands_obs and self._mem_cands_time is not None and \
                (now - self._mem_cands_time).nanoseconds * 1e-9 < self.obs_memory_sec:
            cands_obs = self._gather_obstacles_ahead(self._mem_cands_obs, lookahead)
        obs_ahead = [o for _, o in cands_obs]
        if not obs_ahead:
            # nothing to avoid -> no avoidance path (state machine stays on the raceline)
            return _empty()
        nearest = obs_ahead[0]
        self.obs_in_interest = nearest
        g_near = (nearest.s_center - self.cur_s) % self.gb_max_s        # forward gap to nearest obs
        obs_half_s = ((nearest.s_end - nearest.s_start) % self.gb_max_s) / 2.0

        # --- maneuver geometry (arc lengths from grid_start) ---
        # Stay on the raceline until close, ramp to d_end over ramp_len ending just before the
        # obstacle, HOLD across it, then ease back to 0 and run a short raceline tail. Returning to
        # the raceline is essential: a held offset would violate the (narrow) corridor somewhere
        # downstream and reject EVERY candidate -> all-red / no avoidance. Slalom is handled by
        # per-cycle replanning (next cycle targets the next obstacle).
        e_psi = float(self.converter.get_e_psi(self.cur_x, self.cur_y, self.cur_yaw))
        cur_dp = float(np.tan(np.clip(e_psi, -0.5, 0.5)))
        L_shift = float(np.clip(g_near - obs_half_s - self.shift_buffer, self.shift_min, lookahead))
        s_ramp0 = max(0.0, L_shift - self.ramp_len)               # start easing off the raceline
        s_hold_end = max(g_near + obs_half_s + self.hold_after, L_shift + 0.2)
        s_ret_end = s_hold_end + self.return_len
        span = min(s_ret_end + self.tail_m, self.gb_max_s * 0.9)

        # --- s-grid for the path ---
        car_idx = int(self.cur_s / wpnt_dist) % self.gb_max_idx
        grid_start_s = gb_wpnts[car_idx].s_m
        n = max(int(span / wpnt_dist), 5)
        idxs = (car_idx + np.arange(n)) % self.gb_max_idx
        s_abs = grid_start_s + np.arange(n) * wpnt_dist
        s_mod = s_abs % self.gb_max_s
        s_local = s_abs - grid_start_s
        gap_wp = (s_abs - self.cur_s) % self.gb_max_s

        d_left_arr = np.array([gb_wpnts[j].d_left for j in idxs])
        d_right_arr = np.array([gb_wpnts[j].d_right for j in idxs])   # magnitude of right half-width
        v_gb_arr = np.array([gb_wpnts[j].vx_mps for j in idxs])

        # --- terminal-offset samples: corridor AT THE OBSTACLE (where clearance actually matters) ---
        obs_j = int(nearest.s_center / wpnt_dist) % self.gb_max_idx
        d_hi = gb_wpnts[obs_j].d_left - sample_margin
        d_lo = -(gb_wpnts[obs_j].d_right - sample_margin)
        if d_hi <= d_lo:
            d_ends = np.array([0.0])
        else:
            d_ends = np.linspace(d_lo, d_hi, self.n_d_samples)
            d_ends[int(np.argmin(np.abs(d_ends)))] = 0.0   # snap nearest sample onto the raceline
        N = len(d_ends)

        # --- d(s): flat cur_d | ramp cur_d->d_end | hold d_end across obstacle | ramp d_end->0 | tail 0 ---
        m_ramp = (s_local > s_ramp0) & (s_local <= L_shift)
        m_hold = (s_local > L_shift) & (s_local <= s_hold_end)
        m_ret = (s_local > s_hold_end) & (s_local <= s_ret_end)
        ramp_ok = L_shift > s_ramp0 + 1e-3
        d_cands = np.zeros((N, n))
        for k, d_end in enumerate(d_ends):
            de = float(d_end)
            dv = np.full(n, self.cur_d)
            if ramp_ok and m_ramp.any():
                dp0 = cur_dp if s_ramp0 == 0.0 else 0.0   # match car heading only if ramp starts at the car
                p_in = BPoly.from_derivatives([s_ramp0, L_shift], [[self.cur_d, dp0, 0.0], [de, 0.0, 0.0]])
                dv[m_ramp] = p_in(s_local[m_ramp])
            else:
                dv[m_ramp] = de
            dv[m_hold] = de
            if m_ret.any():
                p_out = BPoly.from_derivatives([s_hold_end, s_ret_end], [[de, 0.0, 0.0], [0.0, 0.0, 0.0]])
                dv[m_ret] = p_out(s_local[m_ret])
            dv[s_local > s_ret_end] = 0.0
            d_cands[k] = dv

        # --- feasibility 1: track corridor (reject, don't clip) ---
        bound_ok = ~(((d_cands > (d_left_arr - half_car)[None, :]) |
                      (d_cands < -(d_right_arr - half_car)[None, :])).any(axis=1))

        # --- feasibility 2: inflated obstacle boxes ---
        obs_ok = np.ones(N, dtype=bool)
        for o in obs_ahead:
            g0 = (o.s_start - self.cur_s) % self.gb_max_s - obs_margin
            g1 = (o.s_end - self.cur_s) % self.gb_max_s + obs_margin
            if g1 < g0:   # box straddles the s seam; skip this frame (handled once past the seam)
                continue
            d_box_lo = min(o.d_right, o.d_left) - obs_margin
            d_box_hi = max(o.d_right, o.d_left) + obs_margin
            s_in = (gap_wp >= g0) & (gap_wp <= g1)
            d_in = (d_cands >= d_box_lo) & (d_cands <= d_box_hi)
            obs_ok &= ~(d_in & s_in[None, :]).any(axis=1)

        # cartesian for ALL candidates in one converter call (viz + downstream checks)
        resp = self.converter.get_cartesian(np.tile(s_mod, N), d_cands.reshape(-1))
        xy_all = (resp.T if resp.ndim == 2 else resp).reshape(N, n, 2)

        # --- heavy checks (grid, curvature, cost) only on geometric survivors ---
        obs_xy = np.array([[o.x_m, o.y_m] for o in obs_ahead], dtype=float)
        best_k, best_J, best = -1, np.inf, None
        status = ["reject"] * N
        n_bounds = n_obs = n_grid = n_curv = 0   # per-stage reject counters (diagnostics)
        for k in range(N):
            if not bound_ok[k]:
                n_bounds += 1
                continue
            if not obs_ok[k]:
                n_obs += 1
                continue
            xy = xy_all[k]
            if self.use_grid_check and self._path_hits_grid(xy):
                n_grid += 1
                continue
            psi_, kappa_ = tph.calc_head_curv_num.calc_head_curv_num(
                path=xy, el_lengths=wpnt_dist * np.ones(len(xy) - 1), is_closed=False)
            if np.max(np.abs(kappa_)) > self.kappa_max:
                n_curv += 1
                continue
            j_d = self.w_d * float(np.sum(np.abs(d_cands[k])))
            j_k = self.w_k * float(np.sum(kappa_ ** 2))
            j_c = self.w_c * abs(float(d_ends[k]) - self._d_end_prev)
            if obs_xy.shape[0]:
                mind = np.sqrt(((xy[:, None, :] - obs_xy[None, :, :]) ** 2).sum(-1)).min(axis=1)
                j_o = self.w_obs * float(np.sum(np.exp(-mind / self.obs_sigma)))
            else:
                j_o = 0.0
            J = j_d + j_k + j_c + j_o
            status[k] = "feasible"
            if J < best_J:
                best_J, best_k, best = J, k, (xy, psi_, kappa_)

        if best is None:
            # Diagnostics: which stage killed every candidate? corridor@obs vs obstacle box vs grid
            # vs curvature, with the geometry so you can see if it's genuinely impassable or a knob.
            self.get_logger().warn(
                f"[{self.name}] NO feasible candidate ({N} sampled) -> TRAILING | "
                f"reject bounds={n_bounds} obs_box={n_obs} grid={n_grid} curv={n_curv} | "
                f"g_near={g_near:.2f} obs_half_s={obs_half_s:.2f} L_shift={L_shift:.2f} | "
                f"sample d_range=[{d_lo:.2f},{d_hi:.2f}] corridor@obs "
                f"L={gb_wpnts[obs_j].d_left:.2f}/R={gb_wpnts[obs_j].d_right:.2f} | "
                f"obs d=[{min(nearest.d_right, nearest.d_left):.2f},{max(nearest.d_right, nearest.d_left):.2f}] "
                f"obs_margin={obs_margin:.2f} sample_margin={sample_margin:.2f}",
                throttle_duration_sec=0.5)
            self._publish_feasible(False)
            wpnts.wpnts = []
            return wpnts, self._candidate_markers(xy_all, status, -1)

        status[best_k] = "selected"
        self._d_end_prev = float(d_ends[best_k])
        xy, psi_, kappa_ = best

        if SMOOTH_OTWPNTS and kappa_.size > SMOOTH_OTWPNTS_POLYORDER + 2:
            win = min(SMOOTH_OTWPNTS_WINDOW, kappa_.size)
            if win % 2 == 0:
                win -= 1
            if win > SMOOTH_OTWPNTS_POLYORDER:
                kappa_ = savgol_filter(kappa_, win, SMOOTH_OTWPNTS_POLYORDER)

        # velocity: scaled-raceline speed clamped by the local lateral-accel limit
        v_curv = np.sqrt(self.a_lat_max / np.maximum(np.abs(kappa_), 1e-3))
        v_arr = np.minimum(v_gb_arr, v_curv)

        d_sel = d_cands[best_k]
        for i in range(len(xy)):
            wpnts.wpnts.append(
                self.xyv_to_wpnts(x=xy[i, 0], y=xy[i, 1], s=s_mod[i], d=d_sel[i],
                                  v=float(v_arr[i]), psi=psi_[i] + np.pi / 2,
                                  kappa=kappa_[i], wpnts=wpnts))

        self._publish_feasible(True)
        return wpnts, self._candidate_markers(xy_all, status, best_k)

    def _path_hits_grid(self, xy: np.ndarray) -> bool:
        """True if any path point lands on an occupied (eroded-map) cell. Early-exits."""
        if getattr(self.map_filter, "eroded_image", None) is None:
            return False
        for x, y in xy:
            if self.map_filter.is_point_inside(float(x), float(y)):
                return True
        return False

    def _publish_feasible(self, feasible: bool):
        self._last_feasible = bool(feasible)
        self.feasible_pub.publish(Bool(data=bool(feasible)))

    ######################
    # VIZ + MSG WRAPPING #
    ######################
    def _candidate_markers(self, cands_xy: np.ndarray, status: List[str], sel_idx: int) -> MarkerArray:
        """One LINE_STRIP per sampled candidate: grey=feasible, red=rejected, green=selected."""
        mrks = MarkerArray()
        del_mrk = Marker()
        del_mrk.header.frame_id = "map"
        del_mrk.action = Marker.DELETEALL
        mrks.markers.append(del_mrk)
        for k in range(cands_xy.shape[0]):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = self.get_clock().now().to_msg()
            mrk.ns = "avoidance_candidates"
            mrk.id = k
            mrk.type = Marker.LINE_STRIP
            mrk.action = Marker.ADD
            mrk.pose.orientation.w = 1.0
            if k == sel_idx:
                mrk.scale.x = 0.10
                mrk.color.r, mrk.color.g, mrk.color.b, mrk.color.a = 0.0, 1.0, 0.0, 1.0
            elif status[k] == "reject":
                mrk.scale.x = 0.04
                mrk.color.r, mrk.color.g, mrk.color.b, mrk.color.a = 1.0, 0.0, 0.0, 0.6
            else:
                mrk.scale.x = 0.04
                mrk.color.r, mrk.color.g, mrk.color.b, mrk.color.a = 0.6, 0.6, 0.6, 0.5
            mrk.points = [Point(x=float(cands_xy[k, i, 0]), y=float(cands_xy[k, i, 1]), z=0.0)
                          for i in range(cands_xy.shape[1])]
            mrks.markers.append(mrk)
        return mrks

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, psi: float,
                     kappa: float, wpnts: OTWpntArray) -> Wpnt:
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
