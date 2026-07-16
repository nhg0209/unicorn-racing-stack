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
# Savitzky-Golay window for the velocity profile (odd, waypoints). Smooths the raceline-speed lookup
# (sector-boundary steps / index quantization) and the final min()-crossover corners.
SMOOTH_VEL_WINDOW = 9
# Publish the (heavy) candidate MarkerArray only every Nth 20 Hz cycle. Building ~n_d_samples x ~100
# Points every cycle and flooding RViz starves the control loop / lags RViz; the path itself is still
# planned at 20 Hz, only the debug viz is decimated (4 -> 5 Hz).
MARKER_DECIM = 4


def _savgol_safe(arr: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing that no-ops on arrays too short for the window/polyorder."""
    if arr.size <= SMOOTH_OTWPNTS_POLYORDER + 2:
        return arr
    win = min(window, arr.size)
    if win % 2 == 0:
        win -= 1
    if win <= SMOOTH_OTWPNTS_POLYORDER:
        return arr
    return savgol_filter(arr, win, SMOOTH_OTWPNTS_POLYORDER)


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
        self._marker_i = 0           # candidate-marker publish decimation counter
        self._emit_markers = True    # build+publish candidate markers only on decimated cycles

        # --- sampling-planner param defaults (all overridable via ROS params / config yaml) ---
        self.kernel_size = 3         # GridFilter erosion (cells); 8 ate ~0.2 m and rejected the raceline
        self.lookahead_min = 8.0     # [m]
        self.lookahead_k = 1.5       # [s]  lookahead = max(lookahead_min, k * cur_vs)
        self.n_d_samples = 13        # terminal offsets sampled across the width
        # Curvature feasibility is corner-fair: budget the curvature the MANEUVER adds over the
        # raceline (kappa_add_max) AND keep an absolute steering ceiling (kappa_abs_max = physical
        # min turn radius). An absolute-only check rejected every offset in a corner because the
        # raceline curvature alone already ate the budget -> flat spline, no avoidance.
        self.kappa_add_max = 2.0     # [1/m] max curvature the maneuver may ADD over the raceline
        self.kappa_abs_max = 3.5     # [1/m] absolute curvature ceiling (min turn radius)
        self.a_lat_max = 6.0         # [m/s^2] lateral-accel cap for the velocity profile
        self.a_long_max = 4.0        # [m/s^2] longitudinal DECEL for the backward pass (brake into the apex)
        self.a_long_accel = 3.0      # [m/s^2] longitudinal ACCEL for the forward pass (gentle exit ramp-up;
                                     # lower = more gradual "fast-out" acceleration off the apex)
        self.safety_margin = 0.16    # [m] extra clearance around the obstacle box (beyond half car).
                                     # obs_margin = half_car(0.15)+safety must cover the sim ego collision
                                     # radius (0.29 m = half car LENGTH); 0.16 -> 0.31 clears it (+0.02).
        self.wall_margin = 0.05      # [m] clearance to the wall the candidate may reach (corridor)
        self.shift_min = 1.0         # [m] min arc length over which the lateral maneuver completes
        self.shift_buffer = 0.5      # [m] finish the shift this far before the obstacle near-edge
        self.ramp_len = 4.0          # [m] gentle entry-ramp length (raceline -> apex)
        self.hold_after = 0.5        # [m] (unused in apex-loaded profile; kept for param compatibility)
        self.return_len = 2.5        # [m] gentle exit-ramp length (apex -> raceline)
        self.apex_bulge = 0.10       # [m] extra offset at the box CENTRE (apex) beyond the clearance
                                     # value: higher = car swings WIDER around the obstacle. 0 = flat hold.
        self.max_weave = 3           # max obstacles woven into one path (slalom); 1 = single-apex only
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
            'kernel_size', 'lookahead_min', 'lookahead_k', 'n_d_samples', 'kappa_max',
            'kappa_add_max', 'kappa_abs_max', 'a_lat_max', 'a_long_max', 'a_long_accel',
            'safety_margin', 'wall_margin', 'shift_min', 'shift_buffer', 'ramp_len', 'hold_after',
            'return_len', 'apex_bulge', 'max_weave', 'width_car', 'tail_m', 'w_d', 'w_k', 'w_c', 'w_obs', 'obs_sigma',
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
        self.declare_parameter('kappa_max', 2.0, dbl(0.1, 10.0, "DEPRECATED alias -> kappa_abs_max [1/m]"))
        self.declare_parameter('kappa_add_max', 2.0, dbl(0.1, 10.0, "max curvature the maneuver may ADD over the raceline [1/m]"))
        self.declare_parameter('kappa_abs_max', 3.5, dbl(0.1, 10.0, "absolute curvature ceiling / min turn radius [1/m]"))
        self.declare_parameter('a_lat_max', 6.0, dbl(1.0, 20.0, "lateral-accel cap [m/s^2]"))
        self.declare_parameter('a_long_max', 4.0, dbl(0.5, 20.0, "longitudinal decel for backward speed pass [m/s^2]"))
        self.declare_parameter('a_long_accel', 3.0, dbl(0.5, 20.0, "longitudinal accel for forward pass (gentle exit) [m/s^2]"))
        self.declare_parameter('safety_margin', 0.16, dbl(0.0, 1.0, "clearance around obstacle box [m]"))
        self.declare_parameter('wall_margin', 0.05, dbl(0.0, 1.0, "clearance to wall a candidate may reach [m]"))
        self.declare_parameter('shift_min', 1.0, dbl(0.3, 10.0, "min arc length for the lateral maneuver [m]"))
        self.declare_parameter('shift_buffer', 0.5, dbl(0.0, 5.0, "finish the shift this far before the obstacle [m]"))
        self.declare_parameter('ramp_len', 4.0, dbl(0.5, 15.0, "ramp length onto the offset [m]"))
        self.declare_parameter('hold_after', 0.5, dbl(0.0, 5.0, "hold the offset past the obstacle far-edge [m]"))
        self.declare_parameter('return_len', 2.5, dbl(0.5, 10.0, "ramp length back to the raceline [m]"))
        self.declare_parameter('apex_bulge', 0.10, dbl(0.0, 1.0, "extra apex offset beyond clearance: higher=wider avoidance [m]"))
        self.declare_parameter('max_weave', 3, intd(1, 5, "max obstacles woven into one path (slalom); 1=single-apex"))
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
                self.kappa_abs_max = float(p.value)   # deprecated alias -> absolute ceiling
            elif n == 'kappa_add_max':
                self.kappa_add_max = float(p.value)
            elif n == 'kappa_abs_max':
                self.kappa_abs_max = float(p.value)
            elif n == 'a_lat_max':
                self.a_lat_max = float(p.value)
            elif n == 'a_long_max':
                self.a_long_max = float(p.value)
            elif n == 'a_long_accel':
                self.a_long_accel = float(p.value)
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
            elif n == 'apex_bulge':
                self.apex_bulge = float(p.value)
            elif n == 'max_weave':
                self.max_weave = int(p.value)
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
        new_wpnts = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        changed = (self.waypoints is None or new_wpnts.shape != self.waypoints.shape
                   or not np.allclose(new_wpnts, self.waypoints))
        self.waypoints = new_wpnts
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m
        # The global line can CHANGE at runtime (static re-optimization swaps in an obstacle-aware
        # line). Rebuild the FrenetConverter so avoidance splines are generated relative to the
        # CURRENT line the car follows — not the startup (clean) one (else they are offset). Only
        # after the initial converter exists, and only on an ACTUAL change (no per-message churn).
        if changed and getattr(self, "converter", None) is not None:
            self.converter = self.initialize_converter()

    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        # decimate the (heavy) candidate markers to ~5 Hz so viz load never starves the 20 Hz plan
        self._marker_i += 1
        self._emit_markers = (self._marker_i % MARKER_DECIM == 0)
        if self.measuring:
            start = time.perf_counter()
        wpnts, mrks = self.do_spline(gb_wpnts=self.gb_scaled_wpnts.wpnts)
        if self.measuring:
            self.latency_pub.publish(Float32(data=float(time.perf_counter() - start)))
        self.evasion_pub.publish(wpnts)
        if self._emit_markers:
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
            # detection-gated: only avoid an obstacle we currently SEE. The tracker keeps confirmed
            # statics in memory (is_visible=False when remembered-but-unseen) for continuity, but
            # planning off a remembered position looks like the car "knows" the box in advance.
            # Brief close-range dropouts are bridged by obs_memory_sec below.
            if not o.is_visible:
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

        # --- avoidance knots: ONE smooth hump per obstacle, peaking at the obstacle centre ---
        # Single knot per obstacle at its centre: the path is one clean quintic hump -- gentle
        # monotonic rise from the raceline, WIDEST at the apex (beside the obstacle), gentle monotonic
        # fall back. The s-inflated obstacle box is verified by the feasibility filter (obs_ok); the
        # sampled offset (+ apex_bulge) is chosen high enough that the hump clears it. Several obstacles
        # -> a woven chain of humps (one apex each).
        e_psi = float(self.converter.get_e_psi(self.cur_x, self.cur_y, self.cur_yaw))
        cur_dp = float(np.tan(np.clip(e_psi, -0.5, 0.5)))
        knots = []          # [(s_centre, obstacle, corridor_idx), ...] strictly increasing in s
        for o in obs_ahead:
            s_c = float(np.clip((o.s_center - self.cur_s) % self.gb_max_s, 0.3, lookahead))
            if knots and s_c <= knots[-1][0] + 0.4:
                continue                                   # too close in s to the previous apex -> merge
            knots.append((s_c, o, int(o.s_center / wpnt_dist) % self.gb_max_idx))
            if len(knots) >= self.max_weave:
                break
        g_near = (nearest.s_center - self.cur_s) % self.gb_max_s       # forward gap to nearest obstacle
        obs_half_s = ((nearest.s_end - nearest.s_start) % self.gb_max_s) / 2.0
        s_entry0 = max(0.0, knots[0][0] - self.ramp_len)              # gentle ramp OUT starts here
        s_exit_end = knots[-1][0] + self.return_len                   # ease back to the raceline after the LAST apex
        span = min(s_exit_end + self.tail_m, self.gb_max_s * 0.9)

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
        kappa_ref = np.array([gb_wpnts[j].kappa_radpm for j in idxs])  # raceline curvature (corner-fair check)

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

        # --- d(s): raceline -> [hold across box_1] -> ... -> [hold across box_m] -> raceline ---
        # The nearest apex offset is SAMPLED (d_end); each LATER apex offset is auto-chosen to clear
        # that obstacle on the side nearer the previous one (smooth weave). One knot per obstacle at its
        # centre -> a single clean hump per obstacle (raceline -> apex -> raceline), no flat shoulders.
        def _pass_offset(cor_idx, o, prev_d):
            dl = gb_wpnts[cor_idx].d_left
            dr = gb_wpnts[cor_idx].d_right
            obox_lo = min(o.d_right, o.d_left) - obs_margin   # car-centre keep-out, right edge
            obox_hi = max(o.d_right, o.d_left) + obs_margin   # car-centre keep-out, left edge
            opts = []
            if obox_hi <= (dl - sample_margin) + 1e-6:        # room to pass on the LEFT of the obstacle
                opts.append(obox_hi)
            if obox_lo >= -(dr - sample_margin) - 1e-6:       # room to pass on the RIGHT of the obstacle
                opts.append(obox_lo)
            if not opts:
                return prev_d                                  # blocked -> keep prev (obs_ok will reject)
            return min(opts, key=lambda d: abs(d - prev_d))   # side nearer the previous apex -> smooth

        m_span = (s_local > s_entry0) & (s_local <= s_exit_end)
        span_ok = s_exit_end > s_entry0 + 1e-3
        dp0 = cur_dp if s_entry0 == 0.0 else 0.0              # match car heading only if the ramp starts at the car
        d_cands = np.zeros((N, n))
        for k, d_end in enumerate(d_ends):
            d_apex = [float(d_end)]
            for i in range(1, len(knots)):
                d_apex.append(_pass_offset(knots[i][2], knots[i][1], d_apex[-1]))
            dv = np.full(n, self.cur_d)
            if span_ok and m_span.any():
                # One knot per obstacle centre -> a single smooth quintic hump (raceline -> apex ->
                # raceline). d'=0 at each apex makes it the peak. apex_bulge pushes the peak FURTHER
                # from the obstacle (wider swing); the feasibility filter verifies box clearance.
                bp_s = [s_entry0]
                bp_d = [[self.cur_d, dp0, 0.0]]
                for (s_c, _o, _cor), da in zip(knots, d_apex):
                    d_peak = da + float(np.sign(da)) * self.apex_bulge
                    bp_s.append(s_c)
                    bp_d.append([d_peak, 0.0, 0.0])
                bp_s.append(s_exit_end)
                bp_d.append([0.0, 0.0, 0.0])
                dv[m_span] = BPoly.from_derivatives(bp_s, bp_d)(s_local[m_span])
            dv[s_local > s_exit_end] = 0.0
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
            if self.use_grid_check and self._path_off_track(xy):
                n_grid += 1
                continue
            psi_, kappa_ = tph.calc_head_curv_num.calc_head_curv_num(
                path=xy, el_lengths=wpnt_dist * np.ones(len(xy) - 1), is_closed=False)
            # Corner-fair curvature: allow what the raceline already curves, bound only the
            # curvature the maneuver ADDS, plus an absolute steering ceiling. (An absolute-only
            # check rejected every offset in a corner -> flat spline, no avoidance.)
            if (np.max(np.abs(kappa_ - kappa_ref)) > self.kappa_add_max or
                    np.max(np.abs(kappa_)) > self.kappa_abs_max):
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
                f"g_near={g_near:.2f} obs_half_s={obs_half_s:.2f} n_box={len(knots)} apex_bulge={self.apex_bulge:.2f} | "
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

        if SMOOTH_OTWPNTS:
            kappa_ = _savgol_safe(kappa_, SMOOTH_OTWPNTS_WINDOW)

        # velocity: slow-in / fast-out around the apex, jitter-free. Smooth EVERYTHING first, run the
        # accel/decel passes LAST so the final profile is both smooth AND shape-guaranteed:
        #   0) smooth the raceline-speed lookup (kills sector-boundary steps / index quantization),
        #      the curvature (above), and the min()-crossover corner -> no high-frequency noise
        #   1) point limit  v_curv = sqrt(a_lat/|kappa|)  -> minimum sits AT the apex (max curvature)
        #   2) backward decel pass -> brake EARLY so the car is already slow entering the apex
        #   3) forward accel pass  -> leave the apex and accelerate out GRADUALLY (bounded a_long_accel)
        v_gb_s = _savgol_safe(v_gb_arr, SMOOTH_VEL_WINDOW)
        v_curv = np.sqrt(self.a_lat_max / np.maximum(np.abs(kappa_), 1e-3))
        v_arr = np.clip(_savgol_safe(np.minimum(v_gb_s, v_curv), SMOOTH_VEL_WINDOW), 0.0, v_gb_s)
        # backward: v[i]^2 <= v[i+1]^2 + 2*a_brake*ds  (ds = wpnt_dist)
        for i in range(len(v_arr) - 2, -1, -1):
            v_arr[i] = min(v_arr[i], float(np.sqrt(v_arr[i + 1] ** 2 + 2.0 * self.a_long_max * wpnt_dist)))
        # forward: v[i]^2 <= v[i-1]^2 + 2*a_accel*ds  -> gentle exit ramp-up (lower a_long_accel = gentler)
        for i in range(1, len(v_arr)):
            v_arr[i] = min(v_arr[i], float(np.sqrt(v_arr[i - 1] ** 2 + 2.0 * self.a_long_accel * wpnt_dist)))

        d_sel = d_cands[best_k]
        for i in range(len(xy)):
            wpnts.wpnts.append(
                self.xyv_to_wpnts(x=xy[i, 0], y=xy[i, 1], s=s_mod[i], d=d_sel[i],
                                  v=float(v_arr[i]), psi=psi_[i] + np.pi / 2,
                                  kappa=kappa_[i], wpnts=wpnts))

        self._publish_feasible(True)
        return wpnts, self._candidate_markers(xy_all, status, best_k)

    def _path_off_track(self, xy: np.ndarray) -> bool:
        """True if any path point is NOT in free/drivable space (on/near a wall). Early-exits.

        NOTE: GridFilter.is_point_inside() returns True when the point is INSIDE the free
        (eroded) drivable area and False on/near a wall -- so a candidate is rejected when a
        point is NOT inside. The map-not-loaded guard is essential: without it every point
        reads 'not inside' and all candidates would be rejected.
        """
        if getattr(self.map_filter, "eroded_image", None) is None:
            return False   # no map yet -> rely on the waypoint corridor bounds only
        for x, y in xy:
            if not self.map_filter.is_point_inside(float(x), float(y)):
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
        if not self._emit_markers:
            return MarkerArray()   # decimated cycle: skip the ~n_d_samples x ~100 Point build entirely
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
