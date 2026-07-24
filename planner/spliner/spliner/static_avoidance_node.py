#!/usr/bin/env python3
import time
from typing import List, Optional, Tuple

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
        - ``/planner/avoidance/static_feasible`` (Bool)        False if 0 feasible candidates
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
        self.sample_gaps = True      # sample the drivable gaps beside the obstacle (vs a uniform
                                     # corridor sweep that skips the narrow gap on a lopsided corridor)
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
        # --- corridor source: measure the drivable width from the MAP, not from the waypoints ---
        # d_left/d_right come from gb_optimizer, which labels the two track contours left/right with
        # ONE global decision taken from the start pose. When that decision comes out inverted the
        # whole lap ships with d_left/d_right exchanged (map f: 125/128 waypoints swapped, values
        # exact mirrors; map ifac: 0 swapped). The planner then believes the roomy side is the side
        # the wall is actually on, samples terminal offsets straight into that wall, and the corridor
        # filter rejects the genuinely free side -> "candidates on the opposite side of the gap".
        # The occupancy grid is the only wall source that cannot be mislabelled, so measure the free
        # lateral extent there and use it as the authority whenever a map is loaded.
        self.trust_grid_bounds = True
        self.grid_scan_max = 3.0     # [m] half-width of the lateral scan around the raceline
        self.grid_scan_step = 0.05   # [m] lateral scan resolution (one map cell)
        self.bounds_warn_m = 0.5     # [m] warn when waypoint bounds and the grid disagree by more
        # Raceline-clear gate: when the CURRENT global line (d=0 in its own frenet frame) already
        # clears every static obstacle ahead — the obstacle-aware line swapped in by static_reopt —
        # this planner must stay IDLE. Planning anyway re-recorded apexes on top of the swapped
        # line (the re-opt then walked outward every lap) and made the SM commit pointless
        # OVERTAKEs. Hysteresis: going idle needs clear_hyst_m EXTRA clearance; once idle, only a
        # genuine keep-out violation resumes planning. Gated on |cur_d| so a maneuver in progress
        # is never abandoned mid-hump.
        self.clear_gate_enable = True
        self.clear_hyst_m = 0.03     # [m] extra clearance required to ENTER the idle state
        self.clear_max_cur_d = 0.15  # [m] gate only applies with the car ON the raceline
        self._line_clear = False     # idle latch (hysteresis state)

        # --- path commitment (temporal consistency) ------------------------------------------
        # Once a feasible evasion path is chosen, COMMIT to it: keep republishing that SAME
        # world-fixed path each cycle (re-slicing only the portion still ahead of the car)
        # instead of re-solving from the car's instantaneous pose. Re-solving every cycle
        # re-anchored the entry ramp to the moving/erroring car the moment the obstacle came
        # within ramp_len (s_entry0 clamps to 0, dp0 = cur_dp) -> the hump compressed and the
        # selected candidate shifted as the gap shrank, so the SM (which re-latches its cached
        # spline on a >=0.15 m peak-d change) kept swapping the tracked path and the car
        # "re-avoided the same obstacle weirdly" on approach. The commit is DROPPED and a fresh
        # plan taken only on a real trigger: the committed slice no longer clears the LIVE
        # obstacle boxes / corridor (safety -> republish feasible=False), the car drifted off the
        # committed path (controller lost it), the triggering box moved/vanished while still
        # ahead, or the maneuver finished. During static OVERTAKE sustain the SM does NO
        # independent obstacle re-check -- the static_feasible flag is the sole interlock -- so
        # feasibility is RE-DERIVED against live obstacles every cycle here: the geometry is
        # frozen, the safety verdict is not.
        self.commit_enable = True
        self.commit_dev_max = 0.35   # [m] drop the commit if |cur_d - committed_d(car)| exceeds this
        self.commit_obs_ds = 0.75    # [m] drop the commit if the triggering box's s drifts this far
        self.commit_obs_dd = 0.40    # [m] ... or its d drifts this far (re-plan the apex around it)
        self._committed = None       # cached selected path (frenet + cartesian arrays) or None

        # Static params
        self.declare_parameters(namespace='', parameters=[('from_bag', False), ('measure', False)])
        self.from_bag = self.get_parameter('from_bag').get_parameter_value().bool_value
        self.measuring = self.get_parameter('measure').get_parameter_value().bool_value

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        self.declare_all_parameters()
        # Sync members from loaded params (yaml/defaults), then register live-reconfigure callback.
        self.dyn_param_cb(self.get_parameters([
            'kernel_size', 'lookahead_min', 'lookahead_k', 'n_d_samples', 'sample_gaps', 'kappa_max',
            'kappa_add_max', 'kappa_abs_max', 'a_lat_max', 'a_long_max', 'a_long_accel',
            'safety_margin', 'wall_margin', 'shift_min', 'shift_buffer', 'ramp_len', 'hold_after',
            'return_len', 'apex_bulge', 'max_weave', 'width_car', 'tail_m', 'w_d', 'w_k', 'w_c', 'w_obs', 'obs_sigma',
            'use_grid_check', 'trust_grid_bounds', 'grid_scan_max', 'grid_scan_step', 'bounds_warn_m',
            'clear_gate_enable', 'clear_hyst_m', 'clear_max_cur_d',
            'commit_enable', 'commit_dev_max', 'commit_obs_ds', 'commit_obs_dd',
        ]))
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        # Subscribers
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacles_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)

        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers", 10)
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", 10)
        # published on the CANONICAL name the SM subscribes (no launch remap needed): a partial
        # bring-up without the remap must not leave the SM's feasibility gate silently open
        self.feasible_pub = self.create_publisher(Bool, "/planner/avoidance/static_feasible", 10)
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
        self.declare_parameter('sample_gaps', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Sample the drivable gaps beside the obstacle (vs a uniform corridor sweep)"))
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
        self.declare_parameter('trust_grid_bounds', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Measure the drivable corridor from the occupancy grid instead of waypoint d_left/d_right"))
        self.declare_parameter('grid_scan_max', 3.0, dbl(0.5, 10.0, "half-width of the lateral grid corridor scan [m]"))
        self.declare_parameter('grid_scan_step', 0.05, dbl(0.01, 0.5, "lateral grid corridor scan resolution [m]"))
        self.declare_parameter('bounds_warn_m', 0.5, dbl(0.0, 5.0, "warn when waypoint bounds and the grid disagree by more [m]"))
        self.declare_parameter('clear_gate_enable', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Stay idle when the current raceline already clears every obstacle ahead"))
        self.declare_parameter('clear_hyst_m', 0.03, dbl(0.0, 0.5, "extra clearance to ENTER the idle state [m]"))
        self.declare_parameter('clear_max_cur_d', 0.15, dbl(0.0, 1.0, "clear gate applies only when |cur_d| below this [m]"))
        self.declare_parameter('commit_enable', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Commit to a chosen evasion path and reuse it (temporal consistency) instead of re-solving every cycle"))
        self.declare_parameter('commit_dev_max', 0.35, dbl(0.05, 2.0, "drop the commit if the car deviates this far from the committed path [m]"))
        self.declare_parameter('commit_obs_ds', 0.75, dbl(0.05, 5.0, "drop the commit if the triggering obstacle s drifts this far [m]"))
        self.declare_parameter('commit_obs_dd', 0.40, dbl(0.05, 2.0, "drop the commit if the triggering obstacle d drifts this far [m]"))

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
            elif n == 'sample_gaps':
                self.sample_gaps = bool(p.value)
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
            elif n == 'trust_grid_bounds':
                self.trust_grid_bounds = bool(p.value)
            elif n == 'grid_scan_max':
                self.grid_scan_max = float(p.value)
            elif n == 'grid_scan_step':
                self.grid_scan_step = float(p.value)
            elif n == 'bounds_warn_m':
                self.bounds_warn_m = float(p.value)
            elif n == 'clear_gate_enable':
                self.clear_gate_enable = bool(p.value)
            elif n == 'clear_hyst_m':
                self.clear_hyst_m = float(p.value)
            elif n == 'clear_max_cur_d':
                self.clear_max_cur_d = float(p.value)
            elif n == 'commit_enable':
                self.commit_enable = bool(p.value)
                if not self.commit_enable:
                    self._committed = None
            elif n == 'commit_dev_max':
                self.commit_dev_max = float(p.value)
            elif n == 'commit_obs_ds':
                self.commit_obs_ds = float(p.value)
            elif n == 'commit_obs_dd':
                self.commit_obs_dd = float(p.value)
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
            self._committed = None   # cached path is in the OLD frenet frame -> re-plan on the new line

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
            # Trust the (position-persistence hardened) tracker flag. The old wide velocity
            # fallback (|vs|<0.5) also caught MOVING opponents whenever their EKF speed dipped
            # (slow corners / filter spin-up): this planner then splined around the moving car
            # and the SM committed a STATIC overtake (which has NO sector gate) during the
            # approach window BEFORE the lane-change planner engages (gap > engage_gap_m) --
            # hijacking the head-to-head behavior with a snapshot spline. Keep only a tight
            # near-zero band as a belt for a real box transiently demoted while its EKF
            # re-initializes (its speed then reads ~0, a driving opponent does not).
            near_zero = abs(o.vs) < 0.15 and abs(o.vd) < 0.15
            if not (o.is_static or near_zero):
                continue
            if not o.is_static and near_zero:
                self.get_logger().info(
                    f"[{self.name}] treating dynamic-flagged obs id={o.id} as static "
                    f"(near-zero speed vs={o.vs:.2f} vd={o.vd:.2f})",
                    throttle_duration_sec=2.0)
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

        # --- committed-path reuse: follow the SAME world-fixed evasion path we already chose ---
        # (see the commit_* notes in __init__). Re-solving from the instantaneous pose every cycle
        # is what made the car re-avoid the same obstacle on approach; here we just re-slice and
        # republish the frozen path. Safety is NOT frozen: _reuse_committed re-derives feasibility
        # against the live obstacles and publishes feasible=False the instant the slice stops
        # clearing them. Runs BEFORE the obstacle gather so the committed exit ramp is still
        # followed once the box has dropped out of "ahead".
        if self.commit_enable and self._committed is not None:
            reuse = self._reuse_committed(gb_wpnts, wpnt_dist, obs_margin, half_car)
            if reuse is not None:
                return reuse

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
            self._line_clear = False
            self._committed = None
            return _empty()

        # --- raceline-clear gate: the current global line may ALREADY avoid everything ahead ---
        # (the obstacle-aware line static_reopt swapped in). Idle then: no path -> the SM stays
        # GB_TRACK and no apex is re-recorded on top of the swapped line. Obstacle d values are in
        # the SAME frenet frame as this planner (tracking re-projects on a line swap), so "the
        # keep-out interval does not contain d=0" IS the clearance test of the followed line.
        # The |cur_d| condition guards ENTERING idle only (never abandon a maneuver mid-hump);
        # once LATCHED idle it must not bypass the gate — during the post-swap merge cur_d decays
        # through the threshold with noise, and re-planning on every excursion above it flapped
        # publish/empty at up to 20 Hz -> the SM alternated OVERTAKE<->GB_TRACK and the controller
        # received two different local paths in alternation (the "duplicate path" symptom; the L1
        # point jumped between the two lines). The latch drops only on a REAL keep-out violation.
        if self.clear_gate_enable and (self._line_clear or abs(self.cur_d) < self.clear_max_cur_d):
            need = obs_margin if self._line_clear else obs_margin + self.clear_hyst_m
            clear = all(
                (min(o.d_right, o.d_left) - need) > 0.0 or (max(o.d_right, o.d_left) + need) < 0.0
                for o in obs_ahead)
            if clear and not self._line_clear:
                self.get_logger().info(
                    f"[{self.name}] raceline clears all {len(obs_ahead)} obstacle(s) ahead "
                    f"(margin {need:.2f} m) -> planner idle")
            self._line_clear = clear
            if clear:
                self._committed = None
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

        # --- terminal-offset samples: the DRIVABLE GAPS beside the obstacle ---
        # Sample each free gap (obstacle's left edge -> left wall, and right edge -> right wall)
        # rather than a blind UNIFORM sweep of the whole corridor. On a wide, LOPSIDED corridor
        # (map f's wall-hugging min-curvature raceline: |d_left-d_right|>1 m over ~2/3 of the lap)
        # a uniform n_d_samples sweep lands ~all samples on the roomy side and can give the narrow
        # side ZERO box-clearing candidates -> if the free gap is the narrow side the planner never
        # populates it and avoidance flips to the open-but-wrong side ("candidates on the opposite
        # side of the gap"). ifac's near-symmetric corridor hides this. Gap-anchored sampling
        # always populates BOTH sides that have room, so the correct side is never missed.
        obs_j = int(nearest.s_center / wpnt_dist) % self.gb_max_idx
        d_hi_wp = gb_wpnts[obs_j].d_left - sample_margin       # left corridor limit (car centre)
        d_lo_wp = -(gb_wpnts[obs_j].d_right - sample_margin)   # right corridor limit (car centre)
        # Prefer the corridor MEASURED in the occupancy grid: d_left/d_right are labelled left/right
        # by one global decision in gb_optimizer and ship exchanged on some maps (see the
        # trust_grid_bounds note in __init__), which puts every sample into the wall on the wrong side.
        grid_cor = self._grid_corridor(nearest.s_center) if self.trust_grid_bounds else None
        if grid_cor is not None:
            d_lo, d_hi = grid_cor
            if abs(d_hi - d_hi_wp) > self.bounds_warn_m or abs(d_lo - d_lo_wp) > self.bounds_warn_m:
                self.get_logger().warn(
                    f"[{self.name}] waypoint bounds disagree with the map at s={nearest.s_center:.1f}: "
                    f"wpnt d=[{d_lo_wp:+.2f},{d_hi_wp:+.2f}] (d_left={gb_wpnts[obs_j].d_left:.2f} "
                    f"d_right={gb_wpnts[obs_j].d_right:.2f}) vs grid d=[{d_lo:+.2f},{d_hi:+.2f}] -> using "
                    f"the grid. Near-mirrored values mean global_waypoints.json has d_left/d_right "
                    f"SWAPPED; regenerate the map (or set trust_grid_bounds:=false to force the "
                    f"waypoint bounds).", throttle_duration_sec=5.0)
        else:
            d_lo, d_hi = d_lo_wp, d_hi_wp
        cor_src = "grid" if grid_cor is not None else "wpnt"
        obox_lo = min(nearest.d_right, nearest.d_left) - obs_margin   # car-centre keep-out, right edge
        obox_hi = max(nearest.d_right, nearest.d_left) + obs_margin   # car-centre keep-out, left edge
        n_left = n_right = 0
        if d_hi <= d_lo:
            d_ends = np.array([0.0])
        elif self.sample_gaps:
            n_side = max(2, self.n_d_samples // 2)
            d_list = [0.0]                                     # always try the raceline
            lo_left = max(obox_hi, d_lo)                       # LEFT gap: car centre in [lo_left, d_hi]
            if lo_left <= d_hi + 1e-6:
                left = np.linspace(lo_left, d_hi, n_side); n_left = int(left.size)
                d_list += list(left)
            hi_right = min(obox_lo, d_hi)                      # RIGHT gap: car centre in [d_lo, hi_right]
            if hi_right >= d_lo - 1e-6:
                right = np.linspace(d_lo, hi_right, n_side); n_right = int(right.size)
                d_list += list(right)
            d_ends = np.unique(np.round(np.asarray(d_list, dtype=float), 4))
            d_ends[int(np.argmin(np.abs(d_ends)))] = 0.0       # snap nearest sample onto the raceline
        else:
            d_ends = np.linspace(d_lo, d_hi, self.n_d_samples)  # legacy uniform corridor sweep
            d_ends[int(np.argmin(np.abs(d_ends)))] = 0.0
        N = len(d_ends)

        # --- d(s): raceline -> [hold across box_1] -> ... -> [hold across box_m] -> raceline ---
        # The nearest apex offset is SAMPLED (d_end); each LATER apex offset is auto-chosen to clear
        # that obstacle on the side nearer the previous one (smooth weave). One knot per obstacle at its
        # centre -> a single clean hump per obstacle (raceline -> apex -> raceline), no flat shoulders.
        def _pass_offset(cor, o, prev_d):
            c_lo, c_hi = cor                                  # corridor at this obstacle (car centre)
            obox_lo = min(o.d_right, o.d_left) - obs_margin   # car-centre keep-out, right edge
            obox_hi = max(o.d_right, o.d_left) + obs_margin   # car-centre keep-out, left edge
            opts = []
            if obox_hi <= c_hi + 1e-6:                        # room to pass on the LEFT of the obstacle
                opts.append(obox_hi)
            if obox_lo >= c_lo - 1e-6:                        # room to pass on the RIGHT of the obstacle
                opts.append(obox_lo)
            if not opts:
                return prev_d                                  # blocked -> keep prev (obs_ok will reject)
            return min(opts, key=lambda d: abs(d - prev_d))   # side nearer the previous apex -> smooth

        # Corridor per woven obstacle, measured once (not per candidate): grid first, waypoint bounds
        # as the fallback -- same authority order as the sampled terminal offset above.
        def _corridor_at(cor_idx, s_c):
            g = self._grid_corridor(s_c) if self.trust_grid_bounds else None
            if g is not None:
                return g
            return (-(gb_wpnts[cor_idx].d_right - sample_margin),
                    gb_wpnts[cor_idx].d_left - sample_margin)

        knot_cor = [(d_lo, d_hi)] + [_corridor_at(kc, ks) for (ks, _ko, kc) in knots[1:]]

        m_span = (s_local > s_entry0) & (s_local <= s_exit_end)
        span_ok = s_exit_end > s_entry0 + 1e-3
        dp0 = cur_dp if s_entry0 == 0.0 else 0.0              # match car heading only if the ramp starts at the car
        d_cands = np.zeros((N, n))
        for k, d_end in enumerate(d_ends):
            d_apex = [float(d_end)]
            for i in range(1, len(knots)):
                d_apex.append(_pass_offset(knot_cor[i], knots[i][1], d_apex[-1]))
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
        # Skipped when the grid is the corridor authority: _path_off_track then tests EVERY path
        # point against the real eroded walls, which is the same job with a trustworthy left/right.
        # Keeping the waypoint test as well would re-apply the possibly-swapped d_left/d_right and
        # reject exactly the candidates on the genuinely free side.
        if self._grid_is_authority():
            bound_ok = np.ones(N, dtype=bool)
        else:
            bound_ok = ~(((d_cands > (d_left_arr - half_car)[None, :]) |
                          (d_cands < -(d_right_arr - half_car)[None, :])).any(axis=1))

        # --- feasibility 2: inflated obstacle boxes ---
        # Signed centre-gap + half-span (same idiom as obs_half_s above): mod-ing s_start and
        # s_end separately breaks whenever the box wraps the seam OR the car is already inside
        # the box's s-interval — the old `g1 < g0: continue` skipped the check exactly then,
        # letting candidates cut straight through an obstacle near s=0.
        obs_ok = np.ones(N, dtype=bool)
        for o in obs_ahead:
            o_span = (o.s_end - o.s_start) % self.gb_max_s
            gc = (o.s_center - self.cur_s) % self.gb_max_s
            if gc > self.gb_max_s / 2.0:
                gc -= self.gb_max_s                     # signed: negative = behind the car
            g0 = gc - o_span / 2.0 - obs_margin
            g1 = gc + o_span / 2.0 + obs_margin
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
        n_feas_left = n_feas_right = 0            # feasible candidates per side (which side has room)
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
            if d_ends[k] > 1e-6:
                n_feas_left += 1
            elif d_ends[k] < -1e-6:
                n_feas_right += 1
            if J < best_J:
                best_J, best_k, best = J, k, (xy, psi_, kappa_)

        if best is None:
            # Diagnostics: which stage killed every candidate? corridor@obs vs obstacle box vs grid
            # vs curvature, with the geometry so you can see if it's genuinely impassable or a knob.
            self.get_logger().warn(
                f"[{self.name}] NO feasible candidate ({N} sampled) -> TRAILING | "
                f"reject bounds={n_bounds} obs_box={n_obs} grid={n_grid} curv={n_curv} | "
                f"g_near={g_near:.2f} obs_half_s={obs_half_s:.2f} n_box={len(knots)} apex_bulge={self.apex_bulge:.2f} | "
                f"sample d_range=[{d_lo:.2f},{d_hi:.2f}] ({cor_src}) corridor@obs "
                f"wpnt L={gb_wpnts[obs_j].d_left:.2f}/R={gb_wpnts[obs_j].d_right:.2f} | "
                f"obs d=[{min(nearest.d_right, nearest.d_left):.2f},{max(nearest.d_right, nearest.d_left):.2f}] "
                f"obs_margin={obs_margin:.2f} sample_margin={sample_margin:.2f}",
                throttle_duration_sec=0.5)
            self._committed = None
            self._publish_feasible(False)
            wpnts.wpnts = []
            return wpnts, self._candidate_markers(xy_all, status, -1)

        status[best_k] = "selected"
        self._d_end_prev = float(d_ends[best_k])
        xy, psi_, kappa_ = best

        # Diagnostic (throttled): which side did we take, how many feasible candidates were on each
        # side, how were the samples split, and what killed the rejects. On map f this exposes a
        # "wrong side" pick as either 0 feasible on the free side (sampling) or that side being eaten
        # by grid/curv/bounds. sel_d>0 = LEFT of the raceline, <0 = RIGHT.
        sel_d = float(d_ends[best_k])
        sel_side = "LEFT" if sel_d > 1e-6 else ("RIGHT" if sel_d < -1e-6 else "RACELINE")
        self.get_logger().info(
            f"[{self.name}] avoid {sel_side} d_end={sel_d:+.2f} | feasible L={n_feas_left} R={n_feas_right} | "
            f"sampled {n_left}L+{n_right}R of {N} | reject bounds={n_bounds} obs={n_obs} grid={n_grid} curv={n_curv} | "
            f"corridor d=[{d_lo:.2f},{d_hi:.2f}] ({cor_src}) obs keep-out d=[{obox_lo:.2f},{obox_hi:.2f}]",
            throttle_duration_sec=2.0)

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
        psi_pub = psi_ + np.pi / 2                       # published heading convention (frenet tangent)
        for i in range(len(xy)):
            wpnts.wpnts.append(
                self.xyv_to_wpnts(x=xy[i, 0], y=xy[i, 1], s=s_mod[i], d=d_sel[i],
                                  v=float(v_arr[i]), psi=psi_pub[i],
                                  kappa=kappa_[i], wpnts=wpnts))

        if self.commit_enable:
            self._store_commit(obs_ahead, s_mod, d_sel, xy, v_arr, psi_pub, kappa_)
        self._publish_feasible(True)
        return wpnts, self._candidate_markers(xy_all, status, best_k)

    ######################
    # PATH COMMITMENT    #
    ######################
    def _obs_qualifies(self, o) -> bool:
        """The same static / near-stationary + currently-visible gate _gather_obstacles_ahead
        uses to pick avoidance obstacles, factored out for the committed-path re-check."""
        near_zero = abs(o.vs) < 0.15 and abs(o.vd) < 0.15
        return bool((o.is_static or near_zero) and o.is_visible)

    def _store_commit(self, obs_ahead, s_mod, d_sel, xy, v_arr, psi_pub, kappa_):
        """Snapshot the freshly chosen path (+ the obstacles it was planned around) so later
        cycles republish it verbatim instead of re-solving from the moving car."""
        self._committed = {
            'obs': [(int(o.id), float(o.s_center), float(o.d_center)) for o in obs_ahead],
            's_mod': np.asarray(s_mod, dtype=float).copy(),
            'd':     np.asarray(d_sel, dtype=float).copy(),
            'xy':    np.asarray(xy, dtype=float).copy(),
            'v':     np.asarray(v_arr, dtype=float).copy(),
            'psi':   np.asarray(psi_pub, dtype=float).copy(),   # already the published convention
            'kappa': np.asarray(kappa_, dtype=float).copy(),
        }

    def _reuse_committed(self, gb_wpnts, wpnt_dist, obs_margin, half_car):
        """Try to republish the committed path (the slice still ahead of the car). Returns
        (OTWpntArray, MarkerArray) on reuse, or None -- after dropping the commit -- when a fresh
        plan is needed. Publishes the feasibility verdict itself in every path it returns from."""
        c = self._committed
        L = self.gb_max_s

        # --- forward slice via path-local arc length (robust to the s=0 seam) ---
        # s_local is the committed path's own 0..span arc length -- its points are forward-ordered
        # and < 1 lap long, so it is strictly ascending with no wrap; car_prog is how far the car
        # has advanced along it. Keeping s_local >= car_prog - 0.30 drops only the passed prefix.
        s0 = c['s_mod'][0]
        s_local = (c['s_mod'] - s0) % L
        car_prog = (self.cur_s - s0) % L
        ahead = s_local >= (car_prog - 0.30)
        if int(ahead.sum()) < 3:
            self._committed = None                            # maneuver finished -> replan (idle next)
            return None
        i0 = int(np.argmax(ahead))                            # first point at/ahead of the car
        sel = slice(i0, len(c['s_mod']))

        # --- lateral deviation: has the controller fallen off the committed path? ---
        d_car = float(np.interp(car_prog, s_local, c['d']))   # committed d at the car
        if abs(self.cur_d - d_car) > self.commit_dev_max:
            self._committed = None                            # re-anchor once from the current pose
            return None

        # --- freshness: did a triggering box move a lot (while still ahead) ? ---
        # A box that briefly drops out of tracking is tolerated (skip): it is static, the frozen
        # path already clears it, and the safety re-check below still guards against anything that
        # HAS moved into the path. Only a same-id box that genuinely relocated forces a re-plan.
        live = list(self.obstacles)
        for (oid, os0, od0) in c['obs']:
            gc = (os0 - self.cur_s) % L
            if gc >= L / 2.0:
                continue                                      # that box is already behind -> exit ramp
            match = next((o for o in live if int(o.id) == oid), None)
            if match is None:
                continue                                      # briefly untracked static box
            ds = abs(((match.s_center - os0 + L / 2.0) % L) - L / 2.0)
            if ds > self.commit_obs_ds or abs(match.d_center - od0) > self.commit_obs_dd:
                self._committed = None                        # box moved enough -> re-plan the apex
                return None

        # --- safety: the committed slice must still clear EVERY live box + stay in the corridor ---
        # This is the sole interlock the SM has during static sustain, so it is re-derived here
        # against live obstacles every cycle: geometry frozen, verdict live.
        if not self._commit_slice_clear(c, sel, gb_wpnts, wpnt_dist, obs_margin, half_car):
            self._committed = None
            self._publish_feasible(False)                     # tell the SM to abandon the OVERTAKE
            wpnts = OTWpntArray()
            wpnts.header.stamp = self.get_clock().now().to_msg()
            wpnts.header.frame_id = "map"
            del_mrk = Marker()
            del_mrk.header.frame_id = "map"
            del_mrk.action = Marker.DELETEALL
            m = MarkerArray()
            m.markers = [del_mrk]
            return wpnts, m

        # --- OK: republish the committed forward slice ---
        wpnts = self._commit_to_msg(c, sel)
        self._publish_feasible(True)
        return wpnts, self._commit_markers(c, sel)

    def _commit_slice_clear(self, c, sel, gb_wpnts, wpnt_dist, obs_margin, half_car) -> bool:
        """True if the committed forward slice stays inside the track corridor AND clears every
        live (static / near-stationary, visible) obstacle's inflated box. Same box idiom as the
        obs_ok check in do_spline, evaluated on the frozen path against the CURRENT obstacles."""
        s_mod = c['s_mod'][sel]
        d = c['d'][sel]
        L = self.gb_max_s
        # Corridor: the eroded map when it is the authority (the waypoint bounds can ship with
        # d_left/d_right exchanged, which would drop a perfectly good committed path every cycle),
        # otherwise the waypoint corridor.
        if self._grid_is_authority():
            if self._path_off_track(c['xy'][sel]):
                return False
        else:
            idxs = (s_mod / wpnt_dist).astype(int) % self.gb_max_idx
            d_left = np.array([gb_wpnts[j].d_left for j in idxs])
            d_right = np.array([gb_wpnts[j].d_right for j in idxs])
            if np.any(d > (d_left - half_car)) or np.any(d < -(d_right - half_car)):
                return False
        gap_wp = (s_mod - self.cur_s) % L
        for o in self.obstacles:
            if not self._obs_qualifies(o):
                continue
            o_span = (o.s_end - o.s_start) % L
            gc = (o.s_center - self.cur_s) % L
            if gc > L / 2.0:
                gc -= L
            g0 = gc - o_span / 2.0 - obs_margin
            g1 = gc + o_span / 2.0 + obs_margin
            d_lo = min(o.d_right, o.d_left) - obs_margin
            d_hi = max(o.d_right, o.d_left) + obs_margin
            s_in = (gap_wp >= g0) & (gap_wp <= g1)
            d_in = (d >= d_lo) & (d <= d_hi)
            if np.any(s_in & d_in):
                return False
        return True

    def _commit_to_msg(self, c, sel) -> OTWpntArray:
        wpnts = OTWpntArray()
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"
        xy = c['xy'][sel]
        s_mod = c['s_mod'][sel]
        d = c['d'][sel]
        v = c['v'][sel]
        psi = c['psi'][sel]
        kappa = c['kappa'][sel]
        for i in range(len(xy)):
            wpnts.wpnts.append(
                self.xyv_to_wpnts(x=xy[i, 0], y=xy[i, 1], s=float(s_mod[i]), d=float(d[i]),
                                  v=float(v[i]), psi=float(psi[i]), kappa=float(kappa[i]),
                                  wpnts=wpnts))
        return wpnts

    def _commit_markers(self, c, sel) -> MarkerArray:
        """Single BLUE line for the committed path (distinct from the green fresh-selection)."""
        if not self._emit_markers:
            return MarkerArray()
        mrks = MarkerArray()
        del_mrk = Marker()
        del_mrk.header.frame_id = "map"
        del_mrk.action = Marker.DELETEALL
        mrks.markers.append(del_mrk)
        xy = c['xy'][sel]
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.ns = "avoidance_committed"
        mrk.id = 0
        mrk.type = Marker.LINE_STRIP
        mrk.action = Marker.ADD
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 0.10
        mrk.color.r, mrk.color.g, mrk.color.b, mrk.color.a = 0.0, 0.6, 1.0, 1.0
        mrk.points = [Point(x=float(xy[i, 0]), y=float(xy[i, 1]), z=0.0) for i in range(len(xy))]
        mrks.markers.append(mrk)
        return mrks

    def _free_mask(self, xy: np.ndarray) -> Optional[np.ndarray]:
        """Vectorised GridFilter.is_point_inside(): True where the point is in the eroded free area.

        Same pixel convention as GridFilter.world_to_pixel()/is_point_inside() (row index = y, no
        vertical flip). Returns None when no map has been received yet, so callers can fall back.
        """
        f = self.map_filter
        img = getattr(f, "eroded_image", None)
        if img is None or f.resolution is None or f.origin is None:
            return None
        px = ((xy[:, 0] - f.origin[0]) / f.resolution).astype(int)
        py = ((xy[:, 1] - f.origin[1]) / f.resolution).astype(int)
        ok = (px >= 0) & (py >= 0) & (px < img.shape[1]) & (py < img.shape[0])
        free = np.zeros(px.shape, dtype=bool)
        free[ok] = img[py[ok], px[ok]] == 255
        return free

    def _grid_corridor(self, s_query: float) -> Optional[Tuple[float, float]]:
        """Free lateral extent [d_lo, d_hi] (car-centre limits) at arc length s, MEASURED in the
        eroded occupancy grid rather than read from the waypoints' d_left/d_right.

        Only the CONTIGUOUS free run containing the raceline is kept, so free space that belongs to
        another part of the track further out cannot widen the corridor. The erosion already reserves
        the clearance a car-centre point needs, so only wall_margin is taken off on top of it.
        Returns None when no map is loaded (callers fall back to the waypoint bounds).
        """
        if self.gb_max_s is None or getattr(self, "converter", None) is None:
            return None
        d_scan = np.arange(-self.grid_scan_max, self.grid_scan_max + 1e-9, self.grid_scan_step)
        s_arr = np.full(d_scan.shape, float(s_query) % self.gb_max_s)
        resp = self.converter.get_cartesian(s_arr, d_scan)
        xy = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)
        free = self._free_mask(xy)
        if free is None or not free.any():
            return None
        i0 = int(np.argmin(np.abs(d_scan)))
        if not free[i0]:                       # raceline itself reads blocked -> nearest free sample
            cand = np.flatnonzero(free)
            i0 = int(cand[np.argmin(np.abs(d_scan[cand]))])
        lo_i = hi_i = i0
        while lo_i > 0 and free[lo_i - 1]:
            lo_i -= 1
        while hi_i < free.size - 1 and free[hi_i + 1]:
            hi_i += 1
        d_lo = float(d_scan[lo_i]) + self.wall_margin
        d_hi = float(d_scan[hi_i]) - self.wall_margin
        if d_hi < d_lo:                        # narrower than 2*wall_margin -> collapse to its middle
            d_lo = d_hi = 0.5 * (float(d_scan[lo_i]) + float(d_scan[hi_i]))
        return d_lo, d_hi

    def _grid_is_authority(self) -> bool:
        """True when the occupancy grid both replaces the waypoint corridor AND is checked per path
        point -- i.e. the per-point grid test fully subsumes the waypoint-bounds test."""
        return (self.trust_grid_bounds and self.use_grid_check and
                getattr(self.map_filter, "eroded_image", None) is not None)

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
