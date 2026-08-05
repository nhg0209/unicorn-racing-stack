#!/usr/bin/env python3
import math
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
from std_msgs.msg import Float32, Bool, Float32MultiArray
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
        self.kernel_size = 3         # GridFilter erosion KERNEL (cells), NOT the erosion depth:
                                     # cv2.erode eats floor(k/2) cells, so 3 reserves ONE cell
                                     # (0.05 m at the maps' resolution). 8 ate ~0.2 m and rejected
                                     # the raceline. See _grid_corridor.
        # TWO IMAGES, TWO JOBS. The kernel above is the SAMPLING image: it reserves one cell, which
        # is what keeps narrow sections samplable at all -- raise it and the corridor collapses and
        # the planner concedes TRAILING where it could still pass. But one cell is 0.05 m, and a
        # path point is a CAR CENTRE: with trust_grid_bounds the waypoint corridor test is skipped
        # entirely (bound_ok = ones, see do_spline) and every published point is then vetted only
        # against that 0.05 m reserve. A 0.30 m car driven down such a path has 0.10 m of itself
        # inside the wall.
        #
        # So keep sampling generous and make PUBLISHING body-safe with a second image eroded by
        # half a car: floor(7/2) = 3 cells = 0.15 m = width_car/2. A point free in this image is a
        # centre the whole car fits at (L-inf, so diagonals get 0.21 m -- conservative, never less).
        # Measured on the shipped maps: the raceline is free at k=7 at 368/368 stations on ifac and
        # 766/766 on f, and the narrowest k=7 free span on ifac is 0.45 m, so this floor does not
        # take away a line the car can actually drive. See _path_body_unsafe.
        self.body_kernel_size = 7    # [cells] erosion of the BODY-SAFETY image; floor(k/2)*0.05 m
                                     # must stay >= width_car/2. 1 = no erosion = floor disabled.
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
        # LATERAL half of the keep-out. Split from safety_margin so the two axes obs_margin
        # conflates can move independently: the longitudinal one is bounded by half the car's
        # LENGTH (0.29 m) and the lateral one by the SM's accept lines, the clear gate and the
        # tracker's position error. Equal to safety_margin here, which makes the split a pure
        # refactor -- every published path is bit-identical to before it.
        self.safety_margin_d = 0.16
        self.static_near_zero_mps = 0.15  # speed band for the near-stationary fallback
        self.static_promote_sec = 0.5     # how long it must hold before the fallback is believed
        self.static_demote_mps = 0.35     # clearly-moving band that ends the belief
        self.static_demote_sec = 0.3      # how long that must hold before demoting
        self._near_zero_since = {}        # obstacle id -> time it first read ~0 speed
        self._moving_since = {}           # obstacle id -> time it first read clearly moving
        self._promoted = set()            # ids currently believed static
                                     # obs_margin = half_car(0.15)+safety must cover the sim ego collision
                                     # radius (0.29 m = half car LENGTH); 0.16 -> 0.31 clears it (+0.02).
        self.wall_margin = 0.05      # [m] clearance to the wall the candidate may reach (corridor)
        # SQUEEZE PASS. When every candidate is rejected at the full design margins, retry the same
        # geometry at reduced ones before conceding TRAILING. This is not a loosening of the design
        # margins -- it is the answer to sections where they are arithmetically unavailable: ifac
        # narrows below 1.20 m, and a box in the middle of that needs half_car + safety_margin per
        # side plus its own half-width, which does not fit. Without this the planner's only output
        # there is feasible=False, and behind a STATIONARY box the trailing gap PID's fixed point is
        # v = 0 -- so "stop forever, 1.5 m short" becomes the correct behaviour. Gated on speed
        # because trading clearance for motion only makes sense where a mis-clearance is survivable,
        # and marked on the published path (ot_line = "squeeze") so the SM caps the speed it is
        # driven at rather than the planner silently issuing a normal-looking path.
        #
        # WHAT THE SQUEEZE MAY NOT TOUCH: the BODY floor (body_kernel_size / _path_body_unsafe).
        # The two numbers below are clearances the car could give up and still fit -- reserve on top
        # of its own width. The body floor is the width itself, so relaxing it does not buy a
        # tighter pass, it buys a collision. Structurally it cannot be relaxed either: the squeeze
        # re-enters do_spline with new safety_margin/wall_margin scalars and nothing else, while the
        # body image is eroded once from body_kernel_size and read by the same check on every
        # candidate of every pass. A section too narrow for the body floor has no answer here, and
        # TRAILING is the correct one.
        self.squeeze_enable = True
        self.squeeze_steps = 2            # reduced-margin attempts between the design value and the floor
        self.squeeze_safety_floor_m = 0.05  # [m] tightest obstacle clearance the pass may ask for
        self.squeeze_wall_floor_m = 0.08    # [m] tightest wall reserve the pass may ask for
        self.squeeze_max_speed_mps = 3.0    # [m/s] above this, "no candidate" still means TRAILING
        self.relax_hold_s = 2.0             # [s] a deadlock relax request forces the pass this long
        self._relax_until = 0.0             # wall time the current relax request expires
        # [m] how close in s two apexes may be before the second is merged into the first. It was
        # a literal 0.4; promoted because it interacts with how far ahead the first knot can sit,
        # and that combination is what emptied the path beside a box (see the knot loop).
        self.knot_merge_s_m = 0.4
        self.shift_min = 1.0         # [m] min arc length over which the lateral maneuver completes
        self.shift_buffer = 0.5      # [m] finish the shift this far before the obstacle near-edge
        # Peak curvature of the maneuver scales as amplitude/length^2 and the speed it can be
        # driven at as sqrt(a_lat/kappa), so the ramp lengths are a speed knob, not just a comfort
        # one. Kept equal to static_avoidance_params.yaml so `ros2 run` does not silently plan a
        # sharper maneuver than the documented configuration.
        self.ramp_len = 4.5          # [m] gentle entry-ramp length (raceline -> apex)
        self.hold_after = 0.5        # [m] (unused in apex-loaded profile; kept for param compatibility)
        self.return_len = 4.5        # [m] gentle exit-ramp length (apex -> raceline)
        # Floor for the ADAPTIVE shortening of the two above (see _fit_ramp). A ramp is shortened
        # only where the corridor will not accept the offset over its full length; below this the
        # curvature cost (A/L^2) stops being worth the feasibility it buys.
        self.ramp_len_min_m = 2.5    # [m]
        self.apex_bulge = 0.05       # [m] extra offset at the box CENTRE (apex) beyond the clearance
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
        # genuine keep-out violation resumes planning.
        #
        # The clearance is EVALUATED every cycle, unconditionally. It used to be evaluated only
        # while |cur_d| < clear_max_cur_d, on the reasoning that a maneuver in progress must not be
        # abandoned mid-hump — but the documented steady-state tracking error is ~0.5 m, three
        # times that threshold, so on the swapped line the gate simply never ran and the planner
        # re-avoided an obstacle the global line already cleared. It also cost nothing to protect:
        # by the time control reaches the gate the committed-path branch above has already
        # returned, so there is no maneuver left to abandon, and the gate's own predicate says the
        # FOLLOWED line clears every box ahead — returning to it is safe from anywhere.
        # clear_max_cur_d now decides only whether standing down would CANCEL an excursion: off
        # the raceline the full entry margin is required, so a cancel can never ride on a latch.
        self.clear_gate_enable = True
        self.clear_margin_m = 0.10   # [m] extra beyond half_car for the raceline-CLEAR trigger
        self.reframe_warn_m = 0.05   # [m] warn when incoming obstacle d must be re-anchored by more
        self.clear_hyst_m = 0.03     # [m] extra clearance required to ENTER the idle state
        self.clear_max_cur_d = 0.15  # [m] above this |cur_d| a stand-down CANCELS an excursion
        self.clear_latch_ttl_s = 10.0  # [s] how long a per-obstacle clear latch survives unseen
        self._line_clear = False     # aggregate idle state (logging / "all boxes ahead are clear")
        # Per-OBSTACLE-ID clear latch: id -> wall time it last passed. Kept while the obstacle is
        # out of the lookahead, so a box that leaves and re-enters the horizon does not have to
        # re-earn idle at the entry margin every time — that reset is what let a cleared obstacle
        # re-trigger a maneuver once per approach.
        self._clear_latch = {}
        # THE clearance definition, fed by static_reopt from the line it actually publishes.
        self._clr_feed = []          # [(x, y, r, clearance)] from /static_reopt/clearance
        self._clr_feed_t = -1e18     # arrival time; staleness is "how long since I last heard"
        self.clearance_feed_ttl_s = 3.0
        self.clearance_match_m = 0.50

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
        # [m] deviation at which the committed path's ENTRY is re-anchored onto the car (it is no
        # longer a drop -- see _reanchor_commit). Must sit ABOVE the controller's steady-state
        # tracking error (~0.5 m, documented in state_machine_params.yaml recovery_exit_d_m), or
        # the correction fires continuously on error the controller has not actually failed at.
        self.commit_dev_max = 0.6
        self.commit_reanchor_len_m = 2.0   # [m] arc length the entry correction is faded out over
        self.commit_reanchor_max_m = 1.0   # [m] beyond this it is not an entry fix -> full re-plan
        self.preramp_len_m = 3.0     # [m] decay the car's current d to the raceline within this
                                     #     much of the path BEFORE the hump's entry ramp
        self.commit_obs_ds = 0.75    # [m] drop the commit if the triggering box's s drifts this far
        self.commit_obs_dd = 0.40    # [m] ... or its d drifts this far (re-plan the apex around it)
        self._committed = None       # cached selected path (frenet + cartesian arrays) or None

        # Static params
        self.declare_parameters(namespace='', parameters=[('from_bag', False), ('measure', False)])
        self.from_bag = self.get_parameter('from_bag').get_parameter_value().bool_value
        self.measuring = self.get_parameter('measure').get_parameter_value().bool_value

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)
        # Second, more strongly eroded view of the SAME map — the body-safety floor (see
        # body_kernel_size). Its own subscription because GridFilter owns the erosion: one image
        # cannot serve two kernels, and the two are deliberately different.
        self.body_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.body_filter.set_erosion_kernel_size(self.body_kernel_size)

        self.declare_all_parameters()
        # Sync members from loaded params (yaml/defaults), then register live-reconfigure callback.
        self.dyn_param_cb(self.get_parameters([
            'kernel_size', 'body_kernel_size',
            'lookahead_min', 'lookahead_k', 'n_d_samples', 'sample_gaps', 'kappa_max',
            'kappa_add_max', 'kappa_abs_max', 'a_lat_max', 'a_long_max', 'a_long_accel',
            'safety_margin', 'safety_margin_d', 'static_near_zero_mps', 'static_promote_sec',
            'static_demote_mps', 'static_demote_sec',
            'wall_margin', 'knot_merge_s_m', 'shift_min', 'shift_buffer', 'ramp_len', 'hold_after',
            'return_len', 'ramp_len_min_m',
            'apex_bulge', 'max_weave', 'width_car', 'tail_m', 'w_d', 'w_k', 'w_c', 'w_obs', 'obs_sigma',
            'use_grid_check', 'trust_grid_bounds', 'grid_scan_max', 'grid_scan_step', 'bounds_warn_m',
            'clear_gate_enable', 'clear_hyst_m', 'clear_max_cur_d', 'clear_margin_m', 'reframe_warn_m',
            'clear_latch_ttl_s', 'squeeze_enable', 'squeeze_steps', 'squeeze_safety_floor_m',
            'squeeze_wall_floor_m', 'squeeze_max_speed_mps', 'relax_hold_s',
            'commit_enable', 'commit_dev_max', 'commit_obs_ds', 'commit_obs_dd',
            'commit_reanchor_len_m', 'commit_reanchor_max_m', 'preramp_len_m',
        ]))
        self.add_on_set_parameters_callback(self.dyn_param_cb)

        # Subscribers
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacles_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)
        # Deadlock recovery request from the state machine: the car has been stopped behind a
        # static obstacle this planner reported infeasible. Absolute name, deliberately NOT
        # remapped per-planner (mirrors /planner/avoidance/static_feasible).
        self.create_subscription(Bool, "/planner/avoidance/relax", self.relax_cb, 10)
        # See _clears_obstacle for why this node stopped deriving its own notion of "clear".
        self.create_subscription(Float32MultiArray, "/static_reopt/clearance",
                                 self.clearance_cb, 1)

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
        self.declare_parameter('body_kernel_size', 7,
                               intd(1, 31, "erosion kernel of the BODY-SAFETY image [cells]: "
                                           "floor(k/2)*resolution must cover width_car/2. 1 = off"))
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
        self.declare_parameter('safety_margin_d', 0.16, dbl(0.0, 1.0, "LATERAL clearance around the obstacle box [m]"))
        self.declare_parameter('static_near_zero_mps', 0.15,
                               dbl(0.0, 1.0, "speed band for the near-stationary fallback [m/s]"))
        self.declare_parameter('static_promote_sec', 0.5,
                               dbl(0.0, 5.0, "how long ~0 speed must hold before believing it [s]"))
        self.declare_parameter('static_demote_mps', 0.35,
                               dbl(0.0, 2.0, "clearly-moving band that ends the static belief [m/s]"))
        self.declare_parameter('static_demote_sec', 0.3,
                               dbl(0.0, 5.0, "how long that must hold before demoting [s]"))
        self.declare_parameter('wall_margin', 0.05, dbl(0.0, 1.0, "clearance to wall a candidate may reach [m]"))
        self.declare_parameter('squeeze_enable', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Retry an all-rejected plan at reduced margins before conceding TRAILING"))
        self.declare_parameter('squeeze_steps', 2,
                               intd(1, 5, "reduced-margin attempts between the design margin and the floor"))
        self.declare_parameter('squeeze_safety_floor_m', 0.05,
                               dbl(0.0, 0.5, "tightest obstacle clearance the squeeze pass may ask for [m]"))
        self.declare_parameter('squeeze_wall_floor_m', 0.08,
                               dbl(0.0, 0.5, "tightest wall reserve the squeeze pass may ask for [m]"))
        self.declare_parameter('squeeze_max_speed_mps', 3.0,
                               dbl(0.0, 10.0, "above this speed, no feasible candidate still means TRAILING [m/s]"))
        self.declare_parameter('relax_hold_s', 2.0,
                               dbl(0.0, 30.0, "how long a deadlock relax request forces the squeeze pass [s]"))
        self.declare_parameter('knot_merge_s_m', 0.4,
                               dbl(0.0, 5.0, "merge a second apex closer than this in s [m]"))
        self.declare_parameter('shift_min', 1.0, dbl(0.3, 10.0, "min arc length for the lateral maneuver [m]"))
        self.declare_parameter('shift_buffer', 0.5, dbl(0.0, 5.0, "finish the shift this far before the obstacle [m]"))
        self.declare_parameter('ramp_len', 4.5, dbl(0.5, 15.0, "ramp length onto the offset [m]"))
        self.declare_parameter('hold_after', 0.5, dbl(0.0, 5.0, "hold the offset past the obstacle far-edge [m]"))
        self.declare_parameter('return_len', 4.5, dbl(0.5, 10.0, "ramp length back to the raceline [m]"))
        self.declare_parameter('ramp_len_min_m', 2.5,
                               dbl(0.5, 10.0, "floor for the adaptive ramp shortening [m]"))
        self.declare_parameter('apex_bulge', 0.05, dbl(0.0, 1.0, "extra apex offset beyond clearance: higher=wider avoidance [m]"))
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
        self.declare_parameter('reframe_warn_m', 0.05,
                               dbl(0.0, 1.0, "warn when obstacle d is re-anchored by more than this [m]"))
        self.declare_parameter('clear_margin_m', 0.10,
                               dbl(0.0, 0.5, "extra clearance beyond half_car for the raceline-clear trigger [m]"))
        self.declare_parameter('clear_gate_enable', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Stay idle when the current raceline already clears every obstacle ahead"))
        self.declare_parameter('clear_hyst_m', 0.03, dbl(0.0, 0.5, "extra clearance to ENTER the idle state [m]"))
        self.declare_parameter('clear_max_cur_d', 0.15,
                               dbl(0.0, 1.0, "above this |cur_d| a stand-down cancels an excursion, "
                                             "so it needs the full entry margin (never a latch) [m]"))
        self.declare_parameter('clear_latch_ttl_s', 10.0,
                               dbl(0.0, 120.0, "how long a per-obstacle clear latch survives while "
                                               "that obstacle is out of the lookahead [s]"))
        self.declare_parameter('commit_enable', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description="Commit to a chosen evasion path and reuse it (temporal consistency) instead of re-solving every cycle"))
        self.declare_parameter('commit_dev_max', 0.6, dbl(0.05, 2.0, "re-anchor the committed path's entry if the car deviates this far from it [m]"))
        self.declare_parameter('commit_reanchor_len_m', 2.0, dbl(0.5, 10.0, "arc length the entry re-anchor is faded out over [m]"))
        self.declare_parameter('commit_reanchor_max_m', 1.0, dbl(0.1, 3.0, "deviation beyond which a full re-plan is taken instead [m]"))
        self.declare_parameter('preramp_len_m', 3.0, dbl(0.0, 15.0, "decay the car's current d to the raceline within this much of the path before the hump [m]"))
        self.declare_parameter('commit_obs_ds', 0.75, dbl(0.05, 5.0, "drop the commit if the triggering obstacle s drifts this far [m]"))
        self.declare_parameter('commit_obs_dd', 0.40, dbl(0.05, 2.0, "drop the commit if the triggering obstacle d drifts this far [m]"))

    def dyn_param_cb(self, params: List[Parameter]):
        for p in params:
            n = p.name
            if n == 'kernel_size':
                self.kernel_size = int(p.value)
                self.map_filter.set_erosion_kernel_size(self.kernel_size)
            elif n == 'body_kernel_size':
                self.body_kernel_size = int(p.value)
                self.body_filter.set_erosion_kernel_size(self.body_kernel_size)
                res = getattr(self.body_filter, "resolution", None) or 0.05
                reserve = (self.body_kernel_size // 2) * res
                if reserve + 1e-9 < 0.5 * self.width_car:
                    self.get_logger().warn(
                        f"[{self.name}] body_kernel_size={self.body_kernel_size} reserves "
                        f"{reserve:.3f} m at {res:.3f} m/cell, under half a car "
                        f"({0.5 * self.width_car:.3f} m): published paths may put the car body "
                        f"into a wall. Needs k >= {int(2 * (0.5 * self.width_car / res)) + 1}.")
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
            elif n == 'safety_margin_d':
                self.safety_margin_d = float(p.value)
            elif n == 'wall_margin':
                self.wall_margin = float(p.value)
            elif n == 'knot_merge_s_m':
                self.knot_merge_s_m = float(p.value)
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
            elif n == 'ramp_len_min_m':
                self.ramp_len_min_m = float(p.value)
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
            elif n == 'squeeze_enable':
                self.squeeze_enable = bool(p.value)
            elif n == 'squeeze_steps':
                self.squeeze_steps = int(p.value)
            elif n == 'squeeze_safety_floor_m':
                self.squeeze_safety_floor_m = float(p.value)
            elif n == 'squeeze_wall_floor_m':
                self.squeeze_wall_floor_m = float(p.value)
            elif n == 'squeeze_max_speed_mps':
                self.squeeze_max_speed_mps = float(p.value)
            elif n == 'relax_hold_s':
                self.relax_hold_s = float(p.value)
            elif n == 'clear_gate_enable':
                self.clear_gate_enable = bool(p.value)
            elif n == 'clear_margin_m':
                self.clear_margin_m = float(p.value)
            elif n == 'reframe_warn_m':
                self.reframe_warn_m = float(p.value)
            elif n == 'clear_hyst_m':
                self.clear_hyst_m = float(p.value)
            elif n == 'clear_max_cur_d':
                self.clear_max_cur_d = float(p.value)
            elif n == 'clear_latch_ttl_s':
                self.clear_latch_ttl_s = float(p.value)
            elif n == 'commit_enable':
                self.commit_enable = bool(p.value)
                if not self.commit_enable:
                    self._committed = None
            elif n == 'commit_dev_max':
                self.commit_dev_max = float(p.value)
            elif n == 'commit_reanchor_len_m':
                self.commit_reanchor_len_m = float(p.value)
            elif n == 'commit_reanchor_max_m':
                self.commit_reanchor_max_m = float(p.value)
            elif n == 'preramp_len_m':
                self.preramp_len_m = float(p.value)
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
        self.obstacles = self._reframe_obstacles(data.obstacles)
        self._track_near_zero(self.obstacles)

    def _reframe_obstacles(self, obstacles):
        """Re-express each obstacle's (s,d) in THIS node's frenet frame, from its map (x_m,y_m).

        Everything else in this planner is already in the current line's frame: gb_cb rebuilds the
        converter whenever static_reopt swaps the global line, the corridor is measured through that
        converter, and cur_s/cur_d come from the frenet republisher on the same /global_waypoints.
        The obstacles were the one input that was not — they arrived carrying whatever (s,d) the
        tracker computed in ITS frame, and that frame is a different node's copy of the line.

        When the two disagree the planner places the obstacle at the wrong lateral position in its
        own frame, so it dodges something that is not there. That has two visible outcomes, both
        reported on the car: an obstacle the re-optimized line already clears reads as on-path (the
        clear gate never idles, every lap re-avoids), and the evasion is solved for a phantom offset
        — the line is already displaced by its own hump, so a dodge computed against a wrong
        obstacle d can put the terminal offset into the wall side of the corridor.

        x_m,y_m is frame-independent, so this is correct by construction regardless of upstream.
        Box extents are preserved as offsets from the centre rather than rebuilt from `size`, so the
        longitudinal/lateral footprint the downstream checks see is unchanged.
        """
        conv = getattr(self, "converter", None)
        if conv is None or not obstacles:
            return obstacles
        worst = 0.0
        for o in obstacles:
            if o.x_m == 0.0 and o.y_m == 0.0:
                continue                       # no map position to re-anchor from
            try:
                fr = conv.get_frenet(np.array([o.x_m]), np.array([o.y_m]))
                s_new, d_new = float(fr[0, 0]), float(fr[1, 0])
            except Exception:
                continue
            half = max(o.size, 0.05) * 0.5
            ds_b = (o.s_center - o.s_start) if o.s_end != o.s_start else half
            ds_f = (o.s_end - o.s_center) if o.s_end != o.s_start else half
            dd_r = (o.d_center - o.d_right) if o.d_left != o.d_right else half
            dd_l = (o.d_left - o.d_center) if o.d_left != o.d_right else half
            worst = max(worst, abs(d_new - o.d_center))
            o.s_center, o.d_center = s_new, d_new
            o.s_start, o.s_end = s_new - ds_b, s_new + ds_f
            o.d_right, o.d_left = d_new - dd_r, d_new + dd_l
        if worst > self.reframe_warn_m:
            self.get_logger().warning(
                f"[{self.name}] obstacle (s,d) arrived up to {worst:.2f} m off this node's frenet "
                f"frame and was re-anchored from (x_m,y_m). Upstream tracking is not re-projecting "
                f"on a static_reopt line swap — planning would have dodged a phantom offset.",
                throttle_duration_sec=5.0)
        return obstacles

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

    def relax_cb(self, msg: Bool):
        """The state machine has certified a static TRAILING deadlock: stopped behind an obstacle
        this planner reported infeasible, for longer than its timeout.

        That is a stronger statement than anything this node can observe on its own -- it knows
        only that no candidate passed, not that the consequence was a standstill -- so it overrides
        both squeeze gates for relax_hold_s: the enable flag (an operator may have turned the pass
        off without knowing it is the only way out of this section) and the speed gate (whose whole
        purpose is to establish that a mis-clearance would be survivable, which being stationary
        establishes far better than any threshold).

        The committed path is dropped: it is what the car is currently stuck behind.
        """
        if not msg.data:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self._relax_until = now + self.relax_hold_s
        self._committed = None
        self.get_logger().warn(
            f"[{self.name}] relax requested (static TRAILING deadlock) -> forcing a reduced-margin "
            f"retry for {self.relax_hold_s:.1f} s", throttle_duration_sec=1.0)

    def _relax_active(self) -> bool:
        return (self.get_clock().now().nanoseconds * 1e-9) < self._relax_until

    def _gate_samples(self, obs_ahead, anchor, corridor, obs_margin_d: float,
                      obs_margin_s: float = None) -> List[float]:
        """Terminal offsets aimed at the lanes left free by ALL boxes overlapping `anchor` in s.

        The gap sampling above measures its two free intervals against the anchor box alone. With a
        second box at overlapping s that is not enough: its keep-out cuts one or both of those
        intervals, so a uniform linspace across them puts samples inside it and hits the lane
        BETWEEN the two boxes only by luck. On a cluster the free lane is narrow and the luck runs
        out -- which is the reported "every candidate rejected" with a passable gap present.

        Subtract every overlapping keep-out from the corridor and return, for each surviving
        sub-interval, its midpoint and the edge nearer the raceline. Bounded to two samples per
        overlapping obstacle, and only sub-intervals a car actually fits through are offered.
        """
        try:
            d_lo, d_hi = corridor
            L = self.gb_max_s
            m_s = obs_margin_d if obs_margin_s is None else obs_margin_s
            a_lo = (anchor.s_start - m_s - self.cur_s) % L
            a_hi = (anchor.s_end + m_s - self.cur_s) % L
            free = [(d_lo, d_hi)]
            n_over = 0
            for o in obs_ahead:
                o_lo = (o.s_start - m_s - self.cur_s) % L
                o_hi = (o.s_end + m_s - self.cur_s) % L
                if o is not anchor and (o_lo > a_hi or o_hi < a_lo):
                    continue                            # disjoint in s -> its keep-out is elsewhere
                if o is not anchor:
                    n_over += 1                         # budget counts the EXTRA boxes only...
                cut_lo = min(o.d_right, o.d_left) - obs_margin_d
                cut_hi = max(o.d_right, o.d_left) + obs_margin_d
                nxt = []
                for lo, hi in free:
                    if cut_hi <= lo or cut_lo >= hi:    # cut misses this interval
                        nxt.append((lo, hi))
                        continue
                    if lo < cut_lo:
                        nxt.append((lo, cut_lo))
                    if cut_hi < hi:
                        nxt.append((cut_hi, hi))
                free = nxt                              # ...but the anchor IS subtracted, or the
                                                        # lane BETWEEN the boxes never appears
            if not n_over:
                return []                               # single box: the linspace gaps already cover it
            # widest-first is the wrong order here: a wide lane already gets linspace coverage, the
            # narrow gate between two boxes is the one nothing is aiming at. Offer the lanes
            # NEAREST the raceline, which is also what the cost function would pick.
            free = [(lo, hi) for lo, hi in free if (hi - lo) > 1e-6]
            free.sort(key=lambda iv: abs(0.5 * (iv[0] + iv[1])))
            out = []
            for lo, hi in free[:max(1, 2 * n_over)]:
                mid = 0.5 * (lo + hi)
                out.append(mid)
                out.append(lo if abs(lo) < abs(hi) else hi)     # the edge nearer the raceline
                if len(out) >= 2 * n_over:
                    break
            return [float(v) for v in out]
        except Exception:                                # sampling extra candidates is never critical
            return []

    @staticmethod
    def _line_clears_obstacle(o, need: float) -> bool:
        """Does the FOLLOWED line (d = 0 in this node's frenet frame) pass `o` by `need`?

        The keep-out interval around the box, widened by `need` on both sides, simply must not
        contain d = 0. This is a LATERAL distance in this node's own frenet frame, and it is the
        FALLBACK now rather than the definition -- see _clears_obstacle.
        """
        return ((min(o.d_right, o.d_left) - need) > 0.0
                or (max(o.d_right, o.d_left) + need) < 0.0)

    def clearance_cb(self, msg: Float32MultiArray):
        d = list(msg.data)
        self._clr_feed = [tuple(d[i:i + 4]) for i in range(0, len(d) - 3, 4)]
        self._clr_feed_t = self.get_clock().now().nanoseconds * 1e-9

    def _published_clearance(self, o):
        """The measured clearance of the FOLLOWED line past `o`, or None if there is none fresh.

        Matched by MAP POSITION: this node carries tracker ids while static_reopt carries the
        static layer's marker ids, and the two id spaces are unrelated."""
        feed = getattr(self, "_clr_feed", None)
        if not feed:
            return None
        if (self.get_clock().now().nanoseconds * 1e-9
                - getattr(self, "_clr_feed_t", -1e18)) > getattr(self, "clearance_feed_ttl_s", 3.0):
            return None
        ox, oy = float(getattr(o, "x_m", float("nan"))), float(getattr(o, "y_m", float("nan")))
        if ox != ox or oy != oy:
            return None
        best, best_d = None, getattr(self, "clearance_match_m", 0.50)
        for (x, y, _r, clr) in feed:
            dd = float(np.hypot(x - ox, y - oy))
            if dd <= best_d:
                best, best_d = float(clr), dd
        return best

    def _clears_obstacle(self, o, need: float) -> bool:
        """Does the FOLLOWED line pass `o` by `need`? ONE definition, with a fail-closed fallback.

        The definition is static_reopt's: the 2-D distance from the published line to the obstacle
        EDGE. This node used to derive its own LATERAL distance in its own frenet frame, the state
        machine derived a third in ITS frame, and all three were compared against thresholds only
        2 cm apart -- so a curve was enough for one layer to believe the line clears a box while
        another believed it does not, and in a fail-closed chain that becomes a permanent TRAILING.

        With no fresh measurement it falls back to the lateral test, which is what it has always
        done: the feed may only ever REPLACE a measurement, never remove one.
        """
        clr = self._published_clearance(o)
        if clr is None:
            return self._line_clears_obstacle(o, need)
        return clr >= need

    def _bulge_away_from(self, da: float, o) -> float:
        """Signed apex_bulge that pushes the peak FURTHER FROM THE OBSTACLE.

        It used to be `sign(da) * apex_bulge` -- further from d = 0, which is only the same thing
        while the obstacle straddles the raceline. It does not while the obstacle sits to one side
        of the line, and that is the normal case on a swapped static_reopt line, where the followed
        line is itself displaced and every box reads off-centre in d.

        Concretely: a box at d_center = +0.50 passed on the RIGHT gives da = +0.05 (its keep-out
        edge). sign(da) is positive, so the bulge moved the peak to +0.15 -- 0.10 m TOWARD the
        obstacle and inside the keep-out obs_ok then tests it against. The candidate is rejected,
        and since the bulge is applied to every candidate, so is every other one: feasible=False on
        geometry that was fine before the bulge was added to it.

        Degenerate case (the apex sits exactly on the obstacle's centreline) keeps the old sign;
        there is no "away" to pick and obs_ok will reject it either way.
        """
        d_c = float(getattr(o, "d_center", 0.0))
        if abs(da - d_c) < 1e-6:
            return float(np.sign(da)) * self.apex_bulge
        return math.copysign(self.apex_bulge, da - d_c)

    def _anchor_car_idx(self, s_idx: int, gb_wpnts, wpnt_dist: float, search_m: float = 3.0) -> int:
        """Re-anchor the s-derived grid start to the station NEAREST THE CAR.

        Same idiom, and the same reason, as the state machine's anchor_gb_index. cur_s comes from
        the frenet republisher on /global_waypoints, while this grid is indexed into
        /global_waypoints_scaled, which sector_tuner only re-publishes on its 0.5 s timer. After a
        static_reopt line swap the two disagree for up to that long: `cur_s / wpnt_dist` then names
        the right NUMBER on the wrong parameterisation, and the whole planned grid -- entry ramp,
        apex station, corridor lookups -- slides along the track away from the car.

        Bounded to +-search_m for the same reason as there: a free nearest-point search over the
        closed loop snaps to the wrong branch wherever the raceline runs close to itself.
        """
        if self.cur_x is None or self.cur_y is None or not gb_wpnts:
            return s_idx
        n = self.gb_max_idx
        k = max(1, int(search_m / max(wpnt_dist, 1e-3)))
        idx = (s_idx + np.arange(-k, k + 1)) % n
        dx = np.fromiter((gb_wpnts[j].x_m - self.cur_x for j in idx), float, len(idx))
        dy = np.fromiter((gb_wpnts[j].y_m - self.cur_y for j in idx), float, len(idx))
        return int(idx[int(np.argmin(dx * dx + dy * dy))])

    def _squeeze_schedule(self):
        """(safety_margin, wall_margin) pairs to retry an all-rejected plan with, widest first.

        Empty means "do not squeeze", which is the normal answer at racing speed: trading clearance
        for motion is only defensible where a mis-clearance is survivable, and above
        squeeze_max_speed_mps the honest response to "no candidate" is still TRAILING. It is also
        empty when the design margins are already at or below the floors -- there is nothing left to
        give -- so the caller falls through to the existing diagnostic + feasible=False.

        A live relax request (see relax_cb) overrides the enable flag and the speed gate, but never
        the floors: the SM has established that the car is stopped and stuck, which is the thing
        those two gates were approximating.

        The schedule interpolates linearly to the floor in squeeze_steps attempts rather than
        jumping straight to it, so a section that only needs a couple of centimetres is driven with
        the couple of centimetres it needs.
        """
        forced = self._relax_active()
        if not self.squeeze_enable and not forced:
            return []
        v = abs(self.cur_vs) if self.cur_vs is not None else 0.0
        if v >= self.squeeze_max_speed_mps and not forced:
            return []
        s0, w0 = self.safety_margin, self.wall_margin
        s1 = min(self.squeeze_safety_floor_m, s0)
        w1 = min(self.squeeze_wall_floor_m, w0)
        if (s0 - s1) < 1e-6 and (w0 - w1) < 1e-6:
            return []                          # already at (or below) the floor: nothing to give
        steps = max(1, int(self.squeeze_steps))
        return [(s0 + (i / steps) * (s1 - s0), w0 + (i / steps) * (w1 - w0))
                for i in range(1, steps + 1)]

    def _eval_clear_gate(self, obs_ahead, half_car: float) -> bool:
        """Does the FOLLOWED global line clear every obstacle ahead? (updates the per-id latches)

        The test per obstacle is "the keep-out interval around the box does not contain d = 0",
        with `d` already re-anchored into this node's frenet frame, so d = 0 is the line the car
        is on. Two thresholds, and which one applies is the whole point of this method:

          stay  = half_car + clear_margin_m                  (already latched, car on the raceline)
          entry = stay + clear_hyst_m                        (fresh, or standing down off-line)

        Standing down while |cur_d| >= clear_max_cur_d CANCELS an excursion the car is already
        committed to in the physical sense, so it must be earned at `entry` — never on a latch.
        On the raceline a latched obstacle keeps `stay`, which is what stops the publish/empty
        flapping as cur_d decays through the threshold with noise (the "duplicate path" symptom,
        SM alternating OVERTAKE<->GB_TRACK at up to 20 Hz).

        The latch is per tracker id and survives the obstacle leaving the lookahead; only a REAL
        keep-out violation drops it.
        """
        now_s = self.get_clock().now().nanoseconds * 1e-9
        base = half_car + self.clear_margin_m
        cancelling = abs(self.cur_d) >= self.clear_max_cur_d
        clear = True
        for o in obs_ahead:
            oid = int(o.id)
            latched = (now_s - self._clear_latch.get(oid, -1e18)) < self.clear_latch_ttl_s
            need = base if (latched and not cancelling) else base + self.clear_hyst_m
            if self._clears_obstacle(o, need):
                self._clear_latch[oid] = now_s
            else:
                self._clear_latch.pop(oid, None)
                clear = False
        self._prune_clear_latch(now_s)
        return clear

    def _prune_clear_latch(self, now_s: float = None) -> None:
        """Drop clear latches for obstacles not seen for clear_latch_ttl_s.

        The latch deliberately survives an obstacle leaving the lookahead, so it cannot be keyed
        off the live obstacle set; a TTL is what keeps tracker ids from accumulating over a
        session. Anything this old is a track that is gone for good, and a re-detected box gets a
        fresh id from the tracker anyway."""
        if not self._clear_latch:
            return
        if now_s is None:
            now_s = self.get_clock().now().nanoseconds * 1e-9
        stale = [k for k, t in self._clear_latch.items() if (now_s - t) >= self.clear_latch_ttl_s]
        for k in stale:
            del self._clear_latch[k]

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
            near_zero = self._near_zero_static(o)
            if not (o.is_static or near_zero):
                if (not o.is_static and abs(o.vs) < self.static_near_zero_mps
                        and abs(o.vd) < self.static_near_zero_mps):
                    since = self._near_zero_since.get(o.id)
                    held = (self.get_clock().now().nanoseconds * 1e-9 - since) if since else 0.0
                    self.get_logger().info(
                        f"[{self.name}] obs id={o.id} reads ~0 speed for only {held:.2f}s "
                        f"(< {self.static_promote_sec:.2f}) -- not yet treated as static",
                        throttle_duration_sec=2.0)
                continue
            if not o.is_static and near_zero:
                self.get_logger().info(
                    f"[{self.name}] treating dynamic-flagged obs id={o.id} as static "
                    f"(vs={o.vs:.2f} vd={o.vd:.2f}; demotes above {self.static_demote_mps:.2f})",
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

    def do_spline(self, gb_wpnts, safety_margin: float = None, wall_margin: float = None,
                  squeeze: bool = False) -> Tuple[OTWpntArray, MarkerArray]:
        """Plan one static-avoidance path. `squeeze` marks a reduced-margin RETRY (see
        _squeeze_schedule); such a call returns None instead of an empty result so the caller can
        try the next step, and never publishes the feasibility verdict itself."""
        # The squeeze lowers `safety_margin`; both axes give up the SAME amount, so a squeeze after
        # the split behaves exactly as it did before it.
        safety_margin_d = self.safety_margin_d - (self.safety_margin - (
            self.safety_margin if safety_margin is None else safety_margin))
        safety_margin = self.safety_margin if safety_margin is None else safety_margin
        wall_margin = self.wall_margin if wall_margin is None else wall_margin
        wpnts = OTWpntArray()
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"
        if squeeze:
            wpnts.ot_line = "squeeze"

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
            return None if squeeze else _empty()

        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m
        half_car = self.width_car / 2.0
        # TWO AXES, NOT ONE. obs_margin was used for two different jobs: the LATERAL keep-out
        # (how far the line must pass the box) and the LONGITUDINAL one (how far the box's
        # s-interval is inflated before the lateral test is applied at all). They are bounded by
        # different things -- the lateral one by the SM's accept lines, the clear gate and the
        # tracker's position error; the longitudinal one by HALF THE CAR'S LENGTH, because a car
        # whose nose is level with the box edge is still half a length from having passed it. Tying
        # them together meant lowering the lateral margin silently spent the longitudinal one, and
        # that is what put the nose into the box.
        obs_margin_s = half_car + safety_margin         # box s-interval inflation
        obs_margin_d = half_car + safety_margin_d       # lateral keep-out half-width
        obs_margin = obs_margin_d                       # legacy name, LATERAL, for the log lines
        sample_margin = half_car + wall_margin          # how close to the wall a candidate may reach

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
        # A squeeze RETRY skips this: whether to reuse or re-plan was already decided by the
        # full-margin call that is now retrying (and which left _committed None to get here).
        if self.commit_enable and self._committed is not None and not squeeze:
            reuse = self._reuse_committed(gb_wpnts, wpnt_dist, obs_margin_d, half_car,
                                          obs_margin_s)
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
        if squeeze and not obs_ahead:
            return None
        if not obs_ahead:
            # nothing to avoid -> no avoidance path (state machine stays on the raceline).
            # The AGGREGATE flag is a statement about the boxes ahead, so it resets with them --
            # but the per-obstacle latches must NOT: an obstacle leaves the lookahead on every
            # approach, and wiping its latch there made it re-earn idle at the entry margin each
            # lap, which is one re-triggered maneuver per pass over an already-cleared box.
            self._line_clear = False
            self._committed = None
            self._prune_clear_latch()
            return _empty()

        # --- raceline-clear gate: the current global line may ALREADY avoid everything ahead ---
        # (the obstacle-aware line static_reopt swapped in). Idle then: no path -> the SM stays
        # GB_TRACK and no apex is re-recorded on top of the swapped line. Obstacle d values are
        # re-anchored into THIS node's frenet frame on arrival (_reframe_obstacles), so "the
        # keep-out interval does not contain d=0" IS the clearance test of the followed line.
        #
        # EVALUATION IS UNCONDITIONAL. It used to be skipped unless |cur_d| < clear_max_cur_d or
        # the aggregate latch was already set, to avoid abandoning a maneuver mid-hump. But the
        # steady-state tracking error is ~0.5 m -- three times that threshold -- so after a line
        # swap the gate mostly did not run at all, and every pass re-avoided a box the global line
        # already cleared. Nothing was being protected either: the committed-path branch above has
        # already returned by the time control gets here, so there is no maneuver in flight, and
        # the predicate itself certifies the followed line clears every box ahead.
        # Skipped on a squeeze RETRY: the full-margin call already ran the gate this cycle and did
        # not idle; re-running it would only re-latch on the same obstacles.
        if self.clear_gate_enable and not squeeze:
            # The trigger threshold is NOT obs_margin. obs_margin (half_car + safety_margin = 0.30)
            # is what a NEW path is designed to, and reusing it here asks "is the current line
            # planned to my full design clearance?" instead of "does the car fit past this box?".
            # That re-triggers avoidance on a line that is perfectly safe: static_reopt builds to
            # reopt_obs_margin = 0.35, so with the hysteresis the gate had 0.35 - 0.33 = 2 cm of
            # headroom, and any error past 2 cm reads the cleared obstacle as on-path. The planner
            # then has to find a candidate at the full 0.30 in a corridor the line already used up,
            # rejects every one, publishes feasible=False, and the SM falls to TRAILING behind an
            # obstacle it had already solved.
            clear = self._eval_clear_gate(obs_ahead, half_car)
            if clear and not self._line_clear:
                self.get_logger().info(
                    f"[{self.name}] raceline clears all {len(obs_ahead)} obstacle(s) ahead "
                    f"(margin {half_car + self.clear_margin_m:.2f} m, |cur_d|="
                    f"{abs(self.cur_d):.2f}) -> planner idle")
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
        n_already_clear = 0
        for o in obs_ahead:
            # An obstacle the FOLLOWED line already passes at the full keep-out needs no apex: the
            # path is welcome to stay where it is beside it. Spending a knot on one costs twice --
            # it consumes a max_weave slot a genuinely blocking box needed, and it bends the line
            # away from a raceline that was already correct there, which then gets recorded as this
            # obstacle's reactive apex and walks the re-optimized line outward every lap.
            # Deliberately tested at obs_margin, NOT at the clear gate's own (smaller) threshold:
            # obs_ok enforces obs_margin on every obstacle whether or not it got a knot, so this is
            # exactly the condition under which skipping the knot cannot make obs_ok reject.
            if self._clears_obstacle(o, obs_margin_d):
                n_already_clear += 1
                continue
            # A box the car is already BESIDE gets no knot. Its s_center is a fraction of a metre
            # ahead (or behind, mod L), and the old floor of 0.3 m pinned its knot there -- so the
            # 0.4 m merge window that follows swallowed the NEXT box's knot while obs_ok went on
            # rejecting that box, and every candidate died. With the two boxes 0.5 m apart there
            # was no recovery at all: the car sat beside the first one publishing nothing.
            #
            # Dropping the knot is not dropping the obstacle. obs_ok still enforces the keep-out of
            # EVERY box in the lookahead, knot or no knot, so this stays fail-closed: the path is
            # not shaped around a box it has already passed, and it is still not allowed to hit it.
            #
            # The gap has to be SIGNED. `(s_center - cur_s) % L` for a centre a hand's breadth
            # BEHIND the car is L - 0.1, which the old clip turned into `lookahead` -- a knot
            # fifteen metres ahead for a box beside the mirror.
            L = self.gb_max_s
            gap_c = ((o.s_center - self.cur_s + L / 2.0) % L) - L / 2.0     # SIGNED
            if gap_c <= 0.0:
                continue                                   # centre level with or behind the car
            s_c = float(min(gap_c, lookahead))
            if knots and s_c <= knots[-1][0] + self.knot_merge_s_m:
                continue                                   # too close in s to the previous apex -> merge
            knots.append((s_c, o, int(o.s_center / wpnt_dist) % self.gb_max_idx))
            if len(knots) >= self.max_weave:
                break
        # An obstacle inside the lookahead that did NOT get a knot is not merely uncovered -- it
        # still contributes its keep-out to obs_ok, so the path is shaped around max_weave boxes and
        # then judged against all of them. The return ramp after the last apex runs straight through
        # whatever was left out, and every candidate is rejected. Name them: this is otherwise
        # indistinguishable from a genuinely impassable section in the all-rejected diagnostic.
        if len(obs_ahead) > len(knots):
            missed = [o for o in obs_ahead if all(o is not ko for (_s, ko, _c) in knots)]
            self.get_logger().warn(
                f"[{self.name}] {len(missed)} of {len(obs_ahead)} obstacle(s) ahead got NO knot "
                f"(max_weave={self.max_weave}): "
                + "; ".join(f"id={o.id} s={o.s_center:.1f} d={o.d_center:+.2f}" for o in missed)
                + ". They are still enforced by obs_ok, so the path must clear them without being "
                  "shaped around them — raise max_weave if this keeps rejecting every candidate.",
                throttle_duration_sec=2.0)
        if not knots:
            # Reachable only with the clear gate disabled -- with it on, an obstacle cleared at
            # obs_margin is also cleared at the gate's smaller threshold, so the gate has already
            # idled. Nothing to shape around either way.
            self.get_logger().info(
                f"[{self.name}] all {len(obs_ahead)} obstacle(s) ahead are already cleared by the "
                f"followed line (>= {obs_margin:.2f} m) -> no avoidance needed",
                throttle_duration_sec=2.0)
            self._committed = None
            return None if squeeze else _empty()
        # Anchor the gap sampling on the first box we are actually shaping around, not on the
        # nearest one in the list -- that one may have been skipped as already cleared.
        nearest = knots[0][1]
        self.obs_in_interest = nearest
        if n_already_clear:
            self.get_logger().info(
                f"[{self.name}] {n_already_clear} obstacle(s) ahead already cleared by the line; "
                f"knots spent on the {len(knots)} that are not", throttle_duration_sec=2.0)
        g_near = (nearest.s_center - self.cur_s) % self.gb_max_s       # forward gap to nearest obstacle
        obs_half_s = ((nearest.s_end - nearest.s_start) % self.gb_max_s) / 2.0
        s_entry0 = max(0.0, knots[0][0] - self.ramp_len)              # gentle ramp OUT starts here
        s_exit_end = knots[-1][0] + self.return_len                   # ease back to the raceline after the LAST apex
        span = min(s_exit_end + self.tail_m, self.gb_max_s * 0.9)

        # --- s-grid for the path ---
        car_idx = self._anchor_car_idx(int(self.cur_s / wpnt_dist) % self.gb_max_idx,
                                       gb_wpnts, wpnt_dist)
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
        # `wall_margin`, not self.wall_margin: on a SQUEEZE retry the caller solved this path at a
        # reduced wall reserve, and measuring the corridor at the design value threw that reduction
        # away -- under trust_grid_bounds (the shipped setting) the squeeze's wall relaxation had no
        # effect at all, so the retry could only ever give up obstacle clearance. The body gate
        # (body_kernel_size, half a car against the eroded map) is unaffected and stays the floor.
        grid_cor = (self._grid_corridor(nearest.s_center, wall_margin=wall_margin)
                    if self.trust_grid_bounds else None)
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
        obox_lo = min(nearest.d_right, nearest.d_left) - obs_margin_d  # car-centre keep-out, right
        obox_hi = max(nearest.d_right, nearest.d_left) + obs_margin_d  # car-centre keep-out, left
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
            # The two gaps above are measured against the NEAREST box only, so with a second box at
            # overlapping s the linspace samples land wherever they land -- including inside the
            # other box's keep-out, and the genuinely free lane BETWEEN two boxes gets a sample only
            # by luck. That is the clustered-obstacle all-rejected case: the gate exists, nothing
            # aimed at it. Subtract the overlapping keep-outs from both gaps and aim at what
            # survives explicitly.
            d_list += self._gate_samples(obs_ahead, nearest, (d_lo, d_hi),
                                         obs_margin_d, obs_margin_s)
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
            obox_lo = min(o.d_right, o.d_left) - obs_margin_d   # car-centre keep-out, right edge
            obox_hi = max(o.d_right, o.d_left) + obs_margin_d   # car-centre keep-out, left edge
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
            g = (self._grid_corridor(s_c, wall_margin=wall_margin)
                 if self.trust_grid_bounds else None)
            if g is not None:
                return g
            return (-(gb_wpnts[cor_idx].d_right - sample_margin),
                    gb_wpnts[cor_idx].d_left - sample_margin)

        # The obstacle's ABSOLUTE station, not knots[i][0]. That first element is the distance
        # from the CAR (s_c = min(gap_c, lookahead)), while _corridor_at hands it to
        # _grid_corridor as an absolute s (`s_query % gb_max_s`). So every knot after the first had
        # its corridor read at a phantom point somewhere else on the track, the apex bulge was
        # clipped to that phantom corridor, and when the clip landed exactly on d_box_hi the
        # inclusive keep-out test in obs_ok rejected every candidate -- an empty publication beside
        # the second box. Only under trust_grid_bounds: the waypoint fallback below already used
        # cor_idx, which is the obstacle's own index.
        #
        # knots[i][0] itself stays s-local: it is consumed that way by the ramp fit, the BPoly
        # breakpoints, the span mask and the merge window.
        knot_cor = [(d_lo, d_hi)] + [_corridor_at(kc, _ko.s_center) for (_ks, _ko, kc) in knots[1:]]

        # --- ADAPTIVE RAMP LENGTH ------------------------------------------------------------
        # The corridor was tested at the obstacle's own station and nowhere else, while the path
        # carries offset over ramp_len + return_len = 9 m of track. Where that 9 m runs through a
        # pinch the ramp is already off the raceline when it gets there, and every candidate dies
        # -- measured on ifac as the single largest feasibility term: 57 of 123 stations at 4.5 m
        # against 83 at 2.5 m.
        #
        # Shortening the ramps globally is the wrong trade: a quintic's peak curvature goes as
        # A/L^2, so 4.5 -> 2.5 m more than triples it, and these lengths were RAISED to carry speed
        # (the straight-line cap falls from 5.0 to 3.75 m/s at 3.75 m). So the length is chosen per
        # candidate instead: keep 4.5 m wherever the corridor accepts the offset over the whole
        # ramp, and shorten only where it does not -- a shorter ramp decays faster, so it asks the
        # pinch for less. Floored at ramp_len_min_m; below that the curvature cost stops being
        # worth the feasibility.
        #
        # Scanned ONCE per cycle at 0.5 m (about 20 grid-corridor calls, ~1.4 ms), not per
        # candidate. The grid is built from the FULL ramp length, so a shortened ramp is a subset
        # of it and no candidate needs its own grid.
        _RAMP_LADDER = (1.0, 0.85, 0.7, 0.6, 0.5)
        scan_lo_s = knots[0][0] - self.ramp_len
        scan_hi_s = knots[-1][0] + self.return_len
        scan_s = np.arange(scan_lo_s, scan_hi_s + 1e-9, 0.5)
        scan_lo = scan_hi = None
        if self.trust_grid_bounds:
            # only as wide as the path can go: the sampled offsets plus the bulge and a cell
            scan_d_max = float(np.max(np.abs(d_ends))) + self.apex_bulge + 0.10
            scan_lo, scan_hi = self._grid_corridor_batch(
                scan_s, d_max=max(scan_d_max, 0.5), d_step=max(self.grid_scan_step, 0.10),
                wall_margin=wall_margin)
        if scan_lo is None:
            scan_lo = np.empty(len(scan_s)); scan_hi = np.empty(len(scan_s))
            scan_lo[:] = np.nan; scan_hi[:] = np.nan
        miss = ~np.isfinite(scan_lo)             # unmeasurable -> the waypoint corridor
        if miss.any():
            jj = ((scan_s[miss] % self.gb_max_s) / wpnt_dist).astype(int) % self.gb_max_idx
            scan_lo[miss] = np.array([-(gb_wpnts[j].d_right - sample_margin) for j in jj])
            scan_hi[miss] = np.array([gb_wpnts[j].d_left - sample_margin for j in jj])

        def _ramp_limits(s_c, full_len, entry):
            """For each candidate ramp length, the widest offset that ramp can carry through the
            corridor: (R, amp_max_positive, amp_min_negative) per rung, longest first.

            Precomputed because only the AMPLITUDE differs between candidates -- the ramp shape and
            the corridor under it do not -- so the per-candidate choice becomes a lookup instead of
            a scan, and the whole feature costs one batched corridor read per cycle."""
            out = []
            for frac in _RAMP_LADDER:
                R = max(full_len * frac, self.ramp_len_min_m)
                ds = (s_c - scan_s) if entry else (scan_s - s_c)
                m = (ds >= 0.0) & (ds <= R)
                if not m.any():
                    out.append((R, np.inf, -np.inf))
                    continue
                t = np.clip(1.0 - ds[m] / max(R, 1e-6), 0.0, 1.0)
                shape = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))          # 0 at the ramp end, 1 at the apex
                sig = shape > 1e-3
                if not sig.any():
                    out.append((R, np.inf, -np.inf))
                    continue
                hi_m, lo_m, sh = scan_hi[m][sig], scan_lo[m][sig], shape[sig]
                out.append((R, float(np.min(hi_m / sh)), float(np.max(lo_m / sh))))
                if R <= self.ramp_len_min_m + 1e-9:
                    break
            return out

        ramp_lim_in = _ramp_limits(knots[0][0], self.ramp_len, entry=True)
        ramp_lim_out = _ramp_limits(knots[-1][0], self.return_len, entry=False)

        def _fit_ramp(amp, limits, full_len):
            """Longest ramp whose offset profile fits the corridor along its WHOLE length."""
            if abs(amp) < 1e-6 or full_len <= self.ramp_len_min_m:
                return full_len
            for R, amp_hi, amp_lo in limits:
                if amp_lo - 1e-9 <= amp <= amp_hi + 1e-9:
                    return R
            return limits[-1][0] if limits else full_len

        dp0_full = cur_dp
        d_cands = np.zeros((N, n))
        cand_entry_i = np.zeros(N, dtype=int)                 # per-candidate prefix boundary
        for k, d_end in enumerate(d_ends):
            d_apex = [float(d_end)]
            for i in range(1, len(knots)):
                d_apex.append(_pass_offset(knot_cor[i], knots[i][1], d_apex[-1]))
            # this candidate's own ramps, from the offsets IT carries (see _fit_ramp)
            r_in = _fit_ramp(d_apex[0], ramp_lim_in, self.ramp_len)
            r_out = _fit_ramp(d_apex[-1], ramp_lim_out, self.return_len)
            s_entry0 = max(0.0, knots[0][0] - r_in)
            s_exit_end = knots[-1][0] + r_out
            m_span = (s_local > s_entry0) & (s_local <= s_exit_end)
            span_ok = s_exit_end > s_entry0 + 1e-3
            dp0 = dp0_full if s_entry0 == 0.0 else 0.0   # match the car heading only at the car
            cand_entry_i[k] = int(np.clip(np.searchsorted(s_local, s_entry0), 0, max(n - 3, 0)))
            # PRE-RAMP, i.e. everything before the hump's entry. This used to hold the car's CURRENT
            # lateral offset flat all the way to s_entry0 -- with a 15 m lookahead and a 4.5 m entry
            # ramp that is up to 10.5 m of published path carrying the instantaneous tracking error
            # as a DC offset, and the hump itself then started from that offset rather than from the
            # raceline. The controller reads it as "stay off the line", which is the opposite of
            # what is wanted while the obstacle is still far away.
            # Decay it to the raceline with a quintic (C2 at both ends) inside min(preramp_len_m,
            # s_entry0), then hold zero until the hump opens. When s_entry0 == 0 the maneuver starts
            # AT the car and the car-anchored entry is kept, heading and all.
            # The decay runs at its DESIGNED rate (preramp_len_m) and no faster. Clipping the
            # length to s_entry0 instead -- "finish the decay before the hump opens, whatever it
            # takes" -- demands the impossible when the hump opens close to the car: with a 0.5 m
            # tracking error and 0.5 m of room it asked for the whole offset to be given back in
            # half a metre, and every candidate then died on curvature and on the grid (measured:
            # 74 curvature rejections where there had been none). If there is no room to finish,
            # the hump simply starts from whatever offset is left -- which is what the car is
            # actually at, and what the s_entry0 <= 0 branch below already does.
            dv = np.zeros(n)
            pre = max(self.preramp_len_m, 1e-6)

            def _decay(x):                                    # 1 at the car, 0 after `pre`
                t = np.clip(np.asarray(x, float) / pre, 0.0, 1.0)
                return 1.0 - t * t * t * (10.0 + t * (-15.0 + 6.0 * t))

            if s_entry0 > 0.0:
                d_start = float(self.cur_d * _decay(s_entry0))
                if abs(self.cur_d) > 1e-9:
                    m_pre = s_local <= s_entry0
                    dv[m_pre] = self.cur_d * _decay(s_local[m_pre])
            else:
                d_start = self.cur_d
                dv[:] = self.cur_d
            if span_ok and m_span.any():
                # One knot per obstacle centre -> a single smooth quintic hump (raceline -> apex ->
                # raceline). d'=0 at each apex makes it the peak. apex_bulge pushes the peak FURTHER
                # from the obstacle (wider swing); the feasibility filter verifies box clearance.
                # Breakpoints must be STRICTLY increasing (BPoly raises otherwise). A knot can now
                # sit arbitrarily close to the car, so the entry and the first apex can coincide.
                bp_s = [s_entry0]
                bp_d = [[d_start, dp0, 0.0]]
                for i_k, ((s_c, _o, _cor), da) in enumerate(zip(knots, d_apex)):
                    # CLIP the bulge to this knot's corridor instead of letting it push the peak
                    # out of the track. The pass offset `da` is already inside the corridor (the
                    # terminal offsets are sampled from it and _pass_offset picks a corridor-tested
                    # side), so only the bulge can leave it -- and it did: on a section narrower
                    # than the design margins the peak landed at the sampling limit + 0.10 m, with
                    # zero terminal reserve, and under squeeze the reserve was NEGATIVE. Clipping
                    # keeps the candidate instead of rejecting it, which is what preserves
                    # feasibility exactly where feasibility is scarce; the box-clearance test
                    # (obs_ok) still decides whether the clipped peak actually passes the obstacle.
                    lo_k, hi_k = knot_cor[i_k]
                    d_peak = float(np.clip(da + self._bulge_away_from(da, _o),
                                           min(lo_k, hi_k), max(lo_k, hi_k)))
                    bp_s.append(max(s_c, bp_s[-1] + 1e-3))
                    bp_d.append([d_peak, 0.0, 0.0])
                bp_s.append(max(s_exit_end, bp_s[-1] + 1e-3))
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
            g0 = gc - o_span / 2.0 - obs_margin_s
            g1 = gc + o_span / 2.0 + obs_margin_s
            d_box_lo = min(o.d_right, o.d_left) - obs_margin_d
            d_box_hi = max(o.d_right, o.d_left) + obs_margin_d
            s_in = (gap_wp >= g0) & (gap_wp <= g1)
            d_in = (d_cands >= d_box_lo) & (d_cands <= d_box_hi)
            obs_ok &= ~(d_in & s_in[None, :]).any(axis=1)

        # cartesian for ALL candidates in one converter call (viz + downstream checks)
        resp = self.converter.get_cartesian(np.tile(s_mod, N), d_cands.reshape(-1))
        xy_all = (resp.T if resp.ndim == 2 else resp).reshape(N, n, 2)

        # THE PREFIX IS NOT A CANDIDATE'S FAULT. Everything before s_entry0 is the decay of the
        # car's CURRENT lateral offset back to the raceline (see the pre-ramp above) plus the car's
        # own position. It is identical in every candidate and the planner cannot change it, so
        # rejecting candidates for a violation there rejects ALL of them and leaves TRAILING as the
        # only output -- with no alternative that would have been better. Measured on ifac with a
        # 0.5 m tracking error and a box 12 m ahead: 310 of 437 grid rejections and 8 of 13 body
        # rejections lie in the prefix; at 0.3 m it is 107 of 115 body rejections. With the box 2 m
        # ahead there is no prefix at all (s_entry0 clamps to 0), which is precisely why the
        # feasibility cliff only appears at distance and only with a tracking error.
        #
        # So the geometry checks start where the planner's own geometry starts. The prefix is not
        # ignored -- a violation there is REPORTED, because it means the car is being tracked into
        # a wall and that is worth knowing; it is just not a reason to discard the escape route.
        i_entry_min = int(cand_entry_i.min()) if N else 0
        if i_entry_min > 0 and self.use_grid_check:
            pre_xy = xy_all[0][:i_entry_min]
            if self._path_off_track(pre_xy) or self._path_body_unsafe(pre_xy):
                self.get_logger().warn(
                    f"[{self.name}] the path PREFIX (the car at d={self.cur_d:+.2f} decaying to "
                    f"the raceline over {self.preramp_len_m:.1f} m) leaves the "
                    f"drivable area. Not a candidate rejection -- every candidate shares it and "
                    f"none can change it -- but the car is being tracked close to a wall.",
                    throttle_duration_sec=2.0)

        # --- heavy checks (grid, curvature, cost) only on geometric survivors ---
        obs_xy = np.array([[o.x_m, o.y_m] for o in obs_ahead], dtype=float)
        best_k, best_J, best = -1, np.inf, None
        status = ["reject"] * N
        n_bounds = n_obs = n_grid = n_curv = n_body = 0   # per-stage reject counters (diagnostics)
        n_feas_left = n_feas_right = 0            # feasible candidates per side (which side has room)
        for k in range(N):
            if not bound_ok[k]:
                n_bounds += 1
                continue
            if not obs_ok[k]:
                n_obs += 1
                continue
            xy = xy_all[k]
            xy_own = xy[cand_entry_i[k]:]         # the part of the path this candidate decides
            if self.use_grid_check and self._path_off_track(xy_own):
                n_grid += 1
                continue
            # BODY floor, deliberately NOT gated on use_grid_check: the check above asks whether the
            # path point is on a free CELL, this one asks whether the whole car fits there. Under
            # trust_grid_bounds nothing else asks that question -- the waypoint corridor test, which
            # reserves half_car + wall_margin, is skipped as redundant with the per-point grid test,
            # and the per-point grid test reserves 0.05 m.
            if self._path_body_unsafe(xy_own):
                n_body += 1
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
            if squeeze:
                return None                     # caller tries the next margin step, then gives up
            # SQUEEZE PASS. Every candidate was rejected at the FULL design margins, which is not
            # the same as "impassable". On ifac the track narrows below 1.20 m, and a box in the
            # middle of that needs width_car/2 + safety_margin = 0.30 m of clearance per side plus
            # its own half-width -- arithmetically unavailable. Without a reduced-margin retry the
            # only remaining behaviour is TRAILING, and behind a STATIONARY box the trailing gap
            # PID's fixed point is v = 0: the "correct" answer becomes stopping forever. Trying the
            # same section at a tighter margin is worse than a clean pass and better than a
            # standstill, so it is offered -- but only slowly (see _squeeze_schedule) and marked, so
            # the SM can cap the speed it is driven at.
            for sm_try, wm_try in self._squeeze_schedule():
                res = self.do_spline(gb_wpnts, safety_margin=sm_try, wall_margin=wm_try,
                                     squeeze=True)
                if res is not None and len(res[0].wpnts) > 0:
                    self.get_logger().warn(
                        f"[{self.name}] SQUEEZE: no candidate at safety_margin="
                        f"{safety_margin:.2f}/wall_margin={wall_margin:.2f}; passing at "
                        f"{sm_try:.2f}/{wm_try:.2f} instead (marked ot_line='squeeze', the state "
                        f"machine caps the speed). The alternative here is a TRAILING standstill.",
                        throttle_duration_sec=1.0)
                    return res
            # Diagnostics: which stage killed every candidate? corridor@obs vs obstacle box vs grid
            # vs curvature, with the geometry so you can see if it's genuinely impassable or a knob.
            self.get_logger().warn(
                f"[{self.name}] NO feasible candidate ({N} sampled) -> TRAILING | "
                f"reject bounds={n_bounds} obs_box={n_obs} grid={n_grid} body={n_body} "
                f"curv={n_curv} | "
                f"g_near={g_near:.2f} obs_half_s={obs_half_s:.2f} n_box={len(knots)} apex_bulge={self.apex_bulge:.2f} | "
                f"sample d_range=[{d_lo:.2f},{d_hi:.2f}] ({cor_src}) corridor@obs "
                f"wpnt L={gb_wpnts[obs_j].d_left:.2f}/R={gb_wpnts[obs_j].d_right:.2f} | "
                f"obs d=[{min(nearest.d_right, nearest.d_left):.2f},{max(nearest.d_right, nearest.d_left):.2f}] "
                f"obs_margin d={obs_margin_d:.2f} s={obs_margin_s:.2f} "
                f"sample_margin={sample_margin:.2f}",
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
            f"sampled {n_left}L+{n_right}R of {N} | reject bounds={n_bounds} obs={n_obs} "
            f"grid={n_grid} body={n_body} curv={n_curv} | "
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
            self._store_commit(obs_ahead, s_mod, d_sel, xy, v_arr, psi_pub, kappa_,
                               obs_margin=obs_margin_d, obs_margin_s=obs_margin_s, squeeze=squeeze,
                               s_entry0=float(s_local[cand_entry_i[best_k]]))
        self._publish_feasible(True)
        return wpnts, self._candidate_markers(xy_all, status, best_k)

    ######################
    # PATH COMMITMENT    #
    ######################
    def _track_near_zero(self, obstacles):
        """Promotion state per obstacle, with HYSTERESIS.

        A bare promote-delay flickers: whenever the tracked speed brushes past the band the
        timer restarts and the avoidance path blinks out for another static_promote_sec. So
        promote below static_near_zero_mps (held static_promote_sec) and demote only above
        static_demote_mps (held static_demote_sec) -- between the two the current belief is
        kept, which is the whole point of a deadband.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        seen = set()
        for o in obstacles:
            seen.add(o.id)
            low = abs(o.vs) < self.static_near_zero_mps and abs(o.vd) < self.static_near_zero_mps
            high = abs(o.vs) >= self.static_demote_mps or abs(o.vd) >= self.static_demote_mps
            if low:
                self._near_zero_since.setdefault(o.id, now)
            else:
                self._near_zero_since.pop(o.id, None)
            if high:
                self._moving_since.setdefault(o.id, now)
            else:
                self._moving_since.pop(o.id, None)
            if o.id in self._promoted:
                t = self._moving_since.get(o.id)
                if t is not None and now - t >= self.static_demote_sec:
                    self._promoted.discard(o.id)
            else:
                t = self._near_zero_since.get(o.id)
                if t is not None and now - t >= self.static_promote_sec:
                    self._promoted.add(o.id)
        for d in (self._near_zero_since, self._moving_since):
            for k in [k for k in d if k not in seen]:
                d.pop(k, None)
        self._promoted &= seen

    def _near_zero_static(self, o) -> bool:
        """May a DYNAMIC-flagged obstacle be treated as static because it reads ~0 speed?

        Only once it has read that way continuously for static_promote_sec, and it stays
        promoted until it reads clearly MOVING for static_demote_sec (see _track_near_zero). On first sight --
        typically as the ego rounds a corner and the opponent enters the FOV -- the tracker's
        KF has just been initialised, so vs/vd read 0.00 for a few cycles while a MOVING
        opponent is spun up. Promoting on that built a snapshot avoidance spline around a car
        that was driving away, which the ego followed until the estimate settled and the path
        was withdrawn: the twitch out of the corner. Speed alone cannot separate "parked" from
        "just seen" -- both read 0.00 -- but time can, and a genuinely parked obstacle keeps
        reading 0.00 for as long as you look at it.
        """
        return o.id in self._promoted

    def _obs_qualifies(self, o) -> bool:
        """The same static / near-stationary + currently-visible gate _gather_obstacles_ahead
        uses to pick avoidance obstacles, factored out for the committed-path re-check."""
        return bool((o.is_static or self._near_zero_static(o)) and o.is_visible)

    def _store_commit(self, obs_ahead, s_mod, d_sel, xy, v_arr, psi_pub, kappa_,
                      obs_margin=None, obs_margin_s=None, squeeze=False, s_entry0=0.0):
        """Snapshot the freshly chosen path (+ the obstacles it was planned around) so later
        cycles republish it verbatim instead of re-solving from the moving car.

        The MARGIN it was solved at is part of the snapshot. A squeeze path re-checked against the
        full design margin fails its own safety re-check on the very next cycle, so the commit
        would be dropped, feasible=False published, and the next cycle would squeeze again -- the
        car flapping in and out of an avoidance it had already committed to. The re-check has to
        ask the question the path was built to answer.
        """
        self._committed = {
            'obs_margin': obs_margin,          # LATERAL keep-out this path was solved at
            'obs_margin_s': obs_margin_s,      # ...and the box s-inflation it was solved at
            'squeeze': bool(squeeze),
            # path-local arc length at which this path's OWN geometry starts (see the prefix note
            # in do_spline). The re-check has to skip the same prefix the candidate check did, or
            # a path accepted here is rejected on its first reuse and the car flaps back out.
            's_entry0': float(s_entry0),
            'obs': [(int(o.id), float(o.s_center), float(o.d_center)) for o in obs_ahead],
            's_mod': np.asarray(s_mod, dtype=float).copy(),
            'd':     np.asarray(d_sel, dtype=float).copy(),
            'xy':    np.asarray(xy, dtype=float).copy(),
            'v':     np.asarray(v_arr, dtype=float).copy(),
            'psi':   np.asarray(psi_pub, dtype=float).copy(),   # already the published convention
            'kappa': np.asarray(kappa_, dtype=float).copy(),
        }

    def _reuse_committed(self, gb_wpnts, wpnt_dist, obs_margin, half_car, obs_margin_s=None):
        """Try to republish the committed path (the slice still ahead of the car). Returns
        (OTWpntArray, MarkerArray) on reuse, or None -- after dropping the commit -- when a fresh
        plan is needed. Publishes the feasibility verdict itself in every path it returns from."""
        c = self._committed
        L = self.gb_max_s
        # Re-check at the margin this path was SOLVED at, not the current design value -- see
        # _store_commit. A squeeze path judged by the full margin fails on its first reuse.
        obs_margin = c.get('obs_margin') or obs_margin
        # ...and the s-axis margin it was solved at, for the same reason
        obs_margin_s = c.get('obs_margin_s') or (obs_margin_s if obs_margin_s is not None
                                                 else obs_margin)

        # --- forward slice via path-local arc length (robust to the s=0 seam) ---
        # s_local is the committed path's own 0..span arc length -- its points are forward-ordered
        # and < 1 lap long, so it is strictly ascending with no wrap; car_prog is how far the car
        # has advanced along it. Keeping s_local >= car_prog - 0.30 drops only the passed prefix.
        s0 = c['s_mod'][0]
        s_local = (c['s_mod'] - s0) % L
        car_prog = (self.cur_s - s0) % L
        ahead = s_local >= (car_prog - 0.30)
        if int(ahead.sum()) < 3:
            self.get_logger().info(
                "[static_avoidance] commit released: the car is past its end (maneuver finished)",
                throttle_duration_sec=1.0)
            self._committed = None                            # maneuver finished -> replan (idle next)
            return None
        i0 = int(np.argmax(ahead))                            # first point at/ahead of the car
        sel = slice(i0, len(c['s_mod']))

        # --- lateral deviation: has the controller fallen off the committed path? ---
        # SOFT re-anchor, not a drop. Dropping the commit here means re-planning from the car's
        # instantaneous pose, and that is a loop: the fresh plan starts at the displaced car, so
        # its entry is displaced too, the controller tracks it with the same error, and the next
        # cycle drops it again. What the geometry actually needs is not a new apex -- the apex is
        # still exactly where the obstacle is -- it is a new ENTRY from where the car really is.
        # So bend the first commit_reanchor_len_m of the committed path to start at the car's
        # current d and blend back onto the committed geometry, leaving everything from there on
        # (apex included) untouched.
        d_car = float(np.interp(car_prog, s_local, c['d']))   # committed d at the car
        if abs(self.cur_d - d_car) > self.commit_dev_max:
            dev = float(self.cur_d) - d_car
            if not self._reanchor_commit(c, car_prog, s_local):
                # Visible on purpose: a dropped commit is a full re-plan from the displaced car,
                # which is the entry to the oscillation loop the re-anchor exists to break. When
                # this appears repeatedly, commit_reanchor_max_m is the number to look at.
                self.get_logger().warn(
                    f"[static_avoidance] commit DROPPED: car is {dev:+.2f} m off the committed "
                    f"path (max {self.commit_dev_max:.2f}, re-anchor limit "
                    f"{self.commit_reanchor_max_m:.2f}) — re-planning from the car",
                    throttle_duration_sec=1.0)
                self._committed = None                        # blend impossible -> full re-plan
                return None
            self.get_logger().info(
                f"[static_avoidance] commit RE-ANCHORED: entry bent {dev:+.2f} m onto the car over "
                f"{self.commit_reanchor_len_m:.2f} m; apex and exit untouched",
                throttle_duration_sec=1.0)
            d_car = float(np.interp(car_prog, s_local, c['d']))

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
                self.get_logger().info(
                    f"[static_avoidance] commit released: box {oid} moved ds={ds:.2f} m "
                    f"(tol {self.commit_obs_ds:.2f}) / dd="
                    f"{abs(match.d_center - od0):.2f} m (tol {self.commit_obs_dd:.2f}) — "
                    f"re-planning the apex",
                    throttle_duration_sec=1.0)
                self._committed = None                        # box moved enough -> re-plan the apex
                return None

        # --- safety: the committed slice must still clear EVERY live box + stay in the corridor ---
        # This is the sole interlock the SM has during static sustain, so it is re-derived here
        # against live obstacles every cycle: geometry frozen, verdict live.
        if not self._commit_slice_clear(c, sel, gb_wpnts, wpnt_dist, obs_margin, half_car,
                                        obs_margin_s):
            self.get_logger().warn(
                f"[static_avoidance] commit released: the frozen slice no longer clears the live "
                f"obstacles/corridor at the margin it was solved with ({obs_margin:.2f} m) — "
                f"publishing feasible=False",
                throttle_duration_sec=1.0)
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

    def _reanchor_commit(self, c, car_prog, s_local) -> bool:
        """Bend the START of the committed path onto the car, keeping the apex where it is.

        The tracking error this handles is a controller property, not a planning error: the apex is
        still exactly where the obstacle is, and re-planning from the displaced car only reproduces
        the same displacement one cycle later -- a fresh plan is anchored AT the car, so its entry
        carries the error into the new geometry and the next cycle drops it again. That loop is the
        oscillation. Correcting only the entry breaks it: the offset the car actually has is faded
        out over commit_reanchor_len_m with a smootherstep (C2 at both ends, so no kink is handed
        to the controller), and every station past the blend -- the apex, the clearance, the exit
        ramp -- is untouched.

        Safety is not weakened: _reuse_committed re-runs _commit_slice_clear on the bent path
        against the LIVE obstacles and the corridor immediately after this returns, and publishes
        feasible=False if the bend has broken it.

        Returns False when the correction is too large to be an entry adjustment, in which case the
        caller does the full re-plan it always did.
        """
        try:
            d = c['d']
            i0 = int(np.searchsorted(s_local, car_prog))
            i0 = int(np.clip(i0, 0, len(d) - 1))
            delta = float(self.cur_d) - float(np.interp(car_prog, s_local, d))
            if abs(delta) > self.commit_reanchor_max_m:
                return False
            blend = max(self.commit_reanchor_len_m, 1e-3)
            t = np.clip((s_local - car_prog) / blend, 0.0, 1.0)
            w = 1.0 - t * t * t * (10.0 + t * (-15.0 + 6.0 * t))   # 1 at the car, 0 after `blend`
            w[s_local < car_prog] = 1.0                            # behind the car: carried along
            c['d'] = d + delta * w
            # geometry, heading and curvature must describe the BENT path, not the old one
            resp = self.converter.get_cartesian(c['s_mod'], c['d'])
            c['xy'] = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)
            if len(c['xy']) > 2:
                # the committed path's own station spacing, same idiom do_spline uses
                el = np.maximum(np.diff(s_local), 1e-3)
                psi_, kappa_ = tph.calc_head_curv_num.calc_head_curv_num(
                    path=c['xy'], el_lengths=el, is_closed=False)
                c['psi'] = psi_ + np.pi / 2.0
                c['kappa'] = kappa_
            self.get_logger().info(
                f"[{self.name}] commit re-anchored: entry bent {delta:+.2f} m over {blend:.1f} m; "
                f"apex and clearance kept", throttle_duration_sec=1.0)
            return True
        except Exception as e:                       # never let this be the thing that crashes
            self.get_logger().warn(f"[{self.name}] commit re-anchor failed: {e}")
            return False

    def _commit_slice_clear(self, c, sel, gb_wpnts, wpnt_dist, obs_margin, half_car,
                            obs_margin_s=None) -> bool:
        """True if the committed forward slice stays inside the track corridor AND clears every
        live (static / near-stationary, visible) obstacle's inflated box. Same box idiom as the
        obs_ok check in do_spline, evaluated on the frozen path against the CURRENT obstacles."""
        s_mod = c['s_mod'][sel]
        d = c['d'][sel]
        L = self.gb_max_s
        # Corridor: the eroded map when it is the authority (the waypoint bounds can ship with
        # d_left/d_right exchanged, which would drop a perfectly good committed path every cycle),
        # otherwise the waypoint corridor.
        # Skip this path's PREFIX, exactly as the candidate check does (see do_spline): it is the
        # decay of the car's own lateral offset, the planner cannot change it, and re-rejecting it
        # here would drop -- on its very first reuse -- a path that was accepted a cycle ago.
        s0 = c['s_mod'][0]
        s_loc_all = (c['s_mod'] - s0) % L
        own_mask = s_loc_all >= float(c.get('s_entry0', 0.0))
        own = np.flatnonzero(own_mask[sel]) + (sel.start or 0)
        if own.size < 3:
            own = np.arange(sel.start or 0, len(c['xy']))
        if self._grid_is_authority():
            if self._path_off_track(c['xy'][own]):
                return False
        else:
            idxs = (s_mod / wpnt_dist).astype(int) % self.gb_max_idx
            d_left = np.array([gb_wpnts[j].d_left for j in idxs])
            d_right = np.array([gb_wpnts[j].d_right for j in idxs])
            if np.any(d > (d_left - half_car)) or np.any(d < -(d_right - half_car)):
                return False
        # BODY floor on the geometry actually being republished, under either authority. This one
        # check covers the whole commitment path: the forward slice is re-derived here every cycle,
        # and _reanchor_commit's blend runs immediately before it -- a bent entry that pushes the
        # car's own displacement into a wall is caught here rather than published.
        if self._path_body_unsafe(c['xy'][own]):
            return False
        gap_wp = (s_mod - self.cur_s) % L
        for o in self.obstacles:
            if not self._obs_qualifies(o):
                continue
            o_span = (o.s_end - o.s_start) % L
            gc = (o.s_center - self.cur_s) % L
            if gc > L / 2.0:
                gc -= L
            m_s = obs_margin if obs_margin_s is None else obs_margin_s
            g0 = gc - o_span / 2.0 - m_s
            g1 = gc + o_span / 2.0 + m_s
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
        # The squeeze marking must survive commitment: it is republished for the whole maneuver,
        # and the SM's speed cap keys off it. Dropping it on reuse would cap only the first cycle.
        # It also OUTRANKS the commit-slice tag below: a squeeze path that is now being republished
        # as a slice is still a squeeze path, and losing the tag would silently lift the speed cap
        # for the rest of the maneuver.
        if c.get('squeeze'):
            wpnts.ot_line = "squeeze"
        elif sel.start:
            # REPUBLISHED SLICE, not a fresh plan: everything before the car has been cut off, so
            # this path starts AT the car and its widest point is whatever is left of the maneuver
            # -- usually the exit ramp. Consumers that read a published path as evidence of what
            # the planner decided (the re-opt's apex recorder above all) need to be able to tell
            # the difference; without it the exit ramps were recorded as apexes and walked the
            # hump station up to a metre downstream of the box.
            wpnts.ot_line = "commit_slice"
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

    def _free_mask(self, xy: np.ndarray, filt=None) -> Optional[np.ndarray]:
        """Vectorised GridFilter.is_point_inside(): True where the point is in the eroded free area.

        Same pixel convention as GridFilter.world_to_pixel()/is_point_inside() (row index = y, no
        vertical flip). Returns None when no map has been received yet, so callers can fall back.
        `filt` selects which eroded view to read: the sampling image by default, the body-safety
        image (body_filter) for the publish-side floor.
        """
        f = self.map_filter if filt is None else filt
        img = getattr(f, "eroded_image", None)
        if img is None or f.resolution is None or f.origin is None:
            return None
        px = ((xy[:, 0] - f.origin[0]) / f.resolution).astype(int)
        py = ((xy[:, 1] - f.origin[1]) / f.resolution).astype(int)
        ok = (px >= 0) & (py >= 0) & (px < img.shape[1]) & (py < img.shape[0])
        free = np.zeros(px.shape, dtype=bool)
        free[ok] = img[py[ok], px[ok]] == 255
        return free

    def _grid_corridor(self, s_query: float, wall_margin: float = None) -> Optional[Tuple[float, float]]:
        """Free lateral extent [d_lo, d_hi] (car-centre limits) at arc length s, MEASURED in the
        eroded occupancy grid rather than read from the waypoints' d_left/d_right.

        Only the CONTIGUOUS free run containing the raceline is kept, so free space that belongs to
        another part of the track further out cannot widen the corridor. Only wall_margin is taken
        off on top of the measured extent, because the erosion already reserves some of what a
        car-centre point needs -- but note HOW MUCH: cv2.erode with a kernel_size x kernel_size
        rect kernel eats floor(kernel_size/2) cells, so the shipped kernel of 3 reserves ONE cell,
        0.05 m at the maps' resolution, not the ~half car an older comment claimed. The effective
        car-centre wall reserve on this path is therefore 0.05 + wall_margin = 0.15 m, i.e. exactly
        half the car with nothing to spare, against half_car + wall_margin = 0.25 m on the
        waypoint-corridor path. Raise kernel_size (or wall_margin) if the car runs too close to
        walls with trust_grid_bounds on.
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
        wm = self.wall_margin if wall_margin is None else float(wall_margin)
        d_lo = float(d_scan[lo_i]) + wm
        d_hi = float(d_scan[hi_i]) - wm
        if d_hi < d_lo:                        # narrower than 2*wall_margin -> collapse to its middle
            d_lo = d_hi = 0.5 * (float(d_scan[lo_i]) + float(d_scan[hi_i]))
        return d_lo, d_hi

    def _grid_corridor_batch(self, s_query: np.ndarray, d_max: float = None, d_step: float = None,
                             wall_margin: float = None):
        """_grid_corridor for MANY stations at once: (lo[n], hi[n]), NaN where unmeasurable.

        One converter call and one image lookup for the whole set. The per-station version costs
        ~0.07 ms each, which is affordable twice a cycle and not affordable twenty times -- and the
        adaptive ramp scan needs it over the ramp's whole length.

        `d_max`/`d_step` narrow the lateral sweep. The ramp scan only has to resolve offsets the
        path can actually reach, and sweeping the full +-grid_scan_max at 5 cm for that is most of
        the cost of the feature. A run that reaches the edge of a narrowed sweep is reported as
        ending there, which UNDERSTATES the corridor -- the conservative direction for a scan whose
        only output is "shorten this ramp".
        """
        n_s = len(s_query)
        if self.gb_max_s is None or getattr(self, "converter", None) is None or n_s == 0:
            return None, None
        d_lim = float(self.grid_scan_max if d_max is None else d_max)
        d_res = float(self.grid_scan_step if d_step is None else d_step)
        wm = self.wall_margin if wall_margin is None else float(wall_margin)
        d_scan = np.arange(-d_lim, d_lim + 1e-9, d_res)
        n_d = len(d_scan)
        ss = np.repeat(np.asarray(s_query, float) % self.gb_max_s, n_d)
        dd = np.tile(d_scan, n_s)
        resp = self.converter.get_cartesian(ss, dd)
        xy = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)
        free = self._free_mask(xy)
        if free is None:
            return None, None
        free = free.reshape(n_s, n_d)
        i0 = int(np.argmin(np.abs(d_scan)))
        lo = np.full(n_s, np.nan)
        hi = np.full(n_s, np.nan)
        for r in range(n_s):
            fr = free[r]
            if not fr.any():
                continue
            j0 = i0
            if not fr[j0]:                       # raceline blocked here -> nearest free sample
                cand = np.flatnonzero(fr)
                j0 = int(cand[np.argmin(np.abs(d_scan[cand]))])
            a_i = b_i = j0
            while a_i > 0 and fr[a_i - 1]:
                a_i -= 1
            while b_i < n_d - 1 and fr[b_i + 1]:
                b_i += 1
            l, h = float(d_scan[a_i]) + wm, float(d_scan[b_i]) - wm
            if h < l:
                l = h = 0.5 * (float(d_scan[a_i]) + float(d_scan[b_i]))
            lo[r], hi[r] = l, h
        return lo, hi

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

    def _path_body_unsafe(self, xy: np.ndarray) -> bool:
        """True if any path point puts the CAR BODY into a wall (body-safety floor).

        Same question as _path_off_track, asked of the image eroded by half a car instead of by one
        cell (see body_kernel_size). The distinction is the whole point of having two images: the
        sampling image decides what may be CONSIDERED, and being generous there is what keeps a
        narrow section passable at all; this one decides what may be PUBLISHED, and being strict
        here is what keeps the published line off the wall. A candidate rejected here is rejected
        outright -- the squeeze pass lowers safety_margin and wall_margin, never this.

        Vectorised (one array lookup for the whole path) because it runs on every candidate that
        survives the cheaper filters and again on every committed slice.

        Silent when no map has arrived: the sampling-side checks are all there is then, exactly as
        _path_off_track already assumes, and vetoing everything on a bench without a map would
        disable the planner instead of protecting it.
        """
        f = getattr(self, "body_filter", None)
        if f is None or getattr(f, "eroded_image", None) is None:
            return False
        free = self._free_mask(np.asarray(xy, dtype=float), f)
        return free is not None and not bool(np.all(free))

    def _publish_feasible(self, feasible: bool):
        # TRANSITIONS are logged UNTHROTTLED. The state machine acts on the edge -- feasible False
        # is what drops it out of the avoidance -- so an edge suppressed by a throttle window is
        # exactly the line missing from the bag when the question is "why did it go TRAILING
        # there?". The steady state is not logged at all, so this stays quiet: two lines per
        # maneuver, not 40 Hz.
        prev = getattr(self, "_last_feasible", None)
        feasible = bool(feasible)
        if prev is not feasible:
            self.get_logger().info(
                f"[static_avoidance] static_feasible {prev} -> {feasible}"
                + (f" @ s={self.cur_s:.2f} m" if self.cur_s is not None else ""))
        self._last_feasible = feasible
        self.feasible_pub.publish(Bool(data=feasible))

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
