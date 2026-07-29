#!/usr/bin/env python3
"""Lane-hold dynamic overtaking planner (node ``planner_change``).

Replaces the snapshot-anchored cosine evasion with a phased lane-change maneuver
so the published path no longer chases a moving opponent:

  IDLE  -> planner stays SILENT until the gap to the opponent has closed to
           engage_gap_m: the approach is owned by TRAILING (controller gap PID)
           or plain raceline driving, at full speed.
  HOLD  -> engage: latch the side, then re-solve the lane every cycle against the
           opponent's latest position and follow it. Entered directly from IDLE
           and held for the whole approach and pass.
  CLOSE -> once the opponent is passed (wrapped signed s-gap > pass_gap_m, held
           for pass_hyst_s), ramp back to the raceline at a latched s.

WHAT THE LANE ACTUALLY IS (this is not a constant offset, and not centerline-
parallel -- earlier revisions of this docstring said both). It is a lateral
profile d(s) in the RACELINE frame, solved on a lane_solve_ds grid by
reachable-interval propagation with a slope limit (_solve_profile). The profile
PREFERS d = 0 (the raceline) and is pushed off it only inside the meeting band
[meet_s - pass_overlap_m, meet_s + _pass_hold_m()], where the corridor is
additionally clamped away from the opponent. So the car stays on the racing line
until the place where the two cars are actually level, and only there does it
step aside -- which is why the state machine must NOT judge this path at the
opponent's current s (see free_check_dynamic_ot_slow in the SM).

There is no separate "open" phase: the profile is anchored at the car
(_solve_profile pins d[0] = current_d) and re-gridded from current_s every
cycle, so a distinct blend-on phase would have terminated on its first cycle by
construction. The blend onto the lane lives in _build_path, not in the phase
machine.

The lane geometry is stable by construction, so the state machine's cached-path
splicing sees a consistent trajectory instead of a per-cycle re-anchored spline
(the source of the old wobble). Velocities are published as 0: the state machine
velocity-replans every received path from its curvature.

COUPLED TO THE STATE MACHINE: the SM runs its own independent free-check on this
path. The clearance numbers are mirrored from its config and derived in
_apply_margins(); stack_master/scripts/check_avoidance_margins.py verifies the
mirrors offline. Do not hand-edit sep_margin_m.

Publishes (unchanged interface):
    /planner/avoidance/otwpnts  OTWpntArray   evasion path (empty topic silence when IDLE)
    /planner/avoidance/merger   Float32MultiArray  [target s_end, path end s]
    /planner/avoidance/markers_sqp  MarkerArray    path viz
"""
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import (
    FloatingPointRange,
    ParameterDescriptor,
    ParameterType,
    SetParametersResult,
)
from rclpy.parameter import Parameter

from nav_msgs.msg import Odometry
from f110_msgs.msg import (
    Wpnt,
    WpntArray,
    Obstacle,
    ObstacleArray,
    OTWpntArray,
    OpponentTrajectory,
    BehaviorStrategy,
)
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, Float32, Bool, Header

from scipy.interpolate import BPoly
from frenet_conversion.frenet_converter import FrenetConverter

from ccma import CCMA
import trajectory_planning_helpers as tph
from transforms3d.euler import quat2euler
from grid_filter.grid_filter import GridFilter

PHASE_IDLE = "IDLE"
PHASE_HOLD = "HOLD"
PHASE_CLOSE = "CLOSE"


def euler_from_quaternion(quaternion):
    """quaternion is [x, y, z, w]; returns (roll, pitch, yaw)."""
    x, y, z, w = quaternion
    return quat2euler([w, x, y, z])


class ChangeAvoidanceNode(Node):
    def __init__(self):
        super().__init__('change_avoidance_node')

        # --- lane / raceline data ---
        self.scaled_wpnts_msg = WpntArray()
        self.scaled_vmax = None
        self.scaled_max_idx = None
        self.scaled_max_s = None
        self.scaled_delta_s = None
        self.center_wpnts_msg = WpntArray()
        self.center_wpnts_received = False
        self.center_s = None            # centerline sampled in the raceline frenet frame
        self.center_d = None            # centerline d(s); corridor reference for the lane
        self.global_waypoints = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.converter = None

        # --- perception state ---
        self.obs_all: List[Obstacle] = []
        self.obs_dynamic: List[Obstacle] = []
        self.opponent_waypoints = []
        self.opp_on_trajectory = False
        self.opponent_wpnts_sm = None

        # --- ego state ---
        self.current_s = None
        self.current_d = None
        self.current_vs = None
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.behavior_state = ""
        self.local_wpnts = None

        # --- maneuver state ---
        self.phase = PHASE_IDLE
        self.side = None                # latched at engage: 'left' | 'right'
        self.lane_sign = 0              # +1 = left of the centerline, -1 = right
        # The lane is an s-varying profile, not a scalar offset: one constant offset had to
        # clear the walls AND the opponent at every sample of the window, so the narrowest
        # point vetoed the whole maneuver even when the opponent was nowhere near it.
        self.meet_s = None              # latched s where the pass happens
        self.lane_s = None              # unwrapped s grid of the solved lane profile
        self.lane_d = None              # lateral profile d(s) [m], raceline frenet frame
        self.lane_offset_cur = 0.0      # peak |d| of the current profile [m] (reporting only)
        self.engaged_offset = 0.0       # peak |d| at engagement
        self.target = None              # dict: id/s/d/vs/size/last_seen
        self.pass_cnt = 0
        self.clear_fail_cnt = 0    # consecutive cycles the lane failed the target-clearance check
        self.close_s = None             # wrapped s where the return ramp starts
        self.close_frozen = False
        self.blocked_since = None
        self.idle_until = 0.0           # re-engage block after finishing/aborting
        self.last_loop_t = None
        self._dt = 0.05
        self.mrk_decim = 0

        # --- constants ---
        self.width_car = 0.30
        self.ot_section_check = False
        self.ot_section_check_t = None

        # --- static params ---
        self.declare_parameter('measure', False,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.measure = self.get_parameter('measure').get_parameter_value().bool_value
        # Lane-change is only validated in overtaking sectors; outside them trailing is the
        # safe fallback in head-to-head, so the ENGAGE gate fails CLOSED when the SM stops
        # publishing. A maneuver already in progress is never aborted by sector state.
        self.declare_parameter('require_ot_section', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.require_ot_section = self.get_parameter('require_ot_section').get_parameter_value().bool_value
        self.declare_parameter('debug_plot', False,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.debug_plot = self.get_parameter('debug_plot').get_parameter_value().bool_value

        # --- tunable params (defaults; overridable via yaml / live reconfigure) ---
        # DISPLAY ONLY since the lane became an s-varying profile: the pass offset is set by
        # the opponent clearance and the walls at each s, never by a constant floor. Kept
        # because the RViz reference lanes are useful when eyeballing a new map.
        self.lane_offset_m = 0.35       # reference lanes drawn in RViz [m]
        self.obs_traj_tresh = 1.0       # engage: max |obs d - ego d|
        self.spline_bound_mindist = 0.25
        self.engage_gap_m = 5.0         # stay silent (TRAILING closes in at speed) until the
                                        # s-gap to the target is below this
        self.offset_slew_mps = 0.6      # how fast the lane offset may grow/shrink [m/s]
        self.open_ramp_min_m = 3.0
        self.open_ramp_time_s = 0.8     # blend length = max(min, t * v)
        self.blend_min_m = 1.5          # HOLD re-anchor blend
        self.close_ramp_min_m = 3.0
        self.close_ramp_time_s = 0.8
        self.close_arm_m = 1.0          # ramp starts this far ahead of the car
        self.tail_m = 2.0               # raceline tail after the return ramp
        self.hold_horizon_m = 22.0      # published path length (> SM interest_horizon 20 m,
                                        # else an escaping opponent past the path end makes
                                        # the SM free-check read NOT-free and aborts)
        self.pass_gap_m = 1.2           # ego must lead by this to trigger CLOSE
        self.pass_hyst_s = 0.3
        # --- SM coupling: sep_margin_m and the monitor abort line are DERIVED, not declared ---
        # The state machine runs its own independent free-check on the path we publish. Both
        # nodes must agree about required clearance, and the coupling used to be a prose comment
        # that had already drifted (it claimed 0.05 m of slack while the value gave 0.02). These
        # three mirrors carry the SM's numbers; _apply_margins() derives the rest and refuses to
        # let the planner ask for LESS than the SM does. Verified offline by
        # stack_master/scripts/check_avoidance_margins.py.
        self.sm_gb_ego_width_m = 0.4    # == state_machine_params.yaml: gb_ego_width_m
        self.sm_lateral_width_m = 0.15  # == planners/dynamic_avoidance_planner.yaml: lateral_width_m
        self.sep_slack_m = 0.07         # solver target above the SM accept line
        self.sep_monitor_slack_m = 0.02  # monitor abort line above it (>=0 => we bail first)
        self.sep_margin_m = 0.42        # DERIVED in _apply_margins; this is only the pre-declare seed
        self._sm_required_m = 0.35      # DERIVED: the SM's accept line
        self._sep_monitor_m = 0.37      # DERIVED: monitors abort below this
        self.target_lost_s = 1.0
        self.reengage_block_s = 1.0
        # Loop rate. Used by the timer AND to convert the dwell params (pass_hyst_s,
        # hold_clear_fail_s) into cycle counts -- it was hardcoded as a bare `* 20` next to a
        # separate `1.0 / 20.0` timer, so changing the rate silently redefined pass_hyst_s.
        self.rate_hz = 20.0
        # Per-cycle re-verification that the lane still clears the target during HOLD.
        self.hold_clear_check = True
        self.hold_clear_fail_s = 0.15   # dwell before acting on a clearance failure [s]
        # Lane profile solver. The opponent clearance is only required where the pass actually
        # overlaps them; everywhere else the lane just has to fit between the walls, which the
        # car alone always does (corridor needs 2*wall_need = 0.54 m, ifac's narrowest is 0.87).
        self.pass_overlap_m = 0.55      # +-s where clearance is enforced = one car length
        self.lane_max_slope = 0.35      # preferred max |dd/ds| of the lane (19 deg heading)
        self.lane_max_slope_close = 0.55  # only if nothing fits at the preferred one (29 deg)
        self.lane_solve_ds = 0.25       # profile solve grid [m]
        self.pass_lead_max_s = 5.0      # cap on the predicted time-to-meet [s]
        self.pass_hold_max_m = 12.0     # cap on how far the pass offset is held [m]
        self.veh_length_m = 0.55        # must match veh_params.length in the SM's ttc/tt0
        self.opp_pred_slack_m = 0.20    # how far the GP may move the opponent beyond vd*t_meet
        self._opp_d_used = {1: 0.0, -1: 0.0}   # diagnostics: opponent d the solve actually used
        self.meet_s_slew_m = 0.15       # how fast the latched meeting point may move [m/cycle]
        self.pass_behind_m = 5.0        # a meeting point within this far BEHIND counts as passed
        # Keep the profile this far inside the corridor to absorb the CCMA smoothing in
        # _publish_path. Measured on real solved profiles through ifac's curvature: the lateral
        # displacement is ~16 mm (median of the per-path 99th pct), and the smoothing does NOT
        # cut the bulge (peak excursion grows by 4-5 mm), so the opponent clearance is not the
        # side at risk. 10 mm off a 120 mm spline_bound_mindist is a 13% bite; the real wall
        # guard is the post-smoothing map check in _publish_path. A 20 mm pad cost 6.8 points
        # of lap feasibility for no measured benefit.
        self.lane_smooth_pad_m = 0.01
        self._lane_dbg = ""
        self.kernel_size = 3            # GridFilter erosion (7 ate ~0.35 m and killed the lanes)

        # --- opponent prediction (P1 temporal, P2 spatial) ---
        self.use_prediction = True           # master toggle; False = react to CURRENT opponent pos only
        self.kd_obs_pred = 1.0               # d->raceline relaxation gain for adaptive propagation
        self.pred_lead_scale = 0.5           # lead time = (s-gap / closing-speed) * this; 0 = no time-forward
        self.engage_min_closing_mps = -0.5   # don't engage an opponent pulling away faster than this

        self._declare_tunables()

        # CCMA init
        self.ccma = CCMA(w_ma=10, w_cc=5)

        # Publishers
        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers_sqp", QoSProfile(depth=10))
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", QoSProfile(depth=10))
        self.merger_pub = self.create_publisher(Float32MultiArray, "/planner/avoidance/merger", QoSProfile(depth=10))
        self.lanes_pub = self.create_publisher(MarkerArray, "/planner/avoidance/lanes", QoSProfile(depth=10))
        if self.measure:
            # NOT /planner/avoidance/latency -- that one belongs to the STATIC planner
            # (spliner/static_avoidance_node.py). Both run in the same graph, so sharing the
            # topic would interleave two planners' loop times into one meaningless histogram.
            # (Was /planner/pspliner_sqp/latency: a leftover from the dormant sqp_planner.)
            self.measure_pub = self.create_publisher(Float32, "/planner/lane_change/latency", QoSProfile(depth=10))

        # Subscribers
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obs_cb, QoSProfile(depth=10))
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, QoSProfile(depth=10))
        self.create_subscription(Odometry, "/car_state/odom", self.state_cartesian_cb, QoSProfile(depth=10))
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, QoSProfile(depth=10))
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, QoSProfile(depth=10))
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.scaled_wpnts_cb, QoSProfile(depth=10))
        self.create_subscription(WpntArray, "/centerline_waypoints", self.center_wpnts_cb, QoSProfile(depth=10))
        self.create_subscription(OpponentTrajectory, "/opponent_trajectory", self.opponent_trajectory_cb, QoSProfile(depth=10))
        self.create_subscription(Bool, "/ot_section_check", self.ot_sections_check_cb, QoSProfile(depth=10))

        self.add_on_set_parameters_callback(self.dyn_param_cb)

        self.converter = self.initialize_converter()

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        # Wait for the centerline waypoints, then project them into the raceline frenet frame
        self.wait_for_message_attr('center_wpnts_received')
        self.build_center_table(center_wpnts=self.center_wpnts_msg)

        self.get_logger().info("[LaneChange] Waiting for messages and services...")
        self.wait_for_loop_messages()
        self.get_logger().info("[LaneChange] Ready!")

        self.create_timer(1.0 / self.rate_hz, self.loop)

    #################### PARAMS ####################
    def _declare_tunables(self):
        def dbl(name, default, lo, hi, desc=""):
            self.declare_parameter(name, default, ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE, description=desc,
                floating_point_range=[FloatingPointRange(from_value=float(lo), to_value=float(hi), step=0.001)]))
            # declare_parameter only seeds the parameter server: a yaml/CLI override lands there
            # but NOT on the attribute the loop reads. Without this read-back every dbl param
            # silently ran on its in-code default -- spline_bound_mindist 0.25 instead of the
            # yaml's 0.12 -- so _choose_lane demanded 0.13 m more room off each wall than
            # configured and rejected every lane on the whole lap.
            setattr(self, name, self.get_parameter(name).value)

        dbl('lane_offset_m', self.lane_offset_m, 0.15, 0.8,
            "RViz reference lanes only -- does not affect the planned path")
        dbl('obs_traj_tresh', self.obs_traj_tresh, 0.1, 2.0, "engage: max |obs d - ego d|")
        dbl('spline_bound_mindist', self.spline_bound_mindist, 0.05, 1.0, "min distance to track bounds")
        dbl('engage_gap_m', self.engage_gap_m, 2.0, 15.0, "trail until the target gap is below this")
        dbl('offset_slew_mps', self.offset_slew_mps, 0.1, 2.0, "lane offset grow/shrink rate")
        dbl('open_ramp_min_m', self.open_ramp_min_m, 1.0, 10.0, "min open blend length")
        dbl('open_ramp_time_s', self.open_ramp_time_s, 0.0, 3.0, "open blend = max(min, t*v)")
        dbl('blend_min_m', self.blend_min_m, 0.5, 5.0, "HOLD re-anchor blend length")
        dbl('close_ramp_min_m', self.close_ramp_min_m, 1.0, 10.0, "min close ramp length")
        dbl('close_ramp_time_s', self.close_ramp_time_s, 0.0, 3.0, "close ramp = max(min, t*v)")
        dbl('close_arm_m', self.close_arm_m, 0.0, 5.0, "close ramp starts this far ahead of car")
        dbl('tail_m', self.tail_m, 0.0, 10.0, "raceline tail after the return ramp")
        dbl('hold_horizon_m', self.hold_horizon_m, 10.0, 40.0, "published path length")
        dbl('pass_gap_m', self.pass_gap_m, 0.3, 5.0, "lead needed to trigger the return")
        dbl('pass_hyst_s', self.pass_hyst_s, 0.0, 2.0, "pass condition dwell time")
        # sep_margin_m is DERIVED from the SM mirrors below -- do not declare it directly.
        dbl('sm_gb_ego_width_m', self.sm_gb_ego_width_m, 0.1, 1.0,
            "MIRROR of state_machine_params.yaml: gb_ego_width_m")
        dbl('sm_lateral_width_m', self.sm_lateral_width_m, 0.0, 1.0,
            "MIRROR of planners/dynamic_avoidance_planner.yaml: lateral_width_m")
        dbl('sep_slack_m', self.sep_slack_m, 0.0, 0.5,
            "solver clearance target above the SM accept line")
        # Negative is allowed on purpose: it reproduces the historical monitor line
        # (sep_margin_m * 0.8) for A/B, even though that sat BELOW the SM's reject line.
        dbl('sep_monitor_slack_m', self.sep_monitor_slack_m, -0.2, 0.5,
            "monitor abort line above the SM accept line; >=0 means we bail before the SM does")
        dbl('pass_overlap_m', self.pass_overlap_m, 0.2, 4.0,
            "+-s around the opponent where lateral clearance is enforced")
        dbl('lane_max_slope', self.lane_max_slope, 0.05, 1.0, "preferred max |dd/ds| of the lane")
        dbl('lane_max_slope_close', self.lane_max_slope_close, 0.05, 1.0,
            "max |dd/ds| used only when nothing fits at the preferred one")
        dbl('pass_lead_max_s', self.pass_lead_max_s, 0.5, 15.0, "cap on the time-to-meet [s]")
        dbl('pass_hold_max_m', self.pass_hold_max_m, 1.0, 25.0, "cap on the offset hold length [m]")
        dbl('opp_pred_slack_m', self.opp_pred_slack_m, 0.0, 2.0,
            "how far the prediction may move the opponent beyond vd*t_meet [m]")
        dbl('meet_s_slew_m', self.meet_s_slew_m, 0.01, 2.0, "how fast the aim point may move [m]")
        dbl('pass_behind_m', self.pass_behind_m, 1.0, 15.0, "meeting point this far behind = passed")
        dbl('target_lost_s', self.target_lost_s, 0.2, 5.0, "coast time before target is dropped")
        dbl('reengage_block_s', self.reengage_block_s, 0.0, 5.0, "idle time after finishing")
        dbl('hold_clear_fail_s', self.hold_clear_fail_s, 0.0, 2.0,
            "dwell before acting on a HOLD target-clearance failure [s]")
        self.declare_parameter('hold_clear_check', self.hold_clear_check, ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description="re-verify every HOLD cycle that the lane still clears the target; "
                        "False restores the old behaviour (checked only once, at engage)"))
        self.hold_clear_check = self.get_parameter('hold_clear_check').value

        # prediction (P1/P2)
        dbl('kd_obs_pred', self.kd_obs_pred, 0.1, 10.0, "d->raceline relaxation gain")
        dbl('pred_lead_scale', self.pred_lead_scale, 0.0, 2.0, "opponent propagation lead-time scale")
        dbl('engage_min_closing_mps', self.engage_min_closing_mps, -3.0, 3.0, "min closing speed to engage")
        self.declare_parameter('use_prediction', self.use_prediction, ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL, description="enable prediction-aware engage/offset/clearance"))
        # dbl() reads its own params back; use_prediction is declared separately, so it needs
        # the same treatment here.
        self.use_prediction = self.get_parameter('use_prediction').value
        self._apply_margins()
        # Print what actually took effect: a yaml key that never reaches an attribute is
        # otherwise invisible until the geometry silently rejects every lane.
        self.get_logger().info(
            f"[LaneChange] params in effect: sm_required={self._sm_required_m:.3f} "
            f"sep_margin={self.sep_margin_m:.3f} sep_monitor={self._sep_monitor_m:.3f} "
            f"bound_mindist={self.spline_bound_mindist:.2f} "
            f"engage_gap={self.engage_gap_m:.1f} obs_traj_tresh={self.obs_traj_tresh:.2f} "
            f"pass_overlap=+-{self.pass_overlap_m:.2f} "
            f"lane_slope={self.lane_max_slope:.2f}/{self.lane_max_slope_close:.2f} "
            f"use_prediction={self.use_prediction} "
            f"engage_min_closing={self.engage_min_closing_mps:+.2f}")

    def _apply_margins(self):
        """Derive the clearance numbers from the mirrored SM parameters.

        Single source of truth for a threshold that TWO nodes must agree on. The SM's
        free-check rejects a path when |d_lane - d_opp| - size/2 < gb_ego_width_m/2 +
        lateral_width_m; that sum is _sm_required_m. The solver aims sep_slack_m above it, and
        the monitors abort sep_monitor_slack_m above it -- so the planner gives up marginally
        BEFORE the SM would, instead of holding a lane the SM rejects (which is exactly the
        OVERTAKE<->TRAILING flapping this replaced: the old monitor line was sep_margin_m * 0.8
        = 0.336, below the SM's 0.40 reject line).

        Clamps rather than raises: a conservative planner beats a dead one."""
        self._sm_required_m = self.sm_gb_ego_width_m / 2.0 + self.sm_lateral_width_m
        self.sep_margin_m = self._sm_required_m + self.sep_slack_m
        self._sep_monitor_m = self._sm_required_m + self.sep_monitor_slack_m
        if self.sep_slack_m < 0.0:
            self.get_logger().error(
                f"[LaneChange] sep_slack_m={self.sep_slack_m:.3f} < 0: the solver would target "
                f"LESS clearance than the SM requires ({self._sm_required_m:.3f} m) and every "
                f"lane would be rejected downstream. Clamping to 0.")
            self.sep_slack_m = 0.0
            self.sep_margin_m = self._sm_required_m
        if self._sep_monitor_m > self.sep_margin_m:
            self.get_logger().error(
                f"[LaneChange] sep_monitor_slack_m={self.sep_monitor_slack_m:.3f} exceeds "
                f"sep_slack_m={self.sep_slack_m:.3f}: the monitors would abort a lane the solver "
                f"just produced. Clamping the monitor to the solver target.")
            self._sep_monitor_m = self.sep_margin_m

    def dyn_param_cb(self, params: List[Parameter]):
        for param in params:
            if param.name in (
                    'lane_offset_m', 'obs_traj_tresh', 'spline_bound_mindist',
                    'engage_gap_m', 'offset_slew_mps',
                    'open_ramp_min_m', 'open_ramp_time_s', 'blend_min_m',
                    'close_ramp_min_m', 'close_ramp_time_s', 'close_arm_m', 'tail_m',
                    'hold_horizon_m', 'pass_gap_m', 'pass_hyst_s',
                    'sm_gb_ego_width_m', 'sm_lateral_width_m',
                    'sep_slack_m', 'sep_monitor_slack_m',
                    'pass_overlap_m', 'lane_max_slope', 'lane_max_slope_close',
                    'pass_lead_max_s', 'meet_s_slew_m',
                    'pass_behind_m', 'pass_hold_max_m', 'opp_pred_slack_m',
                    'target_lost_s', 'reengage_block_s',
                    'hold_clear_fail_s', 'hold_clear_check',
                    'kd_obs_pred', 'pred_lead_scale', 'engage_min_closing_mps',
                    'use_prediction'):
                setattr(self, param.name, param.value)
        # Re-derive: live-tuning any mirror or slack must not leave sep_margin_m/_sep_monitor_m
        # stale, or the solver and the monitors would silently disagree with the SM.
        self._apply_margins()
        self.get_logger().info(
            f"[LaneChange] params: lane_offset_min={self.lane_offset_m:.2f} "
            f"engage_gap={self.engage_gap_m:.1f} pass_gap={self.pass_gap_m:.2f} "
            f"sm_required={self._sm_required_m:.3f} sep_margin={self.sep_margin_m:.3f} "
            f"sep_monitor={self._sep_monitor_m:.3f} hold_horizon={self.hold_horizon_m:.1f}")
        return SetParametersResult(successful=True)

    #################### CALLBACKS ####################
    def obs_cb(self, data: ObstacleArray):
        self.obs_all = list(data.obstacles)
        self.obs_dynamic = [o for o in data.obstacles if not o.is_static]

    def state_frenet_cb(self, data: Odometry):
        self.current_s = data.pose.pose.position.x
        self.current_d = data.pose.pose.position.y
        self.current_vs = data.twist.twist.linear.x

    def state_cartesian_cb(self, data: Odometry):
        self.current_x = data.pose.pose.position.x
        self.current_y = data.pose.pose.position.y
        quaternion = [data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        _, _, self.current_yaw = euler_from_quaternion(quaternion)

    def gb_cb(self, data: WpntArray):
        new_wp = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        changed = (self.global_waypoints is None or new_wp.shape != self.global_waypoints.shape
                   or not np.allclose(new_wp, self.global_waypoints))
        self.global_waypoints = new_wp
        self.gb_max_idx = data.wpnts[-1].id
        self.gb_max_s = data.wpnts[-1].s_m
        # The global line can CHANGE at runtime (static re-opt swap): rebuild the converter AND
        # re-project the centerline table into the CURRENT raceline frame.
        if changed and self.converter is not None:
            self.converter = self.initialize_converter()
            if self.center_wpnts_received:
                self.build_center_table(center_wpnts=self.center_wpnts_msg)

    def scaled_wpnts_cb(self, data: WpntArray):
        self.scaled_wpnts_msg = data
        v_max = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
        if self.scaled_vmax != v_max:
            self.scaled_vmax = v_max
            self.scaled_max_idx = data.wpnts[-1].id
            self.scaled_max_s = data.wpnts[-1].s_m
            self.scaled_delta_s = data.wpnts[1].s_m - data.wpnts[0].s_m

    def center_wpnts_cb(self, data: WpntArray):
        self.center_wpnts_msg = data
        self.center_wpnts_received = True

    def behavior_cb(self, data: BehaviorStrategy):
        self.behavior_state = data.state
        self.local_wpnts = data.local_wpnts

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.opponent_waypoints = data.oppwpnts
        self.opp_on_trajectory = bool(data.opp_is_on_trajectory)
        self.opponent_wpnts_sm = np.array([wpnt.s_m for wpnt in data.oppwpnts])

    def ot_sections_check_cb(self, data: Bool):
        self.ot_section_check = data.data
        self.ot_section_check_t = self.now_sec()

    #################### SMALL HELPERS ####################
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _sdiff(self, a: float, b: float) -> float:
        """Signed wrapped s difference a-b in (-L/2, L/2]."""
        L = self.scaled_max_s
        return ((a - b + L / 2.0) % L) - L / 2.0

    def wait_for_message_attr(self, attr_name: str):
        while not getattr(self, attr_name, False):
            rclpy.spin_once(self)

    def wait_for_loop_messages(self):
        while (self.scaled_max_s is None or self.current_x is None
               or self.current_s is None or self.local_wpnts is None):
            rclpy.spin_once(self)

    def initialize_converter(self) -> "FrenetConverter":
        while self.global_waypoints is None:
            rclpy.spin_once(self)
        converter = FrenetConverter(self.global_waypoints[:, 0], self.global_waypoints[:, 1])
        self.get_logger().info("[LaneChange] initialized FrenetConverter object")
        return converter

    #################### LANE GENERATION (kept from the original node) ####################
    def resample_lane(self, xy: np.ndarray, resolution: float = 0.1) -> np.ndarray:
        deltas = np.diff(xy, axis=0)
        dists = np.hypot(deltas[:, 0], deltas[:, 1])
        s = np.concatenate([[0], np.cumsum(dists)])
        s_new = np.arange(0, s[-1], resolution)
        x_new = np.interp(s_new, s, xy[:, 0])
        y_new = np.interp(s_new, s, xy[:, 1])
        return np.stack([x_new, y_new], axis=1)

    def _center_d(self, s_query):
        """Centerline lateral offset in the raceline frenet frame, interpolated over UNWRAPPED s.

        Any overtaking lane is center_d(s) + lane_sign * lane_offset_cur, so the offset can be
        re-sized per engagement (and grown live) without regenerating any geometry.
        """
        L = self.center_s[-1] + (self.center_s[1] - self.center_s[0]) if self.scaled_max_s is None \
            else self.scaled_max_s
        cs_ext = np.concatenate([self.center_s, self.center_s + L, self.center_s + 2 * L])
        cd_ext = np.concatenate([self.center_d, self.center_d, self.center_d])
        return np.interp(s_query, cs_ext, cd_ext)

    def build_center_table(self, center_wpnts: WpntArray):
        xy = np.array([[w.x_m, w.y_m] for w in center_wpnts.wpnts])
        xy = self.resample_lane(xy, resolution=0.1)
        cs, cd = self.converter.get_frenet(xy[:, 0], xy[:, 1])
        order = np.argsort(np.asarray(cs))
        self.center_s = np.asarray(cs)[order]
        self.center_d = np.asarray(cd)[order]
        self.get_logger().info(
            f"[LaneChange] centerline table built: {len(self.center_s)} pts, "
            f"d range [{float(np.min(self.center_d)):.2f}, {float(np.max(self.center_d)):.2f}]")
        self._publish_lane_markers()

    def _publish_lane_markers(self):
        """Draw two fixed reference lanes for eyeballing a map in RViz. The planned path is an
        s-varying profile solved per cycle and is NOT one of these lanes."""
        if self.center_s is None:
            return
        mrks = MarkerArray()
        for i, sign in enumerate((+1, -1)):
            resp = self.converter.get_cartesian(self.center_s, self.center_d + sign * self.lane_offset_m)
            resp = resp.T if resp.ndim == 2 else resp
            mrk = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
            mrk.ns = "ot_lanes"
            mrk.id = i
            mrk.type = Marker.LINE_STRIP
            mrk.action = Marker.ADD
            mrk.pose.orientation.w = 1.0
            mrk.scale.x = 0.03
            mrk.color.a = 0.6
            mrk.color.r = 0.1 if sign > 0 else 0.9
            mrk.color.b = 0.9 if sign > 0 else 0.1
            mrk.color.g = 0.5
            mrk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.02) for p in resp]
            mrks.markers.append(mrk)
        self.lanes_pub.publish(mrks)

    #################### TARGET TRACKING ####################
    def _update_target(self, now: float, dt: float):
        if self.target is None:
            return
        match = None
        for o in self.obs_dynamic:
            if o.id == self.target['id']:
                match = o
                break
        if match is None:
            # id churn in the tracker: re-associate by nearest wrapped s
            cands = [(abs(self._sdiff(o.s_center, self.target['s'])), o) for o in self.obs_dynamic]
            cands = [c for c in cands if c[0] < 3.0]
            if cands:
                match = min(cands, key=lambda c: c[0])[1]
        if match is not None:
            self.target.update(id=match.id, s=match.s_center, d=match.d_center,
                               vs=match.vs, vd=match.vd, size=max(match.size, 0.25), last_seen=now)
        else:
            # coast on the EKF velocity so the pass check keeps evolving while occluded
            self.target['s'] = (self.target['s'] + self.target['vs'] * dt) % self.scaled_max_s

    def _target_lost_for(self, now: float) -> float:
        return now - self.target['last_seen'] if self.target is not None else float('inf')

    def _pick_engage_target(self) -> Optional[dict]:
        """Nearest dynamic obstacle ahead once TRAILING has closed the gap to engage_gap_m.
        Beyond that gap the planner stays silent: the approach happens at trailing/raceline
        speed, not crawling behind a premature avoidance path."""
        best = None
        for o in self.obs_dynamic:
            if not o.is_visible:
                continue
            gap = (o.s_center - self.current_s) % self.scaled_max_s
            if gap > self.engage_gap_m:
                continue
            if abs(o.d_center - self.current_d) > self.obs_traj_tresh:
                continue
            # don't engage an opponent we cannot close on (it is pulling away): the lane
            # change would only ride the spline and slow us. TRAILING stays the fallback
            # until it slows (e.g. into a corner) and the gap closes for real.
            #
            # This is a CLOSING-SPEED gate, not a prediction feature -- it reads the tracker's
            # measured vs and nothing else. It used to be and-ed with use_prediction, so
            # setting use_prediction:=false silently disabled it too and the planner would
            # engage opponents that were pulling away, burn the maneuver and then eat
            # reengage_block_s. To disable it deliberately, set engage_min_closing_mps very
            # negative (e.g. -3.0); that is its own off-switch.
            if (self.current_vs or 0.0) - o.vs < self.engage_min_closing_mps:
                continue
            if best is None or gap < best[0]:
                best = (gap, o)
        if best is None:
            return None
        o = best[1]
        return dict(id=o.id, s=o.s_center, d=o.d_center, vs=o.vs, vd=o.vd,
                    size=max(o.size, 0.25), last_seen=self.now_sec())

    def _predict_ahead(self, s: float, d: float, vs: float, vd: float = 0.0):
        """Adaptive time-forward propagation of a DYNAMIC obstacle to roughly where it will
        be when the ego reaches it (ported from spliner_node._predict_obs_movement, adaptive
        mode). Lead time grows with time-to-reach; the lateral position relaxes toward the
        raceline, exp-damped by how far off-line the obstacle currently is so a mid-overtake
        opponent is not assumed to snap back. Pass DYNAMIC obstacles only -- a static one has
        vs~=0 but the relaxation term would still pull its d toward the raceline."""
        gap = (s - self.current_s) % self.scaled_max_s
        rel_speed = float(np.clip((self.current_vs or 0.0) - vs, 0.1, 10.0))
        lead = float(np.clip(gap / rel_speed, 0.0, 5.0)) * self.pred_lead_scale
        delta_d = lead * vd
        delta_d = -(d + delta_d) * float(np.exp(-abs(self.kd_obs_pred * d)))
        return (s + lead * vs) % self.scaled_max_s, d + delta_d

    #################### SIDE SELECTION ####################
    def _opp_d_band(self, s_query: np.ndarray) -> np.ndarray:
        """Expected opponent lateral position over s (their line if known, else current d)."""
        if self.opp_on_trajectory and len(self.opponent_waypoints) > 10:
            os_ = np.array([w.s_m for w in self.opponent_waypoints])
            od_ = np.array([w.d_m for w in self.opponent_waypoints])
            order = np.argsort(os_)
            os_, od_ = os_[order], od_[order]
            L = self.scaled_max_s
            os_ext = np.concatenate([os_, os_ + L, os_ + 2 * L])
            od_ext = np.concatenate([od_, od_, od_])
            return np.interp(s_query, os_ext, od_ext)
        return np.full_like(s_query, self.target['d'])

    def _lane_horizon(self) -> float:
        """Solve the lane over exactly what gets published, so the geometry that passes the
        feasibility test is the geometry the SM then evaluates. The old code checked a short
        window and then published hold_horizon_m of the same constant offset, hard-clipped to
        the corridor -- the clip fired at essentially every engage point, so what went out was
        routinely not what had been validated."""
        return min(self.hold_horizon_m, self.scaled_max_s * 0.9)

    def _pass_speed(self) -> float:
        """Speed the ego will run once the pass is committed.

        NOT current_vs: while TRAILING the controller holds station behind the opponent, so
        current_vs measures the state we are trying to leave. Closing reads ~0, the meeting
        point collapses onto the opponent's current s, and the solver is asked to complete the
        lane change in the length of the trailing gap -- which needs ~0.5 dd/ds (about 1 g at
        race pace) and is correctly refused, so the car trails forever. The SM velocity-replans
        every avoidance path from its curvature (see _publish_path), so on commit the ego runs
        the local raceline pace."""
        floor = max(self.current_vs or 0.0, 1.0)
        if self.scaled_wpnts_msg is None or not self.scaled_max_s:
            return floor
        # Averaged over the next few metres, not sampled at one point: the raceline pace swings
        # along the track and this value ends up in the denominator of gap/closing, where a
        # small wobble moves the meeting point by metres.
        span = max(int(5.0 / self.scaled_delta_s), 1)
        i0 = int((self.current_s % self.scaled_max_s) / self.scaled_delta_s)
        idxs = (np.arange(i0, i0 + span) % self.scaled_max_idx)
        v = float(np.mean([self.scaled_wpnts_msg.wpnts[k].vx_mps for k in idxs]))
        return max(v, floor)

    def _update_meet_s(self):
        """Latch the meeting point and let it drift, instead of re-aiming every 50 ms.

        gap/closing is hypersensitive when the closing speed is small (0.4-0.8 m/s here): the
        raw estimate swung 4 m cycle to cycle, the clearance band chased it, and the re-solve
        failed on ~70% of the cycles of an approach. A pass is a commitment to a PLACE."""
        raw = self._meet_s_raw()
        if self.meet_s is None:
            self.meet_s = raw
        else:
            step = float(np.clip(self._sdiff(raw, self.meet_s),
                                 -self.meet_s_slew_m, self.meet_s_slew_m))
            self.meet_s = (self.meet_s + step) % self.scaled_max_s

    def _meet_s(self) -> float:
        return self.meet_s if self.meet_s is not None else self._meet_s_raw()

    def _solve_gentlest(self, s_lin, side_sign, d_prev=None, relax=0.0):
        """Solve at the preferred slope, escalating only if nothing fits.

        The cap is not a dynamics limit -- the SM re-plans the avoidance path's velocity from
        its curvature, so a steeper lane costs pass SPEED, not grip. But it is not free either:
        the solver keeps the lane on the raceline as long as the rate lets it, so raising the
        cap globally makes every pass move later and sharper. Trying the gentle cap first keeps
        the passes that already worked identical, and spends the steep one only where the
        alternative is not passing at all -- a close trailing gap, where 0.35 refused every
        pass on the lap from 1.8 m.
        """
        for slope in (self.lane_max_slope, self.lane_max_slope_close):
            p = self._solve_profile(s_lin, side_sign, d_prev=d_prev, relax=relax, slope=slope)
            if p is not None:
                return p
            if slope >= self.lane_max_slope_close:
                break
        return None

    def _pass_hold_m(self) -> float:
        """How far PAST the meeting point the lane must stay out.

        Minimum excursion pulled the lane back to the raceline one car length past the meeting
        point. Drawing level is not passing: at ~1 m/s of closing the ego needs another second
        or more to get pass_gap_m clear, and it covers several metres of track doing it. A lane
        that has already merged back by then steers into the opponent, and the SM free-check
        reads the merged part as blocked -- which is what made the car flap between TRAILING
        and OVERTAKE. Hold the offset until the pass is done; CLOSE owns the return."""
        if self.target is None:
            return self.pass_overlap_m
        v = self._pass_speed()
        closing = v - self.target['vs']
        lead_needed = self.pass_gap_m + self.pass_overlap_m
        hold = (self.pass_hold_max_m if closing <= 0.2
                else self.pass_overlap_m + v * lead_needed / closing)
        # Mirror of the SM's alongside window (free_check_pass_speed): it now times ttc/tt0
        # from the committed pass speed too, so both nodes look at the same stretch. Keep this
        # in step with _check_free_frenet -- when the two disagreed, the SM checked 11 m ahead
        # while we planned 7, and the hold had to stretch across narrow track to cover it,
        # which then vetoed the pass outright.
        clip_vs = max((self.current_vs or 0.0) - self.target['vs'],
                      self._pass_speed() - self.target['vs'], 0.5)
        gap = self._sdiff(self.target['s'], self.current_s)
        sm_end = gap + self.target['vs'] * (gap + 0.3 * self.veh_length_m) / clip_vs
        meet_fwd = (self._meet_s() - self.current_s) % self.scaled_max_s
        hold = max(hold, sm_end - meet_fwd)
        return float(np.clip(hold, self.pass_overlap_m, self.pass_hold_max_m))

    def _meet_s_raw(self) -> float:
        """s where the ego and the opponent will actually be side by side.

        Deliberately NOT _predict_ahead: that damps its lead time by pred_lead_scale (0.5,
        tuned for the LATERAL relaxation), which puts the meeting point systematically short
        AND makes it drift forward during the approach. The clearance band then sweeps across
        metres of track and the re-solve fails the moment it crosses a narrow section. With
        constant speeds gap/closing is exact and the meeting point does not move at all."""
        L = self.scaled_max_s
        s_o = self.target['s']
        gap = (s_o - self.current_s + L / 2.0) % L - L / 2.0
        closing = self._pass_speed() - self.target['vs']
        if gap <= 0.0 or closing <= 0.2:
            return s_o          # alongside already, or not closing: they are where they are
        # The ego covers gap * (1 + v_opp/closing) of its OWN track before they draw level --
        # that, not the gap, is the run-up available for the lane change.
        return (s_o + self.target['vs'] * min(gap / closing, self.pass_lead_max_s)) % L

    def _lane_bounds(self, s_lin: np.ndarray, side_sign: int):
        """Per-sample lateral corridor [lo, hi] for one side.

        Walls apply everywhere. The opponent clearance applies over one band: from their
        CURRENT s (what the SM free-check reads each cycle) through _pass_hold_m past the
        meeting point (where the pass finishes). Outside it the lane is free to sit on the
        raceline, which is what keeps a narrow section elsewhere from vetoing the maneuver."""
        wp = self.scaled_wpnts_msg.wpnts
        idxs = ((s_lin % self.scaled_max_s) / self.scaled_delta_s).astype(int) % self.scaled_max_idx
        d_left = np.array([wp[j].d_left for j in idxs])
        d_right = np.array([wp[j].d_right for j in idxs])
        wall_need = self.width_car / 2.0 + self.spline_bound_mindist + self.lane_smooth_pad_m
        lo = -(d_right - wall_need)
        hi = d_left - wall_need
        if self.target is not None:
            need = self.target['size'] / 2.0 + self.sep_margin_m
            # The opponent is at ONE lateral position when the cars are level: their d around
            # the MEETING POINT. Sampling their predicted line at every s of the band -- which
            # now runs up to pass_hold_max_m -- instead demands clearance from every place they
            # could ever be, and the hold region is where the EGO will be later, not them. A
            # line that swings then blocks BOTH sides at once: observed live on a 2.7 m wide
            # track as "L:corridor short 0.43m R:corridor short 0.47m", which needs a ~1.9 m
            # lateral swing in the sampled line. The GP predictor logs when its fit has not
            # converged ("length_scale close to the specified upper bound"); that output must
            # not be able to veto the whole track.
            opp_d_ref = self.target['d']
            if self.use_prediction:
                w = self.pass_overlap_m
                q = self._meet_s() + np.arange(-w, w + 1e-9, self.lane_solve_ds)
                pred = self._opp_d_band(q % self.scaled_max_s)
                pred = pred[np.isfinite(pred)]
                if pred.size:
                    # The prediction may only move the opponent as far sideways as the tracker
                    # actually saw them moving. Without this a diverged GP fit (the predictor
                    # logs "length_scale close to the specified upper bound") returns a line
                    # that swings metres and vetoes both sides of the track at once, while the
                    # MEASURED d sat at ~0 the whole time. Measurement is the anchor; the
                    # prediction refines it, it does not get to overrule it.
                    meet_fwd = (self._meet_s() - self.current_s) % self.scaled_max_s
                    t_meet = meet_fwd / max(self._pass_speed(), 0.5)
                    dev = abs(self.target.get('vd', 0.0)) * t_meet + self.opp_pred_slack_m
                    p = float(np.max(pred)) if side_sign > 0 else float(np.min(pred))
                    p = float(np.clip(p, self.target['d'] - dev, self.target['d'] + dev))
                    opp_d_ref = max(p, opp_d_ref) if side_sign > 0 else min(p, opp_d_ref)
            opp_d = np.full(len(s_lin), opp_d_ref)
            self._opp_d_used[side_sign] = opp_d_ref
            # Enforce the clearance where the cars will actually be SIDE BY SIDE. Pinning the
            # band to the opponent's current s constrains a place the two are never at
            # together -- while the ego closes the last metres the opponent has moved on.
            #
            # As a FORWARD DISTANCE, never a signed s-difference: the lap is 36.7 m, so a
            # meeting point 25 m ahead wraps to "11.7 m behind" and the exemption below would
            # fire, applying no opponent constraint at all and calling an empty lane feasible.
            L = self.scaled_max_s
            reach = float(s_lin[-1] - s_lin[0])
            fwd = (self._meet_s() - s_lin[0]) % L
            if fwd > L - self.pass_behind_m:
                # meeting point already behind the car: we are alongside or just ahead, which is
                # exactly when the offset must NOT be given up. Anchor the hold at the car.
                fwd = 0.0
            elif fwd > reach + self.pass_overlap_m + self.lane_solve_ds:
                return lo, hi, False     # past the solved horizon: nothing would be checked
            # The band runs from just before the meeting point through _pass_hold_m past it --
            # long enough to cover the window the SM free-check reads (see _pass_hold_m). One
            # solve-grid step of slack on each edge, because the constraint is applied at 0.25 m
            # samples while the path is published and read at 0.1 m.
            fs = s_lin - s_lin[0]
            start = fwd - self.pass_overlap_m - self.lane_solve_ds
            end = fwd + self._pass_hold_m() + self.lane_solve_ds
            band = (fs >= start) & (fs <= end)
            if not band.any():
                return lo, hi, False
            if side_sign > 0:
                lo = np.where(band, np.maximum(lo, opp_d + need), lo)
            else:
                hi = np.where(band, np.minimum(hi, opp_d - need), hi)
        return lo, hi, True

    def _solve_profile(self, s_lin, side_sign, d_prev=None, relax=0.0, slope=None):
        """Rate-limited minimum-excursion lateral profile inside the corridor, or None.

        Reachable-interval propagation: the forward pass keeps what the car can still get to,
        the backward pass keeps what can also satisfy everything downstream. An empty interval
        anywhere means the pass is genuinely impossible -- a narrow spot away from the opponent
        no longer counts against it, which is the whole point of going s-varying.

        The profile is ANCHORED at the car's current d. Without that the solver is free to
        start the lane already 0.6 m off the raceline and calls a pass feasible that the car
        physically cannot reach: _build_path then blends onto it from the real car state, the
        blend is still short of the lane where the opponent is, and _check_lane_clear_vs_target
        aborts one cycle after engaging."""
        lo, hi, band_ok = self._lane_bounds(s_lin, side_sign)
        if not band_ok or np.any(lo > hi):
            return None
        n = len(s_lin)
        rate = (self.lane_max_slope if slope is None else slope) * float(s_lin[1] - s_lin[0])
        flo, fhi = lo.copy(), hi.copy()
        if self.current_d is not None:
            flo[0] = fhi[0] = float(np.clip(self.current_d, lo[0], hi[0]))
        for i in range(1, n):
            flo[i] = max(flo[i], flo[i - 1] - rate)
            fhi[i] = min(fhi[i], fhi[i - 1] + rate)
            if flo[i] > fhi[i]:
                return None
        for i in range(n - 2, -1, -1):
            flo[i] = max(flo[i], flo[i + 1] - rate)
            fhi[i] = min(fhi[i], fhi[i + 1] + rate)
            if flo[i] > fhi[i]:
                return None
        # preference: hold the previous cycle's lane (continuity for the controller), relaxed
        # toward the raceline by `relax` so the car comes back once the room returns
        want = np.zeros(n) if d_prev is None else d_prev - np.clip(d_prev, -relax, relax)
        d = np.empty(n)
        d[0] = float(np.clip(want[0], flo[0], fhi[0]))
        for i in range(1, n):
            d[i] = float(np.clip(np.clip(want[i], flo[i], fhi[i]),
                                 d[i - 1] - rate, d[i - 1] + rate))
        return d

    def _lane_profile_at(self, s_query):
        """Lane lateral position at (unwrapped) s. Before the first solve this falls back to
        the centerline-parallel lane so nothing can dereference a missing profile."""
        if self.lane_s is None or self.lane_d is None:
            return self._center_d(s_query) + self.lane_sign * self.lane_offset_cur
        return np.interp(s_query, self.lane_s, self.lane_d)

    def _choose_lane(self) -> bool:
        """Latch the overtaking side and solve the lateral profile over the whole published
        horizon. Returns False when neither side admits a drivable profile."""
        if self.scaled_max_s is None or self.target is None:
            return False
        self.meet_s = None            # fresh engagement: do not inherit an old aim point
        self._update_meet_s()
        s_lin = self.current_s + np.arange(0.0, self._lane_horizon(), self.lane_solve_ds)
        cands, dbg = {}, []
        for name, sign in (('left', +1), ('right', -1)):
            prof = self._solve_gentlest(s_lin, sign)
            if prof is not None:
                cands[name] = (float(np.max(np.abs(prof))), sign, prof)
                continue
            lo, hi, band_ok = self._lane_bounds(s_lin, sign)
            if not band_ok:
                dbg.append(f"{name[0].upper()}:meeting point past the {self._lane_horizon():.0f} m horizon")
                continue
            short = float(np.max(lo - hi))
            k = int(np.argmax(lo - hi))
            wp = self.scaled_wpnts_msg.wpnts
            kk = int((s_lin[k] % self.scaled_max_s) / self.scaled_delta_s) % self.scaled_max_idx
            dbg.append(f"{name[0].upper()}@+{s_lin[k] - s_lin[0]:.1f}m"
                       f"[dl={wp[kk].d_left:.2f} dr={wp[kk].d_right:.2f} "
                       f"oppd={self._opp_d_used.get(sign, 0.0):+.2f}]")
            dbg.append(f"{name[0].upper()}:" +
                       (f"corridor short {short:.2f}m" if short > 0.0 else "unreachable at "
                        f"|dd/ds|<={self.lane_max_slope_close:.2f}"))
        if not cands:
            self._lane_dbg = (f"horizon={self._lane_horizon():.0f}m "
                              f"need={self.target['size'] / 2.0 + self.sep_margin_m:.2f} "
                              f"overlap=+-{self.pass_overlap_m:.2f} "
                              f"opp_d={self.target['d']:+.2f} "
                              f"meet_in={(self._meet_s() - self.current_s) % self.scaled_max_s:.1f}m "
                              + " ".join(dbg))
            return False
        self._lane_dbg = ""
        # least lateral excursion wins; slight stickiness to the previous side damps flip-flop
        side = max(cands, key=lambda k: -cands[k][0] + (0.05 if k == self.side else 0.0))
        peak, sign, prof = cands[side]
        self.side, self.lane_sign = side, sign
        self.lane_s, self.lane_d = s_lin, prof
        self.lane_offset_cur = self.engaged_offset = peak
        return True

    def _update_offset(self, dt: float):
        """Re-solve the lane against the opponent's latest position. Continuity comes from
        preferring the previous cycle's profile; it relaxes back toward the raceline at
        offset_slew_mps once the room returns. Not re-solved once the target has been passed,
        so CLOSE has stable geometry to ramp off from."""
        if self.target is None or self.lane_sign == 0:
            return
        if self._sdiff(self.target['s'], self.current_s) < -self.pass_gap_m:
            return
        self._update_meet_s()
        s_lin = self.current_s + np.arange(0.0, self._lane_horizon(), self.lane_solve_ds)
        prev = self._lane_profile_at(s_lin) if self.lane_s is not None else None
        prof = self._solve_gentlest(s_lin, self.lane_sign, d_prev=prev,
                                    relax=self.offset_slew_mps * dt)
        if prof is None:
            # keep the last good lane: the phase steps own the abort decision, and dropping the
            # geometry here would make _build_path jump laterally mid-maneuver
            return
        self.lane_s, self.lane_d = s_lin, prof
        self.lane_offset_cur = float(np.max(np.abs(prof)))

    #################### PATH BUILD + PUBLISH ####################
    def _lane_at(self, s: float) -> float:
        return float(self._lane_profile_at(np.array([float(s)]))[0])

    def _lane_slope(self, s: float) -> float:
        return (self._lane_at(s + 0.3) - self._lane_at(s - 0.3)) / 0.6

    def _build_path(self, closing: bool) -> Optional[dict]:
        """Sample the maneuver path from the car forward. Returns dict with arrays or None."""
        L = self.scaled_max_s
        s0 = self.current_s
        v = max(self.current_vs or 0.0, 1.0)

        horizon = self.hold_horizon_m
        close_len = max(self.close_ramp_min_m, self.close_ramp_time_s * v)
        cs_u = None
        if closing and self.close_s is not None:
            cs_u = s0 + ((self.close_s - s0) % L)
            horizon = max(horizon, (cs_u - s0) + close_len + self.tail_m + 1.0)
        horizon = min(horizon, L * 0.9)

        n = max(int(horizon / 0.1), 20)
        s_lin = s0 + np.arange(n) * 0.1
        d_arr = self._lane_profile_at(s_lin)

        # --- entry blend from the actual car state ---
        e_psi = float(self.converter.get_e_psi(self.current_x, self.current_y, self.current_yaw))
        dp0 = float(np.tan(np.clip(e_psi, -0.5, 0.5)))
        d_gap = abs((self.current_d or 0.0) - float(d_arr[0]))
        open_len = max(self.open_ramp_min_m, self.open_ramp_time_s * v)
        blend_len = float(np.clip(d_gap * 8.0, self.blend_min_m, open_len))
        # The blend must finish BEFORE the stretch where the cars are side by side, else it
        # overwrites the exact piece of lane the solver sized for the clearance and the
        # maneuver is aborted by its own monitor. Anchoring the profile at the car already
        # makes d_arr[0] == current_d, so the blend only has to absorb the heading error and
        # can be short. Applied on every engaged (non-closing) cycle: this is a min(), so it
        # can only shorten the blend, and it used to be gated on a phase that lasted exactly
        # one cycle.
        if not closing and self.target is not None:
            band_start = ((self._meet_s() - s0) % L) - self.pass_overlap_m
            if band_start > 0.0:
                blend_len = min(blend_len, max(0.5, band_start))

        ce = None
        direct_close = closing and cs_u is not None and (cs_u - s0) < blend_len + 0.5
        if direct_close:
            # too little room to rejoin the lane first: ramp straight home from the car
            ce = max(cs_u, s0 + 0.5) + close_len
            bp = BPoly.from_derivatives([s0, ce], [[self.current_d, dp0, 0.0], [0.0, 0.0, 0.0]])
            m = s_lin <= ce
            d_arr[m] = bp(s_lin[m])
            d_arr[s_lin > ce] = 0.0
        else:
            sb = s0 + blend_len
            bp = BPoly.from_derivatives(
                [s0, sb],
                [[self.current_d, dp0, 0.0], [self._lane_at(sb), self._lane_slope(sb), 0.0]])
            m = s_lin <= sb
            d_arr[m] = bp(s_lin[m])
            if closing and cs_u is not None:
                cs_u = max(cs_u, sb + 0.5)
                ce = cs_u + close_len
                bp2 = BPoly.from_derivatives(
                    [cs_u, ce],
                    [[self._lane_at(cs_u), self._lane_slope(cs_u), 0.0], [0.0, 0.0, 0.0]])
                m2 = (s_lin >= cs_u) & (s_lin <= ce)
                d_arr[m2] = bp2(s_lin[m2])
                d_arr[s_lin > ce] = 0.0

        # --- clip to the track corridor (never demand more room than the walls give) ---
        wp = self.scaled_wpnts_msg.wpnts
        idxs = ((s_lin % L) / self.scaled_delta_s).astype(int) % self.scaled_max_idx
        d_left = np.array([wp[j].d_left for j in idxs])
        d_right = np.array([wp[j].d_right for j in idxs])
        wall_need = self.width_car / 2.0 + self.spline_bound_mindist
        d_arr = np.clip(d_arr, -(d_right - wall_need), d_left - wall_need)

        return dict(s_lin=s_lin, d_arr=d_arr, cs_u=cs_u, ce=ce)

    def _publish_path(self, path: dict) -> bool:
        """Convert, safety-check, smooth and publish. Returns False if the path is infeasible."""
        L = self.scaled_max_s
        s_wrapped = path['s_lin'] % L
        resp = self.converter.get_cartesian(s_wrapped, path['d_arr'])
        resp = resp.T if resp.ndim == 2 else resp
        samples = np.asarray(resp, dtype=float).reshape(-1, 2)
        if samples.shape[0] < 3 or not np.all(np.isfinite(samples)):
            return False

        smoothed_xy = self.ccma.filter(samples)
        # Check what actually gets published, not the pre-smoothing samples: CCMA moves the
        # path laterally (~16 mm typical) and it is the smoothed geometry the SM and the
        # controller consume.
        for i in range(smoothed_xy.shape[0]):
            if not self.map_filter.is_point_inside(smoothed_xy[i, 0], smoothed_xy[i, 1]):
                return False
        smoothed_sd = self.converter.get_frenet(smoothed_xy[:, 0], smoothed_xy[:, 1])
        evasion_x = np.asarray(smoothed_xy[:, 0])
        evasion_y = np.asarray(smoothed_xy[:, 1])
        evasion_s = np.asarray(smoothed_sd[0]) % L
        evasion_d = np.asarray(smoothed_sd[1])
        evasion_coords = np.column_stack((evasion_x, evasion_y))

        # Guard: drop coincident points (zero-length segments -> NaN psi/kappa/velocity).
        if len(evasion_coords) >= 2:
            seg = np.linalg.norm(np.diff(evasion_coords, axis=0), axis=1)
            keep = np.concatenate([[True], seg > 1e-6])
            if not np.all(keep):
                evasion_x = evasion_x[keep]
                evasion_y = evasion_y[keep]
                evasion_s = evasion_s[keep]
                evasion_d = evasion_d[keep]
                evasion_coords = evasion_coords[keep]
        if len(evasion_coords) < 3 or not np.all(np.isfinite(evasion_coords)):
            return False

        evasion_psi, evasion_kappa = tph.calc_head_curv_num.calc_head_curv_num(
            path=evasion_coords,
            el_lengths=0.1 * np.ones(len(evasion_coords) - 1),
            is_closed=False,
        )
        evasion_psi += np.pi / 2

        msg = OTWpntArray(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
        msg.ot_side = self.side or ""
        # Velocity intentionally zero: the state machine velocity-replans every received
        # avoidance path from its curvature (update_velocity), planner owns geometry only.
        msg.wpnts = [
            Wpnt(id=i, s_m=float(s), d_m=float(d), x_m=float(x), y_m=float(y),
                 psi_rad=float(p), kappa_radpm=float(k), vx_mps=0.0)
            for i, (x, y, s, d, p, k) in enumerate(
                zip(evasion_x, evasion_y, evasion_s, evasion_d, evasion_psi, evasion_kappa))
        ]
        self.evasion_pub.publish(msg)

        if self.target is not None:
            self.merger_pub.publish(Float32MultiArray(
                data=[(self.target['s'] + self.target['size'] / 2.0) % L, float(evasion_s[-1])]))
        self._visualize_path(evasion_x, evasion_y)
        return True

    #################### SAFETY MONITORS ####################
    def _path_blocked_ahead(self, path: dict, now: float) -> bool:
        """True when an obstacle sits ON the maneuver path ahead (persistent).

        The latched target while BESIDE the car is exempt: bailing to the raceline
        next to an opponent is worse than holding the lane.
        """
        blocked = False
        for o in self.obs_all:
            rel = self._sdiff(o.s_center, self.current_s)
            if not (-0.5 < rel < 12.0):
                continue
            if self.target is not None and o.id == self.target['id']:
                # The latched target is _check_lane_clear_vs_target's job, and only that one
                # asks the question in the meeting-point frame. Testing it here at its current
                # s would re-introduce the abort this monitor is not meant to own.
                continue
            i = int(np.clip(rel / 0.1, 0, len(path['d_arr']) - 1))
            sep = abs(float(path['d_arr'][i]) - o.d_center) - max(o.size, 0.25) / 2.0
            if sep < self._sep_monitor_m:   # derived: sits just ABOVE the SM's reject line
                blocked = True
                break
        if blocked:
            if self.blocked_since is None:
                self.blocked_since = now
        else:
            self.blocked_since = None
        return blocked and (now - self.blocked_since) > 0.3

    def _ot_gate_open(self) -> bool:
        if not self.require_ot_section:
            return True
        now = self.now_sec()
        if self.ot_section_check_t is None or (now - self.ot_section_check_t) > 1.0:
            # The SM only publishes /ot_section_check on some transition paths, so staleness
            # here can also mean "SM idle in GB_TRACK" -- still fail closed, trailing is safe.
            self.get_logger().warn(
                "[LaneChange] /ot_section_check stale or never received -- lane change suppressed",
                throttle_duration_sec=5.0)
            return False
        return self.ot_section_check

    #################### PHASE MACHINE ####################
    def _to_idle(self, now: float, why: str):
        self.get_logger().info(f"[LaneChange] {self.phase} -> IDLE ({why})")
        self.phase = PHASE_IDLE
        self.target = None
        self.meet_s = None
        self.lane_s = None
        self.lane_d = None
        self.close_s = None
        self.close_frozen = False
        self.pass_cnt = 0
        self.clear_fail_cnt = 0    # consecutive cycles the lane failed the target-clearance check
        self.blocked_since = None
        self.idle_until = now + self.reengage_block_s
        self._clear_markers()

    def _passed(self) -> bool:
        if self.target is None:
            return False
        rel = self._sdiff(self.current_s, self.target['s'])   # >0: ego is ahead
        rel_v = (self.current_vs or 0.0) - self.target['vs']
        return rel > self.pass_gap_m and rel_v > -0.3

    def loop(self):
        start_time = time.perf_counter()
        now = self.now_sec()
        dt = 0.05 if self.last_loop_t is None else float(np.clip(now - self.last_loop_t, 0.01, 0.2))
        self.last_loop_t = now
        self._dt = dt

        self._update_target(now, dt)

        if self.phase == PHASE_IDLE:
            self._step_idle(now)
        elif self.phase == PHASE_HOLD:
            self._step_hold(now)
        elif self.phase == PHASE_CLOSE:
            self._step_close(now)

        if self.measure:
            self.measure_pub.publish(Float32(data=time.perf_counter() - start_time))

    def _step_idle(self, now: float):
        if now < self.idle_until:
            self._idle_why(f"re-engage blocked for {self.idle_until - now:.1f} s")
            return
        if not self._ot_gate_open():
            self._idle_why("ot_section gate closed/stale")
            return
        target = self._pick_engage_target()
        if target is None:
            self._idle_why(self._engage_reject_reason())
            return
        self.target = target
        if not self._choose_lane():
            self._idle_why(f"no lane fits: {self._lane_dbg}")
            self.target = None
            return
        self.phase = PHASE_HOLD
        self.pass_cnt = 0
        self.clear_fail_cnt = 0
        self.get_logger().info(
            f"[LaneChange] IDLE -> HOLD (engage): target id={target['id']} "
            f"gap={self._sdiff(target['s'], self.current_s):.1f} m "
            f"side={self.side} offset={self.lane_offset_cur:.2f} m "
            f"meet_in={(self._meet_s() - self.current_s) % self.scaled_max_s:.1f} m "
            f"v_pass={self._pass_speed():.1f} v_opp={target['vs']:.1f}")

    def _idle_why(self, why: str):
        """One throttled line explaining why IDLE did not engage. IDLE publishes nothing, so
        without this the node is silently indistinguishable from dead."""
        self.get_logger().info(f"[LaneChange] IDLE: {why}", throttle_duration_sec=1.0)

    def _engage_reject_reason(self) -> str:
        """Per-gate verdict for the nearest visible dynamic obstacle ahead."""
        if not self.obs_dynamic:
            return f"no dynamic obstacles (obs_all={len(self.obs_all)})"
        if self.scaled_max_s is None or self.current_s is None:
            return "waiting for /global_waypoints_scaled or odom"
        best, best_gap = None, None
        for o in self.obs_dynamic:
            gap = (o.s_center - self.current_s) % self.scaled_max_s
            if best_gap is None or gap < best_gap:
                best, best_gap = o, gap
        closing = (self.current_vs or 0.0) - best.vs
        return (f"nearest dyn id={best.id} gap={best_gap:.2f}(<={self.engage_gap_m:.1f}?"
                f"{'ok' if best_gap <= self.engage_gap_m else 'NO'}) "
                f"|dd|={abs(best.d_center - (self.current_d or 0.0)):.2f}"
                f"(<={self.obs_traj_tresh:.1f}?"
                f"{'ok' if abs(best.d_center - (self.current_d or 0.0)) <= self.obs_traj_tresh else 'NO'}) "
                f"closing={closing:+.2f}(>={self.engage_min_closing_mps:+.1f}?"
                f"{'ok' if closing >= self.engage_min_closing_mps else 'NO'}) "
                f"visible={best.is_visible}")

    def _safe_to_abort(self) -> bool:
        """Are we far enough behind the target that bailing to the raceline is safe?

        Every abort site asks this one question. It used to be asked with two different
        numbers: -(width_car*2 + 0.6) = -1.2 m (a dimensionally meaningless expression -- a
        car WIDTH used as a longitudinal threshold) and -(pass_gap_m + 0.5) = -1.7 m. Bailing
        while alongside steers us back into the opponent, so the conservative one wins."""
        if self.target is None:
            return True
        return self._sdiff(self.current_s, self.target['s']) < -(self.pass_gap_m + 0.5)

    def _step_hold(self, now: float):
        lost = self._target_lost_for(now)
        rel = self._sdiff(self.current_s, self.target['s']) if self.target else 0.0
        # target ran away faster than we can close: nothing to overtake any more. (Inherited
        # from the deleted OPEN phase, which could only ever check it on its single cycle.)
        if self.target is not None and rel < -(self.engage_gap_m + 4.0):
            self._to_idle(now, "target pulled away")
            return
        # lost while already ahead (typical: opponent left the lidar FOV behind us) counts as
        # passing; lost for a long time while still nominally behind means nothing left to pass
        passed_now = self._passed() or (lost > self.target_lost_s and rel > 0.0)
        if not passed_now and lost > 3.0 * self.target_lost_s:
            self._arm_close(now, "target gone")
            return

        self.pass_cnt = self.pass_cnt + 1 if passed_now else 0
        if self.pass_cnt >= max(int(self.pass_hyst_s * self.rate_hz), 1):
            self._arm_close(now, f"passed (lead {rel:.1f} m)")
            return

        self._update_offset(self._dt)
        path = self._build_path(closing=False)
        # Re-verify the lane still clears the opponent we are overtaking. Nothing else does:
        # _path_blocked_ahead deliberately EXCLUDES the latched target, and this check used to
        # run only from the deleted one-cycle OPEN phase -- so for the entire hold the planner
        # trusted a promise the solver made once, even though _build_path then post-processes
        # the profile with an entry blend and a corridor clip that can both eat into it.
        # Checked BEFORE publishing so a lane that fails is never handed to the SM.
        if path is not None and self.hold_clear_check:
            if self._check_lane_clear_vs_target(path):
                self.clear_fail_cnt = 0
            else:
                self.clear_fail_cnt += 1
                if self.clear_fail_cnt >= max(int(self.hold_clear_fail_s * self.rate_hz), 1):
                    if self._safe_to_abort():
                        self._to_idle(now, "lane no longer clears target")
                        return
                    # Alongside or ahead: publishing a lane that is a few cm short beats
                    # steering back onto the raceline right next to the opponent. Same
                    # reasoning as _path_blocked_ahead's target exemption.
                    self.get_logger().warn(
                        f"[LaneChange] lane short of target clearance for "
                        f"{self.clear_fail_cnt / self.rate_hz:.2f} s but too close alongside to "
                        f"abort (rel={rel:+.2f} m) -- holding the lane",
                        throttle_duration_sec=1.0)
        if path is None or not self._publish_path(path):
            self._to_idle(now, "path infeasible")
            return
        if self._path_blocked_ahead(path, now) and self._safe_to_abort():
            # blocked and clearly behind the target again -> give up, SM falls back to TRAILING
            self._to_idle(now, "blocked, dropping back")
            return
        self._visualize_phase()

    def _arm_close(self, now: float, why: str):
        self.phase = PHASE_CLOSE
        self.close_s = (self.current_s + self.close_arm_m) % self.scaled_max_s
        self.close_frozen = False
        self.get_logger().info(f"[LaneChange] HOLD -> CLOSE ({why})")
        # publish the close path in the SAME cycle so the SM freshness gate never sees a gap
        self._step_close(now)

    def _step_close(self, now: float):
        L = self.scaled_max_s
        # opponent re-passed us while we were closing -> reopen the hold
        if self.target is not None and self._target_lost_for(now) < self.target_lost_s:
            rel = self._sdiff(self.current_s, self.target['s'])
            if rel < 0.3:
                self.phase = PHASE_HOLD
                self.close_s = None
                self.close_frozen = False
                self.pass_cnt = 0
                self.get_logger().info("[LaneChange] CLOSE -> HOLD (target re-passed)")
                return

        on_lane = abs((self.current_d or 0.0) - self._lane_at(self.current_s)) < 0.10
        if not self.close_frozen:
            if on_lane:
                # slide the ramp start so it always begins just ahead of the car: whenever the
                # SM adopts the newest path there is no lateral step
                cand = (self.current_s + self.close_arm_m) % L
                if self._sdiff(cand, self.close_s) > 0.0:
                    self.close_s = cand
            else:
                self.close_frozen = True   # car started ramping: freeze the geometry

        # postpone the merge while an obstacle occupies the return corridor (only while the
        # ramp is still sliding: once the car is ramping the geometry stays frozen)
        if not self.close_frozen:
            close_len = max(self.close_ramp_min_m,
                            self.close_ramp_time_s * max(self.current_vs or 0.0, 1.0))
            cs_rel = (self.close_s - self.current_s) % L
            for o in self.obs_all:
                rel_o = self._sdiff(o.s_center, self.current_s)
                if not (-0.5 < rel_o < cs_rel + close_len + 2.0):
                    continue
                lane_d_at_o = self._lane_at(self.current_s + max(rel_o, 0.0))
                corridor_lo = min(0.0, lane_d_at_o) - 0.3
                corridor_hi = max(0.0, lane_d_at_o) + 0.3
                if corridor_lo - o.size / 2.0 < o.d_center < corridor_hi + o.size / 2.0:
                    push = rel_o + o.size / 2.0 + 0.5
                    cand = (self.current_s + max(self.close_arm_m, push)) % L
                    if self._sdiff(cand, self.close_s) > 0.0:
                        self.close_s = cand

        path = self._build_path(closing=True)
        if path is None or not self._publish_path(path):
            self._to_idle(now, "close path infeasible")
            return
        self._visualize_phase()

        # merge complete: on the raceline and past the latched ramp region
        past_ramp = self._sdiff(self.current_s, self.close_s) > 0.5
        if path['ce'] is not None:
            past_ramp = past_ramp or self._sdiff(self.current_s, path['ce'] % L) > 0.0
        if abs(self.current_d or 0.0) < 0.12 and past_ramp:
            self._to_idle(now, "merge complete")

    def _check_lane_clear_vs_target(self, path: dict) -> bool:
        """Verify on the BUILT path what the solver promised: clearance from the opponent over
        the stretch where the two are actually side by side.

        This used to check the opponent's CURRENT s. Once the lane became an s-varying profile
        that contradicts the solver by construction -- minimum excursion keeps the lane on the
        raceline until the meeting point, so the check saw it running straight at where the
        opponent is standing now and aborted every engagement one cycle after OPEN. Checking
        the built path (blend and corridor clip included) against the meeting band is the same
        question the solver answered, asked of the geometry that actually goes out."""
        if self.target is None:
            return True
        n = len(path['d_arr'])
        fwd = (self._meet_s() - self.current_s) % self.scaled_max_s
        if fwd > (n - 1) * 0.1:
            return True          # meeting point past the published path, or already behind us
        i0 = int(np.clip((fwd - self.pass_overlap_m) / 0.1, 0, n - 1))
        i1 = int(np.clip((fwd + self.pass_overlap_m) / 0.1, 0, n - 1))
        seg = np.asarray(path['d_arr'][i0:i1 + 1], dtype=float)
        if seg.size == 0:
            return True
        d_opp = self.target['d']
        if self.use_prediction:
            pred = float(self._opp_d_band(np.array([self.current_s + fwd]))[0])
            d_opp = max(pred, d_opp) if self.lane_sign > 0 else min(pred, d_opp)
        sep = float(np.min(np.abs(seg - d_opp))) - self.target['size'] / 2.0
        return sep >= self._sep_monitor_m

    #################### VIZ ####################
    def _clear_markers(self):
        mrks = MarkerArray()
        del_mrk = Marker(header=Header(stamp=self.get_clock().now().to_msg()))
        del_mrk.action = Marker.DELETEALL
        mrks.markers.append(del_mrk)
        self.mrks_pub.publish(mrks)

    def _visualize_path(self, xs, ys):
        self.mrk_decim = (self.mrk_decim + 1) % 4
        if self.mrk_decim != 0:
            return
        mrks = MarkerArray()
        mrk = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
        mrk.ns = "lane_change_path"
        mrk.id = 0
        mrk.type = Marker.LINE_STRIP
        mrk.action = Marker.ADD
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 0.06
        mrk.color.a = 1.0
        mrk.color.r = 0.63
        mrk.color.g = 0.13
        mrk.color.b = 0.94
        mrk.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in zip(xs, ys)]
        mrks.markers.append(mrk)
        self.mrks_pub.publish(mrks)

    def _visualize_phase(self):
        if self.mrk_decim != 0:
            return
        mrks = MarkerArray()
        mrk = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
        mrk.ns = "lane_change_phase"
        mrk.id = 1
        mrk.type = Marker.TEXT_VIEW_FACING
        mrk.action = Marker.ADD
        rel = self._sdiff(self.current_s, self.target['s']) if self.target else 0.0
        mrk.text = f"{self.phase} {self.side or ''} rel={rel:+.1f}m"
        mrk.pose.position.x = float(self.current_x)
        mrk.pose.position.y = float(self.current_y)
        mrk.pose.position.z = 0.8
        mrk.pose.orientation.w = 1.0
        mrk.scale.z = 0.35
        mrk.color.a = 1.0
        mrk.color.r = 1.0
        mrk.color.g = 1.0
        mrk.color.b = 1.0
        mrks.markers.append(mrk)
        self.mrks_pub.publish(mrks)


def main(args=None):
    rclpy.init(args=args)
    node = ChangeAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
