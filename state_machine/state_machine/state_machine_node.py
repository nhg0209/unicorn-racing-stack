#!/usr/bin/env python3
"""
UNICORN racing state machine - ROS2 (Jazzy / rclpy) port.

Ported from the ROS1 (catkin/rospy) `state_machine` package. This is the racing
"brain": it subscribes to perception / planning / localization topics, computes a
set of boolean conditions, runs the state-transition graph and publishes the chosen
driving behaviour (local waypoints + BehaviorStrategy).

The full UNICORN feature set is preserved (RECOVERY / START / multi-planner
sustainability / prediction-aware free checks / velocity replanning / BehaviorStrategy
trailing & overtaking targets). The race_stack ROS2 template was used only for the
ament/rclpy structural idioms.
"""
import copy
import math
import os
import time
import json
import configparser

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

import transforms3d
from ament_index_python.packages import get_package_share_directory

from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from std_msgs.msg import String, Float32, Float32MultiArray, Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import (
    ObstacleArray,
    OTWpntArray,
    WpntArray,
    BehaviorStrategy,
    PredictionArray,
)

import trajectory_planning_helpers as tph

from frenet_conversion.frenet_converter import FrenetConverter

from vel_planner.vel_planner import calc_vel_profile

# OPTIONAL BY DESIGN. rate_check only ever prints a warning, so a workspace where it has not been
# built yet loses the warning and nothing else -- which is exactly the state every node was in
# before it existed. Hard-failing a live node on a missing diagnostic would be a worse trade.
try:
    from rate_check.rate_check import RateCheck
except ImportError:                          # pragma: no cover - deployment shape, not logic
    RateCheck = None
from state_machine.states_types import StateType
from state_machine import states
from state_machine import state_transitions
from state_machine.state_machine_params import StateMachineParams

try:
    # if we are in the car, vesc msgs are built and we read them
    from vesc_msgs.msg import VescStateStamped
except Exception:
    pass


class WaypointData:
    """Holds the latest waypoints of a given planner together with its (dynamic)
    parameters. In ROS1 these parameters were served by a per-planner
    `dynamic_reconfigure` server (dyn_planner_tuner.cfg). In ROS2 they are declared on
    the state-machine node as nested parameters `<planner_name>.<param>` (loaded from
    the planner yaml in this package's config/planners directory).
    """

    def __init__(self, node: "StateMachine", planner_name: str, is_closed: bool):
        self.node = node
        self.name = planner_name
        self.list = []
        self.array = None
        self.stamp = None
        self.is_init = False
        self.is_gb_track_wpnts = False
        # is_ot_wpnts is True for BOTH avoidance paths (static and dynamic). Where a check must
        # distinguish them -- the two planners have opposite geometry contracts -- use
        # is_dynamic_ot_wpnts, which is set only on cur_avoidance_wpnts.
        self.is_ot_wpnts = False
        self.is_dynamic_ot_wpnts = False
        self.closest_target = None
        self.closest_gap = None
        self.is_closed = is_closed
        self.vel_planner_safety_factor = 1.0
        # Sec this cache was last selected as local_wpnts_src (None until first use).
        self.last_used_sec = None
        self.update_param()

    def update_param(self):
        get = self.node.get_planner_param
        self.min_horizon = get(self.name, "min_horizon")
        self.max_horizon = get(self.name, "max_horizon")
        self.lateral_width_m = get(self.name, "lateral_width_m")
        self.free_scaling_reference_distance_m = get(self.name, "free_scaling_reference_distance_m")
        self.latest_threshold = get(self.name, "latest_threshold")
        self.on_spline_front_horizon_thres_m = get(self.name, "on_spline_front_horizon_thres_m")
        self.on_spline_min_dist_thres_m = get(self.name, "on_spline_min_dist_thres_m")
        self.hyst_timer_sec = get(self.name, "hyst_timer_sec")
        self.killing_timer_sec = get(self.name, "killing_timer_sec")

    def initialize_traj(self, wpnt):
        if len(wpnt.wpnts) != 0:
            self.stamp = wpnt.header.stamp
            self.list = wpnt.wpnts
            self.array = np.array([[w.x_m, w.y_m, w.s_m, w.d_m] for w in wpnt.wpnts])
            self.is_init = True


def time_to_float(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class StateMachine(Node):
    """
    This state machine subscribes to topics and calculates flags/conditions.
    State transitions and state behaviors are described in `transitions.py` and `states.py`
    """

    def __init__(self) -> None:
        super().__init__(
            "state_machine",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = "state_machine"

        self.main_loop = None  # set later, referenced by params callback

        # Load planner configs (planner_name -> {param: value}) before declaring params
        self._planner_param_cache = {}
        self._load_planner_configs()

        # PARAMETER DECLARATION (replaces rospy.get_param + dyn_reconfigure)
        self.params = StateMachineParams(self)
        self.add_on_set_parameters_callback(self.params.parameters_callback)

        # Convenience aliases (kept as attributes for parity with the ROS1 code which
        # read these directly off `self`). They mirror self.params.* values.
        self.rate_hz = self.params.rate_hz
        self._rate_check = (RateCheck(
            self, nominal_hz=self.rate_hz, name="state_machine",
            consequence="every timeout counted in cycles rather than seconds -- the "
                        "static-deadlock timer, the feasibility-loss timer, the "
                        "relax repeat -- fires late by the same factor")
            if RateCheck else None)
        self.n_loc_wpnts = self.params.n_loc_wpnts
        self.timetrials_only = self.params.timetrials_only
        self.racecar_version = self.params.racecar_version
        self.ot_planner = self.params.ot_planner
        self.track_length = self.params.track_length
        self.volt_threshold = self.params.volt_threshold

        self.local_wpnts = WpntArray()
        self.waypoints_dist = 0.1  # [m]
        self.measuring = self.params.measuring

        # sectors: read the map yamls at startup, live-update from the sector tuner nodes
        # (ROS1: /map_params + /ot_map_params and the dyn_sector_* servers)
        self.map_name = self._get_str_param("map", "")
        self.sectors_params = {}
        self.ot_sectors_params = {}
        self.only_ftg_zones = []
        self.ftg_counter = 0
        # Static-obstacle trailing deadlock (see _check_static_trailing_deadlock). Shorter than
        # ftg_timer_sec: behind a stationary box there is nothing to wait for, the gap PID has
        # already converged to its only fixed point (v = 0).
        self._static_deadlock_counter = 0
        self.static_deadlock_speed_mps = 0.3
        # Mirrored so limit_local_window_accel can read them off the node like every other
        # _check_* condition does. Both are in StateMachineParams._NODE_MIRRORED_PARAMS; the alias
        # here is the half that was missing last time this pair went in, and its absence was an
        # AttributeError out of a timer callback that took the whole state machine down.
        self.local_window_accel_limit_enable = self.params.local_window_accel_limit_enable
        self.local_window_a_long_mps2 = self.params.local_window_a_long_mps2
        self.static_deadlock_timeout_s = 1.5
        # Rate limit for the /planner/avoidance/relax request the deadlock raises. Re-sent (not
        # one-shot) because the planner may not be able to act on the first one; rate-limited (not
        # per-cycle) because each request drops the planner's committed path.
        self._relax_sent_t = None
        self.relax_repeat_sec = 2.0
        # Velocity-profile cache, one slot per path source (see update_velocity). The quantum is
        # how much v_start may drift before the profile is re-solved: fine enough that the
        # commanded head speed tracks the car, coarse enough that a frozen path is solved once.
        self._vel_cache = {}
        self.vel_cache_quant_mps = 0.25
        # How many points of the path's TAIL identify its geometry for the profile cache. The tail
        # is what a forward re-slice leaves alone (the head is consumed as the car drives), so it
        # is what the key can be built from. It is only a PREFILTER -- the reuse is confirmed by
        # comparing the cached geometry against this one -- so 40 points (4 m) is enough.
        self.vel_cache_sig_pts = 40
        # obstacle id -> last time it was reported is_visible (STATIC obstacles only); see
        # obstacle_perception_cb's debounce.
        self._last_visible_t = {}
        # Tolerances for "this is the same committed path, re-sliced further forward" (see
        # _is_forward_reslice). Tight on purpose: the whole value of the rule is that adopting such
        # a path moves the window's START without moving its GEOMETRY.
        self.reslice_d_tol_m = 0.03
        self.reslice_end_s_tol_m = 0.20

        self.cur_s = 0.0
        self.cur_d = 0.0
        self.cur_vs = 0.0

        # Velocity Planning - load racecar config from stack_master
        self._load_vehicle_dynamics()

        # overtaking variables
        self.n_ot_sectors = 0
        self.overtake_wpnts = None
        self.overtake_zones = []
        self.ot_begin_margin = 0.5
        # read the map sector yamls, then build only_ftg_zones / overtake_zones
        self._load_sector_yamls()
        self._load_sector_params()
        self.cur_volt = 11.69  # default value for sim
        self.static_overtaking_mode = False

        # waypoint variables
        self.cur_id_ot = 1
        self.max_speed = -1
        self.max_s = 0
        self.current_position = None
        self.gb_wpnts = None
        self.recovery_wpnts = None
        self.gb_max_idx = None
        self.wpnt_dist = self.waypoints_dist
        self.num_glb_wpnts = 0
        self.num_ot_points = 0
        self.previous_index = 0

        # dynamic-parameter-backed attributes (aliases onto params)
        self.gb_ego_width_m = self.params.gb_ego_width_m
        self.recovery_exit_d_m = self.params.recovery_exit_d_m
        self.recovery_entry_d_m = self.params.recovery_entry_d_m
        self.lateral_width_gb_m = self.params.lateral_width_gb_m
        self.lateral_width_static_gb_m = self.params.lateral_width_static_gb_m
        # THE clearance feed (see _published_clearance). Not parameters: a staleness window and a
        # position-match radius are properties of the feed, not knobs to tune per track.
        self._clr_feed = []
        self._clr_feed_t = -1e18
        self.clearance_feed_ttl_s = 3.0
        self.clearance_match_m = 0.50
        self.gb_horizon_m = self.params.gb_horizon_m
        self.interest_horizon_m = self.params.interest_horizon_m
        self.reframe_warn_m = self.params.reframe_warn_m
        self.squeeze_speed_cap_mps = self.params.squeeze_speed_cap_mps
        self.avoidance_ay_max = self.params.avoidance_ay_max
        self.static_invisible_grace_sec = self.params.static_invisible_grace_sec
        # Frenet frame of the line the car is ACTUALLY following (/global_waypoints), rebuilt
        # whenever static_reopt swaps it. Incoming obstacles are re-anchored through this; see
        # _reframe_obstacles. None until the first global line arrives -> obstacles pass through.
        self.converter = None
        self._converter_xy = None
        self.getting_closer_rel_vel_mps = self.params.getting_closer_rel_vel_mps

        self.last_recovery_update_time = None
        self.cur_gb_wpnts = WaypointData(self, "global_tracking", True)
        self.cur_recovery_wpnts = WaypointData(self, "recovery_planner", False)
        self.cur_avoidance_wpnts = WaypointData(self, "dynamic_avoidance_planner", False)
        self.cur_static_avoidance_wpnts = WaypointData(self, "static_avoidance_planner", False)
        self.cur_start_wpnts = WaypointData(self, "start_planner", False)

        self.cur_avoidance_wpnts.is_ot_wpnts = True
        self.cur_avoidance_wpnts.is_dynamic_ot_wpnts = True
        self.cur_static_avoidance_wpnts.is_ot_wpnts = True
        self.cur_gb_wpnts.is_gb_track_wpnts = True
        self.cur_recovery_wpnts.vel_planner_safety_factor = 0.5

        self.gb_closest_target = None
        self.gb_closest_gap = None
        self.recovery_closest_target = None
        self.recovery_closest_gap = None
        self.ot_closest_target = None
        self.ot_closest_gap = None

        self.behavior_strategy = BehaviorStrategy()

        # mincurv spline
        self.mincurv_spline_x = None
        self.mincurv_spline_y = None
        # ot spline
        self.ot_spline_x = None
        self.ot_spline_y = None
        self.ot_spline_d = None
        self.recompute_ot_spline = True
        # live sector retune from the sector tuner nodes (after recompute_ot_spline exists)
        self._setup_sector_live_update()

        # obstacle avoidance variables
        self.obstacles = []
        self.obstacles_in_interest = []
        self.cur_obstacles_in_interest = []
        self.obstacles_perception = []
        self.obstacles_prediction_id = None
        self.obstacles_prediction = []
        self.obstacle_was_here = True
        self.side_by_side_threshold = 0.6
        self.force_trailing = False
        self.use_force_trailing = self.params.use_force_trailing

        # spliner variables
        self.splini_ttl = self.params.splini_ttl
        self.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)
        self.avoidance_wpnts = None
        self.static_avoidance_wpnts = None
        self.start_wpnts = None
        self.start_wpnts_array = None
        self.last_valid_avoidance_wpnts = None
        self.last_valid_avoidance_array = None
        self.last_valid_static_avoidance_wpnts = None

        self.overtaking_horizon_m = self.params.overtaking_horizon_m
        self.lateral_width_ot_m = self.params.lateral_width_ot_m
        self.splini_hyst_timer_sec = self.params.splini_hyst_timer_sec
        self.emergency_break_horizon = self.params.emergency_break_horizon
        self.emergency_break_d = 0.12  # [m]

        # Graph based variables
        self.graph_based_wpts = None
        self.gb_wpnts_arr = None
        # Frenet variables
        self.frenet_wpnts = WpntArray()

        # FTG params
        self.ftg_speed_mps = self.params.ftg_speed_mps
        self.ftg_timer_sec = self.params.ftg_timer_sec
        self.ftg_disabled = not self.params.ftg_active

        # Force GBTRACK state
        self.force_gbtrack_state = self.params.force_GBTRACK

        self.overtaking_ttl_sec = self.params.overtaking_ttl_sec
        self.overtaking_ttl_count = 0
        self.overtaking_ttl_count_threshold = int(self.overtaking_ttl_sec * self.rate_hz)

        # Feasibility signal from the static avoidance planner. FAIL-CLOSED: the planner publishes
        # it every cycle (20 Hz), so a stale or never-received signal means the planner is dead or
        # miswired — the static OVERTAKE commit must then stay blocked (trailing is the safe
        # fallback), not silently open as the old default-True did.
        self.static_avoidance_feasible = False
        self._static_feasible_t = None
        self.static_feasible_stale_sec = 0.5
        # sustain side of the same gate: last time the planner said feasible=True. A static
        # OVERTAKE is dropped (-> TRAILING) once the planner reports infeasible for longer
        # than this (single-cycle blips tolerated; riding the stale cached spline is not).
        self._static_feasible_true_t = None
        self.static_feasible_lost_sec = 0.4
        # ...and the SYMMETRIC debounce on the other sustain term. The static branch dropped
        # OVERTAKE on a single failed _check_availability while the feasibility term next to it
        # tolerated 0.4 s of blips -- so the cheaper, noisier term decided the exit. Availability
        # depends on message freshness against an executor that runs 0.3-0.5 s behind, so one late
        # publish was enough. Same window, same reason.
        self._static_avail_true_t = None
        self.static_avail_lost_sec = 0.4
        # After a static OVERTAKE drop, refuse to re-commit for this long. The drop and the
        # re-entry gate read almost the same inputs, so without it the pair oscillates at the
        # message rate: drop -> path still fresh -> re-enter -> drop. Short enough to stay a
        # debounce (24 cycles at 80 Hz), not a lockout.
        self._static_ot_cooldown_until = None
        self.static_ot_reentry_cooldown_sec = 0.3
        # Same idea for the DYNAMIC path: last time the avoidance path read free. Entry still
        # demands free right now; only staying in OVERTAKE is debounced.
        self._ot_free_true_t = None
        self.ot_free_lost_sec = self.params.ot_free_lost_sec
        self.free_check_predict_dynamic = self.params.free_check_predict_dynamic
        self.free_check_pass_speed = self.params.free_check_pass_speed
        self.free_check_dynamic_ot_slow = self.params.free_check_dynamic_ot_slow

        # Transition hysteresis (anti-chatter): a state must be held >= min_dwell_sec before it may
        # switch to a NON-safe state. Switches toward the safe states bypass this. The counter/timer
        # live on the node (not in the pure transition functions).
        self.min_dwell_sec = self.params.min_dwell_sec
        self._last_transition_time = self.now_sec()
        self._committed_src = None
        # Targets that may be entered IMMEDIATELY (bypass min_dwell): the safe-direction states
        # (TRAILING, FTGONLY) AND OVERTAKE. OVERTAKE must never be delayed by the dwell -- while
        # approaching, the SM legitimately flickers GB_TRACK<->TRAILING, which keeps resetting the
        # dwell timer; gating OVERTAKE behind it would perpetually veto the overtake commit. The
        # dwell therefore only damps the return-to-raceline direction (->GB_TRACK/RECOVERY/...).
        self._IMMEDIATE_STATES = {StateType.TRAILING, StateType.FTGONLY, StateType.OVERTAKE}

        self.save_start_traj = False
        self.cur_start_wpnts_candidate = OTWpntArray()
        self.need_start_traj = False
        # visualization variables
        self.first_visualization = True
        self.x_viz = 0
        self.y_viz = 0

        # STATES
        self.cur_state = StateType.GB_TRACK
        self.local_wpnts_src = StateType.GB_TRACK
        self.static_avoid = False
        self.fail_trailing = False

        self.states = {
            StateType.GB_TRACK: states.GlobalTracking,
            StateType.OVERTAKE: states.Overtaking,
            StateType.FTGONLY: states.FTGOnly,
            StateType.RECOVERY: states.RECOVERY,
            StateType.START: states.START,
        }
        self.state_transitions = {
            StateType.GB_TRACK: state_transitions.GlobalTrackingTransition,
            StateType.RECOVERY: state_transitions.RecoveryTransition,
            StateType.TRAILING: state_transitions.TrailingTransition,
            StateType.ATTACK: state_transitions.TrailingTransition,
            StateType.OVERTAKE: state_transitions.OvertakingTransition,
            StateType.FTGONLY: state_transitions.FTGOnlyTransition,
            StateType.START: state_transitions.StartTransition,
        }

        self.opponent = ObstacleArray()

        qos = QoSProfile(depth=10)

        # SUBSCRIPTIONS
        self.create_subscription(Odometry, "/car_state/odom", self.odom_cb, qos)
        self._wait_for_attr("current_position", "/car_state/odom")

        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.glb_wpnts_cb, qos)
        self.create_subscription(WpntArray, "/planner/recovery/wpnts", self.recovery_wpnts_cb, qos)
        self.create_subscription(WpntArray, "/global_waypoints/overtaking", self.overtake_cb, qos)
        self._wait_for_attr("gb_wpnts", "/global_waypoints_scaled")
        self._wait_for_attr("overtake_wpnts", "/global_waypoints/overtaking")

        self.create_subscription(Odometry, "/car_state/odom_frenet", self.frenet_pose_cb, qos)
        self.create_subscription(WpntArray, "/global_waypoints", self.glb_wpnts_og_cb, qos)
        # THE clearance definition: how far the line static_reopt publishes passes each confirmed
        # static obstacle. Consumed in the static free-check instead of re-deriving a lateral
        # distance in this node's frame -- see the note there.
        self.create_subscription(Float32MultiArray, "/static_reopt/clearance",
                                 self.static_clearance_cb, 1)

        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacle_perception_cb, qos)
        self.create_subscription(
            PredictionArray, "/opponent_prediction/obstacles_pred", self.obstacle_prediction_cb, qos
        )

        if self.ot_planner == "spliner" or self.ot_planner == "predictive_spliner":
            self.create_subscription(OTWpntArray, "/planner/avoidance/otwpnts", self.avoidance_cb, qos)
            if self.ot_planner == "predictive_spliner":
                self.create_subscription(
                    OTWpntArray, "/planner/avoidance/static_otwpnts", self.static_avoidance_cb, qos
                )
                # Feasibility gate from the static (Frenet-sampling) avoidance planner: False means
                # it found no passable candidate -> the SM must not commit to a static OVERTAKE.
                self.create_subscription(
                    Bool, "/planner/avoidance/static_feasible", self.static_feasible_cb, qos
                )
        if self.ot_planner == "predictive_spliner":
            self.create_subscription(Bool, "/opponent_prediction/force_trailing", self.force_trailing_cb, qos)
            self.create_subscription(Bool, "planner/avoidance/fail_trailing", self.fail_trailing_cb, qos)

        if not self.params.sim:
            self.create_subscription(VescStateStamped, "/vesc/sensors/core", self.vesc_state_cb, qos)

        self.create_subscription(OTWpntArray, "/planner/start_wpnts", self.start_wpnts_cb, qos)
        self.create_subscription(Bool, "/save_start_traj", self.save_start_traj_cb, qos)

        # PUBLICATIONS
        self.behavior_strategy_pub = self.create_publisher(BehaviorStrategy, "behavior_strategy", 1)
        self.trailing_marker_pub = self.create_publisher(Marker, "/state_machine/trailing_target", 10)
        self.overtaking_marker_pub = self.create_publisher(Marker, "/state_machine/overtaking_target", 10)
        self.loc_wpnt_pub = self.create_publisher(WpntArray, "local_waypoints", 1)
        self.vis_loc_wpnt_pub = self.create_publisher(MarkerArray, "local_waypoints/markers", 10)
        self.state_pub = self.create_publisher(String, "state_machine", 1)
        self.state_mrk = self.create_publisher(Marker, "/state_marker", 10)
        self.emergency_pub = self.create_publisher(Marker, "/emergency_marker", 5)
        self.ot_section_check_pub = self.create_publisher(Bool, "/ot_section_check", 1)
        # Deadlock recovery request to the static planner: "you reported no feasible path and the
        # car has now been stopped behind that obstacle for static_deadlock_timeout_s -- retry it
        # at reduced margins". Absolute name and deliberately NOT remapped per-planner, mirroring
        # /planner/avoidance/static_feasible: it is a static-planner interlock, not an OT lane.
        self.relax_pub = self.create_publisher(Bool, "/planner/avoidance/relax", 10)
        # ROS1 published this from dynamic_statemachine_server when the save_start_traj
        # rqt button was pressed; re-homed here as a momentary param (see loop()).
        self.save_start_traj_pub = self.create_publisher(Bool, "/save_start_traj", 1)
        self._save_start_traj_requested = False
        self._save_params_requested = False
        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/state_machine/latency", 10)

        # MAIN LOOP at fixed rate
        self.main_loop = self.create_timer(1.0 / self.rate_hz, self.loop)

    # ---------------------------------------------------------------------- #
    # SETUP HELPERS                                                           #
    # ---------------------------------------------------------------------- #
    def _wait_for_attr(self, attr, topic):
        """rclpy equivalent of rospy.wait_for_message."""
        while rclpy.ok() and getattr(self, attr, None) is None:
            self.get_logger().info(f"Waiting for message on {topic}", throttle_duration_sec=1.0)
            rclpy.spin_once(self, timeout_sec=0.1)

    def _load_planner_configs(self):
        """Load the per-planner yaml files shipped in this package's config/planners dir
        and declare them as nested ROS2 parameters (<planner>.<key>)."""
        import yaml

        try:
            share = get_package_share_directory("state_machine")
        except Exception:
            share = None

        planner_names = [
            "global_tracking",
            "recovery_planner",
            "dynamic_avoidance_planner",
            "static_avoidance_planner",
            "start_planner",
        ]
        for pname in planner_names:
            data = {}
            if share is not None:
                cfg = os.path.join(share, "config", "planners", pname + ".yaml")
                if os.path.exists(cfg):
                    with open(cfg, "r") as f:
                        data = yaml.safe_load(f) or {}
            self._planner_param_cache[pname] = data
            for key, val in data.items():
                pname_param = f"{pname}.{key}"
                try:
                    self.declare_parameter(pname_param, val)
                except Exception:
                    pass

    def _get_str_param(self, name, default=""):
        try:
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            v = self.get_parameter(name).value
            return v if v is not None else default
        except Exception:
            return default

    def _load_sector_yamls(self):
        # read the map sector yamls into sectors_params / ot_sectors_params (ROS1 /map_params, /ot_map_params)
        import yaml
        try:
            maps_dir = os.path.join(get_package_share_directory("stack_master"), "maps", self.map_name)
        except Exception:
            self.get_logger().warn(f"[{self.name}] could not locate stack_master maps dir; no sectors loaded")
            return
        sp = os.path.join(maps_dir, "speed_scaling.yaml")
        if os.path.exists(sp):
            with open(sp, "r") as f:
                d = yaml.safe_load(f) or {}
            self.sectors_params = (d.get("speed_sector_tuner", {}) or {}).get("ros__parameters", {}) or {}
        else:
            self.get_logger().warn(f"[{self.name}] {sp} not found; no FTG-only zones")
        op = os.path.join(maps_dir, "ot_sectors.yaml")
        if os.path.exists(op):
            with open(op, "r") as f:
                d = yaml.safe_load(f) or {}
            self.ot_sectors_params = (d.get("ot_sector_tuner", {}) or {}).get("ros__parameters", {}) or {}
            self.ot_begin_margin = float(self.ot_sectors_params.get("ot_sector_begin", self.ot_begin_margin))
        else:
            self.get_logger().warn(f"[{self.name}] {op} not found; no overtake zones")

    def _load_sector_params(self):
        # build zones from the sector dicts (ROS1 sector_dyn_param_cb / ot_dyn_param_cb)
        self.only_ftg_zones = []
        self.n_sectors = int(self.sectors_params.get("n_sectors", 0))
        for i in range(self.n_sectors):
            sec = self.sectors_params.get(f"Sector{i}", {}) or {}
            if sec.get("only_FTG", False):
                # end+1 == next sector's start: close the 1-index gap so adjacent FTG
                # sectors don't briefly drop to GB_TRACK (ROS1 used [start, end]).
                self.only_ftg_zones.append([sec.get("start", 0), sec.get("end", 0) + 1])

        self.overtake_zones = []
        self.n_ot_sectors = int(self.ot_sectors_params.get("n_sectors", 0))
        for i in range(self.n_ot_sectors):
            sec = self.ot_sectors_params.get(f"Overtaking_sector{i}", {}) or {}
            if sec.get("ot_flag", False):
                self.overtake_zones.append([sec.get("start", 0), sec.get("end", 0) + 1])

    def _setup_sector_live_update(self):
        # ROS2 replacement of ROS1 /dyn_sector_speed & /dyn_sector_overtake subscriptions
        from rclpy.parameter_event_handler import ParameterEventHandler
        self._sector_evt_handler = ParameterEventHandler(self)
        self._sector_evt_cb_handle = self._sector_evt_handler.add_parameter_event_callback(
            self._sector_param_event_cb)

    @staticmethod
    def _param_msg_value(p):
        # rcl_interfaces/Parameter -> python value (bool/int/double only needed here)
        t = p.value.type
        if t == 1:
            return p.value.bool_value
        if t == 2:
            return p.value.integer_value
        if t == 3:
            return p.value.double_value
        return None

    def _sector_param_event_cb(self, event):
        node = event.node.lstrip("/")
        if node == "speed_sector_tuner":
            for p in list(event.new_parameters) + list(event.changed_parameters):
                if p.name.startswith("Sector") and p.name.endswith(".only_FTG"):
                    key = p.name.split(".")[0]
                    self.sectors_params.setdefault(key, {})["only_FTG"] = bool(self._param_msg_value(p))
            self._load_sector_params()
        elif node == "ot_sector_tuner":
            for p in list(event.new_parameters) + list(event.changed_parameters):
                if p.name.startswith("Overtaking_sector") and p.name.endswith(".ot_flag"):
                    key = p.name.split(".")[0]
                    self.ot_sectors_params.setdefault(key, {})["ot_flag"] = bool(self._param_msg_value(p))
                elif p.name == "ot_sector_begin":
                    self.ot_begin_margin = float(self._param_msg_value(p))
                    self.recompute_ot_spline = True
            self._load_sector_params()

    def get_planner_param(self, planner_name, key):
        """Read a planner parameter; falls back to cached yaml value."""
        full = f"{planner_name}.{key}"
        if self.has_parameter(full):
            return self.get_parameter(full).value
        return self._planner_param_cache.get(planner_name, {}).get(key)

    def _load_vehicle_dynamics(self):
        """Load veh params + ggv / ax_max machine info from stack_master config."""
        self.pars = {}
        try:
            stack_master_path = get_package_share_directory("stack_master")
        except Exception:
            stack_master_path = None

        parser = configparser.ConfigParser()
        ini_ok = False
        if stack_master_path is not None:
            ini_path = os.path.join(
                stack_master_path, "config", self.params.racecar_version, "racecar_f110.ini"
            )
            ini_ok = bool(parser.read(ini_path))

        if not ini_ok:
            # Sim / missing config fallback: provide sane defaults so the node still runs.
            self.get_logger().warn(
                "racecar_f110.ini not found; using default vehicle params (velocity replanning degraded)"
            )
            self.pars["veh_params"] = {
                "v_max": 7.0, "length": 0.535, "width": 0.3,
                "mass": 3.5, "dragcoeff": 0.0136, "g": 9.81,
            }
            self.pars["vel_calc_opts"] = {"dyn_model_exp": 1.0, "vel_profile_conv_filt_window": None}
            self.ggv = None
            self.ax_max_machines = None
            self.b_ax_max_machines = None
            return

        self.pars["veh_params"] = json.loads(parser.get("GENERAL_OPTIONS", "veh_params"))
        self.pars["vel_calc_opts"] = json.loads(parser.get("GENERAL_OPTIONS", "vel_calc_opts"))
        vdyn = os.path.join(stack_master_path, "config", self.params.racecar_version, "veh_dyn_info")
        ggv_path = os.path.join(vdyn, "ggv.csv")
        ax_max_path = os.path.join(vdyn, "ax_max_machines.csv")
        b_ax_max_path = os.path.join(vdyn, "b_ax_max_machines.csv")
        self.ggv, self.ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
            ggv_import_path=ggv_path, ax_max_machines_import_path=ax_max_path
        )
        _, self.b_ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
            ggv_import_path=ggv_path, ax_max_machines_import_path=b_ax_max_path
        )

    def now_sec(self) -> float:
        return time_to_float(self.get_clock().now().to_msg())

    def _commit_state(self, proposed_state, proposed_src, force=False):
        """Apply a proposed (state, wpnts_src) with min_dwell transition hysteresis.

        A switch to a dwell-gated state is vetoed if it comes sooner than ``min_dwell_sec`` after the
        last committed switch; on veto the previous state and its behaviour source are held for this
        cycle. Switches into an immediate state (TRAILING, FTGONLY, OVERTAKE), staying in the same
        state, and forced overrides (force_GBTRACK / FTGONLY sector) always commit immediately.
        """
        allow = (
            force
            or proposed_state == self.cur_state
            or proposed_state in self._IMMEDIATE_STATES
            or (self.now_sec() - self._last_transition_time) >= self.min_dwell_sec
        )
        if allow:
            if proposed_state != self.cur_state:
                self._last_transition_time = self.now_sec()
            self.cur_state = proposed_state
            self.local_wpnts_src = proposed_src
            self._committed_src = proposed_src
        else:
            # hold the current state; reuse the last committed behaviour source for consistency
            self.local_wpnts_src = self._committed_src if self._committed_src is not None else proposed_src

    def _update_overtake_ttl(self, prev_state, proposed_state):
        """Node-owned replacement for the counter mutation that used to live in
        OvertakingTransition (which violated the 'transitions have no side effects' rule). Mirrors
        the old latch: while staying in OVERTAKE, count up as long as the OT path is sustainable but
        no enemy is directly ahead; reset on enemy / loss of sustainability / leaving OVERTAKE."""
        if prev_state == StateType.OVERTAKE and proposed_state == StateType.OVERTAKE:
            if self._check_enemy_in_front():
                self.overtaking_ttl_count = 0
            else:
                self.overtaking_ttl_count += 1
        else:
            self.overtaking_ttl_count = 0

    #############
    # CALLBACKS #
    #############
    def save_start_traj_cb(self, msg):
        if len(self.cur_start_wpnts_candidate.wpnts) != 0:
            self.update_velocity(self.cur_start_wpnts_candidate,
                                 self.cur_start_wpnts.vel_planner_safety_factor,
                                 cache_key="start")
            self.cur_start_wpnts.initialize_traj(self.cur_start_wpnts_candidate)
            self.cur_state = StateType.START

    def vesc_state_cb(self, data):
        self.cur_volt = data.state.voltage_input

    def frenet_planner_cb(self, data: WpntArray):
        self.frenet_wpnts = data

    def recovery_wpnts_cb(self, data: WpntArray):
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_recovery_wpnts.vel_planner_safety_factor,
                                 cache_key="recovery")
        self.recovery_wpnts = data

    def avoidance_cb(self, data: OTWpntArray):
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_avoidance_wpnts.vel_planner_safety_factor,
                                 ay_max=self.avoidance_ay_max, cache_key="dynamic")
        self.avoidance_wpnts = data

    def static_avoidance_cb(self, data: OTWpntArray):
        if len(data.wpnts) != 0:
            # ot_line == "squeeze": the planner found no candidate at its design margins and solved
            # this one at reduced clearance instead (the alternative there being a TRAILING
            # standstill). The geometry is legal but the error budget is spent, so it must not be
            # driven at raceline pace -- the planner cannot enforce that itself because this node
            # owns the velocity profile of every path it receives.
            cap = self.squeeze_speed_cap_mps if data.ot_line == "squeeze" else None
            self.update_velocity(data, self.cur_static_avoidance_wpnts.vel_planner_safety_factor,
                                 v_cap=cap, ay_max=self.avoidance_ay_max, cache_key="static")
        self.static_avoidance_wpnts = data

    def start_wpnts_cb(self, data: OTWpntArray):
        if len(data.wpnts) != 0:
            self.cur_start_wpnts_candidate = data

    def overtake_cb(self, data):
        self.overtake_wpnts = data.wpnts
        self.num_ot_points = len(self.overtake_wpnts)
        if self.recompute_ot_spline and self.num_ot_points != 0:
            self.ot_splinification()
            self.recompute_ot_spline = False

    def glb_wpnts_cb(self, data: WpntArray):
        # last point's s == loop length (ROS1 read this from /global_republisher/track_length)
        track_len = data.wpnts[-1].s_m
        data.wpnts = data.wpnts[:-1]  # exclude last point (== first)
        self.gb_wpnts = data
        self.num_glb_wpnts = len(data.wpnts)
        self.n_loc_wpnts = min(self.n_loc_wpnts, int(self.num_glb_wpnts / 2))
        self.max_s = data.wpnts[-1].s_m
        if track_len > 1.0:
            self.track_length = track_len
        self.wpnt_dist = data.wpnts[1].s_m - data.wpnts[0].s_m
        self.gb_max_idx = data.wpnts[-1].id
        if self.ot_planner == "graph_based":
            self.gb_wpnts_arr = np.array([
                [w.s_m, w.d_m, w.x_m, w.y_m, w.d_right, w.d_left, w.psi_rad,
                 w.kappa_radpm, w.vx_mps, w.ax_mps2] for w in data.wpnts
            ])

    def glb_wpnts_og_cb(self, data):
        if self.max_speed == -1:
            self.max_speed = max([wpnt.vx_mps for wpnt in data.wpnts])
        # /global_waypoints -- NOT the scaled copy this node's geometry comes from -- is the line
        # the frenet republisher parameterises cur_s/cur_d on and the line static_reopt swaps. It
        # is therefore the frame in which "d = 0" means "on the line the car is following", which
        # is exactly what the free-check assumes when it compares an obstacle's d against the
        # raceline. Rebuild the converter on an ACTUAL geometry change only (no per-message churn:
        # the topic is republished on a keep-alive timer).
        try:
            xy = np.array([[w.x_m, w.y_m] for w in data.wpnts], dtype=float)
            if (self._converter_xy is not None and xy.shape == self._converter_xy.shape
                    and np.allclose(xy, self._converter_xy)):
                return
            psi = np.array([w.psi_rad for w in data.wpnts], dtype=float)
            self.converter = FrenetConverter(xy[:, 0], xy[:, 1], psi)
            self._converter_xy = xy
            self.get_logger().info(f"[{self.name}] (re)built FrenetConverter on /global_waypoints "
                                   f"({len(data.wpnts)} pts)")
        except Exception as e:  # noqa: BLE001 -- never let this kill the SM's obstacle input
            self.get_logger().warn(f"[{self.name}] FrenetConverter rebuild failed: {e}")

    def _reframe_obstacles(self, obstacles):
        """Re-express each obstacle's (s,d) in THIS node's frenet frame, from its map (x_m,y_m).

        Same fix as fa0b974 made in static_avoidance_node, applied to the consumer that decides
        whether the car goes TRAILING. The free-check's static branch reduces to

            |ot_d - obs.d_center| - obs.size/2 - gb_ego_width_m/2  >=  required margin

        with ot_d = 0 on the global line, so obs.d_center is asked to be the obstacle's lateral
        offset from the line the car is following RIGHT NOW. What arrives is whatever (s,d) the
        tracker computed in ITS copy of the line. Before the first static_reopt swap the two agree;
        after one they do not, and the SM then judges the swapped line using offsets measured on
        the line it replaced -- reading an obstacle the new line drives 0.5 m around as sitting on
        it (phantom TRAILING), or an obstacle it passes close to as cleared.

        x_m,y_m is frame-independent, so re-projecting from it is correct by construction whatever
        upstream did. Box extents are carried as offsets from the centre rather than rebuilt from
        `size`, so the footprint every downstream check sees is unchanged.

        NOT covered: /opponent_prediction/obstacles_pred carries its own pred_s/pred_d, produced by
        the predictor in its own frame. That is the DYNAMIC branch of the free-check and needs the
        predictor to re-project; it is not something this node can fix from x_m,y_m.
        """
        conv = self.converter
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
                f"on a static_reopt line swap — the free-check would have judged the swapped line "
                f"against offsets measured on the old one.",
                throttle_duration_sec=5.0)
        return obstacles

    def graphbased_wpts_cb(self, data):
        arr = np.asarray(data.data)
        self.graph_based_wpts = arr.reshape(data.layout.dim[0].size, data.layout.dim[1].size)
        self.graph_based_action = data.layout.dim[0].label

    def obstacle_perception_cb(self, data):
        if not self.timetrials_only:
            # Re-anchor FIRST: every (s,d) read below -- the interest-window gap, the free-check's
            # d_center, the trailing target -- has to be in the frame of the line the car is
            # following, not the one the tracker happened to hold.
            data.obstacles = self._reframe_obstacles(data.obstacles)
            self.obstacles_perception = data.obstacles[:]
            self.obstacles = data.obstacles
            obstacles_in_interest = []
            now = self.now_sec()
            for obs in data.obstacles:
                # detection-gated: ignore a remembered (currently-unseen) STATIC obstacle so state
                # decisions track what the car actually sees, not a stored position ("knows in
                # advance"). Dynamic obstacles are left as-is (handled via prediction + short ttl).
                #
                # DEBOUNCED. is_visible is a per-frame verdict from a lidar that a static box
                # occludes, clips at the FOV edge, and returns few points from at range -- so it
                # drops out for single frames while the obstacle is plainly still there. Acting on
                # each drop emptied obstacles_in_interest for that cycle, which flipped the state
                # (ObstacleTransition -> NonObstacleTransition) and the trailing target with it.
                # A static obstacle cannot leave, so a brief loss of sight is a sensing artifact,
                # not news: hold it for static_invisible_grace_sec after it was last seen. The
                # detection gate is preserved -- an obstacle never seen, or gone for longer than
                # the grace period, is still ignored.
                if obs.is_static:
                    if obs.is_visible:
                        self._last_visible_t[obs.id] = now
                    else:
                        seen = self._last_visible_t.get(obs.id)
                        if seen is None or (now - seen) > self.static_invisible_grace_sec:
                            continue
                gap = (obs.s_start - self.cur_s) % self.track_length
                if gap < self.interest_horizon_m:
                    obstacles_in_interest.append((gap, obs))
            # Sort by forward gap so [0] is always the nearest obstacle ahead. Several
            # checks (_check_getting_closer) only look at index 0, which is only correct
            # if the list is ordered (perception does not guarantee any order).
            obstacles_in_interest.sort(key=lambda g_obs: g_obs[0])
            self.obstacles_in_interest = [obs for _, obs in obstacles_in_interest]
            # Bound the visibility memory: a track that has been out of sight for well past the
            # grace period is gone, and its id will not come back (the tracker issues a new one).
            if len(self._last_visible_t) > 32:
                cutoff = now - 2.0 * self.static_invisible_grace_sec
                self._last_visible_t = {k: t for k, t in self._last_visible_t.items() if t > cutoff}

    def obstacle_prediction_cb(self, data):
        if len(data.predictions) != 0:
            self.obstacles_prediction_id = data.id
            self.obstacles_prediction = data.predictions
        else:
            self.obstacles_prediction = []

    def static_clearance_cb(self, msg: Float32MultiArray):
        d = list(msg.data)
        self._clr_feed = [tuple(d[i:i + 4]) for i in range(0, len(d) - 3, 4)]
        self._clr_feed_t = time.time()

    def _published_clearance(self, obs):
        """static_reopt's measured clearance of the GB line past `obs`, or None if none is fresh.

        Matched by MAP POSITION: this node carries tracker ids while static_reopt carries the
        static layer's marker ids, and the two id spaces are unrelated."""
        feed = getattr(self, "_clr_feed", None)
        if not feed or (time.time() - getattr(self, "_clr_feed_t", -1e18)) > self.clearance_feed_ttl_s:
            return None
        ox, oy = float(getattr(obs, "x_m", float("nan"))), float(getattr(obs, "y_m", float("nan")))
        if ox != ox or oy != oy:
            return None
        best, best_d = None, self.clearance_match_m
        for (x, y, _r, clr) in feed:
            dd = float(np.hypot(x - ox, y - oy))
            if dd <= best_d:
                best, best_d = float(clr), dd
        return best

    def frenet_pose_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x
        if self.num_ot_points != 0:
            self.cur_id_ot = int(self._find_nearest_ot_s())

    def odom_cb(self, data):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        q = data.pose.pose.orientation
        # transforms3d uses [w, x, y, z]
        _, _, theta = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])
        self.current_position = [x, y, theta]

    def force_trailing_cb(self, data):
        self.force_trailing = data.data if self.use_force_trailing else False

    def fail_trailing_cb(self, data):
        self.fail_trailing = data.data

    def static_feasible_cb(self, data):
        self.static_avoidance_feasible = data.data
        self._static_feasible_t = self.now_sec()
        if data.data:
            self._static_feasible_true_t = self.now_sec()

    ######################################
    # ATTRIBUTES/CONDITIONS CALCULATIONS #
    ######################################
    def _check_only_ftg_zone(self) -> bool:
        ftg_only = False
        if len(self.only_ftg_zones) != 0:
            for sector in self.only_ftg_zones:
                if sector[0] <= self.cur_s / self.wpnt_dist <= sector[1]:
                    ftg_only = True
                    break
        return ftg_only

    def _check_close_to_raceline(self, threshold_m=None) -> bool:
        if threshold_m is None:
            return np.abs(self.cur_d) < self.gb_ego_width_m
        else:
            return np.abs(self.cur_d) < threshold_m

    def _check_line_lost(self) -> bool:
        """RECOVERY ENTRY gate: are we far enough off the raceline that the recovery spline is
        worth following?

        Deliberately NOT the `close_to_raceline` flag the transitions pass around. That flag is
        the EXIT hysteresis (recovery_exit_d_m, plus a 20 deg heading term) and it is computed
        differently per state: GlobalTrackingTransition uses gb_ego_width_m (0.4 m, lateral only)
        while Trailing/Overtaking/Recovery/Start/FTGOnly use the much tighter recovery_exit_d_m.
        Driving RECOVERY entry off it therefore made the bar state-dependent -- at |d| = 0.3 m the
        car happily stays GB_TRACK, but the moment an obstacle put it in TRAILING the very same
        offset read as 'off the line' and dropped it into RECOVERY. It then followed the recovery
        spline, which is anchored at the car and so preserved the offset, keeping |d| large and
        the state latched. That is why the symptom only ever showed up while trailing.
        """
        return bool(np.abs(self.cur_d) >= self.recovery_entry_d_m)

    def _check_close_to_raceline_heading(self, threshold_deg=20) -> bool:
        # True when the ego heading is aligned with the closest raceline waypoint within
        # threshold_deg. The heading error is wrapped to (-pi, pi] so the seam (psi near
        # +/-pi) doesn't produce a spurious ~2*pi error.
        # NOTE: the previous threshold_deg branch compared self.cur_d (lateral metres)
        # against deg2rad(threshold_deg) (radians) -- it never checked heading at all.
        cloest_wpnt_idx = int(self.cur_s / self.wpnt_dist) % self.num_glb_wpnts   # live spacing
        cloest_wpnt_psi = self.cur_gb_wpnts.list[cloest_wpnt_idx].psi_rad
        heading_err = (self.current_position[2] - cloest_wpnt_psi + np.pi) % (2 * np.pi) - np.pi
        return np.abs(heading_err) < np.deg2rad(threshold_deg)

    def _gb_speed_at(self, s: float) -> float:
        """Raceline speed at s -- what the ego runs once an overtake is committed."""
        if self.gb_wpnts is None or not self.wpnt_dist:
            return 0.0
        wp = self.gb_wpnts.wpnts
        if not wp:
            return 0.0
        return float(wp[int(s / self.wpnt_dist) % len(wp)].vx_mps)

    def _check_ot_sector(self) -> bool:
        """Is the car inside an overtaking sector? PURE -- see _publish_ot_section_check."""
        # ROS1: no overtake zone matching cur_s -> not in an OT sector (return False).
        # (An empty overtake_zones means overtaking is suppressed, as in ROS1.)
        for sector in self.overtake_zones:
            if sector[0] <= self.cur_s / self.wpnt_dist <= sector[1]:
                return True
        return False

    def _publish_ot_section_check(self):
        """Publish /ot_section_check every cycle, from the main loop.

        This used to be a side effect inside _check_ot_sector(), which only runs as the first
        term of _check_overtaking_mode()'s and-chain -- i.e. only from ObstacleTransition. So
        the topic fell silent in exactly the states where it matters most:

          * while the SM holds OVERTAKE (OvertakingTransition never calls it),
          * whenever ObstacleTransition returns early into RECOVERY,
          * and entirely when force_trailing short-circuits before the call.

        The lane-change planner fails its gate CLOSED after 1 s of staleness, so the moment its
        maneuver ended it could not re-engage: it went quiet, the SM rode the cached path until
        it aged out, dropped to TRAILING, published again, and the whole thing repeated. That
        limit cycle is what the car felt as the overtake path appearing and vanishing.

        Whether the car is in an overtaking sector is a property of its POSITION, not of which
        branch the state machine happened to evaluate this tick.
        """
        if self.cur_s is None or not self.wpnt_dist:
            return
        self.ot_section_check_pub.publish(Bool(data=bool(self._check_ot_sector())))

    def _check_getting_closer(self, threshold_m=3.0) -> bool:
        # True when the nearest obstacle ahead is within threshold_m AND we are closing on it.
        # NOTE: threshold_m was previously declared but never used -- the distance gate was
        # silently dropped, so this returned True for a closing obstacle anywhere on the track.
        # Honour it now so the overtake decision commits inside a sane window (the callers pass
        # 7-10 m, matching the overtaking horizon) instead of from across the lap.
        if len(self.obstacles_in_interest) == 0:
            return False
        nearest = self.obstacles_in_interest[0]
        gap = (nearest.s_start - self.cur_s) % self.track_length
        closing = (self.cur_vs - nearest.vs) > self.getting_closer_rel_vel_mps
        return bool(gap < threshold_m and closing)

    def _check_enemy_in_front(self) -> bool:
        horizon = self.gb_horizon_m
        for obs in self.obstacles:
            gap = (obs.s_start - self.cur_s) % self.track_length
            if gap < horizon:
                return True
        return False

    def _check_latest_wpnts(self, src_wpnts, wpnts_data: WaypointData):
        """Is the planner publishing a fresh path AND is the car on the cached one?

        PURE. This used to ADOPT the published path into the cache as a side effect, which made it
        the second writer of a cache update_waypoints also writes -- with a different policy. The
        two disagreed: update_waypoints deliberately KEEPS a committed path through small
        cycle-to-cycle changes, while this adopted whatever arrived, from whichever gate happened
        to run first that cycle. Which one won depended on the transition path taken, so the cached
        geometry (and therefore the published window) could change without any of the rules that
        are supposed to govern it firing. Adoption now lives only in update_waypoints.
        """
        if src_wpnts is None or len(src_wpnts.wpnts) == 0:
            return False
        if (self.now_sec() - time_to_float(src_wpnts.header.stamp)) > wpnts_data.latest_threshold:
            return False
        return bool(self._check_on_spline(wpnts_data))

    def _check_static_trailing_deadlock(self) -> bool:
        """TRAILING behind a STATIC obstacle is a dead end, and nothing said so.

        The gap PID targets `trailing_vel_gain*v + trailing_gap` behind its target. Behind a moving
        opponent that settles at the opponent's speed; behind a stationary box the only fixed point
        is v = 0, so the car creeps to a standstill and sits there. Measured on the real car (bag
        verify_0731_2114): after the reactive planner went infeasible at t=53.4 the car ran
        3.10 -> 1.50 -> 0.06 m/s and then held 0.00 m/s for the last 8 s of the run, ~1.4 m short of
        the box, with no state change and no log line explaining it.

        Detecting it is not enough: this used to log and hand over to FTG, which is off by default
        (`ftg_active: false`) and, on the real car, has no /scan to work from at all -- so the
        "escape" was a log line and the car sat there. The deadlock now REQUESTS one, on
        /planner/avoidance/relax: the static planner answers it with a reduced-margin retry
        (squeeze pass) of the section it just failed to solve. The request is what makes this a
        recovery rather than a post-mortem, and it does not depend on FTG being enabled.

        Re-sent every `relax_repeat_sec` while the deadlock persists rather than once, because the
        planner may not be able to act on the first one (a stale commit still being unwound, the
        obstacle momentarily untracked), and rate-limited rather than sent at 80 Hz because each
        one drops the planner's committed path.
        """
        target = None
        if self.obstacles_in_interest:
            target = self.obstacles_in_interest[0]
        stalled = (self.cur_state == StateType.TRAILING
                   and abs(self.cur_vs) < self.static_deadlock_speed_mps
                   and target is not None and target.is_static)
        if not stalled:
            self._static_deadlock_counter = 0
            return False
        self._static_deadlock_counter += 1
        if self._static_deadlock_counter <= self.static_deadlock_timeout_s * self.rate_hz:
            return False
        gap = (target.s_start - self.cur_s) % self.track_length
        now = self.now_sec()
        asked = False
        if (self._relax_sent_t is None) or (now - self._relax_sent_t) >= self.relax_repeat_sec:
            self.relax_pub.publish(Bool(data=True))
            self._relax_sent_t = now
            asked = True
        self.get_logger().error(
            f"[{self.name}] STATIC TRAILING DEADLOCK: stopped ({self.cur_vs:+.2f} m/s) "
            f"{gap:.2f} m behind static obstacle id={target.id} for "
            f"{self._static_deadlock_counter / self.rate_hz:.1f} s. The avoidance planner is not "
            f"offering a usable path (static_feasible={self.static_avoidance_feasible}). "
            + ("Requested a reduced-margin retry on /planner/avoidance/relax."
               if asked else "Reduced-margin retry already requested; waiting."),
            throttle_duration_sec=2.0)
        return True

    def _check_ftg(self) -> bool:
        threshold = self.ftg_timer_sec * self.rate_hz
        # A static-obstacle deadlock is exactly the situation FTG exists for, and the car can be
        # fully stopped there (cur_vs == 0), so it satisfies the speed test below anyway — but the
        # dedicated check reports it and uses its own, shorter timeout.
        deadlock = self._check_static_trailing_deadlock()
        if self.ftg_disabled:
            return False
        elif deadlock:
            return True
        else:
            if (self.cur_state == StateType.TRAILING or self.cur_state == StateType.ATTACK) and \
                    self.cur_vs < self.ftg_speed_mps:
                self.ftg_counter += 1
                self.get_logger().warn(
                    f"[{self.name}] FTG counter: {self.ftg_counter}/{threshold}",
                    throttle_duration_sec=0.5,
                )
            else:
                self.ftg_counter = 0
            return self.ftg_counter > threshold

    def _check_on_spline(self, wpnt_data) -> bool:
        if wpnt_data.is_init:
            gap = (wpnt_data.list[-1].s_m - self.cur_s) % self.track_length
            min_dist = np.min(np.linalg.norm(wpnt_data.array[:, 0:2] - self.current_position[:2], axis=1))
            if gap > wpnt_data.on_spline_front_horizon_thres_m and min_dist < wpnt_data.on_spline_min_dist_thres_m:
                return True
        return False

    def _check_free_frenet(self, wpnts_data) -> bool:
        is_free = True
        closest_obs = None
        min_gap = 2.0
        max_horizon = wpnts_data.max_horizon
        is_gb_track_wpnts = wpnts_data.is_gb_track_wpnts
        is_ot_wpnts = wpnts_data.is_ot_wpnts
        is_dynamic_ot_wpnts = wpnts_data.is_dynamic_ot_wpnts
        free_scaling_reference_distance_m = wpnts_data.free_scaling_reference_distance_m
        lateral_width_m = wpnts_data.lateral_width_m

        obstacles = self.cur_obstacles_in_interest
        obstacle_predictions = self.obstacles_prediction

        if wpnts_data.is_init:
            max_gap = (wpnts_data.array[-1, 2] - self.cur_s) % self.max_s
            for obs in obstacles:
                obs_s = obs.s_center
                gap = (obs_s - self.cur_s) % self.max_s
                # Closing speed for the alongside window (ttc..tt0). NOT the current one:
                # while TRAILING the controller holds station behind the opponent, so
                # cur_vs - obs.vs collapses to ~0 and the 0.5 m/s floor then places the window
                # metres further down the track than the pass will ever reach. The avoidance
                # path is velocity-replanned from its curvature on commit, so the ego runs the
                # local raceline pace -- that is the speed the pass actually happens at.
                # Observed: v_pass 4.0 vs opponent 2.0 (real closing 2.0) was timed at 0.5,
                # pushing the check to 11 m ahead and forcing a 7 m offset hold that any
                # narrow spot in between then vetoed.
                relative_vs = self.cur_vs - obs.vs
                if self.free_check_pass_speed:
                    relative_vs = max(relative_vs, self._gb_speed_at(self.cur_s) - obs.vs)
                clip_vs = max(relative_vs, 0.5)
                ttc = (gap - self.pars["veh_params"]["length"]) / clip_vs
                tt0 = (gap + 0.3 * self.pars["veh_params"]["length"]) / clip_vs

                # Treat near-stationary obstacles as static regardless of the (noisy, laggy)
                # tracking is_static flag: a static obstacle transiently classified dynamic would
                # otherwise be checked against a bogus predicted trajectory, making the static
                # avoidance spline read "not free" and delaying the TRAILING->OVERTAKE switch.
                #
                # ...but NOT when judging a DYNAMIC overtaking path. The static branch evaluates
                # the lane at the obstacle's CURRENT s, and the lane-hold planner's geometry
                # contract is the opposite: minimum excursion keeps the lane on the raceline
                # (d ~ 0) until the meeting point and only clears the opponent inside
                # [meet_s - pass_overlap_m, meet_s + pass_hold]. With meet_s = s_o + v_opp*gap/
                # closing, an opponent at 0.49 m/s has its clearance band starting PAST obs_s, so
                # the static branch samples the lane where it is still on the raceline, reads
                # NOT-free, and blocks OVERTAKE structurally and permanently. The reclassification
                # was written to protect the STATIC spline (see the comment above) -- the dynamic
                # path was never its intended target. A genuinely stationary obstacle routed to
                # the dynamic branch is harmless: its ttc..tt0 propagation is a no-op.
                treat_as_static = obs.is_static or (abs(obs.vs) < 0.5 and abs(obs.vd) < 0.5)
                if (treat_as_static and not obs.is_static
                        and is_dynamic_ot_wpnts and self.free_check_dynamic_ot_slow):
                    treat_as_static = False
                if treat_as_static:
                    if not wpnts_data.is_closed and gap > max_gap:
                        is_free = False
                        if closest_obs is None or min_gap > gap:
                            closest_obs = obs
                            min_gap = gap
                    elif gap < max_horizon:
                        obs_d = obs.d_center
                        ot_d = 0
                        if not is_gb_track_wpnts:
                            avoid_wpnt_idx = np.argmin(abs(wpnts_data.array[:, 2] - obs_s))
                            ot_d = wpnts_data.list[avoid_wpnt_idx].d_m
                        min_dist = abs(ot_d - obs_d)
                        free_dist = min_dist - obs.size / 2 - self.gb_ego_width_m / 2
                        # For a STATIC obstacle the required clearance must be DISTANCE-
                        # INDEPENDENT: the object isn't moving, so a line that geometrically
                        # clears it is just as valid at 8 m as at 1 m. The original gap-scaling
                        # (meant for moving opponents: "only trust the lateral gap once close")
                        # made a clearing path read NOT-free while far -> the "trail up close,
                        # then switch" artifact.
                        #   - avoidance path: full lateral_width_m, unscaled.
                        #   - raceline (GB):  the SMALLER static-specific margin. The obstacle-
                        #     aware line from static_reopt clears the box by keep-out+apex_bulge
                        #     (~0.40 m); the scaled requirement (ego/2 + 0.3 = 0.50 m at range)
                        #     read that line as blocked -> phantom TRAILING + pointless
                        #     re-avoidance of an obstacle the line already avoids.
                        if is_ot_wpnts and not is_gb_track_wpnts:
                            required_margin = lateral_width_m
                        elif is_gb_track_wpnts:
                            required_margin = self.lateral_width_static_gb_m
                        else:
                            required_margin = lateral_width_m * np.clip(
                                gap / free_scaling_reference_distance_m, 0.0, 1.0)
                        # ONE DEFINITION OF CLEAR. `free_dist` above is a LATERAL distance in this
                        # node's frenet frame; static_reopt measures the 2-D distance from the line
                        # it publishes to the obstacle EDGE, and the reactive planner used to
                        # derive a third. The three agree on a straight and diverge on a curve --
                        # while being compared against thresholds 2 cm apart, so a curve was enough
                        # for this node to call BLOCKED a line the planner had already gone idle
                        # over, which is a TRAILING nothing can leave. On the GB line (the one
                        # static_reopt owns and measures) its number is the answer; everywhere else
                        # -- an overtaking spline it never saw -- the local computation stands.
                        # No fresh measurement means the local computation too: the feed may only
                        # ever REPLACE a measurement, never remove one.
                        if is_gb_track_wpnts:
                            pub_clr = self._published_clearance(obs)
                            if pub_clr is not None:
                                free_dist = pub_clr - self.gb_ego_width_m / 2
                        if free_dist < required_margin:
                            is_free = False
                            self.get_logger().info(
                                "[State Machine] FREE False, obs dist to ot lane: {} m".format(free_dist),
                                throttle_duration_sec=1.0,
                            )
                            if closest_obs is None or min_gap > gap:
                                closest_obs = obs
                                min_gap = gap
                else:
                    if len(obstacle_predictions) != 0 and self.obstacles_prediction_id == obs.id:
                        start_idx = 0
                        end_idx = len(obstacle_predictions)
                        if is_ot_wpnts:
                            # Seconds per prediction index — MUST match the 'dt' param of
                            # /opponent_propagation_predictor (opp_prediction.py, default 0.02).
                            pred_dt = 0.02
                            if ttc > 0:
                                start_idx = min(int(ttc / pred_dt), len(obstacle_predictions))
                            if tt0 > 0:
                                end_idx = min(int(tt0 / pred_dt), len(obstacle_predictions))
                        for obs_pred in obstacle_predictions[start_idx:end_idx]:
                            wpnt_idx = np.argmin(abs(wpnts_data.array[:, 2] - obs_pred.pred_s))
                            wpnt_d = wpnts_data.list[wpnt_idx].d_m
                            min_dist = abs(wpnt_d - obs_pred.pred_d)
                            free_dist = min_dist - obs.size / 2 - self.gb_ego_width_m / 2
                            scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                            if is_ot_wpnts and free_dist < lateral_width_m * scaling_factor:
                                # np.argmin returns the nearest sample even when pred_s is off
                                # the END of the path -- then wpnt_d belongs to a completely
                                # different piece of track and the comparison is meaningless.
                                # ds_lookup near 0 means the lookup is real; metres means the
                                # published path does not cover where the opponent is predicted.
                                # age tells the other story: a cached path (killing_timer_sec
                                # 10 s) being judged against live obstacles.
                                ds_lookup = abs(float(wpnts_data.array[wpnt_idx, 2])
                                                - float(obs_pred.pred_s))
                                ds_lookup = min(ds_lookup, self.max_s - ds_lookup)
                                age = (self.now_sec() - time_to_float(wpnts_data.stamp)
                                       if wpnts_data.stamp is not None else -1.0)
                                self.get_logger().warn(
                                    f"[{self.name}] OT path NOT free: free_dist={free_dist:+.2f} "
                                    f"(need {lateral_width_m * scaling_factor:.2f}) "
                                    f"lane_d={wpnt_d:+.2f} opp_d={obs_pred.pred_d:+.2f} "
                                    f"| path age={age:.2f}s len={len(wpnts_data.list)} "
                                    f"lookup_err={ds_lookup:.2f}m",
                                    throttle_duration_sec=1.0,
                                )
                            if free_dist < lateral_width_m * scaling_factor:
                                is_free = False
                                if closest_obs is None or min_gap > gap:
                                    closest_obs = obs
                                    min_gap = gap
                    else:
                        if not wpnts_data.is_closed and gap > max_gap:
                            is_free = False
                            if closest_obs is None or min_gap > gap:
                                closest_obs = obs
                                min_gap = gap
                        elif gap < max_horizon:
                            # DYNAMIC obstacle with no usable prediction (predictor not up yet,
                            # or tracking a different id). Propagate it ourselves over ttc..tt0
                            # -- the same window the prediction branch above uses, i.e. the time
                            # the ego is actually alongside. Testing where the opponent stands
                            # NOW asks the path to clear a place the two are never at together;
                            # an overtaking lane deliberately stays on the raceline until the
                            # pass, so that test reads it as blocked and OVERTAKE flaps.
                            # Near-stationary obstacles never reach here (handled as static
                            # above), so nothing is propagated that should not be.
                            ot_d = 0
                            min_dist = abs(ot_d - obs.d_center)
                            if not is_gb_track_wpnts:
                                if self.free_check_predict_dynamic:
                                    t_lo = max(ttc, 0.0)
                                    t_hi = max(tt0, t_lo)
                                    t_samples = np.linspace(t_lo, t_hi, 5)
                                else:
                                    t_samples = np.zeros(1)
                                min_dist = float("inf")
                                for t_p in t_samples:
                                    s_p = (obs.s_center + obs.vs * t_p) % self.max_s
                                    j = np.argmin(abs(wpnts_data.array[:, 2] - s_p))
                                    min_dist = min(min_dist,
                                                   abs(wpnts_data.list[j].d_m - obs.d_center))
                            free_dist = min_dist - obs.size / 2 - self.gb_ego_width_m / 2
                            scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                            if free_dist < lateral_width_m * scaling_factor:
                                if is_ot_wpnts:
                                    # The ONLY other free-check log lives in the prediction branch
                                    # above, but predictions need a clean opponent half-lap first --
                                    # so early in a run every dynamic rejection happens HERE and was
                                    # completely silent. predict_dyn=False means the lane was tested
                                    # where the opponent stands NOW, which an overtaking lane never
                                    # clears by design (it stays on the raceline until the pass).
                                    self.get_logger().warn(
                                        f"[{self.name}] OT path NOT free (no pred): "
                                        f"free_dist={free_dist:+.2f} "
                                        f"(need {lateral_width_m * scaling_factor:.2f}) "
                                        f"opp_d={obs.d_center:+.2f} gap={gap:.2f} "
                                        f"ttc={ttc:.2f} tt0={tt0:.2f} "
                                        f"predict_dyn={self.free_check_predict_dynamic}",
                                        throttle_duration_sec=1.0,
                                    )
                                is_free = False
                                if closest_obs is None or min_gap > gap:
                                    closest_obs = obs
                                    min_gap = gap
        else:
            is_free = True

        wpnts_data.closest_target = closest_obs
        wpnts_data.closest_gap = min_gap
        return is_free

    def _check_free_cartesian(self, wpnts_data) -> bool:
        is_free = True
        closest_obs = None
        min_gap = None
        min_horizon = wpnts_data.min_horizon
        max_horizon = wpnts_data.max_horizon
        free_scaling_reference_distance_m = wpnts_data.free_scaling_reference_distance_m
        lateral_width_m = wpnts_data.lateral_width_m

        obstacles = self.cur_obstacles_in_interest
        if wpnts_data.is_init:
            for obs in obstacles:
                obs_s = obs.s_center
                gap = (obs_s - self.cur_s) % self.max_s
                if gap < max_horizon or min_horizon < (gap - self.max_s):
                    dists = np.linalg.norm(wpnts_data.array[:, 0:2] - np.array([obs.x_m, obs.y_m]), axis=1)
                    min_dist = np.min(dists)
                    free_dist = min_dist - obs.size / 2 - self.gb_ego_width_m / 2
                    scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                    if free_dist < lateral_width_m * scaling_factor:
                        is_free = False
                        if closest_obs is None or min_gap > gap:
                            closest_obs = obs
                            min_gap = gap
                        self.get_logger().info(
                            f"[{self.name}] RECOVERY_FREE False, obs dist to recovery lane: {min_dist} m",
                            throttle_duration_sec=1.0,
                        )
        else:
            is_free = True
        wpnts_data.closest_target = closest_obs
        wpnts_data.closest_gap = min_gap
        return is_free

    def _expire_unused_ot_cache(self, wpnts_data, ttl_sec):
        # Reference = last_used_sec, else the cached stamp (time the path was received).
        if not wpnts_data.is_init:
            return
        ref = wpnts_data.last_used_sec
        if ref is None:
            ref = time_to_float(wpnts_data.stamp) if wpnts_data.stamp is not None else None
        if ref is None:
            return
        if self.now_sec() - ref > ttl_sec:
            wpnts_data.is_init = False
            wpnts_data.closest_target = None
            wpnts_data.last_used_sec = None

    def _check_availability(self, wpnts, wpnts_data) -> bool:
        if (self.now_sec() - time_to_float(wpnts_data.stamp)) > wpnts_data.killing_timer_sec:
            wpnts_data.is_init = False
            return bool(self._check_latest_wpnts(wpnts, wpnts_data))

        if (self.now_sec() - time_to_float(wpnts_data.stamp)) > wpnts_data.hyst_timer_sec:
            if self._check_latest_wpnts(wpnts, wpnts_data):
                return True

        if not self._check_on_spline(wpnts_data):
            return bool(self._check_latest_wpnts(wpnts, wpnts_data))

        return True

    def _check_sustainability(self, src_wpnts, wpnts_data) -> bool:
        if self._check_availability(src_wpnts, wpnts_data) and self._check_free_frenet(wpnts_data):
            return True
        return False

    def _ot_gate_dbg(self, why: str):
        """One throttled line naming the FIRST failing AND-term of _check_overtaking_mode and
        its measured value vs threshold. Dynamic analogue of the 'static_OT check' log below.

        Without it every dynamic no-engage is indistinguishable from a dead node: all four
        terms fail silently, while the static gate prints a full per-term verdict every 0.5 s.
        That asymmetry is why a single run could not attribute a dynamic overtaking failure."""
        self.get_logger().info(f"[{self.name}] dyn_OT blocked: {why}", throttle_duration_sec=0.5)

    def _check_overtaking_mode(self) -> bool:
        # Predictor veto (opt-in via use_force_trailing): True while the opponent is off its
        # learned line or fewer than one opponent-lap has been collected, i.e. the predictions
        # this gate relies on are unreliable. Entry gate only — an OVERTAKE already committed
        # is governed by _check_overtaking_mode_sustainability (bailing out beside an opponent
        # is worse than finishing the maneuver).
        #
        # The chain below MUST stay short-circuited: _check_latest_wpnts adopts the published
        # path into the cache (initialize_traj) and _check_free_frenet writes closest_target/
        # closest_gap, which get_farthest_target consumes. Evaluating every term up-front to
        # build a log line would silently change both. So each term logs inside its own failure
        # branch, reading only cached/pure state.
        if self.force_trailing:
            self._ot_gate_dbg("force_trailing (opponent off its learned line, or < 1 opponent lap)")
            return False
        # --- 1/4 overtaking sector -------------------------------------------------
        if not self._check_ot_sector():
            self._ot_gate_dbg(
                f"ot_sector: wpnt_idx={self.cur_s / self.wpnt_dist:.0f} not in "
                f"overtake_zones={self.overtake_zones} (empty => overtaking suppressed for this map)"
            )
            return False
        # --- 2/4 closing on the nearest obstacle ahead ------------------------------
        if not self._check_getting_closer(threshold_m=10.0):
            if len(self.obstacles_in_interest) == 0:
                self._ot_gate_dbg(
                    f"getting_closer: 0 obstacles within interest_horizon_m={self.interest_horizon_m:.1f}"
                )
            else:
                o = self.obstacles_in_interest[0]
                g = (o.s_start - self.cur_s) % self.track_length
                c = self.cur_vs - o.vs
                self._ot_gate_dbg(
                    f"getting_closer: id={o.id} gap={g:.2f}(<10.00?{'ok' if g < 10.0 else 'NO'}) "
                    f"closing={c:+.2f}(>{self.getting_closer_rel_vel_mps:+.2f}?"
                    f"{'ok' if c > self.getting_closer_rel_vel_mps else 'NO'}) "
                    f"static={o.is_static} vs={o.vs:+.2f}"
                )
            return False
        # --- 3/4 fresh publish + car on the spline ---------------------------------
        if not self._check_latest_wpnts(self.avoidance_wpnts, self.cur_avoidance_wpnts):
            av = self.avoidance_wpnts
            n_av = len(av.wpnts) if av is not None else 0
            av_age = (self.now_sec() - time_to_float(av.header.stamp)) if n_av > 0 else -1.0
            wd = self.cur_avoidance_wpnts
            gap_dbg = md_dbg = -1.0
            if wd.is_init and wd.array is not None:
                gap_dbg = (wd.list[-1].s_m - self.cur_s) % self.track_length
                md_dbg = float(np.min(np.linalg.norm(wd.array[:, 0:2] - self.current_position[:2], axis=1)))
            self._ot_gate_dbg(
                f"latest+on_spline: n={n_av},age={av_age:.2f}(<{wd.latest_threshold}),"
                f"gap={gap_dbg:.2f}(>{wd.on_spline_front_horizon_thres_m}),"
                f"min_dist={md_dbg:.2f}(<{wd.on_spline_min_dist_thres_m})"
            )
            return False
        # --- 4/4 path free -----------------------------------------------------------
        if not self._check_free_frenet(self.cur_avoidance_wpnts):
            wd = self.cur_avoidance_wpnts
            t = wd.closest_target
            self._ot_gate_dbg(
                f"free_frenet: blocked by id={t.id if t is not None else -1} "
                f"d={t.d_center if t is not None else float('nan'):+.2f} "
                f"static={t.is_static if t is not None else '?'} "
                f"gap={wd.closest_gap:.2f} need lateral_width_m={wd.lateral_width_m}"
            )
            return False
        self.static_overtaking_mode = False
        self._ot_free_true_t = self.now_sec()
        return True

    def _check_static_overtaking_mode(self) -> bool:
        # SIMPLIFIED: the static spliner now owns the go/no-go decision — it publishes an evasion
        # path ONLY when a static obstacle has a wide-enough lateral gap to pass (and clamps it
        # inside the track). So the state machine just commits when that path is fresh and the car
        # is on it. The old distance gate (c_closer) and the redundant free-check (c_free) — which
        # made the car slow-trail right up to the obstacle before switching — are gone.
        # Feasibility gate, FAIL-CLOSED: the planner publishes the signal every cycle, so stale
        # (> static_feasible_stale_sec) or never-received means planner dead/miswired -> block
        # the commit and keep TRAILING. (The old speed guard static_ot_speed_mps is gone: the
        # live config had it disabled at 10.0, and entry speed is owned by the avoidance path's
        # slow-in velocity profile, not the commit gate.)
        # Re-entry cooldown after a drop: the drop and this gate read almost the same inputs, so
        # without a pause the pair oscillates at the message rate (drop -> path still fresh ->
        # re-enter -> drop), which is the OVERTAKE<->TRAILING flapping.
        if (self._static_ot_cooldown_until is not None
                and self.now_sec() < self._static_ot_cooldown_until):
            self.get_logger().info(
                f"[{self.name}] static_OT check: in re-entry cooldown for "
                f"{self._static_ot_cooldown_until - self.now_sec():.2f} s after a drop",
                throttle_duration_sec=0.5)
            return False
        feas_age = (self.now_sec() - self._static_feasible_t) if self._static_feasible_t is not None else -1.0
        c_feasible = (self.static_avoidance_feasible
                      and self._static_feasible_t is not None
                      and feas_age <= self.static_feasible_stale_sec)
        c_latest = self._check_latest_wpnts(self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts)
        # debug: why isn't a fresh on-spline path available?
        sa = self.static_avoidance_wpnts
        n_sa = len(sa.wpnts) if sa is not None else 0
        age = (self.now_sec() - time_to_float(sa.header.stamp)) if n_sa > 0 else -1.0
        gap_dbg = md_dbg = -1.0
        wd = self.cur_static_avoidance_wpnts
        if wd.is_init and wd.array is not None:
            gap_dbg = (wd.list[-1].s_m - self.cur_s) % self.track_length
            md_dbg = float(np.min(np.linalg.norm(wd.array[:, 0:2] - self.current_position[:2], axis=1)))
        self.get_logger().info(
            f"[{self.name}] static_OT check: feasible={c_feasible}"
            f"[val={self.static_avoidance_feasible},age={feas_age:.2f}(<{self.static_feasible_stale_sec})] "
            f"latest+on_spline={c_latest}[n={n_sa},age={age:.2f}(<{wd.latest_threshold}),"
            f"gap={gap_dbg:.2f}(>{wd.on_spline_front_horizon_thres_m}),min_dist={md_dbg:.2f}(<{wd.on_spline_min_dist_thres_m})] "
            f"=> {c_feasible and c_latest}",
            throttle_duration_sec=0.5,
        )
        if c_feasible and c_latest:
            self.static_overtaking_mode = True
            return True
        else:
            return False

    def _hold_static_avoidance_reference(self) -> bool:
        """While TRAILING off the raceline, is the last static-avoidance slice still the right
        local reference?

        ObstacleTransition returns (TRAILING, GB_TRACK), so the instant a static OVERTAKE drops the
        reference snaps from the avoidance spline the car is physically ON back to the raw
        raceline -- which, for a static obstacle, is the line that runs INTO it. The controller then
        steers toward the obstacle while the gap PID brakes for it, and the resulting swerve-toward
        is what re-triggers the whole cycle. Holding the avoidance geometry until the car is
        genuinely back near the raceline turns the drop into a deceleration instead.

        Deliberately conservative: it requires the car to still be off the line by more than
        recovery_exit_d_m AND to still be ON the cached path. Once either fails, GB_TRACK is the
        correct reference and the normal cache TTL expires the path.

        Pure predicate -- transitions must not have side effects.
        """
        if abs(self.cur_d) <= self.recovery_exit_d_m:
            return False
        wd = self.cur_static_avoidance_wpnts
        if not wd.is_init or wd.array is None or wd.list is None or len(wd.list) == 0:
            return False
        return bool(self._check_on_spline(wd))

    def _check_overtaking_mode_sustainability(self) -> bool:
        if self.static_overtaking_mode:
            # Stay in OVERTAKE while the (spliner-vetted) static path is still available and the
            # car is on it. The spliner stops publishing once the gap closes / obstacle is passed,
            # so availability naturally drops and we exit — no redundant free re-check needed.
            # BUT: if the static planner explicitly reports NO feasible candidate (fresh
            # feasible=False for longer than a blip), the situation has changed since the commit
            # (obstacle moved/reclassified, gap closed). Riding the CACHED spline then is how the
            # car lunged into a box 0.7 m ahead: the stale path aged into a raceline extension
            # THROUGH the obstacle while availability hysteresis (hyst 4 s / kill 10 s) kept
            # OVERTAKE alive. Fail over to TRAILING instead (gap PID brakes behind the obstacle).
            feas_true_age = (self.now_sec() - self._static_feasible_true_t) \
                if self._static_feasible_true_t is not None else float('inf')
            if feas_true_age > self.static_feasible_lost_sec:
                self.get_logger().warn(
                    f"[{self.name}] static OVERTAKE dropped: planner reports infeasible "
                    f"for {feas_true_age:.2f} s", throttle_duration_sec=1.0)
                self._static_ot_cooldown_until = self.now_sec() + self.static_ot_reentry_cooldown_sec
                return False
            if self._check_availability(self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts):
                self._static_avail_true_t = self.now_sec()
                return True
            # Debounce, SYMMETRIC with static_feasible_lost_sec above. Availability is a freshness
            # test against an executor that runs 0.3-0.5 s behind, so a single late publish failed
            # it -- and dropping OVERTAKE on that hands the car back to the raceline, which for a
            # static obstacle is the line that runs into it. Without this the noisier of the two
            # sustain terms decided the exit while the other tolerated 0.4 s of blips.
            avail_age = ((self.now_sec() - self._static_avail_true_t)
                         if self._static_avail_true_t is not None else float("inf"))
            if avail_age <= self.static_avail_lost_sec:
                return True
            self.get_logger().warn(
                f"[{self.name}] static OVERTAKE dropped: avoidance path unavailable for "
                f"{avail_age:.2f} s (> {self.static_avail_lost_sec:.2f})",
                throttle_duration_sec=1.0)
            self._static_ot_cooldown_until = self.now_sec() + self.static_ot_reentry_cooldown_sec
            return False
        else:
            if self._check_availability(self.avoidance_wpnts, self.cur_avoidance_wpnts):
                self.get_logger().debug("AVAILABLE")
                if self._check_free_frenet(self.cur_avoidance_wpnts):
                    self._ot_free_true_t = self.now_sec()
                    return True
                # Debounce, mirroring static_feasible_lost_sec above. The free-check reads the
                # path at ONE point per obstacle, so a single noisy tracker cycle fails it --
                # and the cost of believing that cycle is high: dropping OVERTAKE steers the
                # car back toward the raceline, i.e. toward the opponent it is alongside, and
                # the check then passes again. That loop is the flapping between the avoidance
                # spline and the global line. A sustained not-free still exits.
                free_age = ((self.now_sec() - self._ot_free_true_t)
                            if self._ot_free_true_t is not None else float("inf"))
                if free_age <= self.ot_free_lost_sec:
                    return True
                self.get_logger().warn(
                    f"[{self.name}] OVERTAKE dropped: avoidance path not free for "
                    f"{free_age:.2f} s (> {self.ot_free_lost_sec:.2f})",
                    throttle_duration_sec=1.0)
            else:
                self.get_logger().warn(
                    f"[{self.name}] OVERTAKE dropped: avoidance path unavailable",
                    throttle_duration_sec=1.0)
        return False

    ################
    # HELPER FUNCS #
    ################
    def update_velocity(self, wpnts_msg, safety_factor=1.0, v_cap=None, ay_max=None,
                        cache_key="default"):
        """Re-profile a planner's path with THIS node's vehicle dynamics, in place.

        `ay_max` overrides the ggv's lateral-accel column for this path only (see
        avoidance_ay_max): the global ggv is tuned for the raceline, and an avoidance maneuver is a
        deliberate, brief excursion that the planner itself already sizes at a higher a_lat_max --
        re-profiling it at the raceline's limit is what made the avoidance spline crawl.

        `cache_key` selects a per-source slot in the profile cache; see the note on the key below.
        """
        if self.ggv is None or self.gb_wpnts is None:
            return  # velocity replanning unavailable (no veh dyn info / no gb wpnts yet)
        wpnts = wpnts_msg.wpnts
        if len(wpnts) < 3:
            return
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        el_lengths = np.array([
            np.linalg.norm([
                wpnts[i + 1].x_m - wpnts[i].x_m,
                wpnts[i + 1].y_m - wpnts[i].y_m,
            ])
            for i in range(len(wpnts) - 1)
        ])
        # Bail if the path is degenerate: a zero-length segment or any non-finite input makes
        # calc_vel_profile divide by zero -> NaN velocities that propagate into the local path
        # and eventually the base_link TF. Leaving the original vx_mps untouched is the safe path.
        if (el_lengths <= 1e-6).any() or not np.all(np.isfinite(el_lengths)) \
                or not np.all(np.isfinite(kappa)):
            self.get_logger().warn(
                f"[{self.name}] degenerate path in update_velocity; keeping planner velocities",
                throttle_duration_sec=1.0,
            )
            return

        glb_start_idx = int(wpnts_msg.wpnts[-1].s_m / self.wpnt_dist)
        v_end = self.gb_wpnts.wpnts[glb_start_idx % len(self.gb_wpnts.wpnts)].vx_mps

        # --- profile cache -------------------------------------------------------------------
        # calc_vel_profile is the expensive part of this node, and a COMMITTED planner path is
        # geometrically FROZEN: the planner republishes the same points at 20 Hz and this solved
        # them again for every one. That is what put the single-threaded executor 0.3-0.5 s behind
        # (documented in static_avoidance_planner.yaml's latest_threshold), and the freshness gate
        # built on top of that lag is what then blocked the OVERTAKE commit -- so the cost of the
        # recomputation was not lost time, it was a lost maneuver.
        #
        # The key has to survive the FORWARD RE-SLICE, or it never hits at all. A committed path is
        # frozen, but the planner republishes only the part still ahead of the car, so the array
        # loses points from the FRONT: at 3 m/s a 0.1 m station is passed every 34 ms against this
        # node's 50 ms cycle, so the length changed nearly every cycle. Hashing the raw float64
        # bytes of the whole array therefore missed every single time -- a cache with a 0% hit rate,
        # which is why the 0.3-0.5 s of executor lag it was added to remove was still in the log
        # (`age=1.41(<1.0)`).
        #
        # So the signature is the TAIL, which a forward re-slice does not touch, quantized below
        # the solver's own sensitivity rather than compared bit-for-bit. The head is handled by
        # VALIDATION instead of by the key: a cached profile is reused only if, sliced to the
        # current window, it starts within one quantum of the speed the car is actually doing --
        # exactly the tolerance the old key claimed for v_start, but checked against the profile
        # that is about to be published instead of against the inputs.
        # The key is a FIXED-LENGTH tail signature -- a count that shrank with the array would be
        # exactly as length-sensitive as the old whole-array hash -- and it is only a prefilter.
        # What makes a reuse correct is the explicit comparison below: the cached geometry, sliced
        # to this window, must actually equal this one, so two paths that happen to share a tail
        # cannot be confused. That comparison is a few array ops; the solve it avoids is milliseconds.
        n_now = len(kappa)
        # A path at least sig points long is signed by its LAST sig points, which a forward
        # re-slice does not touch. A shorter one is signed by all of it -- there the count cannot
        # be fixed, but such a path is below the local window anyway.
        tail = min(n_now, int(self.vel_cache_sig_pts))
        key = (hash(np.round(np.asarray(kappa[-tail:], dtype=np.float64), 3).tobytes()),
               hash(np.round(np.asarray(el_lengths[-tail:], dtype=np.float64), 4).tobytes()),
               round(float(v_end) / self.vel_cache_quant_mps),
               round(float(safety_factor), 3),
               None if v_cap is None else round(float(v_cap), 3),
               None if ay_max is None else round(float(ay_max), 3))
        hit = self._vel_cache.get(cache_key)
        vx_profile = ax_profile = None
        if (hit is not None and hit[0] == key
                and len(hit[1]) >= n_now and len(hit[3]) >= n_now and n_now > 1):
            same = (np.allclose(kappa, hit[3][-n_now:], atol=1e-3, rtol=0.0)
                    and np.allclose(el_lengths, hit[4][-(n_now - 1):], atol=1e-4, rtol=0.0))
            if same:
                vx_try = hit[1][-n_now:]
                ax_try = hit[2][-(n_now - 1):]
                # ...and the profile must still start where the CAR is. The head is validated
                # rather than keyed: a cached profile is a function of the v_start it was solved
                # with, and reusing it after the car has left that speed publishes a reference the
                # car is not on.
                if abs(float(vx_try[0]) - float(self.cur_vs)) <= self.vel_cache_quant_mps:
                    vx_profile, ax_profile = vx_try, ax_try
        if vx_profile is None:
            ax_max_machines_sf = self.ax_max_machines.copy()
            b_ax_max_machines_sf = self.b_ax_max_machines.copy()
            ax_max_machines_sf[:, 1] *= safety_factor
            b_ax_max_machines_sf[:, 1] *= safety_factor

            # ggv columns are [v, ax_max, ay_max]. Overriding the lateral column decouples an
            # avoidance path from the raceline's cornering budget without touching the global one.
            ggv = self.ggv
            if ay_max is not None and ay_max > 0.0:
                ggv = self.ggv.copy()
                ggv[:, 2] = float(ay_max)

            vx_profile = calc_vel_profile(
                ax_max_machines=ax_max_machines_sf,
                kappa=kappa,
                el_lengths=el_lengths,
                closed=False,
                drag_coeff=self.pars["veh_params"]["dragcoeff"],
                m_veh=self.pars["veh_params"]["mass"],
                b_ax_max_machines=b_ax_max_machines_sf,
                ggv=ggv,
                # The squeeze cap goes IN, not on afterwards. Applied as np.minimum to the solved
                # profile it rewrote vx[0] -- which is the solver's v_start, i.e. the speed the car
                # is doing right now -- from cur_vs to the cap, so the reference began with a step
                # that the controller's slew limiter then took ~300 ms to serve. Handed to the
                # solver as v_max it is a BOUND the profile is built to: v_start is still the
                # current speed, and what comes out is the deceleration this comment claimed.
                v_max=(min(float(self.pars["veh_params"]["v_max"]), float(v_cap))
                       if (v_cap is not None and v_cap > 0.0)
                       else self.pars["veh_params"]["v_max"]),
                filt_window=self.pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
                dyn_model_exp=self.pars["vel_calc_opts"]["dyn_model_exp"],
                v_start=self.cur_vs,
                v_end=v_end,
            )

            ax_profile = tph.calc_ax_profile.calc_ax_profile(
                vx_profile=vx_profile, el_lengths=el_lengths, eq_length_output=False
            )
            self._vel_cache[cache_key] = (key, vx_profile, ax_profile,
                                          np.asarray(kappa, float).copy(),
                                          np.asarray(el_lengths, float).copy())

        # THE OUTPUT IS CHECKED TOO. The inputs are validated above, but the solver can still
        # return a non-finite profile (a zero el_length, a degenerate kappa run), and writing NaN
        # into vx_mps poisons everything downstream at once: the controller's lookahead, the
        # marker array (RViz drops the WHOLE array on one bad pose, which is what "the local
        # waypoints disappeared" looks like), and the SM's own speed logic. Keeping the planner's
        # own speeds is the fail-closed answer -- they are a real profile, just not re-fitted to
        # this node's dynamics.
        if not (np.all(np.isfinite(vx_profile)) and np.all(np.isfinite(ax_profile))):
            self.get_logger().warn(
                f"[{self.name}] velocity re-profile produced non-finite values "
                f"({int(np.count_nonzero(~np.isfinite(vx_profile)))} of {len(vx_profile)} vx); "
                f"keeping the planner's own speeds for this path",
                throttle_duration_sec=1.0)
            self._vel_cache.pop(cache_key, None)          # never serve it from the cache
            return
        for i in range(len(vx_profile)):
            wpnts_msg.wpnts[i].vx_mps = vx_profile[i]
        for i in range(len(ax_profile)):
            wpnts_msg.wpnts[i].ax_mps2 = ax_profile[i]
        wpnts[len(ax_profile)].ax_mps2 = ax_profile[-1]

    def anchor_gb_index(self, s_idx: int, search_m: float = 3.0) -> int:
        """Re-anchor an s-derived global-waypoint index to the station NEAREST THE CAR.

        `cur_s` and `cur_gb_wpnts` can briefly disagree about what station a given s is: the frenet
        converter re-takes /global_waypoints the moment static_reopt swaps the line, while this
        node's copy comes from /global_waypoints_scaled, which sector_tuner only re-publishes on
        its 0.5 s timer. In that gap an s-derived index points at the right NUMBER on the wrong
        parameterisation and the local window slides along the track, away from the car.

        The search is restricted to +-`search_m` around `s_idx` on purpose. A free nearest-point
        search over the whole closed loop would snap to the wrong branch wherever the raceline runs
        close to itself (chicane, hairpin); the s index is a good coarse anchor and only ever needs
        a local correction. Returns `s_idx` unchanged if the position or the line is unavailable.
        """
        if self.current_position is None or not self.cur_gb_wpnts.is_init:
            return s_idx
        arr = self.cur_gb_wpnts.array
        n = self.num_glb_wpnts
        if arr is None or n == 0:
            return s_idx
        k = max(1, int(search_m / max(self.wpnt_dist, 1e-3)))
        idx = (s_idx + np.arange(-k, k + 1)) % n
        d = np.hypot(arr[idx, 0] - self.current_position[0],
                     arr[idx, 1] - self.current_position[1])
        return int(idx[int(np.argmin(d))])

    def mincurv_splinification(self):
        coords = np.empty((len(self.cur_gb_wpnts.list), 4))
        for i, wpnt in enumerate(self.cur_gb_wpnts.list):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.vx_mps
        self.mincurv_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.mincurv_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.mincurv_spline_v = Spline(coords[:, 0], coords[:, 3])
        self.get_logger().info(f"[{self.name}] Splinified Min Curve")

    def ot_splinification(self):
        coords = np.empty((len(self.overtake_wpnts), 5))
        for i, wpnt in enumerate(self.overtake_wpnts):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.d_m
            coords[i, 4] = wpnt.vx_mps
        coords = coords[coords[:, 0].argsort()]
        # Drop non-finite rows and duplicate/non-increasing s: scipy Spline requires a
        # strictly increasing x or it raises / returns NaN. A reversed or seam-jumped
        # overtake path would otherwise poison every downstream spline eval with NaN.
        coords = coords[np.isfinite(coords).all(axis=1)]
        if len(coords) >= 2:
            keep = np.concatenate([[True], np.diff(coords[:, 0]) > 1e-6])
            coords = coords[keep]
        if len(coords) < 4:
            self.get_logger().warn(
                f"[{self.name}] overtake wpnts degenerate ({len(coords)} usable); skipping splinification",
                throttle_duration_sec=1.0,
            )
            return
        self.ot_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.ot_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.ot_spline_d = Spline(coords[:, 0], coords[:, 3])
        self.ot_spline_v = Spline(coords[:, 0], coords[:, 4])
        self.get_logger().info(f"[{self.name}] Splinified Overtaking Curve")

    def _find_nearest_ot_s(self) -> float:
        half_search_dim = 5
        idxs = [
            i % self.num_ot_points
            for i in range(self.cur_id_ot - half_search_dim, self.cur_id_ot + half_search_dim)
        ]
        ses = np.array([self.overtake_wpnts[i].s_m for i in idxs])
        dists = np.abs(self.cur_s - ses)
        chose_id = np.argmin(dists)
        s_ot = idxs[chose_id]
        s_ot %= self.num_ot_points
        return s_ot

    def _splini_anchor_index(self, wpnts: WaypointData,
                             search_m: float = 3.0, back_pts: int = 2) -> int:
        """Index of the avoidance path to start the published window at.

        Same idiom as anchor_gb_index: bound the search to an s-window around the car, then pick
        the nearest point by POSITION inside it. Two failure modes are being avoided at once, and
        the old code had one branch for each, which is why they fought.

        A free XY argmin over the whole path snaps to the wrong branch wherever the path runs close
        to itself -- and an avoidance path deliberately does, since its entry and exit ramps sit on
        the same raceline a few metres apart. That is how the window came to start at a point the
        car had already passed. The s-window makes that impossible.

        The other branch spliced from index 0 whenever the car was within splice_start_dist_m of
        the path start, to preserve the entry ramp. But the path is republished with its start
        re-anchored at the car, and the SM adopts it 0.3-0.5 s late, so "near the start" stayed true
        long after the car had driven past it -- pinning the window to a stale point behind the car
        for the whole approach. What that branch actually needed is a small BACK MARGIN, which is
        `back_pts` (2 points ~ 0.2 m): enough to keep the entry ramp continuous across a re-slice,
        far too little to strand the window behind the car.
        """
        arr = wpnts.array
        n = len(arr)
        if n == 0:
            return 0
        L = self.track_length if self.track_length else self.max_s
        cand = np.arange(n)
        if L:
            # signed, wrap-aware s offset of every path point from the car
            ds = (arr[:, 2] - self.cur_s + L / 2.0) % L - L / 2.0
            near = np.flatnonzero(np.abs(ds) <= search_m)
            if near.size:
                cand = near
        d2 = ((arr[cand, 0] - self.current_position[0]) ** 2
              + (arr[cand, 1] - self.current_position[1]) ** 2)
        nearest = int(cand[int(np.argmin(d2))])
        return max(0, nearest - int(back_pts))

    def _splice_index(self, tail, tag: str, search_m: float = 3.0) -> int:
        """Index of the GLOBAL waypoint that continues from `tail`, found by POSITION.

        Two arrays are being joined here, and s only lines them up when they are parameterised by
        the SAME line. They are not, whenever static_reopt has swapped one: the cached avoidance
        path carries the s of the line it was planned on, cur_gb_wpnts comes from
        /global_waypoints_scaled (sector_tuner's 0.5 s timer), and an obstacle-aware line has a
        different arc length -- measured at +0.95 m per lap. searchsorted then answers a question
        about the wrong parameterisation, correctly, and the padding starts several waypoints from
        where the path actually ends. The run log: `the padded join steps 0.595 m
        (> 1.5 x wpnt_dist = 0.152)`, plus two of 0.303 m -- three to six times the nominal
        spacing, i.e. a hole in the published local waypoints.

        (x, y) is frame-independent, so the nearest global waypoint to the path's LAST POINT is the
        right continuation whatever either array's s means. The search is restricted to an s-window
        around the tail for the same reason _splini_anchor_index restricts its own: a free XY argmin
        snaps to the wrong branch wherever the line runs close to itself. The s used for the window
        may be off by the arc-length difference; a 3 m window absorbs that.
        """
        arr = self.cur_gb_wpnts.array
        n = len(arr)
        if n == 0:
            return 0
        last_s = float(getattr(tail, "s_m", 0.0))
        x, y = getattr(tail, "x_m", None), getattr(tail, "y_m", None)
        if x is None or y is None:                      # no geometry -> the old s question
            try:
                return int(np.searchsorted(arr[:, 2], last_s, side="right"))
            except Exception:
                return int(last_s / self.wpnt_dist) + 1
        L = self.track_length if self.track_length else self.max_s
        cand = np.arange(n)
        if L:
            ds = (arr[:, 2] - last_s + L / 2.0) % L - L / 2.0
            near = np.flatnonzero(np.abs(ds) <= search_m)
            if near.size:
                cand = near
        d2 = (arr[cand, 0] - float(x)) ** 2 + (arr[cand, 1] - float(y)) ** 2
        nearest = int(cand[int(np.argmin(d2))])
        return (nearest + 1) % n

    def _warn_splice_step(self, tail, head, tag: str) -> None:
        """A splice is a JOIN: if the two ends are further apart than one waypoint spacing, the
        local path has a step in it, and a step is what the controller feels."""
        try:
            step = float(np.hypot(head.x_m - tail.x_m, head.y_m - tail.y_m))
        except Exception:
            return
        if step > 1.5 * self.wpnt_dist:
            self.get_logger().warn(
                f"[{self.name}] {tag}: the padded join steps {step:.3f} m "
                f"(> 1.5 x wpnt_dist = {1.5 * self.wpnt_dist:.3f}); the local path has a "
                f"discontinuity at the hand-over to the global line",
                throttle_duration_sec=2.0)

    def get_splini_wpts(self) -> WpntArray:
        if self.static_overtaking_mode:
            wpnts = self.cur_static_avoidance_wpnts
        else:
            wpnts = self.cur_avoidance_wpnts

        min_idx = self._splini_anchor_index(wpnts)
        avoidance_wpnts = wpnts.list[min_idx:min_idx + self.n_loc_wpnts]

        if len(avoidance_wpnts) < self.n_loc_wpnts:
            glb_start_idx = self._splice_index(wpnts.list[-1], "avoidance")
            extra_wpnts = [
                self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                for i in range(self.n_loc_wpnts - len(avoidance_wpnts))
            ]
            if avoidance_wpnts and extra_wpnts:
                self._warn_splice_step(avoidance_wpnts[-1], extra_wpnts[0], "avoidance")
            avoidance_wpnts.extend(extra_wpnts)
        return avoidance_wpnts

    def get_recovery_wpts(self) -> WpntArray:
        if self.cur_recovery_wpnts.is_init:
            diff = np.linalg.norm(self.cur_recovery_wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
            min_idx = np.argmin(diff)
            wpnts = self.cur_recovery_wpnts.list[min_idx:min_idx + self.n_loc_wpnts]
            if len(wpnts) < self.n_loc_wpnts:
                # NB the missing +1: this one restarted the padding ON the last point it already
                # had, so the joined path carried a duplicated waypoint.
                glb_start_idx = self._splice_index(self.cur_recovery_wpnts.list[-1], "recovery")
                extra_wpnts = [
                    self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                    for i in range(self.n_loc_wpnts - len(wpnts))
                ]
                if wpnts and extra_wpnts:
                    self._warn_splice_step(wpnts[-1], extra_wpnts[0], "recovery")
                wpnts.extend(extra_wpnts)
            return wpnts

    def get_start_wpts(self) -> WpntArray:
        if self.cur_start_wpnts.is_init:
            diff = np.linalg.norm(self.cur_start_wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
            min_idx = np.argmin(diff)
            start_wpnts = self.cur_start_wpnts.list[min_idx:min_idx + self.n_loc_wpnts]
            if len(start_wpnts) < self.n_loc_wpnts:
                glb_start_idx = self._splice_index(self.cur_start_wpnts.list[-1], "start")
                extra_wpnts = [
                    self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                    for i in range(self.n_loc_wpnts - len(start_wpnts))
                ]
                if start_wpnts and extra_wpnts:
                    self._warn_splice_step(start_wpnts[-1], extra_wpnts[0], "start")
                start_wpnts.extend(extra_wpnts)
            return start_wpnts
        else:
            self.get_logger().debug(f"[{self.name}] No valid avoidance waypoints, passing global waypoints")

    #######
    # VIZ #
    #######
    def limit_local_window_accel(self, wpts, v_seed):
        """Bound d(vx)/ds over the ASSEMBLED window: one backward pass, then one forward pass.

        RETURNS COPIES, ALWAYS. states.GlobalTracking and get_splini_wpts hand back the very Wpnt
        objects held by cur_gb_wpnts.list -- editing vx_mps in place would poison the cached
        global line one station per cycle, permanently, and the damage would look like a speed
        profile that decays over a session rather than like this function.

        The backward pass makes every deceleration reachable; the forward pass, seeded with the
        car's CURRENT speed, makes every acceleration reachable from where the car actually is.
        Forward can only lower speeds, so it cannot undo the backward pass.

        This is a defence, not a cure, for one of the two seams it covers: the s = 0 discontinuity
        is written into every map's global_waypoints.json by the vendored tph's __solver_fb_closed,
        which runs its backward pass over a doubled array and returns the second lap -- whose last
        element has no successor and is therefore never decelerated. Fixing that would rewrite
        every map's raceline and is deliberately NOT done here.
        """
        if not self.local_window_accel_limit_enable or not wpts or len(wpts) < 2:
            return wpts
        a_max = float(self.local_window_a_long_mps2)
        if a_max <= 0.0:
            return wpts
        out = [copy.copy(w) for w in wpts]
        v = np.array([float(w.vx_mps) for w in out], dtype=float)
        s = np.array([float(w.s_m) for w in out], dtype=float)
        ds = np.diff(s)
        # the window can wrap the start/finish line; a negative or absurd step there is the wrap,
        # not a reversal, so fall back to the nominal spacing rather than to a huge ds that would
        # make any jump look reachable
        nominal = float(getattr(self, "wpnt_dist", 0.1)) or 0.1
        ds = np.where((ds > 1e-6) & (ds < 10.0 * nominal), ds, nominal)
        two_a = 2.0 * a_max
        for i in range(len(v) - 2, -1, -1):
            lim = math.sqrt(v[i + 1] * v[i + 1] + two_a * ds[i])
            if v[i] > lim:
                v[i] = lim
        seed = max(float(v_seed), 0.0)
        lim0 = math.sqrt(seed * seed + two_a * ds[0])
        if v[0] > lim0:
            v[0] = lim0
        for i in range(1, len(v)):
            lim = math.sqrt(v[i - 1] * v[i - 1] + two_a * ds[i - 1])
            if v[i] > lim:
                v[i] = lim
        for w, vi in zip(out, v):
            w.vx_mps = float(vi)
        return out

    def _pub_local_wpnts(self, wpts):
        # DELETEALL as the first element of the SAME array (atomic clear+draw in
        # one message) instead of a separate publish, so RViz2 doesn't flicker.
        # Net result matches ROS1 (clear stale markers, then draw the new ones).
        loc_markers = MarkerArray()
        del_mrk = Marker()
        del_mrk.header.stamp = self.get_clock().now().to_msg()
        # set always, so RViz does not drop the DELETEALL for an empty frame and leave the previous
        # markers on screen -- the same reason it is set on the trailing/overtaking targets below
        del_mrk.header.frame_id = "map"
        del_mrk.action = Marker.DELETEALL
        loc_markers.markers.append(del_mrk)

        loc_wpnts = WpntArray()
        loc_wpnts.wpnts = wpts if wpts is not None else []
        loc_wpnts.header.stamp = self.get_clock().now().to_msg()
        loc_wpnts.header.frame_id = "map"

        v_ref = max(float(getattr(self, "max_speed", 0.0) or 0.0), 1e-3)
        for i, wpnt in enumerate(loc_wpnts.wpnts):
            # ONE bad pose makes RViz drop the WHOLE MarkerArray, which is indistinguishable from
            # "the local waypoints vanished". Skip the point instead of losing the path.
            if not (np.isfinite(wpnt.x_m) and np.isfinite(wpnt.y_m)):
                self.get_logger().warn(
                    f"[{self.name}] local waypoint {i} has a non-finite position; skipping it in "
                    f"the marker array so the rest of the path still draws",
                    throttle_duration_sec=2.0)
                continue
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.type = mrk.SPHERE
            mrk.scale.x = 0.15
            mrk.scale.y = 0.15
            mrk.scale.z = 0.15
            mrk.color.a = 1.0
            mrk.id = i + 1  # 0 reserved for the DELETEALL marker (avoid duplicate id in the array)
            mrk.pose.position.x = wpnt.x_m
            mrk.pose.position.y = wpnt.y_m
            # z = 0.0, NOT the speed. Drawing the beads at "speed height" lifted them metres off
            # the ground and dropped them again through every braking zone -- so beside an obstacle,
            # exactly where the path slows, the trail appeared to switch off. The speed is still
            # there, as COLOUR: green = fast, red = slow, against the configured maximum.
            mrk.pose.position.z = 0.0
            v = float(wpnt.vx_mps) if np.isfinite(wpnt.vx_mps) else 0.0
            f = float(np.clip(v / v_ref, 0.0, 1.0))
            mrk.color.g = f
            mrk.color.r = 1.0 - f
            mrk.pose.orientation.w = 1.0
            loc_markers.markers.append(mrk)

        self.loc_wpnt_pub.publish(loc_wpnts)
        self.vis_loc_wpnt_pub.publish(loc_markers)

    def visualize_state(self, state: str):
        if self.first_visualization:
            self.first_visualization = False
            x0 = self.cur_gb_wpnts.list[0].x_m
            y0 = self.cur_gb_wpnts.list[0].y_m
            x1 = self.cur_gb_wpnts.list[1].x_m
            y1 = self.cur_gb_wpnts.list[1].y_m
            xy_norm = (
                -np.array([y1 - y0, x0 - x1]) / np.linalg.norm([y1 - y0, x0 - x1])
                * 1.25 * self.cur_gb_wpnts.list[0].d_left
            )
            self.x_viz = x0 + xy_norm[0]
            self.y_viz = y0 + xy_norm[1]

        mrk = Marker()
        mrk.type = mrk.SPHERE
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.color.a = 1.0
        mrk.pose.position.x = float(self.x_viz)
        mrk.pose.position.y = float(self.y_viz)
        mrk.pose.position.z = 0.0
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 1.0
        mrk.scale.y = 1.0
        mrk.scale.z = 1.0

        if state == "GB_TRACK":
            mrk.color.b = 1.0
        elif state == "OVERTAKE":
            mrk.color.r = 1.0
            mrk.color.g = 0.0
            mrk.color.b = 0.0
        elif state == "TRAILING":
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 0.0
        elif state == "ATTACK":
            mrk.color.r = 1.0
            mrk.color.g = 0.0
            mrk.color.b = 1.0
        elif state == "FTGONLY":
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 1.0
        elif state == "RECOVERY":
            mrk.color.r = 0.0
            mrk.color.g = 1.0
            mrk.color.b = 0.0
        else:
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 1.0
        self.state_mrk.publish(mrk)

    def publish_not_ready_marker(self):
        mrk = Marker()
        mrk.type = mrk.TEXT_VIEW_FACING
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.color.a = 1.0
        mrk.color.r = 1.0
        mrk.color.g = 0.0
        mrk.color.b = 0.0
        mrk.pose.position.x = float(np.mean([wpnt.x_m for wpnt in self.cur_gb_wpnts.list]))
        mrk.pose.position.y = float(np.mean([wpnt.y_m for wpnt in self.cur_gb_wpnts.list]))
        mrk.pose.position.z = 1.0
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 4.69
        mrk.scale.y = 4.69
        mrk.scale.z = 4.69
        mrk.text = "BATTERY TOO LOW!!!"
        self.emergency_pub.publish(mrk)

    def update_waypoints(self):
        # Rebuild the cached ARRAY whenever the message actually changed, not just on the first
        # one. The old `else` branch refreshed `.list` but left `.array` frozen at the very first
        # /global_waypoints_scaled, so after a static-reopt line swap every position-based query
        # (anchor_gb_index) would have been answered against the PRE-swap geometry. Keyed on the
        # stamp so this costs one array build per publish (2 Hz), not per SM cycle (80 Hz).
        if not self.cur_gb_wpnts.is_init or self.cur_gb_wpnts.stamp != self.gb_wpnts.header.stamp:
            self.cur_gb_wpnts.initialize_traj(self.gb_wpnts)
        else:
            self.cur_gb_wpnts.list = self.gb_wpnts.wpnts
        # Refresh the FOLLOWED avoidance trajectory only when the newly published spline MEANINGFULLY
        # differs from the cached one (a re-detected / new obstacle changed the required lateral offset).
        # Refreshing every cycle made the controller chase the planner's per-cycle re-anchoring jitter
        # (path oscillation); never refreshing left OVERTAKE stuck on a stale path when a new obstacle
        # appeared. So: keep the STABLE cached path for small cycle-to-cycle wiggles (only bump its
        # freshness stamp) and swap to the new one only on a real change. Empty/stale publishes are left
        # cached so the availability timers still expire OVERTAKE once the obstacle is passed.
        # Refresh the FOLLOWED path only on a genuine NEW requirement: a BIGGER offset (a closer / new
        # obstacle needs more room) or the OPPOSITE side (a new obstacle on the other side). A smaller
        # same-side peak is just the planner returning to the raceline as the CURRENT obstacle slides
        # behind -- following that mid-maneuver cuts the car back while still beside the box (the "path
        # twists" symptom), so keep the committed spline (its exit ramp already eases back AFTER the box).
        # When we KEEP the cached path we do NOT touch its stamp: _check_on_spline already sustains
        # OVERTAKE while the car is on it, and letting the stamp age lets the availability timers expire
        # OVERTAKE normally once the obstacle is passed (bumping the stamp used to wedge OVERTAKE on so it
        # never exited / re-triggered).
        # Freeze the obstacle set for this cycle BEFORE the adoption rules below: rule (iii) asks
        # the free-check about the fresh path and must see the same obstacles every other consumer
        # will see this cycle.
        self.cur_obstacles_in_interest = self.obstacles_in_interest

        OT_REFRESH_D_THRESH = 0.15   # [m] peak-offset change that counts as a new path (not tracking jitter)
        for src, cur in ((self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts),
                         (self.avoidance_wpnts, self.cur_avoidance_wpnts)):
            if src is None or len(src.wpnts) == 0:
                continue
            if (self.now_sec() - time_to_float(src.header.stamp)) > cur.latest_threshold:
                continue
            if not cur.is_init or cur.list is None or len(cur.list) == 0:
                cur.initialize_traj(src)
                continue
            peak_src = max((w.d_m for w in src.wpnts), key=abs, default=0.0)
            peak_cur = max((w.d_m for w in cur.list), key=abs, default=0.0)
            if peak_src * peak_cur >= 0.0:            # same side (or one is ~raceline)
                refresh = abs(peak_src) - abs(peak_cur) > OT_REFRESH_D_THRESH   # only if it needs MORE offset
            else:                                     # opposite side -> a new obstacle the other way
                refresh = abs(peak_src) > OT_REFRESH_D_THRESH
            # (ii) A FORWARD RE-SLICE of the geometry already committed to. The planner freezes its
            # path in the world and republishes the part still ahead of the car, so the shape does
            # not change -- only where it starts. The rule above sees no NEW requirement in that and
            # kept the cache, which froze the cached START while the car drove on: the published
            # window then began metres behind the car and stayed there until the availability
            # timers expired (up to hyst_timer_sec = 4 s). Adopting it every cycle is precisely
            # what makes the window follow the car, and it cannot change the geometry, because
            # agreeing with the cache everywhere they overlap is the test.
            if not refresh and self._is_forward_reslice(src, cur):
                refresh = True
            # (iii) The cached path is BLOCKED and the fresh one is not. Without this the SM can
            # sit on a cached path the free-check rejects while the planner is already publishing a
            # good one, and every gate keyed on the cache says "not free" -- the planner re-plans,
            # the SM refuses to look. A path that passes when the held one fails is never something
            # to hold out against.
            if not refresh and not self._check_free_frenet(cur):
                probe = copy.copy(cur)
                probe.initialize_traj(src)
                if self._check_free_frenet(probe):
                    refresh = True
                    self.get_logger().info(
                        f"[{self.name}] adopting fresh {cur.name} path: the cached one reads "
                        f"NOT-free and this one is free", throttle_duration_sec=1.0)
            if refresh:
                cur.initialize_traj(src)              # real new requirement -> follow the new path

        # RECOVERY has no committed-path semantics -- it exists to rejoin the raceline -- so it is
        # adopted whenever it is fresh. It used to be adopted as a side effect of
        # _check_latest_wpnts; now that that function is pure, this is its only writer.
        if self.recovery_wpnts is not None and len(self.recovery_wpnts.wpnts) != 0:
            if (self.now_sec() - time_to_float(self.recovery_wpnts.header.stamp)) \
                    <= self.cur_recovery_wpnts.latest_threshold:
                self.cur_recovery_wpnts.initialize_traj(self.recovery_wpnts)
        return

    def _is_forward_reslice(self, src, cur) -> bool:
        """Is `src` the same world-fixed path as `cur`, just re-sliced further forward?

        Tested where the two OVERLAP in s, not by comparing peaks: once the car passes the apex the
        fresh slice no longer contains it, so a peak comparison reads a large change and calls a
        pure re-slice a new path -- exactly when keeping the window on the car matters most. The
        end station is compared too, because that is what a genuinely NEW plan moves and a re-slice
        does not.
        """
        try:
            L = self.track_length if self.track_length else self.max_s
            if not L:
                return False
            s_src = np.fromiter((w.s_m for w in src.wpnts), float, len(src.wpnts))
            d_src = np.fromiter((w.d_m for w in src.wpnts), float, len(src.wpnts))
            s_cur, d_cur = cur.array[:, 2], cur.array[:, 3]
            if abs(((s_src[-1] - s_cur[-1] + L / 2.0) % L) - L / 2.0) > self.reslice_end_s_tol_m:
                return False
            ds = np.abs(((s_src[:, None] - s_cur[None, :] + L / 2.0) % L) - L / 2.0)
            j = np.argmin(ds, axis=1)
            near = ds[np.arange(len(s_src)), j] <= 0.5 * max(self.wpnt_dist, 1e-3) + 1e-6
            if not near.any():
                return False
            return bool(np.max(np.abs(d_src[near] - d_cur[j[near]])) <= self.reslice_d_tol_m)
        except Exception:
            return False

    def get_overtaking_target(self):
        if self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target]
        if self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target]
        else:
            return []

    def get_traling_target(self):
        if self.local_wpnts_src == StateType.GB_TRACK and self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target]
        elif self.local_wpnts_src == StateType.RECOVERY and self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target]
        elif self.local_wpnts_src == StateType.OVERTAKE and self.ot_closest_target is not None:
            return [self.ot_closest_target]
        else:
            return []

    def get_farthest_target(self, local_wpnts_src):
        if local_wpnts_src == StateType.GB_TRACK and self.cur_gb_wpnts.closest_target is not None:
            closest_target = self.cur_gb_wpnts.closest_target
            closest_gap = self.cur_gb_wpnts.closest_gap
            if self.cur_avoidance_wpnts.closest_target is not None and closest_gap <= self.cur_avoidance_wpnts.closest_gap:
                closest_gap = self.cur_avoidance_wpnts.closest_gap
                closest_target = self.cur_avoidance_wpnts.closest_target
                local_wpnts_src = StateType.OVERTAKE
            if self.cur_static_avoidance_wpnts.closest_target is not None and \
                    closest_gap < self.cur_static_avoidance_wpnts.closest_gap:
                closest_gap = self.cur_static_avoidance_wpnts.closest_gap
                closest_target = self.cur_static_avoidance_wpnts.closest_target
                local_wpnts_src = StateType.OVERTAKE
            if self.cur_start_wpnts.closest_target is not None and closest_gap < self.cur_start_wpnts.closest_gap:
                closest_gap = self.cur_start_wpnts.closest_gap
                closest_target = self.cur_start_wpnts.closest_target
                local_wpnts_src = StateType.START
            return [closest_target], local_wpnts_src

        if local_wpnts_src == StateType.RECOVERY and self.cur_recovery_wpnts.closest_target is not None:
            closest_target = self.cur_recovery_wpnts.closest_target
            closest_gap = self.cur_recovery_wpnts.closest_gap
            if self.cur_avoidance_wpnts.closest_target is not None and closest_gap < self.cur_avoidance_wpnts.closest_gap:
                closest_gap = self.cur_avoidance_wpnts.closest_gap
                closest_target = self.cur_avoidance_wpnts.closest_target
                local_wpnts_src = StateType.OVERTAKE
            if self.cur_static_avoidance_wpnts.closest_target is not None and \
                    closest_gap < self.cur_static_avoidance_wpnts.closest_gap:
                closest_gap = self.cur_static_avoidance_wpnts.closest_gap
                closest_target = self.cur_static_avoidance_wpnts.closest_target
                local_wpnts_src = StateType.OVERTAKE
            if self.cur_start_wpnts.closest_target is not None and closest_gap < self.cur_start_wpnts.closest_gap:
                closest_gap = self.cur_start_wpnts.closest_gap
                closest_target = self.cur_start_wpnts.closest_target
                local_wpnts_src = StateType.START
            return [closest_target], local_wpnts_src

        if local_wpnts_src == StateType.OVERTAKE:
            # Reached while TRAILING holds the avoidance geometry as its reference (see
            # _hold_static_avoidance_reference). The SOURCE changed; the target did not -- the car
            # is still keeping a gap to the same obstacle. Without this the branch fell through to
            # `return []` and TRAILING lost its target the moment the reference switched, i.e. the
            # gap PID had nothing to brake for on the exact path where it must.
            for wd in (self.cur_static_avoidance_wpnts, self.cur_gb_wpnts):
                if wd.closest_target is not None:
                    return [wd.closest_target], local_wpnts_src

        return [], local_wpnts_src

    def check_ot_cloest_target(self):
        if self.gb_closest_target is not None and self.ot_closest_target is not None and \
                self.local_wpnts_src == StateType.GB_TRACK:
            if self.ot_closest_gap > self.gb_closest_gap:
                self.local_wpnts_src = StateType.OVERTAKE
        elif self.cur_recovery_wpnts.closest_target is not None and self.ot_closest_target is not None and \
                self.local_wpnts_src == StateType.RECOVERY:
            if self.ot_closest_gap > self.cur_recovery_wpnts.closest_gap:
                self.local_wpnts_src = StateType.OVERTAKE

    def save_params_to_yaml(self):
        # ROS1 dynamic_statemachine_server.save_yaml: persist the dynamic tunables to
        # state_machine_params.yaml, preserving the other keys.
        import yaml
        try:
            path = os.path.join(get_package_share_directory("stack_master"),
                                "config", "state_machine_params.yaml")
        except Exception:
            self.get_logger().error(f"[{self.name}] cannot locate state_machine_params.yaml")
            return
        keys = ["lateral_width_gb_m", "lateral_width_ot_m", "overtaking_ttl_sec",
                "splini_hyst_timer_sec", "splini_ttl", "pred_splini_ttl",
                "emergency_break_horizon", "ftg_speed_mps", "ftg_timer_sec",
                "ftg_active", "force_GBTRACK"]
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            block = data.setdefault("state_machine", {}).setdefault("ros__parameters", {})
            for k in keys:
                if self.has_parameter(k):
                    block[k] = self.get_parameter(k).value
            block["save_params"] = False
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f"[{self.name}] saved params to {path}")
        except Exception as e:
            self.get_logger().error(f"[{self.name}] failed to save params: {e}")

    def _handle_momentary_params(self):
        # Act on the rqt buttons outside the on-set callback so set_parameters() is safe.
        if self._save_start_traj_requested:
            self._save_start_traj_requested = False
            self.save_start_traj_pub.publish(Bool(data=True))
            self.set_parameters([rclpy.parameter.Parameter('save_start_traj', rclpy.Parameter.Type.BOOL, False)])
        if self._save_params_requested:
            self._save_params_requested = False
            self.save_params_to_yaml()
            self.set_parameters([rclpy.parameter.Parameter('save_params', rclpy.Parameter.Type.BOOL, False)])

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        if self._rate_check is not None:
            self._rate_check.tick()
        self._handle_momentary_params()
        if self.measuring:
            start = time.perf_counter()

        # Unconditional, every cycle, before any transition logic runs: the planner gates its
        # engage on this and fails closed when it goes stale.
        self._publish_ot_section_check()

        self.update_waypoints()
        self.gb_closest_target = None
        self.ot_closest_target = None
        need_vel_planner = False

        self.cur_gb_wpnts.closest_target = None
        self.cur_recovery_wpnts.closest_target = None
        self.cur_avoidance_wpnts.closest_target = None
        self.cur_static_avoidance_wpnts.closest_target = None
        self.cur_start_wpnts.closest_target = None

        # Drop an OT path (dynamic/static) not selected as local_wpnts_src for >2 s, else the
        # stale near-raceline path keeps passing _check_on_spline and flips GB<->OVERTAKE.
        self._expire_unused_ot_cache(self.cur_avoidance_wpnts, 2.0)
        self._expire_unused_ot_cache(self.cur_static_avoidance_wpnts, 2.0)

        # safety check
        if self.cur_volt < self.volt_threshold:
            self.get_logger().error(
                f"[{self.name}] VOLTS TOO LOW, STOP THE CAR", throttle_duration_sec=1.0
            )
            self.publish_not_ready_marker()

        if self.force_gbtrack_state:
            self._commit_state(StateType.GB_TRACK, StateType.GB_TRACK, force=True)
        elif self._check_only_ftg_zone():
            self._commit_state(StateType.FTGONLY, StateType.FTGONLY, force=True)
            self.get_logger().warn(f"[{self.name}] FTGONLY sector !!!")
        else:
            prev_state = self.cur_state
            proposed_state, proposed_src = self.state_transitions[self.cur_state](self)
            # Own the overtaking-ttl side-effect that used to live in OvertakingTransition (keeps
            # the transition functions pure) and apply the min_dwell transition hysteresis.
            self._update_overtake_ttl(prev_state, proposed_state)
            self._commit_state(proposed_state, proposed_src)

        if self.cur_state == StateType.TRAILING:
            self.check_ot_cloest_target()
            self.behavior_strategy.trailing_targets, self.local_wpnts_src = \
                self.get_farthest_target(self.local_wpnts_src)
        else:
            self.behavior_strategy.trailing_targets = []

        # Mark the chosen overtake cache as used so it isn't expired next frame.
        if self.local_wpnts_src == StateType.OVERTAKE:
            used = self.cur_static_avoidance_wpnts if self.static_overtaking_mode else self.cur_avoidance_wpnts
            used.last_used_sec = self.now_sec()

        self.behavior_strategy.overtaking_targets = self.get_overtaking_target()

        local_wpnts = self.states[self.local_wpnts_src](self)
        # ONE PLACE, EVERY STATE. The assembled window has seams nothing else bounds: the
        # avoidance-to-global-padding join (measured |dvx| up to 0.894 m/s = 41-55 m/s^2) and the
        # global raceline's own s = 0 seam (0.867 m/s = 35.6 m/s^2, in 21.5% of windows). Applying
        # the limit HERE covers /behavior_strategy (which the controller reads) and
        # /local_waypoints together, and covers GB_TRACK, OVERTAKE, RECOVERY and START without
        # four copies of the same pass.
        local_wpnts = self.limit_local_window_accel(local_wpnts, self.cur_vs)

        if self.cur_state == StateType.LOSTLINE:
            self.cur_state = StateType.GB_TRACK

        need_vel_planner = False
        self.behavior_strategy.header.stamp = self.get_clock().now().to_msg()
        self.behavior_strategy.local_wpnts = local_wpnts if local_wpnts is not None else []
        self.behavior_strategy.state = self.cur_state.value
        self.behavior_strategy.need_vel_planner = need_vel_planner

        self.behavior_strategy_pub.publish(self.behavior_strategy)

        self.state_pub.publish(String(data=self.cur_state.value))
        self.visualize_state(state=self.cur_state.value)

        self._pub_local_wpnts(local_wpnts)

        if self.cur_state != StateType.TRAILING and self.cur_state != StateType.ATTACK:
            self.ftg_counter = 0

        overtaking_target_mrk = Marker()
        overtaking_target_mrk.header.frame_id = "map"  # set always so the DELETEALL marker isn't dropped by RViz (empty frame)
        if len(self.behavior_strategy.overtaking_targets) != 0:
            overtaking_target_mrk.type = Marker.SPHERE
            overtaking_target_mrk.scale.x = 0.5
            overtaking_target_mrk.scale.y = 0.5
            overtaking_target_mrk.scale.z = 0.5
            overtaking_target_mrk.color.a = 1.0
            overtaking_target_mrk.color.b = 1.0
            overtaking_target_mrk.pose.position.x = self.behavior_strategy.overtaking_targets[0].x_m
            overtaking_target_mrk.pose.position.y = self.behavior_strategy.overtaking_targets[0].y_m
            overtaking_target_mrk.pose.orientation.w = 1.0
        else:
            overtaking_target_mrk.action = Marker.DELETEALL
        self.overtaking_marker_pub.publish(overtaking_target_mrk)

        trailing_target_mrk = Marker()
        trailing_target_mrk.header.frame_id = "map"  # set always so the DELETEALL marker isn't dropped by RViz (empty frame)
        if len(self.behavior_strategy.trailing_targets) != 0:
            trailing_target_mrk.type = Marker.SPHERE
            trailing_target_mrk.scale.x = 0.5
            trailing_target_mrk.scale.y = 0.5
            trailing_target_mrk.scale.z = 0.5
            trailing_target_mrk.color.a = 1.0
            trailing_target_mrk.color.g = 1.0
            trailing_target_mrk.pose.position.x = self.behavior_strategy.trailing_targets[0].x_m
            trailing_target_mrk.pose.position.y = self.behavior_strategy.trailing_targets[0].y_m
            trailing_target_mrk.pose.orientation.w = 1.0
        else:
            trailing_target_mrk.action = Marker.DELETEALL
        self.trailing_marker_pub.publish(trailing_target_mrk)

        if self.measuring:
            end = time.perf_counter()
            self.latency_pub.publish(Float32(data=1.0 / (end - start)))


# defined as entry point in setup.py:
def main(args=None):
    rclpy.init(args=args)
    state_machine = StateMachine()
    try:
        rclpy.spin(state_machine)
    except KeyboardInterrupt:
        pass
    state_machine.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
