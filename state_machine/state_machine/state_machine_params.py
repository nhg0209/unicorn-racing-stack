from __future__ import annotations

from typing import List, TYPE_CHECKING

from rcl_interfaces.msg import (
    SetParametersResult,
    ParameterDescriptor,
    ParameterType,
    FloatingPointRange,
    IntegerRange,
)
from rclpy.parameter import Parameter

if TYPE_CHECKING:
    from state_machine.state_machine_node import StateMachine

"""
Replaces the ROS1 `dynamic_reconfigure` server (`dyn_statemachine_tuner.cfg`) and the
plain `rospy.get_param` reads in the ROS1 state machine node.

All tunable parameters are declared here with descriptors (replacing the .cfg ranges) and
live-updated through a single `add_on_set_parameters_callback` registered on the node.
"""


class StateMachineParams:
    # dynamic params that the node also mirrors as a same-named attribute
    _NODE_MIRRORED_PARAMS = {
        "lateral_width_gb_m",
        "lateral_width_static_gb_m",
        "lateral_width_ot_m",
        "splini_hyst_timer_sec",
        "emergency_break_horizon",
        "ftg_speed_mps",
        "ftg_timer_sec",
        "gb_ego_width_m",
        "recovery_exit_d_m",
        "recovery_entry_d_m",
        "gb_horizon_m",
        "interest_horizon_m",
        "reframe_warn_m",
        "squeeze_speed_cap_mps",
        "avoidance_ay_max",
        "static_invisible_grace_sec",
        "overtaking_horizon_m",
        "getting_closer_rel_vel_mps",
        "local_window_accel_limit_enable",
        "local_window_a_long_mps2",
        "min_dwell_sec",
        "ot_free_lost_sec",
        "free_check_predict_dynamic",
        "free_check_pass_speed",
        "free_check_dynamic_ot_slow",
        "relax_preemptive_enable",
        "relax_preemptive_gap_m",
        "relax_preemptive_dwell_s",
        "relax_preemptive_max_speed_mps",
    }

    def __init__(self, node: "StateMachine") -> None:
        self.node = node

        # ------------------------------------------------------------------ #
        # NON-DYNAMIC / STRUCTURAL PARAMETERS (read once)
        # ------------------------------------------------------------------ #
        self._declare(
            "rate", 80,
            ParameterDescriptor(
                description="Rate at which the state machine runs in Hz",
                type=ParameterType.PARAMETER_INTEGER,
                integer_range=[IntegerRange(from_value=1, to_value=200, step=1)],
            ),
        )
        self.rate_hz: int = node.get_parameter("rate").value

        self._declare(
            "n_loc_wpnts", 80,
            ParameterDescriptor(
                description="Number of local waypoints published",
                type=ParameterType.PARAMETER_INTEGER,
            ),
        )
        self.n_loc_wpnts: int = node.get_parameter("n_loc_wpnts").value

        self._declare("timetrials_only", False)
        self.timetrials_only: bool = node.get_parameter("timetrials_only").value

        self._declare("sim", True)
        self.sim: bool = node.get_parameter("sim").value

        self._declare("measure", False)
        self.measuring: bool = node.get_parameter("measure").value

        self._declare("racecar_version", "NUCX")
        self.racecar_version: str = node.get_parameter("racecar_version").value

        self._declare(
            "ot_planner", "predictive_spliner",
            ParameterDescriptor(
                description="Overtaking planner: spliner, predictive_spliner or graph_based",
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        self.ot_planner: str = node.get_parameter("ot_planner").value

        self._declare("track_length", 1.0)
        self.track_length: float = node.get_parameter("track_length").value

        self._declare(
            "volt_threshold", 11.0,
            ParameterDescriptor(
                description="Voltage threshold below which the car is considered low bat",
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        self.volt_threshold: float = node.get_parameter("volt_threshold").value

        self._declare(
            "gb_ego_width_m", 0.4,
            ParameterDescriptor(
                description="Distance from gb path for rejoining in meters",
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        self.gb_ego_width_m: float = node.get_parameter("gb_ego_width_m").value

        # The assembled local window is where the two speed seams meet -- the avoidance-to-global
        # padding join and the global raceline's own s = 0 discontinuity -- and nothing downstream
        # bounds d(vx)/ds. The state machine does it once, on the copy it publishes.
        self._declare(
            "local_window_accel_limit_enable", True,
            ParameterDescriptor(
                description=(
                    "Bound d(vx)/ds over the ASSEMBLED local window (one backward + one forward "
                    "pass) before publishing. Measured with it off: required |a_long| p50/p95/max "
                    "4.88/35.6/56.8 m/s^2 against a ggv ax_max of 5.0, and 104 of the 137 "
                    "stations over 6 m/s^2 come from the raceline's s = 0 seam alone. False "
                    "publishes the seams as assembled."),
                type=ParameterType.PARAMETER_BOOL,
            ),
        )
        self.local_window_accel_limit_enable: bool = node.get_parameter(
            "local_window_accel_limit_enable").value

        self._declare(
            "local_window_a_long_mps2", 5.0,
            ParameterDescriptor(
                description=(
                    "[m/s^2] the longitudinal bound the pass above enforces. 5.0 is the ggv / "
                    "ax_max_machines limit the velocity profile is itself solved to, so this only "
                    "removes what the ASSEMBLY introduced and never re-shapes a feasible "
                    "profile."),
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        self.local_window_a_long_mps2: float = node.get_parameter(
            "local_window_a_long_mps2").value

        self._declare(
            "recovery_exit_d_m", 0.2,
            ParameterDescriptor(
                description=(
                    "|d| below which the car counts as back on the raceline when LEAVING a "
                    "non-GB_TRACK state (RECOVERY/TRAILING/OVERTAKE/START/FTGONLY). Entry into "
                    "RECOVERY uses gb_ego_width_m, so this is the lower half of the hysteresis "
                    "band and must stay below it. Was hardcoded at 0.05 m -- tighter than normal "
                    "tracking error, so RECOVERY never exited [m]"
                ),
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        self.recovery_exit_d_m: float = node.get_parameter("recovery_exit_d_m").value

        self._declare(
            "recovery_entry_d_m", 0.4,
            ParameterDescriptor(
                description=(
                    "|d| at or above which the car counts as having lost the raceline, i.e. the "
                    "RECOVERY ENTRY threshold (_check_line_lost). Upper half of the hysteresis "
                    "band and must stay above recovery_exit_d_m. Entry used to be driven by the "
                    "per-state close_to_raceline flag instead, which made the bar tighter while "
                    "TRAILING than while GB_TRACKing and latched the car onto the recovery "
                    "spline whenever it trailed an opponent [m]"
                ),
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        self.recovery_entry_d_m: float = node.get_parameter("recovery_entry_d_m").value

        self._declare(
            "gb_horizon_m", 15.0,
            ParameterDescriptor(
                description="Horizon considered on the global waypoints to check for obstacles [m]",
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        self.gb_horizon_m: float = node.get_parameter("gb_horizon_m").value

        self._declare(
            "interest_horizon_m", 20.0,
            ParameterDescriptor(
                description="Horizon in which obstacles are considered of interest [m]",
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        self.interest_horizon_m: float = node.get_parameter("interest_horizon_m").value

        self._declare(
            "reframe_warn_m", 0.05,
            ParameterDescriptor(
                description="Warn when an incoming obstacle's (s,d) has to be re-anchored into "
                            "this node's frenet frame by more than this [m]. A large value means "
                            "upstream tracking is not re-projecting on a static_reopt line swap.",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)],
            ),
        )
        self.reframe_warn_m: float = node.get_parameter("reframe_warn_m").value

        self._declare(
            "squeeze_speed_cap_mps", 2.5,
            ParameterDescriptor(
                description="Speed ceiling [m/s] applied to a static-avoidance path the planner "
                            "marked ot_line='squeeze' -- one it could only solve by reducing its "
                            "clearance margins. The geometry is legal but the error budget is "
                            "spent, so it must not be driven at raceline pace.",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.5, to_value=10.0, step=0.1)],
            ),
        )
        self.squeeze_speed_cap_mps: float = node.get_parameter("squeeze_speed_cap_mps").value

        self._declare(
            "avoidance_ay_max", 5.0,
            ParameterDescriptor(
                description="Lateral-accel limit [m/s^2] used when THIS node re-profiles an "
                            "avoidance path, replacing the global ggv's ay column for that path "
                            "only. The ggv is tuned for the raceline; an avoidance is a brief "
                            "deliberate excursion the planner itself already sizes at a higher "
                            "a_lat_max, so re-profiling it at the raceline limit is what made the "
                            "avoidance spline crawl. Does not affect the global line.",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=1.0, to_value=12.0, step=0.1)],
            ),
        )
        self.avoidance_ay_max: float = node.get_parameter("avoidance_ay_max").value

        self._declare(
            "static_invisible_grace_sec", 1.5,
            ParameterDescriptor(
                description="How long [s] a STATIC obstacle stays in the interest list after it "
                            "was last reported is_visible. The flag is a per-frame lidar verdict "
                            "that drops out on occlusion / FOV edge / sparse returns at range, "
                            "while a static obstacle cannot actually leave -- so acting on each "
                            "drop flipped the state and the trailing target for that cycle. 0 "
                            "restores the undebounced behaviour.",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.1)],
            ),
        )
        self.static_invisible_grace_sec: float = \
            node.get_parameter("static_invisible_grace_sec").value

        self._declare(
            "overtaking_horizon_m", 6.9,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.overtaking_horizon_m: float = node.get_parameter("overtaking_horizon_m").value

        self._declare("emergency_break_horizon", 0.5)
        self.emergency_break_horizon: float = node.get_parameter("emergency_break_horizon").value

        # ------------------------------------------------------------------ #
        # DYNAMIC PARAMETERS (replaces dyn_statemachine_tuner.cfg)
        # ------------------------------------------------------------------ #
        self._declare(
            "lateral_width_gb_m", 0.3,
            ParameterDescriptor(
                description="Threshold to raceline for GB_FREE in meters",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.1, to_value=1.75, step=0.05)],
            ),
        )
        self.lateral_width_gb_m: float = node.get_parameter("lateral_width_gb_m").value

        self._declare(
            "lateral_width_static_gb_m", 0.05,
            ParameterDescriptor(
                description="GB_FREE margin vs STATIC obstacles, distance-independent [m]. The "
                            "requirement gb_ego_width_m/2 + this must stay at or below the "
                            "reactive planner's clear-gate STAY threshold (width_car/2 + "
                            "clear_margin_m), which in turn stays below the enforced re-opt "
                            "clearance floor (reopt_obs_margin) minus slack. Anything above the "
                            "clear-gate threshold opens a dead band where the SM reads the "
                            "swapped line as blocked while the planner idles -> TRAILING behind "
                            "a cleared line. Checked by check_avoidance_margins.py.",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)],
            ),
        )
        self.lateral_width_static_gb_m: float = node.get_parameter("lateral_width_static_gb_m").value

        self._declare(
            "lateral_width_ot_m", 0.3,
            ParameterDescriptor(
                description="Threshold to raceline for O_FREE in meters",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.1, to_value=1.75, step=0.05)],
            ),
        )
        self.lateral_width_ot_m: float = node.get_parameter("lateral_width_ot_m").value

        self._declare(
            "splini_hyst_timer_sec", 0.7,
            ParameterDescriptor(
                description="Time between switching overtaking sides [s]",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.1, to_value=1.5, step=0.05)],
            ),
        )
        self.splini_hyst_timer_sec: float = node.get_parameter("splini_hyst_timer_sec").value

        self._declare(
            "splini_ttl", 2.0,
            ParameterDescriptor(
                description="Spliner ttl caching in seconds",
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )

        self._declare(
            "pred_splini_ttl", 2.0,
            ParameterDescriptor(
                description="Predictive spliner ttl caching in seconds",
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        # splini_ttl actually used depends on the planner (matches ROS1 logic)
        if self.ot_planner == "spliner":
            self.splini_ttl: float = node.get_parameter("splini_ttl").value
        else:
            self.splini_ttl = node.get_parameter("pred_splini_ttl").value

        self._declare(
            "overtaking_ttl_sec", 3.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.overtaking_ttl_sec: float = node.get_parameter("overtaking_ttl_sec").value

        self._declare(
            "ftg_speed_mps", 0.1,
            ParameterDescriptor(
                description="Speed threshold below which ftg counter is incremented [mps]",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.1, to_value=2.0, step=0.05)],
            ),
        )
        self.ftg_speed_mps: float = node.get_parameter("ftg_speed_mps").value

        # ---- PRE-EMPTIVE RELAX ------------------------------------------- #
        # _check_static_trailing_deadlock already asks the static planner for a
        # reduced-margin retry on /planner/avoidance/relax, but only after the car has
        # been under static_deadlock_speed_mps for static_deadlock_timeout_s -- i.e.
        # after it has already stopped. For an obstacle that only becomes visible late
        # (a corner exit, a box occluded until the car was committed) that is too late
        # to be a recovery: the gap PID converges on v = 0 first and the run ends with
        # the car parked short of a box it had the room to pass. These raise the SAME
        # request on the symptom -- no feasible candidate, obstacle close, still moving.
        self._declare(
            "relax_preemptive_enable", True,
            ParameterDescriptor(
                description="Ask the static planner for a reduced-margin retry BEFORE the car "
                            "has to stop, instead of only after the trailing deadlock",
                type=ParameterType.PARAMETER_BOOL,
            ),
        )
        self.relax_preemptive_enable: bool = node.get_parameter("relax_preemptive_enable").value

        self._declare(
            "relax_preemptive_gap_m", 6.0,
            ParameterDescriptor(
                description="Ask only once the nearest static obstacle is within this gap [m]",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=1.0, to_value=20.0, step=0.5)],
            ),
        )
        self.relax_preemptive_gap_m: float = node.get_parameter("relax_preemptive_gap_m").value

        self._declare(
            "relax_preemptive_dwell_s", 0.3,
            ParameterDescriptor(
                description="How long static_feasible must stay False before asking [s]. The flag "
                            "drops for a cycle whenever the planner is mid-solve, and each request "
                            "costs it its committed path",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=2.0, step=0.05)],
            ),
        )
        self.relax_preemptive_dwell_s: float = node.get_parameter("relax_preemptive_dwell_s").value

        self._declare(
            "relax_preemptive_max_speed_mps", 6.0,
            ParameterDescriptor(
                description="Do not ask above this speed [m/s]. A relax lets the planner squeeze "
                            "down to its floors (5 cm to the box, 8 cm to the wall), and the "
                            "squeeze speed gate it overrides exists because that trade is only "
                            "defensible where a mis-clearance is survivable. Setting this at or "
                            "below squeeze_max_speed_mps makes the feature a no-op: under that "
                            "speed the planner already squeezes without being asked",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=15.0, step=0.25)],
            ),
        )
        self.relax_preemptive_max_speed_mps: float = node.get_parameter(
            "relax_preemptive_max_speed_mps").value

        self._declare(
            "ftg_timer_sec", 3.0,
            ParameterDescriptor(
                description="Time threshold at which ftg is triggered [s]",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.5, to_value=7.0, step=0.05)],
            ),
        )
        self.ftg_timer_sec: float = node.get_parameter("ftg_timer_sec").value

        self._declare("ftg_active", False)
        self.ftg_active: bool = node.get_parameter("ftg_active").value

        self._declare("force_GBTRACK", False)
        self.force_GBTRACK: bool = node.get_parameter("force_GBTRACK").value

        self._declare("use_force_trailing", False)
        self.use_force_trailing: bool = node.get_parameter("use_force_trailing").value

        # static_ot_speed_mps REMOVED: the live config had it at 10.0 (= disabled), and a speed
        # guard on the commit is conceptually wrong for a STATIC obstacle — trailing one means
        # stopping behind it; the avoidance path's own slow-in velocity profile handles entry
        # speed. The commit gate is: fresh on-spline static path AND (fresh) feasible signal.

        self._declare(
            "getting_closer_rel_vel_mps", -0.5,
            ParameterDescriptor(
                description="Min (ego - obstacle) s-velocity to count as 'getting closer' [mps]",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=-5.0, to_value=5.0, step=0.1)],
            ),
        )
        self.getting_closer_rel_vel_mps: float = node.get_parameter("getting_closer_rel_vel_mps").value

        self._declare(
            "min_dwell_sec", 0.2,
            ParameterDescriptor(
                description="Minimum time [s] a state must be held before switching to a non-safe "
                            "state (transition hysteresis / anti-chatter). Switches TOWARD the safe "
                            "states (TRAILING, FTGONLY) bypass this and are allowed immediately.",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=2.0, step=0.05)],
            ),
        )
        self.min_dwell_sec: float = node.get_parameter("min_dwell_sec").value

        self._declare(
            "ot_free_lost_sec", 0.4,
            ParameterDescriptor(
                description="How long the dynamic avoidance path may read NOT-free before "
                            "OVERTAKE is dropped [s]. The free-check is a single-point test "
                            "against a tracked opponent, so one noisy cycle fails it; without "
                            "this debounce the car flaps between the overtake spline and the "
                            "raceline. Mirrors static_feasible_lost_sec on the static path. "
                            "ENTRY still requires the path to be free right now.",
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.0, step=0.05)],
            ),
        )
        self.ot_free_lost_sec: float = node.get_parameter("ot_free_lost_sec").value

        self._declare(
            "free_check_predict_dynamic", True,
            ParameterDescriptor(
                description="Free-check: for a DYNAMIC obstacle with no usable prediction, "
                            "propagate it over the ttc..tt0 window (when the ego is alongside) "
                            "instead of testing the path at its current s. Matches what the "
                            "prediction branch already does. Near-stationary obstacles are "
                            "unaffected -- they take the static branch and are never "
                            "propagated. False restores the current-s test.",
                type=ParameterType.PARAMETER_BOOL,
            ),
        )
        self.free_check_predict_dynamic: bool = node.get_parameter("free_check_predict_dynamic").value

        self._declare(
            "free_check_pass_speed", True,
            ParameterDescriptor(
                description="Free-check: time the alongside window (ttc/tt0) from the speed the "
                            "ego will run once the overtake is committed (local raceline pace), "
                            "not from the TRAILING closing speed. The trailing value collapses "
                            "to the 0.5 m/s floor and places the window metres too far ahead, "
                            "which forces the planner to hold its offset across track the pass "
                            "never reaches. False restores the old timing.",
                type=ParameterType.PARAMETER_BOOL,
            ),
        )
        self.free_check_pass_speed: bool = node.get_parameter("free_check_pass_speed").value

        self._declare(
            "free_check_dynamic_ot_slow", True,
            ParameterDescriptor(
                description="Free-check: keep a SLOW-but-moving opponent on the dynamic branch "
                            "when testing a dynamic OVERTAKE path, instead of reclassifying it "
                            "as static. The near-stationary reclassification exists to protect "
                            "the STATIC avoidance spline from bogus predicted trajectories; on "
                            "the dynamic path it is actively harmful, because the static branch "
                            "evaluates the lane at the obstacle's CURRENT s -- where an "
                            "overtaking lane sits on the raceline by design -- so the path reads "
                            "NOT-free forever and OVERTAKE can never commit. For a genuinely "
                            "stationary obstacle the ttc..tt0 propagation is a no-op, so nothing "
                            "regresses. False restores the old routing.",
                type=ParameterType.PARAMETER_BOOL,
            ),
        )
        self.free_check_dynamic_ot_slow: bool = node.get_parameter("free_check_dynamic_ot_slow").value

        # Momentary rqt buttons (ROS1: served by dynamic_statemachine_server). When set
        # true they trigger an action and reset to false (done in the node timer, not
        # here, so set_parameters() isn't called inside the on-set callback).
        self._declare("save_start_traj", False)
        self._declare("save_params", False)

    def _declare(self, name, default, descriptor=None):
        """Declare a parameter unless it has already been auto-declared from a yaml
        override (the node is created with
        automatically_declare_parameters_from_overrides=True). When it already exists
        we just (re)apply the descriptor so the GUI ranges still show up."""
        if self.node.has_parameter(name):
            if descriptor is not None:
                try:
                    self.node.set_descriptor(name, descriptor)
                except Exception:
                    pass
            return
        if descriptor is not None:
            self.node.declare_parameter(name, default, descriptor)
        else:
            self.node.declare_parameter(name, default)

    def parameters_callback(self, parameters: List[Parameter]) -> SetParametersResult:
        """Single callback that mirrors the ROS1 dyn_param_cb live update behaviour."""
        for param in parameters:
            name = param.name
            value = param.value

            if name == "rate":
                self.rate_hz = value
                if self.node.main_loop is not None:
                    self.node.main_loop.timer_period_ns = int(1e9 / value)
            elif name == "splini_ttl":
                if self.node.ot_planner == "spliner":
                    self.splini_ttl = value
                    self.node.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)
            elif name == "pred_splini_ttl":
                if self.node.ot_planner != "spliner":
                    self.splini_ttl = value
                    self.node.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)
            elif name == "overtaking_ttl_sec":
                self.overtaking_ttl_sec = value
                self.node.overtaking_ttl_count_threshold = int(value * self.rate_hz)
            elif name == "ftg_active":
                self.ftg_active = value
                self.node.ftg_disabled = not value
            elif name == "save_start_traj":
                # momentary: act + reset in the node timer (not inside this on-set cb)
                if value:
                    self.node._save_start_traj_requested = True
            elif name == "save_params":
                if value:
                    self.node._save_params_requested = True
            else:
                # generic live update for the remaining tunables
                setattr(self, name, value)

            # The node mirrors several of these dynamic params as its own attributes
            # (the _check_* conditions read them directly off the node, matching the
            # ROS1 dyn_param_cb live-update behaviour). Keep them in sync.
            if name in self._NODE_MIRRORED_PARAMS and hasattr(self.node, name):
                setattr(self.node, name, value)
            if name == "force_GBTRACK":
                self.node.force_gbtrack_state = value
            if name == "use_force_trailing":
                self.node.use_force_trailing = value

            self.node.get_logger().info(f"Parameter '{name}' was set to {value}")

        return SetParametersResult(successful=True)
