#!/usr/bin/env python3

import logging
import os

import numpy as np

from visualization_msgs.msg import Marker, MarkerArray


def load_ggv_ay(path):
    """Read a veh_dyn_info/ggv.csv into the [v_mps, ay_max_mps2] table this module needs.

    A five-line numpy read rather than a dependency, on purpose. The two other ggv readers in
    the stack (state_machine_node, stack_master/scripts/global_velocity_planner.py) both go
    through `tph.import_veh_dyn_info`, and trajectory_planning_helpers is NOT importable from
    the interpreter this control node runs under -- reusing them would buy consistency at the
    price of a new way for the controller to fail at startup. Folding all three onto one
    veh_dyn library is real work that is DEFERRED UNTIL AFTER THE RACE; when it happens, this
    function is the caller to delete.

    THERE IS NOW A SECOND COPY: planner/spliner/spliner/static_avoidance_node.py grew
    load_veh_dyn_tables / resolve_veh_dyn_limits reading the same files, for the same reason
    (one k in ggv.csv has to reach every consumer on race day). They are deliberately not shared
    -- a control node and a planner in different ROS packages should not depend on each other to
    read config -- and both are on the same deferred list.

    The full table is kept (not min/mean-folded to a scalar): every row reads the same 5.7
    today, so any reduction looks right, and would go on looking right the day the table
    becomes speed-dependent while being silently wrong.
    """
    return _ggv(path)[:, [0, 2]]


def _ggv(path):
    tbl = np.atleast_2d(np.loadtxt(path, delimiter=",", comments="#"))
    if tbl.shape[1] < 3:
        raise ValueError(f"{path}: expected 3 columns (v_mps, ax_max, ay_max), got {tbl.shape[1]}")
    return tbl


def load_veh_dyn(cfg_dir):
    """Everything the friction circle needs, from a config/<SIM|CAR> directory.

    Returns (ggv_ay, ggv_ax, ax_machines, b_ax_machines, dyn_model_exp): four [v_mps, value]
    tables to be INTERPOLATED at the current speed, and the exponent p of the combined-
    acceleration model. Nothing here is a constant in the controller -- ay_max and ax_max come
    from veh_dyn_info/ggv.csv, the drivetrain cap from veh_dyn_info/ax_max_machines.csv (which is
    genuinely speed-dependent: 9.5 at rest, 4.62 at 15 m/s), the BRAKE cap from
    veh_dyn_info/b_ax_max_machines.csv, and p from racecar_f110.ini's vel_calc_opts. They are the
    SAME inputs the offline velocity profile was solved against, which is the only reason the
    online check and the plan can agree.

    b_ax_max_machines joined this loader on 2026-08-12, when the slew limit stopped being a
    hand-copied 5.0 in controller.yaml and started being derived -- see _slew_limits.

    The .ini is read by per-key regex rather than parsed as a dict literal, because those
    blocks carry inline '#' comments that break literal_eval. That technique is lifted from
    planner/gb_optimizer/.../static_reopt_core.py::_load_veh_dyn, which reads the same files
    for the offline path; the code is not imported because it lives in a planner package and
    the control node has no business depending on one to read its own config. Two readers of
    one set of files is a duplication that belongs in the veh_dyn library deferred until after
    the race -- see load_ggv_ay.

    Raises if anything is missing; the caller reports that and leaves the limit inert.
    """
    import re

    vdi = os.path.join(cfg_dir, "veh_dyn_info")
    ggv = _ggv(os.path.join(vdi, "ggv.csv"))

    def _two_col(name):
        t = np.atleast_2d(np.loadtxt(os.path.join(vdi, name), delimiter=",", comments="#"))
        if t.shape[1] < 2:
            raise ValueError(f"{name}: expected 2 columns, got {t.shape[1]}")
        return t[:, [0, 1]]

    axm = _two_col("ax_max_machines.csv")
    bax = _two_col("b_ax_max_machines.csv")
    ini = os.path.join(cfg_dir, "racecar_f110.ini")
    with open(ini) as f:
        txt = f.read()
    m = re.search(r'"dyn_model_exp"\s*:\s*([-+0-9.eE]+)', txt)
    if not m:
        raise ValueError(f"{ini}: no dyn_model_exp in vel_calc_opts")
    p = float(m.group(1))
    if not 1.0 <= p <= 2.0:
        raise ValueError(f"{ini}: dyn_model_exp {p} outside the model's range [1.0, 2.0]")
    return ggv[:, [0, 2]], ggv[:, [0, 1]], axm, bax, p


class Controller:
    """This class implements the L1 / Pure-Pursuit controller for autonomous driving.
    Input and output topics are managed by the controller manager.

    ROS2 port note: the ROS1 MAP (steering-lookup) branch was intentionally removed;
    only Pure-Pursuit lateral control remains. The marker publisher and loggers are
    injected by the manager so this class stays ROS-node-free.
    """

    def __init__(self,
                t_clip_min,
                t_clip_max,
                m_l1,
                q_l1,

                curvature_factor,

                KP,
                KI,
                KD,
                heading_error_thres,
                steer_gain_for_speed,

                future_constant,

                speed_lookahead,
                lat_err_coeff,
                acc_scaler_for_steer,
                dec_scaler_for_steer,
                start_scale_speed,
                end_scale_speed,
                downscale_factor,
                speed_lookahead_for_steer,

                trailing_gap,
                trailing_vel_gain,
                trailing_p_gain,
                trailing_i_gain,
                trailing_d_gain,
                blind_trailing_speed,

                loop_rate,
                wheelbase,

                speed_factor_for_lat_err,
                speed_factor_for_curvature,

                speed_diff_thres,
                start_speed,
                start_curvature_factor,

                AEB_thres,

                converter,

                predict_pub=None,
                logger_info=logging.info,
                logger_warn=logging.warning,
            ):

        # Parameters from manager
        self.t_clip_min = t_clip_min
        self.t_clip_max = t_clip_max
        self.m_l1 = m_l1
        self.q_l1 = q_l1
        self.speed_lookahead = speed_lookahead
        self.lat_err_coeff = lat_err_coeff
        self.acc_scaler_for_steer = acc_scaler_for_steer
        self.dec_scaler_for_steer = dec_scaler_for_steer
        self.start_scale_speed = start_scale_speed
        self.end_scale_speed = end_scale_speed
        self.downscale_factor = downscale_factor
        self.speed_lookahead_for_steer = speed_lookahead_for_steer

        # marker publisher injected by the manager (ROS2: a Node-created publisher)
        self.predict_pub = predict_pub

        # L1 dist calc param
        self.curvature_factor = curvature_factor

        self.speed_factor_for_lat_err = speed_factor_for_lat_err
        self.speed_factor_for_curvature = speed_factor_for_curvature

        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.heading_error_thres = heading_error_thres
        self.steer_gain_for_speed = steer_gain_for_speed

        self.future_constant = future_constant

        self.trailing_gap = trailing_gap
        self.trailing_vel_gain = trailing_vel_gain
        self.trailing_p_gain = trailing_p_gain
        self.trailing_i_gain = trailing_i_gain
        self.trailing_d_gain = trailing_d_gain
        self.blind_trailing_speed = blind_trailing_speed

        self.loop_rate = loop_rate
        self.AEB_thres = AEB_thres
        self.AEB_thres_overtake = AEB_thres   # manager overrides from yaml after construction
        self.AEB_offline_d_thres = 0.1        # [m] max|d| above which the local window counts
                                              #     as an OFFSET line -> use AEB_thres_overtake
        # AEB LATCH (see AEB_for_weird_local_wpnt). A single threshold with no hysteresis and no
        # minimum hold turns any jitter around it into a per-cycle 2 m/s clamp toggling on and
        # off -- the sawtooth this function's own docstring records as 2.87 -> 2.00 -> 3.24 in two
        # cycles, and 86.8 m/s^2 on the real car. Release needs the distance to fall
        # AEB_release_hyst_m below the bar AND the clamp to have been held AEB_min_hold_s.
        self.AEB_release_hyst_m = 0.25        # [m] overridden from the yaml by the manager
        self.AEB_min_hold_s = 0.2             # [s]
        self._aeb_engaged = False
        self._aeb_cycles = 0                  # cycles the clamp has been held (this loop is fixed-rate)
        self.l1_lat_err_cap = t_clip_max      # manager overrides from yaml (uncapped by default)
        self.converter = converter
        # Longitudinal limits of the PUBLISHED speed command. FALLBACKS ONLY: _slew_limits derives
        # the real pair from config/<SIM|CAR>/veh_dyn_info at the current speed, and these are what
        # it uses when those files cannot be read (warned once). The manager still overrides them
        # from controller.yaml, so the yaml value is the fallback that ships.
        # They were the live values until 2026-08-12, hardcoded 5.0 here and in the yaml with a
        # comment claiming they mirrored ggv ax_max and ax_max_machines. That claim went stale on
        # 2026-08-11 (ggv ax_max 5.0 -> 7.0) and nothing noticed, which is why it is derived now.
        self.max_accel_mps2 = 5.0
        self.max_decel_mps2 = 5.0
        self._slew_warned = False
        self._speed_cmd_prev = None           # last PUBLISHED command, for the slew limit
        self._trailing_handoff = None         # ramp state while merging out of TRAILING
        self._trailing_entry = None           # ...and while merging INTO it (see calc_speed_command)
        # TYRE LIMIT on the steering command -- see _clip_to_tyre_limit. ggv_table is the
        # [v_mps, ay_max] table from config/<SIM|CAR>/veh_dyn_info/ggv.csv, delivered by the
        # manager (load_ggv_ay); without it the clip is inert and says so once. The three
        # parameters are overridden from controller.yaml and are live-togglable.
        # a_lat_margin is NOT 1.0, and that is the whole point -- see _clip_to_tyre_limit.
        self.a_lat_limit_enable = True
        self.a_lat_margin = 1.35              # [-] multiplies the ggv's ay_max
        self.a_lat_v_floor = 0.5              # [m/s] floor in the v^2 denominator
        self.ggv_table = None
        self._ggv_warned = False
        # FRICTION CIRCLE on the speed command -- see _ax_avail. Same delivery convention as the
        # tyre limit above: the manager reads the three tables (load_veh_dyn) and assigns them,
        # the two parameters come from controller.yaml and are live-togglable.
        self.a_comb_limit_enable = True
        self.a_comb_margin = 1.0              # [-] multiplies ay_max in the circle -- see _ax_avail
        # Fusion weights in calc_future_position. 1.0 = PURE KINEMATIC MODEL, 0.0 = pure IMU.
        # Both were hardcoded 1.0, i.e. the measured yaw rate reached the controller and was
        # then multiplied by zero. Overridden from controller.yaml, live-togglable.
        self.lambda_weight = 1.0              # slip angle: model vs IMU-derived
        self.gamma_weight = 1.0               # future heading: model vs measured yaw rate
        self.ggv_ax_table = None              # [v, ax_max] from ggv.csv
        self.ax_machines_table = None         # [v, ax_max_machines] from ax_max_machines.csv
        self.b_ax_table = None                # [v, b_ax_max_machines] from b_ax_max_machines.csv
        self.dyn_model_exp = None             # p, from racecar_f110.ini vel_calc_opts
        self._comb_warned = False

        # Parameters in the controller
        self.curr_steering_angle = 0
        self.idx_nearest_waypoint = None  # index of nearest waypoint to car
        self.track_length = None
        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0
        self.trailing_command = 2
        self.speed_command = None
        self.curvature_waypoints = 0
        self.current_steer_command = 0
        self.yaw_rate = 0

        self.logger_info = logger_info
        self.logger_warn = logger_warn

        self.speed_diff_thres = speed_diff_thres
        self.start_speed = start_speed
        self.start_curvature_factor = start_curvature_factor

        self.wheelbase = wheelbase

        self.start_mode = False
        self.future_lat_err = 0.0
        self.future_lat_e_norm = 0.0
        self.boost_mode = False

    def main_loop(self, state, position_in_map, waypoint_array_in_map, speed_now, opponent, position_in_map_frenet, acc_now, track_length):
        # Updating parameters from manager
        self.state = state
        self.position_in_map = position_in_map

        #-------------------------------Future Position-----------------------------
        self.future_position = np.zeros((1, 3))
        #-------------------------------Future Position-----------------------------

        self.waypoint_array_in_map = waypoint_array_in_map
        self.speed_now = speed_now
        self.opponent = opponent
        self.position_in_map_frenet = position_in_map_frenet
        self.acc_now = acc_now
        self.track_length = track_length

        ## PREPROCESS ##
        # speed vector
        yaw = self.position_in_map[0, 2]

        v = [np.cos(yaw)*self.speed_now, np.sin(yaw)*self.speed_now]

        #-------------------------------Future Position-----------------------------

        self.future_position = self.calc_future_position(self.future_constant)

        #-------------------------------Future Position-----------------------------

        self.idx_nearest_waypoint = self.nearest_waypoint(self.position_in_map[0, :2], self.waypoint_array_in_map[:, :2])

        # if all waypoints are equal set self.idx_nearest_waypoint to 0
        if np.isnan(self.idx_nearest_waypoint):
            self.idx_nearest_waypoint = 0

        if len(self.waypoint_array_in_map[self.idx_nearest_waypoint:]) > 2:
            # calculate curvature of global optimizer waypoints
            self.curvature_waypoints = np.mean(abs(self.waypoint_array_in_map[self.idx_nearest_waypoint+10:self.idx_nearest_waypoint+20, 5]))

        # calculate future lateral error and future lateral error norm

        self.future_lat_e_norm, self.future_lat_err = self.calc_future_lateral_error_norm()

        ### LONGITUDINAL CONTROL ###

        #-----------------------------------------Future-------------------------------------------
        self.speed_command = self.calc_speed_command(v, self.future_lat_e_norm)
        #-----------------------------------------Future-------------------------------------------

        self.speed_command = self.speed_adjust_heading(self.speed_command)

        # POSTPROCESS for acceleration/speed decision

        if self.speed_command is not None:
            speed = max(self.speed_command, 0)
            acceleration = 0
            jerk = 0

        else:
            speed = 0
            jerk = 0
            acceleration = 0
            self.logger_warn("[Controller] speed was none")

        ### LATERAL CONTROL ###

        steering_angle = None
        self.future_idx_nearest_waypoint = self.nearest_waypoint(self.future_position[0, :2], self.waypoint_array_in_map[:, :2])

        #-----------------------------------------Future-------------------------------------------
        L1_point, L1_distance = self.calc_future_L1_point(self.future_lat_err)
        #-----------------------------------------Future-------------------------------------------

        if L1_point.any() is not None:

            #-----------------------------------------Future-------------------------------------------
            steering_angle = self.calc_steering_angle_for_future(L1_point, L1_distance, yaw, self.future_lat_e_norm, v)
            #-----------------------------------------Future-------------------------------------------

            self.current_steer_command = steering_angle

        else:
            raise Exception("L1_point is None")

        speed = self.AEB_for_weird_local_wpnt(speed)
        speed = self._slew_limit_speed(speed)

        return speed, acceleration, jerk, steering_angle, L1_point, L1_distance, self.idx_nearest_waypoint, self.curvature_waypoints, self.future_position

    def _slew_limit_speed(self, speed):
        """Bound |d(speed_command)/dt| by the vehicle's own longitudinal limits.

        Nothing else in this loop does: calc_speed_command -> speed_adjust_heading ->
        AEB_for_weird_local_wpnt all hand their result straight to the mux. Measured on the real
        car (bag ot_speed_0731_2034, 55.7 s): the published command moved at p99 12.2 m/s^2 and up
        to 86.8 m/s^2, against a 5.0 m/s^2 ggv ax_max. Every discrete source feeds it -- the
        TRAILING handoff, a static-reopt line swap, the AEB clamp engaging and releasing (2.87 ->
        2.00 -> 3.24 in two cycles), any state change. Rather than patch each one, bound the
        output: the car cannot execute a step anyway, so a reference containing one only costs
        tracking error and drivetrain shock.

        This is a SAFETY NET, not the fix -- a step still means something upstream is wrong, and
        the handoff/anchor fixes remove the two known sources. Deliberately NOT applied to the
        AEB's own decision, only to its rate: a genuine garbage-path clamp still reaches 2.0 m/s,
        it just gets there at the braking limit instead of instantly.
        """
        if speed is None:
            return speed
        prev = self._speed_cmd_prev
        if prev is None:                       # first cycle / after a reset: adopt as-is
            self._speed_cmd_prev = speed
            return speed
        dt = 1.0 / max(self.loop_rate, 1.0)
        # FRICTION CIRCLE: the UP allowance is whatever longitudinal acceleration the tyre has
        # left after the corner has taken its share (_ax_avail). Outside the circle that is 0
        # and the command simply may not rise. The DOWN allowance is untouched on purpose --
        # this refuses acceleration, it does not command braking.
        accel, decel = self._slew_limits()
        if self.a_comb_limit_enable:
            avail = self._ax_avail()
            if avail is None:
                if not self._comb_warned:
                    self._comb_warned = True
                    self.logger_warn("[Controller] a_comb_limit_enable is set but the veh_dyn "
                                     "tables are incomplete -> the friction circle is INERT")
            else:
                accel = min(accel, avail)
        up, dn = accel * dt, decel * dt
        limited = min(max(speed, prev - dn), prev + up)
        self._speed_cmd_prev = limited
        return limited

    def _slew_limits(self):
        """(accel, decel) allowance on the PUBLISHED speed command [m/s^2], at the current speed.

        Read from config/<SIM|CAR>/veh_dyn_info -- the same pair the offline velocity profile
        bounds itself with:
            accel = min(ggv ax_max, ax_max_machines)      friction bound AND drive bound
            decel = min(ggv ax_max, b_ax_max_machines)    friction bound AND brake bound
        interpolated at speed_now, because ax_max_machines is the one table that is genuinely
        speed-dependent (9.5 at rest, 4.62 at 15 m/s) and a scalar would hide that.

        WHY NOT controller.yaml. Those two keys shipped at 5.0 with a comment stating they
        mirrored ggv ax_max and ax_max_machines -- true when written, false from 2026-08-11
        (ggv ax_max 5.0 -> 7.0) and more so from 2026-08-12 (b_ax 5.0 -> 10.0). A number that has
        to be hand-updated in step with a csv is a number that will be stale, and the cost of
        this one being stale is the slew limit clipping a plan the car could actually follow.
        Same reason and same shape as static_avoidance_node's _a_limits.

        THE ACCEL SIDE IS NOT THE BINDING ONE while a_comb_limit_enable is set: _ax_avail returns
        min(ax_machines, ax_max*(1-used^p)^(1/p)), which is <= this accel everywhere, so the
        caller's min() picks the friction circle. Deriving it still matters -- it stops the yaml
        scalar being a second, cruder cap underneath the circle, which is exactly what 5.0 was.
        The DECEL side has no circle by design (refusing acceleration is not commanding braking),
        so it is governed here alone.

        The yaml keys REMAIN as the fallback used when the tables are absent, warned once and
        naming what is in force. That makes `ros2 param set max_accel_mps2 ...` inert on a car
        whose config IS readable -- deliberate, and free: a re-grip restarts the stack anyway.
        Built without the tables (the offline gates stub this class) it answers from them too.
        """
        ax_t = getattr(self, "ggv_ax_table", None)
        axm_t = getattr(self, "ax_machines_table", None)
        bax_t = getattr(self, "b_ax_table", None)
        if ax_t is None or axm_t is None or bax_t is None:
            if not getattr(self, "_slew_warned", False):
                self._slew_warned = True
                warn = getattr(self, "logger_warn", None)
                if warn is not None:
                    warn(f"[Controller] veh_dyn tables unavailable -> the slew limit falls back "
                         f"to controller.yaml (accel {self.max_accel_mps2}, decel "
                         f"{self.max_decel_mps2} m/s^2). A change to ggv.csv, ax_max_machines.csv "
                         f"or b_ax_max_machines.csv will NOT reach the published speed command.")
            return float(self.max_accel_mps2), float(self.max_decel_mps2)
        v = float(getattr(self, "speed_now", 0.0) or 0.0)
        ax = self._interp(ax_t, v)
        return (float(min(ax, self._interp(axm_t, v))),
                float(min(ax, self._interp(bax_t, v))))

    def AEB_for_weird_local_wpnt(self, speed):
        nearest_local_wpnt = self.waypoint_array_in_map[self.idx_nearest_waypoint, :2]

        local_wpnt_dist = np.sqrt((self.position_in_map[0, 0] - nearest_local_wpnt[0])**2 + (self.position_in_map[0, 1] - nearest_local_wpnt[1])**2)

        # Choose the threshold from the PATH, not from the state name. An avoidance line is
        # legitimately offset from the car (the hump itself, plus the SM's adoption lag), so the
        # tight garbage-path threshold false-fires while following one -- and a sudden 2 m/s clamp
        # at speed is itself a spin risk. Keying that off state == "OVERTAKE" was the wrong
        # question: the state machine hands the SAME avoidance geometry to the controller while
        # TRAILING (holding the avoidance reference through a drop, and during the trailing
        # approach), and in those states the 0.5 m threshold fired against a path that was never
        # meant to sit under the car -- a 2.0 m/s clamp toggling on and off, which is a sawtooth in
        # the speed command. What actually justifies the looser bar is the geometry being offset,
        # so test that: max |d| over the local window.
        d_col = self.waypoint_array_in_map[:, 8]
        offline_d = float(np.max(np.abs(d_col))) if d_col.size else 0.0
        off_line = offline_d > self.AEB_offline_d_thres
        thres = self.AEB_thres_overtake if off_line else self.AEB_thres

        # LATCHED, with hysteresis and a minimum hold. A bare threshold makes every wobble of
        # local_wpnt_dist around the bar a fresh engage/release: measured on the real car at 20 Hz
        # that is the 2.87 -> 2.00 -> 3.24 sawtooth in this function's own docstring, and it fired
        # under OVERTAKE, TRAILING and GB_TRACK alike -- once at 4.60 m against a 0.9 m bar.
        # Splitting the bar in two (0.5 / 0.9) widened it; it did not add hysteresis.
        # time is counted in CYCLES: this loop is fixed-rate, and the controller has no clock
        min_hold = int(round(self.AEB_min_hold_s * max(self.loop_rate, 1.0)))
        if not self._aeb_engaged:
            if local_wpnt_dist >= thres:
                self._aeb_engaged = True
                self._aeb_cycles = 0
                self.logger_warn(
                    f"[Controller] AEB ENGAGED: nearest local wpnt {local_wpnt_dist:.2f} m away "
                    f"(state={self.state}, max|d|={offline_d:.2f}, thres={thres:.2f}) "
                    f"-> capping speed at 2.0 m/s",
                    throttle_duration_sec=1.0)
        else:
            self._aeb_cycles += 1
            if (local_wpnt_dist < thres - self.AEB_release_hyst_m
                    and self._aeb_cycles >= min_hold):
                self._aeb_engaged = False
                self.logger_warn(
                    f"[Controller] AEB released: nearest local wpnt {local_wpnt_dist:.2f} m "
                    f"(< {thres - self.AEB_release_hyst_m:.2f}) after {self._aeb_cycles} cycles",
                    throttle_duration_sec=1.0)
        # A CAP, not an assignment. `return 2.0` raised the command whenever the car was already
        # slower than the cap -- an emergency brake that accelerates.
        return min(speed, 2.0) if self._aeb_engaged else speed

    def calc_steering_angle_for_future(self, future_L1_point, L1_distance, yaw, furture_lat_e_norm, v):
        """
        The purpose of this function is to calculate the steering angle based on the L1 point, desired lateral acceleration and velocity

        Inputs:
            future_L1_point: future_L1_point in frenet coordinates at L1 distance in front of the car
            L1_distance: distance of the L1 point to the car
            yaw: yaw angle of the car
            furture_lat_e_norm: future normed lateral error
            v : future speed vector

        Returns:
            steering_angle: calculated steering angle


        """
        marks = MarkerArray()
        for i in range(1):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.type = mrk.SPHERE
            mrk.scale.x = 0.3
            mrk.scale.y = 0.3
            mrk.scale.z = 0.3
            mrk.color.a = 1.0
            mrk.color.b = 1.0

            mrk.id = i
            mrk.pose.position.x = self.future_position[0, 0]
            mrk.pose.position.y = self.future_position[0, 1]
            mrk.pose.orientation.w = 1.0
            marks.markers.append(mrk)

        if self.predict_pub is not None:
            self.predict_pub.publish(marks)

        if (self.state == "TRAILING") and (self.opponent is not None):
            speed_la_for_lu = self.speed_now
        else:
            adv_ts_st = self.speed_lookahead_for_steer
            la_position_steer = [self.future_position[0, 0] + v[0]*adv_ts_st, self.future_position[0, 1] + v[1]*adv_ts_st]
            idx_future_la_steer = self.nearest_waypoint(la_position_steer, self.waypoint_array_in_map[:, :2])
            speed_la_for_lu = self.waypoint_array_in_map[idx_future_la_steer, 2]

        speed_for_lu = self.speed_adjust_lat_err(speed_la_for_lu, furture_lat_e_norm)

        Future_L1_vector = np.array([future_L1_point[0] - self.future_position[0, 0], future_L1_point[1] - self.future_position[0, 1]])

        if np.linalg.norm(Future_L1_vector) == 0:
            self.logger_warn("[Controller] norm of L1 vector was 0, eta is set to 0")
            eta = 0
        else:
            eta = np.arcsin(np.dot([-np.sin(yaw), np.cos(yaw)], Future_L1_vector)/np.linalg.norm(Future_L1_vector))

        # Pure-Pursuit steering (ROS1 MAP/steering-lookup branch removed)
        steering_angle = np.arctan(2*self.wheelbase*np.sin(eta)/L1_distance)

        dt = 1.0 / self.loop_rate

        #-------------------------Steering Scaling-----------------------------

        # modifying steer based on heading

        steering_angle += self.compute_future_heading_correction(Future_L1_vector, yaw, dt, self.speed_now)

        # modifying steer based on acceleration
        #########################################
        steering_angle = self.acc_scaling(steering_angle)
        #########################################

        # modifying steer based on speed

        steering_angle = self.speed_steer_scaling(steering_angle, speed_for_lu)

        # modifying steer based on velocity

        steering_angle *= np.clip(1 + (self.speed_now/10), 1, self.steer_gain_for_speed)

        # modifying steer based on lateral error

        steering_angle = self.steer_scaling_for_lat_err(steering_angle, self.future_lat_err)

        #-------------------------Steering Scaling-----------------------------

        # TYRE LIMIT. Every modifier above is a tracking DEMAND; none of them knows what the
        # tyres can deliver. Measured on ~/ggv_0812_1645 (5 clean laps, offline replay through
        # this very class, full coverage): this loop commands a lateral acceleration of p90
        # 14.36 and max 22.37 m/s^2, against a car that never once achieved more than 11.71.
        # A saturated tyre has lost cornering force AND steering authority, so the unachievable
        # part of that demand does not merely fail to help, it slows the line recovery it was
        # asked for. The ceiling sits ABOVE the raceline's own worst demand of 5.70 and at the
        # p90 of what the car demonstrably delivers -- see _clip_to_tyre_limit for why equality
        # with 5.70 was a regression and why a "6.0-6.5 knee" was an artefact of a short window.
        # Placed BEFORE the rate limit and the +-0.53 saturation so both still act on a
        # physically meaningful command, and the assignment at the end of this function carries
        # the clipped value into curr_steering_angle -- which is the rate limiter's reference
        # next cycle, so skipping that would walk the reference away from what was published.
        # WHAT THIS DOES NOT PROVE: the replay is OPEN LOOP (recorded positions), so it shows
        # the command falling and cannot show the car returning to the line under the new
        # ceiling. That is a sim/real question. See analysis/replay_steering.py.
        if self.a_lat_limit_enable:
            steering_angle = self._clip_to_tyre_limit(steering_angle)

        # limit change of steering angle
        threshold = 0.4
        if abs(steering_angle - self.curr_steering_angle) > threshold:
            self.logger_info("steering angle clipped")
        steering_angle = np.clip(steering_angle, self.curr_steering_angle - threshold, self.curr_steering_angle + threshold)
        steering_angle = np.clip(steering_angle, -0.53, 0.53)

        self.curr_steering_angle = steering_angle

        return steering_angle

    def a_lat_now(self):
        """Lateral acceleration the car is using right now: v^2 * |kappa| of the PLAN.

        The raceline's curvature at the nearest waypoint, not the gyro. Both were measured on
        bag ggv_0812_1645 and the plan's kappa is the one to use: the gyro reports the
        curvature the car ACHIEVED, which in exactly the situation this limit exists for is
        LOWER than what the path is asking of the tyre (the car is understeering wide -- that
        is what the growing d IS), so a gyro-fed circle opens up precisely when it should be
        closing. The plan's kappa also has no noise and no 90-degree mounting convention.
        """
        try:
            kappa = abs(float(self.waypoint_array_in_map[self.idx_nearest_waypoint, 5]))
        except (TypeError, IndexError):
            return None
        return float(self.speed_now) ** 2 * kappa

    def _ax_avail(self):
        """Longitudinal acceleration still available at the current lateral usage [m/s^2].

            ax_avail = min( ax_machines(v),
                            ax_max(v) * (1 - (a_lat / (ay_max(v)*margin))^p)^(1/p) )

        p = dyn_model_exp from racecar_f110.ini (2.0 today = an ellipse). This is the SAME
        combined-acceleration model the offline velocity profile is solved with, read from the
        same three files, which is what makes "the plan says 4.51 here" and "you may not
        accelerate here" the same statement instead of two opinions.

        WHY THIS EXISTS. Measured on ifac_0807's right-hand slalom (s 2.8-10.0, real bag, laps
        2 and 3 alike): at the apex s=5.2 the raceline asks for kappa -0.27 at 4.51 m/s, the car
        is doing 5.13, and 5.13^2*0.27 = 7.1 m/s^2 of lateral demand against a ggv ay_max of
        5.7. The car is already OUTSIDE the friction circle -- and it goes on accelerating,
        4.90 -> 5.13 -> 5.51, because speed_lookahead 0.25 s at 4.85 m/s reads the profile
        1.21 m ahead, which is the corner EXIT. Every m/s^2 spent going faster there is lateral
        force the tyre no longer has, so d climbs monotonically (-0.04 -> +0.41) and the car
        enters the following straight wide and slow (-0.76 m/s under target at s=8.2).

        a_lat past the circle clamps the used fraction at 1, so ax_avail is 0 and the speed
        command may not RISE. It is deliberately not allowed to force a decrease: braking is a
        different intervention that would move every other section of the lap, and the measured
        fault here is the acceleration.

        Returns None if any of the three inputs is missing; the caller reports that once.
        """
        if (self.ggv_table is None or self.ggv_ax_table is None
                or self.ax_machines_table is None or self.dyn_model_exp is None):
            return None
        a_lat = self.a_lat_now()
        if a_lat is None:
            return None
        v = float(self.speed_now)
        ay = self._interp(self.ggv_table, v) * float(self.a_comb_margin)
        ax = self._interp(self.ggv_ax_table, v)
        axm = self._interp(self.ax_machines_table, v)
        if ay <= 0.0:
            return 0.0
        used = min(a_lat / ay, 1.0)
        p = float(self.dyn_model_exp)
        return float(min(axm, ax * (1.0 - used ** p) ** (1.0 / p)))

    @staticmethod
    def _interp(table, v):
        t = np.atleast_2d(table)
        return float(np.interp(v, t[:, 0], t[:, 1]))

    def a_lat_max_now(self, speed):
        """The controller's lateral-acceleration ceiling [m/s^2] at this speed.

        ggv ay_max INTERPOLATED at the current speed, times a_lat_margin. Interpolated and not
        reduced to a scalar: ggv.csv is a (v_mps, ax_max, ay_max) table, and that it is flat at
        5.7 over the whole range today is this table's value, not a property of the format.

        Returns None when no table was delivered, which the caller reports rather than guessing.
        """
        if self.ggv_table is None:
            return None
        t = np.atleast_2d(self.ggv_table)
        return float(np.interp(speed, t[:, 0], t[:, 1])) * self.a_lat_margin

    def _clip_to_tyre_limit(self, steering_angle):
        """Bound the steering command by the lateral acceleration the tyres can actually make.

        Bicycle model: a_lat = v^2 * tan(delta) / L, so the steering angle that asks for exactly
        the ceiling is delta_max = atan(L * a_lat_ctrl(v) / v^2).

        WHY THE CEILING IS NOT THE ggv's 5.7. That number is the PLANNER's limit, and the
        raceline spends all of it: the published line's own worst demand is vx^2*|kappa| = 5.70.
        Clipping the total to 5.70 therefore leaves EXACTLY ZERO budget for returning to the
        line at a corner apex, and that is not a theory -- it shipped once, at margin 1.0, and
        came back as "path tracking is far too sluggish" on the real car. The prediction that
        goes with the diagnosis is that the sluggishness is CONCENTRATED IN CORNERS, because on
        a straight the line asks for ~0 and the whole budget is free.
        So the controller's ceiling must sit ABOVE the planner's, and the gap IS the recovery
        budget.

        WHERE THE ROOF IS. The first correction to this reasoning put the roof at a "tyre knee"
        of 6.0-6.5, from an achieved/demanded ratio that held at 1.00 up to a demand of 6 and
        collapsed to 0.53 above it. That was measured over 8.2 s of a 50.4 s bag -- the replay
        was silently dying at the first lap boundary. Over the WHOLE bag there is NO CLIFF: the
        ratio falls but the achieved value keeps climbing (demand 6-8 -> achieved 5.55, 8-10 ->
        5.53, 10-14 -> 6.26, 14-25 -> 7.39), and the car demonstrably made p50 4.96, p90 7.62,
        max 11.71 m/s^2 of real lateral acceleration. A ceiling below that band does not stop
        the car asking for the impossible, it stops the car doing what it can: 5.70 removes
        capability the record shows for 41.4% of the run, 6.27 for 29.9%.
        So the roof is what the tyre DEMONSTRABLY DELIVERS, and the ceiling is anchored at the
        p90 of that: 7.62, i.e. a_lat_margin 1.35 -> 7.70. It gives up 9.5% of the run, leaves
        the MEDIAN command untouched (demand p50 6.69, same as with no clip at all) and still
        removes the unachievable tail (demand p90 14.36 -> 7.70, max 22.37 -> 7.70, against an
        achieved max of 11.71 the car never once exceeded).

        v is floored at a_lat_v_floor: as v -> 0, delta_max -> pi/2 and the expression stops
        meaning anything. The +-0.53 saturation downstream does catch it, but by accident.
        """
        a_lat_max = self.a_lat_max_now(self.speed_now)
        if a_lat_max is None or a_lat_max <= 0.0:
            if not self._ggv_warned:
                self._ggv_warned = True
                self.logger_warn("[Controller] a_lat_limit_enable is set but no usable ggv table "
                                 "was delivered -> the steering tyre limit is INERT")
            return steering_angle
        v = max(float(self.speed_now), float(self.a_lat_v_floor))
        delta_max = float(np.arctan(self.wheelbase * a_lat_max / (v * v)))
        return float(np.clip(steering_angle, -delta_max, delta_max))

    def calc_future_L1_point(self, future_lateral_error):

        # calculate future L1 guidance

        if self.speed_now < 2.0:

            speed = np.clip(self.speed_command, self.speed_now - 1, self.speed_now + 1)
            speed_scaler = self.m_l1 * speed

        else:

            speed_scaler = self.m_l1 * self.speed_now

        if self.state == "START":
            curvature_scaler = self.start_curvature_factor*self.curvature_waypoints
        else:
            curvature_scaler = self.curvature_factor*self.curvature_waypoints*self.speed_now*self.speed_now

        L1_distance = (speed_scaler - curvature_scaler) + self.q_l1

        # clip lower bound to avoid ultraswerve when far away from mincurv.
        # The sqrt(2)*lat_err inflation is CAPPED: after a global-line swap that lands with the
        # car off the new line (deadlock-breaker commit, SM source transients) lat_err can reach
        # ~1 m, and an uncapped lower bound stretched the lookahead through corner zones whose
        # normal L1 is ~0.7 m — on a 1.4 m-wide track that is a corner-cut into the wall. The cap
        # bounds the convergence lookahead; below it the anti-swerve behaviour is unchanged.
        lower_bound = max(self.t_clip_min,
                          min(np.sqrt(2)*future_lateral_error, self.l1_lat_err_cap))

        L1_distance = np.clip(L1_distance, lower_bound, self.t_clip_max)

        future_L1_point = self.waypoint_at_distance_before_car(L1_distance, self.waypoint_array_in_map[:, :2], self.future_idx_nearest_waypoint)

        return future_L1_point, L1_distance

    def calc_speed_command(self, v, lat_e_norm):
        """
        The purpose of this function is to isolate the speed calculation from the main control_loop

        Inputs:
            v: speed vector
            lat_e_norm: normed lateral error
            curvature_waypoints: -
        Returns:
            speed_command: calculated and adjusted speed, which can be sent to mux
        """

        # lookahead for speed (speed delay incorporation by propagating position)
        adv_ts_sp = self.speed_lookahead
        offset = 2
        la_position = [self.position_in_map[0, 0] + v[0]*adv_ts_sp, self.position_in_map[0, 1] + v[1]*adv_ts_sp]
        idx_la_position = self.nearest_waypoint(la_position, self.waypoint_array_in_map[:, :2])
        idx_la_position = np.clip(idx_la_position + offset, 0, len(self.waypoint_array_in_map) - 1)
        global_speed = self.waypoint_array_in_map[idx_la_position, 2]
        cur_speed = self.speed_now

        if cur_speed < 0:
            cur_speed = 0

        if (self.state == "START"
            and self.boost_mode
            and self.waypoint_array_in_map[0, 7] > 0):
            if (global_speed-cur_speed) > 0:
                global_speed = self.start_speed
            elif self.cur_state_speed - cur_speed > 0:
                self.cur_state_speed -= self.speed_diff_thres * (self.cur_state_speed - cur_speed)
                global_speed = self.cur_state_speed
            else:
                self.boost_mode = False
        else:
            self.boost_mode = False

        if ((self.state == "TRAILING") and (self.opponent is not None)):  # Trailing controller
            speed_command = self.trailing_controller(global_speed)
            # ENTRY ramp, mirroring the exit ramp below. Leaving TRAILING has been ramped since the
            # +1.7 m/s handoff steps were measured; ENTERING it assigned the PID output outright.
            # Against a stationary box that output is not a small correction: d_value = v_diff *
            # trailing_d_gain is the ego's own speed with the shipped gain of 1.0, so the command
            # collapses from the path speed to ~0 in a single 20 ms cycle -- the same discontinuity
            # in the other direction, and the one that arrives with a box in front of the car.
            # Bounded by the same longitudinal limit the trajectory is planned with (decel here).
            if self._trailing_entry is None:
                self._trailing_entry = self._speed_cmd_prev
            if self._trailing_entry is not None:
                step = self.max_decel_mps2 / max(self.loop_rate, 1.0)
                self._trailing_entry = max(speed_command, self._trailing_entry - step)
                if self._trailing_entry <= speed_command + 1e-3:
                    self._trailing_entry = None         # merged onto the PID
                else:
                    speed_command = self._trailing_entry
            self._trailing_handoff = speed_command      # where the handoff below must start from
        else:
            # HANDOFF out of TRAILING. The gap PID has been holding the car down at the opponent's
            # pace while the path speed ahead is already much higher, so assigning global_speed
            # outright steps the command by the whole speed deficit in one 20 ms cycle. Measured on
            # the real car (bag ot_speed_0731_2034): +1.59/+1.64/+1.67/+1.70 m/s at all four
            # TRAILING->OVERTAKE transitions, i.e. ~80 m/s^2 of demand against a 5.0 ggv ax_max --
            # and the REFERENCE was continuous there (vx[10] 4.94 -> 4.94), so the step was created
            # here, not by the planner. Ramp out at the same longitudinal limit the trajectory is
            # planned with instead; the car still accelerates to pass, it just does it feasibly.
            if self._trailing_handoff is not None:
                step = self.max_accel_mps2 / max(self.loop_rate, 1.0)
                self._trailing_handoff = min(global_speed, self._trailing_handoff + step)
                if self._trailing_handoff >= global_speed - 1e-3:
                    self._trailing_handoff = None       # merged, hand control back to the path
                    speed_command = global_speed
                else:
                    speed_command = self._trailing_handoff
            else:
                speed_command = global_speed
            self.trailing_speed = global_speed
            self.i_gap = 0
            self._trailing_entry = None                 # re-armed for the next entry

        speed_command = self.speed_adjust_lat_err(speed_command, lat_e_norm)

        return speed_command

    def trailing_controller(self, global_speed):
        """
        Adjust the speed of the ego car to trail the opponent at a fixed distance
        Inputs:
            speed_command: velocity of global raceline
            self.opponent: frenet s position and vs velocity of opponent
            self.position_in_map_frenet: frenet s position and vs veloctz of ego car
        Returns:
            trailing_command: reference velocity for trailing
        """

        self.gap = (self.opponent[0] - self.position_in_map_frenet[0]) % self.track_length  # gap to opponent
        self.gap_actual = self.gap
        self.gap_should = self.trailing_vel_gain * self.speed_now + self.trailing_gap

        self.gap_error = self.gap_should - self.gap_actual
        self.v_diff = self.position_in_map_frenet[2] - self.opponent[2]
        self.i_gap = np.clip(self.i_gap + self.gap_error/self.loop_rate, -10, 10)

        p_value = self.gap_error * self.trailing_p_gain
        d_value = self.v_diff * self.trailing_d_gain
        i_value = self.i_gap * self.trailing_i_gain

        self.trailing_command = np.clip(self.opponent[2] - p_value - i_value - d_value, 0, global_speed)
        if not self.opponent[4] and self.gap_actual > self.gap_should:
            self.trailing_command = max(self.blind_trailing_speed, self.trailing_command)

        return self.trailing_command

    def distance(self, point1, point2):
        return np.linalg.norm(point2 - point1)

    def acc_scaling(self, steer):
        """
        Steer scaling based on acceleration
        increase steer when accelerating
        decrease steer when decelerating

        Returns:
            steer: scaled steering angle based on acceleration
        """

        if self.start_mode:
            return steer

        if np.mean(self.acc_now) >= 1:
            steer *= self.acc_scaler_for_steer
        elif np.mean(self.acc_now) <= -3.0:
            if self.state == "START":
                steer *= 0.7
            else:
                steer *= self.dec_scaler_for_steer

        return steer

    def speed_steer_scaling(self, steer, speed):
        """
        Steer scaling based on speed
        decrease steer when driving fast

        Returns:
            steer: scaled steering angle based on speed
        """
        speed_diff = max(0.1, self.end_scale_speed-self.start_scale_speed)  # to prevent division by zero
        factor = 1 - np.clip((speed - self.start_scale_speed)/(speed_diff), 0.0, 1.0) * self.downscale_factor
        steer *= factor
        return steer

    def steer_scaling_for_lat_err(self, steer, lateral_error):

        if self.start_mode:
            return steer

        factor = np.exp(np.log(2)*lateral_error)

        steer *= factor
        return steer

    def calc_future_lateral_error_norm(self):
        """
        Calculates future lateral error

        Returns:
           future lat_e_norm: normalization of the future lateral error
           future lateral_error: future distance from car's position to nearest waypoint
        """
        future_position = self.future_position[0, :2]
        idx_future_local_wpnts = self.nearest_waypoint(future_position, self.waypoint_array_in_map[:, :2])
        future_local_wpnts_d = abs(self.waypoint_array_in_map[idx_future_local_wpnts, 8])
        future_potision_s, future_position_d = self.converter.get_frenet([self.future_position[0, 0]], [self.future_position[0, 1]])
        future_position_d = abs(future_position_d[0])
        future_lat_err = future_position_d - future_local_wpnts_d

        max_lat_e = 1
        min_lat_e = 0.
        lat_e_clip = np.clip(future_lat_err, a_min=min_lat_e, a_max=max_lat_e)
        lat_e_norm = ((lat_e_clip - min_lat_e) / (max_lat_e - min_lat_e))
        return lat_e_norm, future_lat_err

    def speed_adjust_lat_err(self, global_speed, lat_e_norm):
        """
        Reduce speed from the global_speed based on the lateral error
        and curvature of the track. lat_e_coeff scales the speed reduction:
        lat_e_coeff = 0: no account for lateral error
        lat_e_coaff = 1: maximum accounting

        Returns:
            global_speed: the speed we want to follow
        """
        # scaling down global speed with lateral error and curvature
        lat_e_coeff = self.lat_err_coeff  # must be in [0, 1]
        lat_e_norm *= self.speed_factor_for_lat_err
        curv = np.clip(2*(np.mean(self.curvature_waypoints)/0.8) - 2, a_min=0.0, a_max=1.0)  # 0.8 ca. max curvature mean
        curv *= self.speed_factor_for_curvature
        global_speed *= (1.0 - lat_e_coeff + lat_e_coeff*np.exp(-lat_e_norm*curv))
        return global_speed

    def speed_adjust_heading(self, speed_command):
        """
        Reduce speed from the global_speed based on the heading error.
        If the difference between the map heading and the actual heading
        is larger than 10 degrees, the speed gets scaled down linearly up to 0.5x

        Returns:
            global_speed: the speed we want to follow
        """

        heading = self.position_in_map[0, 2]
        map_heading = self.waypoint_array_in_map[self.idx_nearest_waypoint, 6]
        if abs(heading - map_heading) > np.pi:
            heading_error = 2*np.pi - abs(heading - map_heading)
        else:
            heading_error = abs(heading - map_heading)

        if heading_error < self.heading_error_thres*np.pi/180:  # 10 degrees error is okay
            return speed_command
        elif heading_error < np.pi/2:
            scaler = 1 - 0.5 * heading_error/(np.pi/2)
        else:
            scaler = 0.5
        return speed_command * scaler

    def compute_future_heading_correction(self, L1_vector, yaw, dt, speed,
                               alpha=0.1, v_threshold=15.0,
                               use_pid=True, use_filter=True):

        target_heading = np.arctan2(L1_vector[1], L1_vector[0])
        heading_error = target_heading - yaw
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        if use_filter:
            if not hasattr(self, 'filtered_heading_error'):
                self.filtered_heading_error = heading_error
            self.filtered_heading_error = alpha * heading_error + (1 - alpha) * self.filtered_heading_error
            heading_error = self.filtered_heading_error

        if speed < v_threshold:
            dynamic_gain = self.KP * (speed / v_threshold)
        else:
            dynamic_gain = self.KP

        if self.state == "OVERTAKE":
            dynamic_gain *= 0.65

        if not hasattr(self, 'heading_error_integral'):
            self.heading_error_integral = 0.0
        if not hasattr(self, 'prev_heading_error'):
            self.prev_heading_error = heading_error

        if use_pid:
            self.heading_error_integral += heading_error * dt
            derivative = (heading_error - self.prev_heading_error) / dt if dt > 0 else 0.0
            self.prev_heading_error = heading_error

            correction = dynamic_gain * heading_error + self.KI * self.heading_error_integral + self.KD * derivative
        else:
            correction = dynamic_gain * heading_error

        return correction

    def calc_future_position(self, T):
        """
        Predicts the future vehicle state (position and heading) T seconds ahead
        based on the current vehicle state and updates self.position_in_map[0].

        Inputs:
            T: Prediction time (seconds), e.g., 0.25

        Assumes the following variables exist in self:
            self.position_in_map : 2D array with the first row containing [x, y, psi]
            self.speed_now       : Current vehicle speed (v)
            self.current_steer_command : Current steering input (delta)
            self.yaw_rate        : Current yaw rate from the IMU (rad/s)
            self.wheelbase       : Vehicle wheelbase (distance between front and rear axles)
        """

        x_current = self.position_in_map[0, 0]

        # Extract current state
        x_current = self.position_in_map[0, 0]
        y_current = self.position_in_map[0, 1]
        psi_current = self.position_in_map[0, 2]
        v = self.speed_now
        delta = self.current_steer_command  # Steering input

        # Vehicle geometry parameters.
        # Here, L_f and L_r are assumed to be 52% and 48% of the total wheelbase respectively.
        L_total = self.wheelbase
        L_f = 0.52 * L_total
        L_r = 0.48 * L_total

        # 1. Compute geometric slip angle (basic model)
        beta_model = np.arctan((L_r / (L_f + L_r)) * np.tan(delta))

        # 2. Estimate slip angle indirectly using IMU yaw rate data
        # THIS IS NOT A SLIP ANGLE. Substituting the bicycle relation yaw_rate = v*tan(delta)/L
        # gives arcsin(tan(delta)) ~= delta -- the STEERING angle. The slip angle is
        # arctan((L_r/L)*tan(delta)), which is what beta_model above computes, so this term is
        # too large by L/L_r = 1/0.48 = 2.08x. Measured consequence (replay_steering.py --gamma,
        # full bag): mixing it in makes the future-position prediction monotonically WORSE --
        # p50 error 0.0455 m at lambda 1.0, 0.0484 at 0.7, 0.0502 at 0.5, 0.0555 at 0.0 -- which
        # is why lambda_weight stays at 1.0. Correcting the formula is a model change that needs
        # its own sweep and is deliberately NOT bundled with the IMU sign fix.
        if abs(v) > 2.0:
            # If speed is sufficient, estimate slip angle from IMU yaw rate
            beta_imu = np.arcsin(np.clip(((L_f + L_r) * self.yaw_rate / v), -1.0, 1.0))
        else:
            beta_imu = beta_model  # Maintain basic model when speed is very low

        # 3. Fuse the geometric and IMU-based slip angles using weighted average.
        # WEIGHT 1.0 MEANS PURE KINEMATIC MODEL -- the IMU term is multiplied by zero. Both this
        # and gamma_weight below shipped hardcoded at 1.0, which is why the controller had no
        # dynamic feedback at all: it could not tell that a demand of 20 m/s^2 was producing 7.
        # They are instance attributes so controller.yaml can set them; see the note there for
        # why lambda stays at 1.0 while gamma does not.
        lambda_weight = self.lambda_weight
        beta_fused = lambda_weight * beta_model + (1 - lambda_weight) * beta_imu

        # 4. Predict future position using the fused slip angle
        future_x = x_current + v * np.cos(psi_current + beta_fused) * T
        future_y = y_current + v * np.sin(psi_current + beta_fused) * T

        # 5. Predict future heading:
        # Option A: Model-based prediction
        future_psi_model = psi_current + (v / (L_f + L_r)) * np.sin(beta_fused) * T
        # Option B: IMU-based prediction
        future_psi_imu = psi_current + self.yaw_rate * T
        # Fuse the two heading predictions using a weighted average
        gamma_weight = self.gamma_weight
        future_psi = gamma_weight * future_psi_model + (1 - gamma_weight) * future_psi_imu
        # Normalize heading to the range [-pi, pi]
        future_psi = np.arctan2(np.sin(future_psi), np.cos(future_psi))

        # Update the global state: overwrite self.position_in_map[0] with the future state.

        future_position = np.zeros((1, 3))

        future_position[0, 0] = future_x
        future_position[0, 1] = future_y
        future_position[0, 2] = future_psi

        return future_position

    def nearest_waypoint(self, position, waypoints):
        """
        Calculates index of nearest waypoint to the car

        Returns:
            index of nearest waypoint to the car
        """
        position_array = np.array([position]*len(waypoints))
        distances_to_position = np.linalg.norm(abs(position_array - waypoints), axis=1)
        return np.argmin(distances_to_position)

    def waypoint_at_distance_before_car(self, distance, waypoints, idx_waypoint_behind_car):
        """
        Calculates the waypoint at a certain frenet distance in front of the car

        Returns:
            waypoint as numpy array at a ceratin distance in front of the car
        """

        if distance is None:
            distance = self.t_clip_min
        d_distance = distance

        # Extract only waypoints ahead of current index
        waypoints_ahead = waypoints[idx_waypoint_behind_car:]

        # Compute segment-wise distances between waypoints
        deltas = np.diff(waypoints_ahead, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)

        # Compute cumulative distances
        cum_lengths = np.cumsum(seg_lengths)

        # Find the first index where cumulative distance exceeds lookahead
        idx_offset = min(np.searchsorted(cum_lengths, d_distance), len(waypoints_ahead) - 1)

        return waypoints_ahead[idx_offset]
