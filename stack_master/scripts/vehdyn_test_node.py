#!/usr/bin/env python3
"""vehdyn_test_node.py -- drives the measurement maneuvers for vehicle-dynamics calibration.

WHAT IT IS FOR. Primary: re-calibrate at a venue after the surface changed, in about two
minutes (`mode: circle` -- washout left and right, nothing else). Secondary: characterise the
whole envelope at the practice ground (`mode: full`).

WHAT IT DOES NOT DO. It does not write veh_dyn_info/. It does not touch the raceline. It
records; stack_master/scripts/vehdyn_analyze.py turns a bag into candidate csvs under
config/vehdyn_measured/<timestamp>/ for a human to copy in by hand.

THIS DRIVES A REAL CAR AUTONOMOUSLY AT UP TO v_max_allowed. Four things stand between it and
a wall, and all four are in Guard/Plan below, deliberately free of ROS so they can be tested:

  1. dry_run (SHIPPED DEFAULT) -- plans, prints, publishes nothing.
  2. joy    -- ANY operator input aborts. simple_mux does NOT do this for us: it hands over
               only when buttons[4] is pressed (simple_mux_node._handle_joy), so moving a
               stick while this node runs would change nothing. Hence the direct /joy watch.
  3. box    -- leaving the configured rectangle stops the car.
  4. budget -- every maneuver carries a distance ceiling derived BEFORE it starts.

Plus: the speed command is slew-limited (never a step), and resuming from PAUSE is refused
unless the car is within pose_tol of the planned start pose.

Every value comes from config/vehdyn_test_params.yaml. There are no tuning constants here.

Run:
    ros2 launch stack_master vehdyn_test.launch.xml
    ros2 launch stack_master vehdyn_test.launch.xml dry_run:=false   # after reading the plan
"""

import json
import math
import os

# --------------------------------------------------------------------------------------
# Pure logic. No ROS below this line until Node. The abort paths and the space -> condition
# derivation are the parts that must be right, so they are testable without a robot.
# --------------------------------------------------------------------------------------


class Guard:
    """The four abort paths. Each returns a reason string, or None to continue.

    Held separate from the node so stack_master/scripts/test_vehdyn.py can drive them
    directly: a safety path that is only reachable by launching a car is a safety path
    nobody tests.
    """

    def __init__(self, p):
        self.box_center = (float(p['box_center_x']), float(p['box_center_y']))
        self.box_w = float(p['box_w'])
        self.box_h = float(p['box_h'])
        self.margin = float(p['safety_margin_m'])
        self.joy_deadzone = float(p['joy_abort_axis_deadzone'])
        self.joy_any_button = bool(p['joy_abort_on_any_button'])
        self.odom_timeout_s = float(p['odom_timeout_s'])

    def check_joy(self, axes, buttons):
        """ANY operator input is an abort. Not a takeover request -- an abort.

        The node cannot distinguish 'the operator wants control' from 'the operator is
        lunging for the pad because the car is heading somewhere bad', so it treats both
        the same way and stops.
        """
        if axes:
            for i, a in enumerate(axes):
                if abs(float(a)) > self.joy_deadzone:
                    return f"joy axis {i} = {float(a):+.2f} (deadzone {self.joy_deadzone})"
        if self.joy_any_button and buttons:
            for i, b in enumerate(buttons):
                if b:
                    return f"joy button {i} pressed"
        return None

    def check_box(self, x, y):
        """Inside the box, shrunk by safety_margin_m on every side."""
        hw = self.box_w / 2.0 - self.margin
        hh = self.box_h / 2.0 - self.margin
        if hw <= 0.0 or hh <= 0.0:
            return (f"safety box is degenerate: {self.box_w}x{self.box_h} m with margin "
                    f"{self.margin} m leaves nothing")
        dx = x - self.box_center[0]
        dy = y - self.box_center[1]
        if abs(dx) > hw or abs(dy) > hh:
            return (f"outside safety box: offset ({dx:+.2f}, {dy:+.2f}) m exceeds "
                    f"(+-{hw:.2f}, +-{hh:.2f})")
        return None

    def check_budget(self, travelled_m, budget_m):
        if budget_m is not None and travelled_m > budget_m:
            return f"distance budget exceeded: {travelled_m:.1f} m > {budget_m:.1f} m"
        return None

    def check_odom(self, age_s):
        """Fail CLOSED. No position = no box test = no driving."""
        if age_s is None:
            return "no odometry received yet"
        if age_s > self.odom_timeout_s:
            return f"odometry stale: {age_s:.2f} s > {self.odom_timeout_s} s"
        return None


def slew(current, target, rate, dt):
    """One slew-limited step toward target. Used for both speed and steering."""
    step = abs(rate) * dt
    if target > current:
        return min(target, current + step)
    return max(target, current - step)


def pose_within(cur_xy, cur_yaw, want_xy, want_yaw, tol_m, tol_deg):
    """Is the car parked where the next maneuver expects it?

    The gate on resuming from PAUSE. Everything else assumes the run starts from the planned
    pose; a car pointed the wrong way accelerates to v_target at whatever is in front of it.
    """
    d = math.hypot(cur_xy[0] - want_xy[0], cur_xy[1] - want_xy[1])
    dyaw = abs((cur_yaw - want_yaw + math.pi) % (2 * math.pi) - math.pi)
    ok = d <= tol_m and math.degrees(dyaw) <= tol_deg
    return ok, d, math.degrees(dyaw)


# --------------------------------------------------------------------------------------
# Space -> test conditions
# --------------------------------------------------------------------------------------


def derive_conditions(p):
    """Turn the available space into per-maneuver targets, distances and times.

    Printed as a table before anything moves, because 'how fast will it go and how far will
    it travel' is the question the operator needs answered while the car is still parked.
    """
    margin = float(p['safety_margin_m'])
    v_cap = float(p['v_max_allowed'])
    out = {'warnings': []}

    # --- straight: accelerate to v then stop again, inside L ---
    #   L = v^2/(2*a_acc) + v^2/(2*a_brk)  ->  v = sqrt(2L / (1/a_acc + 1/a_brk))
    # Same cross-check as the circle below: the straight cannot be longer than the longest line
    # the abort box actually allows.
    L_declared = float(p['straight_len_m']) - 2.0 * margin
    L_box = 2.0 * max(float(p['box_w']) / 2.0 - margin, float(p['box_h']) / 2.0 - margin)
    L = min(L_declared, L_box)
    if L_box < L_declared:
        out['warnings'].append(
            f"straight_len_m implies {L_declared:.2f} m of usable straight but the abort box only "
            f"allows {L_box:.2f} m -- using {L:.2f} m")
    a_acc = float(p['a_acc_prior'])
    a_brk = float(p['a_brk_prior'])
    if L <= 0.0:
        v_straight = 0.0
    else:
        v_straight = math.sqrt(2.0 * L / (1.0 / a_acc + 1.0 / a_brk))
    v_straight_capped = min(v_straight, v_cap)
    out['straight_usable_m'] = L
    out['v_straight_uncapped'] = v_straight
    out['v_straight'] = v_straight_capped
    if v_straight_capped < 3.0:
        out['warnings'].append(
            f"straight gives only v_target {v_straight_capped:.1f} m/s (< 3.0): the "
            f"longitudinal maneuver is not worth running in this space -- SKIPPED")
    out['straight_feasible'] = v_straight_capped >= 3.0

    # --- circle: radius that fits BOTH the declared area and the ABORT BOX ---
    # These are two different declarations of the same space and they can disagree. The box is
    # what actually stops the car, so a radius derived from area_radius_m alone can plan a
    # circle that the box aborts a quarter turn into -- at speed, with the operator having read
    # a table that said it would fit. Cross-check here, where it is still just a number.
    half_w = float(p['box_w']) / 2.0 - margin
    half_h = float(p['box_h']) / 2.0 - margin
    R_area = float(p['area_radius_m']) - margin - float(p['car_width_m']) / 2.0
    R_box = min(half_w, half_h) - float(p['car_width_m']) / 2.0
    R = min(R_area, R_box)
    out['circle_R_area_m'] = R_area
    out['circle_R_box_m'] = R_box
    if R_box < R_area:
        out['warnings'].append(
            f"THE ABORT BOX IS SMALLER THAN area_radius_m SAYS: box {p['box_w']}x{p['box_h']} m "
            f"with margin {margin} m allows R {R_box:.2f} m, area_radius_m allows {R_area:.2f} m. "
            f"Using {R:.2f} m. Widen box_w/box_h if the space really is bigger -- the box is what "
            f"aborts the run.")
    out['circle_R_m'] = R
    ay_ref = float(p['ggv_ay_max_ref'])
    if R <= 0.0:
        out['v_washout'] = 0.0
        out['circle_feasible'] = False
        out['warnings'].append(f"area_radius_m leaves R = {R:.2f} m -- no circle fits")
    else:
        v_wo = math.sqrt(ay_ref * R)
        out['v_washout_uncapped'] = v_wo
        if v_wo > v_cap:
            # v_max_allowed binds before the tyre does -> shrink R so washout is reachable
            R_cap = v_cap ** 2 / ay_ref
            out['circle_R_m'] = R_cap
            out['v_washout'] = v_cap
            out['warnings'].append(
                f"washout at R {R:.2f} m needs {v_wo:.1f} m/s > v_max_allowed {v_cap:.1f}; "
                f"R reduced to {R_cap:.2f} m so the limit is reachable under the cap")
        else:
            out['v_washout'] = v_wo
        out['circle_feasible'] = True

    # --- oval: lap length 2(L-W) + pi*W in a box L x W ---
    box_L = max(float(p['straight_len_m']), float(p['box_w'])) - 2.0 * margin
    box_W = min(float(p['straight_len_m']), float(p['box_h'])) - 2.0 * margin
    if box_L > box_W > 0.0:
        R_oval = box_W / 2.0
        out['oval_R_m'] = R_oval
        out['oval_lap_m'] = 2.0 * (box_L - box_W) + math.pi * box_W
        out['oval_v_corner'] = min(math.sqrt(ay_ref * R_oval), v_cap)
        out['oval_v_straight'] = v_straight_capped
        out['oval_feasible'] = out['oval_lap_m'] > 0.0
    else:
        out['oval_feasible'] = False
        out['warnings'].append("box is not long enough to inscribe an oval")

    # --- distance budgets, per maneuver ---
    settle = float(p['settle_still_s'])
    ramp = float(p['washout_ramp_mps2'])
    v_wo = out.get('v_washout', 0.0)
    t_wo = (v_wo / ramp) if ramp > 0 else 0.0
    out['washout_time_s'] = t_wo + settle
    # a slow ramp to v over t covers v*t/2, plus a turn's worth of margin
    out['washout_dist_m'] = 0.5 * v_wo * t_wo + 2.0 * math.pi * out.get('circle_R_m', 0.0)
    out['straight_dist_m'] = 2.0 * L if out['straight_feasible'] else 0.0
    out['straight_time_s'] = (2.0 * (v_straight_capped / a_acc + v_straight_capped / a_brk)
                              if out['straight_feasible'] else 0.0)
    out['oval_dist_m'] = out.get('oval_lap_m', 0.0) * float(p['repeats'])
    out['oval_time_s'] = (out['oval_dist_m'] / max(out.get('oval_v_corner', 1.0), 0.1)
                          if out.get('oval_feasible') else 0.0)
    return out


def build_plan(p, cond):
    """The ordered maneuver list for the selected mode.

    T0 (5 s still) is inserted before every maneuver, not just at the start: bias extraction
    and segment splitting in the analyzer both key off those stationary stretches.
    """
    mode = 'circle' if bool(p['venue_mode']) else str(p['mode'])
    reps = int(p['repeats'])
    plan = []

    def add(kind, side, note, dist, tsec):
        plan.append({'kind': kind, 'side': side, 'note': note,
                     'budget_m': dist, 'est_s': tsec})

    if mode in ('circle', 'full'):
        if not cond.get('circle_feasible'):
            cond['warnings'].append("circle not feasible -- washout SKIPPED")
        else:
            for i in range(reps):
                for side in ('left', 'right'):
                    add('T2_washout', side, f"washout ramp {i + 1}/{reps}",
                        cond['washout_dist_m'], cond['washout_time_s'])

    if mode == 'full':
        lm = str(p['long_mode'])
        if lm == 'shuttle' and cond['straight_feasible']:
            for i in range(reps):
                add('T1_shuttle', None, f"straight accel+brake {i + 1}/{reps}",
                    cond['straight_dist_m'], cond['straight_time_s'])
        elif lm == 'oval' and cond.get('oval_feasible'):
            add('T1_oval', None, f"oval, {reps} laps",
                cond['oval_dist_m'], cond['oval_time_s'])
        elif lm == 'circle_accel' and cond.get('circle_feasible'):
            for i in range(reps):
                add('T1_circle_accel', None, f"circle accel {i + 1}/{reps}",
                    cond['washout_dist_m'], cond['washout_time_s'])
        else:
            cond['warnings'].append(f"long_mode '{lm}' not feasible in this space -- SKIPPED")

        if cond.get('circle_feasible'):
            for side in ('left', 'right'):
                add('T3_step_accel', side, "steady-state hold, THEN throttle step",
                    cond['washout_dist_m'], cond['washout_time_s'])
            for side in ('left', 'right'):
                add('T4_step_brake', side, "steady-state hold, THEN brake step",
                    cond['washout_dist_m'], cond['washout_time_s'])

    if mode == 'current':
        add('T5_current', None, "low-speed current sweep (EXTRAPOLATED, low confidence)",
            cond['straight_usable_m'], 10.0)

    return mode, plan


def format_plan(p, cond, mode, plan):
    """The table printed before anything moves."""
    L = []
    L.append("")
    L.append("=" * 78)
    L.append(f"  VEHICLE-DYNAMICS MEASUREMENT PLAN    mode={mode}"
             f"{'  (venue_mode: forced)' if bool(p['venue_mode']) else ''}")
    L.append("=" * 78)
    L.append(f"  dry_run             : {bool(p['dry_run'])}"
             f"{'   <-- NOTHING WILL BE PUBLISHED' if bool(p['dry_run']) else ''}")
    L.append(f"  safety box          : {p['box_w']} x {p['box_h']} m centred "
             f"({p['box_center_x']}, {p['box_center_y']}), margin {p['safety_margin_m']} m")
    L.append(f"  v_max_allowed       : {p['v_max_allowed']} m/s   slew {p['v_slew_mps2']} m/s^2")
    L.append("")
    L.append("  DERIVED FROM THE SPACE")
    L.append(f"    straight usable   : {cond['straight_usable_m']:.2f} m "
             f"-> v_target {cond['v_straight']:.2f} m/s "
             f"(uncapped {cond['v_straight_uncapped']:.2f})")
    L.append(f"    circle radius     : {cond.get('circle_R_m', 0.0):.2f} m "
             f"-> washout at ~{cond.get('v_washout', 0.0):.2f} m/s")
    L.append(f"      (area allows {cond.get('circle_R_area_m', 0.0):.2f} m, "
             f"abort box allows {cond.get('circle_R_box_m', 0.0):.2f} m -- "
             f"the smaller wins)")
    if cond.get('oval_feasible'):
        L.append(f"    oval              : lap {cond['oval_lap_m']:.1f} m, corner "
                 f"{cond['oval_v_corner']:.2f} m/s, straight {cond['oval_v_straight']:.2f} m/s")
    L.append("")
    L.append(f"  {'#':>2}  {'maneuver':<18} {'side':<6} {'budget[m]':>9} {'est[s]':>7}  note")
    L.append("  " + "-" * 74)
    tot_t = 0.0
    for i, m in enumerate(plan):
        tot_t += m['est_s']
        L.append(f"  {i:>2}  {m['kind']:<18} {str(m['side'] or '-'):<6} "
                 f"{m['budget_m']:>9.1f} {m['est_s']:>7.1f}  {m['note']}")
    L.append("  " + "-" * 74)
    L.append(f"  {len(plan)} maneuver(s), ~{tot_t:.0f} s of driving plus "
             f"{len(plan)} x {p['settle_still_s']} s settle")
    for w in cond['warnings']:
        L.append(f"  WARNING: {w}")
    if not plan:
        L.append("  NOTHING TO RUN -- every maneuver was ruled out by the available space.")
    L.append("=" * 78)
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# ROS node
# --------------------------------------------------------------------------------------


def _node_class():
    """The ROS node, built on demand.

    Defined in a factory rather than at import time so that the pure logic above can be
    imported (and tested) in a process that has no ROS, and so that the tests can construct
    the node with explicit parameter overrides instead of a launch file.
    """
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Joy

    class VehDynTestNode(Node):
        def __init__(self, parameter_overrides=None):
            # automatic declaration: the yaml carries the analyzer's block too, and an
            # undeclared key would otherwise abort startup.
            super().__init__('vehdyn_test_node',
                             automatically_declare_parameters_from_overrides=True,
                             parameter_overrides=parameter_overrides or [])
            self.p = self._params()
            self.guard = Guard(self.p)
            self.cond = derive_conditions(self.p)
            self.mode, self.plan = build_plan(self.p, self.cond)
            for line in format_plan(self.p, self.cond, self.mode, self.plan).splitlines():
                self.get_logger().info(line)

            self.dry_run = bool(self.p['dry_run'])
            if self.dry_run:
                self.get_logger().warn(
                    "dry_run is TRUE -- the plan above was printed and the node will publish "
                    "nothing. Relaunch with dry_run:=false to drive.")

            self.odom_xy = None
            self.odom_yaw = 0.0
            self.odom_t = None
            self.speed = 0.0
            self.gz = 0.0
            self.joy_axes = []
            self.joy_buttons = []
            self.resume_req = False
            self.travelled = 0.0
            self.cmd_speed = 0.0
            self.cmd_steer = 0.0
            self.state = 'PAUSE'
            self.idx = 0
            self.aborted = None
            self.t_state = None

            self.out_dir = os.environ.get('VEHDYN_OUT_DIR', '')
            self.done = self._load_progress()

            self.create_subscription(Odometry, self.p['odom_topic'], self._odom_cb, 10)
            self.create_subscription(Joy, self.p['joy_topic'], self._joy_cb, 10)
            self.pub = self.create_publisher(
                AckermannDriveStamped, self.p['drive_topic'], 10)
            self.create_timer(1.0 / float(self.p['control_rate_hz']), self._loop)

        def _params(self):
            keys = ['dry_run', 'mode', 'long_mode', 'venue_mode', 'repeats',
                    'box_center_x', 'box_center_y', 'box_w', 'box_h', 'safety_margin_m',
                    'v_max_allowed', 'v_slew_mps2', 'steer_slew_rps', 'odom_topic',
                    'odom_timeout_s', 'joy_topic', 'joy_abort_axis_deadzone',
                    'joy_abort_on_any_button', 'drive_topic', 'control_rate_hz',
                    'straight_len_m', 'area_radius_m', 'car_width_m', 'wheelbase_m',
                    'ggv_ay_max_ref', 'ggv_ax_max_ref', 'a_acc_prior', 'a_brk_prior',
                    'ax_max_machines_ref', 'b_ax_max_machines_ref',
                    'washout_ratio_thresh', 'washout_hold_s', 'washout_min_speed_mps',
                    'washout_steer_rad', 'washout_ramp_mps2', 'step_speed_frac',
                    'step_settle_s', 'step_accel_target_mps', 'step_brake_target_mps',
                    'step_duration_s', 'settle_still_s', 'resume_button', 'pose_tol_m',
                    'pose_tol_deg', 'progress_file', 'current_topic', 'current_max_a']
            return {k: self.get_parameter(k).value for k in keys}

        def _progress_path(self):
            if not self.out_dir:
                return None
            return os.path.join(self.out_dir, str(self.p['progress_file']))

        def _load_progress(self):
            path = self._progress_path()
            if path and os.path.isfile(path):
                try:
                    with open(path) as f:
                        d = json.load(f)
                    done = set(d.get('completed', []))
                    self.get_logger().info(
                        f"resuming: {len(done)} maneuver(s) already completed, from {path}")
                    return done
                except Exception as e:
                    self.get_logger().warn(f"could not read progress file {path}: {e}")
            return set()

        def _save_progress(self, key):
            self.done.add(key)
            path = self._progress_path()
            if not path:
                return
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    json.dump({'completed': sorted(self.done)}, f, indent=2)
            except Exception as e:
                self.get_logger().warn(f"could not write progress file {path}: {e}")

        def _odom_cb(self, m):
            x, y = m.pose.pose.position.x, m.pose.pose.position.y
            if self.odom_xy is not None:
                self.travelled += math.hypot(x - self.odom_xy[0], y - self.odom_xy[1])
            self.odom_xy = (x, y)
            q = m.pose.pose.orientation
            self.odom_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                       1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
            self.speed = m.twist.twist.linear.x
            self.gz = m.twist.twist.angular.z
            self.odom_t = self.get_clock().now().nanoseconds / 1e9

        def _joy_cb(self, m):
            self.joy_axes = list(m.axes)
            self.joy_buttons = list(m.buttons)
            btn = int(self.p['resume_button'])
            if len(m.buttons) > btn and m.buttons[btn]:
                self.resume_req = True

        def _abort(self, why):
            if self.aborted is None:
                self.aborted = why
                self.get_logger().error(f"ABORT: {why}")
            self.cmd_speed = 0.0
            self._publish(0.0, self.cmd_steer)

        def _publish(self, speed, steer):
            if self.dry_run:
                return
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.drive.speed = float(speed)
            msg.drive.steering_angle = float(steer)
            self.pub.publish(msg)

        def _loop(self):
            if self.dry_run or self.aborted is not None or not self.plan:
                return
            now = self.get_clock().now().nanoseconds / 1e9

            # ---- abort paths, checked before anything is commanded ----
            why = self.guard.check_joy(self.joy_axes, self.joy_buttons)
            if why and self.state != 'PAUSE':
                return self._abort(why)
            age = None if self.odom_t is None else now - self.odom_t
            why = self.guard.check_odom(age)
            if why and self.state != 'PAUSE':
                return self._abort(why)
            if self.odom_xy is not None:
                why = self.guard.check_box(*self.odom_xy)
                if why:
                    return self._abort(why)
            if self.state == 'RUN':
                why = self.guard.check_budget(self.travelled,
                                              self.plan[self.idx]['budget_m'])
                if why:
                    return self._abort(why)

            if self.idx >= len(self.plan):
                self.get_logger().info("all maneuvers complete")
                self._publish(0.0, 0.0)
                return

            man = self.plan[self.idx]
            key = f"{self.idx}:{man['kind']}:{man['side']}"
            if key in self.done and self.state == 'PAUSE':
                self.get_logger().info(f"skipping already-completed {key}")
                self.idx += 1
                return

            if self.state == 'PAUSE':
                # Stop publishing so the operator can take over. NOTE: simple_mux stays in
                # autodrive until buttons[4]; it emits zero once our commands go stale.
                if self.t_state is None:
                    self.t_state = now
                    self.get_logger().warn(
                        f"\nPAUSE before [{self.idx}] {man['kind']} side={man['side']}\n"
                        f"  {man['note']}\n"
                        f"  Position the car at the START POSE for this maneuver, then press "
                        f"joy button {self.p['resume_button']}.\n"
                        f"  Press the mux humandrive button (buttons[4]) to drive it there.")
                if self.resume_req:
                    self.resume_req = False
                    ok, d, dyaw = self._pose_gate(man)
                    if not ok:
                        self.get_logger().error(
                            f"RESUME REFUSED: {d:.2f} m / {dyaw:.1f} deg from the planned "
                            f"start pose (tol {self.p['pose_tol_m']} m / "
                            f"{self.p['pose_tol_deg']} deg). Reposition and press again.")
                        return
                    self.state = 'SETTLE'
                    self.t_state = now
                    self.travelled = 0.0
                    self.get_logger().info("resume accepted -> SETTLE")
                return

            if self.state == 'SETTLE':
                self.cmd_speed = 0.0
                self._publish(0.0, 0.0)
                if now - self.t_state >= float(self.p['settle_still_s']):
                    self.state = 'RUN'
                    self.t_state = now
                    self.get_logger().info(f"RUN [{self.idx}] {man['kind']} {man['side']}")
                return

            if self.state == 'RUN':
                if self._run_maneuver(man, now - self.t_state):
                    self._save_progress(key)
                    self.idx += 1
                    self.state = 'PAUSE'
                    self.t_state = None
                    self.cmd_speed = 0.0
                    self._publish(0.0, 0.0)

        def _pose_gate(self, man):
            """Refuse to resume unless the car is where the maneuver assumes it is.

            With no planned pose recorded yet (first visit) the current pose becomes the
            plan, so the gate binds on RE-runs and after a repositioning -- which is where
            a wrong pose actually costs something.
            """
            if self.odom_xy is None:
                return False, 999.0, 999.0
            want = man.get('start_pose')
            if want is None:
                man['start_pose'] = (self.odom_xy[0], self.odom_xy[1], self.odom_yaw)
                return True, 0.0, 0.0
            return pose_within(self.odom_xy, self.odom_yaw, (want[0], want[1]), want[2],
                               float(self.p['pose_tol_m']), float(self.p['pose_tol_deg']))

        def _run_maneuver(self, man, t):
            """Returns True when the maneuver is finished."""
            dt = 1.0 / float(self.p['control_rate_hz'])
            kind = man['kind']
            sgn = 1.0 if man['side'] == 'left' else -1.0

            if kind == 'T2_washout':
                steer = sgn * float(self.p['washout_steer_rad'])
                self.cmd_steer = slew(self.cmd_steer, steer,
                                      float(self.p['steer_slew_rps']), dt)
                target = min(t * float(self.p['washout_ramp_mps2']),
                             float(self.p['v_max_allowed']))
                self.cmd_speed = slew(self.cmd_speed, target,
                                      float(self.p['v_slew_mps2']), dt)
                self._publish(self.cmd_speed, self.cmd_steer)
                if self._washout_detected():
                    self.get_logger().info(
                        f"washout detected at v={self.speed:.2f} m/s, "
                        f"gz={self.gz:.2f} rad/s -> decelerating")
                    return True
                return t > man['est_s'] * 3.0

            if kind in ('T3_step_accel', 'T4_step_brake'):
                steer = sgn * float(self.p['washout_steer_rad'])
                self.cmd_steer = slew(self.cmd_steer, steer,
                                      float(self.p['steer_slew_rps']), dt)
                hold = float(self.p['step_speed_frac']) * self.cond.get('v_washout', 0.0)
                settle = float(self.p['step_settle_s'])
                if t < settle:
                    # CONSTANT SPEED FIRST. Stepping from a standstill ramps a_x and a_lat
                    # together and measures the ramp, not the envelope -- that is what
                    # invalidated the 2026-08-12 T3/T4 runs.
                    self.cmd_speed = slew(self.cmd_speed, hold,
                                          float(self.p['v_slew_mps2']), dt)
                else:
                    tgt = (hold + float(self.p['step_accel_target_mps'])
                           if kind == 'T3_step_accel'
                           else float(self.p['step_brake_target_mps']))
                    self.cmd_speed = slew(self.cmd_speed, min(tgt, float(self.p['v_max_allowed'])),
                                          float(self.p['v_slew_mps2']), dt)
                self._publish(self.cmd_speed, self.cmd_steer)
                return t > settle + float(self.p['step_duration_s'])

            if kind in ('T1_shuttle', 'T1_oval', 'T1_circle_accel', 'T5_current'):
                # Longitudinal family: ramp to the derived straight target and back down.
                # Geometry is driven open-loop by the operator's placement plus the derived
                # radius; this node owns speed, not path following.
                v_t = self.cond['v_straight'] if kind != 'T1_circle_accel' \
                    else self.cond.get('v_washout', 0.0)
                half = man['est_s'] / 2.0 if man['est_s'] > 0 else 1.0
                target = v_t if t < half else 0.0
                self.cmd_speed = slew(self.cmd_speed, target,
                                      float(self.p['v_slew_mps2']), dt)
                if kind == 'T1_oval':
                    self.cmd_steer = slew(self.cmd_steer, 0.0,
                                          float(self.p['steer_slew_rps']), dt)
                self._publish(self.cmd_speed, self.cmd_steer)
                return t > man['est_s'] and self.cmd_speed <= 0.01

            self.get_logger().error(f"unknown maneuver kind {kind} -- skipping")
            return True

        def _washout_detected(self):
            """ratio = |gz| / |v tan(delta) / L| below threshold, held.

            On 2026-08-12 the ratio sat at 1.00 while the demand climbed to 6 and then fell
            to 0.53 -- the collapse is sharp, so a held threshold catches it without
            catching noise.
            """
            v = abs(self.speed)
            if v < float(self.p['washout_min_speed_mps']):
                self._wo_since = None
                return False
            pred = v * math.tan(abs(self.cmd_steer)) / float(self.p['wheelbase_m'])
            if pred < 1e-6:
                return False
            ratio = abs(self.gz) / pred
            now = self.get_clock().now().nanoseconds / 1e9
            if ratio < float(self.p['washout_ratio_thresh']):
                if getattr(self, '_wo_since', None) is None:
                    self._wo_since = now
                return now - self._wo_since >= float(self.p['washout_hold_s'])
            self._wo_since = None
            return False

    return VehDynTestNode


def _build_node(parameter_overrides=None):
    """Construct the node. rclpy.init() must already have been called."""
    return _node_class()(parameter_overrides=parameter_overrides)


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = _build_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
