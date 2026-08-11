"""PP + Friction Circle Controller.

Follows the global raceline (/local_waypoints) with a speed-adaptive lookahead, and uses
a friction circle to (a) clamp the steering to the available lateral grip and (b) split
the remaining grip into the longitudinal accel budget. A residual heading-error PID trims
what pure-pursuit leaves on the table.

Steering (each tick):  δ_cmd = clip(δ_geo + δ_pid, ±δ_max)
  δ_geo — pure-pursuit to the lookahead point, friction-circle limited:
            kappa_pp  = 2·ly / dist²              (curvature to lookahead point)
            a_lat_use = lat_safety · a_total_max  (usable lateral grip)
            kappa_geo = clip(kappa_pp, ±a_lat_use/vx²)
            δ_geo     = atan(L · kappa_geo)        (valid at all speeds, incl. standstill)
  δ_pid — PID on the path-tangent error e_h = wrap(ψ_path − yaw). δ_geo already pursues the
            lookahead (cross-track + curvature), so the PID only corrects the residual
            heading vs path tangent. Kd on the tangent error == yaw-rate damping; the
            derivative is low-pass filtered (heading_d_tau).

Speed:
  a_lat_used    = max(|vx²·kappa_geo|, |vx·yaw_rate|_filt)    (P1: measured-fed friction circle)
  a_long_budget = min(a_long_max, √(a_total_max² − a_lat_used²))
  v_ref         = raceline vx at a forward look-ahead point (L1-style, anticipatory), lifted
                  toward √(a_lat_use/κ) by corner_push (P2), then capped by a backward
                  braking-feasible pass for slower corners ahead.
  v_cmd         = min(v_ref, vx + a_long_budget·t_cmd_horizon)   [accel] / v_ref [brake]

Latency comp (B): steering is computed for the pose future_constant·v ahead.
H2H: TRAILING gap controller + FTGONLY follow-the-gap fallback + local-wpnt watchdog.

I/O: odom /car_state/odom, waypoints /local_waypoints, drive → drive_topic param.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from f110_msgs.msg import WpntArray, BehaviorStrategy

from controller.ftg.ftg import FTG

PARAMS = {
    # vehicle — unified with global-trajectory dynamics (stack_master/config/<ver>/)
    'wheelbase_L':     0.33,   # vehicle_config wheelbase
    'delta_max':       0.4189, # dynamics.yaml s_max
    'v_min':           0.5,
    'v_max':           15.0,   # racecar_f110.ini veh_params v_max
    'v_min_for_ik':    0.5,    # [m/s] friction-circle / cross-track speed floor
    # control rate
    'control_rate_hz': 50.0,
    # adaptive lookahead (speed-based time-headway): ld = clip(t_headway·vx, ld_min, ld_max)
    'ld_min':          1.0,
    'ld_max':          5.0,    # longer at speed → smoother heading (A)
    't_headway':       0.6,    # [s]      vx → lookahead gain (time headway)
    # friction circle — unified with global-trajectory ggv (ay_max=4.5, ax_max=5.0)
    'a_total_max':     5.0,    # [m/s²]  circle radius = ggv max(ax_max, ay_max)
    'a_long_max':      5.0,    # [m/s²]  = ggv ax_max / ax_max_machines
    'lat_safety':      0.9,    # [-]     a_lat_use = 0.9·5.0 = 4.5 = ggv ay_max
    # measured lateral-accel feedback into the friction circle (P1). a_lat_used =
    # max(commanded vx²·κ_geo, |vx·yaw_rate|_filtered): closes the loop on what the car
    # is REALLY pulling, so the longitudinal budget reflects reality (conservative on
    # corner entry / understeer → no combined-grip over-drive). yaw_rate from odom.
    'use_measured_a_lat': True,
    'a_lat_meas_tau':  0.10,   # [s]  low-pass time constant on the measured lateral accel
    'corner_push':     0.0,    # [0..1] P2: lift corner speed from raceline toward √(a_lat_use/κ); 0=raceline
    # residual heading-error PID on the path-tangent error e_h = wrap(ψ_path − yaw).
    # δ_pid = Kp·e + Ki·∫e + Kd·ė(filtered).  Kp → line convergence, Kd → damp.
    'heading_kp':      0.4,    # [-]
    'heading_ki':      0.0,    # [1/s]
    'heading_kd':      0.05,   # [s]
    'heading_i_max':   0.2,    # [rad]  integral clamp (anti-windup)
    'heading_d_tau':   0.05,   # [s]    derivative low-pass time constant
    # speed
    'a_brake_max':     5.0,    # [m/s²]  anticipatory braking decel = ggv b_ax_max
    't_cmd_horizon':   0.3,    # [s]     가속 명령 horizon
    'speed_lookahead': 0.3,    # [s]     forward time to read the raceline target speed (L1-style)
    'future_constant': 0.05,   # [s]     latency comp: steer for the pose τ seconds ahead (B)
    # ── H2H: TRAILING gap controller (ported from L1 controller_manager) ─────
    # In the TRAILING state, hold a fixed gap behind the opponent. The gap PID
    # output is clipped to v_ref (this controller's friction-circle speed cap) so
    # the trailing speed can never exceed the dynamics-feasible reference.
    'state_machine_rate': 40.0,  # [Hz]  rate used to integrate the gap error
    'trailing_gap':       1.55,  # [m]   base standstill gap to the opponent
    'trailing_vel_gain':  0.25,  # [s]   speed-proportional gap term (gap_should = gain·v + gap)
    'trailing_p_gain':    1.35,  # [-]   gap-error P gain
    'trailing_i_gain':    0.0,   # [-]   gap-error I gain
    'trailing_d_gain':    1.0,   # [-]   relative-velocity D gain
    'blind_trailing_speed': 1.5, # [m/s] floor speed when opponent not visible and gap is large
    # ── H2H: FTG fallback (FTGONLY state) — defaults mirror controller.yaml ──
    'ftg_debug':          False,
    'ftg_safety_radius':  40.0,  # bubble radius (LiDAR samples)
    'ftg_max_lidar_dist': 9.0,   # [m]
    'ftg_max_speed':      6.0,   # [m/s]
    'ftg_range_offset':   180.0, # samples trimmed each side (MUST be >0)
    'ftg_track_width':    2.6,   # [m]
    # ── H2H: local-waypoint freshness watchdog ──────────────────────────────
    'wpnt_timeout_s':     0.5,   # [s]  no fresh /local_waypoints for this long → stop
}

# /pp_debug Float64MultiArray field indices
_D_A_LAT_PP   = 0   # [m/s²]  demanded lateral vx²·κ_pp (pre-clip; > a_lat_geo ⇒ steering clamped)
_D_A_LAT_GEO  = 1   # [m/s²]  commanded lateral vx²·κ_geo (post friction clamp)
_D_A_LAT_MEAS = 2   # [m/s²]  measured lateral |vx·yaw_rate| (filtered, P1)
_D_DELTA_GEO  = 3   # [deg]   pure-pursuit base steering
_D_DELTA_PID  = 4   # [deg]   residual heading-error PID trim
_D_DELTA_CMD  = 5   # [deg]   total commanded steering
_D_E_H        = 6   # [rad]   path-tangent heading error ψ_path−yaw (PID input)
_D_LD         = 7   # [m]     adaptive lookahead distance
_D_V_REF      = 8   # [m/s]   velocity reference (after braking pass)
_D_V_CMD      = 9   # [m/s]   commanded speed
_DEBUG_LEN    = 10


class PPHeadingController(Node):

    def __init__(self):
        super().__init__('pp_heading_controller')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        # drive topic (IFAC mux input by default; string param so declared separately)
        self.declare_parameter(
            'drive_topic', '/vesc/high_level/ackermann_cmd')
        drive_topic = str(self.get_parameter('drive_topic').value)

        self.wheelbase     = float(p('wheelbase_L'))
        self.delta_max     = float(p('delta_max'))
        self.v_min         = float(p('v_min'))
        self.v_max         = float(p('v_max'))
        self.v_min_for_ik  = float(p('v_min_for_ik'))
        self._dt           = 1.0 / float(p('control_rate_hz'))
        self.ld_min        = float(p('ld_min'))
        self.ld_max        = float(p('ld_max'))
        self.t_headway     = float(p('t_headway'))
        self.a_total_max   = float(p('a_total_max'))
        self.a_long_max    = float(p('a_long_max'))
        self.lat_safety    = float(p('lat_safety'))
        self.use_measured_a_lat = bool(p('use_measured_a_lat'))
        self.a_lat_meas_tau = float(p('a_lat_meas_tau'))
        self.corner_push = float(p('corner_push'))
        self.heading_kp    = float(p('heading_kp'))
        self.heading_ki    = float(p('heading_ki'))
        self.heading_kd    = float(p('heading_kd'))
        self.heading_i_max = float(p('heading_i_max'))
        self.heading_d_tau = float(p('heading_d_tau'))
        self.a_brake_max   = float(p('a_brake_max'))
        self.t_cmd_horizon = float(p('t_cmd_horizon'))
        self.speed_lookahead = float(p('speed_lookahead'))
        self.future_constant = float(p('future_constant'))

        # H2H: trailing gap controller params
        self.state_machine_rate   = float(p('state_machine_rate'))
        self.trailing_gap         = float(p('trailing_gap'))
        self.trailing_vel_gain    = float(p('trailing_vel_gain'))
        self.trailing_p_gain      = float(p('trailing_p_gain'))
        self.trailing_i_gain      = float(p('trailing_i_gain'))
        self.trailing_d_gain      = float(p('trailing_d_gain'))
        self.blind_trailing_speed = float(p('blind_trailing_speed'))
        self.wpnt_timeout_s       = float(p('wpnt_timeout_s'))

        self.odom         = None
        self.waypoints    = []
        self._nearest_idx = None
        self._last_pos    = None
        self._he_int      = 0.0    # heading PID integral
        self._he_prev     = None   # previous e_h (None = first tick)
        self._he_deriv    = 0.0    # filtered derivative state
        self._a_lat_meas  = 0.0    # filtered measured lateral accel |vx·yaw_rate| (P1)

        # H2H state (filled by /behavior_strategy, /car_state/odom_frenet, /global_waypoints)
        self.state        = ""
        # opponent = [s_center, d_center, vs, is_static, is_visible] or None
        self.opponent     = None
        self.ego_s        = None
        self.ego_vs       = 0.0
        self.track_length = None
        self._i_gap       = 0.0    # trailing gap-error integral
        self._last_wp_t   = None   # ros time of last /local_waypoints (watchdog)
        self.scan         = None

        # FTG controller for the FTGONLY state (self-contained; node=self enables viz)
        self.ftg = FTG(
            node=self,
            debug=bool(p('ftg_debug')),
            safety_radius=int(p('ftg_safety_radius')),
            max_lidar_dist=float(p('ftg_max_lidar_dist')),
            max_speed=float(p('ftg_max_speed')),
            range_offset=int(p('ftg_range_offset')),
            track_width=float(p('ftg_track_width')),
        )

        # cached numpy arrays — rebuilt only when waypoints change
        self._wx      = None
        self._wy      = None
        self._s_vals  = None
        self._psi_wp  = None
        self._kappa   = None
        self._vx_wp   = None
        self._N       = 0
        self._s_total = 0.0

        self.create_subscription(Odometry,  '/car_state/odom',  self._odom_cb, 10)
        self.create_subscription(WpntArray, '/local_waypoints', self._wp_cb,   10)
        # H2H inputs
        self.create_subscription(BehaviorStrategy, '/behavior_strategy', self._behavior_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom_frenet', self._odom_frenet_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._global_wp_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.drive_pub     = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)
        self.debug_pub     = self.create_publisher(Float64MultiArray, '/pp_debug', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/pp/lookahead', 10)
        self.create_timer(self._dt, self._loop)

        self.get_logger().info(
            f'[PP] drive_topic={drive_topic}  '
            f't_headway={self.t_headway}  ld=[{self.ld_min},{self.ld_max}]  '
            f'a_total_max={self.a_total_max}  a_long_max={self.a_long_max}  '
            f'lat_safety={self.lat_safety} (a_lat_use={self.lat_safety*self.a_total_max:.2f})  '
            f'Kp={self.heading_kp}  Ki={self.heading_ki}  Kd={self.heading_kd}  '
            f'd_tau={self.heading_d_tau}  dt={self._dt*1000:.1f} ms'
        )

    def _odom_cb(self, msg): self.odom = msg

    def _wp_cb(self, msg):
        self.waypoints = msg.wpnts
        self._wx      = np.array([wp.x_m        for wp in self.waypoints])
        self._wy      = np.array([wp.y_m        for wp in self.waypoints])
        self._s_vals  = np.array([wp.s_m        for wp in self.waypoints])
        self._psi_wp  = np.array([wp.psi_rad    for wp in self.waypoints])
        self._kappa   = np.array([wp.kappa_radpm for wp in self.waypoints])
        self._vx_wp   = np.array([wp.vx_mps     for wp in self.waypoints])
        self._N       = len(self.waypoints)
        self._s_total = float(self._s_vals[-1]) if self._N > 0 else 0.0
        self._nearest_idx = None
        self._he_int   = 0.0       # reset PID state on path change
        self._he_prev  = None
        self._he_deriv = 0.0
        self._last_wp_t = self.get_clock().now()   # watchdog: fresh waypoints just arrived

    # ── H2H callbacks ────────────────────────────────────────────────────────
    def _behavior_cb(self, msg: BehaviorStrategy):
        self.state = msg.state
        if len(msg.trailing_targets) != 0:
            o = msg.trailing_targets[0]
            self.opponent = [o.s_center, o.d_center, o.vs, o.is_static, o.is_visible]
        else:
            self.opponent = None

    def _odom_frenet_cb(self, msg: Odometry):
        self.ego_s  = msg.pose.pose.position.x   # frenet s
        self.ego_vs = msg.twist.twist.linear.x   # s-velocity

    def _global_wp_cb(self, msg: WpntArray):
        if len(msg.wpnts) != 0:
            self.track_length = msg.wpnts[-1].s_m

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg
        self.ftg.set_vel(abs(self.odom.twist.twist.linear.x) if self.odom is not None else 0.0)

    # ── arc-length forward walk ──────────────────────────────────────────────

    def _advance(self, start: int, dist: float,
                 wx, wy, s_vals, s_total: float, N: int) -> int:
        """Index ~dist metres ahead of start along the path (forward only)."""
        acc = 0.0
        i   = start
        for _ in range(N - 1):
            j   = (i + 1) % N
            seg = (s_vals[j] - s_vals[i]) % s_total
            if seg <= 1e-6:
                seg = math.hypot(wx[j] - wx[i], wy[j] - wy[i])
            acc += seg
            if acc >= dist:
                return j
            i = j
        return i

    # ── main loop ────────────────────────────────────────────────────────────

    def _loop(self):
        # ── H2H: FTGONLY fallback — bypass the PP/raceline law, drive on /scan ──
        if self.state == "FTGONLY":
            if self.scan is not None:
                speed, steer = self.ftg.process_lidar(self.scan.ranges)
                self._publish_drive(steer, speed)
            else:
                self._publish_drive(0.0, 0.0)
            return

        if self.odom is None or self._wx is None or self._N == 0:
            return

        # ── H2H: local-waypoint freshness watchdog — stop if state machine output went stale ──
        if self._last_wp_t is not None:
            age = (self.get_clock().now() - self._last_wp_t).nanoseconds * 1e-9
            if age > self.wpnt_timeout_s:
                self.get_logger().error(
                    f"[PP] no fresh /local_waypoints for {age:.2f}s — STOPPING",
                    throttle_duration_sec=0.5)
                self._publish_drive(0.0, 0.0)
                return

        vx       = abs(self.odom.twist.twist.linear.x)
        p_x_m    = self.odom.pose.pose.position.x
        p_y_m    = self.odom.pose.pose.position.y
        q        = self.odom.pose.pose.orientation
        yaw_m    = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        yaw_rate = float(self.odom.twist.twist.angular.z)

        # ── Latency / future-position compensation (B) ───────────────────────
        # Steer for where the car WILL be in future_constant seconds, not where it is
        # now: at speed this cancels the sensing+actuation lag that otherwise turns into
        # heading oscillation (mirrors the L1 controller's future_position). Position is
        # advanced along the current heading; yaw is advanced by the measured yaw_rate.
        tau = self.future_constant
        p_x = p_x_m + vx * math.cos(yaw_m) * tau
        p_y = p_y_m + vx * math.sin(yaw_m) * tau
        yaw = yaw_m + yaw_rate * tau

        wx      = self._wx
        wy      = self._wy
        s_vals  = self._s_vals
        N       = self._N
        s_total = self._s_total

        # ── 1. Nearest waypoint ──────────────────────────────────────────────
        # teleport check on the MEASURED pose (prediction is continuous, won't jump)
        if self._last_pos is not None:
            dx = p_x_m - self._last_pos[0]
            dy = p_y_m - self._last_pos[1]
            if dx * dx + dy * dy > 0.25:    # teleport > 0.5 m → reset
                self._nearest_idx = None
        self._last_pos = (p_x_m, p_y_m)

        if self._nearest_idx is None:
            d2      = (wx - p_x) ** 2 + (wy - p_y) ** 2
            aligned = np.cos(yaw - self._psi_wp) > 0.0
            if np.any(aligned):
                d2 = np.where(aligned, d2, np.inf)
            nearest_idx = int(np.argmin(d2))
        else:
            cands       = [(self._nearest_idx + k) % N for k in range(30)]
            nearest_idx = min(cands, key=lambda i: (wx[i] - p_x) ** 2 + (wy[i] - p_y) ** 2)
        self._nearest_idx = nearest_idx

        # ── 2. Adaptive lookahead (speed-based time-headway) ─────────────────
        # ld ∝ vx (clamped). Corner dynamics are handled by the friction-circle
        # speed cap + braking pass below, so no curvature shortening is needed.
        ld = float(np.clip(self.t_headway * vx, self.ld_min, self.ld_max))

        # ── 3. Lookahead point ───────────────────────────────────────────────
        psi_n    = float(self.waypoints[nearest_idx].psi_rad)
        tnx, tny = math.cos(psi_n), math.sin(psi_n)
        extra    = ((p_x - float(wx[nearest_idx])) * tnx
                  + (p_y - float(wy[nearest_idx])) * tny)
        target_idx = self._advance(nearest_idx, max(0.0, ld + extra),
                                   wx, wy, s_vals, s_total, N)

        lx_w = float(wx[target_idx])
        ly_w = float(wy[target_idx])

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        dtx  =  lx_w - p_x
        dty  =  ly_w - p_y
        lx_v =  dtx * cos_y + dty * sin_y
        ly_v = -dtx * sin_y + dty * cos_y
        dist = math.hypot(lx_v, ly_v)

        if dist < 1e-6:
            self._he_int   = 0.0
            self._he_prev  = None
            self._he_deriv = 0.0
            self._publish_drive(0.0, self.v_min)
            return

        # ── 4a. Lateral-accel share → steering (PP, friction-circle limited) ─
        # δ_geo = atan(L·kappa) computed directly so steering is valid at ALL
        # speeds (incl. standstill). The usable lateral grip a_lat_use caps the
        # curvature: kappa_max = a_lat_use/vx² (only binds at high speed). The
        # SAME a_lat_use feeds the corner speed cap in §5, so the two agree.
        kappa_pp  = 2.0 * ly_v / (dist * dist)
        v_safe    = max(vx, self.v_min_for_ik)
        a_lat_use = self.lat_safety * self.a_total_max
        kappa_max = a_lat_use / (v_safe * v_safe)
        kappa_geo = float(np.clip(kappa_pp, -kappa_max, kappa_max))
        delta_geo = math.atan(self.wheelbase * kappa_geo)
        a_lat_pp  = vx * vx * kappa_pp
        a_lat_geo = vx * vx * kappa_geo

        # ── 4b. Residual heading-error PID (on the path-tangent error) ───────
        # e_h = wrap(ψ_path − yaw): the car heading vs the path tangent at the
        # nearest waypoint. δ_geo already pursues the lookahead point (cross-track
        # + curvature), so this is the RESIDUAL pure-pursuit leaves — corrected by
        # Kp (convergence) and Kd (damping; Kd on the tangent error == yaw-rate
        # damping, so no separate δ_damp is needed). The derivative is LOW-PASS
        # FILTERED (d_tau) so Kd is usable — raw 50 Hz de/dt is too noisy.
        e_h = math.atan2(math.sin(psi_n - yaw), math.cos(psi_n - yaw))
        if self._he_prev is None:
            de_raw = 0.0
        else:
            de_raw = (e_h - self._he_prev) / self._dt
        self._he_prev = e_h
        # 1-pole low-pass on the derivative
        a_d = self._dt / (self.heading_d_tau + self._dt) if self.heading_d_tau > 0.0 else 1.0
        self._he_deriv += a_d * (de_raw - self._he_deriv)

        if vx >= self.v_min_for_ik:
            self._he_int += e_h * self._dt
        else:
            self._he_int = 0.0
        i_term = float(np.clip(self.heading_ki * self._he_int,
                               -self.heading_i_max, self.heading_i_max))
        if self.heading_ki > 1e-9:
            self._he_int = i_term / self.heading_ki      # anti-windup back-calc

        delta_pid = (self.heading_kp * e_h
                     + i_term
                     + self.heading_kd * self._he_deriv)

        # ── 4c. Combined steering ────────────────────────────────────────────
        delta_cmd = float(np.clip(delta_geo + delta_pid, -self.delta_max, self.delta_max))

        # ── 5. Speed ─────────────────────────────────────────────────────────
        # Longitudinal grip budget from the friction circle, using the ACTUAL lateral
        # accel (P1): a_lat_used = max(commanded vx²·kappa_geo, measured |vx·yaw_rate|).
        # The larger keeps the budget conservative on corner-entry loading (measured
        # lags) and understeer (front saturated) — never over-driving combined grip.
        # a_lat_used ≤ a_lat_use < a_total_max, so some longitudinal grip is always left.
        a_lat_meas_raw = abs(vx * yaw_rate)
        a_d_lat = self._dt / (self.a_lat_meas_tau + self._dt) if self.a_lat_meas_tau > 0.0 else 1.0
        self._a_lat_meas += a_d_lat * (a_lat_meas_raw - self._a_lat_meas)
        a_lat_used = max(abs(a_lat_geo), self._a_lat_meas) if self.use_measured_a_lat else abs(a_lat_geo)
        a_long_budget = min(self.a_long_max,
                            math.sqrt(max(0.0, self.a_total_max ** 2 - a_lat_used ** 2)))

        # ── v_ref: L1-style anticipatory target + corner-braking safety ──────
        # (1) Target = raceline vx at a FORWARD look-ahead point (like the L1
        # controller's speed_lookahead). Reading ahead — instead of pinning v_ref to
        # the current point's profile speed (a backward-pass min over the whole window,
        # which lags the profile's own ramp) — lets the car accelerate out of corners
        # and onto straights as early as L1 does. On the nominal line this just
        # reproduces the (friction-optimal) global velocity profile.
        s_ahead = (s_vals - float(s_vals[nearest_idx])) % s_total
        la_dist = max(self.ld_min, self.speed_lookahead * vx)
        spd_idx = self._advance(nearest_idx, la_dist, wx, wy, s_vals, s_total, N)
        v_ref   = float(self._vx_wp[spd_idx])
        # P2 — corner-speed release: lift the corner target from the (conservative) raceline
        # vx toward the real friction corner limit √(a_lat_use/κ), by corner_push∈[0,1].
        # 0 = raceline exactly (default). The SAME a_lat_use feeds the steering clamp, so the
        # commanded corner speed stays achievable (no understeer-wide). Straights: v_curve≫,
        # raceline≈v_max → head≈0 → unaffected. Push grip via lat_safety while watching the
        # P1 measured a_lat (/pp_debug idx 11) approach a_total_max.
        if self.corner_push > 0.0:
            k_fwd       = abs(float(self._kappa[spd_idx]))
            v_curve_fwd = math.sqrt(a_lat_use / max(k_fwd, 1e-3))
            v_ref      += self.corner_push * max(0.0, min(v_curve_fwd, self.v_max) - v_ref)

        # (2) Corner braking: never exceed the speed from which we can still brake to a
        # SLOWER point ahead (BEYOND la_dist) at a_brake_max — √(v_target² + 2·a_brake·Δs),
        # v_target = min(raceline vx, curvature cap √(a_lat_use/|κ|)). Points within
        # la_dist are trusted to the raceline vx + steering clamp, so corner-EXIT accel
        # is no longer pinned by the current low profile speed.
        brake_hd = vx ** 2 / (2.0 * self.a_brake_max) + self.ld_min
        mask     = (s_ahead > la_dist) & (s_ahead <= brake_hd)
        if np.any(mask):
            ds        = s_ahead[mask]
            kappa_win = np.abs(self._kappa[mask])
            v_curve   = np.sqrt(a_lat_use / np.maximum(kappa_win, 1e-3))
            v_rl      = self._vx_wp[mask]
            # P2: same corner-speed lift on the braking targets (consistent with v_ref above)
            if self.corner_push > 0.0:
                v_target = v_rl + self.corner_push * np.maximum(
                    0.0, np.minimum(v_curve, self.v_max) - v_rl)
            else:
                v_target = v_rl
            v_target  = np.minimum(v_target, v_curve)   # never exceed the friction corner cap
            v_allow   = np.sqrt(v_target ** 2 + 2.0 * self.a_brake_max * ds)
            v_ref     = min(v_ref, float(np.min(v_allow)))
        v_ref = max(v_ref, self.v_min)

        # Accelerate toward v_ref bounded by the friction-circle longitudinal budget;
        # braking is never budget-limited (let the car slow as needed for corners).
        if v_ref > vx:
            v_cmd = min(v_ref, vx + a_long_budget * self.t_cmd_horizon)
        else:
            v_cmd = v_ref

        # ── H2H: TRAILING overrides the raceline speed with a gap-keeping command,
        # clipped to v_ref so it never exceeds the friction-feasible speed. Allowed
        # down to 0 so the car can stop behind the opponent. Steering is unchanged
        # (still tracks /local_waypoints, incl. any avoidance path baked in upstream).
        if (self.state == "TRAILING" and self.opponent is not None
                and self.ego_s is not None and self.track_length):
            v_cmd = float(np.clip(self._trailing_controller(v_ref), 0.0, self.v_max))
        else:
            self._i_gap = 0.0
            v_cmd = float(np.clip(v_cmd, self.v_min, self.v_max))

        # ── publish ──────────────────────────────────────────────────────────
        self._publish_drive(delta_cmd, v_cmd)
        self._publish_lookahead(lx_w, ly_w, ld)
        self._publish_debug(a_lat_pp, a_lat_geo, self._a_lat_meas,
                            delta_geo, delta_pid, delta_cmd, e_h, ld, v_ref, v_cmd)

    # ── H2H: trailing gap controller ─────────────────────────────────────────

    def _trailing_controller(self, v_cap: float) -> float:
        """Hold a fixed gap behind the opponent (ported from the L1 controller_manager
        trailing_controller). Returns a target speed clipped to [0, v_cap], where
        v_cap is this controller's friction-circle speed reference (v_ref) so the
        trailing command can never exceed the dynamics-feasible speed.

        gap_should = trailing_vel_gain·v + trailing_gap  (speed-dependent target gap)
        cmd        = opp_vs − P·gap_error − I·∫gap_error − D·(ego_vs − opp_vs)
        """
        opp_s, _opp_d, opp_vs, _opp_static, opp_visible = self.opponent
        gap        = (opp_s - self.ego_s) % self.track_length
        gap_should = self.trailing_vel_gain * self.ego_vs + self.trailing_gap
        gap_error  = gap_should - gap
        v_diff     = self.ego_vs - opp_vs
        self._i_gap = float(np.clip(self._i_gap + gap_error / self.state_machine_rate,
                                    -10.0, 10.0))

        p_value = gap_error * self.trailing_p_gain
        i_value = self._i_gap * self.trailing_i_gain
        d_value = v_diff * self.trailing_d_gain

        cmd = float(np.clip(opp_vs - p_value - i_value - d_value, 0.0, v_cap))
        # opponent not visible but still farther than the target gap → keep creeping
        if (not opp_visible) and gap > gap_should:
            cmd = max(self.blind_trailing_speed, cmd)
        return cmd

    # ── publishers ───────────────────────────────────────────────────────────

    def _publish_drive(self, delta: float, speed: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.drive.steering_angle = delta
        msg.drive.speed          = speed
        self.drive_pub.publish(msg)

    def _publish_lookahead(self, x: float, y: float, ld: float) -> None:
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns, m.id        = 'pp_lookahead', 0
        m.type, m.action  = Marker.SPHERE, Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        s = float(np.clip(ld * 0.15, 0.1, 0.4))
        m.scale.x = m.scale.y = m.scale.z = s
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 1.0, 1.0
        self.lookahead_pub.publish(m)

    def _publish_debug(self, a_lat_pp, a_lat_geo, a_lat_meas,
                       delta_geo, delta_pid, delta_cmd, e_h, ld, v_ref, v_cmd):
        d = [0.0] * _DEBUG_LEN
        d[_D_A_LAT_PP]   = a_lat_pp
        d[_D_A_LAT_GEO]  = a_lat_geo
        d[_D_A_LAT_MEAS] = a_lat_meas
        d[_D_DELTA_GEO]  = math.degrees(delta_geo)
        d[_D_DELTA_PID]  = math.degrees(delta_pid)
        d[_D_DELTA_CMD]  = math.degrees(delta_cmd)
        d[_D_E_H]        = e_h
        d[_D_LD]         = ld
        d[_D_V_REF]      = v_ref
        d[_D_V_CMD]      = v_cmd
        msg = Float64MultiArray()
        msg.data = d
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PPHeadingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
