#!/usr/bin/env python3
"""
local_planning.py - Standalone mode-selectable local planner.

Single file: perception + Frenet + planner. PP receives this node's
/local_waypoints via launch-level topic remap (PP.py unchanged).

Pipeline:
    /scan, /vesc/odom, /global_waypoints                          INPUT
        1. perception  : cluster -> l_shape_fitting -> tracking
        2. Frenet      : raceline cubic spline + to_frenet/to_cartesian
        3. mode branch :
             free          - raceline forward window
             trailing      - raceline + trailing speed cap
             spline_avoid  - left/right cubic spline candidates,
                             fall back to trailing if both infeasible
        4. publish WpntArray
    /local_waypoints                                              OUTPUT

Run:
    /usr/bin/python3 local_planning.py --ros-args -p mode:=spline_avoid
"""

import math

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from f110_msgs.msg import Wpnt, WpntArray


# ===========================================================================
#  PERCEPTION - Geometry helpers  (provided)
# ===========================================================================

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def scan_to_xy(ranges: np.ndarray, angle_min: float, angle_inc: float):
    # --- tunable parameters ---
    r_min = 0.05    # [m] ignore returns closer than this
    r_max = 10.0    # [m] ignore returns farther than this

    n = ranges.shape[0]
    angles = angle_min + np.arange(n) * angle_inc
    valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
    x = np.where(valid, ranges * np.cos(angles), np.nan)
    y = np.where(valid, ranges * np.sin(angles), np.nan)
    return x, y

# ===========================================================================
#  PERCEPTION  (Using your code from perception_assignment.py)
# ===========================================================================

def cluster(x: np.ndarray, y: np.ndarray, angle_inc: float):

    # --- tunable parameters ---
    lambda_rad = math.radians(30.0)   # max admissible incidence angle
    sigma = 0.3                      # range-noise term [m]
    min_points = 15                   # drop clusters smaller than this

    clusters = []
    current = []
    prev = None
    



    ### Write your code ###




    return clusters


def l_shape_fitting(clusters):
    
    # --- tunable parameters ---
    max_obs_size = 1.0    # [m] drop boxes whose larger side exceeds this
    min_size     = 0.3    # [m] floor on each side (partial views stay >= this)
    min_edge     = 0.01   # [m] floor on point-to-edge distance (avoid 1/0)
    n_angles     = 90     # orientation search resolution over [0, 90) deg

    thetas = np.linspace(0.0, np.pi / 2 - np.pi / 180, n_angles)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    obstacles = []
    



    ### Write your code ####




    return obstacles


def tracking(obstacles, track, dt: float, ego):
    
    # --- tunable parameters ---
    opp_max_lat = 1.0    # [m] ignore obstacles farther sideways than this
    max_misses = 10      # drop the track after this many missed frames
    q = 0.05              # Kalman process-noise scale
    r = 0.10             # Kalman measurement-noise scale
    lidar_to_base_x = 0.27  # [m] base_link -> laser TF (x); odom is base_link

    ex, ey, eyaw = ego
    meas = None



    ### Write your code ####

    

    if track is None:
        if meas is None:
            return None
        state = np.array([meas[0], meas[1], 0.0, 0.0])
        return (state, np.eye(4), 0)

    state, P, misses = track




    ### Write your code ####




    misses = 0 if meas is not None else misses + 1
    if misses > max_misses:
        return None
    return (state, P, misses)


def trailing(track, ego, ego_v) -> float:
    
    # --- tunable parameters ---
    base_speed   = 4.0    # [m/s] free-running race speed
    desired_gap  = 3.0    # [m] gap to hold behind the opponent
    detect_range = 6.0    # [m] start reacting within this distance
    kp           = 5.0    # P gain on the gap error
    kd           = 2.0    # D gain on the closing speed
    max_speed    = 6.0    # [m/s] absolute speed cap

    speed = base_speed



    ### Write your code ####





    return speed


# ===========================================================================
#  GEOMETRY HELPER (provided)
# ===========================================================================
def geom_psi_kappa(x: np.ndarray, y: np.ndarray):
    """Heading & signed curvature of a non-closed (x, y) sequence."""
    n = len(x)
    psi = np.zeros(n)
    kappa = np.zeros(n)
    for i in range(n):
        if i == 0:
            dx = x[1] - x[0]; dy = y[1] - y[0]
        elif i == n - 1:
            dx = x[-1] - x[-2]; dy = y[-1] - y[-2]
        else:
            dx = (x[i + 1] - x[i - 1]) * 0.5
            dy = (y[i + 1] - y[i - 1]) * 0.5
        psi[i] = math.atan2(dy, dx)
        if 0 < i < n - 1:
            ddx = x[i + 1] - 2 * x[i] + x[i - 1]
            ddy = y[i + 1] - 2 * y[i] + y[i - 1]
            denom = (dx * dx + dy * dy) ** 1.5
            kappa[i] = (dx * ddy - dy * ddx) / max(denom, 1e-9)
    return psi, kappa


# ===========================================================================
#  ROS 2 NODE
# ===========================================================================

class LocalPlanning(Node):

    def __init__(self):
        super().__init__('local_planning')

        gp = lambda name, val: self.declare_parameter(name, val).value

        # ---- mode ('free' | 'trailing' | 'spline_avoid') --------------------
        self.mode = str(gp('mode', 'spline_avoid'))

        # ---- topics ---------------------------------------------------------
        self.scan_topic   = str(gp('scan_topic',   '/scan'))
        self.odom_topic   = str(gp('odom_topic',   '/vesc/odom'))
        self.global_topic = str(gp('global_topic', '/global_waypoints'))
        self.local_topic  = str(gp('local_topic',  '/local_waypoints'))

        # ---- Frenet slice ---------------------------------------------------
        self.local_horizon = float(gp('local_horizon', 5.0))    # [m]
        self.ds_step       = float(gp('ds_step',        0.25))  # [m]
        # ego_d -> target cosine blend distance, ~ 2 * pp_lookahead
        self.s_blend       = float(gp('s_blend',        3.0))   # [m]

        # ---- avoidance (spline_avoid) ---------------------------------------
        self.d_safe        = float(gp('d_safe',       0.7))     # [m] lateral offset
        self.s_in          = float(gp('s_in',         2.0))     # [m] min approach gap
        self.s_out         = float(gp('s_out',        2.5))     # [m] peak -> raceline
        self.trigger_range = float(gp('trigger_range', 8.0))    # [m] forward trigger range
        self.margin        = float(gp('margin',       0.2))     # [m] wall/obstacle margin
        self.obs_radius    = float(gp('obs_radius',   0.4))     # [m] obstacle inflate
        self.track_half_w  = float(gp('track_half_w', 0.8))     # [m] fallback half-width
        self.a_lat_max     = float(gp('a_lat_max',    6.0))     # [m/s^2] lat accel cap
        self.vx_scale_avoid = float(gp('vx_scale_avoid', 0.5))  # avoidance vx multiplier
        # opponents whose tracked |v| exceeds this are treated as dynamic and
        # routed to trailing only (no spline avoidance).
        self.dyn_speed_thresh = float(gp('dyn_speed_thresh', 0.5))  # [m/s]

        # ---- wall clamping (per-sample clamp + PCHIP refit) -----------------
        self.clamp_to_walls = bool(gp('clamp_to_walls', True))
        self.clamp_buffer   = float(gp('clamp_buffer',  0.05))     # [m]

        # ---- Frenet state ---------------------------------------------------
        self._sx = None
        self._sy = None
        self._vx_pchip = None
        self._dl_pchip = None
        self._dr_pchip = None
        self._s_samples = None
        self._x_samples = None
        self._y_samples = None
        self.s_total = 0.0

        # ---- perception / pose state ----------------------------------------
        self.track = None
        self.last_scan_t = None
        self.ex = self.ey = self.eyaw = 0.0
        self.ev = 0.0
        self.have_pose = False
        self.ego_s = 0.0
        self.ego_d = 0.0

        # ---- avoidance commit (hysteresis) ----------------------------------
        # Hold the same spline until ego passes s_d (raceline rejoin point).
        self._avoid_state = None

        # ---- ROS interfaces -------------------------------------------------
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(WpntArray, self.global_topic, self._global_wp_cb, latched)
        self.create_subscription(Odometry,  self.odom_topic,   self._odom_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic,   self._scan_cb, 10)

        self.local_pub  = self.create_publisher(WpntArray,   self.local_topic, latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/local_waypoints/markers', 10)
        self.cand_pub   = self.create_publisher(MarkerArray, '/local_planning/candidates', 5)

        self.get_logger().info(
            f'local_planning up | mode={self.mode} | horizon={self.local_horizon} m | '
            f'global={self.global_topic} -> local={self.local_topic}')

    # ================================================================== #
    # ROS callbacks
    # ================================================================== #
    def _global_wp_cb(self, msg):
        self._build_frenet_spline(list(msg.wpnts))

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.ex, self.ey = p.x, p.y
        self.eyaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.ev   = msg.twist.twist.linear.x
        self.have_pose = True

    def _scan_cb(self, msg):
        if not self.have_pose or self._sx is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.05 if self.last_scan_t is None else max(now - self.last_scan_t, 1e-3)
        self.last_scan_t = now

        # 1) perception
        ranges = np.asarray(msg.ranges, dtype=float)
        x, y = scan_to_xy(ranges, msg.angle_min, msg.angle_increment)
        clusters_xy = cluster(x, y, msg.angle_increment)
        obstacles = l_shape_fitting(clusters_xy)
        ego = (self.ex, self.ey, self.eyaw)
        self.track = tracking(obstacles, self.track, dt, ego)

        # 2) ego frenet
        self.ego_s, self.ego_d = self.to_frenet(self.ex, self.ey)

        # 3) mode branch -> WpntArray
        if self.mode == 'free':
            out, used_mode = self._build_passthrough(), 'free'
        elif self.mode == 'trailing':
            out, used_mode = self._build_trailing(), 'trailing'
        elif self.mode == 'spline_avoid':
            out, used_mode = self._build_spline_avoid_or_fallback()
        else:
            self.get_logger().warn(f"unknown mode '{self.mode}' -> free")
            out, used_mode = self._build_passthrough(), 'free'

        self.local_pub.publish(out)
        self._publish_local_markers(out, used_mode)

    # ================================================================== #
    # Frenet spline
    # ================================================================== #
    def _build_frenet_spline(self, wpnts):
        if len(wpnts) < 4:
            return
        x  = np.array([w.x_m     for w in wpnts], dtype=float)
        y  = np.array([w.y_m     for w in wpnts], dtype=float)
        vx = np.array([w.vx_mps  for w in wpnts], dtype=float)
        dl = np.array([w.d_left  for w in wpnts], dtype=float)
        dr = np.array([w.d_right for w in wpnts], dtype=float)

        if abs(x[0] - x[-1]) > 1e-6 or abs(y[0] - y[-1]) > 1e-6:
            x  = np.append(x,  x[0])
            y  = np.append(y,  y[0])
            vx = np.append(vx, vx[0])
            dl = np.append(dl, dl[0])
            dr = np.append(dr, dr[0])

        ds = np.hypot(np.diff(x), np.diff(y))
        s  = np.concatenate(([0.0], np.cumsum(ds)))

        keep = np.concatenate(([True], np.diff(s) > 1e-9))
        s = s[keep]; x = x[keep]; y = y[keep]
        vx = vx[keep]; dl = dl[keep]; dr = dr[keep]

        # fallback if d_left / d_right are all zero in the CSV
        if np.all(dl < 1e-3):
            dl = np.full_like(dl, self.track_half_w)
        if np.all(dr < 1e-3):
            dr = np.full_like(dr, self.track_half_w)

        self._sx = CubicSpline(s, x, bc_type='periodic')
        self._sy = CubicSpline(s, y, bc_type='periodic')
        self._vx_pchip = PchipInterpolator(s, vx, extrapolate=False)
        self._dl_pchip = PchipInterpolator(s, dl, extrapolate=False)
        self._dr_pchip = PchipInterpolator(s, dr, extrapolate=False)

        self._s_samples = s[:-1]
        self._x_samples = x[:-1]
        self._y_samples = y[:-1]
        self.s_total = float(s[-1])
        self.get_logger().info(
            f'frenet spline built (s_total={self.s_total:.3f} m, N={len(self._s_samples)})')

    def _psi_kappa_at(self, s):
        s = s % self.s_total
        dx  = float(self._sx(s, 1)); dy  = float(self._sy(s, 1))
        ddx = float(self._sx(s, 2)); ddy = float(self._sy(s, 2))
        psi = math.atan2(dy, dx)
        denom = (dx * dx + dy * dy) ** 1.5
        kappa = (dx * ddy - dy * ddx) / denom if denom > 1e-12 else 0.0
        return psi, kappa

    def _vx_at(self, s):
        return float(self._vx_pchip(s % self.s_total))

    def _dl_at(self, s):
        return float(self._dl_pchip(s % self.s_total))

    def _dr_at(self, s):
        return float(self._dr_pchip(s % self.s_total))

    def to_frenet(self, x, y, n_newton=5):
        if self._sx is None:
            return 0.0, 0.0
        i = int(np.argmin((self._x_samples - x) ** 2 + (self._y_samples - y) ** 2))
        s = float(self._s_samples[i])
        for _ in range(n_newton):
            rx = float(self._sx(s)) - x
            ry = float(self._sy(s)) - y
            dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
            ddx = float(self._sx(s, 2)); ddy = float(self._sy(s, 2))
            g  = rx * dx + ry * dy
            gp = dx * dx + dy * dy + rx * ddx + ry * ddy
            if abs(gp) < 1e-12:
                break
            s = (s - g / gp) % self.s_total
        dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
        nrm = math.hypot(dx, dy)
        if nrm < 1e-12:
            return s, 0.0
        nx, ny = -dy / nrm, dx / nrm
        d = (x - float(self._sx(s))) * nx + (y - float(self._sy(s))) * ny
        return s, d

    def to_cartesian(self, s, d):
        if self._sx is None:
            return 0.0, 0.0
        s = s % self.s_total
        x0 = float(self._sx(s)); y0 = float(self._sy(s))
        dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
        nrm = math.hypot(dx, dy)
        if nrm < 1e-12:
            return x0, y0
        nx, ny = -dy / nrm, dx / nrm
        return x0 + d * nx, y0 + d * ny

    # ================================================================== #
    # Builders
    # ================================================================== #
    def _empty_header(self):
        out = WpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        return out

    def _n_pts(self):
        return max(2, int(self.local_horizon / self.ds_step) + 1)

    def _make_local_wpnts(self, target_fn, v_cap=None, use_curvature_cap=False,
                          vx_scale=1.0, use_blend=True):
        """Unified builder: target d(s) + (optional) ego cosine blend + vx.

        With use_blend=True (default, raceline/trailing):
            d(s) = target_fn(s) + (ego_d - target_fn(ego_s)) * cos_alpha(s_off)
        With use_blend=False (spline_avoid): d(s) = target_fn(s) — the
        precomputed avoidance spline is published as-is, so what PP sees
        matches the candidate visualization.

        vx = min( raceline PCHIP vx,
                  v_cap (if given, scalar from trailing),
                  sqrt(a_lat_max / |kappa|) (if use_curvature_cap) ).
        """
        n = self._n_pts()
        if use_blend:
            target_at_ego = float(target_fn(self.ego_s))
            delta = self.ego_d - target_at_ego
        else:
            delta = 0.0

        # (s, d) sequence
        s_arr = np.empty(n)
        d_arr = np.empty(n)
        for k in range(n):
            s = (self.ego_s + k * self.ds_step) % self.s_total
            d_target = float(target_fn(s))
            if not use_blend:
                d = d_target
            else:
                s_off = k * self.ds_step
                if s_off >= self.s_blend:
                    d = d_target
                else:
                    alpha = 0.5 * (1.0 + math.cos(math.pi * s_off / self.s_blend))
                    d = d_target + delta * alpha
            s_arr[k] = s
            d_arr[k] = d

        # cartesian
        xs = np.empty(n)
        ys = np.empty(n)
        for k in range(n):
            xs[k], ys[k] = self.to_cartesian(s_arr[k], d_arr[k])

        # vx
        v_base = np.array([self._vx_at(s_arr[k]) for k in range(n)])
        vx = v_base.copy()
        if v_cap is not None:
            vx = np.minimum(vx, float(v_cap))
        if use_curvature_cap:
            _, kappa_g = geom_psi_kappa(xs, ys)
            v_curv = np.sqrt(self.a_lat_max / np.maximum(np.abs(kappa_g), 1e-6))
            vx = np.minimum(vx, v_curv)
        if vx_scale != 1.0:
            vx = vx * float(vx_scale)

        # build Wpnt array
        out = self._empty_header()
        for k in range(n):
            psi, kp = self._psi_kappa_at(s_arr[k])
            w = Wpnt()
            w.id          = int(k)
            w.s_m         = float(s_arr[k])
            w.d_m         = float(d_arr[k])
            w.x_m         = float(xs[k])
            w.y_m         = float(ys[k])
            w.psi_rad     = float(psi)
            w.kappa_radpm = float(kp)
            w.vx_mps      = float(vx[k])
            w.ax_mps2     = 0.0
            w.d_right     = 0.0
            w.d_left      = 0.0
            out.wpnts.append(w)
        return out

    def _avoid_d_at(self, s, st):
        """Evaluate the committed avoidance cubic spline at s (wrap-safe)."""
        s_abs = st['ego_s_init'] + (s - st['ego_s_init']) % self.s_total
        if s_abs <= st['ego_s_init']:
            return st['ego_d_init']
        if s_abs >= st['s_d']:
            return 0.0
        return float(st['cs'](s_abs))

    def _build_passthrough(self):
        """Raceline (d_target = 0) with ego cosine blend."""
        return self._make_local_wpnts(target_fn=lambda s: 0.0)

    def _build_trailing(self):
        """Passthrough + trailing PD speed cap."""
        ego = (self.ex, self.ey, self.eyaw)
        v_cap = trailing(self.track, ego, self.ev)
        return self._make_local_wpnts(target_fn=lambda s: 0.0, v_cap=v_cap)

    def _build_from_avoid_state(self, st):
        """Publish the committed avoidance spline as-is (no ego blend)."""
        return self._make_local_wpnts(
            target_fn=lambda s: self._avoid_d_at(s, st),
            use_curvature_cap=True,
            vx_scale=self.vx_scale_avoid,
            use_blend=False)

    def _build_spline_avoid_or_fallback(self):
        """Spline avoidance with hysteresis and trailing fallback.

        (A) avoidance committed -> hold until ego passes s_d
        (B) not committed -> trigger check then try a new avoidance
              not in front / off track  -> passthrough
              gap < s_in                -> trailing
              both candidates infeasible-> trailing
              otherwise                 -> commit best candidate
        """
        # (A) committed
        if self._avoid_state is not None:
            st = self._avoid_state
            ahead = (st['s_d'] - self.ego_s) % self.s_total
            if ahead > self.s_total / 2:   # passed
                self.get_logger().info(
                    f"avoidance done (s_d={st['s_d']:.2f}, ego_s={self.ego_s:.2f}) "
                    f"-> raceline")
                self._avoid_state = None
                self._clear_candidates()
            else:
                # Keep the committed candidate visible until ego passes s_d.
                return self._build_from_avoid_state(st), 'spline_avoid'

        # (B) not committed -> try new avoidance
        if self.track is None:
            self._clear_candidates()
            return self._build_passthrough(), 'free'

        ox, oy, vx_obs, vy_obs = self.track[0]
        opp_speed = math.hypot(vx_obs, vy_obs)
        s_obs, d_obs = self.to_frenet(ox, oy)
        gap = (s_obs - self.ego_s) % self.s_total

        dl_obs = self._dl_at(s_obs)
        dr_obs = self._dr_at(s_obs)
        in_front = 0.0 < gap < self.trigger_range
        on_track = -dr_obs < d_obs < dl_obs
        if not (in_front and on_track):
            self._clear_candidates()
            return self._build_passthrough(), 'free'

        # dynamic opponent -> trailing only, never commit a spline avoidance
        if opp_speed > self.dyn_speed_thresh:
            self._clear_candidates()
            return self._build_trailing(), 'trailing'

        if gap < self.s_in:
            self.get_logger().warn(
                f'gap={gap:.2f} < s_in={self.s_in:.2f} -> trailing fallback')
            self._clear_candidates()
            return self._build_trailing(), 'trailing'

        # evaluate left/right candidates
        s_obs_rel = self.ego_s + gap   # monotone (wrap-aware)
        cand_left  = self._make_avoidance_state(s_obs_rel, +self.d_safe, 'left',  d_obs)
        cand_right = self._make_avoidance_state(s_obs_rel, -self.d_safe, 'right', d_obs)

        results = []
        for st in (cand_left, cand_right):
            if st is None:
                continue
            cost = self._evaluate_state(st, ox, oy)
            results.append({'state': st, 'cost': cost})
        self._publish_candidates(results)

        feasible = [r for r in results if r['cost'] is not None]
        if not feasible:
            self.get_logger().warn(
                'both left/right candidates infeasible -> trailing fallback')
            return self._build_trailing(), 'trailing'

        best = min(feasible, key=lambda r: r['cost'])
        self._avoid_state = best['state']  # commit
        self.get_logger().info(
            f"avoidance start: '{best['state']['label']}' "
            f"(cost={best['cost']:.3f}, s_d={best['state']['s_d']:.2f})")
        return self._build_from_avoid_state(self._avoid_state), 'spline_avoid'

    # ================================================================== #
    # Avoidance state + feasibility
    # ================================================================== #
    def _make_avoidance_state(self, s_obs_rel, d_avoid, label, d_obs=0.0):
        """3-ctrl cubic spline, then push samples out of obstacle + clamp to
        walls + PCHIP refit.

        ctrl points: (ego_s_init, ego_d_init), (s_obs_rel, d_avoid),
                     (s_obs_rel + s_out, 0).
        s_d = last ctrl = commit termination check point.

        Post-processing (clamp_to_walls):
          1. dense-sample the raw cubic over [ego_s_init, s_d]
          2. push d laterally so |d - d_obs| >= sqrt(safety^2 - (s - s_obs)^2)
             on the side selected by sign(d_avoid)        (obstacle clearance)
          3. clamp each d into [-dr(s)+margin+buffer, dl(s)-margin-buffer]
             (wall clearance — final, so walls win over obstacle push if they
             collide; _evaluate_state then catches that as infeasible)
          4. refit with PCHIP (shape-preserving, no overshoot)
        """
        ego_s_init = float(self.ego_s)
        ego_d_init = float(self.ego_d)
        s_d = s_obs_rel + self.s_out
        s_ctrl = np.array([ego_s_init, s_obs_rel, s_d], dtype=float)
        d_ctrl = np.array([ego_d_init, float(d_avoid), 0.0], dtype=float)
        if not np.all(np.diff(s_ctrl) > 1e-3):
            return None
        cs_raw = CubicSpline(s_ctrl, d_ctrl, bc_type='natural')

        if self.clamp_to_walls:
            n_clamp = 40
            s_seq = np.linspace(ego_s_init, s_d, n_clamp)
            d_seq = np.asarray(cs_raw(s_seq), dtype=float)
            inset  = self.margin + self.clamp_buffer
            safety = self.obs_radius + self.margin
            side = 1.0 if d_avoid > 0 else -1.0
            for i, s in enumerate(s_seq):
                d = d_seq[i]
                # 1) push outward of obstacle within its s-influence band
                ds_obs = s - s_obs_rel
                if abs(ds_obs) < safety:
                    lat_needed = math.sqrt(safety * safety - ds_obs * ds_obs)
                    target = d_obs + side * lat_needed
                    if side > 0:
                        d = max(d, target)
                    else:
                        d = min(d, target)
                # 2) wall clamp (final authority)
                dl =  self._dl_at(s) - inset
                dr = -self._dr_at(s) + inset
                if dl < dr:                       # corridor narrower than 2*inset
                    d = 0.5 * (dl + dr)
                else:
                    d = min(max(d, dr), dl)
                d_seq[i] = d
            cs = PchipInterpolator(s_seq, d_seq, extrapolate=False)
        else:
            cs = cs_raw

        return {
            'label':       label,
            'd_avoid':     float(d_avoid),
            's_d':         float(s_d),
            'ego_s_init':  ego_s_init,
            'ego_d_init':  ego_d_init,
            'cs':          cs,
        }

    def _sample_state_full(self, st, n=60):
        """Dense sampling over [ego_s_init, s_d] for feasibility & viz."""
        s_seq = np.linspace(st['ego_s_init'], st['s_d'], n)
        d_seq = np.array([self._avoid_d_at(s, st) for s in s_seq])
        return s_seq, d_seq

    def _evaluate_state(self, st, obs_x, obs_y):
        """Wall + obstacle feasibility + cost. Returns cost or None (infeasible)."""
        s_seq, d_seq = self._sample_state_full(st)
        # 1) track width (both walls + margin)
        for s, d in zip(s_seq, d_seq):
            dl = self._dl_at(s) - self.margin
            dr = -self._dr_at(s) + self.margin
            if d > dl or d < dr:
                return None
        # 2) min distance to obstacle (must clear inflated safety distance)
        xs = np.empty_like(s_seq)
        ys = np.empty_like(s_seq)
        for k, (s, d) in enumerate(zip(s_seq, d_seq)):
            xs[k], ys[k] = self.to_cartesian(s, d)
        obs_dist = float(np.min(np.hypot(xs - obs_x, ys - obs_y)))
        safety = self.obs_radius + self.margin
        if obs_dist < safety:
            return None
        # 3) cost: 1/clearance + mean |d|
        w_obs    = 5.0
        w_offset = 1.0
        return (w_obs / max(obs_dist - safety, 0.05)
                + w_offset * float(np.mean(np.abs(d_seq))))

    # ================================================================== #
    # Visualization
    # ================================================================== #
    def _publish_local_markers(self, wpnts, mode):
        color = (0.3, 0.8, 1.0)   # sky blue (always)
        ma = MarkerArray()
        line = Marker()
        line.header.frame_id = 'map'
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = 'local_waypoints_line'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.08
        line.color.r, line.color.g, line.color.b, line.color.a = (*color, 1.0)
        # z above candidate (0.07) and global raceline so the local path
        # always draws on top in RViz.
        for w in wpnts.wpnts:
            p = Point()
            p.x, p.y, p.z = float(w.x_m), float(w.y_m), 0.20
            line.points.append(p)
        ma.markers.append(line)
        self.marker_pub.publish(ma)

    def _clear_candidates(self):
        """Publish DELETEALL to wipe leftover candidate markers."""
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        self.cand_pub.publish(ma)

    def _publish_candidates(self, results):
        """Visualize [{'state': st, 'cost': float|None}, ...]."""
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        feasible = [r for r in results if r['cost'] is not None]
        best_cost = min((r['cost'] for r in feasible), default=None)
        stamp = self.get_clock().now().to_msg()
        for i, r in enumerate(results):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'avoid_candidates'
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            if r['cost'] is None:
                # infeasible
                m.scale.x = 0.03
                m.color.r, m.color.g, m.color.b, m.color.a = 0.7, 0.0, 0.0, 0.4
            elif best_cost is not None and r['cost'] == best_cost:
                # best
                m.scale.x = 0.08
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.95
            else:
                # feasible but not best
                m.scale.x = 0.03
                m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.55, 0.55, 0.6
            s_seq, d_seq = self._sample_state_full(r['state'])
            for s, d in zip(s_seq, d_seq):
                x, y = self.to_cartesian(s, d)
                p = Point(); p.x, p.y, p.z = float(x), float(y), 0.07
                m.points.append(p)
            # lifetime=0 -> persist in RViz until explicit DELETEALL.
            # We only clear when ego passes s_d (avoidance complete).
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0
            ma.markers.append(m)
        self.cand_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanning()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
