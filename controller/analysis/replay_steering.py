#!/usr/bin/env python3
"""Replay a recorded run through the REAL Controller and price steering changes offline.

Why this exists: run_ggv_0812 measured the car demanding p90 14.36 / max 22.37 m/s^2 of lateral
acceleration from a tyre that was never recorded delivering more than 11.71, while the published
raceline only ever asks for 5.70. That is a controller problem, not a ggv one. This harness
answers "what would the commands have been under change X" without a car and without a build.
(An earlier version of this note claimed a saturation knee at 6.0-6.5 from an achieved/demanded
ratio. That came from a replay silently covering 16.3% of the bag; at full coverage the ratio
falls but the achieved value keeps climbing, and there is no knee. See GATE_COVERAGE.)

It drives `combined.src.Controller` ITSELF, not a reimplementation of it: the class is
deliberately ROS-node-free (numpy + injected markers/loggers), so the same code that flies
on the car runs here. Variants subclass it and override one method each.

THE GATE COMES FIRST. A replay that cannot reproduce the RECORDED steering command from the
recorded inputs cannot predict what an alternative would have done, so `--check` fails the
run unless baseline fidelity clears its thresholds. Read a variant table only after it passes.

Known approximations, all of which the gate is what tests:
  * local waypoints were not recorded, so the global line is windowed the way the state machine
    windows it (local_window: 80 points from the nearest index, wrapping). Valid only for clean
    laps -- no avoidance, no reopt swap -- and the bag this was written for is exactly that.
  * `state` is assumed GB_TRACK; /state_machine was not recorded.
  * the FrenetConverter is a nearest-point projection onto the same global line.

Usage:
    source /opt/ros/jazzy/setup.bash && source ~/unicorn_ws/install/setup.bash
    python3 controller/analysis/replay_steering.py --bag ~/ggv_0812_1645 [--check]
    ...                                            --sweep     # price a_lat_margin ceilings
    ...                                            --section   # per-station table, A / E / A+E
"""
import argparse
import collections
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "controller"))
from combined.src.Controller import Controller, load_veh_dyn  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CFG = os.path.join(REPO, "stack_master", "config", "controller.yaml")
VEH = os.path.join(REPO, "stack_master", "config", "CAR")
GGV = os.path.join(VEH, "veh_dyn_info", "ggv.csv")

# The section this was written for: ifac_0807's right-hand slalom. First corner s 4.3-5.8
# (kappa peaks -0.27 at s 5.2), short straight s 7.5-9.5.
SEC_LO, SEC_HI, SEC_STEP = 2.8, 10.0, 0.6

# Fidelity thresholds -- a replay looser than this is not evidence about anything.
# RE-MEASURED after the rolling-window fix, on the FULL bag instead of its first lap: the
# baseline reproduces the record at corr +0.948 / rms 0.0513 rad, against +0.959 / 0.0284 over
# the first 8.2 s. Fidelity is worse because the window now contains the hard part of the lap,
# not because the replay is worse. THESE BARS ARE UNCHANGED: the pre-existing 0.90 / 0.060
# survives the re-measurement, so there was nothing to choose -- moving them to fit the new
# number is the one thing a gate must not do. What DID change is the headroom, from 2.11x the
# rms bar to 1.17x, which is the honest cost of measuring the whole bag; a controller tuning
# change that would once have passed unnoticed can now trip this.
GATE_CORR, GATE_RMS = 0.90, 0.060          # [-], [rad]
# A replay that silently stops producing numbers is the failure this gate exists for: before the
# rolling window, 83.7% of every run was NaN and nothing said so.
GATE_COVERAGE = 0.95                       # [-] finite steering samples / total


# ----------------------------------------------------------------------------- bag reading
def read_bag(path):
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message

    want = {"/car_state/odom", "/car_state/odom_frenet", "/vesc/odom",
            "/vesc/sensors/imu/raw", "/vesc/high_level/ackermann_cmd",
            "/global_waypoints_scaled"}
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    missing = want - set(types)
    if missing:
        sys.exit(f"bag is missing required topics: {sorted(missing)}")
    out = {k: [] for k in want}
    wpnts = None
    while r.has_next():
        topic, data, t_ns = r.read_next()
        if topic not in want:
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        t = t_ns / 1e9
        if topic == "/car_state/odom":
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y ** 2 + q.z ** 2))
            out[topic].append((t, p.x, p.y, yaw))
        elif topic == "/car_state/odom_frenet":
            p = msg.pose.pose.position
            out[topic].append((t, p.x, p.y))
        elif topic == "/vesc/odom":
            out[topic].append((t, msg.twist.twist.linear.x))
        elif topic == "/vesc/sensors/imu/raw":
            a, g = msg.linear_acceleration, msg.angular_velocity
            out[topic].append((t, a.x, a.y, g.z))
        elif topic == "/vesc/high_level/ackermann_cmd":
            out[topic].append((t, msg.drive.speed, msg.drive.steering_angle))
        elif topic == "/global_waypoints_scaled" and wpnts is None:
            # columns the manager builds in behavior_cb, in its order
            wpnts = np.array([[w.x_m, w.y_m, w.vx_mps,
                               (min(w.d_left, w.d_right) / (w.d_right + w.d_left)
                                if (w.d_right + w.d_left) != 0 else 0.0),
                               w.s_m, w.kappa_radpm, w.psi_rad, w.ax_mps2, w.d_m]
                              for w in msg.wpnts])
    if wpnts is None:
        sys.exit("no /global_waypoints_scaled in the bag")
    return {k: np.array(v) for k, v in out.items()}, wpnts


class NearestFrenet:
    """Only `get_frenet` is used by the Controller; a projection onto the same line is it."""

    def __init__(self, wpnts):
        self.xy = wpnts[:, :2]
        self.s = wpnts[:, 4]
        t = np.gradient(self.xy, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        self.nv = np.column_stack([-t[:, 1], t[:, 0]])   # +d to the LEFT, stack convention

    def get_frenet(self, xs, ys):
        s_out, d_out = [], []
        for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
            i = int(np.argmin((self.xy[:, 0] - x) ** 2 + (self.xy[:, 1] - y) ** 2))
            v = np.array([x, y]) - self.xy[i]
            s_out.append(self.s[i])
            d_out.append(float(v @ self.nv[i]))
        return np.array(s_out), np.array(d_out)


# --------------------------------------------------------------------------------- variants
class VarA(Controller):
    """A: tell the controller the tyre limit. delta_max = atan(L*a_lat_max / v^2)."""

    a_lat_max = 5.7

    def calc_steering_angle_for_future(self, *a, **k):
        st = super().calc_steering_angle_for_future(*a, **k)
        v = max(float(self.speed_now), 0.5)
        lim = np.arctan(self.wheelbase * self.a_lat_max / (v * v))
        self.curr_steering_angle = float(np.clip(st, -lim, lim))
        return self.curr_steering_angle


class VarB(Controller):
    """B: drop the unbounded 2**lat_err multiplier (lat_err is already inside eta)."""

    def steer_scaling_for_lat_err(self, steer, lateral_error):
        return steer


class VarAB(VarA, VarB):
    pass


class VarC(Controller):
    """C: raise the L1 floor. t_clip_min is set on the instance, not overridden here."""


class VarD(Controller):
    """D: make the per-cycle steering rate limit real (the shipped 0.4 rad/cycle is ~20 rad/s)."""

    rate_limit_rad_s = 6.0

    def calc_steering_angle_for_future(self, *a, **k):
        prev = float(self.curr_steering_angle)
        st = super().calc_steering_angle_for_future(*a, **k)
        step = self.rate_limit_rad_s / self.loop_rate
        self.curr_steering_angle = float(np.clip(st, prev - step, prev + step))
        return self.curr_steering_angle


# SHIPPED as of the tyre-limit commit: Controller clips to the ggv itself. That silently
# changes what the FIDELITY gate MEANS unless the baseline says otherwise -- the gate asks
# whether the replay reproduces a RECORDING MADE BY THE OLD CODE, so the baseline row (and the
# historical variant rows, which were measured against it) must run with the shipped clip OFF.
# Measured when this was missed: corr +0.959 -> +0.932 and rms 0.0284 -> 0.0319, i.e. worse but
# still inside the thresholds, so it passed quietly. `OFF` is that pin; the shipped
# configuration gets its own row, and the margin sweep its own table.
OFF = {"a_lat_limit_enable": False, "a_comb_limit_enable": False}

# Ceilings to price, as multiples of the ggv's ay_max. 1.00 is in the list because it is the
# one that SHIPPED AND REGRESSED (the raceline's own worst demand is 5.70 = 1.00x, so at 1.00
# the corner-apex recovery budget is exactly zero).
SWEEP_MARGINS = (1.00, 1.05, 1.10, 1.15, 1.25, 1.35)


def achieved_a_lat(D, t):
    """The lateral acceleration the car ACTUALLY made, from the record: v * |yaw rate|.

    Not the IMU's lateral channel: v*|yaw_rate| is the curvature the car actually drove, which
    is the quantity a steering ceiling can take away. Cross-checked against the independently
    measured "the tyre exceeded 6.27 for 29.7% of the run" -- this definition gives 29.9%, the
    IMU channel gives 39.7%.
    """
    v = np.interp(t, D["/vesc/odom"][:, 0], D["/vesc/odom"][:, 1])
    gz = np.interp(t, D["/vesc/sensors/imu/raw"][:, 0], D["/vesc/sensors/imu/raw"][:, 3])
    return v * np.abs(gz)


# ----------------------------------------------------------------------------------- replay
def build(cls, params, conv, **over):
    sig = ["t_clip_min", "t_clip_max", "m_l1", "q_l1", "curvature_factor", "KP", "KI", "KD",
           "heading_error_thres", "steer_gain_for_speed", "future_constant", "speed_lookahead",
           "lat_err_coeff", "acc_scaler_for_steer", "dec_scaler_for_steer", "start_scale_speed",
           "end_scale_speed", "downscale_factor", "speed_lookahead_for_steer", "trailing_gap",
           "trailing_vel_gain", "trailing_p_gain", "trailing_i_gain", "trailing_d_gain",
           "blind_trailing_speed", "loop_rate", "wheelbase", "speed_factor_for_lat_err",
           "speed_factor_for_curvature", "speed_diff_thres", "start_speed",
           "start_curvature_factor", "AEB_thres"]
    # the manager does not read these from the yaml: loop_rate is hardcoded at
    # controller_manager.py:59, wheelbase is a _get_param default. Mirrored, not invented.
    node_defaults = {"loop_rate": 50, "wheelbase": 0.321}
    kw = {}
    for n in sig:
        if n in over:
            kw[n] = over[n]
        elif n in params:
            kw[n] = params[n]
        elif n in node_defaults:
            kw[n] = node_defaults[n]
        else:
            sys.exit(f"controller.yaml has no '{n}' and no default was given")
    c = cls(converter=conv, **kw)
    # the manager also assigns these AFTER construction (controller_manager.py:209-215);
    # setting every remaining yaml key keeps that list from going stale here.
    for k, v in params.items():
        if k not in sig and not hasattr(type(c), k):
            setattr(c, k, v)
    for k, v in over.items():
        setattr(c, k, v)
    return c


Rep = collections.namedtuple("Rep", "t steer v s rec spd fpsi fxy")


def local_window(line, idx, n):
    """What the state machine actually hands the controller, reproduced.

    state_machine/states.py:32 (GB_TRACK): n_loc_wpnts points starting at the nearest global
    index, indexed `(s + i) % num_glb_wpnts` -- a ROLLING FORWARD WINDOW THAT WRAPS at the lap
    boundary. n_loc_wpnts is 80 (state_machine_params.yaml:5), clamped by the node to
    min(80, num_glb_wpnts // 2); glb_wpnts_cb also drops the duplicate last point first.

    Handing the WHOLE line instead -- which this script did until now -- is not merely a coarser
    approximation, it breaks: as the car finishes a lap the nearest index reaches the end of the
    array, main_loop's `np.mean(W[idx+10:idx+20, 5])` averages an EMPTY slice, curvature_waypoints
    goes NaN, the steering goes NaN, and it LATCHES because next cycle's rate limit clips against
    a NaN curr_steering_angle. On ggv_0812_1645 that killed the replay at t+8.2 s of 50.4 s and
    every number this script printed came from that first lap. The s column wraps here exactly as
    it does on the car; nothing in Controller reads it.
    """
    return line[(idx + np.arange(n)) % len(line)]


def replay(cls, D, W, params, conv, n_loc=80, **over):
    cmd = D["/vesc/high_level/ackermann_cmd"]
    t = cmd[:, 0]
    ip = lambda src, col: np.interp(t, src[:, 0], src[:, col])
    x, y = ip(D["/car_state/odom"], 1), ip(D["/car_state/odom"], 2)
    yaw = np.unwrap(ip(D["/car_state/odom"], 3))
    s, d = ip(D["/car_state/odom_frenet"], 1), ip(D["/car_state/odom_frenet"], 2)
    v = ip(D["/vesc/odom"], 1)
    ax = ip(D["/vesc/sensors/imu/raw"], 1)
    gz = ip(D["/vesc/sensors/imu/raw"], 3)
    track_len = float(W[:, 4].max())
    # the SM drops the duplicate last point (its s == the loop length) before windowing
    line = W[:-1]
    n_loc = min(n_loc, len(line) // 2)

    c = build(cls, params, conv, **over)
    acc = np.zeros(10)
    out = np.full(len(t), np.nan)
    spd = np.full(len(t), np.nan)
    fpsi = np.full(len(t), np.nan)
    fxy = np.full((len(t), 2), np.nan)
    for i in range(len(t)):
        acc[1:] = acc[:-1]
        # mirrors controller_manager.imu_cb, INCLUDING its sign fix -- see the note there. Both
        # were negated on the strength of a "vesc is rotated 90 deg" comment that the bag
        # disproves (imu.x tracks dv/dt at +0.956, gz tracks dyaw/dt at +0.986).
        acc[0] = ax[i]
        c.speed_now = float(v[i])
        c.yaw_rate = float(gz[i])
        idx = int(np.argmin((line[:, 0] - x[i]) ** 2 + (line[:, 1] - y[i]) ** 2))
        try:
            sp, _, _, st, _, L1, _, _, fp = c.main_loop(
                "GB_TRACK",
                np.array([[x[i], y[i], np.arctan2(np.sin(yaw[i]), np.cos(yaw[i]))]]),
                local_window(line, idx, n_loc), float(v[i]), None,
                np.array([[s[i], d[i]]]), acc, track_len)
            out[i] = st
            spd[i] = sp
            fpsi[i], fxy[i] = fp[0, 2], fp[0, :2]
        except Exception:
            pass
    return Rep(t, out, v, s, cmd[:, 2], spd, fpsi, fxy)


def longest_run(mask):
    """Longest contiguous True stretch of `mask`, as a slice. Lag needs unbroken time."""
    best = cur = (0, 0)
    for i, m in enumerate(mask):
        cur = (cur[0], i + 1) if m else (i + 1, i + 1)
        if cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    return slice(*best)


def lag_ms(t, a, b, max_lag_s=1.0):
    """Cross-correlation peak lag of `b` behind `a`, in ms, with the peak correlation.

    DIAGNOSTIC ONLY -- nothing here is fixed by this commit. The tyre clip addresses commands
    that are too BIG; sluggishness and overshoot have a second possible cause the clip cannot
    touch, namely how late the loop reacts. Candidate contributors, none measured apart:
    future_constant = 0.05 s of prediction, the heading filter's alpha = 0.1 at 50 Hz (a 0.2 s
    time constant), speed_lookahead = 0.25 s, plus actuator and estimator delay in the record.
    A positive number means the steering command TRAILS the lateral error by that much.
    """
    dt = float(np.median(np.diff(t)))
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return float("nan"), float("nan"), dt
    n = int(round(max_lag_s / dt))
    lags = np.arange(-n, n + 1)
    cc = np.array([float(np.dot(a[max(0, -k):len(a) - max(0, k)],
                                b[max(0, k):len(b) - max(0, -k)])) for k in lags]) / denom
    i = int(np.argmax(np.abs(cc)))
    return lags[i] * dt * 1000.0, cc[i], dt


class VarEgyro(Controller):
    """E, but with the friction circle fed by the GYRO instead of the plan's curvature.

    Kept as a variant, not shipped: the gyro reports the curvature the car ACHIEVED, which in
    the understeer this limit exists for is smaller than what the path is demanding, so the
    circle opens up exactly when it should close.
    """

    def a_lat_now(self):
        return abs(float(self.speed_now) * float(self.yaw_rate))


# The four configurations that have to be read together. A and E act on different signals (A on
# the steering command, E on the speed command) and the apex fault is over-SPEED, not
# over-steer, so A alone can make the excursion worse by giving up lateral force without
# removing the cause. Nothing here is conclusive: the replay is open loop.
CONFIGS = (
    ("기준선 (둘 다 off)", {"a_lat_limit_enable": False, "a_comb_limit_enable": False}),
    ("Ⓐ only  조향 clip", {"a_lat_limit_enable": True, "a_comb_limit_enable": False}),
    ("Ⓔ only  마찰원", {"a_lat_limit_enable": False, "a_comb_limit_enable": True}),
    ("Ⓐ+Ⓔ", {"a_lat_limit_enable": True, "a_comb_limit_enable": True}),
)


def section_table(label, s, d, v, steer, spd, W, wb):
    """Per-station table over the slalom, in the same shape as the hand-measured one.

    Rows are labelled by the left edge of a 0.6 m bin and the plan columns (kappa, target vx)
    are read at that edge, which is the convention the reference table used. The measured
    columns are averaged over the bin across every lap in the bag, so `n` is samples not laps.
    """
    print(f"\n  --- {label} ---")
    print(f"  {'s':>5} {'kappa':>7} {'목표vx':>7} {'명령v':>7} {'실제v':>7} {'a_lat':>6} "
          f"{'delta':>7} {'d':>7} {'n':>5}")
    edges = np.arange(SEC_LO, SEC_HI + 1e-9, SEC_STEP)
    for a in edges[:-1]:
        m = (s >= a) & (s < a + SEC_STEP) & np.isfinite(steer)
        if not m.sum():
            continue
        kap = float(np.interp(a, W[:, 4], W[:, 5]))
        vx = float(np.interp(a, W[:, 4], W[:, 2]))
        vr = float(v[m].mean())
        print(f"  {a:5.1f} {kap:+7.2f} {vx:7.2f} {np.nanmean(spd[m]):7.2f} {vr:7.2f} "
              f"{vr * vr * abs(kap):6.2f} {np.nanmean(steer[m]):+7.2f} {d[m].mean():+7.2f} "
              f"{m.sum():5d}")


def stats(name, st, v, rec, wheelbase, gate=False):
    m = np.isfinite(st) & (v > 2.0)
    dem = v[m] ** 2 * np.abs(np.tan(st[m])) / wheelbase
    line = (f"  {name:22s} |delta| p90 {np.percentile(np.abs(st[m]), 90):5.3f}  "
            f"요구 a_lat p50 {np.median(dem):5.2f} p90 {np.percentile(dem, 90):6.2f} "
            f"max {dem.max():6.2f}  >6.0 {100*(dem > 6.0).mean():5.1f}%")
    if gate:
        r = np.corrcoef(st[m], rec[m])[0, 1]
        rms = float(np.sqrt(np.mean((st[m] - rec[m]) ** 2)))
        ok = (r >= GATE_CORR) and (rms <= GATE_RMS)
        line += f"\n  {'':22s} FIDELITY corr {r:+.3f} (>={GATE_CORR}) rms {rms:.4f} rad " \
                f"(<={GATE_RMS})  {'PASS' if ok else 'FAIL'}"
        return line, ok
    return line, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=os.path.expanduser("~/ggv_0812_1645"))
    ap.add_argument("--check", action="store_true", help="exit 1 unless the fidelity gate passes")
    ap.add_argument("--sweep", action="store_true",
                    help="price the a_lat_margin ceilings in SWEEP_MARGINS and stop")
    ap.add_argument("--section", action="store_true",
                    help=f"per-station table over s={SEC_LO}-{SEC_HI} for A / E / A+E, and stop")
    ap.add_argument("--gamma", action="store_true",
                    help="price gamma_weight (model-vs-measured future heading) and stop")
    args = ap.parse_args()

    params = yaml.safe_load(open(CFG))
    params = next(iter(params.values()))["ros__parameters"]
    ggv = np.atleast_2d(np.loadtxt(GGV, delimiter=",", comments="#"))
    ay = float(ggv[:, 2].min())
    ggv_ay = ggv[:, [0, 2]]
    D, W = read_bag(args.bag)
    conv = NearestFrenet(W)
    wb = params.get("wheelbase", 0.321)
    print(f"bag {args.bag}\ncontroller.yaml {CFG}\nggv ay_max {ay} (min over the table)\n")

    # BASELINE = the code as it was BEFORE the tyre clip, pinned by OFF. Read the note on OFF.
    r0 = replay(Controller, D, W, params, conv, **OFF)
    t, base, v, s, rec = r0.t, r0.steer, r0.v, r0.s, r0.rec
    line, ok = stats("baseline (clip OFF)", base, v, rec, wb, gate=True)
    print("=== 기준선 재현 ===")
    print(f"  recorded              |delta| p90 "
          f"{np.percentile(np.abs(rec[v > 2.0]), 90):5.3f}")
    print(line)
    print("  (기준선 = a_lat_limit_enable:false. 게이트는 clip 이 들어가기 전의 코드를 잰다)")

    # COVERAGE is a GATE, not a note. A NaN steering command is a cycle that produced no
    # evidence, and until the rolling window went in, 83.7% of every run was NaN with nothing
    # saying so -- the whole table came from the first 8.2 s of a 50.4 s bag.
    fin = int(np.isfinite(base).sum())
    cov = fin / len(t)
    cov_ok = cov >= GATE_COVERAGE
    print(f"  재생 커버리지 {fin}/{len(t)} 샘플 = {100 * cov:5.1f}%  (>= {100 * GATE_COVERAGE:.0f}%)  "
          f"{'PASS' if cov_ok else 'FAIL'}   NaN {len(t) - fin} 사이클")
    ok = ok and cov_ok
    if not cov_ok:
        print("  재생이 사이클을 통째로 잃고 있다. 표는 그만큼의 구간만 말한다.")
        if args.check:
            sys.exit(1)
    if not ok:
        print("\n  기준선이 기록을 재현하지 못했다. 아래 표는 근거가 아니다.")
        if args.check:
            sys.exit(1)

    # the veh_dyn inputs the friction circle and the slew limit need, read the way the manager
    # reads them (b_ax_machines joined on 2026-08-12 -- see Controller._slew_limits)
    ggv_ay2, ggv_ax, ax_mach, b_ax, dyn_p = load_veh_dyn(VEH)
    VEHKW = {"ggv_table": ggv_ay2, "ggv_ax_table": ggv_ax, "ax_machines_table": ax_mach,
             "b_ax_table": b_ax, "dyn_model_exp": dyn_p}

    if args.gamma:
        # gamma_weight blends the FUTURE HEADING used for the L1 lookahead:
        #     future_psi = gamma*psi_model + (1-gamma)*(psi + yaw_rate*T)
        # 1.0 (as shipped) is pure kinematic model and multiplies the measured yaw rate by zero.
        #
        # HOW TO CHOOSE, and how NOT to. `vs기록 rms` below is NOT a quality measure: the record
        # was produced by code running gamma=1.0, so any other gamma differs from it BY
        # CONSTRUCTION and 1.0 wins that column trivially. It is here as a magnitude-of-change
        # number only. The column that decides is `psi 예측오차`: the controller predicts where
        # the car will be pointing future_constant seconds from now, and the bag says where it
        # ACTUALLY pointed. That comparison does not care which code wrote the log.
        T = float(params["future_constant"])
        od = D["/car_state/odom"]
        yaw_true = np.unwrap(np.interp(t + T, od[:, 0], np.unwrap(od[:, 3])))
        x_true = np.interp(t + T, od[:, 0], od[:, 1])
        y_true = np.interp(t + T, od[:, 0], od[:, 2])
        print(f"\n=== gamma_weight 스윕 (future_constant {T} s, 예측 대상 = t+{T}s 의 실제 자세) ===")
        print(f"  {'gamma':>6} {'섞는비율':>9}   {'psi 예측오차 p50':>15} {'p90':>7}   "
              f"{'xy 오차 p50':>11}   {'요구 p50':>8} {'p90':>6} {'max':>6}  {'|d| p90':>8}  "
              f"{'vs기록 rms':>10}")
        for g in (1.0, 0.7, 0.5, 0.3, 0.0):
            rr = replay(Controller, D, W, params, conv, **VEHKW, gamma_weight=g)
            k = np.isfinite(rr.steer) & (v > 2.0)
            err = np.abs((rr.fpsi[k] - yaw_true[k] + np.pi) % (2 * np.pi) - np.pi)
            xy = np.hypot(rr.fxy[k, 0] - x_true[k], rr.fxy[k, 1] - y_true[k])
            dem = v[k] ** 2 * np.abs(np.tan(rr.steer[k])) / wb
            rms = float(np.sqrt(np.mean((rr.steer[k] - rec[k]) ** 2)))
            print(f"  {g:6.2f} {100 * (1 - g):8.0f}%   {np.degrees(np.median(err)):12.3f}deg "
                  f"{np.degrees(np.percentile(err, 90)):6.3f}   {np.median(xy):10.4f}m   "
                  f"{np.median(dem):8.2f} {np.percentile(dem, 90):6.2f} {dem.max():6.2f}  "
                  f"{np.percentile(np.abs(rr.steer[k]), 90):8.3f}  {rms:10.4f}")
        print("\n  섞는비율 = 측정 요레이트가 예측에 들어가는 비율 (1-gamma).")
        print("  psi 예측오차 = |예측한 t+T 헤딩 - 실제 t+T 헤딩|. 이것만이 어느 쪽이 맞는지 말한다.")
        print("  vs기록 rms = 기록 명령과의 차이. 기록이 gamma=1.0 으로 만들어졌으므로")
        print("               품질이 아니라 '얼마나 바뀌는가' 이다.")
        print("\n  !! 명령 열(요구/|d|/rms)이 gamma 에 대해 전부 불변인 것에 주목할 것.")
        print("     future_psi 를 읽는 곳은 viz_future_position(RViz 화살표) 하나뿐이고,")
        print("     기능 소비자는 전부 future_position[0,:2] 만 쓴다. eta 는 현재 yaw 로 계산된다.")
        print("     즉 gamma 는 지금 시각화 전용이며, 이것을 노출해도 루프는 닫히지 않는다.")

        # lambda_weight IS live: beta_fused feeds future_x/future_y, which the L1 lookahead,
        # the lateral-error term and the nearest-waypoint search all read.
        print(f"\n=== lambda_weight 스윕 (slip angle: 모델 vs IMU 요레이트 유도) ===")
        print(f"  {'lambda':>6} {'섞는비율':>9}   {'xy 예측오차 p50':>14} {'p90':>8}   "
              f"{'요구 p50':>8} {'p90':>6} {'max':>6}  {'|d| p90':>8}  {'vs기록 rms':>10}")
        for lam in (1.0, 0.7, 0.5, 0.3, 0.0):
            rr = replay(Controller, D, W, params, conv, **VEHKW, lambda_weight=lam)
            k = np.isfinite(rr.steer) & (v > 2.0)
            xy = np.hypot(rr.fxy[k, 0] - x_true[k], rr.fxy[k, 1] - y_true[k])
            dem = v[k] ** 2 * np.abs(np.tan(rr.steer[k])) / wb
            rms = float(np.sqrt(np.mean((rr.steer[k] - rec[k]) ** 2)))
            print(f"  {lam:6.2f} {100 * (1 - lam):8.0f}%   {np.median(xy):11.4f}m "
                  f"{np.percentile(xy, 90):7.4f}m   {np.median(dem):8.2f} "
                  f"{np.percentile(dem, 90):6.2f} {dem.max():6.2f}  "
                  f"{np.percentile(np.abs(rr.steer[k]), 90):8.3f}  {rms:10.4f}")
        print("\n  beta_imu 는 |v| > 2.0 에서만 계산된다(그 아래는 beta_model 로 대체).")
        sys.exit(0)

    if args.section:
        dcol = np.interp(t, D["/car_state/odom_frenet"][:, 0], D["/car_state/odom_frenet"][:, 2])
        print(f"\n=== 구간 표  s={SEC_LO}-{SEC_HI} m, {SEC_STEP} m 간격  "
              f"(ax_max {ggv_ax[:, 1].max():.1f}, ay_max {ay}, p {dyn_p}) ===")
        print(f"  목표vx = 레이스라인 프로파일 | 명령v = 재생된 속도 명령 | 실제v = 기록된 속도")
        print(f"  a_lat = 실제v^2 * |kappa| (레이스라인 kappa)")
        for label, cfg in CONFIGS:
            rr = replay(Controller, D, W, params, conv, **{**VEHKW, **cfg})
            section_table(label, rr.s, dcol, rr.v, rr.steer, rr.spd, W, wb)
        # the same section for the gyro-fed circle, to answer "plan kappa or measured?"
        rr = replay(VarEgyro, D, W, params, conv,
                    **{**VEHKW, "a_lat_limit_enable": False, "a_comb_limit_enable": True})
        section_table("Ⓔ (자이로 a_lat)", rr.s, dcol, rr.v, rr.steer, rr.spd, W, wb)
        print(f"\n  개루프다: '정점에서 명령v 가 오르지 않는다'는 보이지만 "
              f"'그래서 d 가 줄어든다'는 보이지 않는다.")
        sys.exit(0)

    if args.sweep:
        # The ceiling is a_lat_margin * ggv ay_max. Read this table before choosing a default.
        # The raceline's own worst demand is 5.70 = 1.00x, so the corner-apex recovery budget a
        # margin buys is (margin - 1.00) * 5.7 m/s^2 -- at 1.00 it is exactly zero, which is the
        # setting that shipped and came back as "far too sluggish". The tyre knee at 6.0-6.5 is
        # where buying more stops meaning anything, and ">6.0" is how much of the run is spent
        # past it.
        ach = achieved_a_lat(D, t)
        mv = v > 2.0
        print(f"\n=== a_lat_margin 스윕 (ceiling = margin x ggv ay_max {ay}) ===")
        print(f"  {'margin':>6} {'ceiling':>8} {'복귀예산':>9}  {'잘라내는량':>10}   "
              f"{'p50':>5} {'p90':>6} {'max':>6}  {'|delta| p90':>11}")
        for m in SWEEP_MARGINS:
            # E is LIVE here: the steering ceiling has to be chosen against the speeds the car
            # will actually carry once the friction circle is in, not the ones it carried before.
            st = replay(Controller, D, W, params, conv,
                        **VEHKW, a_lat_limit_enable=True, a_lat_margin=m,
                        a_comb_limit_enable=True).steer
            k = np.isfinite(st) & mv
            dem = v[k] ** 2 * np.abs(np.tan(st[k])) / wb
            # WHAT THIS CEILING GIVES UP: how much of the run the car MEASURABLY made more
            # lateral acceleration than the ceiling would have allowed it to ask for.
            cut = 100 * (ach[mv] > m * ay).mean()
            print(f"  {m:6.2f} {m * ay:8.2f} {(m - 1.0) * ay:9.2f}  {cut:9.1f}%   "
                  f"{np.median(dem):5.2f} {np.percentile(dem, 90):6.2f} {dem.max():6.2f}  "
                  f"{np.percentile(np.abs(st[k]), 90):11.3f}")
        k = np.isfinite(base) & mv
        dem = v[k] ** 2 * np.abs(np.tan(base[k])) / wb
        print(f"  {'기준선':>6} {'--':>8} {'--':>9}  {'0.0':>9}%   {np.median(dem):5.2f} "
              f"{np.percentile(dem, 90):6.2f} {dem.max():6.2f}  "
              f"{np.percentile(np.abs(base[k]), 90):11.3f}")
        print(f"\n  복귀예산  = ceiling - 5.70 (레이스라인이 스스로 쓰는 최대 횡가속)")
        print(f"  잘라내는량 = 기록된 달성 a_lat(v*|yaw_rate|)이 그 ceiling 을 넘긴 시간 비율")
        print(f"             = 그 ceiling 이 실제로 포기하는 능력. 달성 p50 "
              f"{np.median(ach[mv]):.2f} p90 {np.percentile(ach[mv], 90):.2f} "
              f"max {ach[mv].max():.2f}")
        # The ">6.0 %" column is GONE on purpose: with the clip active it just reports how often
        # the clip binds (the demand is pinned AT the ceiling), so every margin above 1.05 read
        # the same number and it discriminated nothing.
        print("\n  달성 a_lat 은 요구가 커질수록 계속 오른다 -- 명확한 절벽이 없다:")
        d_rec = v ** 2 * np.abs(np.tan(rec)) / wb
        for lo, hi in ((4, 6), (6, 8), (8, 10), (10, 14), (14, 25)):
            b = mv & (d_rec >= lo) & (d_rec < hi)
            if b.sum():
                print(f"    요구 {lo:2d}-{hi:2d}: 달성 평균 {ach[b].mean():5.2f}  "
                      f"p90 {np.percentile(ach[b], 90):5.2f}  ({b.sum()} samples)")
        sys.exit(0)

    print("\n=== 변형별 조향 요구 (모두 shipped clip OFF 기준) ===")
    VarA.a_lat_max = ay
    for name, cls, over in (("A  a_lat 상한", VarA, {}),
                            ("B  2^lat_err 제거", VarB, {}),
                            ("A+B", VarAB, {}),
                            ("C  t_clip_min 1.0", VarC, {"t_clip_min": 1.0}),
                            ("D  rate 6 rad/s", VarD, {})):
        st = replay(cls, D, W, params, conv, **{**OFF, **over}).steer
        print(stats(name, st, v, rec, wb)[0])

    # A and E, alone and together. They act on different signals, so they have to be read
    # together: A cuts the steering command, E cuts the speed command, and the apex fault is
    # over-speed. `명령v@apex` is the replayed speed command averaged over the s=5.2 bin -- the
    # station where the plan says 4.51, the car was doing 5.27, and a_lat was 7.39 > 5.7.
    print(f"\n=== Ⓐ / Ⓔ / Ⓐ+Ⓔ  (controller.yaml: a_lat_margin {params.get('a_lat_margin')} "
          f"-> ceiling {params.get('a_lat_margin', 1.0) * ay:.2f}, a_comb_margin "
          f"{params.get('a_comb_margin')}) ===")
    s_apex = (s >= 5.2) & (s < 5.8)
    ship = None
    for label, cfg in CONFIGS:
        rr = replay(Controller, D, W, params, conv, **{**VEHKW, **cfg})
        st, sp = rr.steer, rr.spd
        if cfg["a_lat_limit_enable"] and cfg["a_comb_limit_enable"]:
            ship = st
        line = stats(label, st, v, rec, wb)[0]
        print(f"{line}  명령v@apex {np.nanmean(sp[s_apex]):4.2f}")

    # DIAGNOSTIC, nothing here is fixed by this commit. See lag_ms.
    print("\n=== 지연 진단 (진단만, 이번 변경은 명령 크기만 다룬다) ===")
    dcol = np.interp(t, D["/car_state/odom_frenet"][:, 0], D["/car_state/odom_frenet"][:, 2])
    w = longest_run(v > 2.0)
    ms, r, dt = lag_ms(t[w], dcol[w], rec[w])
    print(f"  lat_err -> 기록된 조향     피크 지연 {ms:+7.1f} ms  (r {r:+.3f}, dt {dt * 1000:.0f} ms, "
          f"{w.stop - w.start} samples)")
    print("    음수 = 조향이 lat_err 를 LEAD 한다. 폐루프 기록에서는 당연한 방향이기도 하다")
    print("    (조향이 lat_err 를 만든다). 이 상관만으로 루프 지연과 코너 기하를 가를 수 없다.")
    # Better posed: the replayed command is computed from the state AT THAT INSTANT, so its lag
    # behind the recorded command is delay that exists in the car but not in the code path.
    wf = longest_run(np.isfinite(ship) & (v > 2.0))
    ms, r, dt = lag_ms(t[wf], ship[wf], rec[wf], max_lag_s=0.5)
    print(f"  재생 명령 -> 기록된 명령   피크 지연 {ms:+7.1f} ms  (r {r:+.3f}, dt {dt * 1000:.0f} ms, "
          f"{wf.stop - wf.start} samples)")
    print("    이쪽이 잘 정의된 양이다: 같은 순간 상태로 계산한 명령 대비 실제 발행 명령의 지연.")
    print(f"    창은 v>2 가 끊기지 않는 최장 구간 {(t[wf.stop - 1] - t[wf.start]):.1f} s 다 "
          f"(상호상관에는 연속 시간이 필요하다).")
    print("  후보 기여자(분리 측정 안 됨): future_constant 0.05 s, heading 필터 alpha 0.1 @50Hz")
    print("  (시상수 0.2 s), speed_lookahead 0.25 s, 그리고 기록에 포함된 구동계/추정기 지연.")

    ach = achieved_a_lat(D, t)[v > 2.0]
    print(f"\n  (레이스라인 요구 max 5.70 | 실측 달성 a_lat p50 {np.median(ach):.2f} "
          f"p90 {np.percentile(ach, 90):.2f} max {ach.max():.2f} -- 명확한 무릎은 없다)")
    print("  개루프 재생이다: 명령이 내려간 것만 보이고, 그 상한으로 라인에 복귀하는지는 "
          "sim 실주행에서만 확인된다.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
