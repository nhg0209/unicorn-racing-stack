#!/usr/bin/env python3
"""vehdyn_analyze.py -- a measurement bag -> candidate veh_dyn_info csvs + a report.

    python3 stack_master/scripts/vehdyn_analyze.py ~/vehdyn_0813_1420
    python3 stack_master/scripts/vehdyn_analyze.py ~/vehdyn_0812_1948 --mode full

IT NEVER WRITES veh_dyn_info/. Output goes to config/vehdyn_measured/<timestamp>/ in the
shipped format, headers and all, for a human to read and copy in by hand. Copying them in
REQUIRES REGENERATING THE RACELINE (CLAUDE.md, coupled invariants).

THE PROCEDURE IS NOT NEGOTIABLE, it is what the 2026-08-12 measurement established:

  bias      Subtract the MEDIAN of the stationary samples first. That bag read ax -0.407 and
            ay +0.474 -- a tenth of the signal being measured.
  axes      PROVE the axis assignment, do not assume it: corr(imu.x, dv/dt) and
            corr(imu.y, v*gz) must both clear axis_corr_min or this refuses to emit values.
            The check is what caught controller_manager using -imu.x where the car says
            +imu.x. The longitudinal half is evaluated only where |a_lat| < axis_corr_max_alat:
            over a whole bag it reads 0.76 and would fail a good measurement, because in
            steady cornering dv/dt ~ 0 while body roll leaks gravity into a_x. Restricted, the
            8/12 bag reads 0.929 -- the +0.926 it was reported with.
  dv/dt     Resample to a uniform grid, THEN Savitzky-Golay. The raw stamps carry duplicates
            and a plain difference explodes (-1816..+3188 on that bag).
  long      p95 of ax > +threshold, p05 of ax < -threshold, PER SPEED BIN -- the bins are the
            machines-table curves. The covered speed range is part of the result: if 0-9 m/s
            was not covered, that is reported, not hidden.
  lat       BOTH methods, side by side. Primary omega^2*R from a circle fit to the kiss
            trajectory (with its residual, because a fit residual is the only thing that says
            the path was a circle); secondary the p95 of bias-corrected imu.ay. The
            accelerometer reads higher -- cornering roll leaks 1-2 m/s^2 of gravity into the
            lateral axis -- so the LOWER is adopted and the gap is reported.
  never     Wheel speed x gyro (wheels spin at washout: 21.5 vs a true ~10) and the derivative
            of the kiss pose (attitude jitter amplifies to 17.5 m/s). Neither is computed.

Left and right are reported separately: grip is symmetric but steering geometry is not, so
the same steering angle gives different radii and speeds each way.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # stack_master/
DEFAULT_PARAMS = os.path.join(REPO, 'config', 'vehdyn_test_params.yaml')


# ----------------------------------------------------------------------------------------
# pure helpers -- unit-tested in stack_master/scripts/test_vehdyn.py
# ----------------------------------------------------------------------------------------


def fit_circle(x, y):
    """Algebraic circle fit. Returns (cx, cy, R, rms_residual_m, swept_arc_rad).

    The residual says whether the path was a circle at all; the arc says whether R is
    constrained. A short arc fits ANY radius with a tiny residual, which is exactly the trap
    that turns a 1.4 s run into an 18 m/s^2 lateral acceleration.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    c, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = float(c[0]), float(c[1])
    R = float(np.sqrt(max(c[2] + cx ** 2 + cy ** 2, 0.0)))
    resid = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - R) ** 2)))
    ang = np.unwrap(np.arctan2(y - cy, x - cx))
    return cx, cy, R, resid, float(abs(ang[-1] - ang[0]))


def find_runs(mask, t, max_gap_s, min_dur_s):
    """Contiguous stretches of `mask`, tolerating gaps up to max_gap_s."""
    idx = np.flatnonzero(mask)
    out = []
    if len(idx) == 0:
        return out
    s = p = idx[0]
    for i in idx[1:]:
        if t[i] - t[p] > max_gap_s:
            if t[p] - t[s] >= min_dur_s:
                out.append((int(s), int(p)))
            s = i
        p = i
    if t[p] - t[s] >= min_dur_s:
        out.append((int(s), int(p)))
    return out


def bin_percentiles(v, a, lo_edges, bin_w, min_n, pct, sign):
    """Per-speed-bin percentile of the accelerating (sign>0) or braking (sign<0) samples."""
    rows = []
    for lo in lo_edges:
        m = (v >= lo) & (v < lo + bin_w)
        sel = a[m & (a > 0)] if sign > 0 else a[m & (a < 0)]
        rows.append((lo, int(len(sel)),
                     float(np.percentile(sel, pct)) if len(sel) >= min_n else float('nan')))
    return rows


# ----------------------------------------------------------------------------------------
# bag IO
# ----------------------------------------------------------------------------------------


def read_bag(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
           rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    inv = {v: k for k, v in topics.items()}
    acc = {k: [] for k in topics}
    while r.has_next():
        tp, data, t = r.read_next()
        key = inv.get(tp)
        if key is None or tp not in types:
            continue
        m = deserialize_message(data, get_message(types[tp]))
        ts = t / 1e9
        if key == 'imu':
            acc[key].append((ts, m.linear_acceleration.x, m.linear_acceleration.y,
                             m.linear_acceleration.z, m.angular_velocity.z))
        elif key == 'odom':
            q = m.pose.pose.orientation
            acc[key].append((ts, m.pose.pose.position.x, m.pose.pose.position.y,
                             m.twist.twist.linear.x, m.twist.twist.angular.z,
                             q.x, q.y, q.z, q.w))
        elif key == 'core':
            acc[key].append((ts, m.state.current_motor, m.state.speed))
        elif key == 'cmd':
            acc[key].append((ts, m.drive.speed, m.drive.steering_angle))
    return {k: np.array(v, float) if v else np.zeros((0, 5)) for k, v in acc.items()}


# ----------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------


class Analysis:
    def __init__(self, data, cfg, ref):
        self.cfg = cfg
        self.ref = ref
        self.warn = []
        imu, odom = data['imu'], data['odom']
        if len(imu) == 0 or len(odom) == 0:
            raise SystemExit("bag has no IMU or no odometry -- nothing to analyse")
        self.t0 = imu[0, 0]
        self.ti = imu[:, 0] - self.t0
        self.ax_raw, self.ay_raw, self.gz_raw = imu[:, 1], imu[:, 2], imu[:, 4]
        self.tp = odom[:, 0] - self.t0
        self.px, self.py = odom[:, 1], odom[:, 2]
        self.v_odom = odom[:, 3]
        self.core = data.get('core', np.zeros((0, 3)))

        self.v = np.interp(self.ti, self.tp, self.v_odom)     # speed on the IMU clock
        still = self.v < float(cfg['still_speed_mps'])
        self.n_still = int(still.sum())
        if self.n_still < 50:
            self.warn.append(
                f"only {self.n_still} stationary samples -- the bias estimate is weak. Every "
                f"maneuver is supposed to be preceded by a {ref.get('settle_still_s', 5)} s stop.")
            still = self.v < 0.5
        self.bias = (float(np.median(self.ax_raw[still])),
                     float(np.median(self.ay_raw[still])),
                     float(np.median(self.gz_raw[still])))
        self.ax = self.ax_raw - self.bias[0]
        self.ay = self.ay_raw - self.bias[1]
        self.gz = self.gz_raw - self.bias[2]
        self.alat = self.v * self.gz          # kinematic lateral accel, on the IMU clock

    # -- axis validation ------------------------------------------------------------
    def validate_axes(self):
        from scipy.signal import savgol_filter
        fs = float(self.cfg['resample_hz'])
        win = int(float(self.cfg['savgol_window_s']) * fs) // 2 * 2 + 1
        win = max(win, int(self.cfg['savgol_polyorder']) + 2)
        tu = np.arange(self.ti[0], self.ti[-1], 1.0 / fs)
        vu = np.interp(tu, self.tp, self.v_odom)
        dvdt = savgol_filter(vu, win, int(self.cfg['savgol_polyorder']),
                             deriv=1, delta=1.0 / fs)
        axu = savgol_filter(np.interp(tu, self.ti, self.ax), win,
                            int(self.cfg['savgol_polyorder']))
        ayu = savgol_filter(np.interp(tu, self.ti, self.ay), win,
                            int(self.cfg['savgol_polyorder']))
        gzu = np.interp(tu, self.ti, self.gz)
        alat_u = vu * gzu
        straight = np.abs(alat_u) < float(self.cfg['axis_corr_max_alat'])
        c_long = (float(np.corrcoef(axu[straight], dvdt[straight])[0, 1])
                  if straight.sum() > 100 else float('nan'))
        c_lat = float(np.corrcoef(ayu, alat_u)[0, 1])
        self.axis = {'corr_long': c_long, 'corr_lat': c_lat,
                     'n_straight': int(straight.sum()),
                     'raw_dvdt_min': float(np.nanmin(np.diff(self.v_odom) /
                                                     np.maximum(np.diff(self.tp), 1e-9))),
                     'raw_dvdt_max': float(np.nanmax(np.diff(self.v_odom) /
                                                     np.maximum(np.diff(self.tp), 1e-9)))}
        lim = float(self.cfg['axis_corr_min'])
        self.axis['pass'] = (c_long > lim) and (c_lat > lim)
        return self.axis

    # -- longitudinal ---------------------------------------------------------------
    def longitudinal(self):
        thr = float(self.cfg['long_min_accel_mps2'])
        lat_gate = float(self.ref['ggv_ay_max_ref']) * float(self.cfg['long_max_alat_frac'])
        ok = np.abs(self.alat) < lat_gate
        bw = float(self.cfg['long_speed_bin_mps'])
        vmax = float(np.nanmax(self.v))
        edges = [i * bw for i in range(int(math.ceil(max(vmax, bw) / bw)))]
        acc = bin_percentiles(self.v[ok], np.where(self.ax[ok] > thr, self.ax[ok], 0.0),
                              edges, bw, int(self.cfg['long_min_samples_per_bin']),
                              int(self.cfg['long_accel_pct']), +1)
        brk = bin_percentiles(self.v[ok], np.where(self.ax[ok] < -thr, self.ax[ok], 0.0),
                              edges, bw, int(self.cfg['long_min_samples_per_bin']),
                              int(self.cfg['long_brake_pct']), -1)
        covered = [lo for lo, n, val in acc if not math.isnan(val)]
        self.long = {'accel_bins': acc, 'brake_bins': brk,
                     'speed_min': float(np.nanmin(self.v)), 'speed_max': vmax,
                     'covered_lo': min(covered) if covered else None,
                     'covered_hi': (max(covered) + bw) if covered else None,
                     'lat_gate': lat_gate}
        a = [v for _, _, v in acc if not math.isnan(v)]
        b = [v for _, _, v in brk if not math.isnan(v)]
        self.long['accel_range'] = (min(a), max(a)) if a else None
        self.long['brake_range'] = (min(b), max(b)) if b else None
        if not a or not b:
            self.warn.append("no usable longitudinal bins -- the machines tables cannot be "
                             "measured from this bag")
        if covered and (min(covered) > 1.0 or max(covered) + bw < 8.0):
            self.warn.append(
                f"longitudinal coverage is only {min(covered):.0f}-{max(covered) + bw:.0f} m/s; "
                f"outside it the machines curves are the REFERENCE values, not measurements")
        return self.long

    # -- lateral --------------------------------------------------------------------
    def lateral(self):
        c = self.cfg
        segs = []
        moving = self.v > float(c['circle_min_speed_mps'])
        for side, sgn in (('left', 1), ('right', -1)):
            m = moving & (np.sign(self.gz) == sgn) & (np.abs(self.gz) > float(c['circle_min_gz']))
            for a, b in find_runs(m, self.ti, float(c['circle_max_gap_s']),
                                  float(c['circle_min_dur_s'])):
                if self.v[a:b].max() < float(c['circle_min_peak_speed_mps']):
                    continue          # slow manoeuvring, not a washout run
                segs.append((float(self.ti[a]), float(self.ti[b]), side, a, b))
        segs.sort()
        rows = []
        for tA, tB, side, a, b in segs:
            mp = (self.tp >= tA) & (self.tp <= tB)
            if mp.sum() < 10:
                continue
            _, _, R, resid, arc = fit_circle(self.px[mp], self.py[mp])
            om = float(np.median(np.abs(self.gz[a:b])))
            imu_p95 = float(np.percentile(np.abs(self.ay[a:b]), int(c['lat_imu_pct'])))
            trusted = (resid <= float(c['circle_max_resid_m'])
                       and arc >= float(c['circle_min_arc_rad']))
            rows.append({'side': side, 't0': tA, 't1': tB, 'R': R, 'resid': resid,
                         'arc': arc, 'omega': om, 'a_geom': om * om * R,
                         'a_imu': imu_p95, 'v_peak': float(self.v[a:b].max()),
                         'arc_ok': arc >= float(c['circle_min_arc_rad']),
                         'resid_ok': resid <= float(c['circle_max_resid_m']),
                         'trusted': trusted})
        self.lat_runs = rows
        usable = [r for r in rows if r['arc_ok']]     # residual only downgrades confidence
        geom = [r['a_geom'] for r in usable]
        imu = [r['a_imu'] for r in rows]
        self.lat = {
            'runs': rows,
            'geom_range': (min(geom), max(geom)) if geom else None,
            'imu_range': (min(imu), max(imu)) if imu else None,
            # CONSERVATIVE: the accelerometer reads high because roll leaks gravity in.
            'adopted': min(geom) if geom else (min(imu) if imu else None),
            'adopted_from': 'omega^2*R (circle fit)' if geom else 'imu.ay p95',
            'n_low_conf': sum(1 for r in rows if not r['resid_ok']),
        }
        if geom and imu:
            self.lat['method_gap'] = float(np.median(imu) - np.median(geom))
        if not geom:
            self.warn.append("no circle run had a long enough arc for a trustworthy radius; "
                             "the lateral number falls back to the accelerometer, which reads "
                             "HIGH (gravity leaks in through body roll)")
        if self.lat['n_low_conf']:
            self.warn.append(
                f"{self.lat['n_low_conf']} of {len(rows)} circle runs exceed the "
                f"{c['circle_max_resid_m']} m fit residual -- the path was not a circle "
                f"(usually speed still ramping). Treat those as LOW CONFIDENCE.")
        for side in ('left', 'right'):
            s = [r['a_imu'] for r in rows if r['side'] == side]
            self.lat[f'{side}_imu'] = s
            g = [r['a_geom'] for r in rows if r['side'] == side and r['arc_ok']]
            self.lat[f'{side}_geom'] = g
        return self.lat

    # -- current ---------------------------------------------------------------------
    def current(self):
        if len(self.core) == 0:
            return None
        tc, cur = self.core[:, 0] - self.t0, self.core[:, 1]
        cur_i = np.interp(self.ti, tc, cur)
        m = ((np.abs(cur_i) > float(self.cfg.get('current_fit_min_a', 5.0)))
             & (self.v < float(self.cfg.get('current_fit_max_speed', 3.0)))
             & (np.abs(self.alat) < 2.0))
        if m.sum() < 50:
            return {'ok': False, 'n': int(m.sum()),
                    'range': (float(cur.min()), float(cur.max()))}
        k = float(np.linalg.lstsq(cur_i[m][:, None], self.ax[m], rcond=None)[0][0])
        return {'ok': True, 'n': int(m.sum()), 'k_mps2_per_A': k,
                'range': (float(cur.min()), float(cur.max())),
                'extrapolated_a': k * float(self.ref['current_max_a'])}


# ----------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------


def _speed_grid(ref):
    return [round(i * 0.25, 2) for i in range(61)]     # 0.00 .. 15.00, the shipped grid


def write_csvs(out_dir, res, ref, meta):
    """Candidate csvs in the SHIPPED format, header and all, so a human can diff and copy."""
    hdr = (f"# MEASURED {meta['when']} | mode={meta['mode']} | bag={meta['bag']}\n"
           f"# confidence: {meta['confidence']}\n"
           f"# NOT A LIVE FILE. Copy into config/<CAR|SIM>/veh_dyn_info/ by hand, then\n"
           f"# REGENERATE THE RACELINE -- vx_mps in maps/<map>/global_waypoints.json is an\n"
           f"# offline product and the online path can only lower it.\n")
    grid = _speed_grid(ref)

    ggv = res['ggv']
    with open(os.path.join(out_dir, 'ggv.csv'), 'w') as f:
        f.write(hdr)
        f.write(f"# ax_max {ggv['ax']:.2f}  ay_max {ggv['ay']:.2f}"
                f"  ({ggv['how']})\n")
        f.write("# v_mps,ax_max_mps2,ay_max_mps2\n")
        f.write("\n".join(f"{v:.1f},{ggv['ax']:.2f},{ggv['ay']:.2f}" for v in grid))

    for name, series, how in (('ax_max_machines.csv', res['axm'], res['axm_how']),
                              ('b_ax_max_machines.csv', res['bax'], res['bax_how'])):
        with open(os.path.join(out_dir, name), 'w') as f:
            f.write(hdr)
            f.write(f"# {how}\n")
            f.write("#v_mps, ax_max_machines_mps2\n")
            f.write("\n".join(f"{v:.2f},{series(v):.3f}" for v in grid))


def write_report(out_dir, A, res, ref, meta, cond_note):
    L = []
    ap = L.append
    ap(f"# Vehicle-dynamics measurement -- {meta['when']}")
    ap("")
    ap(f"- bag: `{meta['bag']}`")
    ap(f"- mode: **{meta['mode']}**")
    ap(f"- output: `{out_dir}` (candidates only -- nothing live was written)")
    ap("")
    ap("## Verdict")
    ap("")
    if not A.axis['pass']:
        ap("**FAILED — no values emitted.** The IMU axis assignment could not be proven, so "
           "every number downstream would be a guess about which axis is which.")
    else:
        ap("Axis assignment proven; values below are measurements, not assumptions.")
    ap("")
    ap("| check | value | gate | verdict |")
    ap("|---|---|---|---|")
    ap(f"| corr(imu.x, dv/dt), \\|a_lat\\|<{A.cfg['axis_corr_max_alat']} | "
       f"{A.axis['corr_long']:+.3f} | > {A.cfg['axis_corr_min']} | "
       f"{'PASS' if A.axis['corr_long'] > float(A.cfg['axis_corr_min']) else 'FAIL'} |")
    ap(f"| corr(imu.y, v*gz) | {A.axis['corr_lat']:+.3f} | > {A.cfg['axis_corr_min']} | "
       f"{'PASS' if A.axis['corr_lat'] > float(A.cfg['axis_corr_min']) else 'FAIL'} |")
    ap("")
    ap(f"IMU bias removed (median of {A.n_still} stationary samples): "
       f"ax {A.bias[0]:+.3f}, ay {A.bias[1]:+.3f}, gz {A.bias[2]:+.5f} m/s^2, rad/s.")
    ap(f"Raw dv/dt before resampling spanned {A.axis['raw_dvdt_min']:.0f}.."
       f"{A.axis['raw_dvdt_max']:.0f} m/s^2 -- why it is resampled and Savitzky-Golay "
       f"filtered rather than differenced.")
    ap("")

    ap("## Live vs measured")
    ap("")
    ap("| quantity | live now | measured | ratio | source |")
    ap("|---|---|---|---|---|")
    g = res['ggv']
    ap(f"| ggv ay_max | {ref['ggv_ay_max_ref']:.2f} | {g['ay']:.2f} | "
       f"{g['ay'] / float(ref['ggv_ay_max_ref']):.3f} | {g['how']} |")
    ap(f"| ggv ax_max | {ref['ggv_ax_max_ref']:.2f} | {g['ax']:.2f} | "
       f"{g['ax'] / float(ref['ggv_ax_max_ref']):.3f} | {g['how']} |")
    ap(f"| ax_max_machines (at rest) | {ref['ax_max_machines_ref']:.2f} | "
       f"{res['axm'](0.0):.2f} | — | {res['axm_how']} |")
    ap(f"| b_ax_max_machines | {ref['b_ax_max_machines_ref']:.2f} | "
       f"{res['bax'](0.0):.2f} | — | {res['bax_how']} |")
    ap("")
    ap(f"### k = {res['k']:.3f}")
    ap("")
    ap(f"`k = ay_measured / ggv_ay_max_ref = {res['lat_adopted']:.2f} / "
       f"{ref['ggv_ay_max_ref']:.2f}`")
    ap("")
    if meta['mode'] == 'circle':
        ap("k was applied to **both** ggv columns — ax_max and ay_max. Both are tyre grip and "
           "a resurfaced track scales them together. `ax_max_machines` and "
           "`b_ax_max_machines` are motor and brake properties, independent of the surface: "
           "they were **copied through unchanged** from the live files.")
    else:
        ap("In `full` mode ggv ax_max is the DIRECT longitudinal measurement, not k-scaled; "
           "k is reported for comparison with the venue path only.")
    ap("")

    ap("## Confidence")
    ap("")
    ap(f"- stationary samples for bias: **{A.n_still}**")
    lo, hi = A.long.get('covered_lo'), A.long.get('covered_hi')
    ap(f"- longitudinal speed coverage: **{lo}–{hi} m/s** "
       f"(bag spans {A.long['speed_min']:.1f}..{A.long['speed_max']:.1f})"
       if lo is not None else "- longitudinal speed coverage: **none**")
    ap(f"- longitudinal samples accepted only where |a_lat| < {A.long['lat_gate']:.2f} m/s^2")
    if A.lat['geom_range']:
        ap(f"- lateral, circle fit (omega^2*R): "
           f"**{A.lat['geom_range'][0]:.2f}..{A.lat['geom_range'][1]:.2f}** m/s^2")
    if A.lat['imu_range']:
        ap(f"- lateral, imu.ay p95: "
           f"**{A.lat['imu_range'][0]:.2f}..{A.lat['imu_range'][1]:.2f}** m/s^2")
    if 'method_gap' in A.lat:
        ap(f"- the two lateral methods differ by **{A.lat['method_gap']:+.2f} m/s^2** "
           f"(median). The accelerometer reading higher is expected: cornering roll tips "
           f"gravity into the lateral axis. **The lower was adopted.**")
    ap("")
    ap("| run | side | v peak | R | fit residual | arc | omega^2*R | imu.ay p95 | trusted |")
    ap("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(A.lat['runs']):
        why = "yes" if r['trusted'] else (
            "arc too short" if not r['arc_ok'] else "residual high")
        ap(f"| {i} | {r['side']} | {r['v_peak']:.2f} | {r['R']:.2f} | {r['resid']:.3f} | "
           f"{r['arc']:.2f} | {r['a_geom']:.2f} | {r['a_imu']:.2f} | {why} |")
    ap("")
    ap(f"Left imu.ay p95:  {', '.join(f'{x:.1f}' for x in A.lat.get('left_imu', []))}")
    ap(f"Right imu.ay p95: {', '.join(f'{x:.1f}' for x in A.lat.get('right_imu', []))}")
    ap("")
    ap("Left/right are reported apart on purpose: grip is symmetric, steering geometry is "
       "not, so the same steering angle gives a different radius and speed each way.")
    ap("")
    ap("### Longitudinal, per speed bin")
    ap("")
    ap("| v bin [m/s] | n accel | p95 accel | n brake | p05 brake |")
    ap("|---|---|---|---|---|")
    for (lo_, na, va), (_, nb, vb) in zip(A.long['accel_bins'], A.long['brake_bins']):
        ap(f"| {lo_:.0f}–{lo_ + 1:.0f} | {na} | "
           f"{'—' if math.isnan(va) else f'{va:.2f}'} | {nb} | "
           f"{'—' if math.isnan(vb) else f'{vb:.2f}'} |")
    ap("")

    if res.get('current'):
        c = res['current']
        ap("### Current mode — EXTRAPOLATED, LOW CONFIDENCE")
        ap("")
        ap(f"Measured current spanned {c['range'][0]:.1f}..{c['range'][1]:.1f} A.")
        if c.get('ok'):
            ap(f"Fitted ax = {c['k_mps2_per_A']:.4f} * I over {c['n']} low-speed samples, "
               f"extrapolated to current_max_a = {ref['current_max_a']} A -> "
               f"**{c['extrapolated_a']:.2f} m/s^2**.")
            ap("")
            ap("This is an EXTRAPOLATION well past the measured range. It is not evidence "
               "of what the car does at the limit; it is an estimate to be replaced by a "
               "`full` run as soon as there is room.")
        else:
            ap(f"Only {c['n']} usable samples — no fit.")
        ap("")

    ap("## Venue re-calibration reference")
    ap("")
    ap(f"**ay_max measured here = {res['lat_adopted']:.3f} m/s^2**")
    ap("")
    ap("Record this number. Next time the surface changes, run `mode: circle` and divide: "
       "`k = ay_new / ay_this`. That is the whole venue procedure — about two minutes.")
    ap("")
    ap("## What to do with these files")
    ap("")
    ap("1. Read the numbers above and decide whether they are believable. Anything far from "
       f"the live values ({ref['ggv_ax_max_ref']}/{ref['ggv_ay_max_ref']} ggv, "
       f"{ref['ax_max_machines_ref']} ax_max_machines, {ref['b_ax_max_machines_ref']} b_ax) "
       "is a reason to suspect the measurement, not the car — the live set is known good.")
    ap("2. Copy by hand into `stack_master/config/<CAR|SIM>/veh_dyn_info/`. Nothing here "
       "writes those files.")
    ap("3. **REGENERATE THE RACELINE.** `vx_mps` in `maps/<map>/global_waypoints.json` is an "
       "offline product of the global optimizer; the online path can only lower it, never "
       "raise it. Until it is rebuilt the car will not go any faster. Restarting the stack "
       "is not enough.")
    ap("4. ggv.csv carries a hand-written header explaining this. Whatever regenerates the "
       "file strips it — put it back.")
    if A.warn:
        ap("")
        ap("## Warnings")
        ap("")
        for w in A.warn:
            ap(f"- {w}")
    if cond_note:
        ap("")
        ap(cond_note)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', help='rosbag2 directory (mcap)')
    ap.add_argument('--params', default=DEFAULT_PARAMS)
    ap.add_argument('--mode', default=None, choices=['circle', 'full', 'current'],
                    help='override the mode recorded in the params file')
    ap.add_argument('--out', default=None, help='output dir (default config/vehdyn_measured/<ts>)')
    a = ap.parse_args(argv)

    import yaml
    with open(a.params) as f:
        P = yaml.safe_load(f)
    P = P['/**']['ros__parameters']
    cfg = P['analyze']
    cfg.setdefault('current_fit_min_a', P.get('current_fit_min_a', 5.0))
    cfg.setdefault('current_fit_max_speed', P.get('current_fit_max_speed', 3.0))
    mode = a.mode or P['mode']

    data = read_bag(a.bag, cfg['bag_topics'])
    A = Analysis(data, cfg, P)
    A.validate_axes()
    A.longitudinal()
    A.lateral()

    when = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = a.out or os.path.join(REPO, 'config', 'vehdyn_measured', stamp)
    os.makedirs(out_dir, exist_ok=True)

    if not A.axis['pass']:
        meta = {'when': when, 'mode': mode, 'bag': a.bag, 'confidence': 'FAILED'}
        res = {'ggv': {'ax': float('nan'), 'ay': float('nan'), 'how': 'FAILED'},
               'axm': lambda v: float('nan'), 'axm_how': 'FAILED',
               'bax': lambda v: float('nan'), 'bax_how': 'FAILED',
               'k': float('nan'), 'lat_adopted': float('nan')}
        A.longitudinal()
        rep = write_report(out_dir, A, res, P, meta,
                           "> No csvs were written: the axis check failed.")
        with open(os.path.join(out_dir, 'report.md'), 'w') as f:
            f.write(rep)
        print(rep)
        print(f"\nFAILED -- axis validation did not pass. Report: {out_dir}/report.md")
        return 1

    lat_adopted = A.lat['adopted']
    k = lat_adopted / float(P['ggv_ay_max_ref'])

    if mode == 'full' and A.long['accel_range']:
        ggv_ax = min(abs(A.long['brake_range'][0]), A.long['accel_range'][1])
        how_ax = 'measured directly (full mode)'
    else:
        ggv_ax = float(P['ggv_ax_max_ref']) * k
        how_ax = f'reference x k ({P["ggv_ax_max_ref"]} x {k:.3f})'

    def const(x):
        return lambda v: x

    if mode == 'full' and A.long['accel_range']:
        acc_bins = {lo: val for lo, _, val in A.long['accel_bins'] if not math.isnan(val)}
        brk_bins = {lo: -val for lo, _, val in A.long['brake_bins'] if not math.isnan(val)}

        def mk(bins, fallback):
            los = sorted(bins)
            if not los:
                return const(fallback)
            xs = np.array([lo + 0.5 for lo in los])
            ys = np.array([bins[lo] for lo in los])
            return lambda v: float(np.interp(v, xs, ys))
        axm, bax = mk(acc_bins, float(P['ax_max_machines_ref'])), \
            mk(brk_bins, float(P['b_ax_max_machines_ref']))
        axm_how = 'measured per speed bin (full mode)'
        bax_how = 'measured per speed bin (full mode)'
    else:
        axm, bax = const(float(P['ax_max_machines_ref'])), const(float(P['b_ax_max_machines_ref']))
        axm_how = ('COPIED UNCHANGED from the live file -- motor property, surface-independent, '
                   'not measured in circle mode')
        bax_how = ('COPIED UNCHANGED from the live file -- brake property, surface-independent, '
                   'not measured in circle mode')

    conf = 'circle/washout, low-conf runs: %d' % A.lat['n_low_conf']
    if mode == 'current':
        conf = 'EXTRAPOLATED FROM MOTOR CURRENT -- LOW CONFIDENCE'
    res = {'ggv': {'ax': ggv_ax, 'ay': lat_adopted,
                   'how': f'ay {A.lat["adopted_from"]}; ax {how_ax}'},
           'axm': axm, 'axm_how': axm_how, 'bax': bax, 'bax_how': bax_how,
           'k': k, 'lat_adopted': lat_adopted,
           'current': A.current() if mode == 'current' else None}
    meta = {'when': when, 'mode': mode, 'bag': os.path.abspath(a.bag), 'confidence': conf}

    write_csvs(out_dir, res, P, meta)
    rep = write_report(out_dir, A, res, P, meta, None)
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write(rep)
    with open(os.path.join(out_dir, 'raw.json'), 'w') as f:
        json.dump({'axis': A.axis, 'bias': A.bias, 'long': {
            k2: v for k2, v in A.long.items() if k2 != 'accel_bins' and k2 != 'brake_bins'},
            'lat_runs': A.lat['runs'], 'k': k}, f, indent=2, default=float)
    print(rep)
    print(f"\nWrote {out_dir}/  (ggv.csv, ax_max_machines.csv, b_ax_max_machines.csv, "
          f"report.md, raw.json)")
    print("These are CANDIDATES. Nothing under veh_dyn_info/ was touched.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
