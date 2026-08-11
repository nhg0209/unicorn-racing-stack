#!/usr/bin/env python3
"""Regenerate ifac's raceline at lower OPTIMISATION curvature limits, and price what that buys.

THE HYPOTHESIS UNDER TEST. Three open problems in the static re-optimizer look like one:

    the margin chain has +0.030 m of headroom where 0.050 is required, and obs_margin cannot be
    raised to 0.18 because the hold dies;
    three of nine handovers are the curvature budget (stations 200 x2, 250);
    coverage is 29/38.

All three are downstream of the same fact: ifac's shipped raceline peaks at |kappa| 1.448 against
a curvlim of 1.5, so closed_reopt's curvature budget has 0.052 to spend. Optimise the raceline to
a LOWER curvature and the budget grows. The bill is clean lap time.

WHAT IS AND IS NOT CHANGED. Only the limit the OPTIMISER is given (racecar_f110.ini
veh_params.curvlim, copied into a temp config dir -- the shipped ini is not touched). The speed
planner's curvlim stays 1.5 wherever it is read at runtime, and the gap between the two IS the
budget. Nothing here writes stack_master/maps/ifac/global_waypoints.json; candidates go to
maps/ifac/candidates/ and are a measurement, not an adoption.

    ~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/sweep_raceline_curvlim.py
    ... --limits 1.50 1.35 --skip-reopt      (faster: geometry only)
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/gb_optimizer"))
sys.path.insert(0, str(REPO / "planner/gb_optimizer/gb_optimizer/global_racetrajectory_optimization"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# vel_planner ships as its own ament package (race_utils/f110_utils/libs/vel_planner) and
# trajectory_optimizer imports it by bare name; offline it is not on the path unless the workspace
# is sourced into THIS interpreter, which the conda one is not.
sys.path.insert(0, str(REPO / "race_utils/f110_utils/libs/vel_planner"))

from gb_optimizer import closed_reopt as C                    # noqa: E402
from gb_optimizer.closed_reopt import Obstacle, ReoptParams   # noqa: E402
import compare_reopt as CMP                                   # noqa: E402
from bench_closed_reopt import corridor_from_map              # noqa: E402

MAP = "ifac"
BOX_R = 0.15
OUT = REPO / "stack_master/maps" / MAP / "candidates"
# what static_reopt_node ships with, so a candidate is judged by the solver as configured
SPEED_CURVLIM = 1.5
# the reactive layer's idle entry, line-centre to obstacle edge: width_car/2 + clear_margin_m +
# clear_hyst_m from static_avoidance_params.yaml. A claimed box must beat this or it gets avoided
# twice; check_avoidance_margins.py wants 0.05 m of headroom over it.
IDLE_NEED = 0.28
REQUIRED_HEADROOM = 0.05


def shipped():
    d = json.load(open(REPO / "stack_master/maps" / MAP / "global_waypoints.json"))
    wp = d["global_traj_wpnts_iqp"]["wpnts"][:-1]
    ref = np.array([[w["x_m"], w["y_m"], w["d_right"], w["d_left"]] for w in wp], float)
    vx = np.array([w["vx_mps"] for w in wp], float)
    return ref, vx, float(d["est_lap_time"]["data"])


def temp_config(curvlim: float) -> Path:
    """A copy of config/SIM with veh_params.curvlim replaced. The shipped ini is never written."""
    src = REPO / "stack_master/config/SIM"
    dst = Path(tempfile.mkdtemp(prefix=f"curvlim_{curvlim:.2f}_"))
    shutil.copytree(src, dst / "SIM")
    ini = dst / "SIM" / "racecar_f110.ini"
    txt = ini.read_text()
    new, n = re.subn(r'("curvlim"\s*:\s*)[-+0-9.eE]+', rf'\g<1>{curvlim}', txt)
    if n != 1:
        raise RuntimeError(f"curvlim not found exactly once in {ini} (found {n})")
    ini.write_text(new)
    return dst / "SIM"


def optimise(curvlim: float, safety_width: float):
    """Run the shipped mincurv_iqp pipeline at this optimisation curvlim.

    The centerline goes to /tmp/map_centerline.csv because that is where trajectory_optimizer
    looks for that track name -- the same handoff global_planner_node uses.
    """
    from trajectory_optimizer import trajectory_optimizer
    cent = np.genfromtxt(REPO / "stack_master/maps" / MAP / "centerline.csv",
                         delimiter=",", skip_header=1)
    np.savetxt("/tmp/map_centerline.csv", cent, delimiter=",",
               header="x_m,y_m,w_tr_right_m,w_tr_left_m", comments="# ")
    cfg = temp_config(curvlim)
    t0 = time.perf_counter()
    traj, br, bl, est = trajectory_optimizer(input_path=str(cfg), track_name="map_centerline",
                                             curv_opt_type="mincurv_iqp",
                                             safety_width=safety_width, plot=False)
    return traj, br, bl, float(est), time.perf_counter() - t0


def _densify(a, step=0.02):
    out = []
    b = np.vstack([a, a[:1]])
    for i in range(len(a)):
        el = float(np.hypot(*(b[i + 1] - b[i])))
        for t in np.linspace(0.0, 1.0, max(2, int(el / step) + 1), endpoint=False):
            out.append(b[i] * (1.0 - t) + b[i + 1] * t)
    return np.array(out)


_BOUNDS = None


def track_bounds():
    """Dense right/left bound polylines from centerline.csv + its widths. Cached."""
    global _BOUNDS
    if _BOUNDS is None:
        cent = np.genfromtxt(REPO / "stack_master/maps" / MAP / "centerline.csv",
                             delimiter=",", skip_header=1)
        _psi, nv, _t = C._frame(cent[:, :2])
        _BOUNDS = (_densify(cent[:, :2] + cent[:, 2][:, None] * nv),
                   _densify(cent[:, :2] - cent[:, 3][:, None] * nv))
    return _BOUNDS


def widths(traj_xy):
    """d_right/d_left against the MAP's own bounds, measured identically for every line here.

    NOT the values in global_waypoints.json: those came off the optimizer's spline-interpolated
    reftrack and differ from a direct measurement by up to 0.13 m, which is enough to move the QP
    corridor and therefore the coverage. The variable under test is the raceline's CURVATURE, so
    every line -- the shipped one included -- is re-measured here with one source, and the
    coverage numbers below are consequently not comparable to those taken on the json's widths.
    """
    br, bl = track_bounds()
    dr = np.array([np.min(np.hypot(br[:, 0] - p[0], br[:, 1] - p[1])) for p in traj_xy])
    dl = np.array([np.min(np.hypot(bl[:, 0] - p[0], bl[:, 1] - p[1])) for p in traj_xy])
    return dr, dl


def seam_check(vx, el):
    """The s = 0 velocity seam: __solver_fb_closed does not decelerate the returned lap's LAST
    point, so the wrap from station N-1 to 0 can demand an impossible longitudinal accel.

    Reported, never fixed here -- it lives in the vendored tph.
    """
    a = (np.roll(vx, -1) ** 2 - vx ** 2) / (2.0 * np.maximum(el, 1e-6))
    return float(a[-1]), float(vx[-1]), float(vx[0]), float(np.max(np.abs(a[:-1])))


def clean_metrics(ref, vx):
    xy = ref[:, :2]
    k = np.abs(C.menger_closed(xy))
    el = C._closed_el(xy)
    return dict(peak=float(np.max(k)), peak_at=int(np.argmax(k)),
                budget=SPEED_CURVLIM - float(np.max(k)), length=float(np.sum(el)),
                lap=float(np.sum(el / np.maximum(vx, 1e-3))))


def reopt_on(ref, cor, obs_margin):
    """closed_qp over the twenty-case matrix on THIS raceline."""
    p = ReoptParams(obs_margin=obs_margin)
    need = p.obs_margin + 0.5 * p.w_veh
    n = len(ref)
    cov = tot = 0
    geo = budget = 0
    cl, times = [], []
    kc = float(np.max(np.abs(C.menger_closed(ref[:, :2]))))
    a_ok, b_max = True, -9.0
    for _name, stations in CMP.build_cases(n):
        obs = [Obstacle(float(ref[i % n, 0]), float(ref[i % n, 1]), BOX_R) for i in stations]
        _l, _d, r = C.reoptimize_closed(ref, obs, cor, p)
        tot += len(obs)
        cov += len(obs) - len(r.infeasible)
        for why in r.infeasible_why:
            if "curvature budget" in why:
                budget += 1
            else:
                geo += 1
        cl += list(r.clearances)
        times.append(r.solve_ms)
        if r.ok:
            a_ok &= r.peak_kappa_nodes <= kc + 1e-9
            b_max = max(b_max, r.peak_kappa - kc)
    holds = []
    for dst in (30, 60, 85):
        obs = [Obstacle(float(ref[273, 0]), float(ref[273, 1]), BOX_R),
               Obstacle(float(ref[(273 + dst) % n, 0]), float(ref[(273 + dst) % n, 1]), BOX_R)]
        _l, _d, r = C.reoptimize_closed(ref, obs, cor, p)
        holds.append(r.hold)
    return dict(cov=cov, tot=tot, geo=geo, budget=budget, holds=holds,
                cl_min=min(cl) if cl else float("nan"),
                cl_med=float(np.median(cl)) if cl else float("nan"),
                c5a=a_ok, c5b=b_max, ms=float(np.median(times)), need=need)


def max_obs_margin(ref, cor, base):
    """The largest obs_margin that keeps every hold >= 0.40 AND the coverage of the baseline."""
    best = None
    for om in [round(x, 3) for x in np.arange(0.16, 0.32, 0.01)]:
        r = reopt_on(ref, cor, om)
        if r["cov"] < base["cov"] or any(not (h >= 0.40) for h in r["holds"]):
            break
        best = (om, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limits", nargs="+", type=float, default=[1.50, 1.35, 1.20, 1.10])
    # 0.80 REPRODUCES THE SHIPPED LINE EXACTLY -- 367 stations, peak |kappa| 1.4478 at station
    # 227, length 36.600 m, and a velocity profile whose own lap time (11.300 s) matches the
    # regenerated one to the millisecond. It is global_planner_node's declare_parameter default.
    # Not 0.25: that is reopt_safety_width, which belongs to the RE-optimizer and says nothing
    # about how the raceline itself was made. Measured across 0.15/0.25/0.40/0.60/0.80 -- peak
    # |kappa| 0.945/0.973/1.104/1.199/1.448 -- so the shipped line sits exactly on its curvature
    # limit only at 0.80.
    ap.add_argument("--safety-width", type=float, default=0.80)
    ap.add_argument("--skip-reopt", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ref0, vx0, lap_field = shipped()
    dr0, dl0 = widths(ref0[:, :2])
    ref0 = np.column_stack([ref0[:, :2], dr0, dl0])
    cor0 = corridor_from_map(ref0)
    m0 = clean_metrics(ref0, vx0)
    el0 = C._closed_el(ref0[:, :2])
    s0 = seam_check(vx0, el0)
    print(f"SHIPPED {MAP}/global_waypoints.json (untouched):")
    lap0 = m0["lap"]
    print(f"  {len(ref0)} stations | peak|kappa| {m0['peak']:.4f} @ {m0['peak_at']} | budget "
          f"{m0['budget']:.4f} | length {m0['length']:.3f} m | lap {lap0:.3f} s from its own vx")
    print(f"  NB est_lap_time in the json is {lap_field:.3f} s, which is NOT the lap time of the "
          f"vx profile stored beside it ({lap0:.3f} s). Every % below is against {lap0:.3f}.")
    print(f"  s=0 seam: vx[-1] {s0[1]:.3f} -> vx[0] {s0[2]:.3f} m/s demands {s0[0]:+.1f} m/s^2 "
          f"(worst elsewhere {s0[3]:.1f})")
    base = None if args.skip_reopt else reopt_on(ref0, cor0, 0.16)
    if base:
        print(f"  closed_qp: covered {base['cov']}/{base['tot']} (geometry {base['geo']}, budget "
              f"{base['budget']}) | clearance {base['cl_min']:.3f}/{base['cl_med']:.3f} | holds "
              + "/".join(f"{h:.3f}" for h in base["holds"]))

    rows = []
    for lim in args.limits:
        print(f"\n=== optimisation curvlim {lim:.2f} "
              f"(speed planner stays {SPEED_CURVLIM}) ===")
        try:
            traj, br, bl, est, secs = optimise(lim, args.safety_width)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            rows.append((lim, None, None, None))
            continue
        xy = traj[:-1, 1:3]
        vx = traj[:-1, 5]
        dr, dl = widths(xy)
        ref = np.column_stack([xy, dr, dl])
        m = clean_metrics(ref, vx)
        est = m["lap"]                      # like for like with the baseline, from the profile
        el = C._closed_el(xy)
        s = seam_check(vx, el)
        np.savetxt(OUT / f"raceline_curvlim_{lim:.2f}.csv",
                   np.column_stack([traj[:-1, 0], xy, traj[:-1, 3], traj[:-1, 4], vx,
                                    traj[:-1, 6], dr, dl]),
                   delimiter=",", header="s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,"
                   "d_right,d_left", comments="")
        print(f"  {len(ref)} stations | peak|kappa| {m['peak']:.4f} @ {m['peak_at']} | budget "
              f"{m['budget']:.4f} | length {m['length']:.3f} m | est lap {est:.3f} s | "
              f"solved in {secs:.1f} s")
        print(f"  s=0 seam: vx[-1] {s[1]:.3f} -> vx[0] {s[2]:.3f} m/s demands {s[0]:+.1f} m/s^2 "
              f"(worst elsewhere {s[3]:.1f})")
        r = None
        if not args.skip_reopt:
            cor = corridor_from_map(ref)
            r = reopt_on(ref, cor, 0.16)
            print(f"  closed_qp: covered {r['cov']}/{r['tot']} (geometry {r['geo']}, budget "
                  f"{r['budget']}) | clearance {r['cl_min']:.3f}/{r['cl_med']:.3f} | holds "
                  + "/".join(f"{h:.3f}" for h in r["holds"])
                  + f" | C5a {'ok' if r['c5a'] else 'NO'} C5b {r['c5b']:+.4f} | "
                    f"{r['ms']:.1f} ms")
            best = max_obs_margin(ref, cor, r)
            if best:
                om, rb = best
                head = om + 0.15 - IDLE_NEED
                print(f"  max obs_margin holding coverage and hold: {om:.2f} -> delivers "
                      f"{om + 0.15:.3f}, headroom {head:+.3f} against {REQUIRED_HEADROOM:.2f} "
                      f"-> {'PASSES' if head >= REQUIRED_HEADROOM else 'still short'}")
                r["max_om"] = (om, head)
        rows.append((lim, m, est, r))

    print("\n" + "=" * 100)
    print(f"{'opt curvlim':>11} | {'peak|k|':>8} | {'budget':>7} | {'lap s':>7} | {'vs ship':>8} | "
          f"{'covered':>8} | {'budget handovers':>16} | {'max obs_margin (headroom)':>25}")
    print(f"{'SHIPPED':>11} | {m0['peak']:8.4f} | {m0['budget']:7.4f} | {lap0:7.3f} | "
          f"{'--':>8} | " + (f"{base['cov']:3d}/{base['tot']:<4d}" if base else f"{'--':>8}")
          + f" | {base['budget'] if base else '--':>16} | "
          + (f"{'0.16 (+0.030)':>25}" if base else f"{'--':>25}"))
    for lim, m, est, r in rows:
        if m is None:
            print(f"{lim:11.2f} | {'FAILED':>8}")
            continue
        cov = f"{r['cov']:3d}/{r['tot']:<4d}" if r else "--"
        bud = f"{r['budget']}" if r else "--"
        om = (f"{r['max_om'][0]:.2f} ({r['max_om'][1]:+.3f})"
              if r and "max_om" in r else "--")
        print(f"{lim:11.2f} | {m['peak']:8.4f} | {m['budget']:7.4f} | {est:7.3f} | "
              f"{100 * (est - lap0) / lap0:+7.2f}% | {cov:>8} | {bud:>16} | {om:>25}")
    print(f"\ncandidates written to {OUT} -- NOTHING in maps/{MAP}/global_waypoints.json was "
          f"touched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
