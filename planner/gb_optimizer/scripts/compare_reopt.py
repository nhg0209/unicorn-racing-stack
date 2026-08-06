#!/usr/bin/env python3
"""The hump pipeline and the closed-track QP, same map, same boxes, side by side.

Everything before this measured whether the new core meets its own spec. This measures whether it
is better than what ships, which is the only question a default switch turns on.

  CURRENT   static_reopt_core.reoptimize_local_window -- the C2 quintic hump per reactive apex,
            at the values base_system.launch.xml actually ships (obs_margin 0.35, relax_floor
            0.33, wall_margin 0.05, reach 1.0-6.0, fit_tol 0.005, merge 2.0, hold 8.0/0.3).
  NEW       closed_reopt.reoptimize_closed at its own defaults.

NEITHER SIDE IS ADJUSTED TO LOOK GOOD. In particular the two systems do not owe the same
clearance: the hump is built to clear a box edge by 0.35 m and may relax to 0.33, the QP to
obs_margin + w_veh/2 = 0.30. So the hump clearing MORE is not the QP losing, and the hump
REFUSING is not a measurement error -- it is the behaviour that sends the obstacle back to the
reactive layer for every lap. Refusals are counted as refusals and reported as such.

Every metric is measured on the PUBLISHED geometry of each side with the same function, never
read out of either core's own report:

  clearance   min over the published line of |p - obstacle| - r   [m, centre to edge]
  peak|kappa| core._menger_kappa of the published line            [1/m, real geometry]
  excursions  sign changes of the lateral offset over the span of interest
  hold        min |offset| strictly between two same-side boxes   [m]
  far |d|     median |offset| more than 2.5 m from every box      [m]
  ms          wall clock of the one call

  ~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/compare_reopt.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/gb_optimizer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_optimizer import closed_reopt as C                    # noqa: E402
from gb_optimizer import static_reopt_core as core            # noqa: E402
from gb_optimizer.closed_reopt import Obstacle, ReoptParams   # noqa: E402
from bench_closed_reopt import corridor_from_map, load_ifac   # noqa: E402

MAP = "ifac"
BOX_R = 0.15
# the reactive apex the hump pipeline reshapes: obstacle radius + the reactive keep-out
# (width_car/2 + safety_margin_d = 0.15 + 0.15) + the spliner's apex bulge 0.10
APEX_OFFSET_M = 0.55
# base_system.launch.xml, verbatim
HUMP = dict(obs_margin=0.35, relax_floor=0.33, wall_margin=0.05, w_veh=0.30,
            reach_time=0.0, reach_min=1.0, reach_max=6.0, fit_tol=0.005,
            apex_merge_gap_m=2.0, hold_max_gap_m=8.0, hold_kappa_max=0.3)

# ifac at 0.0997 m/station. Classified by |kappa| smoothed over +-0.5 m: the four flattest
# well-separated stations and the four sharpest (1.14 / 0.56 / 0.42 / 0.39 against 0.23).
STRAIGHT = (0, 99, 135, 273)
CORNER = (227, 119, 63, 186)
GAP_STATIONS = {3.0: 30, 6.0: 60, 8.5: 85}
TRIOS6 = ((275, 360, 200), (100, 160, 260), (200, 260, 40),
          (250, 310, 90), (330, 30, 180), (10, 90, 190))   # last two carry an unavoidable box


def load_full():
    """The published raceline WITH its duplicated closing point -- the hump core wants the loop
    closed, closed_reopt wants it open, and the difference is one row."""
    d = json.load(open(REPO / "stack_master/maps" / MAP / "global_waypoints.json"))
    wp = d["global_traj_wpnts_iqp"]["wpnts"]
    a = np.array([[w["x_m"], w["y_m"], w["d_right"], w["d_left"], w["kappa_radpm"], w["vx_mps"]]
                  for w in wp], float)
    reftrack = np.genfromtxt(REPO / "stack_master/maps" / MAP / "centerline.csv",
                             delimiter=",", skip_header=1)
    return a, reftrack


def offsets_of(pub_xy, clean_xy, nvec):
    """Signed lateral offset of a published line, per clean station, by nearest point.

    The hump core republishes on its own uniform arc-length grid, so the two lines cannot be
    subtracted row by row. Projecting each clean station onto the nearest published point and
    reading it in that station's own normal is what makes the offsets comparable at all.
    """
    d = np.empty(len(clean_xy))
    for i, (p, nv) in enumerate(zip(clean_xy, nvec)):
        j = int(np.argmin(np.hypot(pub_xy[:, 0] - p[0], pub_xy[:, 1] - p[1])))
        d[i] = float((pub_xy[j] - p) @ nv)
    return d


def metrics(pub_xy, d, obs, span, far_mask, floor):
    """Per-obstacle clearance, and how many of them the line actually covers.

    COVERAGE IS THE COMPARABLE QUANTITY, not the minimum clearance. Both systems hand boxes back
    to the reactive layer -- the hump by refusing a window, the QP by naming the box infeasible --
    and a line that avoids two of three boxes has a terrible minimum clearance for the third and a
    perfect one for the two it took. Counting how many boxes each line clears BY ITS OWN FLOOR is
    the only way to put them on the same axis.
    """
    k = np.abs(core._menger_kappa(pub_xy))
    cl = [float(np.min(np.hypot(pub_xy[:, 0] - o.x, pub_xy[:, 1] - o.y)) - o.r) for o in obs]
    sig = np.sign(np.where(np.abs(d[span]) < 0.02, 0.0, d[span]))
    sig = sig[sig != 0]
    exc = int(np.count_nonzero(np.diff(sig) != 0)) if len(sig) else 0
    return dict(clear=min(cl), clears=cl, cover=int(sum(c >= floor - 1e-9 for c in cl)),
                peak=float(np.max(k)), exc=exc,
                far=float(np.median(np.abs(d[far_mask]))) if np.any(far_mask) else float("nan"))


def run_new(ref, cor, obs, span, far_mask, params):
    t0 = time.perf_counter()
    line, d, rep = C.reoptimize_closed(ref, obs, cor, params)
    ms = (time.perf_counter() - t0) * 1e3
    if not rep.ok:
        return dict(ok=False, why=rep.reason or "solver refused", ms=ms,
                    ninf=len(rep.infeasible), cover=0)
    m = metrics(line, d, obs, span, far_mask, params.obs_margin + 0.5 * params.w_veh)
    m.update(ok=True, ms=ms, ninf=len(rep.infeasible), hold=rep.hold,
             why=(rep.infeasible_why[0][:60] if rep.infeasible_why else ""))
    return m


def run_hump(full, reftrack, cfg, corr, obs, stations, span, far_mask):
    xy, dr, dl = full[:, :2], full[:, 2], full[:, 3]
    nvec = core._wrap_normals(xy)
    lo, hi = corr
    apexes, apex_obs = [], []
    for i, o in zip(stations, obs):
        room_hi = hi[i] if np.isfinite(hi[i]) else dr[i]
        room_lo = -lo[i] if np.isfinite(lo[i]) else dl[i]
        side = 1.0 if room_hi >= room_lo else -1.0      # the reactive planner takes the roomier side
        apexes.append(tuple(xy[i] + side * APEX_OFFSET_M * nvec[i]))
        apex_obs.append((float(o.x), float(o.y), float(o.r)))
    t0 = time.perf_counter()
    try:
        res = core.reoptimize_local_window(
            xy, dr, dl, reftrack, apexes, cfg,
            params=core.ModulationParams(obs_margin=HUMP["obs_margin"]),
            w_veh=HUMP["w_veh"], clean_vx=full[:, 5], wall_margin=HUMP["wall_margin"],
            reach_time=HUMP["reach_time"], reach_min=HUMP["reach_min"],
            reach_max=HUMP["reach_max"], clean_kappa=full[:, 4], fit_tol=HUMP["fit_tol"],
            apex_obstacles=apex_obs, relax_floor=HUMP["relax_floor"],
            apex_merge_gap_m=HUMP["apex_merge_gap_m"], hold_max_gap_m=HUMP["hold_max_gap_m"],
            hold_kappa_max=HUMP["hold_kappa_max"], corridor_lo=lo, corridor_hi=hi)
    except Exception as exc:
        return dict(ok=False, why=f"exception:{type(exc).__name__}",
                    ms=(time.perf_counter() - t0) * 1e3, ndrop=len(obs), cover=0)
    ms = (time.perf_counter() - t0) * 1e3
    ndrop = len(res.get("apex_dropped", []))
    if res["n_windows"] == 0:
        why = (res["apex_dropped"][0].get("reason", "?") if res.get("apex_dropped") else "no window")
        return dict(ok=False, why=str(why)[:60], ms=ms, ndrop=ndrop, cover=0)
    pub = res["main"][0][:, 1:3]
    d = offsets_of(pub, xy[:-1], nvec[:-1])
    m = metrics(pub, d, obs, span, far_mask, HUMP["relax_floor"])
    why = ""
    if ndrop:
        why = str(res["apex_dropped"][0].get("reason", "?"))[:60]
    m.update(ok=True, ms=ms, ndrop=ndrop, why=why)
    return m


def hold_between(d, ia, ib, n):
    span = (np.arange(ia + 1, ib) if ib > ia else np.arange(ia + 1, ib + n)) % n
    return float(np.min(np.abs(d[span]))) if len(span) else float("nan")


def build_cases(n):
    cases = []
    for s in STRAIGHT:
        cases.append((f"1box straight {s}", [s]))
    for s in CORNER:
        cases.append((f"1box corner   {s}", [s]))
    for gap, dst in GAP_STATIONS.items():
        cases.append((f"2box straight gap {gap:.1f}m", [273, (273 + dst) % n]))
    for a, b in ((227, 257), (119, 149), (63, 103)):
        cases.append((f"2box corner+straight {a}", [a, b % n]))
    for t in TRIOS6:
        cases.append((f"3box {t}", list(t)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="SIM")
    args = ap.parse_args()
    cfg = str(REPO / "stack_master/config" / args.config)

    ref = load_ifac()
    full, reftrack = load_full()
    n = len(ref)
    cor_new = corridor_from_map(ref)
    # the hump core is handed the corridor on ITS grid and in ITS normal basis -- same map, same
    # erosion, same wall_margin; measuring it once and reindexing would hand one side a corridor
    # measured in the other's frame
    cor_hump = (np.append(cor_new[0], cor_new[0][0]), np.append(cor_new[1], cor_new[1][0]))
    p = ReoptParams()
    need_new = p.obs_margin + p.w_veh / 2.0
    _psi, nv, _t = C._frame(ref[:, :2])

    print(f"map {MAP} | {n} stations | hump: obs_margin {HUMP['obs_margin']} relax_floor "
          f"{HUMP['relax_floor']} (its own floor)")
    print(f"                          | new : obs_margin {p.obs_margin} + w_veh/2 -> clearance "
          f"{need_new:.3f}, w {p.dev_weight}, grid {p.grid_step_m}")
    print()
    hdr = (f"{'case':28s} | {'covered':^9s} | {'min clear (covered)':^19s} | "
           f"{'peak|kappa|':^15s} | {'exc':^7s} | {'far |d|':^13s} | {'ms':^13s}")
    print(hdr)
    print(f"{'':28s} | {'hump':>4s} {'new':>4s} | {'hump':>9s} {'new':>9s} | {'hump':>7s} "
          f"{'new':>7s} | {'h':>3s} {'n':>3s} | {'hump':>6s} {'new':>6s} | {'hump':>6s} {'new':>6s}")
    print("-" * len(hdr))

    rows = []
    for name, stations in build_cases(n):
        obs = [Obstacle(float(ref[i, 0]), float(ref[i, 1]), BOX_R) for i in stations]
        near = np.min([np.hypot(ref[:, 0] - o.x, ref[:, 1] - o.y) for o in obs], axis=0)
        far_mask = near > 2.5
        lo_i, hi_i = min(stations), max(stations)
        span = np.arange(max(0, lo_i - 30), min(n, hi_i + 31))
        h = run_hump(full, reftrack, cfg, cor_hump, obs, stations, span, far_mask)
        nw = run_new(ref, cor_new, obs, span, far_mask, p)
        rows.append((name, stations, h, nw))

        def f(m, k, w, prec=3):
            return f"{m[k]:>{w}.{prec}f}" if m.get("ok") and k in m else f"{'--':>{w}s}"

        def cov(m):
            return f"{m.get('cover', 0):d}/{len(obs):d}"

        def mincov(m, floor):
            got = [c for c in m.get("clears", []) if c >= floor - 1e-9] if m.get("ok") else []
            return f"{min(got):>9.3f}" if got else f"{'--':>9s}"
        print(f"{name:28s} | {cov(h):>4s} {cov(nw):>4s} | {mincov(h, HUMP['relax_floor'])} "
              f"{mincov(nw, need_new)} | {f(h,'peak',7)} {f(nw,'peak',7)}"
              f" | {f(h,'exc',3,0)} {f(nw,'exc',3,0)} | {f(h,'far',6)} {f(nw,'far',6)} | "
              f"{h['ms']:6.1f} {nw['ms']:6.1f}")
        if not h.get("ok"):
            print(f"{'':28s} | hump REFUSED: {h['why']}")
        elif h.get("ndrop"):
            print(f"{'':28s} | hump dropped {h['ndrop']} of {len(obs)}: {h['why']}")
        if not nw.get("ok"):
            print(f"{'':28s} | new  REFUSED: {nw['why']}")
        elif nw.get("ninf"):
            print(f"{'':28s} | new  infeasible {nw['ninf']} of {len(obs)}: {nw['why']}")

    # --- summary ------------------------------------------------------------------------------
    # A refusal and an infeasible classification are the SAME outcome -- the box goes back to the
    # reactive layer -- so neither counts as solving anything. What is compared is how many boxes
    # each line covers by its own floor, and only then what the line costs.
    win, same, lose = [], [], []
    for nm, _s, h, w in rows:
        ch, cn = h.get("cover", 0), w.get("cover", 0)
        if cn > ch:
            win.append((nm, h, w))
        elif cn < ch:
            lose.append((nm, h, w))
        elif h.get("ok") and w.get("ok") and w["peak"] > h["peak"] + 0.02:
            lose.append((nm, h, w))
        elif h.get("ok") and w.get("ok") and w["peak"] < h["peak"] - 0.02:
            win.append((nm, h, w))
        else:
            same.append((nm, h, w))
    tot_h = sum(h.get("cover", 0) for _n, _s, h, _w in rows)
    tot_n = sum(w.get("cover", 0) for _n, _s, _h, w in rows)
    tot_b = sum(len(s) for _n, s, _h, _w in rows)

    print()
    print(f"BOXES COVERED, all {len(rows)} cases: hump {tot_h}/{tot_b} (its floor "
          f"{HUMP['relax_floor']:.2f}) | new {tot_n}/{tot_b} (its floor {need_new:.2f})")
    print(f"CASES: new better {len(win)} | equivalent {len(same)} | worse {len(lose)}")
    for tag, group in (("BETTER", win), ("WORSE", lose)):
        for nm, h, w in group:
            print(f"  {tag:6s} {nm:26s} covered {h.get('cover',0)}->{w.get('cover',0)} | peak "
                  f"{h.get('peak', float('nan')):.3f}->{w.get('peak', float('nan')):.3f} | far "
                  f"{h.get('far', float('nan')):.3f}->{w.get('far', float('nan')):.3f}")
    fars = [w["far"] for _n, _s, _h, w in rows if w.get("ok")]
    print(f"far |d| (median |offset| more than 2.5 m from every box): hump 0.000 everywhere by "
          f"construction | new median {np.median(fars):.3f}, worst {max(fars):.3f}")
    ms_h = sorted(h["ms"] for _n, _s, h, _w in rows)
    ms_n = sorted(w["ms"] for _n, _s, _h, w in rows)
    print(f"  solve time p50/p95: hump {ms_h[len(ms_h)//2]:.1f}/{ms_h[int(.95*len(ms_h))]:.1f} ms | "
          f"new {ms_n[len(ms_n)//2]:.1f}/{ms_n[int(.95*len(ms_n))]:.1f} ms")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.exit(main())
