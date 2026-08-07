#!/usr/bin/env python3
"""Acceptance gate for the closed-track avoidance QP, in the repo's sweep convention.

    ~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/gate_closed_reopt.py --check

Prints every measurement and exits 1 if any check fails. The pytest suite next door covers the
same ground for CI; this exists because the reopt checks in this repo are `--check` scripts with
an exit code (sweep_static_reopt.py, sweep_static_feasibility.py, check_avoidance_margins.py) and
because C5 needs the CURRENT hump pipeline running beside the new core, which takes ~10 s and has
no place in a unit suite.

  C1  no obstacle -> max |d| exactly 0.0, and the input object back
  C2  every box cleared by obs_margin + w_veh/2, or classified infeasible
  C3  W2' -- a corner box outside a straight pair leaves the pair's hold, clearance and
      excursion count alone
  C4  clearance at grid 0.10 / 0.30 / 0.50 / 0.70 (CLEARANCE ONLY -- curvature is grid-dependent
      by construction and is logged, never asserted)
  C5a where the budget is ENFORCED -- at the QP's own stations -- the published |kappa| stays at
      or under the clean raceline's peak. This is the budget's actual contract; between nodes it
      binds nothing.
  C5b what the interpolation adds to the PEAK, measured against the clean raceline rather than
      against the node peak. Subtracting the node peak measures the map, not the cubic: ifac's
      apex is not a QP station, so the CLEAN line itself reads 1.2370 at nodes against 1.4478
      over all of them, and a line that added nothing would score +0.21. Anchored to the clean
      peak the number is what our line actually adds. Allowance = 2x measured, the disc_allow_m
      pattern.
  C6  solve time p95
  C7  disc_allow_m covers the measured upsample sag -- ASSERTED ONLY at the grid it is calibrated
      for (0.50 m). At any other grid this SKIPS with a warning rather than passing quietly.
  C8  an infeasible classification says which station, which side, and how many metres short
  C9  THE MERGE TAIL. Where the avoidance rejoins the raceline the offset is under a millimetre
      and its SIGN oscillates, and at ds = 0.0997 a 0.7 mm second difference is 0.07 1/m of
      published curvature -- visible in sim as ripple on a straight. Two measurements on the
      stations carrying less than 2 mm of offset: the worst |kappa_new - kappa_clean| there, and
      how many times the offset's second difference changes sign. Both fail at grid 0.50, which
      is checked here rather than assumed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/gb_optimizer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_optimizer import closed_reopt as C                    # noqa: E402
from gb_optimizer.closed_reopt import ReoptParams             # noqa: E402
from bench_closed_reopt import (A_I, B_I, TRIOS, box, corridor_from_map,  # noqa: E402
                                excursions, load_ifac, pair_span)
import compare_reopt as CMP                                   # noqa: E402

CALIBRATED_GRID_M = 0.50   # the grid disc_allow_m was measured against
# [1/m] C5b's allowance: twice the measured worst case, the same rule disc_allow_m follows. The
# measurement, over this twenty-case matrix at grid 0.50: the published peak exceeds the clean
# raceline's by at most 0.0112, on the four cases whose offset field passes over ifac's apex,
# where the periodic cubic carries 0.011 m of offset between two QP nodes whose own bound there is
# 0.009 m. 2 mm of offset, 0.0112 of curvature, 0.38% of a corner speed. Calibrated to
# grid_step_m = 0.50 like every other interpolation number here.
INTERP_PEAK_ALLOW = 0.023
# C9's two limits, both set from the measured spread rather than from the value that passes.
# Where the offset is under 2 mm the line IS the raceline, so any curvature difference there is
# manufactured: 0.0634 at grid 0.10 against 0.2457 at 0.50, and 0.10 sits between them so the
# gate can see the failure it was written for. Sign flips run 2-6 at stride 1 and 19-21 above it.
RIPPLE_KAPPA_MAX = 0.10
RIPPLE_FLIPS_MAX = 8
# [ms] C6's ceiling. It was 20 when the QP solved 74 stations; at stride 1 it solves 367 and the
# measured p95 is ~200 ms. Raised to 400 because that is still HALF the hump pipeline's measured
# p95 of 822 ms on the same twenty cases, and because the solve runs on a worker thread, never on
# the executor. Not raised to make a number pass: 200 vs 400 is the same 2x headroom 4 ms had
# against 20.
SOLVE_MS_MAX = 400.0


class Gate:
    def __init__(self):
        self.fails = []

    def check(self, tag, ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {msg}")
        if not ok:
            self.fails.append(tag)
        return ok

    def skip(self, tag, msg):
        print(f"  [SKIP] {tag}: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 on any failure")
    ap.add_argument("--config", default="SIM")
    args = ap.parse_args()

    ref = load_ifac()
    cor = corridor_from_map(ref)
    n = len(ref)
    p = ReoptParams()
    need = p.obs_margin + p.w_veh / 2.0
    g = Gate()
    times = []

    print(f"map ifac | {n} stations | grid {p.grid_step_m} | w {p.dev_weight} | obs_margin "
          f"{p.obs_margin} | disc_allow {p.disc_allow_m} | clearance floor {need:.3f}")

    # C1 ---------------------------------------------------------------------------------------
    print("\nC1  no obstacle is a no-op")
    line, d, rep = C.reoptimize_closed(ref, [], cor, p)
    g.check("C1", bool(np.shares_memory(line, ref)) and np.max(np.abs(d)) == 0.0,
            f"max |d| {np.max(np.abs(d)):.3e}, input object returned "
            f"{bool(np.shares_memory(line, ref))}")

    # C2 ---------------------------------------------------------------------------------------
    print(f"\nC2  cleared by {need:.3f} m or classified infeasible, over the whole matrix")
    cases = CMP.build_cases(n)
    bad = []
    for name, stations in cases:
        obs = [box(ref, i) for i in stations]
        _l, _d, r = C.reoptimize_closed(ref, obs, cor, p)
        times.append(r.solve_ms)
        got = min(r.clearances) if r.clearances else float("inf")
        if not (r.ok and got >= need - 1e-9):
            bad.append(f"{name} ({got:+.3f})")
    g.check("C2", not bad, f"{len(cases) - len(bad)}/{len(cases)} cases"
            + (f" -- {'; '.join(bad)}" if bad else ""))

    # C3 ---------------------------------------------------------------------------------------
    print("\nC3  W2' -- a corner box outside the pair does not change the pair")
    pair = [box(ref, A_I), box(ref, B_I)]
    span = pair_span(n)
    _l, da, ra = C.reoptimize_closed(ref, pair, cor, p)
    ea = excursions(da, span)
    for cs in (60, 120, 200):
        _l, db, rb = C.reoptimize_closed(ref, pair + [box(ref, cs)], cor, p)
        times.append(rb.solve_ms)
        eb = excursions(db, span)
        ok = rb.ok and rb.hold >= 0.40 and min(rb.clearances[:2]) >= need - 1e-9 and eb == ea
        g.check(f"C3 corner {cs}", ok, f"hold {rb.hold:.3f} (>=0.40) | pair clearance "
                f"{min(rb.clearances[:2]):+.3f} | excursions {eb} (was {ea}) | "
                f"|B-A| {np.max(np.abs(db[span] - da[span])) * 1e3:.1f} mm, logged only")

    # C4 ---------------------------------------------------------------------------------------
    print("\nC4  clearance does not depend on the grid (curvature does, and is logged)")
    for step in (0.10, 0.30, 0.50, 0.70):
        _l, _d, r = C.reoptimize_closed(ref, pair, cor, ReoptParams(grid_step_m=step))
        got = min(r.clearances) if r.clearances else float("nan")
        g.check(f"C4 grid {step:.2f}", got >= need - 1e-9,
                f"clearance {got:+.3f} | peak|kappa| {r.peak_kappa:.3f} (log) | {r.solve_ms:.1f} ms")

    # C5a / C5b ---------------------------------------------------------------------------------
    print("\nC5a / C5b  the budget where it is enforced, and what the cubic adds on top")
    full, reftrack = CMP.load_full()
    cor_hump = (np.append(cor[0], cor[0][0]), np.append(cor[1], cor[1][0]))
    cfg = str(REPO / "stack_master/config" / args.config)
    k_clean = float(np.max(np.abs(C.menger_closed(ref[:, :2]))))
    over_a, over_b, deltas = [], [], []
    for name, stations in cases:
        obs = [box(ref, i) for i in stations]
        _l, d, r = C.reoptimize_closed(ref, obs, cor, p)
        if not r.ok:
            continue
        if r.peak_kappa_nodes > k_clean + 1e-9:
            over_a.append(f"{name} {r.peak_kappa_nodes:.4f}")
        add = r.peak_kappa - k_clean
        if add > INTERP_PEAK_ALLOW:
            st = int(np.argmax(np.abs(C.menger_closed(_l))))
            over_b.append(f"{name} +{add:.4f} at station {st} carrying {abs(d[st]):.3f} m")
        near = np.min([np.hypot(ref[:, 0] - o.x, ref[:, 1] - o.y) for o in obs], axis=0)
        sp = np.arange(max(0, min(stations) - 30), min(n, max(stations) + 31))
        h = CMP.run_hump(full, reftrack, cfg, cor_hump, obs, stations, sp, near > 2.5)
        deltas.append((name, r.peak_kappa_nodes, add,
                       (h["peak"] - k_clean) if h.get("ok") else float("nan")))
    print(f"         the clean raceline peaks at {k_clean:.4f} over all stations and "
          f"{max(np.abs(C.menger_closed(ref[:, :2]))[np.arange(0, n, max(1, int(round(p.grid_step_m / np.median(C._closed_el(ref[:, :2]))))))]):.4f} at the QP's")
    print("         case                         | at nodes | published-clean | hump-clean (log)")
    for nm, nodes, dn, dh in deltas:
        print(f"           {nm:26s} | {nodes:8.4f} | {dn:+15.4f} | {dh:+.4f}")
    g.check("C5a", not over_a, f"{len(deltas) - len(over_a)}/{len(deltas)} cases at or under the "
            f"clean peak where the budget binds" + (f" -- OVER: {'; '.join(over_a)}" if over_a
                                                    else ""))
    g.check("C5b", not over_b, f"the cubic adds at most "
            f"{max((d for _n, _k, d, _h in deltas), default=0.0):+.4f} of curvature over the clean "
            f"peak against an allowance of {INTERP_PEAK_ALLOW:.4f}"
            + (f" -- OVER: {'; '.join(over_b)}" if over_b else ""))

    # C6 ---------------------------------------------------------------------------------------
    print("\nC6  solve time")
    ts = sorted(times)
    p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))]
    g.check("C6", p95 <= SOLVE_MS_MAX, f"p50 {ts[len(ts) // 2]:.1f} ms | p95 {p95:.1f} ms | "
            f"max {ts[-1]:.1f} ms over {len(ts)} solves (limit {SOLVE_MS_MAX:.0f} ms, against the "
            f"hump pipeline's measured 822 ms p95)")

    # C7 ---------------------------------------------------------------------------------------
    print("\nC7  disc_allow_m covers the measured upsample sag")
    stride = max(1, int(round(p.grid_step_m / 0.0997)))
    if stride == 1:
        worst = max(C.reoptimize_closed(ref, o, cor, p)[2].sag_mm
                    for o in [pair] + [[box(ref, i) for i in t] for t in TRIOS])
        g.check("C7", worst <= 1e-9,
                f"stride is 1 -- the QP solves every published station and the upsample is "
                f"skipped, so there is nothing between nodes to sag. Measured over the two-box "
                f"case and all twelve field placements: {worst:.4f} mm. disc_allow_m is "
                f"{p.disc_allow_m} and this check is only meaningful above stride 1.")
    elif abs(p.grid_step_m - CALIBRATED_GRID_M) > 1e-9:
        g.skip("C7", f"grid_step_m is {p.grid_step_m}, and disc_allow_m = {p.disc_allow_m} was "
               f"calibrated at {CALIBRATED_GRID_M}. UNCALIBRATED GRID -- the sag scales with it "
               f"(4.11 mm at 0.50, 22.97 mm at 0.70), so re-measure before trusting the clearance")
    else:
        # HOW MUCH OF THE ALLOWANCE THE UPSAMPLE ACTUALLY SPENDS. The QP solves to
        # need + disc_allow_m and the cubic between its stations gives some of that back, so what
        # is left in the published clearance is the headroom. Measuring it this way rather than
        # by re-solving with disc_allow_m = 0 is deliberate: with the allowance removed the
        # contract check simply hands the box to the reactive layer, which measures nothing.
        spent = -np.inf
        for o in [pair] + [[box(ref, i) for i in t] for t in TRIOS]:
            r = C.reoptimize_closed(ref, o, cor, p)[2]
            for c in r.clearances:
                spent = max(spent, p.disc_allow_m - (c - need))
        g.check("C7", spent <= p.disc_allow_m + 1e-9,
                f"the upsample spends at most {spent * 1e3:.2f} mm of the "
                f"{p.disc_allow_m * 1e3:.1f} mm allowance -- {(p.disc_allow_m - spent) * 1e3:.2f} "
                f"mm of headroom left at the calibrated grid")

    # C8 ---------------------------------------------------------------------------------------
    print("\nC8  an infeasible box is named with station, side and metres")
    seen = 0
    for trio in ((330, 30, 180), (10, 90, 190)):
        _l, _d, r = C.reoptimize_closed(ref, [box(ref, i) for i in trio], cor, p)
        for why in r.infeasible_why:
            seen += 1
            ok = ("station" in why and ("left" in why or "right" in why or "side" in why)
                  and " m " in why)
            g.check(f"C8 {trio}", ok, why)
    if not seen:
        g.check("C8", False, "no infeasible case in the matrix -- the classification is untested")

    # C9 ---------------------------------------------------------------------------------------
    print("\nC9  the merge tail: ripple where the line has all but rejoined the raceline")
    k_cln = np.abs(C.menger_closed(ref[:, :2]))

    def ripple(obs, params):
        line, d, r = C.reoptimize_closed(ref, obs, cor, params)
        if not r.ok:
            return float("nan"), 0
        quiet = np.abs(d) < 2e-3
        k_new = np.abs(C.menger_closed(line))
        worst = float(np.max(np.abs(k_new[quiet] - k_cln[quiet]))) if np.any(quiet) else 0.0
        d2 = np.roll(d, -1) - 2.0 * d + np.roll(d, 1)
        sg = np.sign(np.where(np.abs(d2) < 1e-9, 0.0, d2))[quiet]
        sg = sg[sg != 0]
        return worst, int(np.count_nonzero(np.diff(sg) != 0)) if len(sg) else 0

    CASES9 = [("1box straight 310", [310]), ("2box straight pair", [A_I, B_I]),
              ("3box (275,360,200)", [275, 360, 200])]
    for label, stations in CASES9:
        obs = [box(ref, i) for i in stations]
        w, flips = ripple(obs, p)
        g.check(f"C9 {label}", w <= RIPPLE_KAPPA_MAX and flips <= RIPPLE_FLIPS_MAX,
                f"worst |dkappa| where |d| < 2 mm = {w:.4f} (limit {RIPPLE_KAPPA_MAX:.2f}) | "
                f"second-difference sign flips {flips} (limit {RIPPLE_FLIPS_MAX})")
    print("         the same measurement across grid spacings -- the gate must SEE 0.50 fail:")
    print("         grid | ripple | flips | hold  | clearance | ms")
    for step in (0.50, 0.30, 0.20, 0.10):
        pg = ReoptParams(grid_step_m=step)
        w, flips = ripple([box(ref, 310)], pg)
        _l, _d, r2 = C.reoptimize_closed(ref, [box(ref, A_I), box(ref, B_I)], cor, pg)
        verdict = "would FAIL" if (w > RIPPLE_KAPPA_MAX or flips > RIPPLE_FLIPS_MAX) else "ok"
        print(f"         {step:4.2f} | {w:6.4f} | {flips:5d} | {r2.hold:5.3f} | "
              f"{min(r2.clearances) if r2.clearances else float('nan'):9.3f} | {r2.solve_ms:5.1f}"
              f"  <- {verdict}")

    # BRIDGE -------------------------------------------------------------------------------------
    # Not one of C1-C8: this checks the WIRING, i.e. that closed_reopt_bridge hands
    # static_reopt_node the same object reoptimize_local_window does. Everything the node's
    # vetoes read is checked for presence and shape, because a missing key there is a crash on a
    # worker thread inside a timer callback -- which is how the whole state machine went down once
    # already.
    print("\nBRIDGE  the node contract: does closed_qp return what local_window returns?")
    try:
        from gb_optimizer import closed_reopt_bridge as cqb
        from gb_optimizer import static_reopt_core as core
        full, _rt = CMP.load_full()
        obs = [core.Obstacle(float(ref[i, 0]), float(ref[i, 1]), 0.15) for i in (275, 360, 200)]
        res = cqb.reoptimize_closed_window(
            full[:, :2], full[:, 2], full[:, 3], np.zeros((4, 4)), obs,
            str(REPO / "stack_master/config" / args.config),
            w_veh=p.w_veh, clean_vx=full[:, 5], clean_kappa=full[:, 4],
            corridor_lo=np.append(cor[0], cor[0][0]), corridor_hi=np.append(cor[1], cor[1][0]))
        traj = res["main"][0]
        keys = ("main", "d_right", "d_left", "n_windows", "apex_laid", "apex_dropped",
                "kappa_published_ok", "kappa_published_max", "kappa_published_allow",
                "curvlim", "span_m", "span_budget_m", "span_reach_m", "report")
        missing = [k for k in keys if k not in res]
        g.check("BRIDGE keys", not missing, f"all {len(keys)} keys the node reads are present"
                if not missing else f"MISSING {missing}")
        g.check("BRIDGE shape", traj.shape == (len(full), 7) and len(res["d_right"]) == len(full),
                f"traj {traj.shape} against the clean line's {(len(full), 7)}; d_right "
                f"{len(res['d_right'])} -- the point COUNT is what sector_tuner indexes into")
        g.check("BRIDGE finite", bool(np.all(np.isfinite(traj))),
                f"no NaN/inf anywhere in [s,x,y,psi,kappa,vx,ax]; vx "
                f"{traj[:, 5].min():.2f}-{traj[:, 5].max():.2f} m/s, est {res['main'][3]:.3f} s")
        cov = cqb.coverage_records(obs, res, [float(np.min(np.hypot(traj[:, 1] - o.x,
                                                                   traj[:, 2] - o.y)) - o.r)
                                              for o in obs])
        g.check("BRIDGE coverage", len(cov) == len(obs) and all("status" in c for c in cov),
                "; ".join(f"({c['x']:.1f},{c['y']:.1f}) {c['status'].split(':')[0]} "
                          f"{c['clearance_m']:+.3f}" for c in cov))
        g.check("BRIDGE kappa gate", res["kappa_published_ok"],
                f"published max|kappa| {res['kappa_published_max']:.3f} against an allowance of "
                f"{res['kappa_published_allow']:.3f}")
    except Exception as exc:
        g.check("BRIDGE", False, f"{type(exc).__name__}: {exc}")

    print()
    if g.fails:
        print(f"FAILED: {', '.join(g.fails)}")
        return 1 if args.check else 0
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
