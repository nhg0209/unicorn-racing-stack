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
  C5  peak |kappa| no worse than the CURRENT hump pipeline on the same case. No absolute limit is
      used: ifac's own raceline is at 1.448 of a 1.5 curvlim, so an absolute gate would measure
      the map. Cases the hump refuses outright are excluded and listed.
  C6  solve time p95
  C7  disc_allow_m covers the measured upsample sag -- ASSERTED ONLY at the grid it is calibrated
      for (0.50 m). At any other grid this SKIPS with a warning rather than passing quietly.
  C8  an infeasible classification says which station, which side, and how many metres short
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

    # C5 ---------------------------------------------------------------------------------------
    print("\nC5  peak |kappa| against the CURRENT hump pipeline, same cases, no absolute limit")
    full, reftrack = CMP.load_full()
    cor_hump = (np.append(cor[0], cor[0][0]), np.append(cor[1], cor[1][0]))
    cfg = str(REPO / "stack_master/config" / args.config)
    worse, excluded, compared = [], [], 0
    for name, stations in cases:
        obs = [box(ref, i) for i in stations]
        near = np.min([np.hypot(ref[:, 0] - o.x, ref[:, 1] - o.y) for o in obs], axis=0)
        sp = np.arange(max(0, min(stations) - 30), min(n, max(stations) + 31))
        h = CMP.run_hump(full, reftrack, cfg, cor_hump, obs, stations, sp, near > 2.5)
        w = CMP.run_new(ref, cor, obs, sp, near > 2.5, p)
        if not h.get("ok"):
            excluded.append(f"{name} ({h['why']})")
            continue
        compared += 1
        if w["peak"] > h["peak"] + 1e-9:
            worse.append(f"{name} {w['peak']:.3f} vs {h['peak']:.3f}")
    for e in excluded:
        print(f"         excluded, the hump refused it outright: {e}")
    g.check("C5", not worse, f"{compared - len(worse)}/{compared} cases no worse"
            + (f" -- WORSE: {'; '.join(worse)}" if worse else ""))

    # C6 ---------------------------------------------------------------------------------------
    print("\nC6  solve time")
    ts = sorted(times)
    p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))]
    g.check("C6", p95 <= 20.0, f"p50 {ts[len(ts) // 2]:.1f} ms | p95 {p95:.1f} ms | "
            f"max {ts[-1]:.1f} ms over {len(ts)} solves (limit 20 ms)")

    # C7 ---------------------------------------------------------------------------------------
    print("\nC7  disc_allow_m covers the measured upsample sag")
    if abs(p.grid_step_m - CALIBRATED_GRID_M) > 1e-9:
        g.skip("C7", f"grid_step_m is {p.grid_step_m}, and disc_allow_m = {p.disc_allow_m} was "
               f"calibrated at {CALIBRATED_GRID_M}. UNCALIBRATED GRID -- the sag scales with it "
               f"(4.11 mm at 0.50, 22.97 mm at 0.70), so re-measure before trusting the clearance")
    else:
        p0 = ReoptParams(disc_allow_m=0.0)
        worst = max(C.reoptimize_closed(ref, o, cor, p0)[2].sag_mm
                    for o in [pair] + [[box(ref, i) for i in t] for t in TRIOS])
        g.check("C7", worst <= p.disc_allow_m * 1e3,
                f"worst sag {worst:.2f} mm against an allowance of {p.disc_allow_m * 1e3:.1f} mm")

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

    print()
    if g.fails:
        print(f"FAILED: {', '.join(g.fails)}")
        return 1 if args.check else 0
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
