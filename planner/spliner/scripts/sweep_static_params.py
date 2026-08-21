#!/usr/bin/env python3
"""What the two sampling knobs would BUY and what they would COST. Measurement only.

sweep_static_oracle located the whole measured gap on two axes the node already searches:
of 637 cells where a path existed and the planner published none, 76 % had the answer BETWEEN
two of the ~6 terminal offsets it sprays (median spacing 0.150 m) and 23 % sat on one of those
offsets but needed a ramp length its ladder does not carry. Both are parameters. This file prices
them; it does not set them.

Pricing is the whole point, because the two are not independent and the coupling is adverse.
static_avoidance_params.yaml already records it:

    "Do NOT raise n_d_samples alongside it: that takes a rung from 4.31 to 6.53 ms, and the
     budget then cannot even afford rung 1."

ramp_search_max_ms is a budget for the WHOLE ladder, checked before each rung. More terminal
offsets makes every rung more expensive, so raising n_d_samples spends the ladder's budget --
past some point the extra offsets buy fewer cells than the rungs they evict. So each
configuration is run at the SHIPPED budget and reports, next to the cells it wins, how many
re-plans it actually got through.

TIMING IS SINGLE-THREADED, deliberately: the gate's p50/p95 are, and a p95 measured against six
other workers is a number about this machine's scheduler. The benefit sweep, which is a count and
not a duration, is parallel.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_params.py \\
      --missed <oracle_matrix.json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_static_feasibility as sw          # noqa: E402
import sweep_static_oracle as so               # noqa: E402

SHIPPED_ENTRY = [3.15, 2.5, 2.0, 1.5, 1.0]
SHIPPED_EXIT = [4.5, 2.5, 1.5]
# Filled between the shipped rungs, never outside them: the question is resolution on this axis,
# and adding a rung longer than 4.5 m or shorter than 1.0 m would be a different change.
MID_ENTRY = [4.5, 3.15, 2.5, 2.0, 1.5, 1.0]
MID_EXIT = [4.5, 3.0, 2.5, 2.0, 1.5]
DENSE_ENTRY = [4.5, 4.0, 3.5, 3.15, 3.0, 2.5, 2.0, 1.5, 1.0]
DENSE_EXIT = [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]

CONFIGS = [
    ("shipped",            10, SHIPPED_ENTRY, SHIPPED_EXIT),
    ("n_d 14",             14, SHIPPED_ENTRY, SHIPPED_EXIT),
    ("n_d 20",             20, SHIPPED_ENTRY, SHIPPED_EXIT),
    ("n_d 30",             30, SHIPPED_ENTRY, SHIPPED_EXIT),
    ("n_d 40",             40, SHIPPED_ENTRY, SHIPPED_EXIT),
    ("rung 6x5",           10, MID_ENTRY,     MID_EXIT),
    ("rung 9x8",           10, DENSE_ENTRY,   DENSE_EXIT),
    ("n_d 20 + rung 6x5",  20, MID_ENTRY,     MID_EXIT),
    ("n_d 20 + rung 9x8",  20, DENSE_ENTRY,   DENSE_EXIT),
    ("n_d 30 + rung 6x5",  30, MID_ENTRY,     MID_EXIT),
]


def run_cfg(H, i, gap, cur_d, boxes, n_d, entry, exit_, max_ms):
    """One plan under one configuration, at the SHIPPED ladder budget.

    Returns (ok, ms, n_plans). n_plans counts do_spline invocations that reached the sampling
    stage -- the main pass plus every ladder rung and squeeze step that actually ran, which is
    what the budget is spent on.
    """
    n = H.make_node(i, gap, cur_d, ladder=True, max_ms=max_ms, boxes=boxes)
    n.n_d_samples = int(n_d)
    n.ramp_search_entry_m = list(entry)
    n.ramp_search_exit_m = list(exit_)
    cnt = [0]
    orig = n._gate_samples

    def hook(*a, **k):
        cnt[0] += 1
        return orig(*a, **k)

    n._gate_samples = hook
    t0 = time.perf_counter()
    try:
        res = n.do_spline(H.gbw)
    except Exception:
        return False, (time.perf_counter() - t0) * 1e3, cnt[0]
    ms = (time.perf_counter() - t0) * 1e3
    ok = res is not None and res[0] is not None and len(res[0].wpnts) > 0
    return ok, ms, cnt[0]


def _benefit_shard(arg):
    """How many of the cells a config recovers, over the list of cells handed to it."""
    mapname, cells, cfgs, max_ms = arg
    H = sw.Harness(mapname)
    out = {name: 0 for name, *_ in cfgs}
    for (i, gap, boxes, cur_d) in cells:
        for name, n_d, e, x in cfgs:
            ok, _ms, _n = run_cfg(H, i, gap, cur_d, boxes, n_d, e, x, max_ms)
            out[name] += 1 if ok else 0
    return out


def benefit(mapname, missed, cfgs, max_ms, jobs):
    """Recovered cells among the ones sweep_static_oracle proved a path existed for."""
    cells = []
    for m in missed:
        boxes = tuple((o, m["obs_d"]) for o in so.OBS_LAYOUTS[m["layout"]])
        cells.append((m["station"], m["gap"], boxes, m["cur_d"]))
    if not cells:
        return {name: 0 for name, *_ in cfgs}, 0
    if jobs <= 1:
        return _benefit_shard((mapname, cells, cfgs, max_ms)), len(cells)
    import multiprocessing as mp
    shards = [s for s in (cells[k::jobs] for k in range(jobs)) if s]
    with mp.get_context("fork").Pool(len(shards)) as pool:
        parts = pool.map(_benefit_shard, [(mapname, s, cfgs, max_ms) for s in shards])
    out = {name: sum(p[name] for p in parts) for name, *_ in cfgs}
    return out, len(cells)


def cost(H, cells, cfgs, max_ms):
    """Loop time and rungs actually run, single-threaded."""
    out = {}
    for name, n_d, e, x in cfgs:
        ts, oks, plans = [], 0, []
        for (i, gap, boxes, cur_d) in cells:
            ok, ms, npl = run_cfg(H, i, gap, cur_d, boxes, n_d, e, x, max_ms)
            ts.append(ms)
            plans.append(npl)
            oks += 1 if ok else 0
        out[name] = {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95)),
                     "max": float(np.max(ts)), "ok": oks, "n": len(cells),
                     "plans_p50": float(np.percentile(plans, 50)),
                     "plans_max": int(np.max(plans))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="*", default=["ifac", "ifac_0807"])
    ap.add_argument("--missed", required=True, help="oracle_matrix.json")
    ap.add_argument("--jobs", type=int, default=7, help="benefit sweep only; timing is serial")
    ap.add_argument("--cost-stride", type=int, default=40,
                    help="every Nth matrix cell enters the timing set")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    P = sw.load_params()
    max_ms = float(P["ramp_search_max_ms"])
    print(f"ladder budget in use: ramp_search_max_ms = {max_ms} ms "
          f"(static_avoidance_params.yaml), loop period 50 ms")
    print(f"shipped n_d_samples = {P['n_d_samples']}, "
          f"entry {P['ramp_search_entry_m']} x exit {P['ramp_search_exit_m']} "
          f"= {len(P['ramp_search_entry_m']) * len(P['ramp_search_exit_m'])} rungs")

    missed_by_map = {}
    for r in json.load(open(a.missed)):
        missed_by_map[r["map"]] = r["missed"]

    results = {}
    for m in a.map:
        H = sw.Harness(m)
        # TIMING SET: the corner grid, which is what the shipped gate times and where the ladder
        # actually runs, plus a stride through the full matrix so p95 is not a corner artefact.
        corner_cells = [(i, g, ((0.0, 0.0),), 0.0) for i in H.corners for g in (12.0, 8.0, 4.0)]
        matrix_cells = [(i, gap, boxes, cd) for k, (i, gap, ln, boxes, od, cd)
                        in enumerate(so.cells(H, H.stations)) if k % a.cost_stride == 0]
        print(f"\n########## {m} ##########")
        for label, cells in (("corner grid", corner_cells),
                             (f"matrix / {a.cost_stride}", matrix_cells)):
            c = cost(H, cells, CONFIGS, max_ms)
            print(f"\n--- COST | {m} | {label} | {len(cells)} cells | serial ---")
            print(f"  {'config':20s} {'p50':>7s} {'p95':>7s} {'max':>7s}   "
                  f"{'re-plans p50':>12s} {'max':>5s}   {'feasible':>9s}")
            for name, *_ in CONFIGS:
                v = c[name]
                flag = "  <- over the 50 ms period" if v["p95"] > 50 else (
                    "  <- over the 40 ms gate" if v["p95"] > 40 else "")
                print(f"  {name:20s} {v['p50']:6.1f}  {v['p95']:6.1f}  {v['max']:6.1f}   "
                      f"{v['plans_p50']:12.0f} {v['plans_max']:5d}   "
                      f"{v['ok']:4d}/{v['n']:<4d}{flag}")
            results[f"{m}|{label}"] = c
        del H
        b, nb = benefit(m, missed_by_map.get(m, []), CONFIGS, max_ms, a.jobs)
        print(f"\n--- BENEFIT | {m} | the {nb} cells the oracle proved a path for ---")
        for name, *_ in CONFIGS:
            print(f"  {name:20s} recovers {b[name]:4d}/{nb:<4d} "
                  f"({100.0*b[name]/max(nb,1):5.1f} %)")
        results[f"{m}|benefit"] = {"recovered": b, "n": nb}

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1, default=float))
        print(f"\nwritten: {a.out}")
    print("\nNo value was applied. static_avoidance_params.yaml is untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
