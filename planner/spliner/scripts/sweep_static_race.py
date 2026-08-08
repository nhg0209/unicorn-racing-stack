#!/usr/bin/env python3
"""What a RACE presents, what the planner does with it, and how far short it falls when it stops.

Three questions this file answers, in order, because each is the denominator of the next.

  1. THE RATE THAT MATTERS. The uniform matrix walked a box across every station, every lateral
     offset and every tracking error at equal weight, and refused 29.4 % of them. A race does not
     hand out that distribution: three boxes on the first half lap, on ground a car could
     otherwise drive, met by a car that is following the line. Filtering to that gives the refusal
     rate the competition would actually see.

  2. WHAT HAPPENS WHEN IT REFUSES. A refusal is not the end of the chain -- the planner has a
     reduced-margin retry (the squeeze), and the state machine can demand one after
     static_deadlock_timeout_s of standstill (/planner/avoidance/relax). Both are measured here,
     and the measurement is the point: the harness's own cur_vs = 3.0 sits exactly on
     squeeze_max_speed_mps = 3.0, so `v >= squeeze_max_speed_mps` held and the squeeze returned an
     empty schedule for EVERY number this campaign has reported.

  3. HOW FAR SHORT. For the cells that stay closed with both retries spent, the question stops
     being "which algorithm" and becomes "how many centimetres". min_violation answers it as a
     distance, against the floors the squeeze already has authority to reach.

Nothing here changes the planner. cur_vs and the relax flag are set on the harness's throwaway
node, where every other value is already set.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_race.py --jobs 7
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_static_feasibility as sw          # noqa: E402
import sweep_static_oracle as so               # noqa: E402
import sweep_static_corridor as sc             # noqa: E402

SQUEEZE_SPEED = 2.0      # below squeeze_max_speed_mps (3.0): the schedule is non-empty
RACE_SPEED = 3.0         # the harness default, and exactly ON the gate


def _shard(arg):
    mapname, stations, profile = arg
    H = sw.Harness(mapname)
    C = sc.Corridor(H)
    layouts = so.RACE_LAYOUTS if profile == "race" else so.OBS_LAYOUTS
    cur_ds = so.RACE_CUR_D if profile == "race" else so.CUR_D
    o = {"n_raw": 0, "n": 0, "unplaceable": 0, "node_ok": 0, "sq_ok": 0, "relax_ok": 0,
         "closed": 0, "n_empty": 0, "n_split": 0, "tube": [0] * len(sc.Corridor.DPRIME),
         "delta": [], "delta_none": 0, "sq_by_layout": Counter(), "closed_by_layout": Counter(),
         "reject_closed": Counter()}
    for (i, gap, lname, boxes, od, cd) in so.cells(H, stations, layouts, cur_ds):
        o["n_raw"] += 1
        if profile == "race" and not C.placeable(i, boxes):
            o["unplaceable"] += 1
            continue
        o["n"] += 1
        H.cur_vs, H.force_relax = RACE_SPEED, False
        if so.run(H, i, gap, cd, boxes)["ok"]:
            o["node_ok"] += 1
            continue
        # --- the two retries the chain actually has -------------------------------------------
        H.cur_vs = SQUEEZE_SPEED                       # squeeze reachable: the car has slowed
        sq = so.run(H, i, gap, cd, boxes)["ok"]
        H.cur_vs, H.force_relax = RACE_SPEED, True     # relax: the SM certified a deadlock
        rx = so.run(H, i, gap, cd, boxes)["ok"]
        H.force_relax = False
        if sq:
            o["sq_ok"] += 1
            o["sq_by_layout"][lname] += 1
        if rx:
            o["relax_ok"] += 1
        if sq or rx:
            continue
        # --- still closed: how far short, and is it shape or arithmetic ------------------------
        o["closed"] += 1
        o["closed_by_layout"][lname] += 1
        H.cur_vs = RACE_SPEED
        r = so.run(H, i, gap, cd, boxes)
        if r["reject"]:
            sig = "+".join(nm for nm, v in zip(("bounds", "obs_box", "grid", "body", "curv"),
                                               r["reject"][1:]) if v) or "none"
            o["reject_closed"][sig] += 1
        empty_any, tube, _k, _nst, _w = C.test(i, gap, cd, boxes)
        if empty_any:
            o["n_empty"] += 1
        else:
            for t, v in enumerate(tube):
                o["tube"][t] += 1 if v else 0
            if not tube[-1]:
                o["n_split"] += 1
        d = C.min_violation(i, gap, cd, boxes)
        if d is None:
            o["delta_none"] += 1
        else:
            o["delta"].append(round(d, 4))
    return o


def sweep(mapname, jobs, profile, stride=None, limit=None):
    H = sw.Harness(mapname)
    st = list(range(0, len(H.wp) - 1, stride or sw.STATION_STRIDE))
    if limit:
        st = st[:limit]
    del H
    t0 = time.time()
    if jobs <= 1:
        parts = [_shard((mapname, st, profile))]
    else:
        import multiprocessing as mp
        shards = [s for s in (st[k::jobs] for k in range(jobs)) if s]
        with mp.get_context("fork").Pool(len(shards)) as pool:
            parts = pool.map(_shard, [(mapname, s, profile) for s in shards])
    R = {"map": mapname, "profile": profile, "secs": time.time() - t0,
         "dprime": list(sc.Corridor.DPRIME)}
    for k in ("n_raw", "n", "unplaceable", "node_ok", "sq_ok", "relax_ok", "closed",
              "n_empty", "n_split", "delta_none"):
        R[k] = sum(p[k] for p in parts)
    R["tube"] = [sum(p["tube"][t] for p in parts) for t in range(len(sc.Corridor.DPRIME))]
    R["delta"] = [d for p in parts for d in p["delta"]]
    for k in ("sq_by_layout", "closed_by_layout", "reject_closed"):
        c = Counter()
        for p in parts:
            c.update(p[k])
        R[k] = c
    return R


def report(R):
    n = max(R["n"], 1)
    print(f"\n=== {R['map']} | profile {R['profile']} | {R['secs']/60:.1f} min ===")
    if R["unplaceable"]:
        print(f"  {R['n_raw']} cells generated; {R['unplaceable']} dropped because a box would "
              f"not physically fit there ({100.0*R['unplaceable']/R['n_raw']:.1f} %) -> {R['n']} "
              f"race-realistic cells")
    print(f"  planner publishes a path         {R['node_ok']:6d}  ({100.0*R['node_ok']/n:5.1f} %)")
    ref = n - R["node_ok"]
    print(f"  REFUSES at racing speed          {ref:6d}  ({100.0*ref/n:5.1f} %)   <- the rate")
    print(f"    recovered by the SQUEEZE once the car slows below "
          f"squeeze_max_speed_mps:            {R['sq_ok']:6d}  ({100.0*R['sq_ok']/max(ref,1):5.1f} % of refusals)")
    print(f"    recovered by /planner/avoidance/relax at racing speed:                  "
          f"{R['relax_ok']:6d}  ({100.0*R['relax_ok']/max(ref,1):5.1f} % of refusals)")
    print(f"  still closed with BOTH retries spent  {R['closed']:6d}  "
          f"({100.0*R['closed']/n:5.1f} % of all cells, {100.0*R['closed']/max(ref,1):5.1f} % of refusals)")
    c = max(R["closed"], 1)
    print(f"    (i)  a station with no lane at all   {R['n_empty']:6d}  ({100.0*R['n_empty']/c:5.1f} %)")
    print(f"    (ii) lanes but no tube               {R['n_split']:6d}  ({100.0*R['n_split']/c:5.1f} %)")
    for dp, t in zip(R["dprime"], R["tube"]):
        print(f"    (iii) tube at |d'| <= {dp:.1f}          {t:6d}  ({100.0*t/c:5.1f} %)")
    d = np.array(R["delta"]) if R["delta"] else np.array([])
    print("  HOW FAR SHORT (delta off BOTH safety_margin_d and wall_margin, half_car untouched).")
    print("    The squeeze's own floors are safety 0.05 and wall 0.08, so moving both together it "
          "can spend 7 cm before the WALL floor binds -- that is the line that matters, not 10.")
    if len(d):
        n0 = int((d <= 1e-9).sum())
        print(f"    delta = 0: a tube exists at the FULL design margins  {n0:6d}  "
              f"({100.0*n0/c:5.1f} % of closed)   -- nothing to give up; this is shape, not margin")
        print(f"    over the rest: median {100*np.median(d[d > 1e-9]) if (d > 1e-9).any() else 0:5.1f} cm"
              f" | p90 {100*np.percentile(d[d > 1e-9], 90) if (d > 1e-9).any() else 0:5.1f} cm")
        for a, b, lab in ((1e-9, .02, ""), (.02, .05, ""),
                          (.05, .07, "  <- the squeeze can still reach here"),
                          (.07, .10, "  <- past the squeeze's WALL floor"),
                          (.10, .1501, "  <- past every floor it has")):
            k = int(((d > a) & (d <= b)).sum())
            print(f"    {100*a:4.0f} - {100*b:3.0f} cm   {k:6d}  ({100.0*k/c:5.1f} % of closed){lab}")
    print(f"    needs MORE than 15 cm, i.e. more margin than exists  {R['delta_none']:6d}  "
          f"({100.0*R['delta_none']/c:5.1f} %)")
    print("  what killed every candidate in the still-closed cells: " +
          ", ".join(f"{k}={v}" for k, v in R["reject_closed"].most_common(4)))
    print("  by layout | refused-and-closed: " +
          ", ".join(f"{k}={v}" for k, v in sorted(R["closed_by_layout"].items())) +
          "  | squeeze recovered: " +
          ", ".join(f"{k}={v}" for k, v in sorted(R["sq_by_layout"].items())))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="*", default=["ifac_0807", "ifac"])
    ap.add_argument("--profile", default="race", choices=("race", "uniform"))
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    allR = []
    for m in a.map:
        R = sweep(m, a.jobs, a.profile, a.stride, a.limit)
        report(R)
        allR.append(R)
    if a.out:
        Path(a.out).write_text(json.dumps(
            [{k: (dict(v) if isinstance(v, Counter) else v) for k, v in R.items()}
             for R in allR], indent=1, default=float))
        print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
