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

And a fourth, since round 6: WHICH SHAPE. `--method` runs the same cells through both d(s)
generators -- the sampled quintic and the corridor QP -- with the same knots, the same sides, the
same gates and the same two retries, so the difference between the two columns is one branch of the
planner and nothing else. The five things it reports are named R1-R5 in the round's brief:

  R1  refusal rate, per map, per method                       (the r1_table at the end)
  R2  |d'| step at the one splice the published array has     (shape_report; gated at 0.05)
  R3  the speed the shape's own curvature allows              (shape_report; REPORTED, never gated
      -- a slower real path is the trade against a refusal, and a refusal behind a stationary box
      is a TRAILING standstill)
  R4  what a planning pass costs                              (shape_report; --jobs 1, see below)
  R5  the sampled path's own regression                       (NOT here: it is a cell-by-cell
      comparison against the previous commit, and belongs to whatever is being compared)

R4 IS WALL CLOCK. Under --jobs > 1 the shards contend and the timing lines describe the machine,
not the planner; shape_report says so on every such run. Take timings with --jobs 1.

Nothing here changes the planner. cur_vs, the relax flag and the method are set on the harness's
throwaway node, where every other value is already set.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_race.py --jobs 7
  ... --method all          # sample, corridor_qp, corridor_qp with the apex bands pinned
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

A_LAT_MAX = 6.0          # sweep_static_feasibility's own, for the curvature-limited speed cap
SEAM_DPRIME_MAX = 0.05   # R2: |d'| step the splice at s_entry0 may carry
LADDER_BUDGET_MS = 20.0  # static_avoidance_params.yaml: ramp_search_max_ms -- one rung must fit
CYCLE_MS = 50.0          # 20 Hz


def _shape_stats(H, r, o):
    """What the PUBLISHED profile costs: the splice at its one seam, its bending, and the speed its
    curvature allows. Recorded for whichever pass actually published.

      seam      |d'| step at s_entry0, one-sided differences on the station grid -- the only splice
                in the array. (The apex "seams" of the ceiling were artefacts of a per-segment
                splice; one continuous solve has nothing to join there. The exit is a boundary
                condition, not a splice, and with tail_m = 0 it is not in the published grid at all.)
      dpp       max |d'| step over the whole maneuver: the kink measure with nowhere to hide, so a
                seam number cannot be made to look good by choosing where to stand.
      vcap      sqrt(a_lat / peak|kappa|), the speed the shape's own curvature allows. This is the
                price of the trade, and it is reported, never gated.
    """
    d, kap, s0 = r.get("d_pub"), r.get("kappa_pub"), r.get("s_entry0")
    if d is None or kap is None or len(d) < 5:
        return
    ds = H.wp[1]["s_m"] - H.wp[0]["s_m"]
    o["vcap"].append(float(np.sqrt(A_LAT_MAX / max(np.max(np.abs(kap)), 1e-3))))
    o["kpeak"].append(float(np.max(np.abs(kap))))
    j0 = 1 if s0 is None else int(np.clip(round(float(s0) / ds), 1, len(d) - 2))
    dpp = np.abs(np.diff(d, 2)) / ds
    o["seam"].append(float(dpp[j0 - 1]))
    o["dpp"].append(float(np.max(dpp[j0 - 1:])) if j0 - 1 < len(dpp) else 0.0)


def _shard(arg):
    mapname, stations, profile, method, pin_apex = arg
    H = sw.Harness(mapname)
    H.plan_method, H.pin_apex = method, pin_apex
    C = sc.Corridor(H)
    layouts = so.RACE_LAYOUTS if profile == "race" else so.OBS_LAYOUTS
    cur_ds = so.RACE_CUR_D if profile == "race" else so.CUR_D
    o = {"n_raw": 0, "n": 0, "unplaceable": 0, "node_ok": 0, "sq_ok": 0, "relax_ok": 0,
         "closed": 0, "n_empty": 0, "n_split": 0, "tube": [0] * len(sc.Corridor.DPRIME),
         "delta": [], "delta_none": 0, "sq_by_layout": Counter(), "closed_by_layout": Counter(),
         "reject_closed": Counter(),
         "ms": [], "ms1": [], "seam": [], "dpp": [], "vcap": [], "kpeak": []}
    for (i, gap, lname, boxes, od, cd) in so.cells(H, stations, layouts, cur_ds):
        o["n_raw"] += 1
        if profile == "race" and not C.placeable(i, boxes):
            o["unplaceable"] += 1
            continue
        o["n"] += 1
        H.cur_vs, H.force_relax = RACE_SPEED, False
        # R4 wants what ONE pass costs, which is not what a cell costs: the timed run below has the
        # ladder off, so the number is the planning pass itself and can be held against the rung
        # budget the ladder spends it out of. The feasibility verdict still comes from the ladder-on
        # run, exactly as every earlier number in this campaign.
        o["ms1"].append(so.run(H, i, gap, cd, boxes, ladder=False)["ms"])
        r0 = so.run(H, i, gap, cd, boxes)
        o["ms"].append(r0["ms"])
        if r0["ok"]:
            o["node_ok"] += 1
            _shape_stats(H, r0, o)
            continue
        # --- the two retries the chain actually has -------------------------------------------
        H.cur_vs = SQUEEZE_SPEED                       # squeeze reachable: the car has slowed
        r_sq = so.run(H, i, gap, cd, boxes)
        sq = r_sq["ok"]
        H.cur_vs, H.force_relax = RACE_SPEED, True     # relax: the SM certified a deadlock
        r_rx = so.run(H, i, gap, cd, boxes)
        rx = r_rx["ok"]
        H.force_relax = False
        if sq:
            o["sq_ok"] += 1
            o["sq_by_layout"][lname] += 1
        if rx:
            o["relax_ok"] += 1
        if sq or rx:
            _shape_stats(H, r_sq if sq else r_rx, o)
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


def sweep(mapname, jobs, profile, stride=None, limit=None, method=None, pin_apex=None):
    H = sw.Harness(mapname)
    st = list(range(0, len(H.wp) - 1, stride or sw.STATION_STRIDE))
    if limit:
        st = st[:limit]
    del H
    t0 = time.time()
    args = (mapname, st, profile, method, pin_apex)
    if jobs <= 1:
        parts = [_shard(args)]
    else:
        import multiprocessing as mp
        shards = [s for s in (st[k::jobs] for k in range(jobs)) if s]
        with mp.get_context("fork").Pool(len(shards)) as pool:
            parts = pool.map(_shard, [(mapname, s, profile, method, pin_apex) for s in shards])
    R = {"map": mapname, "profile": profile, "secs": time.time() - t0, "jobs": jobs,
         "method": method or "yaml", "pin_apex": bool(pin_apex),
         "dprime": list(sc.Corridor.DPRIME)}
    for k in ("n_raw", "n", "unplaceable", "node_ok", "sq_ok", "relax_ok", "closed",
              "n_empty", "n_split", "delta_none"):
        R[k] = sum(p[k] for p in parts)
    R["tube"] = [sum(p["tube"][t] for p in parts) for t in range(len(sc.Corridor.DPRIME))]
    for k in ("delta", "ms", "ms1", "seam", "dpp", "vcap", "kpeak"):
        R[k] = [d for p in parts for d in p[k]]
    for k in ("sq_by_layout", "closed_by_layout", "reject_closed"):
        c = Counter()
        for p in parts:
            c.update(p[k])
        R[k] = c
    return R


def _q(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def shape_report(R):
    """R2 / R3 / R4: what the published shape costs, per method. Nothing here is a refusal rate --
    it is the price of the paths the refusal rate counts."""
    tag = f"{R['map']} | {R['method']}{' /pin-apex' if R['pin_apex'] else ''}"
    n_pub = len(R["vcap"])
    print(f"\n--- {tag} | shape of the {n_pub} published path(s) ---")
    print(f"  R2 seam |d'| step at s_entry0   median {_q(R['seam'],50):.4f}  p90 {_q(R['seam'],90):.4f}"
          f"  max {max(R['seam']) if R['seam'] else float('nan'):.4f}"
          f"   [gate p90 <= {SEAM_DPRIME_MAX}]")
    print(f"     whole-maneuver max |d'| step  median {_q(R['dpp'],50):.4f}  p90 {_q(R['dpp'],90):.4f}"
          f"  max {max(R['dpp']) if R['dpp'] else float('nan'):.4f}   [reported, not gated]")
    print(f"  R3 curvature-limited speed cap  median {_q(R['vcap'],50):.2f}  p10 {_q(R['vcap'],10):.2f}"
          f"  m/s   (peak|kappa| median {_q(R['kpeak'],50):.3f})   [reported, NEVER gated]")
    print(f"  R4 one planning pass, ladder off  p50 {_q(R['ms1'],50):5.1f}  p95 {_q(R['ms1'],95):5.1f}"
          f"  max {max(R['ms1']) if R['ms1'] else float('nan'):5.1f} ms"
          f"   [rung budget {LADDER_BUDGET_MS:.0f}, cycle {CYCLE_MS:.0f}]")
    print(f"     whole cell, ladder + squeeze + relax as the chain runs it   p95 "
          f"{_q(R['ms'],95):5.1f}  max {max(R['ms']) if R['ms'] else float('nan'):5.1f} ms")
    if R.get("jobs", 1) > 1:
        print(f"     !! WALL CLOCK ON {R['jobs']} CONTENDING SHARDS. These two lines are not a "
              f"budget verdict -- re-run the timing with --jobs 1.")


def report(R):
    n = max(R["n"], 1)
    print(f"\n=== {R['map']} | profile {R['profile']} | method {R['method']}"
          f"{' /pin-apex' if R['pin_apex'] else ''} | {R['secs']/60:.1f} min ===")
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
    shape_report(R)
    return R


def r1_table(allR):
    """R1: the refusal rate, per map, per method. The one number the change is FOR."""
    print("\n=== R1  refusal rate at racing speed (race profile) ===")
    base = {}
    for R in allR:
        if R["profile"] != "race":
            continue
        n = max(R["n"], 1)
        tag = R["method"] + ("/pin-apex" if R["pin_apex"] else "")
        base.setdefault(R["map"], {})[tag] = (100.0 * (n - R["node_ok"]) / n,
                                              100.0 * R["closed"] / n, R["n"])
    for m, rows in base.items():
        print(f"  {m}")
        ref0 = rows["sample"][0] if "sample" in rows else None
        for tag, (ref, closed, nn) in rows.items():
            delta = "" if ref0 is None or tag == "sample" else f"   {ref - ref0:+5.1f} pp vs sample"
            print(f"    {tag:22s} refuses {ref:5.1f} %   still closed after both retries "
                  f"{closed:5.1f} %   (n={nn}){delta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="*", default=["ifac_0807", "ifac"])
    ap.add_argument("--profile", default="race", choices=("race", "uniform"))
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    # The METHOD axis. `both` is the comparison the change has to survive: the same cells, the same
    # gates, the same retries, one branch of the planner different.
    ap.add_argument("--method", default="both",
                    choices=("sample", "corridor_qp", "corridor_qp_pin", "both", "all"))
    a = ap.parse_args()
    runs = {"both": [("sample", False), ("corridor_qp", False)],
            "all": [("sample", False), ("corridor_qp", False), ("corridor_qp", True)],
            "corridor_qp_pin": [("corridor_qp", True)]}.get(a.method, [(a.method, False)])
    allR = []
    for m in a.map:
        for meth, pin in runs:
            R = sweep(m, a.jobs, a.profile, a.stride, a.limit, method=meth, pin_apex=pin)
            report(R)
            allR.append(R)
    r1_table(allR)
    if a.out:
        Path(a.out).write_text(json.dumps(
            [{k: (dict(v) if isinstance(v, Counter) else v) for k, v in R.items()}
             for R in allR], indent=1, default=float))
        print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
