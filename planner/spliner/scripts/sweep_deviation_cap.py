#!/usr/bin/env python3
"""How far off the raceline does the static planner go, and what does bounding it cost?

The planner has NO upper bound on deviation from the raceline. Its limits are the corridor, the
curvature gates and the grid; `corridor_qp_w_dev` -- the only term in the QP that prefers d = 0 --
ships at 0.0, so where the corridor is wide (ifac_0807's start straight reaches 1.9-2.6 m) nothing
in the formulation keeps the published path near the line. The state machine trips RECOVERY at
|cur_d| >= recovery_entry_d_m = 0.4, so a path pushed to a wide corridor's edge and then followed
is a RECOVERY trip by construction.

This file MEASURES that, per configuration, and changes nothing. Two ways of bounding it:

  SOFT   corridor_qp_w_dev in the QP's own objective, min ||D2 d||^2 + w_dev ||d||^2. Smooth, so a
         wide corridor still permits a far excursion -- it only has to pay bending for it.
  HARD   |d| <= cap as a corridor CLIP, the shape closed_reopt takes (its locality_envelope enters
         the QP as bounds, never as a weight). Applied by clipping `_path_corridor`'s two arrays in
         the harness's own copy of the node module -- they ARE qp_lo/qp_hi and are read at exactly
         one place -- so the planner is untouched. Two deliberate differences from
         locality_envelope, both of which make this the WEAKER of the two: that envelope is TAPERED
         (1 near a box, smoothstepped to 0 at infl_len_m, so avoidance simply cannot reach far from
         the box it is for) and it is further scaled by the remaining curvature budget. A constant
         cap is what the brief asked for and is the cheaper thing to implement; read this column as
         the floor of what a hard bound buys, not the ceiling.

         AND IT NEEDS A GATE AS WELL AS BOUNDS. When the cap and a box keep-out close the corridor,
         `_corridor_profile` returns None and the node falls back to the SAMPLED QUINTIC
         (static_avoidance_node.py:1848-1856), which has no cap -- so a bound placed only inside
         the QP is bypassed in exactly the cells that need it. `ok()` below adds the gate a real
         implementation would need; `capfail` counts how often it fires.

Everything else is the shipped configuration, the same cells and the same two retries as
sweep_static_race (race profile, placeable filter, squeeze then relax), so the refusal column is
comparable to that file's R1.

REPORTED PER ROW
  refuse %        planner publishes nothing at racing speed (cur_vs 3.0)
  closed %        still nothing after the squeeze and the relax retry -- the TRAILING standstill
  max|d|          the published path's peak offset from the raceline: median / p90 / max
  >= 0.4 %        published paths whose peak REACHES recovery_entry_d_m -- the SM's own test is
                  `abs(cur_d) >= recovery_entry_d_m`, so this is the RECOVERY contention with zero
                  tracking error allowed for
  seam            |d'| step at the one splice, median / p90            [gate p90 <= 0.05]
  vcap            sqrt(a_lat_max / peak|kappa|), median / p10          [reported, never gated]
  ms              ONE planning pass, ladder off: p50 / p95             [WALL CLOCK -- see below]

MS IS WALL CLOCK. Under --jobs > 1 the shards contend and the timing describes the machine. Take
the timing row from a --jobs 1 run; the report says which it is on every run.

A CAVEAT THE CAP COLUMN CANNOT SHOW: the QP pins d(s0) = cur_d as an equality, so a car already
outside the cap has no feasible corridor at all and the planner would refuse until it recovered.
The race profile's |cur_d| <= 0.1 never reaches that, which is exactly why this sweep cannot price
it -- it is a property of the design, to be answered before a cap ships, not a number in the table.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_deviation_cap.py --jobs 7
  ... --jobs 1 --stride 12          # the timing column, uncontended
  ... --rows w0,w0.1,cap0.4         # a subset (exact tags)
"""
import argparse
import json
import re
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
RACE_SPEED = 3.0         # the harness default, and exactly ON the squeeze gate
SEAM_DPRIME_MAX = 0.05   # the seam gate sweep_static_race holds corridor_qp to

# THE |cur_d| AXIS, and it is not decoration. sweep_static_race's race profile uses |cur_d| <= 0.1
# ("the car is following the line"), which made the QP's d(s0) = cur_d equality pin look harmless.
# The crash run says otherwise: recovery_spliner reported raceline lost at |d| = 0.82 / 0.47 / 0.43
# / 0.42, the controller's AEB saw max|d| = 0.65, and lane_change committed offsets of 0.58 / 0.68.
# So the branch where a car is ALREADY outside a candidate cap -- no feasible corridor, QP returns
# None, node falls back to the uncapped quintic -- is live at exactly the moment the car crashes.
# 0.5 is past every cap under test; 0.3 is between cap0.3 and recovery_entry_d_m.
CUR_D_AXIS = [0.0, 0.1, 0.3, 0.5]

# the node's own account of the fallback, parsed rather than re-derived (same idiom as _Cap.nofeas)
_RE_FALLBACK = re.compile(r"corridor_qp: (\d+) of (\d+) candidate\(s\) fell back")

# state_machine_params.yaml. Read, not hardcoded, so a change there cannot leave this table
# quoting a threshold the state machine no longer uses.
SM_PARAMS = (Path(__file__).resolve().parents[3] / "stack_master" / "config"
             / "state_machine_params.yaml")


def recovery_entry_d():
    import yaml
    y = yaml.safe_load(SM_PARAMS.read_text())
    for block in y.values():
        p = (block or {}).get("ros__parameters", {})
        if "recovery_entry_d_m" in p:
            return float(p["recovery_entry_d_m"])
    raise SystemExit(f"recovery_entry_d_m not found in {SM_PARAMS}")


# --- the rows -------------------------------------------------------------------------------
# ROW = (tag, method, w_dev or None = whatever the yaml ships, hard cap or None)
def build_rows(w_devs, caps, w_dev_shipped):
    rows = [("sample", "sample", None, None)]
    for w in w_devs:
        rows.append((f"w{w:g}", "corridor_qp", w, None))
    for c in caps:
        rows.append((f"cap{c:g}", "corridor_qp", w_dev_shipped, c))
    return rows


def cap_corridor(san, cap):
    """Clip the QP's corridor to |d| <= cap, in the harness's own copy of the node module.

    `_path_corridor`'s two arrays are read at exactly one place -- they ARE qp_lo/qp_hi -- so
    clipping them puts the cap where closed_reopt puts its envelope: in the bounds, hard, not as a
    weight. The planner source is untouched. NaN survives the clip, so 'corridor unmeasurable'
    still reads as unmeasurable rather than as a cap.

    WHAT THIS ALONE DOES NOT DO, and it is the finding this row exists to carry: when the cap and a
    box keep-out close the corridor, `_corridor_profile` returns None and the node FALLS BACK TO THE
    SAMPLED QUINTIC (static_avoidance_node.py:1848-1856), which has no cap. So a bound placed only
    inside the QP is bypassed in precisely the cells that need it. The cap-gate below is what a real
    implementation would have to add alongside it.
    """
    orig = san.ObstacleSpliner._path_corridor

    def capped(self, *a, **k):
        lo, hi = orig(self, *a, **k)
        return np.maximum(np.asarray(lo, float), -cap), np.minimum(np.asarray(hi, float), cap)

    san.ObstacleSpliner._path_corridor = capped


def _peaks(H, r):
    """(peak |d| over the whole published array, peak |d| over the MANEUVER only), or (None, None).

    The two differ by the PREFIX, and the difference is the point. Everything before s_entry0 is the
    decay of the car's CURRENT offset back to the raceline -- it is where the car already is, not
    something the planner chose, and no corridor bound can move it. So a cap measured on the whole
    array reads |cur_d| whenever the car is outside the cap and refuses every cell for a reason the
    planner is not responsible for; a cap has to be judged on the part the planner shapes.
    """
    d, s0 = r.get("d_pub"), r.get("s_entry0")
    if d is None or len(d) < 5:
        return None, None
    ds = H.wp[1]["s_m"] - H.wp[0]["s_m"]
    j0 = 1 if s0 is None else int(np.clip(round(float(s0) / ds), 1, len(d) - 2))
    return float(np.max(np.abs(d))), float(np.max(np.abs(d[j0:])))


def _shape(H, r, o, d_trip):
    """What the PUBLISHED profile is: its peak offset, its splice, and the speed its curvature
    allows. Same definitions as sweep_static_race._shape_stats, plus the peak offset this file is
    about."""
    d, kap, s0 = r.get("d_pub"), r.get("kappa_pub"), r.get("s_entry0")
    if d is None or kap is None or len(d) < 5:
        return
    ds = H.wp[1]["s_m"] - H.wp[0]["s_m"]
    dmax, dman = _peaks(H, r)
    o["dmax"].append(dmax)
    o["dman"].append(dman)
    # THE SM's OWN COMPARISON, which is >= and not > (state_machine_node.py:1000,
    # `abs(cur_d) >= recovery_entry_d_m`). It matters for a cap set AT the threshold: a path that
    # reaches exactly 0.4 trips RECOVERY the moment the car is on it, before any tracking error.
    if dmax >= d_trip - 1e-9:
        o["n_trip"] += 1
    o["vcap"].append(float(np.sqrt(float(H.P["a_lat_max"]) / max(np.max(np.abs(kap)), 1e-3))))
    o["kpeak"].append(float(np.max(np.abs(kap))))
    j0 = 1 if s0 is None else int(np.clip(round(float(s0) / ds), 1, len(d) - 2))
    dpp = np.abs(np.diff(d, 2)) / ds
    o["seam"].append(float(dpp[j0 - 1]))


def _shard(arg):
    mapname, stations, tag, method, w_dev, cap, d_trip = arg
    H = sw.Harness(mapname)
    H.plan_method = method
    # The w_dev axis goes in through H.P, which _node copies onto the node attribute by attribute --
    # the same path the yaml value takes, so this is an override of one shipped value and nothing
    # else. Every other planner value still comes from the yaml.
    if w_dev is not None:
        H.P["corridor_qp_w_dev"] = float(w_dev)
    if cap is not None:
        cap_corridor(H.san, float(cap))
    C = sc.Corridor(H)

    def bucket():
        return {"n_raw": 0, "n": 0, "unplaceable": 0, "node_ok": 0, "sq_ok": 0, "relax_ok": 0,
                "closed": 0, "n_trip": 0, "n_capfail": 0, "n_fb": 0, "n_fb_pub": 0,
                "reject_closed": Counter(),
                "ms1": [], "dmax": [], "dman": [], "dmax_fb": [], "seam": [], "vcap": [],
                "kpeak": []}

    B = {cd: bucket() for cd in CUR_D_AXIS}

    def fell_back(r):
        """Did any candidate fall back to the uncapped sampled quintic this pass? The node warns
        when it happens, so read its own count instead of re-deriving one."""
        for line in (r.get("log").lines if r.get("log") is not None else ()):
            m = _RE_FALLBACK.search(line)
            if m and int(m.group(1)) > 0:
                return True
        return False

    def ok(r, o):
        """Did the planner publish a path this cell would be allowed to drive?

        For the cap rows this is where the CAP-GATE lives: a published path over the cap is one the
        QP did not shape (the sampled fallback), and a real hard bound would have to reject it. The
        approximation is named: a cap-gate in the node would reject that CANDIDATE and could still
        publish a different one, so counting the whole cell refused is an UPPER bound on the cap's
        refusal cost. n_capfail is exactly how many cells that is, so the size of the approximation
        is in the table rather than in this comment.
        """
        if not r["ok"]:
            return False
        if cap is None:
            return True
        _all, man = _peaks(H, r)
        if man is not None and man > cap + 1e-6:
            o["n_capfail"] += 1
            return False
        return True

    for (i, gap, lname, boxes, od, cd) in so.cells(H, stations, so.RACE_LAYOUTS, CUR_D_AXIS):
        o = B[cd]
        o["n_raw"] += 1
        if not C.placeable(i, boxes):
            o["unplaceable"] += 1
            continue
        o["n"] += 1
        H.cur_vs, H.force_relax = RACE_SPEED, False
        o["ms1"].append(so.run(H, i, gap, cd, boxes, ladder=False)["ms"])
        r0 = so.run(H, i, gap, cd, boxes)
        fb = fell_back(r0)
        o["n_fb"] += 1 if fb else 0
        if ok(r0, o):
            o["node_ok"] += 1
            _shape(H, r0, o, d_trip)
            if fb and o["dmax"]:
                o["n_fb_pub"] += 1
                o["dmax_fb"].append(o["dmax"][-1])
            continue
        H.cur_vs = SQUEEZE_SPEED                       # squeeze reachable: the car has slowed
        r_sq = so.run(H, i, gap, cd, boxes)
        sq = ok(r_sq, o)
        H.cur_vs, H.force_relax = RACE_SPEED, True     # relax: the SM certified a deadlock
        r_rx = so.run(H, i, gap, cd, boxes)
        rx = ok(r_rx, o)
        H.force_relax = False
        if sq:
            o["sq_ok"] += 1
        if rx:
            o["relax_ok"] += 1
        if sq or rx:
            _shape(H, r_sq if sq else r_rx, o, d_trip)
            continue
        o["closed"] += 1
        H.cur_vs = RACE_SPEED
        r = so.run(H, i, gap, cd, boxes)
        if r["reject"]:
            sig = "+".join(nm for nm, v in zip(("bounds", "obs_box", "grid", "body", "curv"),
                                               r["reject"][1:]) if v) or "none"
            o["reject_closed"][sig] += 1
        elif cap is not None:
            o["reject_closed"]["cap_gate"] += 1
    return B


def sweep_row(mapname, row, jobs, stations, d_trip):
    tag, method, w_dev, cap = row
    t0 = time.time()
    if jobs <= 1:
        parts = [_shard((mapname, stations, tag, method, w_dev, cap, d_trip))]
    else:
        import multiprocessing as mp
        shards = [s for s in (stations[k::jobs] for k in range(jobs)) if s]
        with mp.get_context("fork").Pool(len(shards)) as pool:
            parts = pool.map(_shard, [(mapname, s, tag, method, w_dev, cap, d_trip)
                                      for s in shards])
    R = {"tag": tag, "map": mapname, "method": method, "w_dev": w_dev, "cap": cap,
         "secs": time.time() - t0, "jobs": jobs, "by_cd": {}}
    for cd in CUR_D_AXIS:
        b = {}
        for k in ("n_raw", "n", "unplaceable", "node_ok", "sq_ok", "relax_ok", "closed",
                  "n_trip", "n_capfail", "n_fb", "n_fb_pub"):
            b[k] = sum(p[cd][k] for p in parts)
        for k in ("ms1", "dmax", "dman", "dmax_fb", "seam", "vcap", "kpeak"):
            b[k] = [v for p in parts for v in p[cd][k]]
        c = Counter()
        for p in parts:
            c.update(p[cd]["reject_closed"])
        b["reject_closed"] = c
        R["by_cd"][cd] = b
    return R


def _q(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def _mx(a):
    return max(a) if len(a) else float("nan")


def table(allR, d_trip, jobs, shipped):
    m = allR[0]["map"]
    ncell = allR[0]["by_cd"][CUR_D_AXIS[0]]["n"]
    print(f"\n=== deviation from the raceline, per bound, per |cur_d| | {m} | race profile | "
          f"{ncell} cells per (row, cur_d) ===")
    print(f"    max|d|  the PUBLISHED path's peak offset from the raceline (whole array).")
    print(f"    man     the same peak over the MANEUVER only (past s_entry0). The prefix is the")
    print(f"            car's own offset decaying back to the line -- no bound can move it, so the")
    print(f"            cap is judged on `man`. Where man << max|d|, the excursion IS the prefix.")
    print(f"    >={d_trip:g}    published paths that REACH recovery_entry_d_m. The SM trips on "
          f"`abs(cur_d) >= {d_trip:g}`, so a path")
    print(f"            AT the cap trips it with no tracking error at all -- a usable cap sits "
          f"BELOW {d_trip:g}, not on it.")
    print(f"    cur_d   the car's lateral offset when it plans. NOT decoration: the crash run "
          f"showed 0.42-0.82 m,")
    print(f"            and the QP pins d(s0) = cur_d as an equality, so a car outside a cap has "
          f"no corridor left.")
    hdr = (f"  {'row':<8}{'cur_d':>7}{'refuse':>8}{'closed':>8}   "
           f"{'max|d| med':>10}{'p90':>7}{'max':>8}{'  man p90':>9}{'man max':>9}{f'  >={d_trip:g}':>9}   "
           f"{'seam p90':>9}{'vcap med':>9}{'ms p50':>8}{'  fallback':>10}{'  capfail':>9}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for R in allR:
        for cd in CUR_D_AXIS:
            b = R["by_cd"][cd]
            n = max(b["n"], 1)
            npub = max(len(b["dmax"]), 1)
            print(f"  {R['tag']:<8}{cd:7.1f}{100.0*(n-b['node_ok'])/n:7.1f}%"
                  f"{100.0*b['closed']/n:7.1f}%   "
                  f"{_q(b['dmax'],50):10.3f}{_q(b['dmax'],90):7.3f}{_mx(b['dmax']):8.4f}"
                  f"{_q(b['dman'],90):9.3f}{_mx(b['dman']):9.4f}"
                  f"{100.0*b['n_trip']/npub:8.1f}%   "
                  f"{_q(b['seam'],90):9.4f}{_q(b['vcap'],50):9.2f}{_q(b['ms1'],50):8.1f}"
                  f"{100.0*b['n_fb']/n:9.1f}%"
                  f"{(str(b['n_capfail']) if R['cap'] is not None else '-'):>9}")
        print("  " + "." * (len(hdr) - 2))

    # THE SPLIT THE CAP ROWS ARE FOR: a bound that lives only inside the QP is not a bound on the
    # PUBLISHED path, because `dv_c is None` falls through to the uncapped sampled quintic. Count
    # the three outcomes apart instead of averaging them into one max|d|.
    print(f"\n=== cap rows: does the bound survive to the published path? ===")
    print(f"    QP ok        published, no candidate fell back -> the cap shaped the path")
    print(f"    fallback     at least one candidate fell back to the UNCAPPED sampled quintic")
    print(f"    fb published the fallback shape is what got published, and its peak offset")
    print(f"\n  {'row':<8}{'cur_d':>7}{'QP ok':>8}{'fallback':>10}{'fb published':>14}"
          f"{'max|d| of those':>17}{'over cap':>10}")
    print("  " + "-" * 74)
    for R in allR:
        if R["cap"] is None:
            continue
        for cd in CUR_D_AXIS:
            b = R["by_cd"][cd]
            n = max(b["n"], 1)
            qp_ok = b["node_ok"] - b["n_fb_pub"]
            fbmax = f"{_mx(b['dmax_fb']):.4f}" if b["dmax_fb"] else "--"
            print(f"  {R['tag']:<8}{cd:7.1f}{qp_ok:8d}{b['n_fb']:10d}{b['n_fb_pub']:14d}"
                  f"{fbmax:>17}{b['n_capfail']:10d}")
    print(f"\n  ships: {shipped}. capfail = published paths over the cap, counted REFUSED here.")
    print(f"  seam gate p90 <= {SEAM_DPRIME_MAX} (over all cur_d): " +
          ", ".join(f"{R['tag']}="
                    f"{'OK' if max((_q(R['by_cd'][cd]['seam'], 90) for cd in CUR_D_AXIS)) <= SEAM_DPRIME_MAX else 'FAIL'}"
                    for R in allR))
    if jobs > 1:
        print(f"  !! ms columns are WALL CLOCK on {jobs} contending shards -- not a budget verdict.")
    print("\n  WHY NO VALUE IS BEING CHANGED FROM THIS TABLE YET: raising refuse raises TRAILING,")
    print("  which raises static_feasible=False, which leaves the SM more windows in which the ONLY")
    print("  adoptable path is lane_change's -- and that planner is the one that just drove the car")
    print("  into a wall (it engaged on stationary boxes with no opponent present). Bounding the")
    print("  static planner's deviation before that is fixed hands authority to the worse actor.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="ifac_0807")
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--w-dev", default="0.0,0.01,0.05,0.1,0.5")
    ap.add_argument("--cap", default="0.5,0.4,0.3")
    ap.add_argument("--rows", default=None, help="comma list of row tags to run (prefix match)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d_trip = recovery_entry_d()
    H = sw.Harness(a.map)
    shipped_w = float(H.P["corridor_qp_w_dev"])
    shipped_m = str(H.P["static_plan_method"])
    st = list(range(0, len(H.wp) - 1, a.stride or sw.STATION_STRIDE))
    if a.limit:
        st = st[:a.limit]
    del H
    print(f"=== ships: static_plan_method={shipped_m}  corridor_qp_w_dev={shipped_w:g}  "
          f"recovery_entry_d_m={d_trip:g} ===")
    print(f"    {len(st)} stations (stride {a.stride or sw.STATION_STRIDE}), jobs {a.jobs}")

    rows = build_rows([float(x) for x in a.w_dev.split(",") if x],
                      [float(x) for x in a.cap.split(",") if x], shipped_w)
    if a.rows:
        want = [t.strip() for t in a.rows.split(",") if t.strip()]
        rows = [r for r in rows if r[0] in want]        # exact: "w0" must not match "w0.01"
    allR = []
    for row in rows:
        R = sweep_row(a.map, row, a.jobs, st, d_trip)
        allR.append(R)
        tot = sum(R["by_cd"][cd]["n"] for cd in CUR_D_AXIS)
        pub = sum(R["by_cd"][cd]["node_ok"] for cd in CUR_D_AXIS)
        dmx = _mx([v for cd in CUR_D_AXIS for v in R["by_cd"][cd]["dmax"]])
        print(f"  {R['tag']:<8} {tot:6d} cells  publishes {pub:6d}  peak|d| max {dmx:.3f}  "
              f"[{R['secs']/60:.1f} min]")
    table(allR, d_trip, a.jobs, f"static_plan_method={shipped_m} "
          f"corridor_qp_w_dev={shipped_w:g}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            [{k: ({str(cd): {kk: (dict(vv) if isinstance(vv, Counter) else vv)
                         for kk, vv in b.items()} for cd, b in v.items()}
              if k == "by_cd" else v)
          for k, v in R.items()} for R in allR], indent=1, default=float))
        print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
