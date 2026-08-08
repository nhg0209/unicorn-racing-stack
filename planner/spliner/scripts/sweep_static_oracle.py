#!/usr/bin/env python3
"""Was there a path, when the planner said there was none?

The field failure this exists to weigh is one line:

    NO feasible candidate (6 sampled) -> TRAILING | reject bounds=0 obs_box=1 grid=3 body=2

Six candidates, all rejected, car stops. That line does not say whether the section was
impassable or whether six was the wrong number of places to look, and everything downstream of
it -- whether the planner needs new sampling or a new formulation -- turns on which it was.

So: run the SAME planner twice per situation. Once as it ships. Once with nothing changed except
how densely it samples the two axes it already searches:

    terminal offset   the node offers ~6 places to end up beside the box; the oracle offers the
                      whole measured corridor at ~1 cm, with the node's own six still in the set
    ramp length       the node's ladder is 5 entries x 3 exits; the oracle fills the gaps between
                      those rungs

EVERY other question is asked by the node's own code, because it IS the node's code: the corridor
comes from _grid_corridor, the keep-out from the same obs_ok arithmetic, the drivability from
_path_off_track against the once-eroded map, the body floor from _path_body_unsafe against the
7-cell one, the curvature from the same kappa_add_max/kappa_abs_max pair. An oracle that answered
a DIFFERENT question -- a smarter search, a relaxed margin, a lattice -- would produce a number
that cannot be compared to the node's, and the comparison is the whole product here.

The oracle is therefore a strict statement about sampling density and nothing else. If it finds a
path the node missed, the node's search grid missed it. If it does not, the section was closed to
this formulation, and no amount of re-sampling was going to open it.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_oracle.py --map ifac

Nothing here modifies static_avoidance_node or its parameters: the two dense axes are set on the
harness's throwaway node instance, exactly where the harness already sets every other value.
"""
import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_static_feasibility as sw          # noqa: E402  (the harness IS the frame)

# --- the two dense axes -------------------------------------------------------------------------
# 101 offsets over a corridor that is at most ~3 m wide is <= 3 cm resolution -- finer than the
# 5 cm occupancy cell the drivability test reads, and an order of magnitude finer than the ~6
# places the node offers. Going finer would resolve the grid, not the planner.
DENSE_D = 101
# The node's shipped rungs are entry [3.15, 2.5, 2.0, 1.5, 1.0] and exit [4.5, 2.5, 1.5]; every one
# of them is in these lists, with the gaps between them filled and the full 4.5 m entry (which the
# ladder never offers, because rung 0 of the main pass is supposed to have covered it) added.
DENSE_ENTRY = [4.5, 4.0, 3.5, 3.15, 3.0, 2.5, 2.0, 1.5, 1.0]
DENSE_EXIT = [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]

# --- the matrix ---------------------------------------------------------------------------------
GAPS = [12.0, 4.0]                  # forward distance car -> first box
OBS_LAYOUTS = {                     # name -> [ds from the station, ...]; the d offset is applied
    "1box": [0.0],                  # to every box in the layout
    "2box_2m": [0.0, 2.0],
    "2box_4m": [0.0, 4.0],
    "2box_8m": [0.0, 8.0],
    "3box": [0.0, 2.0, 4.0],
}
OBS_D = [-0.3, -0.1, 0.0, 0.1, 0.3]
CUR_D = [-0.2, 0.0, 0.2]

# --- what a RACE actually presents -------------------------------------------------------------
# The uniform matrix above sweeps every station, every lateral placement and every tracking error
# at equal weight. A race does not: the organisers put three boxes on the first half lap, on
# ground a car could otherwise drive, and the car meets them while tracking the raceline. The
# spacings here are what three boxes over a ~18 m half lap look like (ifac is 36.6 m round), and
# the clustered 2 m layouts are kept out of this profile deliberately -- they are a stress case,
# not a race case.
RACE_LAYOUTS = {
    "1box": [0.0],
    "2box_6m": [0.0, 6.0],
    "3box_6m": [0.0, 6.0, 12.0],
}
# |cur_d| <= 0.15: the car is following the line, not recovering from something. This is also the
# node's own clear_max_cur_d, i.e. the offset above which it stops treating the car as "on the
# line" at all.
RACE_CUR_D = [-0.1, 0.0, 0.1]

_RE_NOFEAS = re.compile(
    r"NO feasible candidate \((\d+) sampled\).*?reject bounds=(\d+) obs_box=(\d+) grid=(\d+) "
    r"body=(\d+) curv=(\d+)")
_RE_LADDER = re.compile(r"ramp ladder: no candidate at the adaptive ramp geometry; passing with "
                        r"entry=([\d.]+) m exit=([\d.]+)")


class _Cap:
    """The node's logger, kept. The all-rejected diagnostic is the node's own account of which
    gate killed which candidate, and it is already a formatted string -- parsing it is how this
    script learns the reject histogram without adding a single line to the planner."""

    def __init__(self):
        self.lines = []

    def _rec(self, msg, *a, **k):
        self.lines.append(str(msg))

    info = warn = warning = error = debug = _rec

    def nofeas(self):
        """(N, bounds, obs_box, grid, body, curv) of the LAST all-rejected report, or None."""
        for s in reversed(self.lines):
            m = _RE_NOFEAS.search(s)
            if m:
                return tuple(int(x) for x in m.groups())
        return None

    def ladder(self):
        for s in reversed(self.lines):
            m = _RE_LADDER.search(s)
            if m:
                return (float(m.group(1)), float(m.group(2)))
        return None


def _d_ends(cap_args, P, extra):
    """Rebuild the terminal-offset grid do_spline sampled, from the arguments it handed
    _gate_samples. Six lines of do_spline (the `elif self.sample_gaps` branch) restated -- and
    checked against the node's own "(N sampled)" count on every cell that reports one, so a
    restatement that ever stops matching the original is caught rather than believed."""
    obs_ahead, anchor, (d_lo, d_hi), m_d, gate = cap_args
    n_side = max(2, int(P["n_d_samples"]) // 2)
    d_list = [0.0]
    obox_lo = min(anchor.d_right, anchor.d_left) - m_d
    obox_hi = max(anchor.d_right, anchor.d_left) + m_d
    lo_left = max(obox_hi, d_lo)
    if lo_left <= d_hi + 1e-6:
        d_list += list(np.linspace(lo_left, d_hi, n_side))
    hi_right = min(obox_lo, d_hi)
    if hi_right >= d_lo - 1e-6:
        d_list += list(np.linspace(d_lo, hi_right, n_side))
    d_list += list(gate) + list(extra)
    d = np.unique(np.round(np.asarray(d_list, dtype=float), 4))
    d[int(np.argmin(np.abs(d)))] = 0.0
    return d


def run(H, i, gap, cur_d, boxes, dense_d=False, dense_ramp=False):
    """One planning attempt. Returns a dict; `ok` is whether a path came out.

    dense_d / dense_ramp turn on the two oracle axes independently, which is what separates
    "the node looked in too few places" from "the node had the wrong ramp lengths"."""
    cap = _Cap()
    n = H.make_node(i, gap, cur_d, ladder=True, boxes=boxes)
    n.get_logger = lambda: cap
    if dense_ramp:
        n.ramp_search_entry_m = list(DENSE_ENTRY)
        n.ramp_search_exit_m = list(DENSE_EXIT)
    grabs = {"gate": [], "sel": []}
    orig_gate = n._gate_samples

    def gate_hook(obs_ahead, anchor, corridor, m_d, m_s=None):
        out = list(orig_gate(obs_ahead, anchor, corridor, m_d, m_s))
        extra = []
        lo, hi = corridor
        if dense_d and hi > lo:
            extra = list(np.linspace(lo, hi, DENSE_D))
        grabs["gate"].append(((obs_ahead, anchor, corridor, m_d, tuple(out)), tuple(extra)))
        return out + extra

    n._gate_samples = gate_hook
    n._candidate_markers = lambda xy, status, sel: grabs["sel"].append(int(sel)) or H.san.MarkerArray()

    t0 = time.perf_counter()
    try:
        res = n.do_spline(H.gbw)
    except Exception as e:                        # noqa: BLE001 -- reported, not hidden
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "ms": 0.0, "log": cap}
    ms = (time.perf_counter() - t0) * 1e3
    ok = res is not None and res[0] is not None and len(res[0].wpnts) > 0
    out = {"ok": ok, "err": None, "ms": ms, "log": cap, "reject": cap.nofeas(),
           "squeeze": bool(ok and res[0].ot_line == "squeeze"), "ramp": cap.ladder(),
           "d_grid": None, "d_sel": None, "n_sampled": None}
    if grabs["gate"]:
        first, extra0 = grabs["gate"][0]
        out["d_grid"] = _d_ends(first, H.P, extra0)
        out["n_sampled"] = len(out["d_grid"])
        if ok and grabs["sel"] and grabs["sel"][-1] >= 0:
            last, extraN = grabs["gate"][-1]
            d_last = _d_ends(last, H.P, extraN)
            k = grabs["sel"][-1]
            if k < len(d_last):
                out["d_sel"] = float(d_last[k])
    return out


def classify(node_r, d_r, ramp_r, dense_r):
    """Which axis, alone, would have found the path the node missed."""
    d_ok = d_r is not None and d_r["ok"]
    r_ok = ramp_r is not None and ramp_r["ok"]
    if d_ok and r_ok:
        base = "either axis alone"
        win = d_r
    elif d_ok:
        base = "offset sampling"
        win = d_r
    elif r_ok:
        base = "ramp ladder"
        win = ramp_r
    elif dense_r is not None and dense_r["ok"]:
        return "both axes together", None
    else:
        return None, None
    # Where in the node's own grid did the winning offset sit? Between two of its samples means
    # resolution; outside their span means the node never offered that part of the corridor at all.
    g, ds = node_r.get("d_grid"), win.get("d_sel")
    if g is None or ds is None or not len(g):
        return base, None
    if ds < g.min() - 1e-6 or ds > g.max() + 1e-6:
        return base, "outside the node's sampled span"
    if np.min(np.abs(g - ds)) <= 1e-3:
        return base, "on a node sample (ramp geometry decided it)"
    return base, "between two node samples"


def cells(H, stations, layouts=None, cur_ds=None, gaps=None):
    for i in stations:
        for gap in (gaps or GAPS):
            for lname, offs in (layouts or OBS_LAYOUTS).items():
                for od in OBS_D:
                    for cd in (cur_ds or CUR_D):
                        yield (i, gap, lname, tuple((o, od) for o in offs), od, cd)


def sweep(mapname, stations=None, verbose=False):
    H = sw.Harness(mapname)
    st = H.stations if stations is None else stations
    tot = n_node = n_orc = 0
    missed = []
    why = Counter()
    where = Counter()
    rej_hist = Counter()
    rej_sum = Counter()
    n_grid_mismatch = 0
    n_audit = n_audit_lost = 0
    t0 = time.time()
    for (i, gap, lname, boxes, od, cd) in cells(H, st):
        tot += 1
        r = run(H, i, gap, cd, boxes)
        if r["reject"] and r["n_sampled"] is not None and r["reject"][0] != r["n_sampled"]:
            n_grid_mismatch += 1
        if r["ok"]:
            n_node += 1
            n_orc += 1
            # SUPERSET AUDIT, on a subsample. The dense run adds offsets to the node's own six and
            # rungs to the node's own ladder, so it should never LOSE a cell -- except that
            # max|d_ends| also sets the width of the ramp scan's corridor read, and a wider read can
            # move the adaptive ramp fit. That is the one path by which "denser" is not "superset",
            # and a claim that the oracle answers the node's question has to measure it rather than
            # assume it. Cheap here because a cell that plans plans on the first pass.
            if tot % 20 == 0:
                n_audit += 1
                if not run(H, i, gap, cd, boxes, dense_d=True, dense_ramp=True)["ok"]:
                    n_audit_lost += 1
            continue
        # the node published nothing. Ask the two axes separately, then together.
        d_r = run(H, i, gap, cd, boxes, dense_d=True)
        ramp_r = run(H, i, gap, cd, boxes, dense_ramp=True)
        dense_r = None
        if not (d_r["ok"] or ramp_r["ok"]):
            dense_r = run(H, i, gap, cd, boxes, dense_d=True, dense_ramp=True)
        if not (d_r["ok"] or ramp_r["ok"] or (dense_r is not None and dense_r["ok"])):
            if r["reject"]:
                rej_hist["+".join(k for k, v in zip(
                    ("bounds", "obs_box", "grid", "body", "curv"), r["reject"][1:]) if v) or "none"] += 1
                for k, v in zip(("bounds", "obs_box", "grid", "body", "curv"), r["reject"][1:]):
                    rej_sum[k] += v
            continue
        n_orc += 1
        base, spot = classify(r, d_r, ramp_r, dense_r)
        why[base] += 1
        if spot:
            where[spot] += 1
        w = d_r if d_r["ok"] else (ramp_r if ramp_r["ok"] else dense_r)
        missed.append({
            "station": i, "gap": gap, "layout": lname, "obs_d": od, "cur_d": cd,
            "reject": r["reject"], "n_sampled": r["n_sampled"], "why": base, "where": spot,
            "d_sel": w.get("d_sel"), "ramp": w.get("ramp"),
            "d_grid": None if r["d_grid"] is None else [round(float(x), 3) for x in r["d_grid"]],
        })
        if verbose:
            print(f"    MISS st={i:4d} gap={gap:4.1f} {lname:8s} obs_d={od:+.1f} cur_d={cd:+.1f} "
                  f"| node reject {r['reject']} | {base} / {spot} | d_sel={w.get('d_sel')}")
        if r["reject"]:
            for k, v in zip(("bounds", "obs_box", "grid", "body", "curv"), r["reject"][1:]):
                rej_sum["missed_" + k] += v
    return {"map": mapname, "total": tot, "node_ok": n_node, "oracle_ok": n_orc,
            "missed": missed, "why": why, "where": where, "rej_closed": rej_hist,
            "rej_sum": rej_sum, "grid_mismatch": n_grid_mismatch, "secs": time.time() - t0,
            "stations": len(st), "audit": n_audit, "audit_lost": n_audit_lost}


def report(R):
    tot, nn, no = R["total"], R["node_ok"], R["oracle_ok"]
    miss = no - nn
    print(f"\n=== {R['map']} | {R['stations']} stations | {tot} cells | {R['secs']/60:.1f} min ===")
    print(f"  node_ok    {nn:6d}  ({100.0*nn/max(tot,1):5.1f} %)")
    print(f"  oracle_ok  {no:6d}  ({100.0*no/max(tot,1):5.1f} %)")
    print(f"  MISSED     {miss:6d}  ({100.0*miss/max(tot,1):5.1f} % of all cells, "
          f"{100.0*miss/max(tot-nn,1):5.1f} % of the cells the node refused)")
    if R["grid_mismatch"]:
        print(f"  !! the reconstructed offset grid disagreed with the node's own (N sampled) "
              f"count on {R['grid_mismatch']} cells -- the classification below is not trustworthy")
    print(f"  superset audit: {R['audit_lost']} of {R['audit']} sampled node_ok cells were LOST by "
          f"the dense run (must be 0 for 'denser' to mean 'superset')")
    if miss:
        print("  why the node missed it (which single axis would have sufficed):")
        for k, v in R["why"].most_common():
            print(f"    {k:24s} {v:6d}  ({100.0*v/miss:5.1f} %)")
        print("  where the found offset sat in the node's own sample grid:")
        for k, v in R["where"].most_common():
            print(f"    {k:40s} {v:6d}")
    print("  cells NEITHER found (genuinely closed to this formulation): "
          f"{tot - no}. What killed every candidate there:")
    for k, v in R["rej_closed"].most_common(8):
        print(f"    {k:28s} {v:6d} cells")
    return miss


def _shard(arg):
    mapname, stations, verbose = arg
    return sweep(mapname, stations, verbose)


def sweep_parallel(mapname, stations, jobs, verbose):
    """Stations are independent cells, so shard them. Each worker builds its own Harness (~1 s);
    everything else is per-cell work with no shared state."""
    if jobs <= 1:
        return sweep(mapname, stations, verbose)
    import multiprocessing as mp
    shards = [stations[k::jobs] for k in range(jobs)]
    shards = [s for s in shards if s]
    with mp.get_context("fork").Pool(len(shards)) as pool:
        parts = pool.map(_shard, [(mapname, s, verbose) for s in shards])
    R = {"map": mapname, "total": 0, "node_ok": 0, "oracle_ok": 0, "missed": [],
         "why": Counter(), "where": Counter(), "rej_closed": Counter(), "rej_sum": Counter(),
         "grid_mismatch": 0, "secs": 0.0, "stations": len(stations), "audit": 0, "audit_lost": 0}
    for p in parts:
        for k in ("total", "node_ok", "oracle_ok", "grid_mismatch", "audit", "audit_lost"):
            R[k] += p[k]
        R["missed"] += p["missed"]
        for k in ("why", "where", "rej_closed", "rej_sum"):
            R[k].update(p[k])
        R["secs"] = max(R["secs"], p["secs"])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="*", default=["ifac_0807", "ifac"])
    ap.add_argument("--stride", type=int, default=None, help="override STATION_STRIDE")
    ap.add_argument("--limit", type=int, default=None, help="first N stations only (smoke test)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None, help="write the missed cells to this json")
    a = ap.parse_args()

    allR = []
    for m in a.map:
        H = sw.Harness(m)
        H_st = list(range(0, len(H.wp) - 1, a.stride or sw.STATION_STRIDE))
        if a.limit:
            H_st = H_st[:a.limit]
        del H
        R = sweep_parallel(m, H_st, a.jobs, a.verbose)
        report(R)
        allR.append(R)
    if a.out:
        Path(a.out).write_text(json.dumps(
            [{k: (dict(v) if isinstance(v, Counter) else v) for k, v in R.items()} for R in allR],
            indent=1, default=float))
        print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
