#!/usr/bin/env python3
"""Is the corridor EMPTY, or did the hump family just fail to express what was there?

sweep_static_oracle answers "would denser sampling have found a path". It cannot answer "is there
a path at all", because it inherits the node's path family: one quintic hump per knot, apex at the
box centre, max_weave slots, a pre-ramp decay. Everything it declares closed is closed TO THAT
SHAPE, and a rewrite is exactly a proposal to change the shape. So the shape has to be taken out
of the question.

What is left when it is, is arithmetic. At each station the car centre may lie in

    [lo(s), hi(s)]   minus   every obstacle's keep-out band

and if that set is EMPTY at even one station the path must cross, no planner of any shape gets
through -- the track does not contain the clearance. If it is non-empty at every station, then
lateral room existed the whole way and the hump family is what did not use it. The second count is
the CEILING on what any rewrite could buy.

Deliberately generous everywhere the choice arises, because the number it produces is an upper
bound and an upper bound must not be understated:

  * NO SIDE CHOICE. select_sides collapses the corridor to one connected interval so a QP can be
    posed; that is a solver's need, not the track's. Here every band is subtracted from the
    corridor and ANY surviving lane counts -- including the lane between two boxes, which no
    single side assignment can name.
  * NO PATH CONTINUITY. Stations are asked independently. A cell where the free lane jumps from
    the left of one box to the right of the next is counted OPEN even though no continuous line
    may join them. That inflates the ceiling, which is the safe direction for it.
  * DISC, NOT RECTANGLE. obs_ok enforces a rectangle: full lateral keep-out over the whole
    s-inflated interval. The analytic band tapers as sqrt(R^2 - dt^2), so in the s-tails it asks
    for LESS than the node does. Smaller keep-out -> more cells open -> a higher ceiling.
  * ds = 0. obstacle_bounds can widen the band by half a station spacing so a coarse grid cannot
    straddle the box. The published spacing here is 0.1 m and the corridor is read at every one of
    them, so that conservatism is not owed and not taken.

The keep-out itself is closed_reopt.obstacle_bounds, unmodified -- analytic, grid-independent,
with the closed-loop guard that stops a box twelve metres away from constraining a station just
because its tangential projection is small. What this file adds is the frame conversion and the
side-free interval algebra.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_corridor.py --check
"""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "planner" / "gb_optimizer"))
import sweep_static_feasibility as sw          # noqa: E402
import sweep_static_oracle as so               # noqa: E402
from gb_optimizer.closed_reopt import Obstacle, ReoptParams, obstacle_bounds   # noqa: E402


def frame(H, s_win):
    """(pts, nv, tan) at the given absolute stations, in the converter's own d convention.

    get_cartesian adds d along (cos(psi + pi/2), sin(psi + pi/2)), i.e. +d is to the LEFT of
    travel. obstacle_bounds measures du along `nv`, so nv must BE that vector or every sign in
    the keep-out is mirrored -- which is the one way this comparison could quietly stop being
    about the same track.
    """
    resp = H.converter.get_cartesian(s_win, np.zeros_like(s_win))
    pts = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)
    der = np.asarray(H.converter.get_derivative(s_win), dtype=float)
    if der.shape[0] != 2:
        der = der.T
    tan = (der / np.maximum(np.hypot(der[0], der[1]), 1e-12)).T      # (n, 2)
    nv = np.stack([-tan[:, 1], tan[:, 0]], axis=1)                   # rot +90 == +d
    return pts, nv, tan


def corridor(n, H, s_win, idxs, wall_margin, sample_margin):
    """[lo, hi] car-centre bounds, in the node's own authority order: the eroded occupancy grid
    where it can be measured, the waypoint corridor where it cannot."""
    lo, hi = n._grid_corridor_batch(s_win, wall_margin=wall_margin)
    if lo is None:
        lo = np.full(len(s_win), np.nan)
        hi = np.full(len(s_win), np.nan)
    miss = ~np.isfinite(lo)
    if miss.any():
        jj = idxs[miss]
        lo[miss] = np.array([-(H.gbw[j].d_right - sample_margin) for j in jj])
        hi[miss] = np.array([H.gbw[j].d_left - sample_margin for j in jj])
    return lo, hi


def bands(pts, nv, tan, obstacles, lo, hi, params):
    """Each obstacle's keep-out as (band_lo, band_hi) per station, clipped to the corridor.

    obstacle_bounds is called once per side per obstacle. With sides=[+1] it returns
    max(lo, du + h) -- the band's upper edge; with sides=[-1], min(hi, du - h) -- the lower one.
    Untouched stations come back as (hi, lo), an inverted interval that cuts nothing, which is
    how the function's own closed-loop guard survives into the algebra here.
    """
    out = []
    for o in obstacles:
        lo_p, _ = obstacle_bounds(pts, nv, tan, [o], [+1], lo, hi, params, ds=0.0)
        _, hi_m = obstacle_bounds(pts, nv, tan, [o], [-1], lo, hi, params, ds=0.0)
        out.append((hi_m, lo_p))
    return out


def free_intervals(lo, hi, bnds):
    """Per station, the lanes a car CENTRE may occupy: [lo, hi] minus every band, as a sorted
    list of open intervals. No side is chosen, so the lane BETWEEN two boxes appears like any
    other."""
    out = []
    for i in range(len(lo)):
        free = [(lo[i], hi[i])]
        for b_lo, b_hi in bnds:
            blo, bhi = b_lo[i], b_hi[i]
            if not (bhi > blo):
                continue                              # inverted -> this band cuts nothing here
            nxt = []
            for a, b in free:
                if bhi <= a or blo >= b:
                    nxt.append((a, b))
                    continue
                if a < blo:
                    nxt.append((a, blo))
                if bhi < b:
                    nxt.append((bhi, b))
            free = nxt
            if not free:
                break
        out.append([(a, b) for a, b in free if b > a])
    return out


def connected(free, delta):
    """Does one lane thread the WHOLE window?

    Per-station non-emptiness is necessary for a path and not sufficient: d(s) is continuous, so
    a car cannot use the left lane at one station and the right lane at the next. A box on the
    raceline in a corridor too narrow for either side to be taken all the way leaves both lanes
    non-empty at every station and admits no path at all -- counting that cell as "room existed"
    would put a phantom in the rewrite's ceiling.

    So: sweep forward, carrying the set of offsets still reachable, dilated by `delta` per station
    to allow for the lateral movement a real path makes between two stations 0.1 m apart.
    delta = 0 is the strict floor (a path that never moves sideways); the caller also runs a
    permissive delta to show the answer does not live on that choice.
    """
    reach = list(free[0])
    for k in range(1, len(free)):
        if not reach:
            return False
        grown = [(a - delta, b + delta) for a, b in reach]
        nxt = []
        for ga, gb in grown:
            for fa, fb in free[k]:
                a, b = max(ga, fa), min(gb, fb)
                if b > a:
                    nxt.append((a, b))
        nxt.sort()
        merged = []
        for a, b in nxt:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        reach = merged
    return bool(reach)


class Corridor:
    """The arithmetic test, set up once per map."""

    def __init__(self, H):
        self.H = H
        P = H.P
        self.half_car = 0.5 * float(P["width_car"])
        self.obs_margin_d = self.half_car + float(P["safety_margin_d"])
        self.obs_margin_s = self.half_car + float(P["safety_margin"])
        self.sample_margin = self.half_car + float(P["wall_margin"])
        self.wall_margin = float(P["wall_margin"])
        # R = r + obs_margin + w_veh/2 = 0.15 + 0.15 + 0.15 = 0.45, which is EXACTLY the node's
        # own lateral keep-out half-width at the box's own station: obs_ok tests
        # d in [d_center - (0.15 + obs_margin_d), d_center + (0.15 + obs_margin_d)] and
        # obs_margin_d = width_car/2 + safety_margin_d = 0.30. Same number, reached from the two
        # different directions the two files describe it from.
        self.params = ReoptParams(obs_margin=float(P["safety_margin_d"]),
                                  w_veh=float(P["width_car"]), disc_allow_m=0.0)
        self.r_box = sw.Harness.OBS_HALF

    # THE ANSWER MOVES WITH THIS, SO IT IS REPORTED AS A PROFILE AND NOT AS A NUMBER.
    # Connectivity needs to know how far a path may move sideways between two stations 0.1 m
    # apart, i.e. a bound on |d'|. Zero is the strict floor and describes nothing real; the
    # node's own entry slope is tan(clip(e_psi, +-0.5)) = 0.546, and a quintic hump of 0.5 m over
    # a 2.5-4.5 m ramp peaks at |d'| = 1.875 * A / L = 0.21-0.38. So the band that matters is
    # 0.2-0.5, and the ceiling is quoted across the whole range rather than at a chosen point.
    # This is a continuity bound, not a path family: EVERY drivable line obeys one.
    DPRIME = (0.0, 0.1, 0.2, 0.3, 0.5)

    def test(self, i, gap, cur_d, boxes):
        """(empty_any, tube[len(DPRIME)], n_empty, n_stations) for one matrix cell."""
        H = self.H
        n = H.make_node(i, gap, cur_d, ladder=True, boxes=boxes)
        wpnt_dist = H.gbw[1].s_m - H.gbw[0].s_m
        # The window the path has to cross: the car's own anchored station (the node's anchor, not
        # a fresh nearest-point search) through the far edge of the last box's s-inflated interval.
        car_idx = n._anchor_car_idx(int(n.cur_s / wpnt_dist) % n.gb_max_idx, H.gbw, wpnt_dist)
        grid_start_s = H.gbw[car_idx].s_m
        reach = max(((o.s_center - n.cur_s) % H.L) for o in n.obstacles)
        span = reach + self.r_box + self.obs_margin_s
        nst = max(int(math.ceil(span / wpnt_dist)) + 1, 5)
        idxs = (car_idx + np.arange(nst)) % n.gb_max_idx
        s_win = (grid_start_s + np.arange(nst) * wpnt_dist) % H.L
        lo, hi = corridor(n, H, s_win, idxs, self.wall_margin, self.sample_margin)
        pts, nv, tan = frame(H, s_win)
        obs = [Obstacle(x=float(o.x_m), y=float(o.y_m), r=self.r_box) for o in n.obstacles]
        bnds = bands(pts, nv, tan, obs, lo, hi, self.params)
        free = free_intervals(lo, hi, bnds)
        k = sum(1 for f in free if not f)
        if k:
            return True, [False] * len(self.DPRIME), k, nst, 0.0
        # HOW WIDE IS THE TUBE AT ITS TIGHTEST? A lane is a lane at any positive width, and the
        # count of (iii) cells says nothing about whether the room is a lane or a slot. The
        # bottleneck is the widest lane at the station where the widest lane is narrowest.
        widest = min(max(b - a for a, b in f) for f in free)
        return (False, [connected(free, dp * wpnt_dist) for dp in self.DPRIME], 0, nst,
                float(widest))

    def placeable(self, i, boxes):
        """Could a marshal have PUT these boxes here?

        A box only appears in a race where there is drivable ground to put it on. The uniform
        matrix does not ask that: it walks a box across every lateral offset at every station,
        including offsets whose footprint is already inside a wall. Those cells are not the
        planner failing, they are a placement that never happens, and counting them inflates the
        refusal rate against a race.

        Asked of the box's own footprint (half-extent OBS_HALF each side of its centre) against
        the free extent the grid measures with NO wall reserve taken off -- the reserve is a car's
        claim on the corridor, not a box's.
        """
        H = self.H
        n = H.make_node(i, 12.0, 0.0, ladder=True, boxes=boxes)
        for o in n.obstacles:
            g = n._grid_corridor(float(o.s_center), wall_margin=0.0)
            if g is None:
                return False
            if not (g[0] <= o.d_center - self.r_box and o.d_center + self.r_box <= g[1]):
                return False
        return True

    def min_violation(self, i, gap, cur_d, boxes, dprime=0.2, hi_delta=0.15, tol=0.005):
        """How much margin would have to be GIVEN UP for a tube to exist? [m], or None if none.

        Not a path and not a plan -- a distance. delta is subtracted from BOTH negotiable
        reserves at once, which is exactly the pair the squeeze pass trades:
            safety_margin_d 0.15 -> the obstacle keep-out shrinks by delta
            wall_margin     0.15 -> the corridor widens by delta on each side
        half_car (0.15) is NOT in the search and never can be: that is the car, not a margin.
        So delta is bounded at 0.15, where both reserves are zero and the car is being driven at
        its own width against a box and a wall.

        The reference points that matter are the squeeze's own floors: it interpolates to
        safety 0.05 / wall 0.08, i.e. it can already spend delta = 0.10 on the keep-out and 0.07
        on the wall. A cell that opens below that is one the SHIPPED planner could already take
        and is not taking; a cell that needs more is asking for margin nobody has authorised.
        """
        H = self.H
        n = H.make_node(i, gap, cur_d, ladder=True, boxes=boxes)
        wpnt_dist = H.gbw[1].s_m - H.gbw[0].s_m
        car_idx = n._anchor_car_idx(int(n.cur_s / wpnt_dist) % n.gb_max_idx, H.gbw, wpnt_dist)
        reach = max(((o.s_center - n.cur_s) % H.L) for o in n.obstacles)
        nst = max(int(math.ceil((reach + self.r_box + self.obs_margin_s) / wpnt_dist)) + 1, 5)
        idxs = (car_idx + np.arange(nst)) % n.gb_max_idx
        s_win = (H.gbw[car_idx].s_m + np.arange(nst) * wpnt_dist) % H.L
        pts, nv, tan = frame(H, s_win)
        obs = [Obstacle(x=float(o.x_m), y=float(o.y_m), r=self.r_box) for o in n.obstacles]

        def open_at(delta):
            lo, hi = corridor(n, H, s_win, idxs, max(self.wall_margin - delta, 0.0),
                              max(self.sample_margin - delta, 0.0))
            p = ReoptParams(obs_margin=max(float(H.P["safety_margin_d"]) - delta, 0.0),
                            w_veh=float(H.P["width_car"]), disc_allow_m=0.0)
            free = free_intervals(lo, hi, bands(pts, nv, tan, obs, lo, hi, p))
            if any(not f for f in free):
                return False
            return connected(free, dprime * wpnt_dist)

        if open_at(0.0):
            return 0.0
        if not open_at(hi_delta):
            return None                    # not reachable even with every negotiable margin gone
        a, b = 0.0, hi_delta
        while b - a > tol:
            m = 0.5 * (a + b)
            if open_at(m):
                b = m
            else:
                a = m
        return b

    def probe(self, i, gap, cur_d, boxes):
        """The same setup, returning the intermediate quantities for the convention check."""
        H = self.H
        n = H.make_node(i, gap, cur_d, ladder=True, boxes=boxes)
        wpnt_dist = H.gbw[1].s_m - H.gbw[0].s_m
        car_idx = n._anchor_car_idx(int(n.cur_s / wpnt_dist) % n.gb_max_idx, H.gbw, wpnt_dist)
        reach = max(((o.s_center - n.cur_s) % H.L) for o in n.obstacles)
        nst = max(int(math.ceil((reach + self.r_box + self.obs_margin_s) / wpnt_dist)) + 1, 5)
        idxs = (car_idx + np.arange(nst)) % n.gb_max_idx
        s_win = (H.gbw[car_idx].s_m + np.arange(nst) * wpnt_dist) % H.L
        lo, hi = corridor(n, H, s_win, idxs, self.wall_margin, self.sample_margin)
        pts, nv, tan = frame(H, s_win)
        obs = [Obstacle(x=float(o.x_m), y=float(o.y_m), r=self.r_box) for o in n.obstacles]
        return n, s_win, lo, hi, pts, nv, tan, obs


def check_convention(mapname, n_show=8):
    """Before any count is believed: does this file read the same corridor and the same keep-out
    the NODE reads, at the same stations?

    Two independent comparisons, because the conversion has two halves that can fail separately:
      CORRIDOR  the batched grid read here against _grid_corridor called one station at a time,
                which is the call do_spline makes for its sampling limits.
      KEEP-OUT  the analytic band at the obstacle's own station against obs_ok's rectangle,
                d_center +- (box half-width + obs_margin_d). They must agree at dt = 0 and the
                band must be the SMALLER of the two everywhere else.
    """
    H = sw.Harness(mapname)
    C = Corridor(H)
    print(f"\n=== convention check | {mapname} ===")
    print("  CORRIDOR: this file (batched) vs the node's own _grid_corridor, per station")
    print("      s      this lo/hi          node lo/hi         d_lo err  d_hi err")
    worst = 0.0
    rows = 0
    for i in H.stations[::max(1, len(H.stations) // 4)][:4]:
        n, s_win, lo, hi, *_ = C.probe(i, 12.0, 0.0, ((0.0, 0.0),))
        for k in list(range(0, len(s_win), max(1, len(s_win) // 2)))[:2]:
            g = n._grid_corridor(float(s_win[k]), wall_margin=C.wall_margin)
            if g is None:
                continue
            e_lo, e_hi = abs(g[0] - lo[k]), abs(g[1] - hi[k])
            worst = max(worst, e_lo, e_hi)
            rows += 1
            if rows <= n_show:
                print(f"   {s_win[k]:7.2f}  [{lo[k]:+.3f},{hi[k]:+.3f}]   "
                      f"[{g[0]:+.3f},{g[1]:+.3f}]    {e_lo:.2e}  {e_hi:.2e}")
    print(f"  -> worst corridor disagreement over {rows} stations: {worst:.2e} m")

    print("  KEEP-OUT: analytic band vs obs_ok's rectangle, at the obstacle's own station")
    print("      obs_d   band [lo,hi]        obs_ok [lo,hi]      du       err")
    worst_b = 0.0
    for od in (-0.3, -0.1, 0.0, 0.1, 0.3):
        i = H.stations[len(H.stations) // 3]
        n, s_win, lo, hi, pts, nv, tan, obs = C.probe(i, 12.0, 0.0, ((0.0, od),))
        o = n.obstacles[0]
        j = int(np.argmin(np.abs(((s_win - o.s_center + H.L / 2) % H.L) - H.L / 2)))
        b_lo, b_hi = bands(pts, nv, tan, obs, lo, hi, C.params)[0]
        box_lo = min(o.d_right, o.d_left) - C.obs_margin_d
        box_hi = max(o.d_right, o.d_left) + C.obs_margin_d
        du = float((np.array([o.x_m, o.y_m]) - pts[j]) @ nv[j])
        err = max(abs(b_lo[j] - max(lo[j], box_lo)) if box_lo > lo[j] else abs(b_lo[j] - box_lo),
                  abs(b_hi[j] - min(hi[j], box_hi)) if box_hi < hi[j] else abs(b_hi[j] - box_hi))
        worst_b = max(worst_b, min(err, abs(b_lo[j] - box_lo), abs(b_hi[j] - box_hi)))
        print(f"   {od:+.2f}   [{b_lo[j]:+.3f},{b_hi[j]:+.3f}]   "
              f"[{box_lo:+.3f},{box_hi:+.3f}]   {du:+.3f}   "
              f"{max(abs(b_lo[j]-box_lo), abs(b_hi[j]-box_hi)):.3f}")
    print(f"  -> worst keep-out disagreement at dt=0 (after corridor clipping): {worst_b:.3f} m")
    print("     (a nonzero err column with a matching du means the band was CLIPPED to the "
           "corridor, which is correct -- the algebra only ever subtracts inside [lo,hi])")
    return H, C


def _shard(arg):
    mapname, stations, missed_keys = arg
    H = sw.Harness(mapname)
    C = Corridor(H)
    nd = len(Corridor.DPRIME)
    out = {"n": 0, "node_ok": 0, "neither": 0, "n_empty": 0, "n_split": 0, "width": [],
           "tube": [0] * nd, "reject_open": Counter(), "layout_open": Counter(), "examples": []}
    for (i, gap, lname, boxes, od, cd) in so.cells(H, stations):
        out["n"] += 1
        r = so.run(H, i, gap, cd, boxes)
        if r["ok"]:
            out["node_ok"] += 1
            continue
        if (i, gap, lname, od, cd) in missed_keys:
            continue                     # the dense oracle found this one; not a "neither" cell
        out["neither"] += 1
        empty_any, tube, k, nst, wide = C.test(i, gap, cd, boxes)
        if empty_any:
            out["n_empty"] += 1          # (i) some station has no lane at all
            continue
        for t in range(nd):
            out["tube"][t] += 1 if tube[t] else 0
        if not tube[-1]:
            out["n_split"] += 1          # (ii) lanes everywhere, but none threads through
            continue
        out["layout_open"][lname] += 1
        out["width"].append(round(wide, 4))
        if r["reject"]:
            sig = "+".join(nm for nm, v in zip(("bounds", "obs_box", "grid", "body", "curv"),
                                               r["reject"][1:]) if v) or "none"
            out["reject_open"][sig] += 1
        if len(out["examples"]) < 30:
            out["examples"].append({"station": i, "gap": gap, "layout": lname, "obs_d": od,
                                    "cur_d": cd, "n_stations": nst,
                                    "tube": [bool(x) for x in tube],
                                    "reject": r["reject"], "n_sampled": r["n_sampled"]})
    return out


def sweep(mapname, jobs, missed_keys, stride=None, limit=None):
    H = sw.Harness(mapname)
    st = list(range(0, len(H.wp) - 1, stride or sw.STATION_STRIDE))
    if limit:
        st = st[:limit]
    del H
    t0 = time.time()
    if jobs <= 1:
        parts = [_shard((mapname, st, missed_keys))]
    else:
        import multiprocessing as mp
        shards = [s for s in (st[k::jobs] for k in range(jobs)) if s]
        with mp.get_context("fork").Pool(len(shards)) as pool:
            parts = pool.map(_shard, [(mapname, s, missed_keys) for s in shards])
    R = {"map": mapname, "secs": time.time() - t0, "reject_open": Counter(),
         "layout_open": Counter(), "examples": [], "width": [],
         "dprime": list(Corridor.DPRIME)}
    for k in ("n", "node_ok", "neither", "n_empty", "n_split"):
        R[k] = sum(p[k] for p in parts)
    R["tube"] = [sum(p["tube"][t] for p in parts) for t in range(len(Corridor.DPRIME))]
    for p in parts:
        R["reject_open"].update(p["reject_open"])
        R["layout_open"].update(p["layout_open"])
        R["examples"] += p["examples"]
        R["width"] += p["width"]
    return R


def report(R):
    nei = max(R["neither"], 1)
    print(f"\n=== {R['map']} | {R['n']} cells | {R['secs']/60:.1f} min ===")
    print(f"  node_ok {R['node_ok']}; {R['neither']} cells NEITHER the node nor the dense "
          f"oracle found. Asking those the shape-free question:")
    print(f"    (i)  some station has NO lane at all            {R['n_empty']:6d}  "
          f"({100.0*R['n_empty']/nei:5.1f} %)   arithmetic: the track lacks the clearance")
    print(f"    (ii) lanes everywhere, none threads the window  {R['n_split']:6d}  "
          f"({100.0*R['n_split']/nei:5.1f} %)   arithmetic: d(s) is continuous, so no path either")
    print(f"    (iii) a lane threads it -- the CEILING, as a function of the lateral rate a path "
          f"is allowed:")
    for dp, t in zip(R["dprime"], R["tube"]):
        note = ""
        if dp == 0.0:
            note = "   (a line that never moves sideways -- describes nothing real)"
        elif dp == 0.5:
            note = "   (the node's own e_psi clip, tan(0.5) = 0.55)"
        elif dp in (0.2, 0.3):
            note = "   (what the shipped quintic humps actually reach)"
        print(f"          |d'| <= {dp:.1f}   {t:6d}  ({100.0*t/nei:5.1f} %){note}")
    if R["tube"][-1]:
        print("    what killed every node candidate in the (iii) cells:")
        for k, v in R["reject_open"].most_common(6):
            print(f"      {k:26s} {v:6d}")
        w = np.array(R["width"])
        print(f"    the (iii) tube at its TIGHTEST station, over {len(w)} cells: "
              f"median {100*np.median(w):.1f} cm, p10 {100*np.percentile(w,10):.1f}, "
              f"p90 {100*np.percentile(w,90):.1f}")
        for a, b in ((0, .05), (.05, .10), (.10, .20), (.20, .40), (.40, 9)):
            k = int(((w >= a) & (w < b)).sum())
            print(f"      {100*a:5.0f} - {100*b if b < 9 else 999:>3.0f} cm  {k:6d}  "
                  f"({100.0*k/max(len(w),1):5.1f} %)")
        print("    (iii) by obstacle layout: " +
              ", ".join(f"{k}={v}" for k, v in sorted(R["layout_open"].items())))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="*", default=["ifac_0807", "ifac"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--missed", default=None, help="oracle_matrix.json, to subtract the 637")
    ap.add_argument("--convention-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    for m in a.map:
        check_convention(m)
    if a.convention_only:
        return 0

    missed = {}
    if a.missed:
        for r in json.load(open(a.missed)):
            missed[r["map"]] = {(x["station"], x["gap"], x["layout"], x["obs_d"], x["cur_d"])
                                for x in r["missed"]}
    allR = []
    for m in a.map:
        R = sweep(m, a.jobs, missed.get(m, set()), a.stride, a.limit)
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
