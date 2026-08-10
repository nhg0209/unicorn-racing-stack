#!/usr/bin/env python3
"""WHERE the hump fails, and what a corridor-aware ramp would buy if it did not.

Three questions, in the order the answers depend on each other. All of them are asked ONLY of the
cells that stay closed with every retry spent AND whose min_violation is ZERO -- the cells where a
corridor exists at the full design margins and the planner still publishes nothing. Those are 36.1 %
(ifac) / 46.4 % (ifac_0807) of the still-closed cells, and by construction none of them is short of
margin. Whatever closes them is shape.

  (1) WHICH PART of the hump violates. Entry ramp, apex band, weave, or return ramp; how far the
      violating station sits from the apex it belongs to; and how wide the corridor is there against
      how wide it is at the apex. The standing hypothesis is that the ramps pass through NARROWER
      ground than the apex does. This file does not assume it -- it measures it, and reports it
      false if it is false.

  (2) THE CEILING. Keep the node's own apex offsets and knot placement exactly as it chose them,
      and re-solve ONLY the ramp segments' d(s) as a corridor-constrained minimum-bending QP:

          min ||D2 d||^2   s.t.   lo(s) <= d(s) <= hi(s),  both endpoints pinned

      then put the spliced path through EVERY check the node itself runs -- the waypoint bounds, the
      obstacle keep-out rectangle, the eroded sampling grid, the body floor at kernel 7, and the
      corner-fair curvature pair. Same principle as the oracle: an answer to a different question
      cannot be compared to the node's.

  (3) WHAT THE CURVATURE COSTS. A QP ramp meets the node's quintic apex at a breakpoint the QP never
      saw, so the junction can kink. (2)'s pass rate is reported with the curvature gate OFF and ON,
      and the junction's d' jump is measured. If most of (2) dies here, the direction is closed.

NOTHING in this file modifies static_avoidance_node, its parameters, or any margin. The QP is
offline arithmetic on the candidates the node already built.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_ramp.py --jobs 7

--convention-only prints just the corridor-convention table, which is the thing that has to be
believed before any count below it is.
"""
import argparse
import json
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
import sweep_static_corridor as sc             # noqa: E402
from gb_optimizer.closed_reopt import _second_difference   # noqa: E402

import quadprog                                # noqa: E402

SQUEEZE_SPEED = 2.0      # below squeeze_max_speed_mps (3.0): the schedule is non-empty
RACE_SPEED = 3.0         # the harness default, and exactly ON the gate

# do_spline's own _RAMP_LADDER, which is a local inside the function and therefore cannot be
# imported. Restated here and covered by the same fidelity check as the rest of the restatement:
# if this tuple ever stops matching the node's, the reject histogram stops matching too.
RAMP_LADDER = (1.0, 0.85, 0.7, 0.6, 0.5)

SEGMENTS = ("prefix", "entry", "apex", "weave", "return")


# =====================================================================================
# the open-segment QP
# =====================================================================================
def _open_second_difference(n: int, ds: float) -> np.ndarray:
    """The interior rows of closed_reopt._second_difference -- i.e. the SAME stencil with the
    periodic wrap removed.

    closed_reopt solves a closed track: its D2 has D2[0, -1] = D2[-1, 0] = 1 so the lap joins up.
    A ramp is an OPEN segment with both endpoints pinned, and keeping those two rows would charge
    it for a bend between its first station and its last -- two points several metres apart with
    nothing between them. Dropping them is exactly `[1:-1]`, and the remaining n-2 rows are the
    ordinary three-point stencil. Reusing the function rather than rewriting the stencil is what
    keeps the two solvers' definition of "bending" the same one.
    """
    return _second_difference(n, ds)[1:-1]


def solve_open(lo, hi, ds, w=0.0):
    """min ||D2 d||^2 + w||d||^2 s.t. lo <= d <= hi, over an OPEN station run.

    Same quadprog posing as closed_reopt.solve_offsets (C = [I, -I], b = [lo, -hi], meq = 0) and
    the same 1e-9 ridge for strict convexity; the only difference is the operator above. Endpoints
    are pinned by the caller collapsing lo and hi onto the value it wants -- a box of width 2e-6 is
    an equality to within a micrometre and stays inside quadprog's inequality form.
    """
    n = len(lo)
    if n < 3:
        return None
    D2 = _open_second_difference(n, ds)
    G = 2.0 * (D2.T @ D2 + w * np.eye(n)) + 1e-9 * np.eye(n)
    C = np.hstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([lo, -hi])
    try:
        return quadprog.solve_qp(G, np.zeros(n), C, b, 0)[0]
    except Exception:
        return None


# =====================================================================================
# the restatement: what do_spline's main pass builds, made inspectable
# =====================================================================================
class Plan:
    """Everything do_spline's first pass computes for one cell, kept instead of discarded.

    The node builds its candidate set inside one 300-line function and keeps none of it: the
    profiles, the per-candidate ramp lengths and the knot stations all die with the call. Both
    questions this file asks are about those objects, so they are rebuilt here from the node's own
    state using the node's own methods wherever a method exists (_gather_obstacles_ahead,
    _eval_clear_gate, _clears_obstacle, _anchor_car_idx, _gate_samples, _grid_corridor,
    _grid_corridor_batch, _bulge_away_from, _free_mask).

    What is RESTATED rather than called -- the knot loop, the terminal-offset grid, the ramp scan,
    the BPoly assembly, the bounds and keep-out tests -- is checked against the node's own account
    of the same pass: `(N sampled)` and the `reject bounds=/obs_box=/grid=/body=/curv=` histogram it
    prints when everything is rejected. Every closed cell prints that line, so the check runs on
    every cell in the population, not on a sample. See `fid_bad` in the shard.
    """

    __slots__ = ("ok", "why", "n", "N", "wpnt_dist", "s_local", "s_mod", "gap_wp", "idxs",
                 "d_cands", "d_ends", "entry_i", "knot_s", "knot_o", "knot_cor", "d_peak",
                 "r_in", "r_out", "s_entry0", "s_exit_end", "obs_enforce", "kappa_ref",
                 "lo_path", "hi_path", "obs_half_s", "obs_margin_d", "obs_margin_s",
                 "half_car", "d_left_arr", "d_right_arr", "seg", "reject", "counts")

    def __init__(self):
        self.ok = False
        self.why = ""


def build_plan(H, n):
    """do_spline's main pass (full margins, adaptive ramps, no ladder rung, no squeeze), rebuilt."""
    P = Plan()
    gbw = H.gbw
    wpnt_dist = gbw[1].s_m - gbw[0].s_m
    half_car = n.width_car / 2.0
    safety_margin, safety_margin_d, wall_margin = n.safety_margin, n.safety_margin_d, n.wall_margin
    obs_margin_s = half_car + safety_margin
    obs_margin_d = half_car + safety_margin_d
    sample_margin = half_car + wall_margin

    lookahead = min(max(n.lookahead_min, n.lookahead_k * (n.cur_vs or 0.0)), n.gb_max_s / 2.0)
    gather = min(lookahead + max(0.0, n.obs_gather_extra_m), n.gb_max_s / 2.0)

    cands_obs = n._gather_obstacles_ahead(n.obstacles, gather)
    obs_reach = [o for _g, o in cands_obs]
    obs_ahead = [o for g, o in cands_obs if g <= lookahead]
    if not obs_ahead:
        P.why = "no obstacle ahead"
        return P
    if n.clear_gate_enable and n._eval_clear_gate(obs_ahead, half_car):
        P.why = "clear gate idles"
        return P

    # --- knots (do_spline's own loop) ---
    knots = []
    L = n.gb_max_s
    for o in obs_reach:
        if n._clears_obstacle(o, obs_margin_d):
            continue
        gap_c = ((o.s_center - n.cur_s + L / 2.0) % L) - L / 2.0
        if gap_c <= 0.0:
            continue
        s_c = float(min(gap_c, gather))
        if knots and s_c <= knots[-1][0] + n.knot_merge_s_m:
            continue
        knots.append((s_c, o, int(o.s_center / wpnt_dist) % n.gb_max_idx))
        if len(knots) >= n.max_weave:
            break
    if not knots:
        P.why = "no knot"
        return P
    knotted = {id(ko) for (_s, ko, _c) in knots}
    obs_enforce = obs_ahead + [o for o in obs_reach if o not in obs_ahead and id(o) in knotted]

    nearest = knots[0][1]
    obs_half_s = ((nearest.s_end - nearest.s_start) % L) / 2.0
    s_exit_end_b = knots[-1][0] + n.return_len
    span = min(s_exit_end_b + n.tail_m, L * 0.9)

    car_idx = n._anchor_car_idx(int(n.cur_s / wpnt_dist) % n.gb_max_idx, gbw, wpnt_dist)
    grid_start_s = gbw[car_idx].s_m
    nst = max(int(span / wpnt_dist), 5)
    idxs = (car_idx + np.arange(nst)) % n.gb_max_idx
    s_abs = grid_start_s + np.arange(nst) * wpnt_dist
    s_mod = s_abs % L
    s_local = s_abs - grid_start_s
    gap_wp = (s_abs - n.cur_s) % L

    d_left_arr = np.array([gbw[j].d_left for j in idxs])
    d_right_arr = np.array([gbw[j].d_right for j in idxs])
    kappa_ref = np.array([gbw[j].kappa_radpm for j in idxs])

    # --- terminal offsets ---
    obs_j = int(nearest.s_center / wpnt_dist) % n.gb_max_idx
    d_hi_wp = gbw[obs_j].d_left - sample_margin
    d_lo_wp = -(gbw[obs_j].d_right - sample_margin)
    grid_cor = (n._grid_corridor(nearest.s_center, wall_margin=wall_margin)
                if n.trust_grid_bounds else None)
    d_lo, d_hi = grid_cor if grid_cor is not None else (d_lo_wp, d_hi_wp)
    obox_lo = min(nearest.d_right, nearest.d_left) - obs_margin_d
    obox_hi = max(nearest.d_right, nearest.d_left) + obs_margin_d
    if d_hi <= d_lo:
        d_ends = np.array([0.0])
    elif n.sample_gaps:
        n_side = max(2, int(n.n_d_samples) // 2)
        d_list = [0.0]
        lo_left = max(obox_hi, d_lo)
        if lo_left <= d_hi + 1e-6:
            d_list += list(np.linspace(lo_left, d_hi, n_side))
        hi_right = min(obox_lo, d_hi)
        if hi_right >= d_lo - 1e-6:
            d_list += list(np.linspace(d_lo, hi_right, n_side))
        d_list += n._gate_samples(obs_ahead, nearest, (d_lo, d_hi), obs_margin_d, obs_margin_s)
        d_ends = np.unique(np.round(np.asarray(d_list, dtype=float), 4))
        d_ends[int(np.argmin(np.abs(d_ends)))] = 0.0
    else:
        d_ends = np.linspace(d_lo, d_hi, int(n.n_d_samples))
        d_ends[int(np.argmin(np.abs(d_ends)))] = 0.0
    N = len(d_ends)

    def _corridor_at(cor_idx, s_c):
        g = (n._grid_corridor(s_c, wall_margin=wall_margin) if n.trust_grid_bounds else None)
        if g is not None:
            return g
        return (-(gbw[cor_idx].d_right - sample_margin), gbw[cor_idx].d_left - sample_margin)

    knot_cor = [(d_lo, d_hi)] + [_corridor_at(kc, _ko.s_center) for (_ks, _ko, kc) in knots[1:]]

    def _pass_offset(cor, o, prev_d):
        c_lo, c_hi = cor
        b_lo = min(o.d_right, o.d_left) - obs_margin_d
        b_hi = max(o.d_right, o.d_left) + obs_margin_d
        opts = []
        if b_hi <= c_hi + 1e-6:
            opts.append(b_hi)
        if b_lo >= c_lo - 1e-6:
            opts.append(b_lo)
        if not opts:
            return prev_d
        return min(opts, key=lambda d: abs(d - prev_d))

    # --- adaptive ramp scan ---
    scan_lo_s = knots[0][0] - n.ramp_len
    scan_hi_s = knots[-1][0] + n.return_len
    scan_s = np.arange(scan_lo_s, scan_hi_s + 1e-9, 0.5)
    scan_s_abs = (n.cur_s + scan_s) % L
    scan_lo = scan_hi = None
    if n.trust_grid_bounds:
        scan_d_max = float(np.max(np.abs(d_ends))) + n.apex_bulge + 0.10
        scan_lo, scan_hi = n._grid_corridor_batch(
            scan_s_abs, d_max=max(scan_d_max, 0.5), d_step=max(n.grid_scan_step, 0.10),
            wall_margin=wall_margin)
    if scan_lo is None:
        scan_lo = np.full(len(scan_s), np.nan)
        scan_hi = np.full(len(scan_s), np.nan)
    miss = ~np.isfinite(scan_lo)
    if miss.any():
        jj = (scan_s_abs[miss] / wpnt_dist).astype(int) % n.gb_max_idx
        scan_lo[miss] = np.array([-(gbw[j].d_right - sample_margin) for j in jj])
        scan_hi[miss] = np.array([gbw[j].d_left - sample_margin for j in jj])

    def _ramp_limits(s_c, full_len, entry):
        out = []
        for frac in RAMP_LADDER:
            R = max(full_len * frac, n.ramp_len_min_m)
            ds = (s_c - scan_s) if entry else (scan_s - s_c)
            m = (ds >= 0.0) & (ds <= R)
            if not m.any():
                out.append((R, np.inf, -np.inf))
                continue
            t = np.clip(1.0 - ds[m] / max(R, 1e-6), 0.0, 1.0)
            shape = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
            sig = shape > 1e-3
            if not sig.any():
                out.append((R, np.inf, -np.inf))
                continue
            hi_m, lo_m, sh = scan_hi[m][sig], scan_lo[m][sig], shape[sig]
            out.append((R, float(np.min(hi_m / sh)), float(np.max(lo_m / sh))))
            if R <= n.ramp_len_min_m + 1e-9:
                break
        return out

    ramp_lim_in = _ramp_limits(knots[0][0], n.ramp_len, entry=True)
    ramp_lim_out = _ramp_limits(knots[-1][0], n.return_len, entry=False)

    def _fit_ramp(amp, limits, full_len):
        if abs(amp) < 1e-6 or full_len <= n.ramp_len_min_m:
            return full_len
        for R, amp_hi, amp_lo in limits:
            if amp_lo - 1e-9 <= amp <= amp_hi + 1e-9:
                return R
        return limits[-1][0] if limits else full_len

    # --- the candidates ---
    e_psi = float(n.converter.get_e_psi(n.cur_x, n.cur_y, n.cur_yaw))
    dp0_full = float(np.tan(np.clip(e_psi, -0.5, 0.5)))
    d_cands = np.zeros((N, nst))
    entry_i = np.zeros(N, dtype=int)
    r_in_a = np.zeros(N)
    r_out_a = np.zeros(N)
    s_e0_a = np.zeros(N)
    s_ee_a = np.zeros(N)
    d_peak_a = np.zeros((N, len(knots)))
    pre = max(n.preramp_len_m, 1e-6)

    def _decay(x):
        t = np.clip(np.asarray(x, float) / pre, 0.0, 1.0)
        return 1.0 - t * t * t * (10.0 + t * (-15.0 + 6.0 * t))

    BPoly = H.san.BPoly
    for k, d_end in enumerate(d_ends):
        d_apex = [float(d_end)]
        for i in range(1, len(knots)):
            d_apex.append(_pass_offset(knot_cor[i], knots[i][1], d_apex[-1]))
        r_in = _fit_ramp(d_apex[0], ramp_lim_in, n.ramp_len)
        r_out = _fit_ramp(d_apex[-1], ramp_lim_out, n.return_len)
        s_entry0 = max(0.0, knots[0][0] - r_in)
        s_exit_end = knots[-1][0] + r_out
        m_span = (s_local > s_entry0) & (s_local <= s_exit_end)
        span_ok = s_exit_end > s_entry0 + 1e-3
        dp0 = dp0_full if s_entry0 == 0.0 else 0.0
        entry_i[k] = int(np.clip(np.searchsorted(s_local, s_entry0), 0, max(nst - 3, 0)))
        r_in_a[k], r_out_a[k], s_e0_a[k], s_ee_a[k] = r_in, r_out, s_entry0, s_exit_end
        dv = np.zeros(nst)
        if s_entry0 > 0.0:
            d_start = float(n.cur_d * _decay(s_entry0))
            if abs(n.cur_d) > 1e-9:
                m_pre = s_local <= s_entry0
                dv[m_pre] = n.cur_d * _decay(s_local[m_pre])
        else:
            d_start = n.cur_d
            dv[:] = n.cur_d
        if span_ok and m_span.any():
            bp_s = [s_entry0]
            bp_d = [[d_start, dp0, 0.0]]
            for i_k, ((s_c, _o, _cor), da) in enumerate(zip(knots, d_apex)):
                lo_k, hi_k = knot_cor[i_k]
                d_peak = float(np.clip(da + n._bulge_away_from(da, _o),
                                       min(lo_k, hi_k), max(lo_k, hi_k)))
                d_peak_a[k, i_k] = d_peak
                bp_s.append(max(s_c, bp_s[-1] + 1e-3))
                bp_d.append([d_peak, 0.0, 0.0])
            bp_s.append(max(s_exit_end, bp_s[-1] + 1e-3))
            bp_d.append([0.0, 0.0, 0.0])
            dv[m_span] = BPoly.from_derivatives(bp_s, bp_d)(s_local[m_span])
        dv[s_local > s_exit_end] = 0.0
        d_cands[k] = dv

    # --- the corridor over the PATH stations (finest read; see check_convention) ---
    lo_p, hi_p = n._grid_corridor_batch(s_mod, wall_margin=wall_margin)
    if lo_p is None:
        lo_p = np.full(nst, np.nan)
        hi_p = np.full(nst, np.nan)
    miss = ~np.isfinite(lo_p)
    if miss.any():
        lo_p[miss] = np.array([-(gbw[j].d_right - sample_margin) for j in idxs[miss]])
        hi_p[miss] = np.array([gbw[j].d_left - sample_margin for j in idxs[miss]])

    P.ok = True
    P.n, P.N, P.wpnt_dist = nst, N, wpnt_dist
    P.s_local, P.s_mod, P.gap_wp, P.idxs = s_local, s_mod, gap_wp, idxs
    P.d_cands, P.d_ends, P.entry_i = d_cands, d_ends, entry_i
    P.knot_s = [ks for (ks, _o, _c) in knots]
    P.knot_o = [ko for (_s, ko, _c) in knots]
    P.knot_cor, P.d_peak = knot_cor, d_peak_a
    P.r_in, P.r_out, P.s_entry0, P.s_exit_end = r_in_a, r_out_a, s_e0_a, s_ee_a
    P.obs_enforce, P.kappa_ref = obs_enforce, kappa_ref
    P.lo_path, P.hi_path = lo_p, hi_p
    P.obs_half_s, P.obs_margin_d, P.obs_margin_s, P.half_car = \
        obs_half_s, obs_margin_d, obs_margin_s, half_car
    P.d_left_arr, P.d_right_arr = d_left_arr, d_right_arr
    P.seg = segment_labels(P)
    return P


def segment_labels(P):
    """Which part of the hump each path station belongs to.

    The apex BAND, not the apex point: a single knot is one station, and "the violation was at the
    apex" has to mean "where the car is beside the box". That is the box's own s-inflated interval,
    half_car + safety_margin each side of its half-extent -- the same interval obs_ok enforces.
    Everything before the first band is entry, everything after the last is return, and anything
    between two bands is the weave a multi-box path carries. `prefix` is split off per candidate at
    use time, because s_entry0 moves with that candidate's own ramp length.
    """
    lab = np.empty(P.n, dtype=object)
    lab[:] = "entry"
    half = P.obs_half_s + P.obs_margin_s
    bands = [(ks - half, ks + half) for ks in P.knot_s]
    lab[P.s_local > bands[-1][1]] = "return"
    for (a, b) in bands:
        lab[(P.s_local >= a) & (P.s_local <= b)] = "apex"
    for i in range(len(bands) - 1):
        m = (P.s_local > bands[i][1]) & (P.s_local < bands[i + 1][0])
        lab[m] = "weave"
    return lab


# =====================================================================================
# the node's own checks, applied to an arbitrary offset profile
# =====================================================================================
def check_profiles(H, n, P, d_arr, entry_i, want_stations=False):
    """Every gate do_spline applies to a candidate, run on `d_arr` (M, n).

    Returns a dict of boolean (M,) verdicts per gate plus, optionally, the per-station violation
    masks the (1) analysis needs. The order and the definitions are the node's:

      bounds  skipped entirely when the grid is the corridor authority (_grid_is_authority)
      obs_box the same signed-centre-gap rectangle, over obs_enforce
      grid    _path_off_track on the sampling image, from cand_entry_i onward
      body    _path_body_unsafe on the body image (kernel 7), from cand_entry_i onward
      curv    |kappa - kappa_ref| <= kappa_add_max AND |kappa| <= kappa_abs_max, over the WHOLE
              path -- the node computes kappa on `xy`, not on `xy_own`
    """
    M = d_arr.shape[0]
    out = {}
    if n._grid_is_authority():
        bounds = np.ones(M, dtype=bool)
        bmask = np.zeros((M, P.n), dtype=bool)
    else:
        bmask = ((d_arr > (P.d_left_arr - P.half_car)[None, :]) |
                 (d_arr < -(P.d_right_arr - P.half_car)[None, :]))
        bounds = ~bmask.any(axis=1)
    out["bounds"] = bounds

    obs = np.ones(M, dtype=bool)
    omask = np.zeros((M, P.n), dtype=bool)
    L = n.gb_max_s
    for o in P.obs_enforce:
        o_span = (o.s_end - o.s_start) % L
        gc = (o.s_center - n.cur_s) % L
        if gc > L / 2.0:
            gc -= L
        g0 = gc - o_span / 2.0 - P.obs_margin_s
        g1 = gc + o_span / 2.0 + P.obs_margin_s
        d_box_lo = min(o.d_right, o.d_left) - P.obs_margin_d
        d_box_hi = max(o.d_right, o.d_left) + P.obs_margin_d
        s_in = (P.gap_wp >= g0) & (P.gap_wp <= g1)
        d_in = (d_arr >= d_box_lo) & (d_arr <= d_box_hi)
        hit = d_in & s_in[None, :]
        omask |= hit
        obs &= ~hit.any(axis=1)
    out["obs_box"] = obs

    resp = n.converter.get_cartesian(np.tile(P.s_mod, M), d_arr.reshape(-1))
    xy = (resp.T if resp.ndim == 2 else resp).reshape(M, P.n, 2)
    out["xy"] = xy

    free_s = n._free_mask(xy.reshape(-1, 2), n.map_filter)
    free_b = n._free_mask(xy.reshape(-1, 2), n.body_filter)
    gmask = np.zeros((M, P.n), dtype=bool) if free_s is None else ~free_s.reshape(M, P.n)
    ymask = np.zeros((M, P.n), dtype=bool) if free_b is None else ~free_b.reshape(M, P.n)
    own = np.arange(P.n)[None, :] >= np.asarray(entry_i)[:, None]
    out["grid"] = (~(gmask & own).any(axis=1) if n.use_grid_check else np.ones(M, dtype=bool))
    out["body"] = ~(ymask & own).any(axis=1)

    tph = H.san.tph
    el = P.wpnt_dist * np.ones(P.n - 1)
    kap = np.empty((M, P.n))
    for k in range(M):
        _psi, kk = tph.calc_head_curv_num.calc_head_curv_num(
            path=xy[k], el_lengths=el, is_closed=False)
        kap[k] = kk
    cmask = (np.abs(kap - P.kappa_ref[None, :]) > n.kappa_add_max) | (np.abs(kap) > n.kappa_abs_max)
    out["curv"] = ~cmask.any(axis=1)
    out["kappa"] = kap
    if want_stations:
        out["m_bounds"], out["m_obs"] = bmask, omask
        out["m_grid"], out["m_body"] = gmask & own, ymask & own
        out["m_curv"] = cmask
        out["own"] = own
    return out


def node_order_reject(chk):
    """The node's SHORT-CIRCUIT accounting: a candidate is charged to the FIRST gate that refuses
    it, in the order bounds -> obs_box -> grid -> body -> curv, and the later gates never run.
    Counting a candidate under every gate it fails would produce a histogram the node never prints
    and the fidelity check would fail against its own evidence."""
    M = len(chk["bounds"])
    cnt = dict.fromkeys(("bounds", "obs_box", "grid", "body", "curv"), 0)
    feas = np.zeros(M, dtype=bool)
    charged = np.empty(M, dtype=object)
    for k in range(M):
        for g in ("bounds", "obs_box", "grid", "body", "curv"):
            if not chk[g][k]:
                cnt[g] += 1
                charged[k] = g
                break
        else:
            feas[k] = True
            charged[k] = "feasible"
    return cnt, feas, charged


# =====================================================================================
# (1) where the violation is
# =====================================================================================
def locate(P, chk, charged):
    """For every REJECTED candidate: the station it violates, which segment that falls in, how far
    it sits from the apex it belongs to, and the corridor width there against the apex's.

    Two independent notions of violation are kept apart deliberately:

      GATE       the station that actually made the node reject -- the first failing gate's first
                 offending station. This is the node's own verdict, located.
      CORRIDOR   d(s) outside [lo(s), hi(s)] -- the interval the sampling and the ramp scan are
                 built from, and the one the QP in (2) constrains. A candidate can fail a gate with
                 no corridor violation at all (the keep-out rectangle is not part of the corridor,
                 and curvature is not a lateral question), and that difference is itself a finding:
                 it bounds what a corridor-aware ramp can possibly fix.
    """
    rows = []
    d = P.d_cands
    viol = np.maximum(d - P.hi_path[None, :], P.lo_path[None, :] - d)      # >0 == outside
    kn = np.asarray(P.knot_s)
    width = P.hi_path - P.lo_path
    for k in range(P.N):
        g = charged[k]
        if g == "feasible":
            continue
        own = np.arange(P.n) >= P.entry_i[k]
        inspan = own & (P.s_local <= P.s_exit_end[k] + 1e-9)
        mk = {"bounds": "m_bounds", "obs_box": "m_obs", "grid": "m_grid",
              "body": "m_body", "curv": "m_curv"}[g]
        gm = chk[mk][k] & inspan
        cm = (viol[k] > 1e-6) & inspan
        for nm, mask in (("gate", gm), ("corridor", cm)):
            idx = np.flatnonzero(mask)
            if not idx.size:
                if nm == "corridor":
                    rows.append({"kind": "corridor", "gate": g, "none": True})
                continue
            if nm == "corridor":
                j = int(idx[np.argmax(viol[k][idx])])          # the worst one: what a fix must move
            else:
                j = int(idx[0])                                # the first one the node would hit
            amount = float(max(viol[k][j], 0.0))
            need = float(np.clip(d[k, j], P.lo_path[j], P.hi_path[j]))
            ka = int(np.argmin(np.abs(kn - P.s_local[j])))
            j_apex = int(np.argmin(np.abs(P.s_local - kn[ka])))
            seg = P.seg[j]
            if seg == "entry" and P.s_local[j] <= P.s_entry0[k]:
                seg = "prefix"
            rows.append({
                "kind": nm, "gate": g, "none": False, "seg": seg,
                "ds_apex": float(P.s_local[j] - kn[ka]),
                "w_here": float(width[j]), "w_apex": float(width[j_apex]),
                "amount": amount, "d_here": float(d[k, j]), "d_need": need,
                "n_st": int(idx.size), "r_in": float(P.r_in[k]), "r_out": float(P.r_out[k]),
                "d_end": float(P.d_ends[k]),
            })
    return rows


# =====================================================================================
# (2) the ceiling: corridor-decided ramps
# =====================================================================================
def _pin(lo, hi, i, v):
    """Pin station i to v, WIDENING the box if v lies outside the corridor there.

    The apex offset is given, not negotiable: it is what the node chose and what (2) is defined to
    keep. Where the corridor at the apex station does not contain it -- which happens, because the
    node clips the peak to the corridor read at the OBSTACLE's station while the path is evaluated
    at the nearest STATION -- refusing to solve would silently drop the candidate and understate the
    ceiling. Widening the single pinned station instead keeps the apex exactly where the node put it
    and lets the QP shape everything around it, and the full check afterwards still judges the
    result. `widened` is counted and reported.
    """
    wide = v < lo[i] - 1e-6 or v > hi[i] + 1e-6
    lo[i] = hi[i] = v
    return wide


def qp_solve(P, k, mode, w_reg=0.0):
    """Re-solve part of candidate k's offset profile as corridor-constrained minimum bending.

    mode = "ramps"  round-4 item (2) as specified: ONLY the entry ramp (s_entry0 -> first knot) and
                    the return ramp (last knot -> s_exit_end). Each is its own open QP, pinned at
                    both ends to the value the node's own quintic has at those stations, so the
                    spliced profile stays continuous with the parts that were not re-solved and
                    "0 at one end, the apex offset at the other" is discretised onto the station
                    grid rather than asserted between stations. On a single-box layout this is the
                    whole maneuver; on a woven one it leaves the inter-apex segments alone.

    mode = "span"   the same question asked of every station the maneuver owns, with the APEX BANDS
                    pinned station by station to the node's own values. Strictly wider than "ramps"
                    -- it adds the inter-apex weave, which "ramps" cannot touch -- and it exists
                    because (1) says that is where the violations are. It still keeps what item (2)
                    said to keep: the apex offsets and the knot placement are the node's.

    Returns (d_new, info) or (None, reason).
    """
    d0 = P.d_cands[k]
    d_new = d0.copy()
    ds = P.wpnt_dist
    kn = P.knot_s
    half = P.obs_half_s + P.obs_margin_s
    info = {"n": 0, "widened": 0}
    if mode == "ramps":
        runs = [(P.s_entry0[k], kn[0]), (kn[-1], P.s_exit_end[k])]
    else:
        runs = [(P.s_entry0[k], P.s_exit_end[k])]
    for (a, b) in runs:
        m = np.flatnonzero((P.s_local >= a - 1e-9) & (P.s_local <= b + 1e-9))
        if m.size < 4:
            continue                       # shorter than three stations: nothing to bend
        lo = P.lo_path[m].copy()
        hi = P.hi_path[m].copy()
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            return None, "corridor_unmeasurable"
        info["widened"] += int(_pin(lo, hi, 0, d0[m[0]]))
        info["widened"] += int(_pin(lo, hi, -1, d0[m[-1]]))
        if mode == "span":
            for ks in kn:                                   # the apex bands stay the node's
                for j in np.flatnonzero((P.s_local[m] >= ks - half) & (P.s_local[m] <= ks + half)):
                    info["widened"] += int(_pin(lo, hi, int(j), d0[m[int(j)]]))
        lo = lo - 1e-6
        hi = hi + 1e-6
        if np.any(hi < lo):
            return None, "corridor_empty"
        sol = solve_open(lo, hi, ds, w_reg)
        if sol is None:
            return None, "qp_failed"
        d_new[m] = sol
        info["n"] += int(m.size)
    return d_new, info


def dprime_jump(P, k, d):
    """|d'| discontinuity at the four seams the splice creates: the two ramp ends and the two apex
    junctions. One-sided differences on the station grid, so the number is the kink the controller
    would actually see at 0.1 m resolution."""
    ds = P.wpnt_dist
    seams = [P.s_entry0[k], P.knot_s[0], P.knot_s[-1], P.s_exit_end[k]]
    worst = 0.0
    for s in seams:
        j = int(np.argmin(np.abs(P.s_local - s)))
        if j < 2 or j > P.n - 3:
            continue
        before = (d[j] - d[j - 1]) / ds
        after = (d[j + 1] - d[j]) / ds
        worst = max(worst, abs(after - before))
    return float(worst)


# =====================================================================================
# the shard
# =====================================================================================
def _shard(arg):
    mapname, stations, cap_rows = arg
    H = sw.Harness(mapname)
    C = sc.Corridor(H)
    o = {"n_raw": 0, "n": 0, "unplaceable": 0, "node_ok": 0, "sq_ok": 0, "relax_ok": 0,
         "closed": 0, "d0": 0, "dpos": 0, "dnone": 0,
         "fid_ok": 0, "fid_bad": 0, "fid_examples": [], "plan_bad": 0, "feas_leak": 0,
         "rows": [], "seg_gate": Counter(), "seg_corr": Counter(), "gate_hist": Counter(),
         "corr_none": Counter(), "seg_by_knots": Counter(),
         "closed_by_layout": Counter(), "d0_by_layout": Counter(), "d0_knots": Counter(),
         "qp_cells": 0, "qp_cand": 0}
    for tag in ("", "s_"):
        o.update({tag + "qp_open_nocurv": 0, tag + "qp_open": 0, tag + "qp_solved": 0,
                  tag + "qp_widened": 0, tag + "qp_fail": Counter(), tag + "qp_gate": Counter(),
                  tag + "qp_kink": [], tag + "qp_kappa": [], tag + "qp_kappa_add": [],
                  tag + "base_kink": [], tag + "open_by_layout": Counter(),
                  tag + "open_nocurv_by_layout": Counter(), tag + "open_ds": []})
    for (i, gap, lname, boxes, od, cd) in so.cells(H, stations, so.RACE_LAYOUTS, so.RACE_CUR_D):
        o["n_raw"] += 1
        if not C.placeable(i, boxes):
            o["unplaceable"] += 1
            continue
        o["n"] += 1
        H.cur_vs, H.force_relax = RACE_SPEED, False
        if so.run(H, i, gap, cd, boxes)["ok"]:
            o["node_ok"] += 1
            continue
        H.cur_vs = SQUEEZE_SPEED
        sq = so.run(H, i, gap, cd, boxes)["ok"]
        H.cur_vs, H.force_relax = RACE_SPEED, True
        rx = so.run(H, i, gap, cd, boxes)["ok"]
        H.force_relax = False
        if sq:
            o["sq_ok"] += 1
        if rx:
            o["relax_ok"] += 1
        if sq or rx:
            continue
        o["closed"] += 1
        o["closed_by_layout"][lname] += 1
        H.cur_vs = RACE_SPEED
        delta = C.min_violation(i, gap, cd, boxes)
        if delta is None:
            o["dnone"] += 1
            continue
        if delta > 1e-9:
            o["dpos"] += 1
            continue
        o["d0"] += 1
        o["d0_by_layout"][lname] += 1

        # ---- the delta = 0 population: rebuild what the node built -------------------------
        r = so.run(H, i, gap, cd, boxes)
        n = H.make_node(i, gap, cd, ladder=True, boxes=boxes)
        try:
            P = build_plan(H, n)
        except Exception as e:                                    # noqa: BLE001
            o["plan_bad"] += 1
            if len(o["fid_examples"]) < 8:
                o["fid_examples"].append({"st": i, "gap": gap, "why": f"{type(e).__name__}: {e}"})
            continue
        if not P.ok:
            o["plan_bad"] += 1
            continue
        chk = check_profiles(H, n, P, P.d_cands, P.entry_i, want_stations=True)
        cnt, feas, charged = node_order_reject(chk)
        # FIDELITY: the node's own account of this same pass, on every cell
        ref = r["reject"]
        mine = (P.N, cnt["bounds"], cnt["obs_box"], cnt["grid"], cnt["body"], cnt["curv"])
        if ref is not None and tuple(ref) == mine:
            o["fid_ok"] += 1
        else:
            o["fid_bad"] += 1
            if len(o["fid_examples"]) < 8:
                o["fid_examples"].append({"st": i, "gap": gap, "layout": lname, "obs_d": od,
                                          "cur_d": cd, "node": ref, "mine": list(mine)})
        if feas.any():
            # the node published nothing, so no candidate of this pass may pass here
            o["feas_leak"] += 1

        # ---- (1) ----
        nk = len(P.knot_s)
        o["d0_knots"][nk] += 1
        for rr in locate(P, chk, charged):
            rr["layout"], rr["nknot"] = lname, nk
            if rr["none"]:
                o["corr_none"][rr["gate"]] += 1
                continue
            kind = rr["kind"]
            (o["seg_gate"] if kind == "gate" else o["seg_corr"])[rr["seg"]] += 1
            o["seg_by_knots"][(kind, nk, rr["seg"])] += 1
            if kind == "gate":
                o["gate_hist"][rr["gate"]] += 1
            if len(o["rows"]) < cap_rows:
                o["rows"].append(rr)

        # ---- (2) + (3), in both scopes ----
        o["qp_cells"] += 1
        for mode, tag in (("ramps", ""), ("span", "s_")):
            d_new, keep = [], []
            for k in range(P.N):
                if charged[k] == "feasible":
                    continue
                if not tag:
                    o["qp_cand"] += 1
                dn, why = qp_solve(P, k, mode)
                if dn is None:
                    o[tag + "qp_fail"][why] += 1
                    continue
                o[tag + "qp_solved"] += 1
                o[tag + "qp_widened"] += int(why["widened"] > 0)
                d_new.append(dn)
                keep.append(k)
            if not d_new:
                continue
            d_new = np.asarray(d_new)
            c2 = check_profiles(H, n, P, d_new, P.entry_i[keep])
            hard = c2["bounds"] & c2["obs_box"] & c2["grid"] & c2["body"]
            allg = hard & c2["curv"]
            for t in range(len(keep)):
                if not hard[t]:
                    for g in ("bounds", "obs_box", "grid", "body"):
                        if not c2[g][t]:
                            o[tag + "qp_gate"][g] += 1
                            break
                elif not c2["curv"][t]:
                    o[tag + "qp_gate"]["curv"] += 1
                else:
                    o[tag + "qp_gate"]["pass"] += 1
            if hard.any():
                o[tag + "qp_open_nocurv"] += 1
                o[tag + "open_nocurv_by_layout"][lname] += 1
            if allg.any():
                o[tag + "qp_open"] += 1
                o[tag + "open_by_layout"][lname] += 1
                o[tag + "open_ds"].append([i, gap, lname, od, cd])
                t = int(np.flatnonzero(allg)[0])
                o[tag + "qp_kink"].append(dprime_jump(P, keep[t], d_new[t]))
                o[tag + "base_kink"].append(dprime_jump(P, keep[t], P.d_cands[keep[t]]))
                kk = c2["kappa"][t]
                o[tag + "qp_kappa"].append(float(np.max(np.abs(kk))))
                o[tag + "qp_kappa_add"].append(float(np.max(np.abs(kk - P.kappa_ref))))
    return o


# =====================================================================================
# the corridor-convention check
# =====================================================================================
def check_convention(mapname, n_show=10):
    """Before (2) is believed: is the corridor this file constrains the QP with the SAME corridor
    the node reads, at the SAME stations?

    Three reads of the same quantity, which can disagree in two different ways:

      A  this file's per-path-station batch (_grid_corridor_batch at the default 5 cm lateral step)
      B  the node's own _grid_corridor, one station at a time -- the call do_spline makes for its
         sampling limits and for every knot's corridor
      C  the node's RAMP SCAN, which is the same batch call at d_step = 0.10 and only every 0.5 m,
         then interpolated

    A vs B is the convention question: they must agree to numerical noise, or the QP is constrained
    by an interval the planner never used. A vs C is not an error -- it is the resolution the ramp
    fit actually ran at, and the gap is worth knowing because it is the node's own blind spot.

    The closed-loop convention is checked too: closed_reopt's corridor is a full lap and its D2
    wraps, while the QP here is an open run with pinned ends. The last line prints both operators'
    bending cost on a ramp-shaped profile, so "the wrap was removed" is measured, not asserted.
    """
    H = sw.Harness(mapname)
    print(f"\n=== corridor convention | {mapname} ===")
    print("   s [m]   A: this file        B: node _grid_corridor   |A-B|      C: ramp scan (0.10 m)"
          "  |A-C|  |A-C'|")
    worstAB = worstAC = worstAW = 0.0
    rows = 0
    for i in H.stations[::max(1, len(H.stations) // 5)][:5]:
        n = H.make_node(i, 12.0, 0.0, ladder=True, boxes=((0.0, 0.0),))
        P = build_plan(H, n)
        if not P.ok:
            continue
        wm = n.wall_margin
        s_scan = np.arange(P.knot_s[0] - n.ramp_len, P.knot_s[-1] + n.return_len + 1e-9, 0.5)
        # the node's own scan_d_max, not a round number: the ramp scan sweeps only as wide as the
        # sampled offsets plus the bulge and a cell, and where the corridor is wider than that the
        # scan reports it ending at the edge of its own sweep
        d_max = max(float(np.max(np.abs(P.d_ends))) + n.apex_bulge + 0.10, 0.5)
        slo, shi = n._grid_corridor_batch((n.cur_s + s_scan) % n.gb_max_s, d_max=d_max,
                                          d_step=max(n.grid_scan_step, 0.10), wall_margin=wm)
        # the same scan with the sweep width taken OFF, so the two causes separate: what the 0.5 m
        # station spacing costs, and what the narrowed sweep costs on top of it
        wlo, whi = n._grid_corridor_batch((n.cur_s + s_scan) % n.gb_max_s, d_max=n.grid_scan_max,
                                          d_step=max(n.grid_scan_step, 0.10), wall_margin=wm)
        for j in range(0, P.n, max(1, P.n // 3)):
            g = n._grid_corridor(float(P.s_mod[j]), wall_margin=wm)
            if g is None or not np.isfinite(P.lo_path[j]):
                continue
            eAB = max(abs(g[0] - P.lo_path[j]), abs(g[1] - P.hi_path[j]))
            sl = float(np.interp(P.s_local[j], s_scan, slo))
            sh = float(np.interp(P.s_local[j], s_scan, shi))
            wl = float(np.interp(P.s_local[j], s_scan, wlo))
            wh = float(np.interp(P.s_local[j], s_scan, whi))
            eAC = max(abs(sl - P.lo_path[j]), abs(sh - P.hi_path[j]))
            eAW = max(abs(wl - P.lo_path[j]), abs(wh - P.hi_path[j]))
            worstAB, worstAC = max(worstAB, eAB), max(worstAC, eAC)
            worstAW = max(worstAW, eAW)
            rows += 1
            if rows <= n_show:
                print(f"  {P.s_mod[j]:7.2f}  [{P.lo_path[j]:+.3f},{P.hi_path[j]:+.3f}]   "
                      f"[{g[0]:+.3f},{g[1]:+.3f}]        {eAB:.2e}   "
                      f"[{sl:+.3f},{sh:+.3f}]         {eAC:.3f}  {eAW:.3f}")
    print(f"  -> A vs B (the convention): worst {worstAB:.2e} m over {rows} stations "
          f"-- must be numerical noise")
    print(f"  -> A vs C (the ramp scan as the node runs it: 0.5 m stations, 0.10 m lateral step, "
          f"sweep narrowed to the offsets the path can reach): worst {worstAC:.3f} m")
    print(f"  -> A vs C with the sweep width restored: worst {worstAW:.3f} m -- so of the "
          f"disagreement, {worstAC - worstAW:.3f} m is the NARROWED SWEEP and the rest is the "
          f"0.5 m station spacing. Neither is an error: it is the resolution _ramp_limits "
          f"decided the ramp length at.")
    nn, ds = 45, 0.1
    t = np.linspace(0.0, 1.0, nn)
    prof = 0.4 * t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
    Dc, Do = _second_difference(nn, ds), _open_second_difference(nn, ds)
    print(f"  -> operator: closed D2 has {Dc.shape[0]} rows (2 of them wrap the lap), open has "
          f"{Do.shape[0]}; on a 0.4 m ramp ||D2c p||^2 = {float(np.sum((Dc @ prof)**2)):.1f} vs "
          f"open {float(np.sum((Do @ prof)**2)):.1f} -- the difference IS the two phantom wrap "
          f"rows, which an open segment must not be charged for")
    return H


# =====================================================================================
def sweep(mapname, jobs, stride=None, limit=None, cap_rows=6000):
    H = sw.Harness(mapname)
    st = list(range(0, len(H.wp) - 1, stride or sw.STATION_STRIDE))
    if limit:
        st = st[:limit]
    del H
    t0 = time.time()
    if jobs <= 1:
        parts = [_shard((mapname, st, cap_rows))]
    else:
        import multiprocessing as mp
        shards = [s for s in (st[k::jobs] for k in range(jobs)) if s]
        with mp.get_context("fork").Pool(len(shards)) as pool:
            parts = pool.map(_shard, [(mapname, s, cap_rows // max(len(shards), 1))
                                      for s in shards])
    R = {"map": mapname, "secs": time.time() - t0}
    ints = ["n_raw", "n", "unplaceable", "node_ok", "sq_ok", "relax_ok", "closed", "d0",
            "dpos", "dnone", "fid_ok", "fid_bad", "plan_bad", "feas_leak", "qp_cells", "qp_cand"]
    cnts = ["seg_gate", "seg_corr", "gate_hist", "corr_none", "closed_by_layout",
            "d0_by_layout", "d0_knots"]
    lists = ["rows", "fid_examples"]
    for tag in ("", "s_"):
        ints += [tag + x for x in ("qp_open_nocurv", "qp_open", "qp_solved", "qp_widened")]
        cnts += [tag + x for x in ("qp_fail", "qp_gate", "open_by_layout",
                                   "open_nocurv_by_layout")]
        lists += [tag + x for x in ("qp_kink", "base_kink", "qp_kappa", "qp_kappa_add", "open_ds")]
    for k in ints:
        R[k] = sum(p[k] for p in parts)
    for k in cnts:
        c = Counter()
        for p in parts:
            c.update(p[k])
        R[k] = c
    for k in lists:
        R[k] = [x for p in parts for x in p[k]]
    sbk = Counter()
    for p in parts:
        sbk.update(p["seg_by_knots"])
    R["seg_by_knots"] = Counter({"|".join(str(x) for x in kk): v for kk, v in sbk.items()})
    return R


def _pct(a, b):
    return 100.0 * a / max(b, 1)


def report(R):
    print(f"\n=== {R['map']} | race profile | {R['secs']/60:.1f} min ===")
    n, ref = R["n"], R["n"] - R["node_ok"]
    print(f"  {R['n_raw']} cells, {R['unplaceable']} unplaceable -> {n} race-realistic")
    print(f"  planner publishes {R['node_ok']} ({_pct(R['node_ok'], n):.1f} %) | REFUSES {ref} "
          f"({_pct(ref, n):.1f} %) | squeeze recovers {R['sq_ok']}, relax {R['relax_ok']}")
    print(f"  still closed with both retries spent: {R['closed']} "
          f"({_pct(R['closed'], ref):.1f} % of refusals)")
    d0 = R["d0"]
    print(f"    delta = 0 (a corridor exists at the FULL design margins): {d0} "
          f"({_pct(d0, R['closed']):.1f} % of closed, {_pct(d0, ref):.1f} % of refusals)"
          f"   <- the population everything below is about")
    print(f"    delta > 0: {R['dpos']} | needs more margin than exists: {R['dnone']}")
    print("    still closed by layout: " +
          ", ".join(f"{k}={v}" for k, v in sorted(R["closed_by_layout"].items())) +
          "  | of those, delta=0: " +
          ", ".join(f"{k}={v}" for k, v in sorted(R["d0_by_layout"].items())))
    print("    delta=0 cells by the number of KNOTS the path carries: " +
          ", ".join(f"{k} knot={v}" for k, v in sorted(R["d0_knots"].items())))

    print("\n  RESTATEMENT FIDELITY (the node's own '(N sampled) reject ...' line against this "
          "file's\n  recomputation of the same pass, on EVERY delta=0 cell):")
    print(f"    match {R['fid_ok']} | MISMATCH {R['fid_bad']} | plan not rebuildable "
          f"{R['plan_bad']} | candidate wrongly feasible {R['feas_leak']}")
    if R["fid_bad"] or R["feas_leak"]:
        print("    !! the numbers below describe this file's restatement, not the node. Examples:")
        for e in R["fid_examples"][:5]:
            print(f"       {e}")

    # --- (1) ---
    print("\n  (1) WHERE THE VIOLATION IS, over the rejected candidates of the delta=0 cells")
    tg = sum(R["seg_gate"].values())
    tc = sum(R["seg_corr"].values())
    print(f"    the station that made the NODE reject (its first failing gate), by segment "
          f"[{tg} candidates]:")
    for s in SEGMENTS:
        v = R["seg_gate"][s]
        print(f"      {s:8s} {v:7d}  ({_pct(v, tg):5.1f} %)")
    print("      by gate: " + ", ".join(f"{k}={v}" for k, v in R["gate_hist"].most_common()))
    print(f"    the worst CORRIDOR violation (d outside [lo,hi]), by segment [{tc}]:")
    for s in SEGMENTS:
        v = R["seg_corr"][s]
        print(f"      {s:8s} {v:7d}  ({_pct(v, tc):5.1f} %)")
    nc = sum(R["corr_none"].values())
    print(f"      candidates rejected with NO corridor violation anywhere: {nc} "
          f"({_pct(nc, nc + tc):5.1f} % of rejected candidates)  by gate: "
          + ", ".join(f"{k}={v}" for k, v in R["corr_none"].most_common(5)))
    print("    BY HOW MANY KNOTS THE PATH CARRIES -- a one-knot path has no weave at all, so this "
          "is\n    the split that says whether 'the ramps' is the right name for the failure:")
    sbk = R["seg_by_knots"]
    for kind in ("gate", "corridor"):
        for nk in (1, 2, 3):
            tot = sum(v for k, v in sbk.items() if k.startswith(f"{kind}|{nk}|"))
            if not tot:
                continue
            parts_ = ", ".join(f"{s}={sbk.get(f'{kind}|{nk}|{s}', 0)} "
                               f"({_pct(sbk.get(f'{kind}|{nk}|{s}', 0), tot):.0f} %)"
                               for s in SEGMENTS if sbk.get(f"{kind}|{nk}|{s}", 0))
            print(f"      [{kind}] {nk} knot(s), {tot:6d} candidates: {parts_}")
    rows = R["rows"]
    for kind in ("gate", "corridor"):
        rr = [x for x in rows if x["kind"] == kind and not x["none"]]
        if not rr:
            continue
        ds = np.array([x["ds_apex"] for x in rr])
        wh = np.array([x["w_here"] for x in rr])
        wa = np.array([x["w_apex"] for x in rr])
        am = np.array([x["amount"] for x in rr])
        print(f"    [{kind}] {len(rr)} sampled violations")
        print(f"      distance from the apex it belongs to: median {np.median(ds):+.2f} m, "
              f"p10 {np.percentile(ds, 10):+.2f}, p90 {np.percentile(ds, 90):+.2f} "
              f"(negative = BEFORE the apex, i.e. on the entry side)")
        for a, b in ((-99, -3), (-3, -1.5), (-1.5, -0.45), (-0.45, 0.45), (0.45, 1.5),
                     (1.5, 3), (3, 99)):
            k = int(((ds >= a) & (ds < b)).sum())
            print(f"        {a:+6.2f} .. {b:+6.2f} m  {k:6d}  ({_pct(k, len(rr)):5.1f} %)")
        print(f"      corridor width AT the violation: median {100*np.median(wh):.1f} cm | "
              f"at its APEX station: median {100*np.median(wa):.1f} cm | "
              f"narrower at the violation in {_pct(int((wh < wa - 1e-6).sum()), len(rr)):.1f} % "
              f"of them (median w_here - w_apex = {100*np.median(wh - wa):+.1f} cm)")
        if kind == "corridor":
            print(f"      how far outside: median {100*np.median(am):.1f} cm, "
                  f"p90 {100*np.percentile(am, 90):.1f} cm, max {100*np.max(am):.1f} cm")

    # --- (2) ---
    print("\n  (2) CEILING: d(s) re-solved as a corridor-constrained minimum-bending QP, with the "
          "node's\n      apex offsets and knot placement kept exactly as it chose them")
    print(f"    cells attempted {R['qp_cells']} | rejected candidates {R['qp_cand']}")
    for tag, title in (("", "A. THE RAMPS ONLY -- entry and return, as item (2) specifies"),
                       ("s_", "B. THE WHOLE SPAN -- the same question also asked of the inter-apex "
                              "weave,\n       apex bands still pinned. Strictly wider than A; it "
                              "exists because (1) says\n       that is where the violations are.")):
        print(f"\n    {title}")
        print(f"      solved {R[tag+'qp_solved']} "
              f"({_pct(R[tag+'qp_solved'], R['qp_cand']):.1f} % of candidates), of which "
              f"{R[tag+'qp_widened']} needed a pinned station's box widened to hold the node's own "
              f"apex")
        if R[tag + "qp_fail"]:
            print("      QP not posed: " +
                  ", ".join(f"{k}={v}" for k, v in R[tag + "qp_fail"].most_common()))
        print(f"      cells OPENED, curvature gate OFF: {R[tag+'qp_open_nocurv']}  "
              f"({_pct(R[tag+'qp_open_nocurv'], d0):.1f} % of delta=0, "
              f"{_pct(R[tag+'qp_open_nocurv'], ref):.1f} % of ALL refused cells)")
        print(f"      cells OPENED, every node check ON: {R[tag+'qp_open']}  "
              f"({_pct(R[tag+'qp_open'], d0):.1f} % of delta=0, "
              f"{_pct(R[tag+'qp_open'], ref):.1f} % of ALL refused cells)")
        print("      what happens to a QP candidate (first failing gate): " +
              ", ".join(f"{k}={v}" for k, v in R[tag + "qp_gate"].most_common()))
        if R[tag + "open_by_layout"]:
            print("      opened by layout: " +
                  ", ".join(f"{k}={v}" for k, v in sorted(R[tag + "open_by_layout"].items())))

    # --- (3) ---
    print("\n  (3) CURVATURE AND CONTINUITY on the cells (2) opens")
    for tag, title in (("", "A. ramps only"), ("s_", "B. whole span")):
        lost = R[tag + "qp_open_nocurv"] - R[tag + "qp_open"]
        print(f"    {title}: geometry opened {R[tag+'qp_open_nocurv']}, curvature takes back "
              f"{lost} ({_pct(lost, max(R[tag+'qp_open_nocurv'], 1)):.1f} %), leaving "
              f"{R[tag+'qp_open']}")
        if R[tag + "qp_kappa"]:
            ka = np.array(R[tag + "qp_kappa"])
            kd = np.array(R[tag + "qp_kappa_add"])
            qk = np.array(R[tag + "qp_kink"])
            bk = np.array(R[tag + "base_kink"])
            print(f"      peak |kappa| of the published QP path: median {np.median(ka):.3f}, "
                  f"p90 {np.percentile(ka, 90):.3f}, max {np.max(ka):.3f}  (kappa_abs_max 5.5)")
            print(f"      peak |kappa - kappa_ref|: median {np.median(kd):.3f}, "
                  f"p90 {np.percentile(kd, 90):.3f}, max {np.max(kd):.3f}  (kappa_add_max 5.0)")
            print(f"      worst d' jump at the seams: QP median {np.median(qk):.3f} "
                  f"p90 {np.percentile(qk, 90):.3f} max {np.max(qk):.3f} | the node's own quintic "
                  f"on the SAME candidate median {np.median(bk):.3f} "
                  f"p90 {np.percentile(bk, 90):.3f}")
            # THE PRICE. Passing the curvature gate is not the same as being worth driving: the
            # maneuver's speed cap is sqrt(a_lat_max / peak|kappa|). sweep_static_feasibility
            # gates the GEOMETRIC half of it (mean sqrt(1/|kappa|) >= MIN_BEND) so the gate does
            # not move with the vehicle file; the speed is reported there and here, at the
            # planner's own a_lat_max. A ceiling bought at 1 m/s is a ceiling on paper.
            cap = np.sqrt(float(sw.load_params()["a_lat_max"]) / np.maximum(ka, 1e-3))
            print(f"      curvature-limited speed cap sqrt(a_lat/kappa): median {np.median(cap):.2f}"
                  f" m/s, p10 {np.percentile(cap, 10):.2f}, min {np.min(cap):.2f}  "
                  f"(the feasibility sweep gates the node's own published cells at "
                  f"{sw.MIN_SPEED_CAP:.2f} mean)")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="*", default=["ifac", "ifac_0807"])
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--convention-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    for m in a.map:
        check_convention(m)
    if a.convention_only:
        return 0
    allR = []
    for m in a.map:
        R = sweep(m, a.jobs, a.stride, a.limit)
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
