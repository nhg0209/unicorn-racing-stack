"""Windowed minimum-curvature re-optimization around static obstacles.

WHAT THIS REPLACES. static_reopt_core builds an avoidance line by GUESSING ITS SHAPE: a quintic
hump per obstacle, a reach chosen by search, a hold bridge, a cluster pre-merge, a side
unification, a span budget -- and sixteen distinct refusal reasons when the guess does not fit.
Here the shape is not guessed. The corridor is narrowed around each obstacle and the line is the
minimum-curvature solution inside it, which is the same objective the offline raceline is built
with. A hold across two boxes is not a special case that has to be offered and gated; it is what
minimum curvature does when the corridor allows it.

WHY IT IS WINDOWED, AND WHY IT ITERATES. Both are measured constraints, not preferences:

  tph.iqp_handler is closed-track only (it builds refline_tmp_cl), its `while True` has no
  iteration cap, and on ifac its 0.01 rad/m tolerance sits under the discretisation noise floor,
  so it does not terminate: 0.911 -> 0.0185 (iter 4) -> 0.0116 (iter 7) -> 0.117 (iter 14) ->
  2.699 (iter 29, diverged). This module therefore keeps its own loop with BOTH a tolerance and a
  hard cap, and returns the BEST iterate rather than the last one.

  A single opt_min_curv call is not enough. On the ifac window 250..389 with two boxes it returns
  peak |kappa| 10.247 against a 1.5 bound, int_k2 36.65, where a hand-built hold profile over the
  same window scores int_k2 1.33 / peak 1.04. The objective is right; the linearisation needs
  iterating.

CONVENTIONS, all verified against the vendored tph rather than assumed:

  reftrack is [x, y, w_tr_right, w_tr_left], and +normvec is the +w_tr_right side. tph's
  calc_normal_vectors and this repo's centerline_frame agree on that, so widths from
  modulate_widths and normals from calc_splines live in the same frame.

  calc_splines on an OPEN path of n points returns n-1 splines, an n-1 row normvec and a
  4*(n-1) row A -- but opt_min_curv(closed=False) wants n normal vectors. The endpoint normal is
  appended from psi_e.

  fix_s / fix_e do NOT pin the endpoints to zero: they set dev_max_left[0] = dev_max_right[0] =
  0.05, a +-5 cm box. A window with too little padding will sit against that box and hand the
  clean line a 5 cm step at the seam, so the seam deviation is measured and reported on every
  solve (see WindowReport.seam_alpha and WindowParams.seam_alpha_max).

Pure functions, no ROS. The caller supplies the corridor it already measures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import trajectory_planning_helpers as tph
from scipy.interpolate import CubicSpline

from .static_reopt_core import (  # noqa: F401  (Obstacle/ModulationParams are re-exported)
    Obstacle,
    ModulationParams,
    modulate_widths,
)


# [m] station spacing the QP is BUILT ON. Not a tuning knob -- a conditioning requirement, and
# the shipped racecar_f110.ini says the same thing for the offline optimizer ("stepsize_reg": 0.2,
# "stepsize during opt."). The published raceline is sampled at 0.1 m, and opt_min_curv linearises
# curvature about the reference: over 0.1 m segments that second derivative is noise, and the loop
# does not converge. Measured on the ifac window around station 275, same problem, same padding:
#     QP grid   curv_err by iteration            peak |kappa|
#      0.10 m   9.6, 50.6, 10.0, 494.8, 1029.8      26.9      (diverges)
#      0.50 m   0.475, 0.389, 0.461, 0.272, 0.802    1.17     (usable)
# The QP's answer is interpolated back onto every 0.1 m station, so nothing downstream sees the
# coarse grid.
_QP_STEP_M = 0.5


@dataclass
class WindowParams:
    """Every knob this module has. Nothing else is tunable on purpose."""
    pad_m: float = 3.0            # clean raceline kept either side of an obstacle, inside the window
    merge_gap_m: float = 6.0      # obstacles closer than this share one window
    max_iters: int = 8            # hard cap -- iqp_handler has none and diverges without one
    iters_min: int = 3            # never accept before this many, the error is not monotone early
    curv_tol: float = 0.02        # [rad/m] 0.01 is under ifac's discretisation floor: never met
    kappa_bound: float = 1.5      # [1/m] the vehicle's own steering limit (veh_dyn curvlim)
    w_veh: float = 0.30           # [m] vehicle width the QP reserves inside the corridor
    seam_alpha_max: float = 0.01  # [m] |alpha| allowed at a window end before the seam is a step


@dataclass
class WindowSpec:
    """One contiguous stretch of stations to re-solve, and the obstacles inside it."""
    idx: np.ndarray               # station indices in travel order (may wrap through 0)
    obs: List[int] = field(default_factory=list)   # indices into the caller's obstacle list
    s0: float = 0.0
    s1: float = 0.0


@dataclass
class WindowReport:
    ok: bool = False
    reason: str = ""
    iters: int = 0
    curv_err: List[float] = field(default_factory=list)
    int_k2: float = float("nan")
    peak_kappa: float = float("nan")
    seam_alpha: float = float("nan")
    clearances: List[float] = field(default_factory=list)
    solve_ms: float = 0.0
    n_stations: int = 0
    span_m: float = 0.0


@dataclass
class Report:
    n_windows: int = 0
    windows: List[WindowReport] = field(default_factory=list)
    infeasible: List[int] = field(default_factory=list)   # obstacle indices with no drivable side
    sides: List[int] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return all(w.ok for w in self.windows)


# ======================================================================================
# geometry helpers
# ======================================================================================
def _closed_el(pts: np.ndarray) -> np.ndarray:
    seg = np.roll(pts, -1, axis=0) - pts
    return np.hypot(seg[:, 0], seg[:, 1])


def _station_s(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Cumulative arc length of a CLOSED station list, and the lap length."""
    el = _closed_el(pts)
    return np.concatenate([[0.0], np.cumsum(el)[:-1]]), float(np.sum(el))


def _frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(psi, normvec) of a closed line, in the same convention as the widths."""
    psi, _ = tph.calc_head_curv_num.calc_head_curv_num(
        path=pts, el_lengths=_closed_el(pts), is_closed=True)
    return psi, tph.calc_normal_vectors.calc_normal_vectors(psi)


def menger_open(pts: np.ndarray) -> np.ndarray:
    """Signed curvature of an OPEN polyline from circumscribed circles.

    Menger rather than a differentiated numeric heading: the latter amplifies the raceline's own
    micro-noise (measured ~5-8x on this track), and every curvature VERDICT in this stack is taken
    on real geometry. Endpoints inherit their neighbour, having no stencil of their own.
    """
    p = np.asarray(pts, float)
    n = len(p)
    if n < 3:
        return np.zeros(n)
    a, b, c = p[:-2], p[1:-1], p[2:]
    v1, v2 = b - a, c - b
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    denom = np.hypot(*(b - a).T) * np.hypot(*(c - b).T) * np.hypot(*(c - a).T)
    k = 2.0 * cross / np.maximum(denom, 1e-12)
    return np.concatenate([[k[0]], k, [k[-1]]])


def _int_k2(pts: np.ndarray) -> float:
    """Integral of kappa^2 along an open polyline -- the objective, measured on the result."""
    k = menger_open(pts)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    ds = np.concatenate([[seg[0]], 0.5 * (seg[:-1] + seg[1:]), [seg[-1]]])
    return float(np.sum(k * k * ds))


# ======================================================================================
# sides
# ======================================================================================
def select_sides(reftrack: np.ndarray,
                 obstacles: Sequence[Obstacle],
                 corridor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 mod_params: Optional[ModulationParams] = None,
                 w_veh: float = 0.30) -> List[int]:
    """Which side of each obstacle the line will pass: +1 (toward +w_tr_right), -1, or 0.

    0 means neither side has room for the car plus its margin. Such an obstacle is dropped from
    the windows and named in Report.infeasible -- the reactive layer owns it. This never raises:
    an unavoidable box is a fact about the track, not an error in the request.

    Room is read from the corridor the caller measured (the eroded grid intersected with the
    waypoint bounds) so the answer matches what the car can actually drive, not what d_left /
    d_right claim.

    Note there is no side UNIFICATION step. Once a side is chosen the corridor is a connected
    interval at every station and the QP solves it directly; two obstacles in one window may take
    different sides and the corridor expresses that without any pre-pass.
    """
    mp = mod_params or ModulationParams()
    pts = np.asarray(reftrack, float)[:, :2]
    _psi, nv = _frame(pts)
    if corridor is not None:
        lo = np.asarray(corridor[0], float)
        hi = np.asarray(corridor[1], float)
    else:
        lo = -np.asarray(reftrack, float)[:, 3]
        hi = np.asarray(reftrack, float)[:, 2]
    sides: List[int] = []
    for o in obstacles:
        j = int(np.argmin(np.hypot(pts[:, 0] - o.x, pts[:, 1] - o.y)))
        d_obs = float((np.array([o.x, o.y]) - pts[j]) @ nv[j])
        need = float(o.r) + float(mp.obs_margin) + 0.5 * float(w_veh)
        room_hi = (hi[j] if np.isfinite(hi[j]) else np.inf) - (d_obs + need)
        room_lo = (d_obs - need) - (lo[j] if np.isfinite(lo[j]) else -np.inf)
        if room_hi < 0.0 and room_lo < 0.0:
            sides.append(0)
        else:
            sides.append(1 if room_hi >= room_lo else -1)
    return sides


# ======================================================================================
# windows
# ======================================================================================
def form_windows(reftrack: np.ndarray,
                 obstacles: Sequence[Obstacle],
                 sides: Sequence[int],
                 params: Optional[WindowParams] = None) -> List[WindowSpec]:
    """One window per obstacle group, merged where they are closer than merge_gap_m.

    This one function does the job of the hump pipeline's hold bridge, cluster pre-merge, side
    unification and span budget: obstacles that interact end up in the same optimization problem,
    and what happens between them is decided by minimising curvature inside the corridor rather
    than by a rule about what shape to attempt.
    """
    p = params or WindowParams()
    pts = np.asarray(reftrack, float)[:, :2]
    s, L = _station_s(pts)
    n = len(pts)

    live = [(i, o) for i, o in enumerate(obstacles) if sides[i] != 0]
    if not live:
        return []

    # obstacle station and its own [s-pad, s+pad]
    raw = []
    for i, o in live:
        j = int(np.argmin(np.hypot(pts[:, 0] - o.x, pts[:, 1] - o.y)))
        raw.append((float(s[j]), i))
    raw.sort()

    # merge in travel order, wrap-aware: walk the sorted stations and start a new group whenever
    # the gap to the previous obstacle exceeds merge_gap_m
    groups: List[List[Tuple[float, int]]] = [[raw[0]]]
    for s_o, i in raw[1:]:
        if (s_o - groups[-1][-1][0]) <= p.merge_gap_m:
            groups[-1].append((s_o, i))
        else:
            groups.append([(s_o, i)])
    # the seam: the last group may join the first one around the lap
    if len(groups) > 1:
        gap_wrap = (groups[0][0][0] - groups[-1][-1][0]) % L
        if gap_wrap <= p.merge_gap_m:
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    out: List[WindowSpec] = []
    for g in groups:
        s_start = (g[0][0] - p.pad_m) % L
        s_end = (g[-1][0] + p.pad_m) % L
        i0 = int(np.argmin(np.abs(((s - s_start + L / 2.0) % L) - L / 2.0)))
        i1 = int(np.argmin(np.abs(((s - s_end + L / 2.0) % L) - L / 2.0)))
        count = (i1 - i0) % n + 1
        if count < 5:                     # too few stations to spline: widen to the minimum
            count = 5
        idx = (i0 + np.arange(count)) % n
        out.append(WindowSpec(idx=idx, obs=[i for _s, i in g],
                              s0=float(s[i0]), s1=float(s[i1])))
    return out


# ======================================================================================
# the solve
# ======================================================================================
def solve_window(reftrack: np.ndarray,
                 win: WindowSpec,
                 obstacles: Sequence[Obstacle],
                 corridor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 params: Optional[WindowParams] = None,
                 mod_params: Optional[ModulationParams] = None
                 ) -> Tuple[Optional[np.ndarray], WindowReport]:
    """Minimum-curvature line inside the obstacle-narrowed corridor, over one window.

    Returns (alpha over the window's stations, report). alpha is measured along the CLEAN line's
    normals so the caller can lay it on the clean geometry; None means no iterate was acceptable
    and the report says why.

    The loop is the whole point of this module, so its shape is deliberate:

      * BOTH a tolerance and a cap (iqp_handler has only the tolerance, and diverges past it).
      * The best iterate wins, not the last: the error is not monotone, and iteration 4 was the
        good one in the measured trace while iteration 29 was garbage.
      * "Best" means lowest int_k2 AMONG THE ACCEPTABLE ones -- every obstacle cleared by
        obs_margin and peak |kappa| inside the bound. An iterate that lowers the objective by
        breaking the clearance is not a candidate at all.
    """
    p = params or WindowParams()
    mp = mod_params or ModulationParams()
    t0 = time.perf_counter()
    rep = WindowReport(n_stations=len(win.idx))

    full = np.asarray(reftrack, float)
    pts_all = full[:, :2]
    _psi_all, nv_all = _frame(pts_all)

    # Corridor for THIS window's obstacles, from the function that already documents this exact
    # use: recenter=False keeps the reference line as the clean raceline and expresses the
    # exclusion purely in the widths (one-sided, possibly negative -- the QP solves that).
    ref_in = full.copy()
    if corridor is not None:
        lo = np.asarray(corridor[0], float)
        hi = np.asarray(corridor[1], float)
        ok = np.isfinite(hi)
        ref_in[ok, 2] = np.minimum(ref_in[ok, 2], hi[ok])
        ok = np.isfinite(lo)
        ref_in[ok, 3] = np.minimum(ref_in[ok, 3], -lo[ok])
    win_obs = [obstacles[i] for i in win.obs]
    mod, _mrep = modulate_widths(ref_in, win_obs, params=mp, recenter=False)

    idx = win.idx
    pts0 = pts_all[idx].copy()
    w_r0 = mod[idx, 2].copy()
    w_l0 = mod[idx, 3].copy()
    nv0 = nv_all[idx]
    rep.span_m = float(np.sum(np.hypot(*np.diff(pts0, axis=0).T)))

    # Headings at the seams come from the CLEAN line, so the window rejoins it without a kink.
    psi_s = float(_psi_all[idx[0]])
    psi_e = float(_psi_all[idx[-1]])

    # THE QP GRID. Sub-sample to _QP_STEP_M and give each QP station the TIGHTEST corridor over
    # the fine stations it stands for -- sampling the width at one station instead lets the QP
    # miss the narrowest point between two samples, which showed up as "constraints are
    # inconsistent" on some offsets and a corridor violation on others.
    ds = float(np.median(np.hypot(*np.diff(pts0, axis=0).T)))
    stride = max(1, int(round(_QP_STEP_M / max(ds, 1e-6))))
    qi = np.arange(0, len(idx), stride)
    if qi[-1] != len(idx) - 1:
        qi = np.append(qi, len(idx) - 1)
    edges = np.concatenate([[0], (qi[:-1] + qi[1:] + 1) // 2, [len(idx)]])
    w_r_q = np.array([np.min(w_r0[edges[k]:edges[k + 1]]) for k in range(len(qi))])
    w_l_q = np.array([np.min(w_l0[edges[k]:edges[k + 1]]) for k in range(len(qi))])
    q_arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts0, axis=0).T))])

    pts = pts0[qi].copy()
    alpha_tot = np.zeros(len(qi))
    best_alpha, best_int = None, float("inf")

    for it in range(p.max_iters):
        # THE SEAM IS BOUNDED ON THE TOTAL, not per iteration. fix_s/fix_e overwrite the end
        # stations' allowance with a fixed +-5 cm box EVERY iteration, so eight iterations can walk
        # the seam 0.4 m off the clean line -- measured at -0.227 / -0.268 m after six. Instead the
        # ends carry an explicit allowance that shrinks by what they have already used.
        # KEEP THE ACCUMULATOR INSIDE THE CORRIDOR. Each QP obeys the box it was given, but the
        # boxes are rebuilt from a linearisation about a line that has since moved, so alpha_tot
        # can drift past the corridor and hand the next iteration an empty feasible set
        # ("constraints are inconsistent"). Clamping is the honest repair: it is the same
        # constraint, applied to the total instead of the step.
        alpha_tot = np.clip(alpha_tot, -(w_l_q - 0.5 * p.w_veh), w_r_q - 0.5 * p.w_veh)
        w_r_it = w_r_q - alpha_tot
        w_l_it = w_l_q + alpha_tot
        for e in (0, -1):
            left = max(p.seam_alpha_max - abs(alpha_tot[e]), 0.0)
            w_r_it[e] = 0.5 * p.w_veh + left
            w_l_it[e] = 0.5 * p.w_veh + left
        try:
            cx, cy, A, nv = tph.calc_splines.calc_splines(
                path=pts, psi_s=psi_s, psi_e=psi_e)
            nv_full = np.vstack([nv, tph.calc_normal_vectors.calc_normal_vectors(
                np.array([psi_e]))])
            a, curv_err = tph.opt_min_curv.opt_min_curv(
                reftrack=np.column_stack([pts, w_r_it, w_l_it]), normvectors=nv_full, A=A,
                kappa_bound=p.kappa_bound, w_veh=p.w_veh, closed=False,
                psi_s=psi_s, psi_e=psi_e, fix_s=False, fix_e=False)
        except Exception as exc:                     # never raise: a hard window is a fact
            rep.reason = f"{type(exc).__name__}: {str(exc)[:80]}"
            break

        pts = pts + nv_full * a[:, None]
        alpha_tot = alpha_tot + a
        rep.curv_err.append(float(curv_err))
        rep.iters = it + 1

        # judge THIS iterate on the FINE geometry it implies, not on the QP grid
        # CUBIC, not linear. A piecewise-linear alpha puts a corner at every QP node, and the
        # fine line inherits it: measured peak |kappa| 1.9-2.7 against a 1.5 bound purely from the
        # interpolation, on a QP solution that satisfied the bound. The QP's own answer is a
        # spline through its points, so the mapping back has to be one too.
        a_fine = CubicSpline(q_arc[qi], alpha_tot, bc_type="clamped")(q_arc)
        pts_fine = pts0 + nv0 * a_fine[:, None]
        kap = np.abs(menger_open(pts_fine))
        peak = float(np.max(kap))
        clears = [float(np.min(np.hypot(pts_fine[:, 0] - obstacles[i].x,
                                        pts_fine[:, 1] - obstacles[i].y)) - obstacles[i].r)
                  for i in win.obs]
        seam = float(max(abs(a_fine[0]), abs(a_fine[-1])))
        acceptable = (peak <= p.kappa_bound + 1e-9
                      and all(c >= mp.obs_margin - 1e-9 for c in clears)
                      and seam <= p.seam_alpha_max + 1e-9)
        if acceptable:
            score = _int_k2(pts_fine)
            if score < best_int:
                best_int, best_alpha = score, a_fine.copy()
                rep.int_k2, rep.peak_kappa = score, peak
                rep.clearances = clears
                rep.seam_alpha = seam
        if it + 1 >= p.iters_min and float(curv_err) <= p.curv_tol:
            break

    rep.solve_ms = (time.perf_counter() - t0) * 1e3
    if best_alpha is None:
        rep.ok = False
        if not rep.reason:
            rep.reason = ("no iterate cleared every obstacle by obs_margin inside the curvature "
                          "bound and the seam budget")
        return None, rep
    rep.ok = True
    return best_alpha, rep


# ======================================================================================
# top level
# ======================================================================================
def reoptimize_windowed(reftrack: np.ndarray,
                        obstacles: Sequence[Obstacle],
                        corridor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        params: Optional[WindowParams] = None,
                        mod_params: Optional[ModulationParams] = None
                        ) -> Tuple[np.ndarray, np.ndarray, Report]:
    """Re-optimize only where obstacles are, and leave the rest of the lap exactly alone.

    Returns (line[N,2], alpha[N], Report). Stations outside every window are byte-identical to
    the input -- not re-splined, not re-sampled, not copied through an interpolator. With no
    obstacles the input array is returned as-is.
    """
    p = params or WindowParams()
    mp = mod_params or ModulationParams()
    t0 = time.perf_counter()
    full = np.asarray(reftrack, float)
    pts_all = full[:, :2]
    rep = Report()

    sides = select_sides(full, obstacles, corridor, mp, p.w_veh)
    rep.sides = sides
    rep.infeasible = [i for i, s in enumerate(sides) if s == 0]

    wins = form_windows(full, obstacles, sides, p)
    rep.n_windows = len(wins)
    if not wins:
        rep.total_ms = (time.perf_counter() - t0) * 1e3
        return pts_all, np.zeros(len(pts_all)), rep

    line = pts_all.copy()
    alpha = np.zeros(len(pts_all))
    _psi_all, nv_all = _frame(pts_all)
    for w in wins:
        a, wrep = solve_window(full, w, obstacles, corridor, p, mp)
        rep.windows.append(wrep)
        if a is None:
            continue                                  # window left on the clean line
        line[w.idx] = pts_all[w.idx] + nv_all[w.idx] * a[:, None]
        alpha[w.idx] = a
    rep.total_ms = (time.perf_counter() - t0) * 1e3
    return line, alpha, rep
