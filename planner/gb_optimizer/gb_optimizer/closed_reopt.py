"""Closed-track minimum-curvature re-optimization around static obstacles.

WHAT THIS REPLACES. static_reopt_core builds an avoidance line by GUESSING ITS SHAPE: a quintic
hump per obstacle, a reach chosen by search, a hold bridge, a cluster pre-merge, a side
unification, a span budget -- and sixteen refusal reasons when the guess does not fit. Here the
shape is not guessed. The corridor is narrowed around each obstacle and the line is the
minimum-curvature solution inside it, the same objective the offline raceline is built with. A
hold across two boxes is not a special case to be offered and gated; it is what minimum curvature
does when the corridor allows it.

WHY THE WHOLE LAP AND NOT A WINDOW. The windowed attempt (6be2564) died on a constraint that only
existed because it was windowed: a window has two seams, and pinning them to the clean line to
keep "everything outside unchanged" was a third constraint that emptied the feasible set. A closed
track has no seam and no window-formation policy, so both problems disappear at once. What is
given up is "byte-identical outside the obstacles" -- the whole line is re-solved. With no
obstacles at all the input is returned untouched, so the clean case still costs nothing.

WHY IT ITERATES, AND WHY IT KEEPS THE BEST ITERATE. opt_min_curv linearises curvature about the
reference line; one call is not the answer. tph.iqp_handler is the loop for that, and is not
usable here: its `while True` has no iteration cap, its tolerance sits under this track's
discretisation noise floor, and it re-interpolates every iteration, which breaks the alignment
between A, the normals and the width arrays. Measured on ifac, its error goes 0.911 -> 0.0185
(iter 4) -> 0.0116 (iter 7) -> 0.117 (iter 14) -> 2.699 (iter 29, diverged). So: our own loop,
with a hard cap, and the best iterate wins rather than the last.

THE GRID IS A CONDITIONING REQUIREMENT, NOT A KNOB. The published raceline is sampled at 0.1 m and
opt_min_curv's second derivative over segments that short is noise. Measured on ifac:
    0.30 m   diverges (clearances -0.06 .. -0.17, and "constraints are inconsistent" with a
             third box)
    0.50 m   converges in 2 iterations, peak |kappa| 0.98, clearance 0.312
    0.70 m   worse again (-0.12 .. -0.20)
0.50 m it is, and the answer is mapped back onto every 0.1 m station with a PERIODIC cubic --
linear interpolation puts a corner at every grid node and the fine line inherits it (measured peak
|kappa| 1.9-2.7 from the interpolation alone, on solutions that satisfied the bound).

Pure functions, no ROS. The caller supplies the corridor it already measures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import trajectory_planning_helpers as tph
from scipy.interpolate import CubicSpline

from .static_reopt_core import (  # noqa: F401  (re-exported for callers)
    Obstacle,
    ModulationParams,
    modulate_widths,
)


@dataclass
class ReoptParams:
    """Every knob this module has. Seven, and no more."""
    grid_step_m: float = 0.50     # QP grid: a conditioning requirement (see the module docstring)
    max_iters: int = 8            # hard cap -- iqp_handler has none and diverges without one
    iters_min: int = 2            # never accept before this many
    step_tol_m: float = 0.005     # converged once the iteration moves the line less than this
    kappa_bound: float = 1.5      # [1/m] the vehicle's steering limit (veh_dyn curvlim)
    w_veh: float = 0.30           # [m] width the QP reserves inside the corridor
    obs_margin: float = 0.17      # [m] 0.15 + 0.02 to absorb the 0.5 m grid's discretisation


@dataclass
class Report:
    ok: bool = False
    reason: str = ""
    iters: int = 0
    steps: List[float] = field(default_factory=list)      # max|a| per iteration
    accepted: List[bool] = field(default_factory=list)    # was each iterate a candidate?
    int_k2: float = float("nan")
    peak_kappa: float = float("nan")
    clearances: List[float] = field(default_factory=list)
    infeasible: List[int] = field(default_factory=list)   # obstacles with no drivable side
    sides: List[int] = field(default_factory=list)
    n_coarse: int = 0
    solve_ms: float = 0.0


# ======================================================================================
# geometry
# ======================================================================================
def _closed_el(pts: np.ndarray) -> np.ndarray:
    seg = np.roll(pts, -1, axis=0) - pts
    return np.hypot(seg[:, 0], seg[:, 1])


def _frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(psi, normvec) of a closed line. +normvec is the +w_tr_right side, which is the convention
    modulate_widths and opt_min_curv both use, so widths and normals live in one frame."""
    psi, _ = tph.calc_head_curv_num.calc_head_curv_num(
        path=pts, el_lengths=_closed_el(pts), is_closed=True)
    return psi, tph.calc_normal_vectors.calc_normal_vectors(psi)


def menger_closed(pts: np.ndarray) -> np.ndarray:
    """Signed curvature of a CLOSED polyline from circumscribed circles.

    Menger rather than a differentiated numeric heading: the latter amplifies the raceline's own
    micro-noise (~5-8x on this track), and every curvature VERDICT in this stack is taken on real
    geometry.
    """
    p = np.asarray(pts, float)
    a, b, c = np.roll(p, 1, axis=0), p, np.roll(p, -1, axis=0)
    v1, v2 = b - a, c - b
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    denom = np.hypot(*(b - a).T) * np.hypot(*(c - b).T) * np.hypot(*(c - a).T)
    return 2.0 * cross / np.maximum(denom, 1e-12)


def _int_k2(pts: np.ndarray) -> float:
    """Integral of kappa^2 around the lap -- the objective, measured on the result."""
    k = menger_closed(pts)
    el = _closed_el(pts)
    ds = 0.5 * (el + np.roll(el, 1))
    return float(np.sum(k * k * ds))


# ======================================================================================
# sides
# ======================================================================================
def select_sides(reftrack: np.ndarray,
                 obstacles: Sequence[Obstacle],
                 corridor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 params: Optional[ReoptParams] = None) -> List[int]:
    """Which side of each obstacle the line passes: +1 (toward +w_tr_right), -1, or 0.

    A box ON the raceline splits the corridor into two disconnected intervals, and a QP cannot
    choose between them -- so the choice is made here and the corridor handed to the QP is the
    connected one. 0 means neither side fits: that obstacle is dropped and named in
    Report.infeasible for the reactive layer. This never raises; an unavoidable box is a fact
    about the track, not an error in the request.

    Room is read from the corridor the caller measured (the eroded grid intersected with the
    waypoint bounds), because that is the space the car actually has -- the waypoint bounds alone
    overstate it, which is why a wall gate exists downstream at all.
    """
    p = params or ReoptParams()
    pts = np.asarray(reftrack, float)[:, :2]
    _psi, nv = _frame(pts)
    if corridor is not None:
        lo = np.asarray(corridor[0], float)
        hi = np.asarray(corridor[1], float)
    else:
        lo = -(np.asarray(reftrack, float)[:, 3] - 0.5 * p.w_veh)
        hi = np.asarray(reftrack, float)[:, 2] - 0.5 * p.w_veh
    out: List[int] = []
    for o in obstacles:
        j = int(np.argmin(np.hypot(pts[:, 0] - o.x, pts[:, 1] - o.y)))
        d_obs = float((np.array([o.x, o.y]) - pts[j]) @ nv[j])
        need = float(o.r) + p.obs_margin
        room_hi = (hi[j] if np.isfinite(hi[j]) else np.inf) - (d_obs + need)
        room_lo = (d_obs - need) - (lo[j] if np.isfinite(lo[j]) else -np.inf)
        if room_hi < 0.0 and room_lo < 0.0:
            out.append(0)
        else:
            out.append(1 if room_hi >= room_lo else -1)
    return out


# ======================================================================================
# the solve
# ======================================================================================
def reoptimize_closed(reftrack_fine: np.ndarray,
                      obstacles: Sequence[Obstacle],
                      corridor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      params: Optional[ReoptParams] = None
                      ) -> Tuple[np.ndarray, np.ndarray, Report]:
    """Minimum-curvature line around the whole lap, inside the obstacle-narrowed corridor.

    Returns (line_fine[N,2], offset_fine[N], report). With no obstacles the input points are
    returned as they are -- not copied, not re-splined, not resampled.
    """
    p = params or ReoptParams()
    t0 = time.perf_counter()
    full = np.asarray(reftrack_fine, float)
    pts_fine = full[:, :2]
    n = len(pts_fine)
    rep = Report()

    live = list(obstacles)
    rep.sides = select_sides(full, live, corridor, p) if live else []
    rep.infeasible = [i for i, sd in enumerate(rep.sides) if sd == 0]
    use = [o for i, o in enumerate(live) if rep.sides[i] != 0]
    if not use:
        rep.ok = True
        rep.reason = "no obstacle to avoid" if not live else "every obstacle is unavoidable"
        rep.solve_ms = (time.perf_counter() - t0) * 1e3
        return pts_fine, np.zeros(n), rep

    # --- the corridor the QP is allowed to use ------------------------------------------------
    # min(waypoint, eroded grid), both expressed as CAR-CENTRE limits and handed to opt_min_curv
    # as widths (it subtracts w_veh/2 itself). The waypoint bounds alone overstate the drivable
    # space -- measured on ifac, the grid is the binding one at p10 (0.15/0.20 m against
    # 0.25/0.26) -- and the wall gate downstream exists because of exactly that gap.
    ref_in = full.copy()
    if corridor is not None:
        lo = np.asarray(corridor[0], float)
        hi = np.asarray(corridor[1], float)
        okh = np.isfinite(hi)
        ref_in[okh, 2] = np.minimum(ref_in[okh, 2] - 0.5 * p.w_veh, hi[okh]) + 0.5 * p.w_veh
        okl = np.isfinite(lo)
        ref_in[okl, 3] = np.minimum(ref_in[okl, 3] - 0.5 * p.w_veh, -lo[okl]) + 0.5 * p.w_veh

    # --- coarse grid --------------------------------------------------------------------------
    el_fine = _closed_el(pts_fine)
    spacing = float(np.median(el_fine))
    stride = max(1, int(round(p.grid_step_m / max(spacing, 1e-6))))
    ci = np.arange(0, n, stride)
    rep.n_coarse = len(ci)
    coarse = ref_in[ci]
    mod, _mrep = modulate_widths(
        coarse, use, params=ModulationParams(obs_margin=p.obs_margin), recenter=False)
    w_r0, w_l0 = mod[:, 2].copy(), mod[:, 3].copy()

    clean_c = coarse[:, :2].copy()
    _psi_c, nv0_c = _frame(clean_c)
    s_fine = np.concatenate([[0.0], np.cumsum(el_fine)[:-1]])
    L = float(np.sum(el_fine))
    s_c = s_fine[ci]
    _psi_f, nv_fine = _frame(pts_fine)

    def upsample(off_c: np.ndarray) -> np.ndarray:
        """Coarse offsets -> every fine station, PERIODIC cubic (see the module docstring)."""
        sc = np.concatenate([s_c, [s_c[0] + L]])
        oc = np.concatenate([off_c, [off_c[0]]])
        return CubicSpline(sc, oc, bc_type="periodic")(s_fine)

    # --- the loop -----------------------------------------------------------------------------
    pts = clean_c.copy()
    best_off, best_int = None, float("inf")
    w2 = 0.5 * p.w_veh
    need_clear = p.obs_margin + w2

    for it in range(p.max_iters):
        off = np.einsum("ij,ij->i", pts - clean_c, nv0_c)
        w_r_it = w_r0 - off
        w_l_it = w_l0 + off
        if np.any((w_r_it - w2) + (w_l_it - w2) < 0.0):
            rep.reason = ("the corridor closed on the moving line: no lateral room left at "
                          f"{int(np.count_nonzero((w_r_it - w2) + (w_l_it - w2) < 0.0))} stations")
            break
        try:
            cx, cy, A, nv = tph.calc_splines.calc_splines(path=np.vstack([pts, pts[0]]))
            a = tph.opt_min_curv.opt_min_curv(
                reftrack=np.column_stack([pts, w_r_it, w_l_it]), normvectors=nv, A=A,
                kappa_bound=p.kappa_bound, w_veh=p.w_veh, closed=True)[0]
        except Exception as exc:                       # never raise: a hard track is a fact
            rep.reason = f"{type(exc).__name__}: {str(exc)[:80]}"
            break

        pts = pts + nv * a[:, None]
        step = float(np.max(np.abs(a)))
        rep.steps.append(step)
        rep.iters = it + 1

        off_fine = upsample(np.einsum("ij,ij->i", pts - clean_c, nv0_c))
        line = pts_fine + nv_fine * off_fine[:, None]
        peak = float(np.max(np.abs(menger_closed(line))))
        clears = [float(np.min(np.hypot(line[:, 0] - o.x, line[:, 1] - o.y)) - o.r) for o in use]
        good = peak <= p.kappa_bound + 1e-9 and all(c >= need_clear - 1e-9 for c in clears)
        rep.accepted.append(bool(good))
        if good:
            score = _int_k2(line)
            if score < best_int:
                best_int, best_off = score, off_fine.copy()
                rep.int_k2, rep.peak_kappa, rep.clearances = score, peak, clears
        if it + 1 >= p.iters_min and step < p.step_tol_m:
            break

    rep.solve_ms = (time.perf_counter() - t0) * 1e3
    if best_off is None:
        rep.ok = False
        if not rep.reason:
            rep.reason = (f"no iterate cleared every obstacle by {need_clear:.2f} m inside "
                          f"|kappa| <= {p.kappa_bound}")
        return pts_fine, np.zeros(n), rep
    rep.ok = True
    return pts_fine + nv_fine * best_off[:, None], best_off, rep
