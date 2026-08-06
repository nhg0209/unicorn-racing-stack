"""Static-obstacle avoidance as one convex QP over the lateral offset from the raceline.

THE PROBLEM, IN FULL:

    minimize   || D2 d ||^2  +  w || d ||^2
    subject to lo(s) <= d(s) <= hi(s)

d is the lateral offset from the CLEAN raceline on a fixed station grid, D2 the periodic second
difference over arc length, and [lo, hi] the track corridor intersected with each obstacle's
keep-out. It is exactly quadratic in d and box-constrained, so it has one solution and there is
nothing to iterate.

WHY THIS AND NOT MINIMUM CURVATURE OF THE LINE. Minimising the lap's own curvature re-solves the
whole raceline instead of laying an avoidance on it: with no obstacles at all that objective still
moves the line, taking ifac's int_k2 from 7.487 to 4.647. Everything that went wrong with the
previous two attempts follows from that -- there was no reason to hold an offset (the raceline is
not the reference), the corridor left the objective nearly flat so the iteration oscillated, and a
box in one corner moved the global optimum everywhere. Here the clean line IS the reference: with
no obstacle the answer is d = 0 exactly, and every metre of offset has to be paid for.

  w is the hold-vs-return knob, and the only shape knob there is. Measured on ifac with two boxes
  8.5 m apart on the straight: w = 0.00 holds 0.416 m between them, w = 0.01 holds 0.404, w = 0.05
  holds 0.227. Low w keeps the offset across the gap; high w brings the line home between the
  boxes. reach, span, ramp curvature, hold bridges and cluster merges are all gone -- their job is
  done by one number.

THE KEEP-OUT IS ANALYTIC, NOT WINDOWED BY STATION. A station-window exclusion (modulate_widths'
`obs.r + long_taper` band) depends on where the grid happens to fall: at 0.30 m spacing the same
obstacle produced a clearance of -0.14 m where 0.10 and 0.50 m gave +0.300. Here each station asks
the obstacle directly how much lateral room it takes at that longitudinal distance,
sqrt(max(0, R^2 - dt^2)) with R = r + obs_margin + w_veh/2, which is grid-independent and exact.

Solved with quadprog -- the same dual method opt_min_curv uses internally, present in both the
conda env and the ROS runtime. scipy.optimize.lsq_linear solves the same problem in 0.1-9.3 s
against quadprog's 0.5 ms.

Pure functions, no ROS. The caller supplies the corridor it already measures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import quadprog
import trajectory_planning_helpers as tph
from scipy.interpolate import CubicSpline

from .static_reopt_core import Obstacle  # noqa: F401  (re-exported for callers)


@dataclass
class ReoptParams:
    """Five numbers. There is no sixth."""
    grid_step_m: float = 0.50        # QP station spacing; the answer is mapped back to every station
    dev_weight: float = 0.01         # w: the hold-vs-return knob (see the module docstring)
    obs_margin: float = 0.15         # [m] lateral clearance owed to an obstacle, beyond its radius
    w_veh: float = 0.30              # [m] vehicle width, reserved on both sides
    kappa_report_only: float = 1.5   # [1/m] REPORTED against, never used to reject or to tune


@dataclass
class Report:
    ok: bool = False
    reason: str = ""
    solve_ms: float = 0.0
    n_coarse: int = 0
    hold: float = float("nan")            # min |d| between the first two obstacles, when there are two
    clearances: List[float] = field(default_factory=list)
    peak_kappa: float = float("nan")      # Menger, on the published fine line
    peak_station: int = -1
    clean_peak_kappa: float = float("nan")
    clean_peak_station: int = -1
    peak_near_obstacle: bool = False      # is the worst curvature ours, or the raceline's own?
    infeasible: List[int] = field(default_factory=list)
    sides: List[int] = field(default_factory=list)
    max_offset: float = 0.0


# ======================================================================================
# geometry
# ======================================================================================
def _closed_el(pts: np.ndarray) -> np.ndarray:
    seg = np.roll(pts, -1, axis=0) - pts
    return np.hypot(seg[:, 0], seg[:, 1])


def _frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(psi, normvec, tangent) of a closed line. +normvec is the +w_tr_right side, matching the
    reftrack's own width columns."""
    el = _closed_el(pts)
    psi, _ = tph.calc_head_curv_num.calc_head_curv_num(path=pts, el_lengths=el, is_closed=True)
    nv = tph.calc_normal_vectors.calc_normal_vectors(psi)
    tan = np.roll(pts, -1, axis=0) - pts
    tan = tan / np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
    return psi, nv, tan


def menger_closed(pts: np.ndarray) -> np.ndarray:
    """Signed curvature of a CLOSED polyline from circumscribed circles. Menger rather than a
    differentiated numeric heading, which amplifies this raceline's own micro-noise ~5-8x."""
    p = np.asarray(pts, float)
    a, b, c = np.roll(p, 1, axis=0), p, np.roll(p, -1, axis=0)
    v1, v2 = b - a, c - b
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    denom = np.hypot(*(b - a).T) * np.hypot(*(c - b).T) * np.hypot(*(c - a).T)
    return 2.0 * cross / np.maximum(denom, 1e-12)


def int_k2(pts: np.ndarray) -> float:
    k = menger_closed(pts)
    el = _closed_el(pts)
    return float(np.sum(k * k * 0.5 * (el + np.roll(el, 1))))


# ======================================================================================
# corridor + keep-out
# ======================================================================================
def obstacle_bounds(pts: np.ndarray,
                    nv: np.ndarray,
                    tan: np.ndarray,
                    obstacles: Sequence[Obstacle],
                    sides: Sequence[int],
                    lo: np.ndarray,
                    hi: np.ndarray,
                    params: ReoptParams) -> Tuple[np.ndarray, np.ndarray]:
    """Narrow [lo, hi] by each obstacle's keep-out, analytically and per station.

    At longitudinal distance dt from the obstacle centre, a disc of radius R blocks a lateral band
    of half-width sqrt(max(0, R^2 - dt^2)) about the obstacle's own offset. R carries the radius,
    the margin owed to it and half the car, so what is returned is a CAR-CENTRE bound.

    Which side the line passes on has already been decided (see select_sides): passing on +normvec
    means the offset must stay above the band's upper edge, so only `lo` moves, and the corridor
    stays a single interval -- which is what makes the problem a box-constrained QP at all.
    """
    lo = lo.copy()
    hi = hi.copy()
    for o, sd in zip(obstacles, sides):
        if sd == 0:
            continue
        rel = np.array([o.x, o.y])[None, :] - pts
        du = np.einsum("ij,ij->i", rel, nv)
        dt = np.einsum("ij,ij->i", rel, tan)
        R = float(o.r) + params.obs_margin + 0.5 * params.w_veh
        h = np.sqrt(np.maximum(R * R - dt * dt, 0.0))
        touched = h > 0.0
        if sd > 0:
            lo[touched] = np.maximum(lo[touched], du[touched] + h[touched])
        else:
            hi[touched] = np.minimum(hi[touched], du[touched] - h[touched])
    return lo, hi


def select_sides(pts: np.ndarray,
                 nv: np.ndarray,
                 obstacles: Sequence[Obstacle],
                 lo: np.ndarray,
                 hi: np.ndarray,
                 params: ReoptParams) -> List[int]:
    """Which side of each obstacle the line passes: +1 (toward +normvec), -1, or 0 for neither.

    A box on the raceline splits the corridor into two disconnected intervals and a box-constrained
    QP cannot choose between them, so the choice is made here and the QP is handed the connected
    one. 0 means the car does not fit either side; that obstacle is dropped and named in
    Report.infeasible for the reactive layer. Never raises -- an unavoidable box is a fact about
    the track.
    """
    out: List[int] = []
    for o in obstacles:
        j = int(np.argmin(np.hypot(pts[:, 0] - o.x, pts[:, 1] - o.y)))
        du = float((np.array([o.x, o.y]) - pts[j]) @ nv[j])
        R = float(o.r) + params.obs_margin + 0.5 * params.w_veh
        room_hi = (hi[j] if np.isfinite(hi[j]) else np.inf) - (du + R)
        room_lo = (du - R) - (lo[j] if np.isfinite(lo[j]) else -np.inf)
        out.append(0 if (room_hi < 0.0 and room_lo < 0.0) else (1 if room_hi >= room_lo else -1))
    return out


# ======================================================================================
# the solve
# ======================================================================================
def _second_difference(n: int, ds: float) -> np.ndarray:
    """Periodic second difference over a uniform station grid."""
    D2 = (np.diag(-2.0 * np.ones(n))
          + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1))
    D2[0, -1] = 1.0
    D2[-1, 0] = 1.0
    return D2 / (ds * ds)


def solve_offsets(lo: np.ndarray, hi: np.ndarray, ds: float, w: float) -> Optional[np.ndarray]:
    """The QP itself: minimize ||D2 d||^2 + w||d||^2 subject to lo <= d <= hi.

    quadprog minimises 1/2 x'Gx - a'x subject to C'x >= b, so the two box sides go in as
    C = [I, -I] and b = [lo, -hi]. The 1e-9 ridge is for strict convexity only -- D2'D2 is singular
    (a constant offset has no second difference) and w alone may be zero.
    """
    n = len(lo)
    D2 = _second_difference(n, ds)
    G = 2.0 * (D2.T @ D2 + w * np.eye(n)) + 1e-9 * np.eye(n)
    C = np.hstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([lo, -hi])
    try:
        return quadprog.solve_qp(G, np.zeros(n), C, b, 0)[0]
    except Exception:
        return None


def reoptimize_closed(reftrack_fine: np.ndarray,
                      obstacles: Sequence[Obstacle],
                      corridor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      params: Optional[ReoptParams] = None
                      ) -> Tuple[np.ndarray, np.ndarray, Report]:
    """Lay a minimum-bending avoidance offset on the clean raceline, around the whole lap.

    Returns (line_fine[N,2], offset_fine[N], report). With no obstacle to avoid the input points
    are returned as they are -- not copied, not re-splined, not resampled.
    """
    p = params or ReoptParams()
    t0 = time.perf_counter()
    full = np.asarray(reftrack_fine, float)
    pts_fine = full[:, :2]
    n = len(pts_fine)
    rep = Report()

    # --- the corridor, as CAR-CENTRE limits ---------------------------------------------------
    # min(waypoint, eroded grid). The waypoint bounds alone overstate the drivable space -- on ifac
    # the grid is the binding one at p10 (0.15/0.20 m against 0.25/0.26) -- and the wall gate
    # downstream exists because of that gap. The kernel-7 erosion already reserves half a car, so
    # only the waypoint side has w_veh/2 taken off it.
    hi_f = full[:, 2] - 0.5 * p.w_veh
    lo_f = -(full[:, 3] - 0.5 * p.w_veh)
    if corridor is not None:
        g_lo = np.asarray(corridor[0], float)
        g_hi = np.asarray(corridor[1], float)
        ok = np.isfinite(g_hi)
        hi_f[ok] = np.minimum(hi_f[ok], g_hi[ok])
        ok = np.isfinite(g_lo)
        lo_f[ok] = np.maximum(lo_f[ok], g_lo[ok])

    _psi_f, nv_fine, _tan_f = _frame(pts_fine)
    rep.sides = select_sides(pts_fine, nv_fine, obstacles, lo_f, hi_f, p) if len(obstacles) else []
    rep.infeasible = [i for i, sd in enumerate(rep.sides) if sd == 0]
    use = [o for i, o in enumerate(obstacles) if rep.sides[i] != 0]
    use_sides = [sd for sd in rep.sides if sd != 0]

    k_clean = np.abs(menger_closed(pts_fine))
    rep.clean_peak_kappa = float(np.max(k_clean))
    rep.clean_peak_station = int(np.argmax(k_clean))
    if not use:
        rep.ok = True
        rep.reason = "no obstacle to avoid" if not len(obstacles) else "every obstacle is unavoidable"
        rep.peak_kappa, rep.peak_station = rep.clean_peak_kappa, rep.clean_peak_station
        rep.solve_ms = (time.perf_counter() - t0) * 1e3
        return pts_fine, np.zeros(n), rep

    # --- coarse grid --------------------------------------------------------------------------
    el_fine = _closed_el(pts_fine)
    stride = max(1, int(round(p.grid_step_m / max(float(np.median(el_fine)), 1e-6))))
    ci = np.arange(0, n, stride)
    rep.n_coarse = len(ci)
    pts_c = pts_fine[ci]
    _psi_c, nv_c, tan_c = _frame(pts_c)
    lo_c, hi_c = obstacle_bounds(pts_c, nv_c, tan_c, use, use_sides, lo_f[ci], hi_f[ci], p)

    empty = lo_c > hi_c
    if np.any(empty):
        # THE WALL WINS AND THE OBSTACLE MARGIN GIVES. Where the corridor and the keep-out do not
        # overlap the car cannot have both, and driving into a wall is worse than passing a box
        # closer than intended -- the same degradation modulate_widths documents ("hugged against
        # the wall furthest from the obstacle, and the station is flagged infeasible"). The
        # corridor bound is kept, the keep-out is relaxed to meet it, and the clearance that comes
        # out is reported so the caller can see what it actually got.
        n_bad = int(np.count_nonzero(empty))
        wall_is_lo = lo_c[empty] > hi_c[empty]
        hi_c[empty] = np.where(wall_is_lo, lo_c[empty] + 1e-6, hi_c[empty])
        lo_c[empty] = np.minimum(lo_c[empty], hi_c[empty] - 1e-6)
        rep.reason = (f"{n_bad} station(s) could not have both the corridor and the keep-out; "
                      f"the corridor won and the clearance below is what was achievable")

    ds = float(np.sum(el_fine)) / len(ci)
    d_c = solve_offsets(lo_c, hi_c, ds, p.dev_weight)
    if d_c is None:
        rep.ok = False
        rep.reason = rep.reason or "quadprog could not solve the box-constrained QP"
        rep.solve_ms = (time.perf_counter() - t0) * 1e3
        return pts_fine, np.zeros(n), rep

    # --- back to every station ----------------------------------------------------------------
    # PERIODIC CUBIC. A linear map puts a corner at every coarse node and the fine line inherits
    # it: measured peak |kappa| 1.9-2.7 from the interpolation alone, on offsets that were smooth.
    s_fine = np.concatenate([[0.0], np.cumsum(el_fine)[:-1]])
    L = float(np.sum(el_fine))
    d_fine = CubicSpline(np.concatenate([s_fine[ci], [s_fine[ci][0] + L]]),
                         np.concatenate([d_c, [d_c[0]]]), bc_type="periodic")(s_fine)
    line = pts_fine + nv_fine * d_fine[:, None]

    k_new = np.abs(menger_closed(line))
    rep.peak_kappa = float(np.max(k_new))
    rep.peak_station = int(np.argmax(k_new))
    rep.max_offset = float(np.max(np.abs(d_fine)))
    rep.clearances = [float(np.min(np.hypot(line[:, 0] - o.x, line[:, 1] - o.y)) - o.r)
                      for o in use]
    # is the worst curvature ours or the raceline's own? "Ours" = within a keep-out radius of a box
    ds_peak = np.min([np.hypot(line[rep.peak_station, 0] - o.x,
                               line[rep.peak_station, 1] - o.y) for o in use])
    rep.peak_near_obstacle = bool(ds_peak < 2.0)
    if len(use) >= 2:
        i0 = int(np.argmin(np.hypot(pts_fine[:, 0] - use[0].x, pts_fine[:, 1] - use[0].y)))
        i1 = int(np.argmin(np.hypot(pts_fine[:, 0] - use[1].x, pts_fine[:, 1] - use[1].y)))
        span = (np.arange(i0, i1 + 1) if i1 >= i0 else np.arange(i0, i1 + n + 1)) % n
        if len(span) > 10:
            rep.hold = float(np.min(np.abs(d_fine[span[5:-5]])))
    rep.ok = True
    rep.solve_ms = (time.perf_counter() - t0) * 1e3
    return line, d_fine, rep
