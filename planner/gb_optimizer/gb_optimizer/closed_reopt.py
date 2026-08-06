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

  w is the hold-vs-return knob and the only shape knob there is: it trades the hold between two
  boxes against how quickly the line comes home away from them. Measured on ifac, two boxes 8.5 m
  apart on the straight (hold / median |d| more than 2.5 m from either box):

      w  0.000  0.005  0.010  0.020  0.050  0.100
   hold  0.504  0.499  0.493  0.461  0.306  0.177
  far|d| 0.190  0.094  0.055  0.038  0.024  0.017

  peak |kappa| over that whole sweep is 1.427-1.435 against a clean line that is already 1.448, so
  curvature does not choose w -- it is 0.02 because that is the lowest peak among the w that still
  hold 0.40, and it also leaves the least offset lying around the rest of the lap. reach, span,
  ramp curvature, hold bridges and cluster merges are all gone; their job is done by this number.

THE KEEP-OUT IS ANALYTIC, NOT WINDOWED BY STATION. A station-window exclusion (modulate_widths'
`obs.r + long_taper` band) depends on where the grid happens to fall: at 0.30 m spacing the same
obstacle produced a clearance of -0.14 m where 0.10 and 0.50 m gave +0.300. Here each station asks
the obstacle directly how much lateral room it takes at that longitudinal distance,
sqrt(max(0, R^2 - dt^2)) with R = r + obs_margin + w_veh/2, which is grid-independent and exact.

grid_step_m IS A MODELLING CHOICE, NOT A NUMERICAL DETAIL. It sets the resolution of the offset
profile. D2 scales as 1/ds^2, so a fine grid lets millimetre ripple in d appear as curvature: the
same two boxes give peak |kappa| 1.57 at 0.50 m and 7.98 at 0.10 m. A 0.30 m car stepping around a
0.40 m box is a half-metre problem and 0.50 m is the scale it lives at. GRID INDEPENDENCE IS
CLAIMED FOR CLEARANCE ONLY -- that is what the analytic keep-out bought and what the grid sweep
asserts. It is not claimed, or wanted, for curvature.

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
    dev_weight: float = 0.02         # w: the hold-vs-return knob (see the module docstring)
    obs_margin: float = 0.15         # [m] lateral clearance owed to an obstacle, beyond its radius
    w_veh: float = 0.30              # [m] vehicle width, reserved on both sides
    # [m] SOLVED-FOR ONLY, never part of the safety requirement. The QP enforces the keep-out at
    # its own stations; the periodic cubic that maps the answer back to every 0.1 m station sags
    # between them, so the published line grazes a millimetre or two inside what was solved. This
    # covers that sag, measured (bench section 6) at 4.17 mm worst on a 0.50 m grid over the
    # two-box case and the twelve-placement field; 0.010 is 2.4x that. IT IS CALIBRATED TO
    # grid_step_m = 0.50 AND ONLY TO IT -- the same sweep measures 5.4 mm at 0.30 m and 22.6 mm at
    # 0.70 m, so a different grid needs a different allowance. It buys the headroom for nothing
    # measurable: 0.005 / 0.010 / 0.020 all clear 12/12 of the field with the same two boxes
    # unavoidable, at a worst clearance of 0.301 / 0.305 / 0.314.
    # obs_margin stays a safety number and this stays a numerical one -- the ACCEPTANCE check is
    # still obs_margin + w_veh/2, with none of this added.
    disc_allow_m: float = 0.010
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
    infeasible_why: List[str] = field(default_factory=list)   # for the reactive layer
    sides: List[int] = field(default_factory=list)
    max_offset: float = 0.0
    sag_mm: float = float("nan")  # worst keep-out violation the cubic upsample introduces [mm]


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
                    params: ReoptParams,
                    ds: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
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
        R = float(o.r) + params.obs_margin + 0.5 * params.w_veh + params.disc_allow_m
        # EACH STATION STANDS FOR THE HALF-SPACING EITHER SIDE OF IT. Asked at the point, a coarse
        # grid can straddle the obstacle: at 0.70 m spacing the nearest station sits up to 0.35 m
        # along, sees a band of sqrt(R^2 - 0.35^2) instead of R, and the line passes 0.11 m closer
        # than it was told to. Asking with the interval's CLOSEST tangential distance makes the
        # sampled keep-out conservative, so the clearance stops depending on where the grid falls.
        dt_eff = np.maximum(np.abs(dt) - 0.5 * ds, 0.0)
        h = np.sqrt(np.maximum(R * R - dt_eff * dt_eff, 0.0))
        # ON A CLOSED LOOP A SMALL TANGENTIAL PROJECTION IS NOT PROXIMITY. The far side of the
        # track runs back the other way, so a station 12 m from the box can still see it at
        # dt ~ 0 and would take the full keep-out band at du ~ -12 m -- a constraint that cannot
        # be met, which the infeasible path then "resolved" by pinning the line to the wall at
        # stations with no obstacle anywhere near them. The band only constrains a station whose
        # corridor it actually overlaps.
        touched = (h > 0.0) & (du + h > lo) & (du - h < hi)
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
                 params: ReoptParams,
                 why: Optional[List[str]] = None) -> List[int]:
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
        blocked = room_hi < 0.0 and room_lo < 0.0
        if blocked and why is not None:
            why.append(f"box at station {j}: neither side fits -- the left corridor is "
                       f"{-room_hi:.3f} m short and the right corridor {-room_lo:.3f} m short "
                       f"of the keep-out at that station")
        out.append(0 if blocked else (1 if room_hi >= room_lo else -1))
    return out


def corridor_deficit(pts: np.ndarray,
                     nv: np.ndarray,
                     tan: np.ndarray,
                     o: Obstacle,
                     sd: int,
                     lo: np.ndarray,
                     hi: np.ndarray,
                     params: ReoptParams,
                     ds: float = 0.0) -> Tuple[float, int, str]:
    """How far short of this obstacle's keep-out the bare corridor falls: (metres, station, side).

    Asked with the SAFETY radius only -- r + obs_margin + w_veh/2, without disc_allow_m. A
    millimetre of numerical allowance is not a reason to call a box unavoidable; a wall is.
    Deficit <= 0 means the corridor can carry the keep-out at every station the obstacle touches.
    """
    rel = np.array([o.x, o.y])[None, :] - pts
    du = np.einsum("ij,ij->i", rel, nv)
    dt = np.einsum("ij,ij->i", rel, tan)
    R = float(o.r) + params.obs_margin + 0.5 * params.w_veh
    dt_eff = np.maximum(np.abs(dt) - 0.5 * ds, 0.0)
    h = np.sqrt(np.maximum(R * R - dt_eff * dt_eff, 0.0))
    t = (h > 0.0) & (du + h > lo) & (du - h < hi)
    if not np.any(t):
        return -np.inf, -1, "none"
    if sd > 0:
        short = (du[t] + h[t]) - np.where(np.isfinite(hi[t]), hi[t], np.inf)
        which = "left"      # the +normvec wall is the one in the way
    else:
        short = np.where(np.isfinite(lo[t]), lo[t], -np.inf) - (du[t] - h[t])
        which = "right"
    k = int(np.argmax(short))
    return float(short[k]), int(np.flatnonzero(t)[k]), which


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
    rep.sides = (select_sides(pts_fine, nv_fine, obstacles, lo_f, hi_f, p, rep.infeasible_why)
                 if len(obstacles) else [])
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

    # --- what the corridor genuinely cannot carry ---------------------------------------------
    # An obstacle can pick a side at its own station and still be impossible a metre later, where
    # the keep-out disc is still wide and the track has narrowed. Forcing it through would eat the
    # wall margin for a clearance that was never available, so it is dropped from the QP and named
    # -- with the station and the side and the metres -- for the reactive layer, which is the one
    # that can still slow down or squeeze. The obstacles that ARE avoidable are solved as normal.
    ds = float(np.sum(el_fine)) / len(ci)
    keep, keep_sides = [], []
    for o, sd in zip(use, use_sides):
        short, k, which = corridor_deficit(pts_c, nv_c, tan_c, o, sd, lo_f[ci], hi_f[ci], p, ds)
        if short > 1e-6:
            j = int(np.argmin(np.hypot(pts_fine[:, 0] - o.x, pts_fine[:, 1] - o.y)))
            rep.infeasible.append(next(i for i, ob in enumerate(obstacles) if ob is o))
            rep.infeasible_why.append(
                f"box at station {j}: the {which} corridor is {short:.3f} m short of the keep-out "
                f"at station {int(ci[k])}")
        else:
            keep.append(o)
            keep_sides.append(sd)
    if not keep:
        rep.ok = True
        rep.reason = "every obstacle is unavoidable"
        rep.peak_kappa, rep.peak_station = rep.clean_peak_kappa, rep.clean_peak_station
        rep.solve_ms = (time.perf_counter() - t0) * 1e3
        return pts_fine, np.zeros(n), rep
    use, use_sides = keep, keep_sides

    ds = float(np.sum(el_fine)) / len(ci)
    lo_c, hi_c = obstacle_bounds(pts_c, nv_c, tan_c, use, use_sides, lo_f[ci], hi_f[ci], p, ds)

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

    # WHAT THE UPSAMPLE COSTS. The QP holds the keep-out exactly at its own stations; the periodic
    # cubic through them dips between. Measured here against the very bounds that were solved, so a
    # positive number is interpolation and nothing else -- it is what disc_allow_m has to cover.
    ko_lo, ko_hi = obstacle_bounds(pts_fine, nv_fine, _tan_f, use, use_sides, lo_f, hi_f, p)
    by_box = np.maximum(ko_lo - d_fine, d_fine - ko_hi)
    binds = (ko_lo > lo_f + 1e-9) | (ko_hi < hi_f - 1e-9)
    rep.sag_mm = float(np.max(by_box[binds])) * 1e3 if np.any(binds) else 0.0

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
