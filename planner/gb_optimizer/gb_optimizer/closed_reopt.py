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
  curvature does not choose w -- it is 0.010 because that keeps the widest margin
  over the 0.40 hold floor while still bringing the line home (see the field above). reach, span,
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
    """Six numbers, and the sixth is why: locality is free to a hump and has to be bought here."""
    # [m] the station spacing the QP solves on. 0.10 IS the published spacing, so stride is 1: the
    # solver constrains every point that goes on the wire and no interpolation stands between the
    # two. Measured over the whole pipeline (ripple = max |kappa_new - kappa_clean| where the
    # offset is under 2 mm; flips = sign changes of the offset's second difference over the merge
    # tail):
    #
    #     grid   ripple   flips   hold    clearance   ms
    #     0.50   0.2457     21    0.442     0.320      4-6
    #     0.30   0.4281     19    0.442     0.321      9-16
    #     0.20   0.0859     21    0.454     0.330     20-22
    #     0.10   0.0634      2    0.454     0.327    178-209
    #
    # 0.20 is the fallback if 200 ms ever matters: same ripple magnitude at a tenth of the time,
    # but the sign oscillation stays. peak |kappa| is 1.441-1.448 at EVERY spacing -- the 7.98 a
    # fine grid produced in an earlier round predates the envelope and the curvature budget, and a
    # solution this constrained no longer has room to ripple. 200 ms is faster than the hump
    # pipeline (395-822 ms) and runs off the executor either way.
    grid_step_m: float = 0.10
    # 0.010, not the 0.020 the "lowest peak that still holds 0.40" rule picks: peak |kappa| over
    # the whole sweep is 1.425-1.447 and does not discriminate, so what is left is hold margin
    # against offset left lying around the lap. 0.01 holds 0.499 (+25% over the 0.40 floor) and
    # leaves 5.8 cm; 0.02 holds 0.469 (+17%) and leaves 4.0 cm. Losing the hold is the QUALITATIVE
    # failure this whole rewrite exists to fix -- a straight pair drawn as a W -- and 1.8 cm of
    # residual offset is a quantitative one. The asymmetry points down.
    dev_weight: float = 0.010         # w: the hold-vs-return knob (see the module docstring)
    # [m] lateral clearance owed to an obstacle beyond its radius. The line DELIVERS
    # obs_margin + w_veh/2 = 0.320 and now PROMISES the same, which is the point of it being 0.17
    # rather than 0.16: the tenth of a centimetre that used to arrive as disc_allow_m was real
    # clearance under an interpolation's name, and the margin chain was told the smaller number.
    # The keep-out radius is bit-identical either way -- (0.16, disc_allow 0.010) and (0.17,
    # disc_allow 0.0) were measured to give the same coverage 29/38, the same clearances to six
    # decimals, the same holds, the same peaks and the same sum of |d| -- so this is a renaming,
    # not a loosening.
    #
    # What it has to beat is the reactive layer's idle entry (width_car/2 + clear_margin_m +
    # clear_hyst_m = 0.28): below that the reactive layer avoids a box the global line already
    # claims. 0.320 leaves 4 cm where check_avoidance_margins.py asks for 5, and that check still
    # FAILS -- deliberately, and not papered over. The next centimetre is not available here:
    # sweep_obs_margin.py measures 0.18 costing seven of 38 boxes and every hold on ifac, because
    # a keep-out radius of 0.49 m no longer fits inside the curvature budget's envelope on a
    # "straight" that is already at 16% of it. Whether 4 cm is enough is a sim question, and the
    # DOUBLE AVOIDANCE warning in static_reopt_node is what answers it.
    obs_margin: float = 0.17
    w_veh: float = 0.30              # [m] vehicle width, reserved on both sides
    # [m] SOLVED-FOR ONLY, never part of the safety requirement: it covers the sag the periodic
    # cubic introduces between QP stations. AT grid_step_m = 0.10 THERE ARE NO STATIONS BETWEEN
    # QP STATIONS -- stride is 1, the upsample is skipped entirely, and the measured sag over the
    # two-box case and all twelve field placements is exactly 0.0000 mm. So this is 0.0, and it is
    # kept rather than deleted because it is the correct value for any coarser grid (4.11 mm at
    # 0.50, 5.40 at 0.30, 22.97 at 0.70 -- roughly 2x those as an allowance).
    #
    # It is NOT quietly left at 0.010 as free margin. At stride 1 that number no longer corrects
    # anything; it would simply be a centimetre of extra clearance wearing an interpolation's
    # name, and mixing those two is exactly what splitting it off obs_margin was for. The
    # centimetre itself was not thrown away -- it moved into obs_margin (0.16 -> 0.17), which is
    # where a safety margin belongs, so the keep-out radius is unchanged at r + 0.320 and only the
    # PROMISE moved: the margin chain is now told 0.320, the number actually delivered.
    disc_allow_m: float = 0.0
    # [m] LOCALITY, BOUGHT EXPLICITLY. Beyond this arc distance from every box the offset is
    # pinned to exactly zero, so an avoidance cannot move the racing line on the far side of the
    # lap. The hump pipeline gets this for free by laying a bump; a global objective does not, and
    # measured without it the QP left a median 0.062 m and up to 0.150 m of offset more than 2.5 m
    # from any box. The LOWER bound is the hold: two boxes 8.5 m apart need the envelope to still
    # be open at the 4.25 m midpoint, and 4.0 m kills that hold outright (0.000).
    infl_len_m: float = 6.0
    # [1/m] the curvature the line is allowed to reach. NOT a rejection threshold -- it is a
    # SPATIAL BUDGET: where the clean raceline already uses most of it, the envelope closes and
    # the line is held near the raceline. On ifac the apex at station 227 is at 1.448 of this, so
    # 3.5% of the budget is left there and the line is all but pinned; on a straight at 0.23 the
    # budget is 85% free and the envelope is untouched. A box that cannot be cleared inside the
    # budget is handed to the reactive layer by name, never cleared by less than it is owed.
    kappa_budget: float = 1.5


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
    # peak |kappa| AT THE QP's OWN STATIONS -- where the envelope is actually enforced, so this is
    # what the curvature budget contracts for. peak_kappa above is over every published station
    # and therefore also carries whatever the periodic cubic adds BETWEEN nodes; the difference of
    # the two is the interpolation excess, and the two are gated separately for that reason.
    peak_kappa_nodes: float = 0.0
    # [m] how far the UPSAMPLED offset steps outside the envelope between QP stations. The bound
    # is enforced at the nodes; the periodic cubic through them is not bound to anything in
    # between, and where the envelope is nearly shut -- at an apex the curvature budget has closed
    # down to millimetres -- that overshoot is the entire story. Metres, not curvature, for the
    # same reason disc_allow_m is metres: it is a property of the interpolation, and what it costs
    # in curvature depends on which corner it lands in.
    env_overshoot_m: float = 0.0


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


# How hard the curvature budget bites: the corridor at a station keeps this power of the budget
# fraction still free there. A module constant on purpose -- it shapes the same trade-off
# infl_len_m does, and two knobs for one trade-off is how the hump pipeline ended up with sixteen.
#
# 1.0 rather than 0.5, measured on the four cases where the QP was still raising ifac's apex
# (peak |kappa| / boxes covered of 36):
#
#     q = 0.50   1.591 1.572 1.574 1.746   32/36
#     q = 1.00   1.410 1.451 1.406 1.417   29/36
#
# Three more boxes go to the reactive layer, which handles them every lap -- that is lap 1's
# normal behaviour and functionally whole. The alternative is 1.746 at station 227, which
# _cap_speed_to_published_curvature reads straight into the speed plan and which slows an
# obstacle-free corner for the whole stint, while asking the speed planner for a curvature
# outside its own curvlim of 1.5. Local and temporary against global and persistent.
_BUDGET_EXP = 1.0


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


def locality_envelope(ci: np.ndarray,
                      track_len: float,
                      obstacles: Sequence[Obstacle],
                      pts_fine: np.ndarray,
                      s_fine: np.ndarray,
                      k_clean: np.ndarray,
                      params: ReoptParams) -> np.ndarray:
    """g(s) in [0, 1]: the fraction of the corridor the offset may use at each station.

    Two factors, multiplied, and both are HARD -- they enter the QP as bounds, not as weights:

      LOCALITY. 1 within infl_len_m/2 of the nearest box, smoothstepped to 0 at infl_len_m. Where
      it reaches 0 the offset is pinned to 0 and the avoidance simply cannot reach that far. This
      is the property the hump pipeline has by construction and a global objective does not.

      CURVATURE BUDGET. sqrt(the fraction of kappa_budget the clean line is NOT already using).
      Minimising the OFFSET's bending says nothing about the TOTAL curvature: the offset's own
      tail is smooth and still adds to a corner that is already at the limit, which is exactly how
      the QP put 2.052 into ifac's apex while the hump stayed at 1.445. The exponent is a module
      constant, not a knob -- 0.5 is chosen so a corner at half the budget keeps 71% of its
      corridor rather than 50%, i.e. the budget bites hard only near the limit.
    """
    if not len(obstacles):
        return np.zeros(len(ci))
    d_arc = np.full(len(s_fine), np.inf)
    for o in obstacles:
        j = int(np.argmin(np.hypot(pts_fine[:, 0] - o.x, pts_fine[:, 1] - o.y)))
        gap = np.abs(s_fine - s_fine[j])
        d_arc = np.minimum(d_arc, np.minimum(gap, track_len - gap))   # wrap-aware
    half = 0.5 * max(params.infl_len_m, 1e-6)
    t = np.clip((d_arc - half) / half, 0.0, 1.0)
    g = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
    frac = np.clip((params.kappa_budget - k_clean) / max(params.kappa_budget, 1e-9), 0.0, 1.0)
    g = g * np.power(frac, _BUDGET_EXP)
    # READ AT THE STATION, NOT OVER ITS INTERVAL -- unlike the keep-out, and measured, not assumed.
    # The keep-out's half-spacing correction is nearly free; the same correction here is not. It
    # fixes a real 2 mm overshoot (ifac's apex sits BETWEEN two nodes, so the cubic carries 11 mm
    # of offset where the budget there allows 9, which is what puts four cases 6/1000 of curvature
    # over the clean line) and it costs, on the same twenty-case matrix:
    #
    #     reduction   boxes covered   peak <= clean+0.005   hold across the 8.5 m pair
    #     pointwise      29/38             16/20                   0.475
    #     interval       25/38             20/20                   none -- a box is handed over
    #
    # The tightest station in a 0.5 m cell then governs the whole cell, and on ifac even a
    # "straight" is at 16% of the budget, so the corridor available to a keep-out that needs
    # 0.46 m falls under it and the box goes to the reactive layer. Buying a curvature gate with
    # the hold that this whole formulation exists to produce is the wrong trade, so the overshoot
    # is reported instead of paid for.
    return g[ci]


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
    ds = float(np.sum(el_fine)) / len(ci)
    s_fine = np.concatenate([[0.0], np.cumsum(el_fine)[:-1]])
    L = float(np.sum(el_fine))
    need = p.obs_margin + 0.5 * p.w_veh

    def name(o):
        return next(i for i, ob in enumerate(obstacles) if ob is o)

    def station(o):
        return int(np.argmin(np.hypot(pts_fine[:, 0] - o.x, pts_fine[:, 1] - o.y)))

    # --- solve, then CHECK WHAT THE LINE ACTUALLY DELIVERED --------------------------------
    # THE CONTRACT OF THIS LAYER: every box the global line claims is cleared by the margin it is
    # owed. A box it cannot clear is handed to the reactive layer BY NAME -- that layer can still
    # brake, squeeze or take a different side, and this one cannot. What is not allowed is
    # claiming a box and clearing it by less, which is how a downstream fail-closed gate ends up
    # re-avoiding a line that was published as safe.
    d_c = d_fine = line = None
    for _round in range(len(obstacles) + 2):
        # the envelope follows the boxes still being avoided, so it is rebuilt every round
        g = locality_envelope(ci, L, use, pts_fine, s_fine, k_clean, p)
        env_hi = np.minimum(hi_f[ci], g * np.maximum(hi_f[ci], 0.0))
        env_lo = np.maximum(lo_f[ci], g * np.minimum(lo_f[ci], 0.0))

        # what the envelope, not the wall, makes impossible
        keep, keep_sides = [], []
        for o, sd in zip(use, use_sides):
            raw, _k0, _w0 = corridor_deficit(pts_c, nv_c, tan_c, o, sd, lo_f[ci], hi_f[ci], p, ds)
            env, k, which = corridor_deficit(pts_c, nv_c, tan_c, o, sd, env_lo, env_hi, p, ds)
            if env <= 1e-6:
                keep.append(o)
                keep_sides.append(sd)
                continue
            j, kj = station(o), int(ci[k])
            rep.infeasible.append(name(o))
            if raw > 1e-6:
                rep.infeasible_why.append(
                    f"box at station {j}: the {which} corridor is {raw:.3f} m short of the "
                    f"keep-out at station {kj}")
            else:
                used = 100.0 * min(k_clean[kj] / max(p.kappa_budget, 1e-9), 1.0)
                gap = abs(s_fine[kj] - s_fine[j])
                need_at = abs(env_hi[k] if which == "left" else env_lo[k]) + env
                rep.infeasible_why.append(
                    f"box at station {j}: the clean line is at {used:.1f}% of the curvature "
                    f"budget {min(gap, L - gap):.1f} m from it; clearing it would need the line "
                    f"moved {need_at:.3f} m where only {need_at - env:.3f} m of budget remains")
        if len(keep) != len(use):
            use, use_sides = keep, keep_sides
            if not use:
                break
            continue

        lo_c, hi_c = obstacle_bounds(pts_c, nv_c, tan_c, use, use_sides, env_lo, env_hi, p, ds)
        empty = lo_c > hi_c
        if np.any(empty):
            # THE WALL WINS ONLY WHERE THE GEOMETRY IS THE PROBLEM. Two keep-outs that overlap in
            # opposite directions leave a station with no interval at all, and something has to
            # give; the corridor bound is kept and the keep-out relaxed to meet it. This path is
            # NOT allowed to absorb a curvature-budget conflict -- that is a budget WE chose, and
            # a box it makes unreachable is handed over above, named, rather than cleared short.
            n_bad = int(np.count_nonzero(empty))
            hi_c[empty] = np.maximum(lo_c[empty] + 1e-6, hi_c[empty])
            lo_c[empty] = np.minimum(lo_c[empty], hi_c[empty] - 1e-6)
            rep.reason = (f"{n_bad} station(s) could not have both the corridor and the keep-out; "
                          f"the corridor won and the clearance below is what was achievable")

        d_c = solve_offsets(lo_c, hi_c, ds, p.dev_weight)
        if d_c is None:
            rep.ok = False
            rep.reason = rep.reason or "quadprog could not solve the box-constrained QP"
            rep.solve_ms = (time.perf_counter() - t0) * 1e3
            return pts_fine, np.zeros(n), rep

        if stride == 1:
            # THE QP ALREADY SOLVED EVERY STATION. Interpolating here would be the identity in
            # exact arithmetic and is not in floating point, so the spline is skipped rather than
            # trusted to be a no-op -- and with it goes the entire ringing mechanism below.
            d_fine = d_c
        else:
            # PERIODIC CUBIC back to every station. A linear map puts a corner at every coarse node
            # and the fine line inherits it: peak |kappa| 1.9-2.7 from the interpolation alone.
            # The cubic has its own failure though, and it is what drove grid_step_m to 0.10: the
            # QP minimises a second difference over 0.5 m while the PUBLISHED curvature is the
            # cubic's second derivative over 0.1 m, and those are different quantities. Where the
            # envelope shuts (d pinned to 0) and meets a ramp, the cubic rings -- offsets under a
            # millimetre alternating sign, which at ds = 0.0997 is 0.7 mm of second difference and
            # therefore 0.07 1/m of curvature the solver never asked for.
            d_fine = CubicSpline(np.concatenate([s_fine[ci], [s_fine[ci][0] + L]]),
                                 np.concatenate([d_c, [d_c[0]]]), bc_type="periodic")(s_fine)
        line = pts_fine + nv_fine * d_fine[:, None]

        got = [float(np.min(np.hypot(line[:, 0] - o.x, line[:, 1] - o.y)) - o.r) for o in use]
        short = [i for i, c in enumerate(got) if c < need - 1e-9]
        if not short:
            break
        worst = min(short, key=lambda i: got[i])
        o = use[worst]
        j = station(o)
        kj = int(ci[int(np.argmin(np.abs(s_fine[ci] - s_fine[j])))])
        used = 100.0 * min(k_clean[kj] / max(p.kappa_budget, 1e-9), 1.0)
        rep.infeasible.append(name(o))
        rep.infeasible_why.append(
            f"box at station {j}: the clean line is at {used:.1f}% of the curvature budget at it; "
            f"the line clears it by {got[worst]:.3f} m where {need:.3f} m is owed, so it is the "
            f"reactive layer's")
        use = [x for i, x in enumerate(use) if i != worst]
        use_sides = [x for i, x in enumerate(use_sides) if i != worst]
        d_c = d_fine = line = None
        if not use:
            break

    if not use or d_fine is None:
        rep.ok = True
        rep.reason = rep.reason or "every obstacle is the reactive layer's"
        rep.peak_kappa, rep.peak_station = rep.clean_peak_kappa, rep.clean_peak_station
        rep.solve_ms = (time.perf_counter() - t0) * 1e3
        return pts_fine, np.zeros(n), rep

    # WHAT THE UPSAMPLE COSTS. The QP holds the keep-out at its own stations; the periodic cubic
    # through them dips between. Measured against the very bounds that were solved, so a positive
    # number is interpolation and nothing else -- it is what disc_allow_m has to cover.
    ko_lo, ko_hi = obstacle_bounds(pts_fine, nv_fine, _tan_f, use, use_sides, lo_f, hi_f, p)
    by_box = np.maximum(ko_lo - d_fine, d_fine - ko_hi)
    binds = (ko_lo > lo_f + 1e-9) | (ko_hi < hi_f - 1e-9)
    rep.sag_mm = float(np.max(by_box[binds])) * 1e3 if np.any(binds) else 0.0

    g_f = locality_envelope(np.arange(n), L, use, pts_fine, s_fine, k_clean, p)
    env_hi_f = np.minimum(hi_f, g_f * np.maximum(hi_f, 0.0))
    env_lo_f = np.maximum(lo_f, g_f * np.minimum(lo_f, 0.0))
    rep.env_overshoot_m = float(max(0.0, np.max(np.maximum(d_fine - env_hi_f,
                                                           env_lo_f - d_fine))))

    k_new = np.abs(menger_closed(line))
    rep.peak_kappa_nodes = float(np.max(k_new[ci]))
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
        i0, i1 = station(use[0]), station(use[1])
        span = (np.arange(i0, i1 + 1) if i1 >= i0 else np.arange(i0, i1 + n + 1)) % n
        if len(span) > 10:
            rep.hold = float(np.min(np.abs(d_fine[span[5:-5]])))
    rep.ok = True
    rep.solve_ms = (time.perf_counter() - t0) * 1e3
    return line, d_fine, rep
