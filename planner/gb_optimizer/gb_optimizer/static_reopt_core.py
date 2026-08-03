"""
static_reopt_core.py — width-modulation + re-optimization core for IFAC static-obstacle handling.

Given a *clean* reference track (centerline + track widths, the `x_m,y_m,w_tr_right_m,w_tr_left_m`
format used everywhere in this stack) and a set of static obstacles (map-frame disks),
this module narrows the drivable corridor around each obstacle — recentring the reference
line onto the free side — and re-runs the closed-loop optimizer to produce an obstacle-aware
raceline (`mincurv_iqp`) and overtaking line (`shortest_path`).

It reuses gb_optimizer's vendored `trajectory_optimizer` as a *library only* — no gb_optimizer
node is modified. The width representation modulated here is exactly the reftrack CSV that the
optimizer already consumes, so the fragile occupancy-grid -> skeleton -> centerline stage is
bypassed entirely (see memory: project-ifac-static-reopt).

Sign convention (matches gb_optimizer / tph):
    normvec = calc_normal_vectors(psi) points toward the RIGHT bound (== +w_tr_right).
    corridor coordinate u along normvec: left bound at u=-w_tr_left, right bound at u=+w_tr_right.
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# --- vendored lib shim: make `global_racetrajectory_optimization` importable as a
#     top-level package, identical to gb_optimizer/global_planner_node.py --------------
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import trajectory_planning_helpers as tph  # noqa: E402

# NOTE: the vendored `trajectory_optimizer` (whole-track offline optimizer) is imported
# LAZILY inside `reoptimize()` only. The fast ONLINE `reoptimize_local_window` path needs
# just tph, so it must not drag in the heavy offline chain (vel_planner, etc.).


# ======================================================================================
# Data classes
# ======================================================================================
@dataclass
class Obstacle:
    """A static obstacle as a disk in the map frame."""
    x: float
    y: float
    r: float = 0.20  # radius [m], should already include the object's physical half-size


@dataclass
class ModulationParams:
    """Parameters for corridor width modulation."""
    obs_margin: float = 0.15     # extra lateral clearance added to the obstacle radius [m].
                                 # Two roles: (1) localization uncertainty, and (2) DOUBLE-
                                 # AVOIDANCE PREVENTION (design choice A): the re-optimized
                                 # line must clear an obstacle by more than the reactive
                                 # planner's trigger band so the reactive layer does NOT
                                 # re-avoid an obstacle already handled by the global line.
                                 # Required:  obs_margin > gb_ego_width_m/2 - safety_width/2
                                 # (+ buffer). With gb_ego_width_m=0.4 that is >-0.05 at
                                 # safety_width=0.5, or >~0.08 at safety_width=0.25.
                                 # NB: reactive static avoidance is not yet tuned — re-check
                                 # this against gb_ego_width_m / evasion_dist after tuning.
    long_taper: float = 0.30     # longitudinal blend distance past the obstacle radius [m]
                                 # over which the narrowing ramps back to zero (avoids kinks)
    min_halfwidth: float = 0.10  # minimum drivable half-width to keep the optimizer feasible [m]


@dataclass
class ModulationReport:
    """Diagnostics from a width-modulation pass."""
    n_stations: int = 0
    n_affected: int = 0
    n_infeasible: int = 0
    min_halfwidth_seen: float = float("inf")
    obstacle_sides: List[str] = field(default_factory=list)   # 'right' | 'left' | 'skip' per obstacle
    infeasible_s_idx: List[int] = field(default_factory=list)


# ======================================================================================
# Reftrack IO + geometry
# ======================================================================================
def load_reftrack(csv_path: str) -> np.ndarray:
    """Load a reftrack CSV `[x_m, y_m, w_tr_right_m, w_tr_left_m]` (header optional).

    Returns an (N, 4) float array, unclosed (path[-1] != path[0]).
    """
    rows: List[List[float]] = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(";", ",").split(",")
            try:
                vals = [float(p) for p in parts[:4]]
            except ValueError:
                continue  # header row
            if len(vals) == 4:
                rows.append(vals)
    if not rows:
        raise IOError(f"No reftrack rows parsed from {csv_path}")
    reftrack = np.asarray(rows, dtype=float)
    # drop a duplicate closing point if present (optimizer/tph want it unclosed)
    if np.allclose(reftrack[0, :2], reftrack[-1, :2]):
        reftrack = reftrack[:-1]
    return reftrack


def _closed_el_lengths(pts: np.ndarray) -> np.ndarray:
    """Element lengths for a CLOSED path (length N, includes the closing segment)."""
    seg = np.roll(pts, -1, axis=0) - pts
    return np.hypot(seg[:, 0], seg[:, 1])


def _cyclic_smooth(a: np.ndarray, win: int = 7) -> np.ndarray:
    """Moving-average smoothing on a closed (cyclic) 1-D signal, length preserved.

    A closed raceline array carries a DUPLICATED closing point (a[-1] == a[0]); wrapping over the
    full length would then treat that duplicate as its own station and smooth with period N instead
    of the true N-1, leaving a seam error at s=0 (0.014 m on ifac bounds) that `alpha_dd`'s 1/h^2
    ~= 100 gain turns into a fake curvature spike at start/finish. Detect the duplicate, smooth the
    unique part cyclically, then restore it."""
    win = max(1, int(win))
    a = np.asarray(a, dtype=float)
    if win == 1 or len(a) < win:
        return a.astype(float)
    if len(a) > win + 1 and np.isclose(a[0], a[-1]):
        inner = _cyclic_smooth(a[:-1], win)          # true period = N-1
        return np.append(inner, inner[0])            # restore the closing duplicate
    k = np.ones(win) / win
    pad = win // 2
    ext = np.concatenate([a[-pad:], a, a[:win - pad - 1]])
    return np.convolve(ext, k, mode="valid")


# Fraction of the lap ONE avoidance hump (entry + exit) may occupy. The re-optimized line must
# remain a local detour that visibly rejoins the racing line; without this bound the entry ramp
# grows until a single obstacle perturbs most of the lap.
_HUMP_SPAN_FRAC = 0.28   # 0.20 -> 0.28: room for the entry/exit ramp stretching that shallows
                         # the merge-zone inflections (user-visible S-kinks where the hump
                         # rejoins the raceline); still keeps a hump under ~28% of the lap.
# ALL-OR-NOTHING apex fit: a hump the corridor forces below this fraction of the recorded
# reactive apex does NOT clear the obstacle (the apex |d| is the reactive-PROVEN clearance) —
# laying the shrunken hump wastes lap time, still triggers the reactive layer every lap
# (clear-gate never idles -> OVERTAKE<->GB_TRACK churn) and re-records apexes. Measured on
# ifac: want -0.46 laid -0.28. Such apexes are DROPPED and reported instead.
_APEX_KEEP_FRAC = 0.90
# JOINT (post-weave) verification of the multi-apex profile. A single hump is verified against the
# corridor and the curvature bar before it is accepted, but where two ramps OVERLAP the BPoly weaves
# them apex-to-apex and the result is a shape neither was tested as. These bound the recovery:
# shrink the reach of the humps covering a violation by _WEAVE_SHRINK, re-weave, up to
# _WEAVE_MAX_PASSES times, then fail honestly. Shrinking separates the ramps, which is what
# restores the per-hump verdicts.
_WEAVE_SHRINK = 0.8
_WEAVE_MAX_PASSES = 6
# Tolerance on the clearance ACCEPTANCE comparison [m]. The obstacle-derived target amplitude is
# d_obs + side*(r + obs_margin), so on a straight the laid line clears the box by EXACTLY
# obs_margin and `gap >= need` is decided by the last bit of a float. Measured: 26 of 70 apexes on
# map f dropped with reason 'clearance' at a reported clear of 0.350 inside a 1.0-1.9 m corridor,
# purely on that comparison. A micrometre is far below anything the margin chain cares about.
_CLEAR_EPS = 1e-6

# Peak |d''| of the C2 quintic hump in closed form: for d(u) = A*(10t^3 - 15t^4 + 6t^5),
# t = u/r, the extrema of d'' sit at t = (1 +- 1/sqrt(3))/2 and give |d''|max = (10/sqrt(3))*|A|/r^2.
# Verified numerically at r = 1, 2, 5 (coefficient 5.7735 in all three). Because the published
# curvature is kappa_clean + d'', this inverts to the SMALLEST reach that keeps a hump of
# amplitude A inside a curvature budget:  r_min = sqrt(_KAPPA_QUINTIC_C * |A| / budget).
_KAPPA_QUINTIC_C = 10.0 / math.sqrt(3.0)          # = 5.7735
# Curvature budget floor as a fraction of curvlim. At a raceline apex kappa_clean already sits at
# ~curvlim, so `curvlim - max|kappa_clean|` alone goes to zero and demands an infinite reach. The
# floor keeps r_min finite; the honest all-or-nothing drop then handles the truly impossible ones.
_KAPPA_BUDGET_FLOOR_FRAC = 0.25
# Default corridor-fit tolerance [m]. The corridor bounds are a `_cyclic_smooth`ed ESTIMATE that
# ripples ~2 mm between adjacent stations, and the amplitude cap parks the hump peak exactly on
# that bound — so a zero-tolerance test rejects reaches over a SUB-MILLIMETRE violation at the
# station next to the apex, and the reach bisects down to whatever is sharp enough to duck under
# the ripple. Measured on ifac: a 0.5 mm violation at station 292 collapsed the reach 5.00 -> 1.24 m,
# which took max|kappa| 1.46 -> 1.98 (curvlim is 1.5) and cost +1.62 s on an 11.30 s lap. 5 mm is
# 4-10% of the wall_margin it eats into, and the true track bound is still wall_margin + w_veh/2 away.
_FIT_TOL_DEFAULT = 0.005


def _hump_values(u_stn: np.ndarray, u_c: float, d: float, r_in: float, r_out: float):
    """One C2 quintic hump (raceline -> apex -> raceline) sampled at the cut-linear stations `u_stn`.

    Closed form, no scipy. `BPoly.from_derivatives([u_c-r_in, u_c, u_c+r_out], [[0,0,0], [d,0,0],
    [0,0,0]])` IS the quintic Hermite interpolant on each half, i.e. d * smootherstep — the two agree
    to 1.3e-15 m over randomised knots. Written out because the corridor and curvature fits evaluate
    this hundreds of times per apex and constructing a BPoly object costs more than the arithmetic;
    replacing it here made the 3-apex solve ~35 ms FASTER than before the curvature checks existed.
    The final laid profile still goes through BPoly, which also weaves several apexes into one
    polynomial. Returns None on degenerate knots."""
    if not (r_in > 1e-6 and r_out > 1e-6 and r_in + r_out > 1e-3):
        return None
    v = np.zeros_like(u_stn)
    rel = u_stn - u_c
    for m, r in (((rel >= -r_in) & (rel <= 0.0), r_in), ((rel > 0.0) & (rel <= r_out), r_out)):
        if np.any(m):
            t = 1.0 - np.abs(rel[m]) / r                  # 0 at the ramp end, 1 at the apex
            v[m] = d * t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
    return v


def _fit_hump_to_corridor(u_stn: np.ndarray, u_c: float, d: float, r0: float, track_len: float,
                          hi_inc: np.ndarray, lo_inc: np.ndarray, reach_floor: float,
                          fit_tol: float = _FIT_TOL_DEFAULT):
    """Shrink one hump until it FITS the corridor, instead of letting the final clip chop it.

    Root cause this addresses: the corridor is zero-width at many scattered stations (on ifac 59/355
    = 17% have d_right - w_veh/2 - wall_margin <= 0, because the min-curvature line hugs the inside
    wall), while a speed-scaled ramp spans ~79 stations. Element-wise clipping of the analytic hump
    against that comb-shaped bound turns a PERFECT single hump (1 local extremum) into a 3-5
    extremum comb, and alpha''s 1/h^2 ~= 100 gain blows those cm-scale steps up into a curvature
    that flips sign every station. Shrinking the hump keeps it ANALYTIC (always exactly 1 extremum,
    C2 by construction) — a narrower hump is sharper but never wavy, and the clip then barely bites.

    `fit_tol` is the feasibility TOLERANCE, and it is load-bearing — see _FIT_TOL_DEFAULT. Step 1
    parks the amplitude exactly ON the bound, so with a zero tolerance the neighbouring stations
    (where the smoothed bound ripples ~2 mm tighter) reject every reach that is still near-peak
    there. The reach then bisects to whatever is sharp enough to fall away from the peak within
    one station — measured on ifac: a 0.5 mm violation at the station 0.1 m from the apex collapsed
    the reach from 5.00 m to 1.24 m, and with it max|kappa| 1.46 -> 1.98 and +1.62 s of lap time.

    Returns (d_fitted, r_fitted). Amplitude is capped at the apex station first, then the reach is
    bisected; the hump shrinks monotonically in both, so bisection is well posed."""
    bound = hi_inc if d > 0 else lo_inc
    # 1) cap the amplitude at the apex station (an apex the corridor cannot hold at all)
    i_c = int(np.argmin(np.abs(u_stn - u_c)))
    cap = bound[i_c]
    d = min(d, cap) if d > 0 else max(d, cap)
    if abs(d) < 0.03:
        return 0.0, r0

    def fits(r, amp):
        v = _hump_values(u_stn, u_c, amp, r, r)
        if v is None:
            return False
        return bool(np.all(v <= hi_inc + fit_tol)) and bool(np.all(v >= lo_inc - fit_tol))

    if fits(r0, d):
        return d, r0
    # 2) bisect the REACH down. A narrower hump is sharper but stays analytic (1 extremum);
    # widening it into a corridor that cannot hold it is what forced the clip to notch it.
    lo_r, hi_r = reach_floor, r0
    if fits(lo_r, d):
        for _ in range(24):
            mid = 0.5 * (lo_r + hi_r)
            if fits(mid, d):
                lo_r = mid
            else:
                hi_r = mid
        return d, lo_r
    # 3) even the narrowest hump does not fit at this amplitude -> bisect the AMPLITUDE at the
    # floor reach. Both shrink the hump monotonically, so this always terminates on a feasible pair.
    lo_a, hi_a = 0.0, d
    for _ in range(24):
        mid = 0.5 * (lo_a + hi_a)
        if fits(reach_floor, mid):
            lo_a = mid
        else:
            hi_a = mid
    return (lo_a if abs(lo_a) >= 0.03 else 0.0), reach_floor


def centerline_frame(reftrack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-point heading psi (north-up), unit normal (toward +w_tr_right) and
    unit tangent for a closed reference line.

    Returns (psi[N], normvec[N,2], tangent[N,2]).
    """
    pts = reftrack[:, :2]
    el = _closed_el_lengths(pts)
    psi, _ = tph.calc_head_curv_num.calc_head_curv_num(
        path=pts, el_lengths=el, is_closed=True
    )
    normvec = tph.calc_normal_vectors.calc_normal_vectors(psi)      # points toward +w_tr_right
    # tangent = normal rotated by +90deg; for north-up psi the tangent is [-sin? ] — derive
    # directly from consecutive geometry to stay independent of angle conventions.
    tangent = np.roll(pts, -1, axis=0) - pts
    nrm = np.hypot(tangent[:, 0], tangent[:, 1])
    nrm[nrm < 1e-9] = 1e-9
    tangent = tangent / nrm[:, None]
    return psi, normvec, tangent


# ======================================================================================
# Width modulation
# ======================================================================================
def modulate_widths(
    reftrack: np.ndarray,
    obstacles: List[Obstacle],
    params: Optional[ModulationParams] = None,
    min_total_width: Optional[float] = None,
    recenter: bool = True,
) -> Tuple[np.ndarray, ModulationReport]:
    """Narrow the drivable corridor around each obstacle and recenter the reference line
    onto the chosen free side.

    `recenter` (default True): move the reference line to the middle of the free corridor
    around each obstacle. This gives the whole-track mincurv_iqp a feasible reference. For
    the ONLINE windowed QP set recenter=False: the reference line is kept UNCHANGED (the
    smooth clean raceline) and the obstacle exclusion is expressed purely in the returned
    widths `[w_tr_right=keep_hi, w_tr_left=-keep_lo]` (which may be negative on the blocked
    side, i.e. the reference lies outside the free corridor — opt_min_curv still solves it
    as a one-sided box). Recentering a narrow (~0.5 m) exclusion zone produces a reference
    spike that a coarse QP grid cannot represent, so recenter=False is required there.

    `min_total_width` (typically the optimizer's safety_width) guarantees every returned
    station stays wide enough for the QP to remain solvable: where the geometric free
    corridor is narrower than this floor, the corridor is clamped to the floor and hugged
    against the wall furthest from the obstacle (maximal avoidance), and the station is
    flagged infeasible — meaning the global line there is NOT guaranteed collision-free
    and the reactive layer must handle it. This never raises; it degrades gracefully.

    Returns (reftrack_mod[N,4], report). Stations not affected by any obstacle are returned
    byte-for-byte unchanged, so an empty obstacle list yields the clean reftrack.
    """
    if params is None:
        params = ModulationParams()

    floor_half = params.min_halfwidth
    if min_total_width is not None:
        floor_half = max(floor_half, 0.5 * min_total_width + 0.02)  # +2cm QP feasibility margin

    reftrack = np.asarray(reftrack, dtype=float)
    N = reftrack.shape[0]
    pts = reftrack[:, :2]
    w_r = reftrack[:, 2].copy()
    w_l = reftrack[:, 3].copy()

    _, normvec, tangent = centerline_frame(reftrack)

    # corridor bounds in the normal coordinate u (u>0 == +normvec == right side)
    keep_lo = -w_l.copy()            # left bound
    keep_hi = w_r.copy()             # right bound
    affected = np.zeros(N, dtype=bool)

    report = ModulationReport(n_stations=N)

    for obs in obstacles:
        p = np.array([obs.x, obs.y])
        rel = p[None, :] - pts                         # (N,2) obstacle relative to each station
        du = np.einsum("ij,ij->i", rel, normvec)       # signed lateral offset (u)
        dt = np.einsum("ij,ij->i", rel, tangent)       # signed longitudinal offset

        infl = obs.r + params.long_taper
        mask = np.abs(dt) < infl
        if not np.any(mask):
            report.obstacle_sides.append("skip")
            continue

        # decide pass side ONCE, at the longitudinally-nearest station, by remaining room
        i0 = int(np.argmin(np.where(mask, np.abs(dt), np.inf)))
        u0 = du[i0]
        room_right = w_r[i0] - (u0 + obs.r)            # space on +side beyond obstacle
        room_left = (u0 - obs.r) - (-w_l[i0])          # space on -side beyond obstacle
        side = "right" if room_right >= room_left else "left"
        report.obstacle_sides.append(side)

        idxs = np.nonzero(mask)[0]
        adt = np.abs(dt[idxs])
        # half-chord of the disk at this longitudinal offset, plus a tapered margin
        base_h = np.sqrt(np.clip(obs.r ** 2 - dt[idxs] ** 2, 0.0, None))
        in_core = adt <= obs.r
        h = np.where(
            in_core,
            base_h + params.obs_margin,
            params.obs_margin * np.clip(1.0 - (adt - obs.r) / max(params.long_taper, 1e-9), 0.0, 1.0),
        )

        # blocked lateral interval of the obstacle disk at each station
        b_lo = du[idxs] - h
        b_hi = du[idxs] + h
        cur_lo = keep_lo[idxs]
        cur_hi = keep_hi[idxs]

        # only act where the blocked interval actually overlaps the current corridor —
        # this auto-ignores far track passes whose |dt| is small but that are laterally
        # metres away (du huge), so no spurious narrowing on the wrong part of the loop.
        overlap = (b_hi > cur_lo) & (b_lo < cur_hi)

        # the two free sub-intervals left/right of the obstacle within the corridor
        left_hi = np.minimum(b_lo, cur_hi)     # left interval  [cur_lo, left_hi]
        right_lo = np.maximum(b_hi, cur_lo)    # right interval [right_lo, cur_hi]
        len_left = left_hi - cur_lo
        len_right = cur_hi - right_lo

        # honour the globally chosen pass side; fall back to the other if it has no room
        eps = 1e-6
        if side == "left":
            choose_left = len_left > eps
        else:
            choose_left = ~(len_right > eps)

        new_lo = np.where(choose_left, cur_lo, right_lo)
        new_hi = np.where(choose_left, left_hi, cur_hi)

        keep_lo[idxs] = np.where(overlap, new_lo, cur_lo)
        keep_hi[idxs] = np.where(overlap, new_hi, cur_hi)
        affected[idxs[overlap]] = True

    # --- tight stations: where the free corridor is narrower than the vehicle floor,
    #     clamp it to a floor-wide window hugging the wall furthest from the obstacle
    #     (best-effort maximal avoidance). Keeps the QP solvable; flag as infeasible.
    width = keep_hi - keep_lo
    tight = affected & (width < 2.0 * floor_half)
    moved_lo = keep_lo - (-w_l)          # how far the left bound was pushed in (>=0)
    moved_hi = w_r - keep_hi             # how far the right bound was pushed in (>=0)
    hug_left = moved_hi >= moved_lo      # obstacle came from the right -> hug left wall
    win_lo = np.where(hug_left, -w_l, np.maximum(w_r - 2.0 * floor_half, -w_l))
    win_hi = np.where(hug_left, np.minimum(-w_l + 2.0 * floor_half, w_r), w_r)
    keep_lo = np.where(tight, win_lo, keep_lo)
    keep_hi = np.where(tight, win_hi, keep_hi)

    # --- reference offset: 0 where the obstacle does not push the centerline out of the
    #     corridor; otherwise the minimal move that keeps a margin `m` on both sides.
    #     The free corridor [keep_lo, keep_hi] is what excludes the obstacle and is
    #     INVARIANT to this offset, so we may smooth the offset freely (kink removal)
    #     without ever compromising obstacle clearance.
    width = keep_hi - keep_lo
    if not recenter:
        # keep the reference line (smooth clean raceline); express exclusion in widths only.
        shift_s = np.zeros(N)
    else:
        m = np.minimum(floor_half, 0.5 * width)
        shift_t = np.where(affected, np.clip(np.zeros(N), keep_lo + m, keep_hi - m), 0.0)
        shift_s = _cyclic_smooth(shift_t, win=7)
        shift_s = np.clip(shift_s, keep_lo, keep_hi)   # never leave the free corridor

    reftrack_mod = reftrack.copy()
    reftrack_mod[:, 0] = pts[:, 0] + normvec[:, 0] * shift_s
    reftrack_mod[:, 1] = pts[:, 1] + normvec[:, 1] * shift_s
    reftrack_mod[:, 2] = keep_hi - shift_s          # w_tr_right (>=0 by the clip when recentered)
    reftrack_mod[:, 3] = shift_s - keep_lo          # w_tr_left  (>=0 by the clip when recentered)

    report.n_affected = int(np.count_nonzero(affected))
    report.n_infeasible = int(np.count_nonzero(tight))
    report.infeasible_s_idx = np.nonzero(tight)[0].tolist()
    aff = np.nonzero(affected)[0]
    if len(aff):
        report.min_halfwidth_seen = float(np.min(np.minimum(reftrack_mod[aff, 2], reftrack_mod[aff, 3])))
    else:
        report.min_halfwidth_seen = float(min(np.min(w_r), np.min(w_l)))
    return reftrack_mod, report


# ======================================================================================
# Re-optimization (thin wrapper over the vendored trajectory_optimizer)
# ======================================================================================
# trajectory_optimizer special-cases these two track names to read /tmp/<name>.csv.
_TMP_TRACK_NAMES = ("map_centerline", "map_centerline_2")


def _write_tmp_reftrack(reftrack: np.ndarray, track_name: str) -> str:
    if track_name not in _TMP_TRACK_NAMES:
        raise ValueError(
            f"track_name must be one of {_TMP_TRACK_NAMES} to use the /tmp mechanism, got {track_name!r}"
        )
    path = os.path.join("/tmp", track_name + ".csv")
    np.savetxt(path, reftrack[:, :4], delimiter=",", fmt="%.6f")
    return path


def reoptimize(
    reftrack: np.ndarray,
    input_path: str,
    curv_opt_type: str = "mincurv_iqp",
    safety_width: float = 0.8,
    track_name: str = "map_centerline",
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Write `reftrack` to /tmp and run the vendored optimizer.

    Returns (traj_race_cl[N,7] = [s,x,y,psi,kappa,vx,ax], bound_r[K,2], bound_l[K,2], est_lap_time).
    """
    from global_racetrajectory_optimization.trajectory_optimizer import trajectory_optimizer
    _write_tmp_reftrack(reftrack, track_name)
    traj, bound_r, bound_l, est_t = trajectory_optimizer(
        input_path=input_path,
        track_name=track_name,
        curv_opt_type=curv_opt_type,
        safety_width=safety_width,
        plot=plot,
    )
    return traj, bound_r, bound_l, est_t


def reoptimize_with_obstacles(
    reftrack: np.ndarray,
    obstacles: List[Obstacle],
    input_path: str,
    params: Optional[ModulationParams] = None,
    safety_width: float = 0.8,
    safety_width_sp: float = 0.8,
    compute_sp: bool = True,
) -> dict:
    """Full core: modulate widths for `obstacles`, then re-optimize the closed loop for the
    main raceline (mincurv_iqp) and — per confirmed design (a) — the overtaking line
    (shortest_path), both from the SAME modulated reftrack.

    The MAIN racing line is ALWAYS min-curvature (`mincurv_iqp`); the shortest-path line is
    ONLY the auxiliary overtaking output and is never used as the main line. If mincurv_iqp
    is infeasible (genuinely tight/blocking obstacle placement) this RAISES — the caller
    (static_reopt_node) then keeps its last valid min-curvature main line (previous
    obstacle-aware, else clean) so a proper racing line is always published.

    Returns a dict with keys: 'reftrack_mod', 'report', 'main' (traj,br,bl,est),
    and optionally 'sp' (traj,br,bl,est).
    """
    # floor the corridor to the widest safety_width in play so both re-opts stay solvable
    min_total = max(safety_width, safety_width_sp) if compute_sp else safety_width
    reftrack_mod, report = modulate_widths(reftrack, obstacles, params, min_total_width=min_total)

    out = {"reftrack_mod": reftrack_mod, "report": report}
    # MAIN line: min-curvature only (never shortest_path). May raise -> node fallback.
    out["main"] = reoptimize(
        reftrack_mod, input_path, "mincurv_iqp", safety_width, "map_centerline"
    )
    if compute_sp:
        out["sp"] = reoptimize(
            reftrack_mod, input_path, "shortest_path", safety_width_sp, "map_centerline"
        )
    return out


# ======================================================================================
# Windowed local re-optimization (fast, ONLINE)
# ======================================================================================
# The whole-track `reoptimize_with_obstacles` runs the vendored mincurv_iqp over ~774 pts
# and takes MINUTES — it is an offline tool and starves the control loop. This is the fast
# online replacement: it reuses the min-curvature QP (`tph.opt_min_curv`) on a SHORT OPEN
# WINDOW around each obstacle and stitches the result into the CLEAN raceline. The QP with
# `closed=False` fixes the window endpoints to the reference and enforces the boundary
# heading (psi_s/psi_e), so the stitched line joins the clean line C0+C1 at the seams.
# Everything outside the windows stays byte-for-byte the clean raceline.

def _wrap_run_indices(mask: np.ndarray) -> List[np.ndarray]:
    """Contiguous index runs where `mask` is True on a CLOSED loop (wrap-aware).

    Returns a list of index arrays (each a contiguous run, possibly wrapping across 0).
    A run that spans the whole loop is returned as a single full-length array.
    """
    N = len(mask)
    if not np.any(mask):
        return []
    if np.all(mask):
        return [np.arange(N)]
    # rotate so index 0 starts a run (mask[0] False, mask[-1..] handled by roll)
    start = int(np.argmax((~np.roll(mask, 1)) & mask))  # first True whose predecessor is False
    idx = (start + np.arange(N)) % N
    m = mask[idx]
    runs = []
    i = 0
    while i < N:
        if m[i]:
            j = i
            while j < N and m[j]:
                j += 1
            runs.append(idx[i:j])
            i = j
        else:
            i += 1
    return runs


def _append_endpoint_normvec(normvec: np.ndarray, coeffs_x: np.ndarray, coeffs_y: np.ndarray) -> np.ndarray:
    """calc_splines returns one normal per spline START (K-1 for a K-point open path); the
    QP wants one per POINT (K). Append the final endpoint normal, computed from the last
    spline's tangent at t=1 with the SAME convention calc_splines uses (normalize([ty,-tx]))."""
    d = np.array([0.0, 1.0, 2.0, 3.0])                     # d/dt of [1,t,t^2,t^3] at t=1
    tx = float(coeffs_x[-1] @ d)
    ty = float(coeffs_y[-1] @ d)
    nv_end = np.array([ty, -tx])
    nv_end /= max(np.hypot(*nv_end), 1e-9)
    return np.vstack([normvec, nv_end])


def _edge_blend(new_arr: np.ndarray, clean_run: np.ndarray, tb_max: int = 15) -> np.ndarray:
    """Cosine-blend `new_arr` toward `clean_run` at both ends (weight 0 at the ends -> exact
    clean, 1 in the middle -> new). Keeps a re-solved quantity C1-continuous with the clean
    profile it's spliced into (no slope jump at the seam)."""
    Kr = len(new_arr)
    tb = int(min(tb_max, Kr // 3))
    wgt = np.ones(Kr)
    if tb >= 2:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, tb)))
        wgt[:tb] = ramp
        wgt[-tb:] = ramp[::-1]
    return wgt * new_arr + (1.0 - wgt) * clean_run


def _densify_run(coarse_xy: np.ndarray, n_out: int) -> np.ndarray:
    """Interpolate a coarse polyline to `n_out` points by cumulative arc length (cubic if
    scipy is available, else linear). Endpoints are preserved."""
    seg = np.hypot(np.diff(coarse_xy[:, 0]), np.diff(coarse_xy[:, 1]))
    u = np.concatenate([[0.0], np.cumsum(seg)])
    if u[-1] < 1e-9:
        return np.repeat(coarse_xy[:1], n_out, axis=0)
    uq = np.linspace(0.0, u[-1], n_out)
    try:
        from scipy.interpolate import CubicSpline
        x = CubicSpline(u, coarse_xy[:, 0])(uq)
        y = CubicSpline(u, coarse_xy[:, 1])(uq)
    except Exception:
        x = np.interp(uq, u, coarse_xy[:, 0])
        y = np.interp(uq, u, coarse_xy[:, 1])
    return np.column_stack([x, y])


def _blas_single_thread():
    """Context manager pinning BLAS to ONE thread for the duration. OpenBLAS spawns a large
    thread pool that, on the tiny matrices of the windowed QP, costs ~1000x the actual compute
    (a 120x120 solve measured 480 ms multi-threaded vs 0.5 ms single-threaded). Returns a
    no-op context if threadpoolctl is unavailable."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def _load_veh_dyn(input_path: str):
    """Load (ggv, ax_max_machines, m_veh, drag_coeff, dyn_model_exp, v_max, curvlim) from a
    config/<version> directory (veh_dyn_info/*.csv + racecar_f110.ini).

    `curvlim` is the vehicle's max trackable |kappa| the OFFLINE optimizer already respects
    (`optim_opts_mincurv` hands it to iqp_handler as kappa_bound). The online path needs it as a
    hard bound too: a hump sharper than curvlim is not steerable, so publishing it makes the
    controller cut the line while the speed plan collapses. Consumers must tolerate a 7-tuple —
    `_offset_lap_time` unpacks `veh[:6]` so the velocity-profile contract is unchanged.
    """
    import re
    vdi = os.path.join(input_path, "veh_dyn_info")
    ggv = np.loadtxt(os.path.join(vdi, "ggv.csv"), delimiter=",", comments="#")
    axm = np.loadtxt(os.path.join(vdi, "ax_max_machines.csv"), delimiter=",", comments="#")
    m_veh, drag, dyn_exp, v_max, curvlim = 3.5, 0.0136, 1.0, 15.0, 1.5
    ini_path = os.path.join(input_path, "racecar_f110.ini")
    if os.path.isfile(ini_path):
        # the .ini holds python-dict blocks with inline '#' comments, so pull individual
        # keys by regex (robust to the comments that break a full-dict literal_eval)
        txt = open(ini_path).read()

        def _key(name, default):
            m = re.search(r'"%s"\s*:\s*([-+0-9.eE]+)' % name, txt)
            return float(m.group(1)) if m else default
        m_veh = _key("mass", m_veh)
        drag = _key("dragcoeff", drag)
        v_max = _key("v_max", v_max)
        dyn_exp = _key("dyn_model_exp", dyn_exp)
        curvlim = _key("curvlim", curvlim)
    return ggv, axm, m_veh, drag, dyn_exp, v_max, curvlim


def build_offset_profile(clean_xy: np.ndarray, s_loop: np.ndarray, track_len: float,
                         nvec_rl: np.ndarray, apexes: List[Tuple[float, float]],
                         clean_vx: Optional[np.ndarray],
                         reach_time: float, reach_min: float, reach_max: float,
                         hi_inc: Optional[np.ndarray] = None,
                         lo_inc: Optional[np.ndarray] = None,
                         entry_scale: float = 1.0,
                         exit_scale: float = 1.0,
                         fit_tol: float = _FIT_TOL_DEFAULT,
                         clean_kappa: Optional[np.ndarray] = None,
                         curvlim: float = 0.0,
                         obstacles: Optional[List[Optional[Tuple[float, float, float]]]] = None,
                         obs_margin: float = 0.0
                         ) -> Tuple[np.ndarray, int, float, list, list]:
    """Lateral offset d(s) on the CLOSED clean loop that PRESERVES each recorded reactive apex
    but re-grows long, gentle entry/exit ramps — the "keep the apex, press the secondary apexes"
    reshape. `apexes` are map-frame (x, y) apex points captured from the reactive spline.

    Per apex: project onto the raceline normal at the nearest station -> signed offset d*; grow a
    speed-scaled reach R = clip(reach_time * local_speed, reach_min, reach_max) SYMMETRIC on both
    sides. A single C2 quintic (via BPoly, zero d'/d'' at the ramp ends, d'=d''-free peak at the
    apex) realises one wide gentle hump; apexes whose ramps overlap are woven into ONE multi-knot
    BPoly (raceline -> apex1 -> apex2 -> raceline) so the line stays C2.

    NB: the ramp deliberately is NOT shortened near corners. The offset is a FIXED-shape decay to
    zero (it cannot re-cut a corner however far it reaches — smootherstep decays to <6% of the peak
    by 80% of R), so a LONGER ramp through a corner adds only ~0.08 1/m of curvature, whereas a
    shortened ramp forces a sharp S-shaped merge (curvature swing +0.6/-0.6) — the exact "unnecessary
    undulation" at the merge. Longer is always gentler here.

    With `curvlim > 0` (and `clean_kappa`) the hump is additionally bound by a HARD curvature limit.
    Two steps, because the cheap estimate and the truth disagree on this track:
      * SIZING — the quintic's peak |d''| is closed-form (_KAPPA_QUINTIC_C * |A| / r^2), so the
        reach that would fit an additive curvature budget needs no search:
        r_min = sqrt(_KAPPA_QUINTIC_C * |A| / budget). Used only to pick how far to GROW the reach.
      * VERDICT — measured on the actual offset GEOMETRY (`_kappa_peak`). The additive
        `kappa_clean + d''` model is a linearisation that reads ~45% low here, so it cannot be the
        thing a safety limit is checked against; see `_kappa_peak`.
    A hump over the bar is grown while the corridor allows, then has its amplitude shaved, and is
    dropped if that would take it under the acceptance floor — publishing a line the car cannot
    steer is never the answer (nothing checked this before: `curvlim` was respected only by the
    offline IQP).

    AMPLITUDE. With `obstacles` the target offset is derived from the BOX (d_obs + side*(r +
    obs_margin)), not replayed from the recorded apex — see the knot loop. The apex only chooses
    the side and stands as evidence that the side is drivable.

    ACCEPTANCE FLOOR. With `obstacles` (map-frame (x, y, r) index-aligned with `apexes`, entries
    may be None) and `obs_margin > 0`, a hump is accepted only when the geometry it would lay
    passes the obstacle's EDGE by at least `obs_margin` — the same hypot scan the node reports on
    the published line, so what is enforced here is exactly what is measured there. A short hump is
    retried ONCE at the amplitude that would reach the floor (re-fitted to the corridor, allowed to
    keep the wider reach), and dropped with reason "clearance" if the corridor cannot hold it. The
    target amplitude above is the LATERAL offset that would clear the box; the floor is a 2-D
    distance on a curving line, so the two differ slightly and the retry is the normal path, not an
    exception. This replaces judging the hump by amplitude RATIO (>= _APEX_KEEP_FRAC of the
    recorded apex), which is a proxy for clearance and a loose one: 0.90 of a 0.55 m apex leaves
    0.345 m off a 0.15 m box where the SM and the reactive planner both expect the line to be built
    to obs_margin. Without obstacle geometry (offline sweeps) the ratio test remains the only
    available proxy and is still used.

    Returns (d_global[N], n_apex_used, entry_kappa, dropped, laid).
    `dropped` lists the apexes that could not be laid at the acceptance floor (all-or-nothing; each
    entry {"xy": (x, y), "want": d*, "fit": best_feasible_d, "reason":
    "corridor"|"curvature"|"clearance", plus "clear"/"need" for the clearance reason}), so the
    caller can report honestly and leave those obstacles to the reactive layer. `laid` mirrors it
    for the humps that WERE laid ({"xy", "want", "laid", "r_in", "r_out", "r_req", "kappa_peak",
    "clear"}) so the node can log why a hump ended up as sharp as it did and what it really clears.
    Wrap is handled by cutting the profile in the largest apex-free gap so no hump straddles s=0.
    Never raises; degrades to zeros on any failure."""
    N = len(s_loop)
    d_global = np.zeros(N)
    if not apexes:
        return d_global, 0, 0.0, [], []
    try:
        from scipy.interpolate import BPoly
    except Exception:
        return d_global, 0, 0.0, [], []

    # --- project apexes -> (s*, d*, R_in, R_out, xy, obs); drop negligible offsets ------------
    # `obs` (map-frame x, y, r or None) rides along so the amplitude and the acceptance floor
    # below can both be derived from the box the hump exists to clear.
    knots = []
    for k_i, (xa, ya) in enumerate(apexes):
        i = int(np.argmin(np.hypot(clean_xy[:, 0] - xa, clean_xy[:, 1] - ya)))
        d_star = float((np.array([xa, ya], float) - clean_xy[i]) @ nvec_rl[i])
        if abs(d_star) < 0.03:
            continue                                    # apex on the raceline -> no avoidance
        ob = obstacles[k_i] if (obstacles is not None and k_i < len(obstacles)) else None
        # AMPLITUDE FROM THE OBSTACLE, not from the recorded apex. The apex is what the REACTIVE
        # planner drove: keep-out (width_car/2 + safety_margin) + apex_bulge, i.e. its own design
        # clearance plus a deliberate extra swing. Replaying it makes the global line a copy of the
        # local one -- same offset, same lap-time cost, no reason to swap. What the global line
        # actually owes the obstacle is r + obs_margin from its centre, which on the shipped
        # numbers is 0.50 m against the reactive 0.55 m: a genuinely tighter line, and the whole
        # point of re-optimizing at all.
        #
        # The recorded apex keeps two jobs it is uniquely good at: it says which SIDE the obstacle
        # was passed on (a decision that needs the corridor and the grid, which this function does
        # not have), and it is standing evidence that a path on that side is drivable.
        d_target = d_star
        if ob is not None and obs_margin > 0.0:
            j = int(np.argmin(np.hypot(clean_xy[:, 0] - ob[0], clean_xy[:, 1] - ob[1])))
            d_obs = float((np.array([ob[0], ob[1]], float) - clean_xy[j]) @ nvec_rl[j])
            side = 1.0 if d_star >= d_obs else -1.0     # the side the reactive layer proved
            need = float(ob[2]) + float(obs_margin)
            # It is a CONSTRAINT, not a set-point: if the raceline already stands off the box by
            # `need` on that side there is nothing to lay, however far the reactive path swung.
            if side * (0.0 - d_obs) >= need - 1e-9:
                continue
            d_target = d_obs + side * need
            if abs(d_target) < 0.03:
                continue
        v = float(clean_vx[i]) if clean_vx is not None else 3.0
        R = float(np.clip(reach_time * v, reach_min, reach_max))
        R = min(R, 0.45 * track_len)                    # never span more than ~half the loop
        knots.append((float(s_loop[i]), d_target, R, R, xa, ya, ob))  # symmetric ramps
    if not knots:
        return d_global, 0, 0.0, [], []

    # --- cut the loop in the largest gap between apex centres (seam falls on d=0) -------------
    knots.sort(key=lambda k: k[0])
    centers = np.array([k[0] for k in knots])
    if len(centers) == 1:
        s_cut = (centers[0] + 0.5 * track_len) % track_len
    else:
        ext = np.concatenate([centers, [centers[0] + track_len]])
        gaps = np.diff(ext)
        g = int(np.argmax(gaps))
        s_cut = (centers[g] + 0.5 * gaps[g]) % track_len

    # --- BPoly breakpoints in the cut-linear coordinate u = (s - s_cut) mod L -----------------
    # Between two apexes whose ramps OVERLAP we weave straight apex->apex (no return-to-0 knot);
    # otherwise each hump opens from and closes back to the raceline. `zero` = [d,d',d''] all 0.
    zero = [0.0, 0.0, 0.0]
    kn_u = sorted((((c - s_cut) % track_len, d, ri, ro, xa, ya, ob)
                   for (c, d, ri, ro, xa, ya, ob) in knots), key=lambda k: k[0])
    dropped: List[dict] = []
    laid: List[dict] = []
    # --- FIT each hump to the corridor (shrink reach, NEVER the clearance) BEFORE laying it ----
    # Without this the downstream element-wise clip does the shaping and combs the hump; see
    # _fit_hump_to_corridor. Fitting keeps every hump analytic (exactly one extremum, C2).
    # ALL-OR-NOTHING: an amplitude below _APEX_KEEP_FRAC of the recorded apex does not clear
    # the obstacle -> drop the hump entirely and report it (reactive layer keeps handling it).
    if hi_inc is not None and lo_inc is not None:
        u_all = (s_loop - s_cut) % track_len
        # Floor well BELOW reach_min: on a tight track (ifac is 1.39 m wide, and the min-curvature
        # line leaves zero headroom over 59/355 stations) the corridor often admits only a ~1-2 m
        # ramp — comparable to the reactive spliner's own return_len (2.5 m). A floor at reach_min
        # would make the fit give up and hand an infeasible hump to the clip, i.e. the comb again.
        floor_r = max(0.5, min(1.0, 0.05 * track_len))
        hi_a = np.asarray(hi_inc, float)
        lo_a = np.asarray(lo_inc, float)
        kap_a = (np.abs(np.asarray(clean_kappa, float))
                 if clean_kappa is not None else np.zeros(N))
        # the clean line's own GEOMETRIC curvature — the reference the laid hump is judged against
        try:
            kap_geo = _menger_kappa(clean_xy)
        except Exception:
            kap_geo = kap_a
        # locality budget for a GROWN reach: a hump must stay a local detour (see _HUMP_SPAN_FRAC)
        r_grow_cap = min(max(1.0, 0.5 * _HUMP_SPAN_FRAC * track_len), 0.45 * track_len)

        def _fits(u_c, r, amp):
            v = _hump_values(u_all, u_c, amp, r, r)
            return (v is not None and bool(np.all(v <= hi_a + fit_tol))
                    and bool(np.all(v >= lo_a - fit_tol)))

        def _kappa_budget(u_c, r):
            """curvlim minus the clean line's own curvature over the hump span — the headroom the
            hump's d'' may use. Only used to SIZE the reach (r_min), so it is floored: at a raceline
            apex kappa_clean already sits at ~curvlim and a zero budget would demand r=inf. The
            floor must never authorise a violation — that is what the geometric `_kappa_peak` vs
            `_kappa_allow` verdict below is for, and it is why the floor may stay this loose."""
            m = (u_all >= u_c - r) & (u_all <= u_c + r)
            kc = float(np.max(kap_a[m])) if np.any(m) else 0.0
            return max(curvlim - kc, _KAPPA_BUDGET_FLOOR_FRAC * curvlim)

        def _kappa_peak(u_c, amp, r_i, r_o):
            """Max |kappa| of the GEOMETRY a candidate hump would lay, over its own span.

            Measured from the actual offset points, NOT from the `kappa_clean + d''` additive model
            the rest of this file publishes. That model is the small-(d, d', kappa) linearisation of
            the Frenet offset curvature, and on this track it is not valid: laying a 0.55 m hump at
            reach 2.1 m next to the ifac chicane (kappa_clean 1.45) gives an additive estimate of
            1.19 while the real geometry measures 2.11 — the offset curve's `1 - d*kappa_r` and
            d'^2 terms are first-order here (d*kappa_r ~ 0.8, d' up to 0.49), not negligible. A
            curvature LIMIT checked against the linearisation would pass lines the car cannot steer,
            so the verdict has to be geometric. Menger is the right estimator: it reproduces the
            stored clean kappa to 0.0067 1/m without differentiating a derived heading."""
            v = _hump_values(u_all, u_c, amp, r_i, r_o)
            if v is None:
                return float("inf")
            m = (u_all >= u_c - r_i) & (u_all <= u_c + r_o)
            if not np.any(m):
                return 0.0
            k = _menger_kappa(clean_xy + v[:, None] * nvec_rl)
            return float(np.max(np.abs(k[m])))

        need_clear = float(obs_margin) if obstacles is not None else 0.0

        def _clearance(u_c, amp, r_i, r_o, ob):
            """Min distance from the geometry ONE hump would lay to the obstacle's EDGE [m].

            The same measurement static_reopt_node makes on the published line
            (`_line_clearances`): a hypot scan over every station, minus the obstacle radius.
            Scanned over the WHOLE loop on purpose — the stations that decide the verdict are
            usually the ones just OUTSIDE the ramp, where the line has already returned to the
            raceline and is therefore back beside the box. A span-limited scan would report the
            apex's clearance and miss exactly the case a too-short ramp creates.

            +inf when this apex has no obstacle attached (nothing to enforce); -inf on degenerate
            knots so a hump that cannot even be evaluated can never be accepted."""
            if ob is None:
                return float("inf")
            v = _hump_values(u_all, u_c, amp, r_i, r_o)
            if v is None:
                return float("-inf")
            pts = clean_xy + v[:, None] * nvec_rl
            return float(np.min(np.hypot(pts[:, 0] - ob[0], pts[:, 1] - ob[1]))) - float(ob[2])

        def _accepts(d_now, d_want, ob, u_c, r_i, r_o):
            """Is this hump good enough to lay? (ok, clearance_or_None).

            With obstacle geometry the bar is ABSOLUTE: the laid line must clear the box edge by
            obs_margin. Without it, fall back to the amplitude-ratio proxy (offline sweeps, and
            the `global` re-opt method, have no per-apex box to measure against)."""
            if need_clear > 0.0 and ob is not None:
                gap = _clearance(u_c, d_now, r_i, r_o, ob)
                return gap >= need_clear - _CLEAR_EPS, gap
            return abs(d_now) >= max(0.03, _APEX_KEEP_FRAC * abs(d_want)), None

        def _kappa_allow(u_c, r):
            """The bar the laid hump must clear. `curvlim` normally, but never stricter than the
            clean raceline's OWN geometric curvature there: if the clean line already exceeds
            curvlim at this corner that is a property of the raceline, and dropping the apex would
            not fix it."""
            m = (u_all >= u_c - r) & (u_all <= u_c + r)
            kc = float(np.max(np.abs(kap_geo[m]))) if np.any(m) else 0.0
            return max(curvlim, kc)

        fitted = []
        for (u, d, ri, ro, xa, ya, ob) in kn_u:
            r_req = max(ri, ro)
            d_f, r_f = _fit_hump_to_corridor(u_all, u, d, r_req, track_len,
                                             hi_a, lo_a, floor_r, fit_tol=fit_tol)
            ok, gap = _accepts(d_f, d, ob, u, r_f, r_f)
            if not ok and gap is not None:
                # ONE retry, at the amplitude that WOULD reach the floor. The obstacle-derived
                # target is the LATERAL offset that clears the box; the floor is a 2-D distance, so
                # on a curving line the first attempt can land just under it and asking for the
                # deficit recovers it. The reach is asked for at r_req again — a wider hump is
                # gentler AND keeps the line away from the box for longer, so it can only help.
                d_want = math.copysign(abs(d_f) + (need_clear - gap), d)
                d_r, r_r = _fit_hump_to_corridor(u_all, u, d_want, r_req, track_len,
                                                 hi_a, lo_a, floor_r, fit_tol=fit_tol)
                ok_r, gap_r = _accepts(d_r, d, ob, u, r_r, r_r)
                if ok_r:
                    ok, d_f, r_f, gap = True, d_r, r_r, gap_r
            if not ok:
                rec = {"xy": (float(xa), float(ya)), "want": float(d), "fit": float(d_f),
                       "reason": "clearance" if gap is not None else "corridor"}
                if gap is not None:
                    rec["clear"], rec["need"] = float(gap), need_clear
                dropped.append(rec)
                continue
            # --- HARD curvature limit -------------------------------------------------------
            # r_min is closed-form, so this costs no search. Widen first (free in curvature terms,
            # ~0.08 m of extra arc), and only shave the proven clearance if the corridor refuses.
            if curvlim > 0.0 and abs(d_f) > 1e-9:
                for _ in range(3):                      # budget depends on the span -> re-converge
                    budget = _kappa_budget(u, r_f)
                    r_min = math.sqrt(_KAPPA_QUINTIC_C * abs(d_f) / budget)
                    if r_min <= r_f + 1e-6:
                        break
                    # The corridor fit already returned the LARGEST feasible reach <= r_req, so
                    # growing is only possible when it did not bite (r_f == r_req).
                    r_top = min(r_min, r_grow_cap)
                    if r_f >= r_req - 1e-9 and r_top > r_f:
                        if _fits(u, r_top, d_f):
                            r_f = r_top
                        else:                            # largest feasible reach in (r_f, r_top]
                            lo_g, hi_g = r_f, r_top
                            for _ in range(20):
                                mid = 0.5 * (lo_g + hi_g)
                                if _fits(u, mid, d_f):
                                    lo_g = mid
                                else:
                                    hi_g = mid
                            r_f = lo_g
                    else:
                        break
                # VERDICT on the actual laid curvature (the floor above may have let r_f through).
                allow = _kappa_allow(u, r_f)
                if _kappa_peak(u, d_f, r_f, r_f) > allow:
                    # Shave the amplitude to whatever the curvature bar admits at this reach.
                    # Bisection, not the closed form: with the signed sum the peak is not exactly
                    # C*|A|/r^2 near a corner, and this lands on the true boundary either way.
                    lo_k, hi_k = 0.0, abs(d_f)
                    for _ in range(20):
                        mid = 0.5 * (lo_k + hi_k)
                        if _kappa_peak(u, math.copysign(mid, d_f), r_f, r_f) <= allow:
                            lo_k = mid
                        else:
                            hi_k = mid
                    d_f = math.copysign(lo_k, d_f)
                    # No retry here: the amplitude was shaved BECAUSE the curvature bar was hit,
                    # so asking for more of it at this reach only fails the same bar again. The
                    # reach was already grown as far as the corridor allows above.
                    ok, gap = _accepts(d_f, d, ob, u, r_f, r_f)
                    if not ok:
                        rec = {"xy": (float(xa), float(ya)), "want": float(d),
                               "fit": float(d_f), "reason": "curvature"}
                        if gap is not None:
                            rec["clear"], rec["need"] = float(gap), need_clear
                        dropped.append(rec)
                        continue
            # ENTRY/EXIT ramps may be stretched independently of the lap-time-optimal reach:
            # a longer ramp cuts the merge-zone curvature (the S-shaped inflection where the
            # hump joins the raceline — it cannot be removed, only made shallower).
            # LOCALITY CAP. An avoidance must stay a LOCAL detour: the lap-time search alone does
            # not bound it (stretching the entry costs <30 ms, so it always "wins"), and on ifac an
            # 18 m entry + 6 m exit put 65% of the 35 m lap off the racing line and pushed the hump
            # across s=0. Budget the whole hump to a fraction of the lap.
            span_cap = max(2.0 * r_f, _HUMP_SPAN_FRAC * track_len)

            def _stretch(scale, r_fixed, stretch_entry):
                """Longest feasible stretched ramp <= scale*r_f within the span budget."""
                if scale <= 1.0:
                    return r_f
                r_best = r_f
                r_try = min(r_f * scale, max(r_f, span_cap - r_fixed))
                for _ in range(12):
                    ri, ro = (r_try, r_fixed) if stretch_entry else (r_fixed, r_try)
                    v = _hump_values(u_all, u, d_f, ri, ro)
                    if v is not None and np.all(v <= hi_a + fit_tol) and np.all(v >= lo_a - fit_tol):
                        r_best = r_try
                        break
                    r_try = 0.5 * (r_try + r_f)
                    if r_try <= r_f + 1e-3:
                        break
                return r_best

            r_in = _stretch(entry_scale, r_f, stretch_entry=True)
            r_out = _stretch(exit_scale, r_in, stretch_entry=False)
            fitted.append((u, d_f, r_in, r_out))
            laid.append({"xy": (float(xa), float(ya)), "want": float(d), "laid": float(d_f),
                         "r_in": float(r_in), "r_out": float(r_out), "r_req": float(r_req),
                         "kappa_peak": 0.0,        # filled from the laid profile below
                         "clear": float("nan"),    # ditto — measured on the WOVEN profile
                         "_obs": ob,               # so the join below can re-measure it
                         "_u": float(u)})          # join key: `fitted` gets sorted, `laid` does not
        kn_u = sorted(fitted)
        if not kn_u:
            return d_global, 0, 0.0, dropped, laid
        verify_ctx = (hi_a, lo_a, kap_geo, floor_r)
    else:
        kn_u = [(u, d, ri, ro) for (u, d, ri, ro, _xa, _ya, _ob) in kn_u]   # no corridor: keep all
        verify_ctx = None

    u_stn = (s_loop - s_cut) % track_len

    def _weave(knots):
        """Lay `knots` as ONE C2 polynomial: raceline -> apex [-> apex ...] -> raceline.

        Humps whose ramps overlap are woven straight apex-to-apex (no return-to-zero knot between
        them), which is what keeps the line C2 through a slalom -- and also what makes the result
        something no per-hump check ever looked at. Returns None on degenerate breakpoints."""
        n_ap = len(knots)
        breaks = [0.0]
        bd = [list(zero)]                               # seam: raceline, C2
        for idx, (u, d, ri, ro) in enumerate(knots):
            at_raceline = (bd[-1] == zero)
            entry = u - ri
            if at_raceline and entry > breaks[-1] + 2e-3:
                breaks.append(entry)                    # open from the raceline before this hump
                bd.append(list(zero))
            u_adj = max(u, breaks[-1] + 1e-3)           # keep breakpoints strictly increasing
            breaks.append(u_adj)
            bd.append([float(d), 0.0, 0.0])             # apex knot (peak, slope 0)
            # close back to the raceline UNLESS the next apex's ramp overlaps this exit (-> weave)
            exit_ = u + ro
            next_entry = (knots[idx + 1][0] - knots[idx + 1][2]) if idx + 1 < n_ap else float("inf")
            if next_entry > exit_ + 2e-3:
                e = min(exit_, track_len - 1e-3)
                if e > breaks[-1] + 1e-3:
                    breaks.append(e)
                    bd.append(list(zero))
        if breaks[-1] < track_len - 1e-6:
            breaks.append(track_len)                    # seam close (raceline, C2 across s=0)
            bd.append(list(zero))
        try:
            return np.asarray(BPoly.from_derivatives(np.asarray(breaks), bd)(u_stn), dtype=float)
        except Exception:
            return None

    def _weave_violations(dg, ctx):
        """Stations of the WOVEN profile that break the corridor or the curvature bar.

        Every check up to here judged ONE hump in isolation. Where two ramps overlap the weave
        produces a shape neither of them was tested as, and the only thing standing behind that was
        the clip + moving-average fallback downstream -- which this file documents as turning an
        analytic 1-extremum hump into a 3-5 extremum comb. So re-ask both questions of what is
        actually laid, with the SAME tolerance the per-hump fit used and the SAME geometric
        curvature the verdict and the published speed cap use."""
        hi_v, lo_v, kap_ref, _fl = ctx
        bad = (dg > hi_v + fit_tol) | (dg < lo_v - fit_tol)
        if curvlim > 0.0:
            try:
                k_laid = np.abs(_menger_kappa(clean_xy + dg[:, None] * nvec_rl))
                # Same bar as the per-hump verdict: curvlim, but never stricter than the clean
                # line's own geometry -- a corner the raceline already over-curves is not the
                # hump's fault and dropping it would not fix it.
                bad |= k_laid > np.maximum(curvlim, kap_ref) + 1e-6
            except Exception:
                pass
        return bad

    if verify_ctx is not None:
        for _pass in range(_WEAVE_MAX_PASSES):
            d_global = _weave(kn_u)
            if d_global is None:
                return np.zeros(N), 0, 0.0, dropped, laid
            bad = _weave_violations(d_global, verify_ctx)
            if not bad.any():
                break
            # Shrink the reach of every hump whose span covers a violation, JOINTLY -- the failure
            # is an interaction, so the humps that produce it have to give ground together.
            # Shrinking is the right lever even for a curvature violation: it separates the ramps,
            # and once they no longer overlap each hump is again the single hump that already
            # passed its own corridor and curvature tests.
            floor_v = verify_ctx[3]
            u_bad = u_stn[bad]
            shrunk = []
            changed_any = False
            for (u, d, ri, ro) in kn_u:
                touches = np.any((u_bad >= u - ri - 1e-9) & (u_bad <= u + ro + 1e-9))
                if touches and max(ri, ro) > floor_v + 1e-6:
                    ri = max(floor_v, ri * _WEAVE_SHRINK)
                    ro = max(floor_v, ro * _WEAVE_SHRINK)
                    changed_any = True
                shrunk.append((u, d, ri, ro))
            if not changed_any:
                # Nothing left to give. Fail honestly instead of handing the profile to the clip:
                # the reactive layer keeps these obstacles, which is strictly better than a comb.
                for rec in laid:
                    dropped.append({"xy": rec["xy"], "want": rec["want"], "fit": rec["laid"],
                                    "reason": "weave"})
                return np.zeros(N), 0, 0.0, dropped, []
            kn_u = sorted(shrunk)
        else:
            for rec in laid:
                dropped.append({"xy": rec["xy"], "want": rec["want"], "fit": rec["laid"],
                                "reason": "weave"})
            return np.zeros(N), 0, 0.0, dropped, []
    else:
        d_global = _weave(kn_u)
        if d_global is None:
            return np.zeros(N), 0, 0.0, dropped, laid
    # RAMP curvature: max |d''| over each hump's entry AND exit ramps — the merge-zone
    # inflections the driver feels joining/leaving the avoidance. It is the tiebreak the
    # reach/stretch search minimises once the lap time is settled. The same pass fills each
    # `laid` entry's kappa_peak — the GEOMETRIC max |kappa| of the laid line over its own span, so
    # the caller logs what was really laid instead of inferring it from the reach.
    ent_k = 0.0
    by_u = {round(e["_u"], 6): e for e in laid if "_u" in e}
    xy_laid = clean_xy + d_global[:, None] * nvec_rl
    try:
        kap_laid = np.abs(_menger_kappa(xy_laid))
    except Exception:
        kap_laid = None
    for (u, d, ri, ro) in kn_u:
        m = (u_stn >= u - ri) & (u_stn <= u + ro)
        if int(np.count_nonzero(m)) < 3:
            continue
        seg = d_global[m]
        h = float(np.median(np.abs(np.diff(u_stn[m])))) or 1.0
        ent_k = max(ent_k, float(np.abs(np.diff(seg, 2)).max() / max(h * h, 1e-9)))
        rec = by_u.get(round(float(u), 6))
        if rec is not None and kap_laid is not None:
            # Geometric, matching the verdict in _kappa_peak. The additive `kappa_clean + d''` the
            # node publishes reads ~45% low here (1.19 vs 2.11 on the ifac chicane hump), so a log
            # built on it would report a healthy line the car in fact cannot steer.
            rec["kappa_peak"] = float(np.max(kap_laid[m]))
    # Clearance of the FINAL WOVEN profile per laid hump. The acceptance floor above judged each
    # hump in isolation, but where two ramps overlap the BPoly weaves them and what gets published
    # is not the single hump that was tested. Re-measure on the profile that is actually returned,
    # so the number the node logs (and vetoes on) describes the line the car will follow.
    for rec in laid:
        ob = rec.pop("_obs", None)
        if ob is None:
            continue
        rec["clear"] = float(np.min(np.hypot(xy_laid[:, 0] - ob[0],
                                             xy_laid[:, 1] - ob[1]))) - float(ob[2])
    return d_global, len(kn_u), ent_k, dropped, laid


def _cap_speed_to_published_curvature(traj: np.ndarray, ggv, axm) -> None:
    """Make the speed plan consistent with the PUBLISHED geometry (in place).

    The velocity profile is solved on the ANALYTIC hump curvature (kappa_clean + alpha''),
    but the published line is sharper after the corridor fit + uniform resample — the same vx
    then demands more lateral acceleration than the vehicle has (measured on ifac: 6.7 m/s^2
    implied vs ggv ay_max 4.5 on a two-hump line). The controller cannot track a plan beyond
    the friction budget, which reads as a sharp tracking-quality collapse on the swapped line.

    Cap vx by ay_max over the TRUE geometry of the published points, then re-run wrap-aware
    backward-decel / forward-accel passes so the capped profile stays reachable. Cheap (two sweeps
    over ~355 points).

    The cap must NOT read traj[:, 4]: that field is deliberately smoothed for the controller's L1
    lookahead (`_republish_kappa` -> Menger + _cyclic_smooth(win=5)), and smoothing it again here
    made the friction cap see a curvature the line does not have. Measured on a 3-obstacle ifac
    line: published |kappa| max 1.36 while the real geometry was 1.70, so the published plan
    demanded 5.20 m/s^2 of lateral accel against a 4.5 budget — 15% over, on the exact stations
    where tracking matters. Take the max of the published field and the RAW Menger curvature of the
    published points, so the cap can never under-report the geometry. Raw, not smoothed: measured on
    the ifac single-hump line, smoothing the cap input costs the friction budget without buying lap
    time — win=5 leaves max ay at 4.77 and win=3 at 4.60, while raw lands on exactly 4.50 (the clean
    line's own value) for +0.004 s of lap time and 0.02 m/s of min speed."""
    ay_cap, a_brake, a_accel = 4.5, 5.0, 3.0
    try:
        if np.ndim(ggv) > 1 and np.shape(ggv)[1] > 2:
            ay_cap = float(np.min(ggv[:, 2]))
            a_brake = float(np.max(ggv[:, 1]))
        if np.ndim(axm) > 1 and np.shape(axm)[1] > 1:
            a_accel = float(np.min(axm[:, 1]))
    except Exception:
        pass
    kap = np.abs(traj[:, 4])
    try:
        kap = np.maximum(kap, np.abs(_menger_kappa(traj[:, 1:3])))
    except Exception:
        pass                                     # fall back to the published field alone
    vx = np.minimum(traj[:, 5], np.sqrt(ay_cap / np.maximum(kap, 1e-3)))
    seg = np.roll(traj[:, 1:3], -1, axis=0) - traj[:, 1:3]
    el = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-6)   # el[i] = dist i -> i+1 (closed)
    n = len(vx)
    for i in range(2 * n - 1, -1, -1):                       # backward decel, wraps the seam
        j, k = i % n, (i + 1) % n
        vx[j] = min(vx[j], float(np.sqrt(vx[k] ** 2 + 2.0 * a_brake * el[j])))
    for i in range(2 * n):                                   # forward accel, wraps the seam
        j, k = i % n, (i - 1) % n
        vx[j] = min(vx[j], float(np.sqrt(vx[k] ** 2 + 2.0 * a_accel * el[k])))
    traj[:, 5] = vx


def _resample_uniform(traj: np.ndarray, d_right: np.ndarray, d_left: np.ndarray,
                      target_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a CLOSED trajectory [s,x,y,psi,kappa,vx,ax] to UNIFORM arc-length spacing over
    EXACTLY `target_n` unique points (+1 duplicated closing point). Laying the avoidance offset on
    the raceline COMPRESSES the point spacing on the inner side of curves (the clean line's uniform
    0.1 m becomes ~0.08-0.13 m); a downstream spline fit through unevenly spaced waypoints (frenet
    converter, controller) can overshoot and WIGGLE between the sparse points. Uniform resampling
    removes that — the SHAPE is preserved (points are only redistributed along the same polyline),
    so clearance/apex are unchanged.

    The COUNT is PINNED to the clean line's, NOT derived from the length. Deriving it (M = L/ds)
    makes the detour's extra path length change the waypoint count (ifac: +0.05 m of detour already
    turns 355 into 356), and the count is a CONTRACT the rest of the stack relies on:
      - sector_tuner.scale_points() indexes its cached scaled array with the NEW array's length and
        has no length guard -> IndexError -> the node dies -> /global_waypoints_scaled STOPS and
        every consumer (state_machine, ot_interpolator) keeps the OLD line forever. That is the
        "new line published but the car still follows the old path" failure.
      - maps/<map>/speed_scaling.yaml and ot_sectors.yaml hard-code sector bounds as INDICES
        (ifac: end: 355), so a changed count silently shifts every sector boundary.
    Spacing therefore becomes L_new/target_n (~1.4% off the clean 0.0998 m for a 0.5 m detour) —
    uniformity, which is what fixed the wiggle, is fully preserved.

    x,y,kappa,vx,ax,d_right,d_left are linearly interpolated along the arc; psi + s are recomputed
    from the resampled xy. Returns (traj_M, d_right_M, d_left_M) with a duplicated closing point."""
    xy = traj[:, 1:3]
    dup = np.allclose(xy[-1], xy[0])
    xyu = xy[:-1] if dup else xy
    n = len(xyu)
    if n < 4 or int(target_n) < 4:
        return traj, d_right, d_left
    seg = np.vstack([np.diff(xyu, axis=0), xyu[0] - xyu[-1]])   # closed-loop segments
    el = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(el)])                 # s[-1] = L (arc length)
    L = float(s[-1])
    M = max(int(target_n), 16)
    new_s = np.linspace(0.0, L, M, endpoint=False)

    def _interp(a):                                            # periodic linear interp onto new_s
        au = a[:-1] if dup else a
        return np.interp(new_s, s, np.concatenate([au, au[:1]]))

    new_xy = np.column_stack([_interp(traj[:, 1]), _interp(traj[:, 2])])
    kap = _interp(traj[:, 4]); vx = _interp(traj[:, 5]); ax = _interp(traj[:, 6])
    dr = _interp(d_right); dl = _interp(d_left)
    # close the loop (duplicate the start point) to match the clean-bundle convention
    new_xy = np.vstack([new_xy, new_xy[:1]])
    kap = np.append(kap, kap[0]); vx = np.append(vx, vx[0]); ax = np.append(ax, ax[0])
    dr = np.append(dr, dr[0]); dl = np.append(dl, dl[0])
    # recompute psi + s on the uniformly-spaced closed line
    segm = np.roll(new_xy, -1, axis=0) - new_xy
    elm = np.hypot(segm[:, 0], segm[:, 1])
    psi_m, _ = tph.calc_head_curv_num.calc_head_curv_num(path=new_xy, el_lengths=elm, is_closed=True)
    s_m = np.concatenate([[0.0], np.cumsum(elm)])[:len(new_xy)]
    traj_m = np.column_stack([s_m, new_xy[:, 0], new_xy[:, 1], psi_m, kap, vx, ax])
    return traj_m, dr, dl


def _menger_kappa(xy_closed: np.ndarray) -> np.ndarray:
    """Signed curvature of a CLOSED polyline from circumscribed circles (Menger). Unlike
    calc_head_curv_num this does not differentiate a numerically-derived heading, so it does not
    amplify the raceline's micro-noise: on the ifac clean line it reproduces the stored
    kappa_radpm to 0.0067 1/m. Returns one value per input point (closing duplicate included)."""
    xy = np.asarray(xy_closed, float)
    dup = len(xy) > 2 and np.allclose(xy[-1], xy[0])
    u = xy[:-1] if dup else xy
    a, b, c = np.roll(u, 1, axis=0), u, np.roll(u, -1, axis=0)
    v1, v2 = b - a, c - b
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    denom = (np.hypot(*(b - a).T) * np.hypot(*(c - b).T) * np.hypot(*(c - a).T))
    k = 2.0 * cross / np.maximum(denom, 1e-12)
    return np.append(k, k[0]) if dup else k


def _republish_kappa(traj: np.ndarray, clean_xy: np.ndarray, clean_kappa: Optional[np.ndarray],
                     dev_tol: float = 0.02) -> np.ndarray:
    """Curvature CONSISTENT with the published points.

    kappa was built as kappa_clean + alpha'' on the PRE-resample stations and then edge-blended, so
    after the uniform resample the published kappa no longer describes the published geometry —
    measured error up to 1.56 1/m (the clean line's own peak curvature is 1.19). The controller sets
    its L1 lookahead from mean|kappa| a few points ahead, so those spikes make the lookahead jump and
    the car wander even though the waypoints themselves are smooth to <3 mm.

    Fix: take the curvature straight from the published points, then restore the EXACT clean value
    wherever the line has rejoined the clean raceline, so outside the avoidance the controller still
    sees byte-identical clean curvature."""
    k_geo = _cyclic_smooth(_menger_kappa(traj[:, 1:3]), win=5)
    if clean_kappa is None:
        return k_geo
    ck = np.asarray(clean_kappa, float)
    cx = np.asarray(clean_xy, float)
    # nearest clean station for every published point (positions, not indices — the resample and
    # the detour both shift arc length, so index/s matching would compare different places)
    d2 = ((traj[:, 1][:, None] - cx[None, :, 0]) ** 2 +
          (traj[:, 2][:, None] - cx[None, :, 1]) ** 2)
    j = np.argmin(d2, axis=1)
    on_clean = np.sqrt(d2[np.arange(len(j)), j]) <= dev_tol
    out = k_geo.copy()
    out[on_clean] = ck[j[on_clean]]
    return out


def _offset_lap_time(d_global: np.ndarray, clean_xy: np.ndarray, nvec_rl: np.ndarray,
                     el_cl: np.ndarray, clean_kappa: Optional[np.ndarray],
                     clean_vx: Optional[np.ndarray], lo_inc: np.ndarray, hi_inc: np.ndarray,
                     veh, N: int) -> float:
    """Estimated LAP TIME for a candidate offset profile — the objective the reach search minimises.

    Mirrors the geometry/velocity math of _reopt_local_window_impl but skips psi, bounds and the
    resample (none of which affect the time), so a whole search over ~10 candidate reaches costs
    about as much as one full solve."""
    if clean_vx is None:
        return float("inf")
    try:
        ggv, axm, m_veh, drag, dyn_exp, v_max = veh[:6]     # veh may carry curvlim as [6]
        alpha = np.clip(d_global, lo_inc, hi_inc)
        stitch = clean_xy + alpha[:, None] * nvec_rl
        if N > 1 and np.allclose(clean_xy[-1], clean_xy[0]):
            stitch[-1] = stitch[0]
        sg = np.roll(stitch, -1, axis=0) - stitch
        elc = np.hypot(sg[:, 0], sg[:, 1])
        h2 = np.maximum((0.5 * (elc + np.roll(elc, 1))) ** 2, 1e-9)
        add = (np.roll(alpha, -1) - 2.0 * alpha + np.roll(alpha, 1)) / h2
        if N > 3:
            h0 = 0.5 * (elc[0] + elc[N - 2])
            add[0] = (alpha[1] - 2.0 * alpha[0] + alpha[N - 2]) / max(h0 ** 2, 1e-9)
            add[N - 1] = add[0]
        kappa_full = (np.asarray(clean_kappa, float) if clean_kappa is not None
                      else np.zeros(N)) + add
        dev = np.hypot(stitch[:, 0] - clean_xy[:, 0], stitch[:, 1] - clean_xy[:, 1])
        sig = dev > 0.02
        vx = np.asarray(clean_vx, float).copy()
        if np.any(sig):
            # same braking-distance margin as the full solve, so the search ranks candidates on the
            # profile that will actually be published
            ds_stn = float(np.sum(elc)) / max(N - 1, 1)
            a_brake = 5.0
            if ggv is not None and np.ndim(ggv) > 1 and np.shape(ggv)[1] > 1:
                a_brake = float(np.max(ggv[:, 1]))
            v_ref = float(np.max(clean_vx))
            mg = int(np.clip(np.ceil(v_ref ** 2 / (2.0 * max(a_brake, 0.5)) / max(ds_stn, 1e-3)),
                             10, max(10, N // 3)))
            mask = np.zeros(N, dtype=bool)
            for run in _wrap_run_indices(sig):
                mask[(run[0] - mg + np.arange(len(run) + 2 * mg)) % N] = True
            for run in _wrap_run_indices(mask):
                vn = tph.calc_vel_profile.calc_vel_profile(
                    ax_max_machines=axm, kappa=kappa_full[run], el_lengths=elc[run[:-1]],
                    closed=False, drag_coeff=drag, m_veh=m_veh, ggv=ggv, dyn_model_exp=dyn_exp,
                    v_max=v_max, v_start=float(clean_vx[run[0]]), v_end=float(clean_vx[run[-1]]))
                vx[run] = _edge_blend(np.minimum(vn, clean_vx[run]), clean_vx[run])
        return float(np.sum(elc / np.maximum(vx, 1e-3)))
    except Exception:
        return float("inf")


def reoptimize_local_window(*args, **kwargs) -> dict:
    """Fast ONLINE obstacle-aware raceline (BLAS pinned to one thread for the QP). See
    `_reopt_local_window_impl` for the full signature and behaviour."""
    with _blas_single_thread():
        return _reopt_local_window_impl(*args, **kwargs)


def _reopt_local_window_impl(
    clean_xy: np.ndarray,
    clean_dr: np.ndarray,
    clean_dl: np.ndarray,
    reftrack: np.ndarray,
    apexes: List[Tuple[float, float]],
    input_path: str,
    params: Optional[ModulationParams] = None,
    w_veh: float = 0.30,
    clean_vx: Optional[np.ndarray] = None,
    wall_margin: float = 0.12,
    reach_time: float = 1.2,
    reach_min: float = 4.0,
    reach_max: float = 10.0,
    clean_kappa: Optional[np.ndarray] = None,
    fit_tol: float = _FIT_TOL_DEFAULT,
    apex_obstacles: Optional[List[Optional[Tuple[float, float, float]]]] = None,
) -> dict:
    """Fast ONLINE obstacle-aware raceline: reshape the REACTIVE avoidance spline into a global
    line. Each `apex` (map-frame (x,y) captured from the reactive spliner on the exploration lap)
    is PRESERVED, but its entry/exit are re-grown as long gentle ramps merging into the clean
    raceline (the "keep the apex, press the secondary apexes" reshape — see build_offset_profile).

    The offset d(s) is a C2 quintic hump per apex (woven where they overlap); it is laid on the
    clean raceline (stitch_xy = clean + d*nvec). Curvature is analytic (`kappa_clean + d''`, since
    calc_head_curv_num inflates it ~5-8x); vx/kappa outside the arc stay exactly the clean values
    (localized; controller lookahead/speed unchanged there). The reactive apex already cleared the
    obstacle (corridor+grid+box+curvature checked), so the global line uses the SAME gap.

    Inputs:
      clean_xy   [N,2] clean raceline points (CLOSED loop; a duplicated closing point is handled).
      clean_dr/dl[N]   distance from each raceline point to the RIGHT/LEFT track bound.
      reftrack   [M,4] centerline `[x,y,w_tr_right,w_tr_left]` (only to reconstruct bound polylines).
      apexes           recorded reactive-spline apex points (map-frame (x,y)); empty -> clean line.
      apex_obstacles   OPTIONAL map-frame (x, y, r) per apex, index-aligned (None entries allowed).
                       Supplying them turns `params.obs_margin` into a HARD acceptance floor: a
                       hump is laid only if the geometry it produces clears that obstacle's edge
                       by obs_margin, else it is retried wider once and then honestly dropped for
                       the reactive layer. Without them the amplitude-ratio proxy is used instead
                       (see build_offset_profile) — offline sweeps have no box to measure against.
      input_path       config/<version> dir (veh_dyn_info + racecar_f110.ini) for the vel profile.
      w_veh            vehicle width [m]; with wall_margin sets the corridor the offset is clamped to.
      reach_time/min/max  ramp reach R = clip(reach_time * local_speed, reach_min, reach_max).
                       Bigger R = gentler ramps (secondary apexes pressed down) reaching the corners.

    Returns the same dict shape as `reoptimize_with_obstacles` MINUS 'sp'
    (keys 'reftrack_mod', 'report', 'main'=(traj[N,7], bound_r, bound_l, est_lap_time), plus
    'd_right'/'d_left', 'n_windows', 'n_failed'). No apexes -> the clean raceline. Never raises.
    """
    if params is None:
        params = ModulationParams()
    clean_xy = np.asarray(clean_xy, dtype=float)[:, :2]
    N = clean_xy.shape[0]
    clean_dr = np.asarray(clean_dr, float)
    clean_dl = np.asarray(clean_dl, float)
    clean_vx_arr = np.asarray(clean_vx, float) if clean_vx is not None else None

    # dense raceline normal (toward +right); lays the offset on the clean line + reconstructs bounds.
    rl_ref = np.column_stack([clean_xy[:, 0], clean_xy[:, 1], clean_dr, clean_dl])
    _, nvec_rl, _ = centerline_frame(rl_ref)
    # The closed raceline has a DUPLICATED closing point (xy[-1]==xy[0], el=0); centerline_frame's
    # normal there is a 0/0 garbage vector. It is the SAME physical point as idx 0, so copy it.
    if N > 1 and np.allclose(clean_xy[-1], clean_xy[0]):
        nvec_rl[-1] = nvec_rl[0]

    # arc length of the closed clean loop
    seg = np.roll(clean_xy, -1, axis=0) - clean_xy
    el_cl = np.hypot(seg[:, 0], seg[:, 1])
    s_loop = np.concatenate([[0.0], np.cumsum(el_cl)])[:N]
    track_len = float(np.sum(el_cl))

    # --- feasible lateral corridor per station (0 always included so a zero-offset station is
    # never pushed off the clean line at a narrow spot). SMOOTH the bounds first: the optimizer
    # corridor widths are bumpy at the 0.1 m scale near tight corners, and forcing a decaying ramp
    # to track a ±cm-jittery wall makes the merge zigzag. The wall_margin buffer keeps it safe.
    hi_off = _cyclic_smooth(clean_dr - 0.5 * w_veh - wall_margin, win=7)
    lo_off = _cyclic_smooth(-(clean_dl - 0.5 * w_veh - wall_margin), win=7)
    lo_inc = np.minimum(lo_off, 0.0)
    hi_inc = np.maximum(hi_off, 0.0)

    ggv, axm, m_veh, drag, dyn_exp, v_max, curvlim = _load_veh_dyn(input_path)
    veh = (ggv, axm, m_veh, drag, dyn_exp, v_max)

    # --- OFFSET d(s): apex-preserving reshape of the reactive spline, with the ramp REACH chosen
    # to MINIMISE LAP TIME (the actual objective) instead of by a speed heuristic ---------------
    # The old rule R = clip(reach_time * v, reach_min, reach_max) came from "a wider arc is gentler
    # and carries more speed". That is false on a short tight track: on ifac (35.3 m lap) R = 10 m
    # spreads one hump over 57% of the lap, and measured against the clean line it costs 1.3-2.4 s
    # versus 0.10-0.31 s at the optimum — i.e. roughly TWICE the time the purely reactive line
    # loses. The optimum reach varies 1.5-4 m per apex and cannot be predicted from local speed, so
    # search it. Asymmetric entry/exit reaches were measured to add only 0-0.03 s, not worth 8x the
    # evaluations. reach_min/reach_max are now the SEARCH BOUNDS.
    # Candidate reaches, additionally bounded by the locality budget: a symmetric hump of reach r
    # spans 2r, so r must fit inside _HUMP_SPAN_FRAC of the lap however large reach_max is set.
    r_cap = max(1.0, 0.5 * _HUMP_SPAN_FRAC * track_len)
    # The lower bound must respect the locality cap too: with reach_min > r_cap (a long-track
    # default like 4.0 m on the 35 m ifac loop, r_cap 3.5) the filter came back EMPTY and the
    # fallback candidate (7.0 m) blew straight past the cap — the intended 1.5-4 m lap-time
    # search never ran and every hump was ~2x wider than optimal.
    r_lo = min(reach_min, r_cap)
    cand_r = [r for r in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
              if r_lo - 1e-9 <= r <= min(reach_max, r_cap) + 1e-9]
    if not cand_r:
        cand_r = [float(np.clip(0.5 * (reach_min + reach_max), r_lo, min(reach_max, r_cap)))]
    def _try(r, e_scale, x_scale=1.0):
        dg, nn, ek, drp, lay = build_offset_profile(
            clean_xy, s_loop, track_len, nvec_rl, apexes,
            clean_vx_arr, 0.0, r, r, hi_inc=hi_inc, lo_inc=lo_inc,
            entry_scale=e_scale, exit_scale=x_scale,
            fit_tol=fit_tol, clean_kappa=clean_kappa, curvlim=curvlim,
            obstacles=apex_obstacles, obs_margin=params.obs_margin)
        if nn == 0:
            return dg, 0, float("inf"), ek, drp, lay
        return dg, nn, _offset_lap_time(dg, clean_xy, nvec_rl, el_cl, clean_kappa,
                                        clean_vx_arr, lo_inc, hi_inc, veh, N), ek, drp, lay

    # STAGE 1 — symmetric reach, minimise lap time.
    d_global, n_solved, best_est, best_ek, best_r = None, 0, float("inf"), 0.0, cand_r[0]
    apex_dropped: List[dict] = []
    apex_laid: List[dict] = []
    for r_try in cand_r:
        d_try, n_try, est_try, ek_try, drp_try, lay_try = _try(r_try, 1.0)
        if n_try == 0:
            if d_global is None:
                d_global, n_solved, apex_dropped, apex_laid = d_try, 0, drp_try, lay_try
            continue
        if est_try < best_est:
            d_global, n_solved, best_est, best_ek, best_r = d_try, n_try, est_try, ek_try, r_try
            apex_dropped, apex_laid = drp_try, lay_try
    # STAGE 2 — stretch the ENTRY ramp (gradual turn-in), then STAGE 3 — stretch the EXIT ramp
    # too: the merge-back inflection (curvature sign flip where the hump rejoins the raceline)
    # cannot be removed, but a longer return ramp makes it shallow instead of a visible S-kink.
    # Each stage keeps the longest stretch that costs at most `tol` of lap time and lowers the
    # ramp curvature tiebreak.
    if n_solved:
        tol = 0.03                                        # [s] lap-time budget for a softer turn-in
        best_e = 1.0
        for e_scale in (1.5, 2.0, 3.0):
            d_try, n_try, est_try, ek_try, drp_try, lay_try = _try(best_r, e_scale)
            if n_try and est_try <= best_est + tol and ek_try < best_ek:
                d_global, n_solved, best_ek, best_e = d_try, n_try, ek_try, e_scale
                apex_dropped, apex_laid = drp_try, lay_try
        tol_exit = 0.05                                   # [s] merge smoothness is worth a bit more
        for x_scale in (1.5, 2.0, 3.0):
            d_try, n_try, est_try, ek_try, drp_try, lay_try = _try(best_r, best_e, x_scale)
            if n_try and est_try <= best_est + tol_exit and ek_try < best_ek:
                d_global, n_solved, best_ek = d_try, n_try, ek_try
                apex_dropped, apex_laid = drp_try, lay_try
    if d_global is None:
        d_global, n_solved = np.zeros(N), 0
    n_failed = 0                                          # apexes with no offset are simply absent
    # The profile was fitted per hump AND verified as woven inside build_offset_profile, both
    # against this corridor with `fit_tol`. So the clip here is a pure guard: it may take back the
    # tolerance the fit was allowed to spend (up to fit_tol), and nothing more.
    #
    # Anything beyond that used to fall through to clip + moving-average smoothing, described in
    # this file as turning an analytic 1-extremum hump into a "3-5 extremum comb" -- i.e. the
    # documented generator of the multi-hump undulation this line is judged on. A fallback that
    # produces a shape the design explicitly rejects is not a fallback, so it is gone: a bite
    # bigger than the tolerance means the verification upstream is wrong, and the honest answer is
    # to lay nothing and leave the obstacles to the reactive layer.
    alpha_full = np.clip(d_global, lo_inc, hi_inc)
    if n_solved:
        bite = float(np.max(np.abs(alpha_full - d_global)))
        if bite > fit_tol + 1e-6:
            print(f"[static_reopt_core] corridor clip bit {bite * 1e3:.1f} mm > fit_tol "
                  f"{fit_tol * 1e3:.1f} mm on the woven profile — refusing to smooth it into a "
                  f"comb; laying the clean line and leaving the obstacles to the reactive layer")
            d_global, alpha_full, n_solved = np.zeros(N), np.zeros(N), 0
            apex_laid = []

    # curvature contribution of the arc = alpha'' (2nd deriv wrt arc length), from the FINAL
    # smoothed+clamped offset: post-smoothing this is the real laid shape (no fake clamp-kink
    # spikes left), so the vel profile sees honest curvature through clamped wedges too.
    elm = np.roll(el_cl, 1)
    h2 = np.maximum((0.5 * (el_cl + elm)) ** 2, 1e-9)
    alpha_dd = (np.roll(alpha_full, -1) - 2.0 * alpha_full + np.roll(alpha_full, 1)) / h2
    # The DUPLICATED closing point breaks the 2nd-difference stencil at idx 0 and N-1 (it uses the
    # duplicate instead of the real neighbour N-2) -> a fake curvature spike at start/finish.
    if N > 3:
        h0 = 0.5 * (el_cl[0] + el_cl[N - 2])
        alpha_dd[0] = (alpha_full[1] - 2.0 * alpha_full[0] + alpha_full[N - 2]) / max(h0 ** 2, 1e-9)
        alpha_dd[N - 1] = alpha_dd[0]

    stitch_xy = clean_xy + alpha_full[:, None] * nvec_rl if n_solved else clean_xy.copy()
    if N > 1 and np.allclose(clean_xy[-1], clean_xy[0]):
        stitch_xy[-1] = stitch_xy[0]                     # keep the closed-loop closing point exact

    # minimal report (no width modulation now); reftrack_mod kept for the return dict shape
    report = ModulationReport(n_stations=N, n_affected=n_solved)
    rl_mod = rl_ref

    # --- recompute geometry over the stitched CLOSED loop ---------------------------------
    seg = np.roll(stitch_xy, -1, axis=0) - stitch_xy
    el_cl = np.hypot(seg[:, 0], seg[:, 1])
    psi_full, _ = tph.calc_head_curv_num.calc_head_curv_num(
        path=stitch_xy, el_lengths=el_cl, is_closed=True)
    # CURVATURE analytically (kappa ≈ kappa_clean + alpha''), NOT via calc_head_curv_num which
    # amplifies the clean normal's micro-noise ~5-8x and would crush the speed profile. alpha_dd
    # was computed above from the SMOOTH pre-clamp arc, so this is a clean gentle curvature.
    kappa_clean = np.asarray(clean_kappa, float) if clean_kappa is not None else np.zeros(N)
    kappa_full = kappa_clean + alpha_dd
    s_full = np.concatenate([[0.0], np.cumsum(el_cl)])[:N]

    # Region to RE-SOLVE: where the line actually detours (>5 cm), extended by a margin. Both
    # vx AND the published kappa are recomputed ONLY here (blended to clean at the edges) and
    # otherwise held at the clean values — so outside the obstacle the controller sees exactly
    # the clean speed AND the clean curvature. The L1 lookahead uses mean|kappa| a few points
    # ahead, so a whole-loop re-derived (noisier) kappa would jitter the lookahead everywhere.
    # Re-solve region = the whole ARC (any deviation), so the slow-in / carry / fast-out speed is
    # computed across the entire gentle arc against the clean corner-context speeds at its ends —
    # not just the sharp part.
    dev_full = np.hypot(stitch_xy[:, 0] - clean_xy[:, 0], stitch_xy[:, 1] - clean_xy[:, 1])
    sig = dev_full > 0.02
    vx_runs: List[np.ndarray] = []
    if np.any(sig):
        # The margin must cover the BRAKING DISTANCE into the arc, not a token 1 m. The arc adds
        # curvature, so the car has to be slower when it ARRIVES — braking therefore starts well
        # upstream of any geometric deviation. With a fixed 10-station (1 m) margin that whole
        # deceleration got crammed into 1 m and _edge_blend then forced the profile back up to the
        # clean speed at the run edge: an impossible decel demand exactly at the junction, and up to
        # 1.84 m/s of speed loss at stations whose GEOMETRY is already back on the clean line.
        # Sizing the run by v^2/(2a) instead lets the re-solved profile meet the clean one on its
        # own, so the blend at the edge becomes a no-op.
        ds_stn = track_len / max(N - 1, 1)
        a_brake = 5.0
        try:
            if ggv is not None and np.ndim(ggv) > 1 and np.shape(ggv)[1] > 1:
                a_brake = float(np.max(ggv[:, 1]))
        except Exception:
            pass
        v_ref = float(np.max(clean_vx_arr)) if clean_vx_arr is not None else 10.0
        margin = int(np.clip(np.ceil(v_ref ** 2 / (2.0 * max(a_brake, 0.5)) / max(ds_stn, 1e-3)),
                             10, max(10, N // 3)))
        vx_mask = np.zeros(N, dtype=bool)
        for run in _wrap_run_indices(sig):
            vx_mask[(run[0] - margin + np.arange(len(run) + 2 * margin)) % N] = True
        vx_runs = _wrap_run_indices(vx_mask)

    # --- velocity ---
    if clean_vx is not None:
        vx = np.asarray(clean_vx, float).copy()
        for run in vx_runs:
            vx_new = tph.calc_vel_profile.calc_vel_profile(
                ax_max_machines=axm, kappa=kappa_full[run], el_lengths=el_cl[run[:-1]], closed=False,
                drag_coeff=drag, m_veh=m_veh, ggv=ggv, dyn_model_exp=dyn_exp, v_max=v_max,
                v_start=float(clean_vx[run[0]]), v_end=float(clean_vx[run[-1]]))
            # never exceed the tuned racing speed; a gentle arc simply REACHES it (no dip = the win)
            vx_new = np.minimum(vx_new, clean_vx[run])
            vx[run] = _edge_blend(vx_new, clean_vx[run])
    else:
        vx = tph.calc_vel_profile.calc_vel_profile(
            ax_max_machines=axm, kappa=kappa_full, el_lengths=el_cl, closed=True,
            drag_coeff=drag, m_veh=m_veh, ggv=ggv, dyn_model_exp=dyn_exp, v_max=v_max)

    # --- published curvature (drives the L1 lookahead) ---
    if clean_kappa is not None:
        kap_pub = np.asarray(clean_kappa, float).copy()
        for run in vx_runs:
            kap_pub[run] = _edge_blend(kappa_full[run], clean_kappa[run])
    else:
        kap_pub = kappa_full

    # longitudinal accel from the closed vx profile (a = (v_{i+1}^2 - v_i^2)/(2 ds))
    vx_next = np.roll(vx, -1)
    ax = (vx_next ** 2 - vx ** 2) / (2.0 * np.maximum(el_cl, 1e-6))

    traj = np.column_stack([s_full, stitch_xy[:, 0], stitch_xy[:, 1],
                            psi_full, kap_pub, vx, ax])
    est = float(np.sum(el_cl / np.maximum(vx, 1e-3)))

    # --- exact d_right/d_left for the stitched line, from the clean widths + lateral shift --
    # The stitched line is the clean raceline shifted laterally by `offset` (signed, +toward
    # the right/+normal) inside each window, 0 elsewhere. So d_right = clean_dr - offset and
    # d_left = clean_dl + offset EXACTLY (no polyline min-distance approximation, and it sidesteps
    # dist_to_bounds' column handling). Reconstructed bound polylines are also returned for
    # callers that still want them (e.g. legacy dist_to_bounds).
    offset = np.einsum("ij,ij->i", stitch_xy - clean_xy, nvec_rl)
    d_right = np.maximum(np.asarray(clean_dr, float) - offset, 0.0)
    d_left = np.maximum(np.asarray(clean_dl, float) + offset, 0.0)
    bound_r = clean_xy + np.asarray(clean_dr, float)[:, None] * nvec_rl
    bound_l = clean_xy - np.asarray(clean_dl, float)[:, None] * nvec_rl

    # UNIFORM-spacing resample: the offset compresses the point spacing on inner curves, and a
    # downstream spline through unevenly spaced waypoints can wiggle. Only when an arc was laid
    # (n_solved) — a clean line is already uniform. Shape is preserved; the COUNT is pinned to the
    # clean line's (N-1 unique + 1 closing point) — see _resample_uniform: a length-derived count
    # kills sector_tuner (IndexError -> /global_waypoints_scaled stops -> the car keeps following
    # the OLD line) and shifts the index-based sector bounds in speed_scaling/ot_sectors.yaml.
    if n_solved:
        traj, d_right, d_left = _resample_uniform(traj, d_right, d_left, N - 1)
        # Curvature must describe the points we actually publish (the controller's lookahead reads
        # it); recompute it from the final geometry and restore the exact clean value where the
        # line has rejoined the raceline. See _republish_kappa.
        traj[:, 4] = _republish_kappa(traj, clean_xy, clean_kappa)
        # FULL-LAP velocity re-solve on the PUBLISHED geometry/curvature, replacing the windowed
        # runs + edge-blend patchwork as the final speed source: one closed-loop profile has no
        # blend seams (the per-run profiles left small steps at run edges and between humps — a
        # user-visible rough speed plan across the swapped line). Ceiling = the tuned clean
        # line's top speed, so the re-solve can never overspeed the racing setup.
        _sg = np.roll(traj[:, 1:3], -1, axis=0) - traj[:, 1:3]
        _el = np.maximum(np.hypot(_sg[:, 0], _sg[:, 1]), 1e-6)
        try:
            vx_full = tph.calc_vel_profile.calc_vel_profile(
                ax_max_machines=axm, kappa=traj[:-1, 4], el_lengths=_el[:-1], closed=True,
                drag_coeff=drag, m_veh=m_veh, ggv=ggv, dyn_model_exp=dyn_exp, v_max=v_max)
            ceil = float(np.max(clean_vx_arr)) if clean_vx_arr is not None else v_max
            traj[:-1, 5] = np.minimum(vx_full, ceil)
            traj[-1, 5] = traj[0, 5]
        except Exception:
            pass                                     # keep the windowed profile on any failure
        # ... and the SPEED must stay feasible over the published curvature in every case
        # (also re-runs the wrap-aware decel/accel sweeps) — see _cap_speed_to_published_curvature.
        _cap_speed_to_published_curvature(traj, ggv, axm)
        # ax likewise has to describe the PUBLISHED vx over the PUBLISHED spacing. It was computed
        # on the pre-resample grid and then linearly interpolated, leaving it inconsistent by up to
        # 0.85 m/s^2 (the clean line's own residual is 0.001) — a wrong feed-forward for any
        # consumer that differentiates the speed plan.
        _sg = np.roll(traj[:, 1:3], -1, axis=0) - traj[:, 1:3]
        _el = np.hypot(_sg[:, 0], _sg[:, 1])
        traj[:, 6] = (np.roll(traj[:, 5], -1) ** 2 - traj[:, 5] ** 2) / (2.0 * np.maximum(_el, 1e-6))
        if len(traj) > 2:
            traj[-1, 6] = traj[0, 6]                 # closing duplicate: el=0 there
        est = float(np.sum(_el[:-1] / np.maximum(traj[:-1, 5], 1e-3)))   # est on the FINAL profile

    return {"reftrack_mod": rl_mod, "report": report,
            "main": (traj, bound_r, bound_l, est), "d_right": d_right, "d_left": d_left,
            "n_windows": n_solved, "n_failed": n_failed, "apex_dropped": apex_dropped,
            "apex_laid": apex_laid, "curvlim": float(curvlim)}


# ======================================================================================
# Distance-to-bounds + waypoint building (replicated from gb_optimizer; nodes untouched)
# ======================================================================================
def dist_to_bounds(traj_xy: np.ndarray, bound_r: np.ndarray, bound_l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Min distance from each trajectory point to the right/left bound polylines."""
    traj_xy = np.asarray(traj_xy)[:, :2] if traj_xy.shape[1] > 2 else np.asarray(traj_xy)
    d_right = np.empty(len(traj_xy))
    d_left = np.empty(len(traj_xy))
    for i, p in enumerate(traj_xy):
        d_right[i] = np.min(np.hypot(bound_r[:, 0] - p[0], bound_r[:, 1] - p[1]))
        d_left[i] = np.min(np.hypot(bound_l[:, 0] - p[0], bound_l[:, 1] - p[1]))
    return d_right, d_left


def conv_psi(psi: float) -> float:
    """tph heading (0 at +y) -> ROS heading (0 at +x), wrapped to (-pi, pi]."""
    new_psi = psi + np.pi / 2.0
    if new_psi > np.pi:
        new_psi -= 2.0 * np.pi
    return new_psi


def build_wpnts(traj: np.ndarray, d_right: np.ndarray, d_left: np.ndarray, second_traj: bool = False):
    """Build (WpntArray, MarkerArray) from an optimizer trajectory [s,x,y,psi,kappa,vx,ax].

    ROS message imports are lazy so the numeric core can be used without a ROS session.
    """
    from f110_msgs.msg import Wpnt, WpntArray
    from visualization_msgs.msg import Marker, MarkerArray

    max_vx = float(np.max(traj[:, 5])) if len(traj) else 1.0
    max_vx = max_vx if max_vx > 1e-6 else 1.0

    wpnts = WpntArray()
    markers = MarkerArray()
    for i, pnt in enumerate(traj):
        w = Wpnt()
        w.id = i
        w.s_m = float(pnt[0])
        w.x_m = float(pnt[1])
        w.y_m = float(pnt[2])
        w.d_right = float(d_right[i])
        w.d_left = float(d_left[i])
        w.psi_rad = float(conv_psi(pnt[3]))
        w.kappa_radpm = float(pnt[4])
        w.vx_mps = float(pnt[5])
        w.ax_mps2 = float(pnt[6])
        wpnts.wpnts.append(w)

        m = Marker()
        m.header.frame_id = "map"
        m.type = Marker.CYLINDER
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = w.vx_mps / max_vx
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0 if second_traj else 0.0
        m.id = i
        m.pose.position.x = float(pnt[1])
        m.pose.position.y = float(pnt[2])
        m.pose.position.z = w.vx_mps / max_vx / 2.0
        m.pose.orientation.w = 1.0
        markers.markers.append(m)

    return wpnts, markers
