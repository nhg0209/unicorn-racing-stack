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

from global_racetrajectory_optimization.trajectory_optimizer import trajectory_optimizer  # noqa: E402
import trajectory_planning_helpers as tph  # noqa: E402


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
    """Moving-average smoothing on a closed (cyclic) 1-D signal, length preserved."""
    win = max(1, int(win))
    if win == 1 or len(a) < win:
        return a.astype(float)
    k = np.ones(win) / win
    pad = win // 2
    ext = np.concatenate([a[-pad:], a, a[:win - pad - 1]])
    return np.convolve(ext, k, mode="valid")


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
) -> Tuple[np.ndarray, ModulationReport]:
    """Narrow the drivable corridor around each obstacle and recenter the reference line
    onto the chosen free side.

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
    m = np.minimum(floor_half, 0.5 * width)
    shift_t = np.where(affected, np.clip(np.zeros(N), keep_lo + m, keep_hi - m), 0.0)
    shift_s = _cyclic_smooth(shift_t, win=7)
    shift_s = np.clip(shift_s, keep_lo, keep_hi)   # never leave the free corridor

    reftrack_mod = reftrack.copy()
    reftrack_mod[:, 0] = pts[:, 0] + normvec[:, 0] * shift_s
    reftrack_mod[:, 1] = pts[:, 1] + normvec[:, 1] * shift_s
    reftrack_mod[:, 2] = keep_hi - shift_s          # w_tr_right (>=0 by the clip above)
    reftrack_mod[:, 3] = shift_s - keep_lo          # w_tr_left  (>=0 by the clip above)

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
