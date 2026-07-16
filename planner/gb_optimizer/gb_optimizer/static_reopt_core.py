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
    """Load (ggv, ax_max_machines, m_veh, drag_coeff, dyn_model_exp, v_max) from a
    config/<version> directory (veh_dyn_info/*.csv + racecar_f110.ini)."""
    import re
    vdi = os.path.join(input_path, "veh_dyn_info")
    ggv = np.loadtxt(os.path.join(vdi, "ggv.csv"), delimiter=",", comments="#")
    axm = np.loadtxt(os.path.join(vdi, "ax_max_machines.csv"), delimiter=",", comments="#")
    m_veh, drag, dyn_exp, v_max = 3.5, 0.0136, 1.0, 15.0
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
    return ggv, axm, m_veh, drag, dyn_exp, v_max


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
    obstacles: List[Obstacle],
    input_path: str,
    params: Optional[ModulationParams] = None,
    w_veh: float = 0.30,
    clean_vx: Optional[np.ndarray] = None,
    wall_margin: float = 0.12,
    reach_time: float = 1.2,
    reach_min: float = 4.0,
    reach_max: float = 10.0,
    clean_kappa: Optional[np.ndarray] = None,
) -> dict:
    """Fast ONLINE obstacle-aware raceline: lay a smooth WIDE avoidance arc (raised smootherstep)
    on the clean raceline per obstacle — gentler & faster than the reactive spline's tight bump.

    The clearance/side/feasibility come from `modulate_widths`; the SHAPE is analytic (not a QP):
    a smootherstep bump peaking at the required clearance over a speed-scaled half-width R. Zero
    1st+2nd derivative at both ends -> C2 seam + low curvature (carries speed). Curvature is taken
    analytically as `kappa_clean + alpha''` (calc_head_curv_num inflates it ~5-8x). vx/kappa outside
    the arc stay exactly the clean values (localized; controller lookahead/speed unchanged there).

    Inputs:
      clean_xy   [N,2] clean raceline points (CLOSED loop; a duplicated closing point is handled).
      clean_dr/dl[N]   distance from each raceline point to the RIGHT/LEFT track bound.
      reftrack   [M,4] centerline `[x,y,w_tr_right,w_tr_left]` (only to reconstruct bound polylines).
      obstacles        confirmed static obstacles (map-frame disks).
      input_path       config/<version> dir (veh_dyn_info + racecar_f110.ini) for the vel profile.
      w_veh            vehicle width [m] for the corridor/clearance floor. Obstacle CLEARANCE is
                       obs.r + obs_margin (in `params`); wall_margin keeps the arc off the walls.
      reach_time/min/max  arc half-width R = clip(reach_time * local_speed, reach_min, reach_max).
                       Bigger R = gentler, faster arc reaching toward the adjacent corners.

    Returns the same dict shape as `reoptimize_with_obstacles` MINUS 'sp'
    (keys 'reftrack_mod', 'report', 'main'=(traj[N,7], bound_r, bound_l, est_lap_time), plus
    'd_right'/'d_left', 'n_windows', 'n_failed'). No obstacles -> the clean raceline. Never raises;
    an obstacle whose corridor is blocked keeps the clean line there (reactive layer covers it).
    """
    if params is None:
        params = ModulationParams()
    clean_xy = np.asarray(clean_xy, dtype=float)[:, :2]
    N = clean_xy.shape[0]

    # --- corridor around the RACELINE: reference = clean line, widths = dist to bounds ----
    rl_ref = np.column_stack([clean_xy[:, 0], clean_xy[:, 1],
                              np.asarray(clean_dr, float), np.asarray(clean_dl, float)])
    # recenter=False: keep the smooth clean raceline as the QP reference; the obstacle
    # exclusion lives in the widths (a recentered reference spikes at the coarse QP grid).
    # Floor the corridor above the QP vehicle width so tight stations keep a small usable
    # band; the +2*wall_margin lets us shave wall_margin off EACH side below (keeping the
    # line off the walls) while the post-shave floor stays w_veh+0.05 (still solvable).
    corridor_floor = w_veh + 0.05 + 2.0 * wall_margin
    rl_mod, report = modulate_widths(rl_ref, obstacles, params,
                                     min_total_width=corridor_floor, recenter=False)
    # dense raceline normal (toward +right); used both to lay the smooth detour on the clean
    # line and to reconstruct the track bounds. Computed once.
    _, nvec_rl, _ = centerline_frame(rl_ref)
    # The closed raceline has a DUPLICATED closing point (xy[-1]==xy[0], el=0); centerline_frame's
    # normal there is a 0/0 garbage vector -> if an obstacle arc reaches start/finish, that one
    # station scatters far off the line (and its d_right/d_left/bounds go garbage). It is the SAME
    # physical point as idx 0, so copy the good normal.
    if N > 1 and np.allclose(clean_xy[-1], clean_xy[0]):
        nvec_rl[-1] = nvec_rl[0]

    # --- DEVIATION as a smooth WIDE arc (analytic), NOT a min-curv QP bump -----------------
    # The windowed min-curv QP concentrates the deviation into a ~2-3 m bump right at the obstacle
    # (as sharp as the reactive spline) and leaves curvature ripple + a seam spike -> it neither
    # steers gently nor carries speed. Instead lay the deviation directly as a raised SMOOTHERSTEP
    # bump per obstacle: peak = the clearance the corridor requires, spread over a speed-scaled
    # half-width R that reaches toward the adjacent corners. smootherstep(t)=6t^5-15t^4+10t^3 has
    # ZERO 1st AND 2nd derivative at both ends -> C2 seam (no curvature spike) + a gentle low-curv
    # arc (v_curv stays high -> no braking). It only ever deviates to CLEAR the obstacle, so it
    # never re-cuts real corners (no undulation) — corner reshaping stays option 2.
    seg = np.roll(clean_xy, -1, axis=0) - clean_xy
    el_cl = np.hypot(seg[:, 0], seg[:, 1])
    s_loop = np.concatenate([[0.0], np.cumsum(el_cl)])[:N]
    track_len = float(np.sum(el_cl))

    # per-station feasible offset band [lo_off, hi_off] for the car (w_veh) + wall_margin, and the
    # REQUIRED signed offset to clear the obstacles (0 where the clean line already clears).
    hi_off = rl_mod[:, 2] - 0.5 * w_veh - wall_margin
    lo_off = -rl_mod[:, 3] + 0.5 * w_veh + wall_margin
    req = np.where(lo_off > 0.0, lo_off, np.where(hi_off < 0.0, hi_off, 0.0))
    feasible_stn = hi_off >= lo_off

    alpha_full = np.zeros(N)
    solved_runs: List[np.ndarray] = []
    n_solved = 0
    n_failed = 0
    for obs in obstacles:
        i_c = int(np.argmin(np.hypot(clean_xy[:, 0] - obs.x, clean_xy[:, 1] - obs.y)))
        if not feasible_stn[i_c]:
            n_failed += 1                                # corridor blocked -> reactive layer covers
            continue
        peak = float(req[i_c])
        if abs(peak) < 1e-3:
            continue                                     # clean line already clears -> no arc
        v_obs = float(clean_vx[i_c]) if clean_vx is not None else 3.0
        R = float(np.clip(reach_time * v_obs, reach_min, reach_max))
        ds = np.abs(s_loop - s_loop[i_c])
        ds = np.minimum(ds, track_len - ds)              # wrap-aware arc distance to the obstacle
        t = np.clip(ds / max(R, 1e-3), 0.0, 1.0)
        shape = 1.0 - (6.0 * t ** 5 - 15.0 * t ** 4 + 10.0 * t ** 3)   # 1 @obstacle -> 0 @R (C2 ends)
        bump = peak * shape
        # max-MAGNITUDE envelope over obstacles: same-side close obstacles merge into one wider
        # plateau that clears both (a signed sum would over-shoot then get clamped short); opposite
        # sides keep the dominant clearance (one line can't slalom two near-coincident blockers).
        alpha_full = np.where(np.abs(bump) > np.abs(alpha_full), bump, alpha_full)
        n_solved += 1

    # curvature contribution of the arc = alpha'' (2nd deriv wrt arc length) — computed from the
    # SMOOTH (pre-clamp) smootherstep so it stays clean (the clamp below can add corridor kinks
    # that a finite-difference would read as huge fake curvature).
    elm = np.roll(el_cl, 1)
    h2 = np.maximum((0.5 * (el_cl + elm)) ** 2, 1e-9)
    alpha_dd = (np.roll(alpha_full, -1) - 2.0 * alpha_full + np.roll(alpha_full, 1)) / h2
    # The closed loop has a DUPLICATED closing point (xy[-1]==xy[0], el=0), which breaks the
    # 2nd-difference stencil at idx 0 and N-1 (it uses the duplicate instead of the real neighbour
    # N-2) -> a huge fake curvature spike right at start/finish. Recompute those two from the real
    # physical neighbours so an arc that wraps past s=0 stays smooth.
    if N > 3:
        h0 = 0.5 * (el_cl[0] + el_cl[N - 2])
        alpha_dd[0] = (alpha_full[1] - 2.0 * alpha_full[0] + alpha_full[N - 2]) / max(h0 ** 2, 1e-9)
        alpha_dd[N - 1] = alpha_dd[0]

    # clamp the arc to the drivable corridor (walls), but keep the bounds inclusive of 0 so a
    # zero-arc station is NEVER pushed off the clean line at a narrow spot (that would deviate the
    # line where there is no obstacle = undulation). A no-op wherever the arc already fits.
    alpha_full = np.clip(alpha_full, np.minimum(lo_off, 0.0), np.maximum(hi_off, 0.0))

    stitch_xy = clean_xy + alpha_full[:, None] * nvec_rl if n_solved else clean_xy.copy()
    if N > 1 and np.allclose(clean_xy[-1], clean_xy[0]):
        stitch_xy[-1] = stitch_xy[0]                     # keep the closed-loop closing point exact

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

    ggv, axm, m_veh, drag, dyn_exp, v_max = _load_veh_dyn(input_path)

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
        margin = 10
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

    return {"reftrack_mod": rl_mod, "report": report,
            "main": (traj, bound_r, bound_l, est), "d_right": d_right, "d_left": d_left,
            "n_windows": n_solved, "n_failed": n_failed}


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
