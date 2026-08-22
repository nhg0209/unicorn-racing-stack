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
# Default corridor-fit tolerance [m]. The corridor bounds are a `_cyclic_smooth`ed ESTIMATE that
# ripples ~2 mm between adjacent stations, and the amplitude cap parks the hump peak exactly on
# that bound — so a zero-tolerance test rejects reaches over a SUB-MILLIMETRE violation at the
# station next to the apex, and the reach bisects down to whatever is sharp enough to duck under
# the ripple. Measured on ifac: a 0.5 mm violation at station 292 collapsed the reach 5.00 -> 1.24 m,
# which took max|kappa| 1.46 -> 1.98 (curvlim is 1.5) and cost +1.62 s on an 11.30 s lap. 5 mm is
# 4-10% of the wall_margin it eats into, and the true track bound is still wall_margin + w_veh/2 away.
_FIT_TOL_DEFAULT = 0.005


def _wrap_normals(xy: np.ndarray) -> np.ndarray:
    """Unit normal per station of a CLOSED line, in the same +right convention as
    `centerline_frame`, from a wrap-around central-difference tangent on the UNIQUE points.

    This is the lateral BASIS the offset is laid on and every downstream check is measured in, so
    an error in it is indistinguishable from an error in the offset -- and it was the larger of the
    two. `centerline_frame` derives the normal from tph's numerically differentiated heading over
    the array AS GIVEN, which for a raceline means two things:

      * THE SEAM. The closed raceline ships with a DUPLICATED closing point (xy[-1] == xy[0]), so
        the last segment has length 0 and the heading there is 0/0. The existing nvec[-1] = nvec[0]
        patch repairs that one vector but not its neighbours, and the basis arrives at s = 0 having
        rotated by the wrong amount: measured against the rotation the line's own curvature
        demands (kappa_clean * el), the seam segment is off by 23 mrad on ifac and 44 mrad on f.
        Laid under a ~0.5 m offset that is a ~2 cm lateral step at one station -- a visible kink in
        the published line, and a |dkappa| of 0.44 where the offset profile itself is perfectly C2.
      * CORNERS. The same comparison shows up to 64 mrad of station-to-station error on ifac,
        which under half a metre of offset is the ripple seen along the humps.

    Dropping the duplicate and taking the tangent from p[i+1] - p[i-1] removes both: the residual
    falls to 9.6 mrad max / 0.5 mrad at the seam on ifac, and 1.3 / 0.6 on f.

    `centerline_frame` itself is left alone -- it has other consumers (modulate_widths) whose
    geometry is not this raceline.
    """
    p = np.asarray(xy, float)[:, :2]
    n = len(p)
    dup = bool(n > 1 and np.allclose(p[-1], p[0]))
    u = p[:-1] if dup else p
    t = np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)     # wrap-around central difference
    nrm = np.hypot(t[:, 0], t[:, 1])
    nrm[nrm < 1e-12] = 1e-12
    t = t / nrm[:, None]
    # +right convention, matching calc_normal_vectors(psi) as used by centerline_frame: verified
    # against it on both shipped maps (dot > 0 at every station of ifac and f).
    nv = np.column_stack([t[:, 1], -t[:, 0]])
    # the duplicated closing point is the SAME physical point as station 0 -- give it that normal
    return np.vstack([nv, nv[:1]]) if dup else nv


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
                      target_n: int,
                      d_m: np.ndarray = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    x,y,kappa,vx,ax,d_right,d_left and d_m are linearly interpolated along the arc; psi + s are
    recomputed from the resampled xy. d_m rides along for the same reason the widths do: the
    redistribution moves every station along the arc, so a lateral field taken before the resample
    no longer lines up with traj row for row (a 0.5 m detour shifts the far end by ~5 points).
    Returns (traj_M, d_right_M, d_left_M, d_m_M) with a duplicated closing point."""
    xy = traj[:, 1:3]
    dup = np.allclose(xy[-1], xy[0])
    xyu = xy[:-1] if dup else xy
    n = len(xyu)
    if n < 4 or int(target_n) < 4:
        return traj, d_right, d_left, d_m
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
    dm = _interp(d_m) if d_m is not None else None
    # close the loop (duplicate the start point) to match the clean-bundle convention
    new_xy = np.vstack([new_xy, new_xy[:1]])
    kap = np.append(kap, kap[0]); vx = np.append(vx, vx[0]); ax = np.append(ax, ax[0])
    dr = np.append(dr, dr[0]); dl = np.append(dl, dl[0])
    if dm is not None:
        dm = np.append(dm, dm[0])
    # recompute psi + s on the uniformly-spaced closed line
    segm = np.roll(new_xy, -1, axis=0) - new_xy
    elm = np.hypot(segm[:, 0], segm[:, 1])
    psi_m, _ = tph.calc_head_curv_num.calc_head_curv_num(path=new_xy, el_lengths=elm, is_closed=True)
    s_m = np.concatenate([[0.0], np.cumsum(elm)])[:len(new_xy)]
    traj_m = np.column_stack([s_m, new_xy[:, 0], new_xy[:, 1], psi_m, kap, vx, ax])
    return traj_m, dr, dl, dm

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


def _on_clean_mask(traj: np.ndarray, clean_xy: np.ndarray, dev_tol: float = 0.02):
    """(mask, nearest_clean_index) for published points that have REJOINED the clean raceline.

    Distance is to the clean POLYLINE (point-to-segment), not to its vertices. That distinction is
    the whole usefulness of this mask: the clean line is sampled every ~0.1 m, so a published point
    lying exactly ON it but between two vertices is up to 0.05 m from the nearest vertex — over any
    sane tolerance. Measured on a 2-hump ifac line, the vertex test called 67% of the points
    "deviating" with a median residual of 31.9 mm, while the segment test gives 21% and a median of
    0.0 mm — and 21% is exactly the span the humps occupy. Everything built on this mask (clean
    curvature and clean speed restoration) was therefore firing on a minority of the stations it
    should have.

    Positions, not indices: the resample and the detour both change arc length, so index/s matching
    would compare different places on the track.
    """
    cx = np.asarray(clean_xy, float)
    ax, ay = cx[:, 0], cx[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    dx, dy = bx - ax, by - ay
    seg2 = np.maximum(dx * dx + dy * dy, 1e-12)
    px, py = traj[:, 1], traj[:, 2]
    t = np.clip(((px[:, None] - ax[None, :]) * dx[None, :]
                 + (py[:, None] - ay[None, :]) * dy[None, :]) / seg2[None, :], 0.0, 1.0)
    qx = ax[None, :] + t * dx[None, :]
    qy = ay[None, :] + t * dy[None, :]
    d2 = (px[:, None] - qx) ** 2 + (py[:, None] - qy) ** 2
    j = np.argmin(d2, axis=1)
    on_clean = np.sqrt(d2[np.arange(len(j)), j]) <= dev_tol
    # Report the nearer ENDPOINT of the winning segment: callers index per-station clean arrays
    # (kappa, vx) with it, and on a clean stretch the two endpoints agree to the sampling error.
    tj = t[np.arange(len(j)), j]
    j_pt = np.where(tj > 0.5, (j + 1) % len(ax), j)
    return on_clean, j_pt


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
    on_clean, j = _on_clean_mask(traj, clean_xy, dev_tol)
    out = k_geo.copy()
    out[on_clean] = ck[j[on_clean]]
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


def build_wpnts(traj: np.ndarray, d_right: np.ndarray, d_left: np.ndarray, second_traj: bool = False,
                d_m: np.ndarray = None):
    """Build (WpntArray, MarkerArray) from an optimizer trajectory [s,x,y,psi,kappa,vx,ax].

    ROS message imports are lazy so the numeric core can be used without a ROS session.

    d_m is the lateral deviation from the CLEAN raceline, signed LEFT-positive, and defaults to 0
    (which is what the clean line itself is). It is not decoration: Controller.py reads it as
    column 8 and takes max |d_m| over the local window to decide whether the path it is following
    is legitimately offset -- an avoidance line -- and therefore gets the looser garbage-path bar
    (AEB_thres_overtake) instead of the tight one (AEB_thres). Leaving it at 0 on a line with a
    hump in it made the tight 0.5 m bar fire against the hump the car was deliberately driving,
    clamping the speed command to 2.0 m/s on approach and releasing it after the obstacle.
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
        w.d_m = float(d_m[i]) if d_m is not None else 0.0
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
