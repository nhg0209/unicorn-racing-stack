"""Per-stage opponent-position sampling for prediction-aware MPCC avoidance.

Pure functions (no ROS/acados) so they unit-test in isolation. Given a GP-
predicted opponent path plus the opponent's current frenet progress and speed,
produce the predicted opponent (x, y) at each MPC shooting-node future time.
The acados solver already accepts a per-stage obstacle position; this fills it
with WHERE THE OPPONENT WILL BE instead of a single frozen current position.
"""
import numpy as np


def stage_times(dT, N, dt_vec=None):
    """Cumulative future time [s] at shooting nodes k = 0..N (length N+1).

    Uniform grid uses dT; a non-uniform grid passes explicit per-step dt_vec
    (length N)."""
    if dt_vec is None:
        return np.arange(N + 1, dtype=float) * float(dT)
    return np.concatenate(([0.0], np.cumsum(np.asarray(dt_vec, dtype=float))))


def project_s(px, py, path_x, path_y, path_s):
    """Frenet progress s of point (px, py) projected onto a polyline path
    (nearest-vertex approximation — adequate for a densely sampled raceline).
    Returns the path_s of the nearest path vertex."""
    x = np.asarray(path_x, float)
    y = np.asarray(path_y, float)
    i = int(np.argmin((x - float(px)) ** 2 + (y - float(py)) ** 2))
    return float(np.asarray(path_s, float)[i])


def sample_opp_xy(opp_s, opp_x, opp_y, s0, v, times, track_len):
    """Predicted opponent (x, y) at each future time.

    opp_s, opp_x, opp_y : 1-D arrays of the predicted opponent path (need not be
                          sorted by s).
    s0                  : opponent current frenet progress [m].
    v                   : opponent longitudinal speed [m/s] (proj_vs_mps).
    times              : future times [s], length M.
    track_len          : closed-track length L [m] for wraparound.
    Returns (M, 2) array. Constant-velocity advance of the opponent along its
    own predicted path: s(t) = s0 + v*t, wrapped into [s_min, s_min + L)."""
    order = np.argsort(opp_s)
    s = np.asarray(opp_s, float)[order]
    x = np.asarray(opp_x, float)[order]
    y = np.asarray(opp_y, float)[order]
    s_pred = float(s0) + float(v) * np.asarray(times, float)
    s_wrapped = s[0] + np.mod(s_pred - s[0], float(track_len))
    return np.column_stack([np.interp(s_wrapped, s, x), np.interp(s_wrapped, s, y)])
