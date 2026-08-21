#!/usr/bin/env python3
"""Where an opponent will BE by the time we get there, without a Gaussian process.

NOTHING IS WIRED TO THIS. It is a reference implementation kept on purpose: the dormant
spliner_node carried four obstacle-propagation modes that the live lane_change_planner has no
equivalent of, and MIGRATION_GAP_ANALYSIS listed them as the first-priority extraction because
they are GP-free, cost nothing, and degrade gracefully -- the properties gp_traj_predictor does
not have. The node those modes lived in is deleted; the arithmetic is here so that deleting the
node did not delete the item. Wire it, or close the item -- but do not let it rot as a file
nobody can find.

Pure functions. The originals read `self.cur_s`, `self.gb_max_s`, `self.gb_scaled_wpnts`,
`self.kd_obs_pred`, `self.fixed_pred_time` and `self.converter` off the node, and published a
marker as a side effect. The marker publishing is NOT carried across: it is node coupling and has
nothing to do with the prediction. Everything else arrives as an argument.

WHAT MUST BE FIXED WHEN THIS IS WIRED, and was already wrong in the original:

  * `gb_scaled_wpnts.wpnts[int(cur_s * 10)]` hardcodes a 0.1 m waypoint spacing. The state
    machine's GlobalTracking inherited the identical defect and it was fixed there by dividing by
    the real `wpnt_dist`; on a map whose spacing is not 0.1 m the local window ended up as much as
    1 m from the car. `ego_vx_at` below is that fix, and the modes take the ego speed as a VALUE so
    the indexing cannot be got wrong twice.

  * `heuristic` never assigned `delta_s`. As shipped it raises UnboundLocalError the first time it
    is selected, which is the strongest evidence available that the mode was never run. It is
    given the same `delta_s` as `adaptive` here -- the only reading consistent with the rest of the
    branch -- and this note is left so the choice is visible rather than silently canonical.

  * `adaptive_velheuristic` divides by `(1 - opponent_scaler) * ego_speed` with no guard, so a
    stationary ego is a division by zero. Clipped like `adaptive`'s own rel_speed.

The modes, all four kept:

  constant                propagate by a fixed time. No d relaxation (the original's line for it
                          is commented out; kept commented, because turning it on is a behaviour
                          change and this file is a preservation, not a redesign).
  adaptive                time-to-reach from the real closing speed, halved, then relax d toward
                          the raceline by exp(-|kd * d|) -- an opponent far off-line is assumed to
                          be coming back to it, and the further off it is the less it relaxes.
  adaptive_velheuristic   the same, with the closing speed assumed to be a fixed fraction of ego
                          speed instead of measured. For when the opponent's own vs is untrusted.
  heuristic               the same, with the closing speed assumed to be 3 m/s flat.
"""
import copy
from typing import Optional

import numpy as np

__all__ = ["MODES", "ego_vx_at", "predict_obs_movement"]

MODES = ("constant", "adaptive", "adaptive_velheuristic", "heuristic")

_DEFAULT_HORIZON_M = 10.0      # the original's `< 10`, which carried a "TODO make param"
_OPPONENT_SCALER = 0.7         # adaptive_velheuristic: opponent speed as a fraction of ego's
_HEURISTIC_REL_SPEED = 3.0     # heuristic: assumed closing speed [m/s]
_MAX_PRED_S = 5.0              # [s] every mode clips time-to-reach here


def ego_vx_at(cur_s: float, wpnts, wpnt_dist: float) -> float:
    """Ego's scaled speed at `cur_s`, indexed by the REAL spacing.

    The original wrote `wpnts[int(cur_s * 10)]`, which is this function with wpnt_dist frozen at
    0.1 m. Pass the spacing the map actually publishes.
    """
    if not wpnts or wpnt_dist <= 0.0:
        return 0.0
    return float(wpnts[int(cur_s / wpnt_dist) % len(wpnts)].vx_mps)


def predict_obs_movement(obs, cur_s: float, gb_max_s: float, ego_vx: float = 0.0,
                         mode: str = "constant", kd_obs_pred: float = 0.5,
                         fixed_pred_time: float = 0.5,
                         horizon_m: float = _DEFAULT_HORIZON_M) -> Optional[object]:
    """A COPY of `obs`, moved to where it is predicted to be. The input is not touched.

    Returns the copy unchanged when the obstacle is further than `horizon_m` ahead -- the original
    propagated nothing out there, on the reasoning that a prediction five seconds old is worse than
    the measurement it replaced.

    `obs` needs s_start / s_center / s_end / d_left / d_center / d_right / vs / vd; a ROS
    f110_msgs/Obstacle and a SimpleNamespace both qualify, which is what keeps this testable
    without ROS.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    out = copy.deepcopy(obs)
    if gb_max_s <= 0.0:
        return out
    ot_distance = (out.s_center - cur_s) % gb_max_s
    if ot_distance >= horizon_m:
        return out

    if mode == "adaptive":
        rel_speed = float(np.clip(ego_vx - out.vs, 0.1, 10.0))
        ot_time_distance = float(np.clip(ot_distance / rel_speed, 0.0, _MAX_PRED_S)) * 0.5
        delta_s = ot_time_distance * out.vs
        delta_d = ot_time_distance * out.vd
        delta_d = -(out.d_center + delta_d) * np.exp(-np.abs(kd_obs_pred * out.d_center))

    elif mode == "adaptive_velheuristic":
        # Clipped low, which the original was not: rel_speed is (1 - scaler) * ego_speed and a
        # stationary ego made that zero.
        rel_speed = float(np.clip((1.0 - _OPPONENT_SCALER) * ego_vx, 0.1, 10.0))
        ot_time_distance = float(np.clip(ot_distance / rel_speed, 0.0, _MAX_PRED_S))
        delta_s = ot_time_distance * _OPPONENT_SCALER * ego_vx
        delta_d = -out.d_center * np.exp(-np.abs(kd_obs_pred * out.d_center))

    elif mode == "constant":
        delta_s = fixed_pred_time * out.vs
        delta_d = fixed_pred_time * out.vd
        # The original relaxes d here too, commented out. Left commented: enabling it is a
        # behaviour change, and this file preserves what was written.
        # delta_d = -(out.d_center + delta_d) * np.exp(-np.abs(kd_obs_pred * out.d_center))

    else:                                   # heuristic
        ot_time_distance = float(np.clip(ot_distance / _HEURISTIC_REL_SPEED, 0.0, _MAX_PRED_S))
        # delta_s was never assigned in the original -- see the module docstring. Given the same
        # reading as `adaptive`, which is the only one the surrounding lines support.
        delta_s = ot_time_distance * out.vs
        delta_d = ot_time_distance * out.vd
        delta_d = -(out.d_center + delta_d) * np.exp(-np.abs(kd_obs_pred * out.d_center))

    out.s_start = (out.s_start + delta_s) % gb_max_s
    out.s_center = (out.s_center + delta_s) % gb_max_s
    out.s_end = (out.s_end + delta_s) % gb_max_s
    out.d_left += delta_d
    out.d_center += delta_d
    out.d_right += delta_d
    return out
