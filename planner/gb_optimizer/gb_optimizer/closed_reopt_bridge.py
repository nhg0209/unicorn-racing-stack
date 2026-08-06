"""Make closed_reopt's answer look exactly like reoptimize_local_window's, and nothing else.

static_reopt_node has a long tail of safety machinery behind the solve -- the clearance veto, the
identity veto, the stale-solve epoch guard, the wall gate, the published-curvature gate, the
coverage-regression refusal, the publish deadband. Every one of them reads the dict the solve
returns. So the way to offer a second solver is to return the SAME dict, not to teach the node
about a second shape: the branch is two lines at the call site and nothing downstream can tell
which solver ran.

WHAT IS REPRODUCED FROM THE HUMP PATH, DELIBERATELY AND IN THE SAME ORDER (see the tail of
_reopt_local_window_impl): uniform resample to the clean line's point count, curvature republished
from the FINAL geometry with the clean values restored where the line has rejoined it, a full-lap
velocity profile capped by the tuned clean speeds, the speed feasibility pass over the published
curvature, ax re-derived on the published spacing, and the published-curvature verdict against
max(curvlim, the clean line's own curvature). Those are not hump-specific: they are what makes a
geometry publishable, and a line that skipped them would be a different kind of object on the wire.

WHAT IS NOT: apexes. The QP needs no reactive apex, so `apex_laid` and `apex_dropped` are
SYNTHESISED from what it did -- one laid record per box the line covers, one dropped record per
box it hands back, carrying closed_reopt's own reason string. That is what the node's coverage
topic and its regression check consume, so the handover reaches the reactive layer through the
channel that already exists rather than a new one.

The floor is closed_reopt's own: obs_margin + w_veh/2. It is NOT the hump's obs_margin, and the
node is told so per obstacle, so the clearance veto holds this line to what this solver promised.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import static_reopt_core as core
from . import closed_reopt as cq


def _closed_pair(clean_xy: np.ndarray) -> Tuple[np.ndarray, bool]:
    """(the unique stations, whether the input repeated its first point to close the loop)."""
    xy = np.asarray(clean_xy, float)
    dup = len(xy) > 1 and bool(np.allclose(xy[-1], xy[0]))
    return (xy[:-1] if dup else xy), dup


def reoptimize_closed_window(clean_xy: np.ndarray,
                             clean_dr: np.ndarray,
                             clean_dl: np.ndarray,
                             reftrack: np.ndarray,
                             obstacles: Sequence,
                             input_path: str,
                             params: Optional[cq.ReoptParams] = None,
                             w_veh: float = 0.30,
                             clean_vx: Optional[np.ndarray] = None,
                             clean_kappa: Optional[np.ndarray] = None,
                             corridor_lo: Optional[np.ndarray] = None,
                             corridor_hi: Optional[np.ndarray] = None) -> dict:
    """Solve the closed-track avoidance QP and return reoptimize_local_window's dict shape.

    `obstacles` are core.Obstacle (x, y, r). No apexes: this solver does not need the reactive
    layer to have driven the gap first, which is the whole reason a box it has never seen can be
    covered on the lap it is confirmed.
    """
    with core._blas_single_thread():
        return _impl(clean_xy, clean_dr, clean_dl, reftrack, obstacles, input_path, params,
                     w_veh, clean_vx, clean_kappa, corridor_lo, corridor_hi)


def _impl(clean_xy, clean_dr, clean_dl, reftrack, obstacles, input_path, params, w_veh,
          clean_vx, clean_kappa, corridor_lo, corridor_hi) -> dict:
    p = params or cq.ReoptParams(w_veh=w_veh)
    xy_all = np.asarray(clean_xy, float)
    xy, dup = _closed_pair(xy_all)
    n = len(xy)
    N = len(xy_all)
    dr = np.asarray(clean_dr, float)
    dl = np.asarray(clean_dl, float)
    nvec_rl = core._wrap_normals(xy_all)
    ggv, axm, m_veh, drag, dyn_exp, v_max, curvlim = core._load_veh_dyn(input_path)
    need = p.obs_margin + 0.5 * p.w_veh

    # the corridor closed_reopt takes is (lo, hi) per unique station; the node measures it on the
    # closed array, so the closing duplicate comes off with everything else
    cor = None
    if corridor_lo is not None and corridor_hi is not None:
        cor = (np.asarray(corridor_lo, float)[:n], np.asarray(corridor_hi, float)[:n])

    ref = np.column_stack([xy[:, 0], xy[:, 1], dr[:n], dl[:n]])
    boxes = [cq.Obstacle(float(o.x), float(o.y), float(o.r)) for o in obstacles]
    line, d_open, rep = cq.reoptimize_closed(ref, boxes, cor, p)

    # --- the records the node's coverage and logs are built from -------------------------------
    covered = [i for i in range(len(boxes)) if i not in rep.infeasible]
    apex_laid, apex_dropped = [], []
    for slot, i in enumerate(covered):
        o = boxes[i]
        j = int(np.argmin(np.hypot(line[:, 0] - o.x, line[:, 1] - o.y)))
        apex_laid.append({
            "obs_i": i, "xy": (float(o.x), float(o.y)),
            "laid": float(d_open[j]), "want": float(d_open[j]),
            "r_in": float(p.infl_len_m), "r_out": float(p.infl_len_m),
            "r_allow": float(p.infl_len_m), "r_req": float(p.infl_len_m),
            "kappa_peak": float(rep.peak_kappa),
            "clear": float(rep.clearances[slot]) if slot < len(rep.clearances) else float("nan"),
            "floor": need})
    for k, i in enumerate(rep.infeasible):
        o = boxes[i]
        why = rep.infeasible_why[k] if k < len(rep.infeasible_why) else "infeasible"
        apex_dropped.append({
            "obs_i": i, "xy": (float(o.x), float(o.y)),
            # the node prints `reason` in its rejection warning and ranks coverage by it; the full
            # sentence goes in `detail`, which is what the coverage record carries to the topic
            "reason": "corridor" if "corridor is" in why or "neither side" in why else "curvature",
            "detail": why, "want": 0.0, "fit": 0.0})

    # --- geometry, exactly as the QP validated it ---------------------------------------------
    stitch_xy = np.vstack([line, line[0]]) if dup else line.copy()
    seg = np.roll(stitch_xy, -1, axis=0) - stitch_xy
    el_cl = np.hypot(seg[:, 0], seg[:, 1])
    import trajectory_planning_helpers as tph
    psi_full, _ = tph.calc_head_curv_num.calc_head_curv_num(
        path=stitch_xy, el_lengths=np.maximum(el_cl, 1e-9), is_closed=True)
    s_full = np.concatenate([[0.0], np.cumsum(el_cl)])[:N]
    # MENGER, never kappa_clean + d''. The additive model is ~45% low on ifac and this line's
    # curvature is measured, not modelled -- it is the number the speed cap and the published
    # curvature gate both read.
    kap = core._menger_kappa(stitch_xy)
    vx = (np.asarray(clean_vx, float).copy() if clean_vx is not None
          else np.full(N, float(v_max)))
    ax = np.zeros(N)
    traj = np.column_stack([s_full, stitch_xy[:, 0], stitch_xy[:, 1], psi_full, kap, vx, ax])

    offset = np.einsum("ij,ij->i", stitch_xy - xy_all, nvec_rl)
    d_right = np.maximum(dr - offset, 0.0)
    d_left = np.maximum(dl + offset, 0.0)
    bound_r = xy_all + dr[:, None] * nvec_rl
    bound_l = xy_all - dl[:, None] * nvec_rl

    n_solved = len(covered)
    if n_solved:
        # UNIFORM resample to the clean line's point count. Not cosmetic: a length-derived count
        # kills sector_tuner (IndexError -> /global_waypoints_scaled stops -> the car keeps
        # following the OLD line) and shifts the index-based sector bounds.
        traj, d_right, d_left = core._resample_uniform(traj, d_right, d_left, N - 1)
        traj[:, 4] = core._republish_kappa(traj, xy_all, clean_kappa)
        _sg = np.roll(traj[:, 1:3], -1, axis=0) - traj[:, 1:3]
        _el = np.maximum(np.hypot(_sg[:, 0], _sg[:, 1]), 1e-6)
        try:
            vx_full = tph.calc_vel_profile.calc_vel_profile(
                ax_max_machines=axm, kappa=traj[:-1, 4], el_lengths=_el[:-1], closed=True,
                drag_coeff=drag, m_veh=m_veh, ggv=ggv, dyn_model_exp=dyn_exp, v_max=v_max)
            ceil = float(np.max(clean_vx)) if clean_vx is not None else v_max
            traj[:-1, 5] = np.minimum(vx_full, ceil)
            traj[-1, 5] = traj[0, 5]
        except Exception:
            pass                                  # keep the clean profile on any failure
        if clean_vx is not None:
            # the tuned speeds carry sector scaling the solver knows nothing about; restore them
            # wherever the geometry has rejoined the raceline
            on_clean, j_cl = core._on_clean_mask(traj, xy_all)
            traj[on_clean, 5] = np.asarray(clean_vx, float)[j_cl[on_clean]]
        core._cap_speed_to_published_curvature(traj, ggv, axm)
        _sg = np.roll(traj[:, 1:3], -1, axis=0) - traj[:, 1:3]
        _el = np.hypot(_sg[:, 0], _sg[:, 1])
        traj[:, 6] = (np.roll(traj[:, 5], -1) ** 2 - traj[:, 5] ** 2) / (2.0 * np.maximum(_el, 1e-6))
        if len(traj) > 2:
            traj[-1, 6] = traj[0, 6]
        est = float(np.sum(_el[:-1] / np.maximum(traj[:-1, 5], 1e-3)))
    else:
        est = float(np.sum(el_cl[:-1] / np.maximum(traj[:-1, 5], 1e-3)))

    # PUBLISHED-CURVATURE VERDICT, the same corner-fair bound the hump path is held to: the
    # maneuver may not ADD steering beyond curvlim, and is never blamed for the corner it drives
    # through. The envelope exists precisely so this stays true.
    k_pub = np.abs(core._menger_kappa(traj[:, 1:3]))
    k_cln = np.abs(core._menger_kappa(xy_all))
    _, j_near = core._on_clean_mask(traj, xy_all, dev_tol=np.inf)
    k_allow = np.maximum(float(curvlim), k_cln[j_near])
    over = k_pub - k_allow
    i_worst = int(np.argmax(over))

    return {"reftrack_mod": np.asarray(reftrack, float), "report": rep,
            "kappa_published_ok": bool(over[i_worst] <= 0.0),
            "kappa_published_max": float(k_pub[i_worst]),
            "kappa_published_allow": float(k_allow[i_worst]),
            "main": (traj, bound_r, bound_l, est), "d_right": d_right, "d_left": d_left,
            "n_windows": n_solved, "n_failed": len(rep.infeasible),
            "apex_dropped": apex_dropped, "apex_laid": apex_laid, "curvlim": float(curvlim),
            "alpha": offset, "clip_bite": 0.0,
            "span_m": float(p.infl_len_m * max(n_solved, 0)),
            "span_budget_m": float(p.infl_len_m * max(n_solved, 1)),
            "span_reach_m": float(p.infl_len_m),
            # closed_qp-only, for the node's log line and nothing else
            "closed_report": rep, "clearance_floor": need}


def coverage_records(obstacles: Sequence, res: dict, clearances: Sequence[float]) -> List[dict]:
    """One record per obstacle in the node's own coverage format, so /static_reopt/coverage --
    and the regression check that reads it -- work unchanged.

    THIS IS THE HANDOVER. A box the global line cannot take is not silently absent: it comes out
    as `dropped:<the reason closed_reopt gave>`, with the station, the side and the metres, on the
    latched topic the reactive layer and any bag reader already look at.
    """
    laid = {a["obs_i"] for a in res.get("apex_laid", [])}
    why = {d["obs_i"]: d.get("detail", d.get("reason", "?")) for d in res.get("apex_dropped", [])}
    out = []
    for i, o in enumerate(obstacles):
        clr = float(clearances[i]) if i < len(clearances) else float("nan")
        status = "laid" if i in laid else "dropped:" + str(why.get(i, "?"))
        out.append({"id": int(getattr(o, "id", -1)), "x": float(o.x), "y": float(o.y),
                    "r": float(o.r), "status": status, "clearance_m": clr})
    return out
