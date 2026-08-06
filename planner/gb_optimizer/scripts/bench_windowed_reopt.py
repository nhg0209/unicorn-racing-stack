#!/usr/bin/env python3
"""Does the windowed minimum-curvature re-opt do what the hump pipeline could not?

Five questions, on the real ifac map:

  (1) NO-OP        with no obstacles, is the output byte-identical to the clean line?
  (2) HOLD         two boxes on the straight, 8.48 m apart -- do they land in ONE window, and
                   does the line hold the offset between them (min |alpha| in the gap) instead of
                   coming home? With int_k2 and peak |kappa|.
  (3) LOCALITY     the same pair PLUS a corner box elsewhere on the lap -- is the straight pair's
                   alpha BIT-IDENTICAL to (2)? This is the whole claim of the design: a window is
                   an independent optimization problem, so an obstacle 15 m away cannot change it.
                   The hump pipeline fails exactly here (measured: the corner hump owns the ramp
                   curvature tiebreak and breaks the pair's hold).
  (4) SEAM         |alpha| at each window end, against seam_alpha_max. fix_s/fix_e pin the ends
                   only to a +-5 cm box, so a short window shows up here as a step.
  (5) COST         iterations, wall clock p50/p95, and the per-iteration curvature-error trace.

  ~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/bench_windowed_reopt.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/gb_optimizer"))

from gb_optimizer import windowed_reopt as W          # noqa: E402
from gb_optimizer.static_reopt_core import ModulationParams, Obstacle  # noqa: E402

MAP = "ifac"
BOX_R = 0.15
STRAIGHT_A, STRAIGHT_B, CORNER = 275, 360, 120


def load_ifac():
    """The published raceline as a reftrack [x, y, w_tr_right, w_tr_left].

    global_waypoints.json closes the loop by repeating point 0 byte-for-byte as its last entry,
    so the last row is dropped: ifac is 367 stations, L = 36.6026 m.
    """
    d = json.load(open(REPO / "stack_master/maps" / MAP / "global_waypoints.json"))
    wp = d["global_traj_wpnts_iqp"]["wpnts"][:-1]
    ref = np.array([[w["x_m"], w["y_m"], w["d_right"], w["d_left"]] for w in wp], float)
    return ref, np.array([w["s_m"] for w in wp], float)


def corridor_from_map(ref):
    """Car-centre corridor: the eroded occupancy grid intersected with the waypoint bounds.

    Same measurement the node makes (kernel 7 reserves half a car, wall_margin comes off on top),
    reproduced here so the bench sees what the car sees.
    """
    import cv2
    import yaml
    from PIL import Image
    mapdir = REPO / "stack_master/maps" / MAP
    meta = yaml.safe_load((mapdir / f"{MAP}.yaml").read_text())
    img = np.array(Image.open(mapdir / meta["image"]).convert("L"))
    occ = img < int(255 * (1.0 - meta["occupied_thresh"]))
    base = np.flipud(np.where(occ, 0, 255).astype(np.uint8))
    er = cv2.erode(base, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    res, org = meta["resolution"], (meta["origin"][0], meta["origin"][1])

    pts = ref[:, :2]
    _psi, nv = W._frame(pts)
    d_scan = np.arange(-3.0, 3.0 + 1e-9, 0.05)
    i0 = int(np.argmin(np.abs(d_scan)))
    xy = pts[:, None, :] + d_scan[None, :, None] * nv[:, None, :]
    px = ((xy[..., 0] - org[0]) / res).astype(int)
    py = ((xy[..., 1] - org[1]) / res).astype(int)
    h, w = er.shape
    inside = (px >= 0) & (py >= 0) & (px < w) & (py < h)
    free = np.zeros(px.shape, bool)
    free[inside] = er[py[inside], px[inside]] == 255
    n = len(pts)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    bh, bl = ~free[:, i0:], (~free[:, :i0 + 1])[:, ::-1]
    kh = np.where(bh.any(1), np.argmax(bh, 1), bh.shape[1])
    kl = np.where(bl.any(1), np.argmax(bl, 1), bl.shape[1])
    ok = free[:, i0]
    hi[ok] = (kh[ok] - 1) * 0.05 - 0.05
    lo[ok] = -((kl[ok] - 1) * 0.05 - 0.05)
    return lo, hi


def box_at(ref, i):
    return Obstacle(float(ref[i, 0]), float(ref[i, 1]), BOX_R)


def alpha_between(alpha, s, ia, ib):
    """min |alpha| strictly between two stations -- the HOLD test."""
    n = len(alpha)
    span = (np.arange(ia, ib + 1) if ib >= ia else np.arange(ia, ib + n + 1)) % n
    inner = span[3:-3] if len(span) > 8 else span
    return float(np.min(np.abs(alpha[inner]))), span


def main():
    ref, s = load_ifac()
    cor = corridor_from_map(ref)
    p = W.WindowParams()
    mp = ModulationParams()
    n = len(ref)
    gap = float((s[STRAIGHT_B] - s[STRAIGHT_A]) % s[-1])
    print(f"map {MAP}: {n} stations, L = {s[-1] + (s[1] - s[0]):.4f} m | "
          f"boxes at {STRAIGHT_A}/{STRAIGHT_B} are {gap:.2f} m apart")
    print(f"WindowParams: pad {p.pad_m} merge {p.merge_gap_m} iters {p.iters_min}..{p.max_iters} "
          f"tol {p.curv_tol} kappa {p.kappa_bound} w_veh {p.w_veh} seam {p.seam_alpha_max}")

    # (1) ---------------------------------------------------------------------------------
    line0, alpha0, rep0 = W.reoptimize_windowed(ref, [], cor, p, mp)
    ident = bool(np.array_equal(line0, ref[:, :2]))
    print(f"\n(1) NO-OP      windows {rep0.n_windows} | byte-identical to the clean line: {ident}"
          f" | same object: {line0 is ref[:, :2] or np.shares_memory(line0, ref)}")

    # (2) ---------------------------------------------------------------------------------
    obs2 = [box_at(ref, STRAIGHT_A), box_at(ref, STRAIGHT_B)]
    t = time.perf_counter()
    line2, alpha2, rep2 = W.reoptimize_windowed(ref, obs2, cor, p, mp)
    ms2 = (time.perf_counter() - t) * 1e3
    hold2, span2 = alpha_between(alpha2, s, STRAIGHT_A, STRAIGHT_B)
    w2 = rep2.windows[0] if rep2.windows else None
    print(f"\n(2) HOLD       windows {rep2.n_windows} (want 1) | sides {rep2.sides} | {ms2:.0f} ms")
    if w2:
        print(f"    ok={w2.ok} reason='{w2.reason}' iters={w2.iters} span={w2.span_m:.2f} m "
              f"stations={w2.n_stations}")
        print(f"    int_k2={w2.int_k2:.3f}  peak|kappa|={w2.peak_kappa:.3f}  "
              f"clearances={[round(c,3) for c in w2.clearances]}  seam={w2.seam_alpha:.4f} m")
    print(f"    min |alpha| between the boxes = {hold2:.3f} m "
          f"({'HELD' if hold2 > 0.25 else 'comes home -- W'})")

    # (3) ---------------------------------------------------------------------------------
    obs3 = obs2 + [box_at(ref, CORNER)]
    line3, alpha3, rep3 = W.reoptimize_windowed(ref, obs3, cor, p, mp)
    hold3, _ = alpha_between(alpha3, s, STRAIGHT_A, STRAIGHT_B)
    pair_win2 = rep2.windows[0] if rep2.windows else None
    # compare the straight pair's own window stations
    same = None
    if rep2.n_windows >= 1 and rep3.n_windows >= 1:
        w_specs3 = W.form_windows(ref, obs3, rep3.sides, p)
        w_specs2 = W.form_windows(ref, obs2, rep2.sides, p)
        idx2 = w_specs2[0].idx
        cand = [ws for ws in w_specs3 if set(ws.obs) == {0, 1}]
        if cand and np.array_equal(cand[0].idx, idx2):
            same = bool(np.array_equal(alpha2[idx2], alpha3[idx2]))
            maxdiff = float(np.max(np.abs(alpha2[idx2] - alpha3[idx2])))
        else:
            maxdiff = float("nan")
    print(f"\n(3) LOCALITY   windows {rep3.n_windows} (want 2) | sides {rep3.sides}")
    print(f"    the straight pair's alpha vs case (2): bit-identical = {same} "
          f"(max |diff| = {maxdiff:.2e} m)")
    print(f"    min |alpha| between the boxes = {hold3:.3f} m")

    # (4) ---------------------------------------------------------------------------------
    print(f"\n(4) SEAM       (seam_alpha_max = {p.seam_alpha_max} m)")
    for tag, r in (("2 boxes", rep2), ("3 boxes", rep3)):
        for k, wr in enumerate(r.windows):
            flag = "OK" if wr.seam_alpha <= p.seam_alpha_max else "STEP"
            print(f"    {tag} window {k}: |alpha| at the ends = {wr.seam_alpha:.4f} m  {flag}")

    # (5) ---------------------------------------------------------------------------------
    print(f"\n(5) COST")
    times, iters = [], []
    for case, obs in (("2 boxes", obs2), ("3 boxes", obs3)):
        for _ in range(5):
            t = time.perf_counter()
            _l, _a, r = W.reoptimize_windowed(ref, obs, cor, p, mp)
            times.append((time.perf_counter() - t) * 1e3)
            iters += [w.iters for w in r.windows]
    print(f"    wall clock over {len(times)} solves: p50 {np.percentile(times,50):.0f} ms  "
          f"p95 {np.percentile(times,95):.0f} ms  max {max(times):.0f} ms")
    print(f"    iterations per window: {sorted(set(iters))} (cap {p.max_iters})")
    for tag, r in (("2 boxes", rep2), ("3 boxes", rep3)):
        for k, wr in enumerate(r.windows):
            tr = ", ".join(f"{e:.4f}" for e in wr.curv_err)
            print(f"    {tag} window {k} curv_err by iteration: [{tr}]  (tol {p.curv_tol})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
