#!/usr/bin/env python3
"""Does the closed-track minimum-curvature re-opt do what the hump pipeline could not?

  (1) NO-OP      no obstacles -> is the input returned untouched?
  (2) HOLD       two boxes on the ifac straight (275, 360) -> does the line hold the offset
                 between them instead of coming home, and at what clearance / peak |kappa| /
                 int_k2?
  (3) LOCALITY   the same pair PLUS a corner box -> how far does the straight pair's own offset
                 move? A whole-lap solve cannot promise zero, so the number itself is the result.
  (4) GRID       0.30 / 0.50 / 0.70 m -- the conditioning claim behind grid_step_m.
  (5) COST       iterations, wall clock, and whether each iterate was acceptable.

  ~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/bench_closed_reopt.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/gb_optimizer"))

from gb_optimizer import closed_reopt as C            # noqa: E402
from gb_optimizer.closed_reopt import Obstacle, ReoptParams  # noqa: E402

MAP = "ifac"
BOX_R = 0.15
A_I, B_I = 275, 360
CORNERS = (120, 60, 200, 300)


def load_ifac():
    """The published raceline as [x, y, w_tr_right, w_tr_left].

    global_waypoints.json repeats point 0 byte-for-byte to close the loop, so the last row is
    dropped: 367 stations, L = 36.6026 m.
    """
    d = json.load(open(REPO / "stack_master/maps" / MAP / "global_waypoints.json"))
    wp = d["global_traj_wpnts_iqp"]["wpnts"][:-1]
    return np.array([[w["x_m"], w["y_m"], w["d_right"], w["d_left"]] for w in wp], float)


def corridor_from_map(ref):
    """Car-centre corridor from the eroded occupancy grid (kernel 7 = half a car reserved,
    wall_margin 0.05 on top) -- the same measurement static_reopt_node makes."""
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
    _psi, nv = C._frame(pts)
    d = np.arange(-3.0, 3.0 + 1e-9, 0.05)
    i0 = int(np.argmin(np.abs(d)))
    xy = pts[:, None, :] + d[None, :, None] * nv[:, None, :]
    px = ((xy[..., 0] - org[0]) / res).astype(int)
    py = ((xy[..., 1] - org[1]) / res).astype(int)
    h, w = er.shape
    ins = (px >= 0) & (py >= 0) & (px < w) & (py < h)
    free = np.zeros(px.shape, bool)
    free[ins] = er[py[ins], px[ins]] == 255
    n = len(pts)
    hi, lo = np.full(n, np.nan), np.full(n, np.nan)
    bh, bl = ~free[:, i0:], (~free[:, :i0 + 1])[:, ::-1]
    kh = np.where(bh.any(1), np.argmax(bh, 1), bh.shape[1])
    kl = np.where(bl.any(1), np.argmax(bl, 1), bl.shape[1])
    ok = free[:, i0]
    hi[ok] = (kh[ok] - 1) * 0.05 - 0.05
    lo[ok] = -((kl[ok] - 1) * 0.05 - 0.05)
    return lo, hi


def box(ref, i):
    return Obstacle(float(ref[i, 0]), float(ref[i, 1]), BOX_R)


def pair_span(n, ia=A_I, ib=B_I):
    return (np.arange(ia, ib + 1) if ib >= ia else np.arange(ia, ib + n + 1)) % n


def main():
    ref = load_ifac()
    cor = corridor_from_map(ref)
    p = ReoptParams()
    n = len(ref)
    print(f"map {MAP}: {n} stations | ReoptParams grid {p.grid_step_m} iters {p.iters_min}.."
          f"{p.max_iters} step_tol {p.step_tol_m} kappa {p.kappa_bound} w_veh {p.w_veh} "
          f"obs_margin {p.obs_margin}")

    # (1) -------------------------------------------------------------------------------------
    line0, off0, r0 = C.reoptimize_closed(ref, [], cor, p)
    print(f"\n(1) NO-OP      returned the input object: {line0 is ref[:, :2] or np.shares_memory(line0, ref)}"
          f" | max |offset| {np.max(np.abs(off0)):.3e} | ok={r0.ok} '{r0.reason}'")

    # (2) -------------------------------------------------------------------------------------
    obsA = [box(ref, A_I), box(ref, B_I)]
    t = time.perf_counter()
    lineA, offA, rA = C.reoptimize_closed(ref, obsA, cor, p)
    msA = (time.perf_counter() - t) * 1e3
    span = pair_span(n)
    inner = span[5:-5]
    holdA = float(np.min(np.abs(offA[inner])))
    print(f"\n(2) HOLD       ok={rA.ok} '{rA.reason}' | {msA:.0f} ms | iters {rA.iters} | "
          f"sides {rA.sides}")
    print(f"    int_k2 {rA.int_k2:.3f}  peak|kappa| {rA.peak_kappa:.3f}  "
          f"clearances {[round(c, 3) for c in rA.clearances]} (need {p.obs_margin + p.w_veh/2:.2f})")
    print(f"    min |offset| between the boxes = {holdA:.3f} m "
          f"({'HELD' if holdA > 0.25 else 'comes home -- W'})")

    # (3) -------------------------------------------------------------------------------------
    print(f"\n(3) LOCALITY   the straight pair's offset when a corner box is added")
    print("    corner station | ok | max |B-A| over the pair span | hold | clearances")
    for cs in CORNERS:
        obsB = obsA + [box(ref, cs)]
        lineB, offB, rB = C.reoptimize_closed(ref, obsB, cor, p)
        if not rB.ok:
            print(f"    {cs:14d} | NO | -- ({rB.reason[:44]})")
            continue
        dmm = float(np.max(np.abs(offB[span] - offA[span]))) * 1e3
        holdB = float(np.min(np.abs(offB[inner])))
        print(f"    {cs:14d} | ok | {dmm:24.2f} mm | {holdB:.3f} | "
              f"{[round(c,3) for c in rB.clearances]}")

    # (4) -------------------------------------------------------------------------------------
    print(f"\n(4) GRID       sensitivity (2 boxes / 3 boxes with the corner at {CORNERS[0]})")
    print("    step | 2 boxes: ok peak|k| clear        | 3 boxes: ok peak|k| clear")
    for step in (0.30, 0.50, 0.70):
        pg = ReoptParams(grid_step_m=step)
        _l2, _o2, r2 = C.reoptimize_closed(ref, obsA, cor, pg)
        _l3, _o3, r3 = C.reoptimize_closed(ref, obsA + [box(ref, CORNERS[0])], cor, pg)
        def fmt(r):
            if not r.ok:
                return f"NO  {r.reason[:34]}"
            return (f"yes {r.peak_kappa:5.3f} "
                    f"{min(r.clearances) if r.clearances else float('nan'):+.3f}")
        print(f"    {step:4.2f} | {fmt(r2):36s} | {fmt(r3)}")

    # (5) -------------------------------------------------------------------------------------
    print(f"\n(5) COST")
    times = []
    for _ in range(5):
        t = time.perf_counter()
        C.reoptimize_closed(ref, obsA, cor, p)
        times.append((time.perf_counter() - t) * 1e3)
    print(f"    2 boxes: p50 {np.percentile(times,50):.0f} ms  p95 {np.percentile(times,95):.0f} ms"
          f"  | {rA.n_coarse} coarse stations")
    print(f"    steps by iteration: {[round(s,4) for s in rA.steps]} (tol {p.step_tol_m})")
    print(f"    acceptable by iteration: {rA.accepted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
