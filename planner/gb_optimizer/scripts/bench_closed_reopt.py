#!/usr/bin/env python3
"""Does the closed-track minimum-curvature re-opt do what the hump pipeline could not?

  (1) NO-OP      no obstacles -> is d exactly 0 and the input returned untouched?
  (2) HOLD       two boxes on the ifac straight (275, 360) -> hold >= 0.40 m, clearance >= 0.30 m
  (3) LOCALITY   a corner box OUTSIDE the pair (60/120/200) -> |B-A| over the pair span [mm]
  (4) GRID       0.10 / 0.30 / 0.50 / 0.70 m -> clearance >= 0.30 at every one of them
  (5) FIELD      twelve three-box placements: clearance, peak |kappa|, hold

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
CORNERS = (60, 120, 200)   # OUTSIDE the pair: 300 sits BETWEEN 275 and 360


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
    _psi, nv, _tan = C._frame(pts)
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
    print(f"map {MAP}: {n} stations | ReoptParams grid {p.grid_step_m} w {p.dev_weight} "
          f"obs_margin {p.obs_margin} w_veh {p.w_veh} kappa(report) {p.kappa_report_only}")
    need = p.obs_margin + p.w_veh / 2.0

    # (1) -------------------------------------------------------------------------------------
    line0, d0, r0 = C.reoptimize_closed(ref, [], cor, p)
    same = line0 is ref[:, :2] or np.shares_memory(line0, ref)
    print(f"\n(1) NO-OP      input object returned: {same} | max |d| {np.max(np.abs(d0)):.3e} | "
          f"clean peak|kappa| {r0.clean_peak_kappa:.3f} @ station {r0.clean_peak_station}")

    # (2) -------------------------------------------------------------------------------------
    obsA = [box(ref, A_I), box(ref, B_I)]
    lineA, dA, rA = C.reoptimize_closed(ref, obsA, cor, p)
    print(f"\n(2) HOLD       ok={rA.ok} '{rA.reason}' | {rA.solve_ms:.1f} ms | {rA.n_coarse} coarse")
    print(f"    hold {rA.hold:.3f} m (need >= 0.40) | clearances "
          f"{[round(c,3) for c in rA.clearances]} (need >= {need:.2f}) | max|d| {rA.max_offset:.3f}")
    print(f"    peak|kappa| {rA.peak_kappa:.3f} @ {rA.peak_station} "
          f"(clean {rA.clean_peak_kappa:.3f} @ {rA.clean_peak_station}; "
          f"near an obstacle: {rA.peak_near_obstacle})")

    # (3) -------------------------------------------------------------------------------------
    span = pair_span(n)
    print(f"\n(3) LOCALITY   corner box added OUTSIDE the pair")
    print("    corner | ok | max |B-A| over the pair span | hold | min clearance")
    for cs in CORNERS:
        _lB, dB, rB = C.reoptimize_closed(ref, obsA + [box(ref, cs)], cor, p)
        if not rB.ok:
            print(f"    {cs:6d} | NO | {rB.reason[:40]}")
            continue
        dmm = float(np.max(np.abs(dB[span] - dA[span]))) * 1e3
        print(f"    {cs:6d} | ok | {dmm:24.3f} mm | {rB.hold:.3f} | {min(rB.clearances):+.3f}")

    # (4) -------------------------------------------------------------------------------------
    print(f"\n(4) GRID       clearance must hold at every spacing (need >= {need:.2f})")
    print("    step | ok | min clearance | hold  | peak|kappa| | ms")
    for step in (0.10, 0.30, 0.50, 0.70):
        pg = ReoptParams(grid_step_m=step)
        _l, _d, rg = C.reoptimize_closed(ref, obsA, cor, pg)
        mc = min(rg.clearances) if rg.clearances else float("nan")
        flag = "ok" if (rg.ok and mc >= need - 1e-9) else "NO"
        print(f"    {step:4.2f} | {flag} | {mc:+13.3f} | {rg.hold:.3f} | {rg.peak_kappa:11.3f} | "
              f"{rg.solve_ms:.1f}")

    # (5) -------------------------------------------------------------------------------------
    print(f"\n(5) FIELD      twelve three-box placements")
    print("    boxes            | ok | min clear | peak|kappa| | near obs | hold  | ms")
    fails = 0
    for trio in ((275, 360, 60), (275, 360, 120), (275, 360, 200),
                 (20, 60, 200), (40, 120, 300), (100, 160, 260),
                 (150, 210, 330), (200, 260, 40), (250, 310, 90),
                 (300, 20, 140), (330, 30, 180), (10, 90, 190)):
        obs = [box(ref, i) for i in trio]
        _l, _d, r = C.reoptimize_closed(ref, obs, cor, p)
        mc = min(r.clearances) if r.clearances else float("nan")
        ok = r.ok and mc >= need - 1e-9
        fails += 0 if ok else 1
        print(f"    {str(trio):16s} | {'ok' if ok else 'NO'} | {mc:+9.3f} | {r.peak_kappa:11.3f} | "
              f"{str(r.peak_near_obstacle):8s} | {r.hold:5.3f} | {r.solve_ms:.1f}")
    print(f"    -> {12 - fails}/12 clear every box by {need:.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
