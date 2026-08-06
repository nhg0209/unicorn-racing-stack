#!/usr/bin/env python3
"""Does the closed-track re-opt QP do what the hump pipeline could not?

  (1) NO-OP      no obstacles -> is d exactly 0 and the input returned untouched?
  (2) HOLD       two boxes on the ifac straight (275, 360) -> hold >= 0.40 m, clearance >= 0.30 m
  (3) W2'        a corner box OUTSIDE the pair must not change the pair's BEHAVIOUR: hold, both
                 clearances and the excursion count all survive. |B-A| in mm is logged, not judged.
  (4) GRID       0.10 / 0.30 / 0.50 / 0.70 m -> CLEARANCE only. Curvature is grid-dependent by
                 construction (D2 ~ 1/ds^2) and is not asserted here.
  (5) FIELD      twelve three-box placements: every box either cleared by 0.30 m or named
                 infeasible with the station, the side and the metres it fell short.
  (6) SAG        what the periodic-cubic upsample costs, measured with disc_allow_m = 0, against
                 the default that has to cover it.
  (7) W SWEEP    hold / peak|kappa| / clearance / ms across w -- the one shape knob.

  ~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/bench_closed_reopt.py
"""
import json
import sys
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
TRIOS = ((275, 360, 60), (275, 360, 120), (275, 360, 200),
         (20, 60, 200), (40, 120, 300), (100, 160, 260),
         (150, 210, 330), (200, 260, 40), (250, 310, 90),
         (300, 20, 140), (330, 30, 180), (10, 90, 190))
DEADBAND = 0.02   # [m] an excursion is a lateral move worth naming, not solver noise


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


def excursions(d, span, deadband=DEADBAND):
    """How many times the offset crosses the line over this span, ignoring noise."""
    sig = np.sign(np.where(np.abs(d[span]) < deadband, 0.0, d[span]))
    sig = sig[sig != 0]
    return int(np.count_nonzero(np.diff(sig) != 0)) if len(sig) else 0


def hold_and_clear(rep):
    return rep.hold, (min(rep.clearances) if rep.clearances else float("nan"))


def main():
    ref = load_ifac()
    cor = corridor_from_map(ref)
    p = ReoptParams()
    n = len(ref)
    need = p.obs_margin + p.w_veh / 2.0
    print(f"map {MAP}: {n} stations | grid {p.grid_step_m} w {p.dev_weight} obs_margin "
          f"{p.obs_margin} w_veh {p.w_veh} disc_allow {p.disc_allow_m} kappa(report) "
          f"{p.kappa_report_only}")
    print(f"    acceptance clearance = obs_margin + w_veh/2 = {need:.3f} m "
          f"(disc_allow_m is NOT in it)")

    # (1) -------------------------------------------------------------------------------------
    line0, d0, r0 = C.reoptimize_closed(ref, [], cor, p)
    same = line0 is ref[:, :2] or np.shares_memory(line0, ref)
    print(f"\n(1) NO-OP      input object returned: {same} | max |d| {np.max(np.abs(d0)):.3e} | "
          f"clean peak|kappa| {r0.clean_peak_kappa:.3f} @ station {r0.clean_peak_station}")

    # (2) -------------------------------------------------------------------------------------
    obsA = [box(ref, A_I), box(ref, B_I)]
    lineA, dA, rA = C.reoptimize_closed(ref, obsA, cor, p)
    hA, cA = hold_and_clear(rA)
    span = pair_span(n)
    eA = excursions(dA, span)
    print(f"\n(2) HOLD       ok={rA.ok} | {rA.solve_ms:.1f} ms | {rA.n_coarse} coarse")
    print(f"    hold {hA:.3f} m (need >= 0.40) | clearances {[round(c, 3) for c in rA.clearances]} "
          f"(need >= {need:.3f}) | max|d| {rA.max_offset:.3f} | sag {rA.sag_mm:+.2f} mm")
    print(f"    peak|kappa| {rA.peak_kappa:.3f} @ {rA.peak_station} (clean {rA.clean_peak_kappa:.3f}"
          f" @ {rA.clean_peak_station}; near an obstacle: {rA.peak_near_obstacle})")
    print(f"    -> {'PASS' if (hA >= 0.40 and cA >= need - 1e-9) else 'FAIL'}")

    # (3) -------------------------------------------------------------------------------------
    print(f"\n(3) W2'        a corner box outside the pair must not change the pair's behaviour")
    print(f"    baseline: hold {hA:.3f} | clearances {[round(c, 3) for c in rA.clearances]} | "
          f"{eA} excursion(s) over the span")
    print("    corner | hold  | min clear | exc | verdict | |B-A| (log only)")
    for cs in CORNERS:
        _lB, dB, rB = C.reoptimize_closed(ref, obsA + [box(ref, cs)], cor, p)
        hB, cB = hold_and_clear(rB)
        # the pair's own clearances are the first two entries; the corner box is the third
        cpair = min(rB.clearances[:2]) if len(rB.clearances) >= 2 else float("nan")
        eB = excursions(dB, span)
        ok = (hB >= 0.40) and (cpair >= need - 1e-9) and (eB == eA)
        dmm = float(np.max(np.abs(dB[span] - dA[span]))) * 1e3
        print(f"    {cs:6d} | {hB:5.3f} | {cpair:+9.3f} | {eB:3d} | {'PASS' if ok else 'FAIL':7s} | "
              f"{dmm:8.3f} mm")

    # (4) -------------------------------------------------------------------------------------
    print(f"\n(4) GRID       clearance only (need >= {need:.3f}); kappa is logged, not asserted")
    print("    step | min clearance | verdict | hold  | peak|kappa| (log) | sag mm | ms")
    for step in (0.10, 0.30, 0.50, 0.70):
        _l, _d, rg = C.reoptimize_closed(ref, obsA, cor, ReoptParams(grid_step_m=step))
        hg, cg = hold_and_clear(rg)
        print(f"    {step:4.2f} | {cg:+13.3f} | {'PASS' if cg >= need - 1e-9 else 'FAIL':7s} | "
              f"{hg:5.3f} | {rg.peak_kappa:17.3f} | {rg.sag_mm:+6.2f} | {rg.solve_ms:.1f}")

    # (5) -------------------------------------------------------------------------------------
    print(f"\n(5) FIELD      cleared by {need:.3f} m, OR classified infeasible with a reason")
    print("    boxes            | clear | infeas | peak|kappa| | hold  | verdict")
    bad = 0
    for trio in TRIOS:
        _l, _d, r = C.reoptimize_closed(ref, [box(ref, i) for i in trio], cor, p)
        cg = min(r.clearances) if r.clearances else float("inf")
        ok = r.ok and (cg >= need - 1e-9)
        bad += 0 if ok else 1
        print(f"    {str(trio):16s} | {cg:+5.3f} | {len(r.infeasible):6d} | {r.peak_kappa:11.3f} | "
              f"{r.hold:5.3f} | {'PASS' if ok else 'FAIL'}")
        for why in r.infeasible_why:
            print(f"                     -> {why}")
    print(f"    -> {len(TRIOS) - bad}/{len(TRIOS)}")

    # (6) -------------------------------------------------------------------------------------
    print(f"\n(6) SAG        keep-out violation the cubic upsample introduces, disc_allow_m = 0")
    print("    grid | worst sag over the 2-box + 12 field cases | default disc_allow_m | verdict")
    for step in (0.30, 0.50, 0.70):
        p0 = ReoptParams(grid_step_m=step, disc_allow_m=0.0)
        worst = -np.inf
        for obs in [obsA] + [[box(ref, i) for i in t] for t in TRIOS]:
            _l, _d, r = C.reoptimize_closed(ref, obs, cor, p0)
            worst = max(worst, r.sag_mm)
        cov = p.disc_allow_m * 1e3 >= worst
        print(f"    {step:4.2f} | {worst:+41.2f} mm | {p.disc_allow_m * 1e3:19.1f} mm | "
              f"{'COVERED' if cov else 'NOT COVERED'}")
    print("    what a larger allowance costs (grid 0.50, the 12 field cases):")
    print("    disc_allow | worst clearance | cleared | infeasible boxes")
    for da in (0.005, 0.010, 0.020):
        pa = ReoptParams(disc_allow_m=da)
        cs, ninf, nok = [], 0, 0
        for t in TRIOS:
            _l, _d, r = C.reoptimize_closed(ref, [box(ref, i) for i in t], cor, pa)
            cs.append(min(r.clearances) if r.clearances else float("inf"))
            ninf += len(r.infeasible)
            nok += 1 if (r.ok and cs[-1] >= need - 1e-9) else 0
        print(f"    {da:10.3f} | {min(cs):+15.3f} | {nok:2d}/{len(TRIOS)} | {ninf}")

    # (7) -------------------------------------------------------------------------------------
    print("\n(7) W SWEEP    the one shape knob (two boxes, grid 0.50)")
    print("    w      | hold  | peak|kappa| | min clear | max|d| | far |d| p50 | ms")
    far = np.min([np.hypot(ref[:, 0] - o.x, ref[:, 1] - o.y) for o in obsA], axis=0) > 2.5
    best = None
    for w in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1):
        _l, dw, rw = C.reoptimize_closed(ref, obsA, cor, ReoptParams(dev_weight=w))
        hw, cw = hold_and_clear(rw)
        print(f"    {w:6.3f} | {hw:5.3f} | {rw.peak_kappa:11.3f} | {cw:+9.3f} | "
              f"{rw.max_offset:6.3f} | {np.median(np.abs(dw[far])):12.3f} | {rw.solve_ms:.1f}")
        if hw >= 0.40 and (best is None or rw.peak_kappa < best[1]):
            best = (w, rw.peak_kappa)
    print(f"    -> lowest peak|kappa| among the w that still hold 0.40: w = {best[0]} "
          f"(peak {best[1]:.3f})" if best else "    -> no w holds 0.40")
    return 0


if __name__ == "__main__":
    sys.exit(main())
