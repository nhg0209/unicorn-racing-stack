#!/usr/bin/env python3
"""What G2 actually counts, and whether corridor_qp fails it for moving or for being steep.

G2 counts cycles whose |d| step at the car + 1 m point exceeds 5 cm. That point is where the
controller is steering for, so a step there is a steering step -- but the point MOVES. The car
advances 0.125 m per cycle at 2.5 m/s and 20 Hz, so between two cycles the metric is evaluated at
two different places on the track, and

    step = d_new(car_1 + 1 m) - d_old(car_0 + 1 m)

silently contains two different things added together:

    SPATIAL   d_old(car_1 + 1 m) - d_old(car_0 + 1 m)
              the SAME plan, read 0.125 m further along it. This is the path's own slope at the
              look-ahead point and nothing else -- a perfectly frozen path scores 0.125 * |d'|
              here, which at |d'| = 0.4 is 0.05 m, exactly the threshold.
    REPLAN    d_new(car_1 + 1 m) - d_old(car_1 + 1 m)
              the SAME point, two consecutive plans. This is instability: the controller's target
              moved without the car moving to it.

Only the second is what a consistency term could fix. If corridor_qp fails G2 on the first, the
gate is measuring the two methods' different d(s) distributions -- the quintic holds d ~ 0 through
the pre-ramp and spends its lateral movement in a short burst near the box, the QP spreads the
same movement over the whole window -- and adding w_prev would buy nothing while costing
feasibility.

The scenario is sweep_avoidance_stability's, verbatim, so the 17 reproduces here before it is
explained.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/diag_g2_step.py
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/spliner/scripts"))
import sweep_static_feasibility as F          # noqa: E402
import sweep_avoidance_stability as S         # noqa: E402


def drive(H, method):
    """S.run's scenario and S.run's arithmetic, keeping the whole published path per cycle."""
    san = H.san
    stamp = san.OTWpntArray().header.stamp
    clock = S._Clock()
    n = H._node(0.0, ladder=True)
    n.get_logger = lambda: _LOG
    n.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=int(clock.t * 1e9), to_msg=lambda: stamp))
    n.commit_enable, n._committed = True, None
    n.commit_dev_max = 0.6
    n.commit_reanchor_len_m, n.commit_reanchor_max_m = 2.0, 1.0
    n.commit_obs_ds, n.commit_obs_dd = 0.75, 0.40
    n.commit_drop_on_new_obstacle = True
    n.cur_vs = S.CAR_V
    n.static_plan_method = method
    n.ramp_search_max_ms = S.SHIPPED_LADDER_MS
    rec = {"fresh": 0}
    real_store = san.ObstacleSpliner._store_commit

    def store(self, *a, **k):
        rec["fresh"] += 1
        return real_store(self, *a, **k)
    n._store_commit = types.MethodType(store, n)
    n._publish_feasible = lambda ok: None

    L, wp = H.L, H.wp
    el = float(np.mean(np.hypot(np.diff([w["x_m"] for w in wp]), np.diff([w["y_m"] for w in wp]))))
    step_i = max(1, int(round(S.BOX_GAP_M / el)))
    boxes = []
    for k in range(S.N_BOX):
        j = (40 + k * step_i) % (len(wp) - 1)
        w = wp[j]
        boxes.append(types.SimpleNamespace(
            id=k + 1, s_start=(w["s_m"] - 0.15) % L, s_end=(w["s_m"] + 0.15) % L,
            s_center=w["s_m"], d_center=0.0, d_right=-0.15, d_left=0.15, size=0.30,
            vs=0.0, vd=0.0, is_static=True, is_visible=True,
            x_m=w["x_m"], y_m=w["y_m"], is_actually_a_gap=False))
    n.obstacles = boxes
    n._promoted = {b.id for b in boxes}

    dt = 1.0 / S.RATE_HZ
    cur_s, cur_d = 0.0, 0.0
    out = []
    for _ in range(int(S.LAPS * L / (S.CAR_V * dt))):
        clock.t += dt
        n.cur_s, n.cur_d = cur_s % L, cur_d
        resp = H.converter.get_cartesian(np.array([n.cur_s]), np.array([cur_d]))
        pxy = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)[0]
        n.cur_x, n.cur_y = float(pxy[0]), float(pxy[1])
        j = int(np.argmin(np.abs(H.s_arr - n.cur_s)))
        j2 = (j + 1) % (len(wp) - 1)
        n.cur_yaw = float(np.arctan2(wp[j2]["y_m"] - wp[j]["y_m"], wp[j2]["x_m"] - wp[j]["x_m"]))
        fresh0 = rec["fresh"]
        try:
            res = n.do_spline(H.gbw)
        except Exception:                                # noqa: BLE001
            out.append({"s": cur_s % L, "sm": None})
            cur_s += S.CAR_V * dt
            continue
        pts = res[0].wpnts if res and res[0] is not None else []
        sm = np.array([p.s_m for p in pts]) if pts else None
        dm = np.array([p.d_m for p in pts]) if pts else None
        d_look = _at(H, sm, dm, (cur_s % L))
        out.append({"s": cur_s % L, "sm": sm, "dm": dm, "d_look": d_look,
                    "fresh": rec["fresh"] > fresh0,
                    "committed": n._committed is not None,
                    "knots": tuple(sorted(oid for (oid, _s, _d)
                                          in (n._committed or {}).get("obs", []))),
                    "side": 0 if not pts else int(np.sign(max((p.d_m for p in pts), key=abs)))})
        cur_d += ((d_look if d_look is not None else 0.0) - cur_d) * min(1.0, dt / S.TRACK_TAU)
        cur_s += S.CAR_V * dt
    return out, L


class _Sink:
    def info(self, *a, **k): pass
    warn = warning = error = debug = info


_LOG = _Sink()


def _at(H, sm, dm, car_s, look=S.LOOK_M):
    """The published d at car + `look`, by S.run's own nearest-gap rule."""
    if sm is None or not len(sm):
        return None
    g = (sm - car_s) % H.L
    k = int(np.argmin(np.abs(g - look)))
    return float(dm[k]) if abs(g[k] - look) < 0.6 else None


def report(H, out, L, method):
    big = []
    steps = []
    for a, b in zip(out, out[1:]):
        if a.get("d_look") is None or b.get("d_look") is None:
            continue
        step = b["d_look"] - a["d_look"]
        # THE DECOMPOSITION. Same old plan, read at the NEW car's look-ahead point: that is the
        # part the car earned by moving. What is left is the part the re-plan moved under it.
        old_at_new = _at(H, a["sm"], a["dm"], b["s"])
        if old_at_new is None:
            continue
        spatial = old_at_new - a["d_look"]
        replan = b["d_look"] - old_at_new
        steps.append((abs(step), abs(spatial), abs(replan)))
        if abs(step) > S.BIG_STEP_M:
            big.append({"s": b["s"], "step": step, "spatial": spatial, "replan": replan,
                        "fresh": b["fresh"], "committed": b["committed"],
                        "knots_changed": b["knots"] != a["knots"],
                        "side_flip": bool(a["side"] and b["side"] and a["side"] != b["side"])})
    arr = np.array(steps) if steps else np.zeros((1, 3))
    print(f"\n--- {method} | {len(out)} cycles, {len(steps)} comparable pairs ---")
    print(f"  G2: {len(big)} cycles step > {S.BIG_STEP_M*100:.0f} cm   (gate {S.MAX_BIG_STEPS})"
          f"   {'FAIL' if len(big) > S.MAX_BIG_STEPS else 'pass'}")
    print(f"  G1: max step {arr[:, 0].max():.4f} m                (gate {S.MAX_STEP_M})")
    print(f"  over ALL pairs   |step| p50 {np.percentile(arr[:,0],50):.4f} "
          f"p90 {np.percentile(arr[:,0],90):.4f}")
    print(f"    of which SPATIAL (same plan, car moved 0.125 m): p50 "
          f"{np.percentile(arr[:,1],50):.4f}  p90 {np.percentile(arr[:,1],90):.4f}  "
          f"max {arr[:,1].max():.4f}")
    print(f"    of which REPLAN  (same point, new plan)        : p50 "
          f"{np.percentile(arr[:,2],50):.4f}  p90 {np.percentile(arr[:,2],90):.4f}  "
          f"max {arr[:,2].max():.4f}")
    if big:
        n_sp = sum(1 for b in big if abs(b["spatial"]) > abs(b["replan"]))
        print(f"  the {len(big)} over-threshold cycles: {n_sp} dominated by SPATIAL, "
              f"{len(big)-n_sp} by REPLAN")
        print(f"    fresh plan on {sum(1 for b in big if b['fresh'])}, "
              f"knot set changed on {sum(1 for b in big if b['knots_changed'])}, "
              f"side flip on {sum(1 for b in big if b['side_flip'])}, "
              f"committed on {sum(1 for b in big if b['committed'])}")
        print("      s        step    spatial     replan   fresh knots side")
        for b in big[:20]:
            print(f"    {b['s']:7.2f}  {b['step']:+.4f}  {b['spatial']:+.4f}  {b['replan']:+.4f}"
                  f"     {int(b['fresh'])}     {int(b['knots_changed'])}    {int(b['side_flip'])}")
    return len(big), arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="ifac")
    ap.add_argument("--method", nargs="*", default=["sample", "corridor_qp"])
    a = ap.parse_args()
    H = F.Harness(a.map)
    print(f"scenario: {S.N_BOX} boxes at {S.BOX_GAP_M} m, v={S.CAR_V} m/s, {S.RATE_HZ} Hz, "
          f"look={S.LOOK_M} m -> the car advances {S.CAR_V/S.RATE_HZ:.3f} m per cycle")
    for m in a.method:
        out, L = drive(H, m)
        report(H, out, L, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
