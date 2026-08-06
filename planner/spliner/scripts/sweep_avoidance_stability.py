#!/usr/bin/env python3
"""How STEADY is the published avoidance path while the car drives past a field of boxes?

Feasibility says a path exists; this asks what the controller is handed cycle to cycle. It drives
the REAL do_spline around the REAL map at a fixed speed with a first-order tracking lag, four
boxes on the raceline, and measures the things that make a car wobble:

  STEP        |d_published(car + 1 m)| between consecutive cycles. The reference the controller is
              tracking, at the point it is steering for. A step here is a steering step.
  REPLANS     cycles that threw the committed path away and solved a new one. Each one is a fresh
              geometry laid under a moving car.
  TOGGLES     feasible True<->False edges. Each False is the state machine's cue to abandon the
              overtake.
  FLIPS       left/right side changes WITHIN one maneuver (same knot set, committed throughout) --
              a maneuver that changes its mind mid-excursion. A knot set that changed is a box
              arriving or leaving, and the plan after it is a different maneuver.
  d_end JUMP  between consecutive fresh plans for the SAME knot set: how far the chosen exit
              offset moves when nothing about the obstacles changed.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_avoidance_stability.py --check
"""
import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/spliner/scripts"))
import sweep_static_feasibility as F  # noqa: E402

# --- gates ---------------------------------------------------------------------------------
MAX_STEP_M = 0.15          # G1  worst |d| step at the car+1 m point
MAX_BIG_STEPS = 4          # G2  cycles whose step exceeds 5 cm
MAX_TOGGLES = 8            # G3  feasible edges
MAX_REPLANS = 20           # G4  fresh plans
MAX_FLIPS = 0              # G5  side flips while committed
MAX_DEND_JUMP_M = 0.15     # G6  |d_end| change between fresh plans with the same knot set
BIG_STEP_M = 0.05
LOOK_M = 1.0               # where the step is measured, ahead of the car
CAR_V = 2.5                # [m/s]
TRACK_TAU = 0.30           # [s] first-order lag of the car's lateral position onto the reference
RATE_HZ = 20.0
N_BOX, BOX_GAP_M, LAPS = 4, 6.0, 2.0


class _Clock:
    def __init__(self): self.t = 0.0
    def now(self):
        return types.SimpleNamespace(nanoseconds=int(self.t * 1e9),
                                     to_msg=lambda: F.load_node_module.__self__ if False else None)


def run(H, verbose=False):
    """Drive one car around the lap and return the per-cycle record."""
    san = H.san
    stamp = san.OTWpntArray().header.stamp
    clock = _Clock()
    n = H._node(0.0, ladder=True)
    n.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=int(clock.t * 1e9), to_msg=lambda: stamp))
    # the shipped commit tunables (static_avoidance_params.yaml); the feasibility harness runs
    # with commits OFF, and this sweep is about exactly what they do
    n.commit_enable = True
    n._committed = None
    n.commit_dev_max = 0.6
    n.commit_reanchor_len_m, n.commit_reanchor_max_m = 2.0, 1.0
    n.commit_obs_ds, n.commit_obs_dd = 0.75, 0.40
    n.commit_drop_on_new_obstacle = True
    n.cur_vs = CAR_V
    # feasibility verdict + fresh-plan counter, observed rather than inferred
    rec = {"feas": [], "fresh": 0, "released": []}
    n._publish_feasible = lambda ok: rec["feas"].append(bool(ok))
    real_store = san.ObstacleSpliner._store_commit

    def store(self, *a, **k):
        rec["fresh"] += 1
        return real_store(self, *a, **k)
    n._store_commit = types.MethodType(store, n)

    # four boxes on the raceline, evenly spaced
    L, wp = H.L, H.wp
    el = float(np.mean(np.hypot(np.diff([w["x_m"] for w in wp]), np.diff([w["y_m"] for w in wp]))))
    step_i = max(1, int(round(BOX_GAP_M / el)))
    i0 = 40
    boxes = []
    for k in range(N_BOX):
        j = (i0 + k * step_i) % (len(wp) - 1)
        w = wp[j]
        boxes.append(types.SimpleNamespace(
            id=k + 1, s_start=(w["s_m"] - 0.15) % L, s_end=(w["s_m"] + 0.15) % L,
            s_center=w["s_m"], d_center=0.0, d_right=-0.15, d_left=0.15, size=0.30,
            vs=0.0, vd=0.0, is_static=True, is_visible=True,
            x_m=w["x_m"], y_m=w["y_m"], is_actually_a_gap=False))
    n.obstacles = boxes
    n._promoted = {b.id for b in boxes}

    dt = 1.0 / RATE_HZ
    cur_s, cur_d = 0.0, 0.0
    out = []
    prev_d_at_look = None
    n_cycles = int(LAPS * L / (CAR_V * dt))
    for _ in range(n_cycles):
        clock.t += dt
        n.cur_s, n.cur_d = cur_s % L, cur_d
        resp = H.converter.get_cartesian(np.array([n.cur_s]), np.array([cur_d]))
        pxy = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)[0]
        n.cur_x, n.cur_y = float(pxy[0]), float(pxy[1])
        j = int(np.argmin(np.abs(H.s_arr - n.cur_s)))
        j2 = (j + 1) % (len(wp) - 1)
        n.cur_yaw = float(np.arctan2(wp[j2]["y_m"] - wp[j]["y_m"], wp[j2]["x_m"] - wp[j]["x_m"]))
        fresh0, nfeas0 = rec["fresh"], len(rec["feas"])
        try:
            res = n.do_spline(H.gbw)
        except Exception as exc:                        # a crash is a failure, not a skip
            out.append({"s": cur_s, "err": repr(exc)})
            cur_s += CAR_V * dt
            continue
        pts = res[0].wpnts if res and res[0] is not None else []
        # the reference at car + LOOK_M, which is what the controller is steering for
        d_look = None
        if pts:
            ds = [((x.s_m - n.cur_s) % L) for x in pts]
            k = int(np.argmin([abs(v - LOOK_M) for v in ds]))
            if abs(ds[k] - LOOK_M) < 0.6:
                d_look = float(pts[k].d_m)
        row = {"s": cur_s, "n": len(pts), "d_look": d_look,
               "fresh": rec["fresh"] > fresh0,
               "feas": rec["feas"][-1] if len(rec["feas"]) > nfeas0 else None,
               "d_end": float(pts[-1].d_m) if pts else None,
               "side": (0 if not pts else int(np.sign(max((x.d_m for x in pts), key=abs)))),
               "committed": n._committed is not None,
               "knots": tuple(sorted(oid for (oid, _s, _d) in (n._committed or {}).get('obs', []))),
               # CONSECUTIVE cycles only. Comparing across a gap of blank cycles measures two
               # different maneuvers a lap apart, not a step the controller was handed.
               "step": (abs(d_look - prev_d_at_look)
                        if (d_look is not None and prev_d_at_look is not None) else None)}
        out.append(row)
        prev_d_at_look = d_look
        # the car lags the reference (first-order), which is what makes a step a real disturbance
        target = d_look if d_look is not None else 0.0
        cur_d += (target - cur_d) * min(1.0, dt / TRACK_TAU)
        cur_s += CAR_V * dt
    return out


def report(out, label):
    steps = [r["step"] for r in out if r.get("step") is not None]
    big = [r for r in out if (r.get("step") or 0.0) > BIG_STEP_M]
    fresh = [r for r in out if r.get("fresh")]
    feas = [r["feas"] for r in out if r.get("feas") is not None]
    toggles = sum(1 for a, b in zip(feas, feas[1:]) if a != b)
    # A flip counts only WITHIN ONE MANEUVER: same knot set, committed on both cycles, no blank
    # in between. A blank ends the excursion, and a knot set that changed means a box arrived or
    # left -- the plan that follows is a different maneuver, not this one changing its mind. Scoped
    # exactly like the d_end gate below, which is the same question asked of the exit offset.
    flips = 0
    prev = None
    for r in out:
        side = r.get("side") or 0
        if not side or not r.get("committed"):
            prev = None
            continue
        if prev is not None and prev[0] == r["knots"] and side != prev[1]:
            flips += 1
        prev = (r["knots"], side)
    dend_jump = 0.0
    prev = None
    for r in fresh:
        if r.get("d_end") is None:
            continue
        if prev is not None and prev[0] == r["knots"]:
            dend_jump = max(dend_jump, abs(r["d_end"] - prev[1]))
        prev = (r["knots"], r["d_end"])
    errs = [r for r in out if "err" in r]
    blank = [r for r in out if r.get("n") == 0]
    print(f"=== {label} | {len(out)} cycles, {N_BOX} boxes, v={CAR_V} m/s, tau={TRACK_TAU} s ===")
    print(f"  max step at car+{LOOK_M:.0f} m   {max(steps) if steps else 0.0:.3f} m   (gate {MAX_STEP_M})")
    print(f"  steps over {BIG_STEP_M*100:.0f} cm        {len(big):3d}     (gate {MAX_BIG_STEPS})")
    print(f"  feasible toggles       {toggles:3d}     (gate {MAX_TOGGLES})")
    print(f"  fresh plans            {len(fresh):3d}     (gate {MAX_REPLANS})")
    print(f"  side flips committed   {flips:3d}     (gate {MAX_FLIPS})")
    print(f"  max d_end jump         {dend_jump:.3f} m   (gate {MAX_DEND_JUMP_M}, same knot set)")
    print(f"  blank publications     {len(blank):3d}     | exceptions {len(errs)}")
    return {"step": max(steps) if steps else 0.0, "big": len(big), "toggles": toggles,
            "fresh": len(fresh), "flips": flips, "dend": dend_jump, "err": len(errs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="ifac")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    H = F.Harness(a.map)
    m = report(run(H), f"stability | map {a.map}")
    fails = []
    if m["step"] > MAX_STEP_M:
        fails.append(f"G1 max published step {m['step']:.3f} m > {MAX_STEP_M}")
    if m["big"] > MAX_BIG_STEPS:
        fails.append(f"G2 {m['big']} cycles step more than {BIG_STEP_M*100:.0f} cm > {MAX_BIG_STEPS}")
    if m["toggles"] > MAX_TOGGLES:
        fails.append(f"G3 {m['toggles']} feasible toggles > {MAX_TOGGLES}")
    if m["fresh"] > MAX_REPLANS:
        fails.append(f"G4 {m['fresh']} fresh plans > {MAX_REPLANS}")
    if m["flips"] > MAX_FLIPS:
        fails.append(f"G5 {m['flips']} side flips while committed > {MAX_FLIPS}")
    if m["dend"] > MAX_DEND_JUMP_M:
        fails.append(f"G6 max d_end jump {m['dend']:.3f} m > {MAX_DEND_JUMP_M}")
    if m["err"]:
        fails.append(f"{m['err']} cycles raised")
    if fails:
        print("\nFAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1 if a.check else 0
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
