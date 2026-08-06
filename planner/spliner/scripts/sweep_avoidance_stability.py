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
SHIPPED_LADDER_MS = 20.0   # static_avoidance_params.yaml: ramp_search_max_ms
# --- ENGAGE scenario (G7/G8): a pair of boxes on the ifac straight ------------------------------
ENGAGE_ANCHOR_S = 29.0
ENGAGE_SPACINGS = (3.0, 5.0, 7.0, 9.0, 12.0)
ENGAGE_BIASES = (0.0, 0.3)
MIN_ENGAGE_GAP_M = 6.0     # G7  how far out the second box is first shaped around
MIN_ENGAGE_RAMP_M = 2.0    # G8  how much of that the hump's entry ramp gets
# --- what the PUBLISHED path itself asks of the car -------------------------------------------
# The planner's own profile is solved with the accel limits; these check the result, station by
# station, so a change to the smoothing, the resample or the velocity pass cannot quietly hand the
# controller a discontinuity. The station-spacing bound (G13) is the one that catches a harness or
# a frame convention folding a path onto itself at the s seam.
MAX_PATH_A_LONG = 6.0      # G9   [m/s^2] demanded by the published speed plan (ggv ax_max 5.0)
MAX_PATH_DVX = 0.30        # G10  [m/s] between adjacent published stations
MAX_PATH_DKAPPA = 0.80     # G11  [1/m] between adjacent published stations
SPACING_LO, SPACING_HI = 0.5, 2.0   # G13  station gap, as a fraction of the nominal spacing


class _Clock:
    def __init__(self): self.t = 0.0
    def now(self):
        return types.SimpleNamespace(nanoseconds=int(self.t * 1e9),
                                     to_msg=lambda: F.load_node_module.__self__ if False else None)


def _boxes(H, anchor_s, spacings, d=0.0, r=0.15):
    """Boxes on the raceline at `anchor_s` and `anchor_s + cumulative(spacings)`."""
    L, wp = H.L, H.wp
    s_arr = H.s_arr
    out, s_here = [], float(anchor_s)
    for k, gap in enumerate([0.0] + list(spacings)):
        s_here = (s_here + gap) % L
        j = int(np.argmin(np.abs(s_arr - s_here)))
        w = wp[j]
        out.append(types.SimpleNamespace(
            id=k + 1, s_start=(w["s_m"] - r) % L, s_end=(w["s_m"] + r) % L, s_center=w["s_m"],
            d_center=d, d_right=d - r, d_left=d + r, size=2 * r, vs=0.0, vd=0.0,
            is_static=True, is_visible=True, x_m=w["x_m"], y_m=w["y_m"], is_actually_a_gap=False))
    return out


def engage_run(H, spacing, bias, anchor_s=29.0, start_back_m=25.0):
    """Drive up to a PAIR of boxes and report when the second one is first shaped around.

    ENGAGE GAP is the distance to box 2 when the published path first bends around it, and ENGAGE
    RAMP is how much of that the hump's own entry ramp occupies. Both are what the driver feels as
    "it reacted late": a plan that appears 1.5 m from a box is a plan that arrives after the
    braking decision, however feasible it is.
    """
    san = H.san
    stamp = san.OTWpntArray().header.stamp
    clock = _Clock()
    n = _driver_node(H, clock, stamp)
    n.cur_d = bias
    boxes = _boxes(H, anchor_s, [spacing], d=0.0)
    n.obstacles = boxes
    n._promoted = {b.id for b in boxes}
    L = H.L
    s2 = boxes[1].s_center
    keep_out = n.width_car / 2.0 + n.safety_margin_d + 0.15      # box half-width + lateral keep-out
    dt = 1.0 / RATE_HZ
    cur_s = (boxes[0].s_center - start_back_m) % L
    cur_d = bias
    engage_gap = engage_ramp = None
    for _ in range(int((start_back_m + spacing + 4.0) / (CAR_V * dt))):
        clock.t += dt
        n.cur_s, n.cur_d = cur_s % L, cur_d
        resp = H.converter.get_cartesian(np.array([n.cur_s]), np.array([cur_d]))
        pxy = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)[0]
        n.cur_x, n.cur_y = float(pxy[0]), float(pxy[1])
        j = int(np.argmin(np.abs(H.s_arr - n.cur_s)))
        j2 = (j + 1) % (len(H.wp) - 1)
        n.cur_yaw = float(np.arctan2(H.wp[j2]["y_m"] - H.wp[j]["y_m"],
                                     H.wp[j2]["x_m"] - H.wp[j]["x_m"]))
        try:
            res = n.do_spline(H.gbw)
        except Exception:
            cur_s += CAR_V * dt
            continue
        pts = res[0].wpnts if res and res[0] is not None else []
        if pts and engage_gap is None:
            # SHAPED means the path would actually pass box 2: its offset at box 2's station is
            # outside that box's keep-out. Any looser test (some offset carried over from box 1's
            # hump, say) reports a merged pair as engaged when nothing was planned for box 2.
            near = [x for x in pts
                    if abs(((x.s_m - s2 + L / 2.0) % L) - L / 2.0) <= 0.30]
            if near and max(abs(x.d_m) for x in near) > keep_out:
                engage_gap = ((s2 - n.cur_s) % L)
                # box 2's OWN entry ramp: walk back from it while the path still carries offset,
                # stopping at box 1 so a merged pair does not report the whole excursion
                cap = min(spacing, L / 2.0)
                back = sorted((((s2 - x.s_m) % L), abs(x.d_m)) for x in pts
                              if ((s2 - x.s_m) % L) <= cap)
                engage_ramp = 0.0
                for ds, ad in back:
                    if ad <= 0.02:
                        break
                    engage_ramp = ds
        target = 0.0
        if pts:
            ds = [((x.s_m - n.cur_s) % L) for x in pts]
            k = int(np.argmin([abs(v - LOOK_M) for v in ds]))
            if abs(ds[k] - LOOK_M) < 0.6:
                target = float(pts[k].d_m)
        cur_d += (target - cur_d) * min(1.0, dt / TRACK_TAU)
        cur_s += CAR_V * dt
    return engage_gap, engage_ramp


def _driver_node(H, clock, stamp):
    """A node wired for a driving run: real commit machinery, stubbed clock and publisher.

    The feasibility harness stubs _store_commit out -- it plans one cycle at a time and has no use
    for a commit -- so a driving run has to put the real one back. Without it `_committed` stays
    None, `_reuse_committed` never runs, and every cycle is a fresh plan: the exact opposite of
    what a commit-behaviour measurement is for.
    """
    n = H._node(0.0, ladder=True)
    n._store_commit = types.MethodType(H.san.ObstacleSpliner._store_commit, n)
    n.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=int(clock.t * 1e9), to_msg=lambda: stamp))
    n.commit_enable = True
    n._committed = None
    n.commit_dev_max = 0.6
    n.commit_reanchor_len_m, n.commit_reanchor_max_m = 2.0, 1.0
    n.commit_obs_ds, n.commit_obs_dd = 0.75, 0.40
    n.commit_drop_on_new_obstacle = True
    n.commit_replan_gap_m = getattr(n, "commit_replan_gap_m", 7.0)
    n.ramp_search_max_ms = SHIPPED_LADDER_MS
    n.cur_vs = CAR_V
    n._publish_feasible = lambda ok: None
    return n


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
        # what the published path asks of the car, station by station (G9-G11, G13)
        a_long = dvx = dkap = 0.0
        gap_lo = gap_hi = None
        if len(pts) > 2:
            vx = np.array([x.vx_mps for x in pts])
            kap = np.array([x.kappa_radpm for x in pts])
            pxy2 = np.array([[x.x_m, x.y_m] for x in pts])
            step = np.hypot(*np.diff(pxy2, axis=0).T)
            ok = step > 1e-9
            if ok.any():
                a_long = float(np.max(np.abs(vx[1:][ok] ** 2 - vx[:-1][ok] ** 2)
                                      / (2.0 * step[ok])))
                gap_lo = float(np.min(step)) / H.nominal_ds
                gap_hi = float(np.max(step)) / H.nominal_ds
            dvx = float(np.max(np.abs(np.diff(vx))))
            dkap = float(np.max(np.abs(np.diff(kap))))
        row = {"a_long": a_long, "dvx": dvx, "dkap": dkap,
               "gap_lo": gap_lo, "gap_hi": gap_hi,
               "s": cur_s, "n": len(pts), "d_look": d_look,
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
    a_long = max((r.get("a_long") or 0.0) for r in out) if out else 0.0
    dvx = max((r.get("dvx") or 0.0) for r in out) if out else 0.0
    dkap = max((r.get("dkap") or 0.0) for r in out) if out else 0.0
    los = [r["gap_lo"] for r in out if r.get("gap_lo") is not None]
    his = [r["gap_hi"] for r in out if r.get("gap_hi") is not None]
    gap_lo = min(los) if los else 1.0
    gap_hi = max(his) if his else 1.0
    print(f"  published |a_long| max {a_long:6.2f} m/s^2   (gate {MAX_PATH_A_LONG})")
    print(f"  published |dvx|   max {dvx:6.3f} m/s     (gate {MAX_PATH_DVX})")
    print(f"  published |dkappa| max {dkap:6.3f} 1/m    (gate {MAX_PATH_DKAPPA})")
    print(f"  station gap        {gap_lo:5.3f}-{gap_hi:5.3f} x nominal (gate "
          f"{SPACING_LO}-{SPACING_HI})")
    return {"step": max(steps) if steps else 0.0, "big": len(big), "toggles": toggles,
            "fresh": len(fresh), "flips": flips, "dend": dend_jump, "err": len(errs),
            "a_long": a_long, "dvx": dvx, "dkap": dkap, "gap_lo": gap_lo, "gap_hi": gap_hi}


def engage_report(H):
    """G7/G8: how early the SECOND of a pair is reacted to, across spacings and tracking bias."""
    rows, gaps, ramps = [], [], []
    for spacing in ENGAGE_SPACINGS:
        for bias in ENGAGE_BIASES:
            g, r = engage_run(H, spacing, bias)
            rows.append((spacing, bias, g, r))
            if g is not None:
                gaps.append(g)
                ramps.append(r if r is not None else 0.0)
    print(f"=== engage | pair on the straight (s={ENGAGE_ANCHOR_S}), "
          f"{len(ENGAGE_SPACINGS)} spacings x {len(ENGAGE_BIASES)} biases ===")
    print("  spacing  bias |  engage gap   entry ramp")
    for spacing, bias, g, r in rows:
        print(f"   {spacing:5.1f} {bias:5.2f} | "
              + (f"{g:9.2f} m {r:10.2f} m" if g is not None
                 else "     never          -   "))
    return (min(gaps) if gaps else 0.0), (min(ramps) if ramps else 0.0), rows


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
    if m["a_long"] > MAX_PATH_A_LONG:
        fails.append(f"G9 the published speed plan demands {m['a_long']:.2f} m/s^2 "
                     f"> {MAX_PATH_A_LONG}")
    if m["dvx"] > MAX_PATH_DVX:
        fails.append(f"G10 published |dvx| {m['dvx']:.3f} m/s between stations > {MAX_PATH_DVX}")
    if m["dkap"] > MAX_PATH_DKAPPA:
        fails.append(f"G11 published |dkappa| {m['dkap']:.3f} 1/m between stations "
                     f"> {MAX_PATH_DKAPPA}")
    if m["gap_lo"] < SPACING_LO or m["gap_hi"] > SPACING_HI:
        fails.append(f"G13 published station gaps run {m['gap_lo']:.3f}-{m['gap_hi']:.3f} x "
                     f"nominal, outside {SPACING_LO}-{SPACING_HI} -- a path folded on itself")
    min_gap, min_ramp, _rows = engage_report(H)
    if min_gap < MIN_ENGAGE_GAP_M:
        fails.append(f"G7 the second box is first shaped around at {min_gap:.2f} m "
                     f"< {MIN_ENGAGE_GAP_M} (a plan that late arrives after the braking decision)")
    if min_ramp < MIN_ENGAGE_RAMP_M:
        fails.append(f"G8 its entry ramp gets {min_ramp:.2f} m < {MIN_ENGAGE_RAMP_M}")
    if fails:
        print("\nFAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1 if a.check else 0
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
