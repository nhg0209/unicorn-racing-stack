#!/usr/bin/env python3
"""Why does the published path get REPLACED mid-drive, and is it the solver or the situation?

Sim reports corridor_qp producing smooth paths and then swapping them for different ones while the
car is on them. A swap has four possible authors and they need different fixes, so the first job is
to tell them apart rather than to guess:

    the box set changed        a box entered or left the horizon -- a different maneuver, correctly
    the side flipped           the plan changed its mind about which way to pass
    the commit was released    one of the five paths in _reuse_committed threw the frozen path away
    NONE OF THOSE              same boxes, same side, still committed -- and a different answer

The last one is the interesting one, and it has a specific cause available: the objective is
min ||D2 d||^2 + w_dev ||d||^2 with w_dev defaulting to 0.0, so nothing in it prefers the previous
answer, and nothing in it is even strictly convex on its own -- an affine profile has zero second
difference, which is why _solve_qp needs a ridge at all. A problem re-posed every cycle on a window
that slid by one station can land somewhere else in a flat direction without anything about the
track having changed.

Two measurements, because the drive alone cannot separate "the situation moved" from "the solver
moved":

  DRIVE      the car is advanced along its own published reference and every cycle re-plans. Each
             consecutive pair is compared on WORLD-FIXED stations -- the same piece of track, not
             the same array index -- and the cycles that moved more than REPLACE_M are classified.

  PERTURB    the car is held still except for a deliberate nudge of the station grid, and nothing
             else changes at all: same boxes, same side, same corridor, same margins. Whatever d
             does here is the solver, because it is the only thing that could have. This is the
             control the drive lacks, and it is what makes "degenerate" a measurement rather than
             an inference.

Both are run for `sample` as well, because "the quintic wobbles this much" is the only scale on
which a corridor_qp number means anything.

Measurement only. No parameter is written, no default is changed.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_plan_consistency.py
"""
import argparse
import json
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/spliner/scripts"))
import sweep_static_feasibility as F          # noqa: E402
import sweep_avoidance_stability as S         # noqa: E402

REPLACE_M = 0.10          # a swap: max |d_new - d_old| over the overlap
SETTLE_M = 2.0            # ignore this much ahead of the car when isolating the solver from the pin
CAR_V = 2.5
RATE_HZ = 20.0
TRACK_TAU = 0.30
N_BOX, BOX_GAP_M = 4, 6.0
LAPS = 1.2
PERTURB = (0.01, 0.05, 0.0997)     # [m] nudge of cur_s; the last is one published station


class _Cap:
    """The node's logger, kept, so a commit release can be attributed to the branch that raised it."""

    def __init__(self):
        self.lines = []

    def _rec(self, msg, *a, **k):
        self.lines.append(str(msg))

    info = warn = warning = error = debug = _rec

    RELEASES = (("past its end", "past-end"), ("moved ds=", "box-moved"),
                ("no longer clears", "slice-unclear"), ("commit released:", "new-obstacle"))

    def releases(self, since):
        out = []
        for s in self.lines[since:]:
            if "commit released" not in s:
                continue
            out.append(next((tag for key, tag in self.RELEASES if key in s), "other"))
        return out


def _node(H, method, clock, stamp):
    """A driving node: the real commit machinery, the chosen d(s) generator, a capturing logger."""
    n = H._node(0.0, ladder=True)
    cap = _Cap()
    n.get_logger = lambda: cap
    n._store_commit = types.MethodType(H.san.ObstacleSpliner._store_commit, n)
    n.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=int(clock.t * 1e9), to_msg=lambda: stamp))
    n.commit_enable, n._committed = True, None
    n.commit_dev_max = 0.6
    n.commit_reanchor_len_m, n.commit_reanchor_max_m = 2.0, 1.0
    n.commit_obs_ds, n.commit_obs_dd = 0.75, 0.40
    n.commit_drop_on_new_obstacle = True
    n.ramp_search_max_ms = S.SHIPPED_LADDER_MS
    n.cur_vs = CAR_V
    n.static_plan_method = method
    n._publish_feasible = lambda ok: None
    return n, cap


def _pose(H, n, cur_s, cur_d):
    n.cur_s, n.cur_d = cur_s % H.L, cur_d
    resp = H.converter.get_cartesian(np.array([n.cur_s]), np.array([cur_d]))
    pxy = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)[0]
    n.cur_x, n.cur_y = float(pxy[0]), float(pxy[1])
    j = int(np.argmin(np.abs(H.s_arr - n.cur_s)))
    j2 = (j + 1) % (len(H.wp) - 1)
    n.cur_yaw = float(np.arctan2(H.wp[j2]["y_m"] - H.wp[j]["y_m"],
                                 H.wp[j2]["x_m"] - H.wp[j]["x_m"]))


def _unwrap(H, s, s_ref):
    """World-fixed coordinate: s as a signed distance from a FIXED anchor, not from the car.

    Comparing two cycles by array index compares different pieces of track (the window slid);
    comparing by distance-from-the-car does the same thing more subtly. The anchor is a station on
    the map and does not move, so equal u is equal ground.
    """
    return ((np.asarray(s, float) - s_ref + H.L / 2.0) % H.L) - H.L / 2.0


def _overlap_diff(u_a, d_a, u_b, d_b, u_min=None):
    """max |d_b - d_a| over the world-fixed stations both publications cover. None if none."""
    if u_a is None or u_b is None or len(u_a) < 2 or len(u_b) < 2:
        return None, 0
    lo = max(u_a[0], u_b[0])
    hi = min(u_a[-1], u_b[-1])
    if u_min is not None:
        lo = max(lo, u_min)
    if hi <= lo:
        return None, 0
    m = (u_b >= lo) & (u_b <= hi)
    if not m.any():
        return None, 0
    return float(np.max(np.abs(np.interp(u_b[m], u_a, d_a) - d_b[m]))), int(m.sum())


def drive(H, method, laps=LAPS):
    """One car, one lap and a bit, re-planning every cycle. Returns the per-cycle record."""
    stamp = H.san.OTWpntArray().header.stamp
    clock = S._Clock()
    n, cap = _node(H, method, clock, stamp)
    L, wp = H.L, H.wp
    el = float(np.mean(np.hypot(np.diff([w["x_m"] for w in wp]), np.diff([w["y_m"] for w in wp]))))
    step_i = max(1, int(round(BOX_GAP_M / el)))
    boxes = []
    for k in range(N_BOX):
        j = (40 + k * step_i) % (len(wp) - 1)
        w = wp[j]
        boxes.append(types.SimpleNamespace(
            id=k + 1, s_start=(w["s_m"] - 0.15) % L, s_end=(w["s_m"] + 0.15) % L,
            s_center=w["s_m"], d_center=0.0, d_right=-0.15, d_left=0.15, size=0.30,
            vs=0.0, vd=0.0, is_static=True, is_visible=True,
            x_m=w["x_m"], y_m=w["y_m"], is_actually_a_gap=False))
    n.obstacles = boxes
    n._promoted = {b.id for b in boxes}
    s_ref = boxes[0].s_center

    dt = 1.0 / RATE_HZ
    cur_s, cur_d = 0.0, 0.0
    recs = []
    for _ in range(int(laps * L / (CAR_V * dt))):
        clock.t += dt
        _pose(H, n, cur_s, cur_d)
        seen = len(cap.lines)
        try:
            res = n.do_spline(H.gbw)
        except Exception as exc:                    # noqa: BLE001
            recs.append({"err": repr(exc), "u": None})
            cur_s += CAR_V * dt
            continue
        pts = res[0].wpnts if res and res[0] is not None else []
        u = _unwrap(H, [p.s_m for p in pts], s_ref) if pts else None
        d = np.array([p.d_m for p in pts]) if pts else None
        if u is not None:
            o = np.argsort(u)
            u, d = u[o], d[o]
        recs.append({
            "cur_s": cur_s % L, "u_car": float(_unwrap(H, [cur_s % L], s_ref)[0]),
            "u": u, "d": d, "n": len(pts),
            "side": 0 if not pts else int(np.sign(max((p.d_m for p in pts), key=abs))),
            "committed": n._committed is not None,
            "knots": tuple(sorted(oid for (oid, _s, _d) in (n._committed or {}).get("obs", []))),
            "released": cap.releases(seen),
        })
        target = 0.0
        if pts:
            g = [((p.s_m - n.cur_s) % L) for p in pts]
            k = int(np.argmin([abs(v - 1.0) for v in g]))
            if abs(g[k] - 1.0) < 0.6:
                target = float(pts[k].d_m)
        cur_d += (target - cur_d) * min(1.0, dt / TRACK_TAU)
        cur_s += CAR_V * dt
    return recs


def drive_report(recs, label):
    """Classify every consecutive pair, and attribute the swaps."""
    swaps, pairs = 0, 0
    diffs = []
    why = Counter()
    rel = Counter()
    settled = []
    for a, b in zip(recs, recs[1:]):
        if a.get("u") is None or b.get("u") is None:
            continue
        dmax, npts = _overlap_diff(a["u"], a["d"], b["u"], b["d"])
        if dmax is None:
            continue
        pairs += 1
        diffs.append(dmax)
        # the same comparison, but only well ahead of the car, where the start pin cannot be the
        # explanation: a legitimately moving pin is a lead-in effect and dies out within a metre
        far, _ = _overlap_diff(a["u"], a["d"], b["u"], b["d"], u_min=b["u_car"] + SETTLE_M)
        if far is not None:
            settled.append(far)
        if dmax <= REPLACE_M:
            continue
        swaps += 1
        rel.update(b["released"])
        if b["knots"] != a["knots"]:
            why["box set changed"] += 1
        elif b["released"]:
            why["commit released"] += 1
        elif b["side"] != a["side"] and a["side"] and b["side"]:
            why["side flipped"] += 1
        elif not a["committed"] or not b["committed"]:
            why["not committed"] += 1
        else:
            why["SAME boxes, SAME side, still committed"] += 1
    d = np.array(diffs) if diffs else np.array([0.0])
    st = np.array(settled) if settled else np.array([0.0])
    print(f"--- DRIVE | {label} | {pairs} consecutive pairs ---")
    print(f"    max |d_new - d_old| over the overlap: p50 {np.percentile(d,50):.4f}  "
          f"p90 {np.percentile(d,90):.4f}  p99 {np.percentile(d,99):.4f}  max {d.max():.4f} m")
    print(f"      same, ignoring the first {SETTLE_M:.0f} m ahead of the car: "
          f"p90 {np.percentile(st,90):.4f}  max {st.max():.4f} m")
    print(f"    swaps (> {REPLACE_M:.2f} m): {swaps} / {pairs}  ({100.0*swaps/max(pairs,1):.1f} %)")
    for k, v in why.most_common():
        print(f"      {k:42s} {v:5d}  ({100.0*v/max(swaps,1):5.1f} %)")
    if rel:
        print("      release branches: " + ", ".join(f"{k}={v}" for k, v in rel.most_common()))
    return {"pairs": pairs, "swaps": swaps, "p90": float(np.percentile(d, 90)),
            "max": float(d.max()), "settled_max": float(st.max()), "why": dict(why),
            "releases": dict(rel)}


def _box(H, s_c, d, oid=1, r=0.15):
    xy = H.converter.get_cartesian(np.array([s_c % H.L]), np.array([d]))
    xy = (xy.T if xy.ndim == 2 else xy).reshape(-1, 2)[0]
    return types.SimpleNamespace(
        id=oid, s_start=(s_c - r) % H.L, s_end=(s_c + r) % H.L, s_center=s_c % H.L,
        d_center=float(d), d_right=float(d) - r, d_left=float(d) + r, size=2 * r,
        vs=0.0, vd=0.0, is_static=True, is_visible=True,
        x_m=float(xy[0]), y_m=float(xy[1]), is_actually_a_gap=False)


# What actually jitters cycle to cycle on a car. The grid nudge is the window sliding under a
# moving car; the obstacle nudges are the tracker's own noise on a box that is not moving. All
# three leave the TRACK, the SIDE and the MARGINS identical, so any d that moves is the solve.
PERTURBATIONS = (("car s   +1 cm", "car", 0.01), ("car s   +5 cm", "car", 0.05),
                 ("car s  +10 cm", "car", 0.0997), ("obs s   +1 cm", "obs_s", 0.01),
                 ("obs s   +3 cm", "obs_s", 0.03), ("obs d   +1 cm", "obs_d", 0.01),
                 ("obs d   +3 cm", "obs_d", 0.03))


def perturb(H, methods, n_pose=60, gap_m=9.0, obs_d=0.0):
    """Nudge one input by a centimetre; measure how far the published path moves.

    Run on poses where EVERY method under test produces a path, so the comparison is like for like
    -- a method that refuses half the poses and is steady on the rest is not steadier.

    Commits are OFF: a committed cycle republishes a frozen slice and would report a stability that
    belongs to the commit machinery, not to the solve. The question here is what the solver does
    when the problem barely changes.

    The first SETTLE_M ahead of the car is excluded. The start pin legitimately moves when the car
    moves, and a lead-in that decays within a metre is not the symptom; what the controller follows
    into the maneuver is.
    """
    stamp = H.san.OTWpntArray().header.stamp
    clock = S._Clock()
    nodes = {}
    for meth in methods:
        n, _cap = _node(H, meth, clock, stamp)
        n.commit_enable = False
        nodes[meth] = n
    L, wp = H.L, H.wp
    out = {m: {lab: [] for lab, _k, _e in PERTURBATIONS} for m in methods}
    n_ok = 0

    def plan(n, car_s, boxes, s_ref):
        n.obstacles = boxes
        n._promoted = {b.id for b in boxes}
        clock.t += 0.05
        _pose(H, n, car_s, 0.0)
        try:
            r = n.do_spline(H.gbw)
        except Exception:                            # noqa: BLE001
            return None, None
        pts = r[0].wpnts if r and r[0] is not None else []
        if not pts:
            return None, None
        u = _unwrap(H, [p.s_m for p in pts], s_ref)
        d = np.array([p.d_m for p in pts])
        o = np.argsort(u)
        return u[o], d[o]

    for t in range(n_pose):
        i0 = (11 + t * 5) % (len(wp) - 1)
        car_s = wp[i0]["s_m"]
        s_box = car_s + gap_m
        s_ref = s_box
        base = [_box(H, s_box, obs_d)]
        first = {m: plan(nodes[m], car_s, base, s_ref) for m in methods}
        if any(v[0] is None for v in first.values()):
            continue
        n_ok += 1
        for lab, kind, eps in PERTURBATIONS:
            cs = car_s + (eps if kind == "car" else 0.0)
            bx = [_box(H, s_box + (eps if kind == "obs_s" else 0.0),
                       obs_d + (eps if kind == "obs_d" else 0.0))]
            u_min = _unwrap(H, [cs % L], s_ref)[0] + SETTLE_M
            for m in methods:
                u1, d1 = plan(nodes[m], cs, bx, s_ref)
                if u1 is None:
                    out[m][lab].append(float("nan"))     # refused after the nudge: also instability
                    continue
                far, npts = _overlap_diff(first[m][0], first[m][1], u1, d1, u_min=u_min)
                if far is not None and npts >= 5:
                    out[m][lab].append(far)
    return out, n_ok


def perturb_report(res, n_ok, methods, label):
    print(f"--- PERTURB | {label} | {n_ok} poses where every method planned | commits OFF ---")
    print(f"      |d| change beyond {SETTLE_M:.0f} m ahead, same track / same side / same margins")
    print("      nudge            " + "".join(f"{m:>26s}" for m in methods))
    print("                       " + "".join(f"{'p50      p90      max':>26s}" for m in methods))
    rows = {}
    for lab, _k, _e in PERTURBATIONS:
        cells = []
        for m in methods:
            a = np.array(res[m][lab], dtype=float)
            nan = int(np.isnan(a).sum())
            a = a[~np.isnan(a)]
            if a.size == 0:
                cells.append(f"{'-':>26s}")
                continue
            rows[f"{m}|{lab}"] = {"p50": float(np.percentile(a, 50)),
                                  "p90": float(np.percentile(a, 90)),
                                  "max": float(a.max()), "n": int(a.size), "refused": nan}
            cells.append(f"{np.percentile(a,50):8.4f} {np.percentile(a,90):8.4f} {a.max():8.4f}")
        print(f"      {lab:16s} " + "".join(cells))
    for m in methods:
        nan = sum(int(np.isnan(np.array(res[m][lab], dtype=float)).sum())
                  for lab, _k, _e in PERTURBATIONS)
        if nan:
            print(f"      ({m}: {nan} nudges turned a published path into a refusal)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="*", default=["ifac"])
    ap.add_argument("--method", nargs="*", default=["sample", "corridor_qp"])
    ap.add_argument("--poses", type=int, default=60)
    ap.add_argument("--obs-d", type=float, default=0.0, dest="obs_d")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for m in a.map:
        H = F.Harness(m)
        print(f"\n########## {m} ##########")
        for meth in a.method:
            res[f"{m}|{meth}|drive"] = drive_report(drive(H, meth), f"{m} | {meth}")
        print()
        r, n_ok = perturb(H, a.method, a.poses, obs_d=a.obs_d)
        res[f"{m}|perturb"] = perturb_report(r, n_ok, a.method, m)
        del H
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"written: {a.out}")
    print("Measurement only: no parameter written, static_plan_method default untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
