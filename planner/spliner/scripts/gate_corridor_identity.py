#!/usr/bin/env python3
"""corridor_qp's published output, byte for byte, across and after the sample deletion.

Deleting the sampled d(s) generator touches do_spline, which corridor_qp also runs through: the
obstacle gather, the side selection, the knot loop, obs_ok, the grid and body gates, the curvature
check, the handover blend and the publishing tail are shared. A deletion that changes any of those
by a bit has deleted something live, and the only way to know is to compare the numbers rather
than to read the diff.

So: 20 cases, the full published waypoint array for each (s, d, x, y, v, psi, kappa), hashed. Not
a tolerance -- a hash. A tolerance is a decision about how much change is acceptable, and the
answer here is none.

    --save  <json>   record the baseline (run BEFORE the deletion)
    --check <json>   re-run and compare (run AFTER each deletion commit)

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/gate_corridor_identity.py \
      --save /tmp/corridor_baseline.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/spliner/scripts"))
import sweep_static_feasibility as F          # noqa: E402

# 20 cases: both maps, corners and straights, one/two/three boxes, boxes on and off the raceline,
# the car on the line and displaced, near and far. Chosen to exercise every branch the deletion
# could plausibly reach -- side selection both ways, the squeeze (kept), a knot merge, a box the
# line already clears, and the wrap at the seam.
CASES = [
    ("ifac", 0, 12.0, 0.0, ((0.0, 0.0),)),
    ("ifac", 0, 4.0, 0.0, ((0.0, 0.0),)),
    ("ifac", 30, 12.0, 0.0, ((0.0, 0.0),)),
    ("ifac", 30, 12.0, 0.2, ((0.0, 0.0),)),
    ("ifac", 60, 12.0, -0.2, ((0.0, 0.3),)),
    ("ifac", 60, 8.0, 0.0, ((0.0, -0.3),)),
    ("ifac", 90, 12.0, 0.0, ((0.0, 0.0), (6.0, 0.0))),
    ("ifac", 90, 4.0, 0.0, ((0.0, 0.0), (2.0, 0.0))),
    ("ifac", 120, 12.0, 0.0, ((0.0, 0.0), (6.0, 0.0), (12.0, 0.0))),
    ("ifac", 150, 12.0, 0.1, ((0.0, 0.1),)),
    ("ifac", 180, 12.0, 0.0, ((0.0, 0.6),)),          # a box the followed line already clears
    ("ifac", 210, 12.0, 0.0, ((0.0, 0.0), (4.0, 0.3))),
    ("ifac", 240, 8.0, -0.2, ((0.0, 0.0),)),
    ("ifac", 300, 12.0, 0.0, ((0.0, 0.0), (2.0, 0.0), (4.0, 0.0))),
    ("ifac", 360, 12.0, 0.0, ((0.0, 0.0),)),          # near the seam: s wraps inside the window
    ("ifac_0807", 0, 12.0, 0.0, ((0.0, 0.0),)),
    ("ifac_0807", 60, 12.0, 0.2, ((0.0, 0.0), (6.0, 0.0))),
    ("ifac_0807", 120, 4.0, 0.0, ((0.0, -0.3),)),
    ("ifac_0807", 200, 12.0, 0.0, ((0.0, 0.0), (6.0, 0.0), (12.0, 0.0))),
    ("ifac_0807", 380, 12.0, 0.0, ((0.0, 0.0),)),     # near the seam on the other map
]
# The squeeze is KEPT by this deletion, so it has to be exercised by the baseline. It is gated on
# `cur_vs < squeeze_max_speed_mps` and the harness's default cur_vs is 3.0, which is exactly the
# threshold -- so at the default speed the reduced-margin retry cannot run and a baseline taken
# there would prove nothing about it. These cases drive at 2.0 m/s, below the gate, in the
# narrowest sections both maps have.
SLOW_MPS = 2.0
# Located by scanning both maps for `ot_line == "squeeze"` rather than guessed at: 45 such cases
# exist on ifac and 39 on ifac_0807 at 2 m/s, and none of them is where a reading of the corridor
# widths would have put them.
SLOW_CASES = [
    ("ifac", 3, 12.0, 0.0, ((0.0, 0.0), (2.0, 0.0))),
    ("ifac", 6, 12.0, 0.0, ((0.0, 0.0),)),
    ("ifac", 6, 8.0, 0.0, ((0.0, 0.0), (2.0, 0.0))),
    ("ifac", 6, 4.0, 0.0, ((0.0, 0.0),)),
    ("ifac_0807", 69, 12.0, 0.0, ((0.0, 0.0),)),
    ("ifac_0807", 69, 4.0, 0.0, ((0.0, 0.0), (2.0, 0.0))),
    ("ifac_0807", 72, 12.0, 0.0, ((0.0, 0.0),)),
]


def _fields(w):
    return (w.s_m, w.d_m, w.x_m, w.y_m, w.vx_mps, w.psi_rad, w.kappa_radpm)


def run_case(H, station, gap, cur_d, boxes):
    """One plan under corridor_qp, hashed. The method is forced, not read from the yaml, so the
    baseline does not depend on which default happens to be in the working tree."""
    n = H.make_node(station % (len(H.wp) - 1), gap, cur_d, ladder=True, boxes=boxes)
    if hasattr(n, "static_plan_method"):
        n.static_plan_method = "corridor_qp"
    try:
        res = n.do_spline(H.gbw)
    except Exception as e:                            # noqa: BLE001
        return {"err": f"{type(e).__name__}: {e}"}
    pts = res[0].wpnts if res and res[0] is not None else []
    if not pts:
        return {"n": 0, "hash": "empty", "squeeze": False}
    arr = np.array([_fields(w) for w in pts], dtype=np.float64)
    return {"n": len(pts), "hash": hashlib.sha256(arr.tobytes()).hexdigest(),
            "squeeze": bool(res[0].ot_line == "squeeze"),
            "d_peak": float(np.max(np.abs(arr[:, 1]))),
            "k_peak": float(np.max(np.abs(arr[:, 6])))}


def collect():
    out = {}
    H = {}
    for speed, cases in ((None, CASES), (SLOW_MPS, SLOW_CASES)):
        for (m, st, gap, cd, boxes) in cases:
            if m not in H:
                H[m] = F.Harness(m)
            H[m].cur_vs = 3.0 if speed is None else speed
            key = f"{m}|{st}|{gap}|{cd}|{boxes}|v{H[m].cur_vs}"
            out[key] = run_case(H[m], st, gap, cd, boxes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--check")
    a = ap.parse_args()
    got = collect()
    n_pub = sum(1 for v in got.values() if v.get("n"))
    n_sq = sum(1 for v in got.values() if v.get("squeeze"))
    print(f"{len(got)} cases | {n_pub} published | {n_sq} via the squeeze | "
          f"{sum(1 for v in got.values() if 'err' in v)} raised")
    if a.save:
        Path(a.save).write_text(json.dumps(got, indent=1, sort_keys=True))
        print(f"baseline written: {a.save}")
        return 0
    if a.check:
        want = json.loads(Path(a.check).read_text())
        bad = []
        for k in sorted(set(want) | set(got)):
            if want.get(k) != got.get(k):
                bad.append((k, want.get(k), got.get(k)))
        if bad:
            print(f"\nNOT IDENTICAL: {len(bad)} of {len(want)} cases changed")
            for k, w, g in bad[:8]:
                print(f"  {k}\n    was {w}\n    now {g}")
            return 1
        print("IDENTICAL: every case matches the baseline bit for bit")
        return 0
    for k, v in sorted(got.items()):
        print(f"  {k}\n    {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
