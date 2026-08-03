#!/usr/bin/env python3
"""
sweep_static_reopt.py — offline quality sweep + regression gate for the ONLINE obstacle-aware
raceline (`reopt_method: local_window`).

Why this exists. The re-opt line's failure mode is not a crash, it is a SHAPE: the corridor fit can
shrink an avoidance hump until the line is too curved to steer and the velocity profile collapses,
and nothing downstream complains. That was invisible for a long time because the node only logged
"N/M apex(es) reshaped" — identical text whether the hump was laid at the 5 m reach the search asked
for or bisected down to 1.2 m. This script walks obstacles all the way around a map and reports the
numbers that actually matter, with no sim, no ROS and no colcon build:

  * geometric max |kappa| of the published line   -> must stay inside the vehicle's `curvlim`
  * implied lateral accel max(vx^2 * |kappa|)      -> must stay inside the ggv `ay_max`
  * estimated lap-time loss vs the clean raceline
  * how many apexes were laid vs honestly dropped, and why

Curvature and lateral accel are measured GEOMETRICALLY (Menger circumscribed circles on the
published points), never from the published `kappa_radpm` field. The field is deliberately smoothed
for the controller's L1 lookahead, and the additive `kappa_clean + alpha''` model the solver uses
internally is the small-(d, d', kappa) linearisation of the Frenet offset curvature — on a tight
track it reads far too low (measured on the ifac chicane: 1.19 modelled vs 2.11 real). Judging this
line by either of those is how an unsteerable path passes review.

Usage:
    python3 sweep_static_reopt.py                          # sweep the default maps, print a table
    python3 sweep_static_reopt.py --maps ifac --step 5     # denser sweep of one map
    python3 sweep_static_reopt.py --wall-margin 0.05 0.12  # compare two wall reserves
    python3 sweep_static_reopt.py --check                  # regression gate; exit 1 on violation

Run it after touching `_fit_hump_to_corridor`, `build_offset_profile`, `_resample_uniform`,
`_cap_speed_to_published_curvature`, or any of the reopt_* launch args.
"""

import argparse
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_STACK_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..")))       # for `gb_optimizer`

from gb_optimizer import static_reopt_core as core                   # noqa: E402

# The reactive apex this sweep simulates: an obstacle sitting ON the raceline forces the reactive
# planner out by obstacle radius + keep-out (width_car/2 + safety_margin) + apex_bulge. With the
# shipped static_avoidance_params.yaml that is 0.15 + 0.30 + 0.10 = 0.55 m.
APEX_OFFSET_M = 0.55
# ...and the obstacle it was driven around: a disk of this radius sitting ON the raceline.
# Passing it to the core is what makes this sweep exercise the REAL acceptance rule (the laid line
# must clear the box edge by obs_margin) instead of the amplitude-ratio proxy that applies only
# when no box is known. OBS_MARGIN_M mirrors reopt_obs_margin in base_system/race.launch.xml.
OBS_RADIUS_M = 0.15
OBS_MARGIN_M = 0.35
# The 0.5 mm regression. Obstacle just past the ifac start/finish straight, apex at the point the
# published line actually passed through in the failing run (2026-07-30). The amplitude cap parks
# the hump peak exactly on the corridor bound; with a zero fit tolerance the station 0.1 m away
# violates it by 0.0005 m, every reach >= 1.5 m is rejected and the fit bisects to 1.24 m.
REGRESSION = {"map": "ifac", "apex": (3.95, 0.24), "wall_margin": 0.12, "min_reach_m": 4.5}


def load_map(name):
    d = os.path.join(_STACK_ROOT, "stack_master", "maps", name)
    wp = json.load(open(os.path.join(d, "global_waypoints.json")))["global_traj_wpnts_iqp"]["wpnts"]
    return {
        "xy": np.array([[w["x_m"], w["y_m"]] for w in wp], float),
        "dr": np.array([w["d_right"] for w in wp], float),
        "dl": np.array([w["d_left"] for w in wp], float),
        "kappa": np.array([w["kappa_radpm"] for w in wp], float),
        "vx": np.array([w["vx_mps"] for w in wp], float),
        "reftrack": np.genfromtxt(os.path.join(d, "centerline.csv"), delimiter=",", skip_header=1),
    }


def clean_metrics(m):
    seg = np.roll(m["xy"], -1, axis=0) - m["xy"]
    el = np.hypot(seg[:, 0], seg[:, 1])
    kg = np.abs(core._menger_kappa(m["xy"]))
    return (float(np.sum(el / np.maximum(m["vx"], 1e-3))), float(kg.max()),
            float(np.max(m["vx"] ** 2 * kg)))


def solve(m, apex, cfg_dir, wall_margin, fit_tol, obstacle=None):
    return core.reoptimize_local_window(
        m["xy"], m["dr"], m["dl"], m["reftrack"], [apex], cfg_dir,
        params=core.ModulationParams(obs_margin=OBS_MARGIN_M),
        w_veh=0.30, clean_vx=m["vx"], wall_margin=wall_margin,
        reach_time=0.0, reach_min=1.0, reach_max=6.0, clean_kappa=m["kappa"], fit_tol=fit_tol,
        apex_obstacles=[obstacle] if obstacle is not None else None)


def sweep(m, cfg_dir, wall_margin, fit_tol, step):
    """Walk an obstacle around the lap; return per-solve rows measured on the published geometry."""
    n = len(m["xy"])
    rl = np.column_stack([m["xy"][:, 0], m["xy"][:, 1], m["dr"], m["dl"]])
    _, nvec, _ = core.centerline_frame(rl)
    nvec[-1] = nvec[0]
    hi = np.maximum(core._cyclic_smooth(m["dr"] - 0.15 - wall_margin, 7), 0.0)
    lo = np.minimum(core._cyclic_smooth(-(m["dl"] - 0.15 - wall_margin), 7), 0.0)
    rows, dropped = [], []
    for i in range(0, n - 1, step):
        side = 1.0 if hi[i] >= -lo[i] else -1.0          # the reactive planner takes the roomier side
        apex = tuple(m["xy"][i] + side * APEX_OFFSET_M * nvec[i])
        obstacle = (float(m["xy"][i][0]), float(m["xy"][i][1]), OBS_RADIUS_M)
        try:
            res = solve(m, apex, cfg_dir, wall_margin, fit_tol, obstacle)
        except Exception as exc:                          # a solver failure is a result, not a stop
            dropped.append({"i": i, "reason": f"exception:{type(exc).__name__}"})
            continue
        if res["n_windows"] == 0:
            why = res["apex_dropped"][0].get("reason", "?") if res["apex_dropped"] else "no-apex"
            dropped.append({"i": i, "reason": why})
            continue
        traj = res["main"][0]
        kg = np.abs(core._menger_kappa(traj[:, 1:3]))
        laid = res["apex_laid"][0]
        rows.append({
            "i": i, "kappa": float(kg.max()), "ay": float(np.max(traj[:, 5] ** 2 * kg)),
            "lap": float(res["main"][3]), "laid": laid["laid"], "want": laid["want"],
            "r_in": laid["r_in"], "r_out": laid["r_out"], "r_req": laid["r_req"],
            "clear": float(laid.get("clear", float("nan"))),
            "curvlim": float(res.get("curvlim", 0.0)),
        })
    return rows, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=["ifac", "f"])
    ap.add_argument("--config", default="SIM", help="stack_master/config/<SIM|CAR>")
    ap.add_argument("--wall-margin", nargs="+", type=float, default=[0.05])
    ap.add_argument("--fit-tol", nargs="+", type=float, default=[core._FIT_TOL_DEFAULT])
    ap.add_argument("--step", type=int, default=11, help="obstacle spacing in waypoints (0.1 m each)")
    ap.add_argument("--check", action="store_true", help="enforce the pass criteria; exit 1 on failure")
    args = ap.parse_args()

    cfg_dir = os.path.join(_STACK_ROOT, "stack_master", "config", args.config)
    ay_max = 4.5
    try:
        ggv = core._load_veh_dyn(cfg_dir)[0]
        ay_max = float(np.min(ggv[:, 2]))
    except Exception:
        pass
    ok = True

    for name in args.maps:
        m = load_map(name)
        lap0, kappa0, ay0 = clean_metrics(m)
        print(f"\n##### map {name}: clean lap {lap0:.3f} s | clean geometric |kappa|max {kappa0:.2f} "
              f"| clean implied ay max {ay0:.2f} | ggv ay_max {ay_max:.2f}")
        print("  wall_m fit_tol | laid drop | curvlim | kappa_true: med   p90   MAX | over | ay MAX "
              "| lap loss: med     p90     MAX | reach med")
        for wm in args.wall_margin:
            for tol in args.fit_tol:
                rows, drops = sweep(m, cfg_dir, wm, tol, args.step)
                if not rows:
                    print(f"  {wm:6.3f} {tol:7.4f} |    0 {len(drops):4d} | every apex dropped")
                    ok = False
                    continue
                kap = np.array([r["kappa"] for r in rows])
                ay = np.array([r["ay"] for r in rows])
                loss = np.array([r["lap"] for r in rows]) - lap0
                reach = np.array([min(r["r_in"], r["r_out"]) for r in rows])
                curvlim = rows[0]["curvlim"]
                # The bar: curvlim, but never stricter than the clean line's own curvature — if the
                # raceline itself exceeds curvlim at a corner, dropping the apex would not fix that.
                bar = max(curvlim, kappa0)
                over = int(np.sum(kap > bar + 1e-3))
                print(f"  {wm:6.3f} {tol:7.4f} | {len(rows):4d} {len(drops):4d} | {curvlim:7.2f} "
                      f"| {np.median(kap):9.2f} {np.percentile(kap, 90):5.2f} {kap.max():5.2f} "
                      f"| {over:4d} | {ay.max():6.2f} | {np.median(loss):+8.3f} "
                      f"{np.percentile(loss, 90):+7.3f} {np.percentile(loss, 100):+7.3f} "
                      f"| {np.median(reach):6.2f}")
                if drops:
                    tally = {}
                    for d in drops:
                        tally[d["reason"]] = tally.get(d["reason"], 0) + 1
                    print(f"         dropped: {', '.join(f'{k}x{v}' for k, v in sorted(tally.items()))}"
                          f"  (left to the reactive layer BY DESIGN)")
                if args.check:
                    if over:
                        ok = False
                        worst = max(rows, key=lambda r: r["kappa"])
                        print(f"    FAIL: {over} line(s) exceed the curvature bar {bar:.2f} "
                              f"(worst {worst['kappa']:.2f} at waypoint {worst['i']}). "
                              f"The car cannot steer that; it must be shaved or dropped.")
                    if ay.max() > ay_max + 1e-2:
                        ok = False
                        print(f"    FAIL: implied lateral accel {ay.max():.2f} > ggv ay_max "
                              f"{ay_max:.2f}. The speed plan is outside the friction budget — check "
                              f"_cap_speed_to_published_curvature reads the real geometry.")
                    # A LAID hump must clear the box by obs_margin. The core enforces this per
                    # hump before accepting it; this re-checks it on the WOVEN + resampled profile
                    # that is actually returned, which is the only place the two can disagree.
                    short = [r for r in rows if r["clear"] == r["clear"]
                             and r["clear"] < OBS_MARGIN_M - 1e-6]
                    if short:
                        ok = False
                        w = min(short, key=lambda r: r["clear"])
                        print(f"    FAIL: {len(short)} laid line(s) clear the box by less than "
                              f"obs_margin {OBS_MARGIN_M:.2f} (worst {w['clear']:+.3f} m at "
                              f"waypoint {w['i']}). The acceptance floor accepted a hump the final "
                              f"profile does not deliver — the reactive layer will re-avoid it and "
                              f"the SM will read the line as blocked.")

    # --- the 0.5 mm corridor-fit regression -------------------------------------------------
    if args.check:
        r = REGRESSION
        print(f"\n--- regression: {r['map']} apex {r['apex']} at wall_margin {r['wall_margin']} ---")
        mr = load_map(r["map"])
        # The obstacle this apex was driven around sat on the raceline; recover it as the raceline
        # point nearest the apex (the same projection build_offset_profile does).
        j = int(np.argmin(np.hypot(mr["xy"][:, 0] - r["apex"][0], mr["xy"][:, 1] - r["apex"][1])))
        obs = (float(mr["xy"][j][0]), float(mr["xy"][j][1]), OBS_RADIUS_M)
        res = solve(mr, r["apex"], cfg_dir, r["wall_margin"], core._FIT_TOL_DEFAULT, obs)
        if res["n_windows"] == 0:
            ok = False
            why = res["apex_dropped"][0].get("reason", "?") if res["apex_dropped"] else "no-apex"
            print(f"FAIL: apex dropped ({why}); it must be laid with a wide reach.")
        else:
            a = res["apex_laid"][0]
            got = min(a["r_in"], a["r_out"])
            if got < r["min_reach_m"]:
                ok = False
                print(f"FAIL: reach {got:.2f} m < {r['min_reach_m']:.2f} m — the corridor fit is "
                      f"collapsing the hump again over a sub-millimetre bound violation. This is "
                      f"the +1.62 s/lap, curvlim-exceeding shape from 2026-07-30.")
            else:
                print(f"OK: reach {a['r_in']:.2f}/{a['r_out']:.2f} m (>= {r['min_reach_m']:.2f}), "
                      f"laid d={a['laid']:+.3f}, max|kappa| {a['kappa_peak']:.2f}, "
                      f"clears {a['clear']:+.3f} m (floor {OBS_MARGIN_M:.2f})")
        print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
