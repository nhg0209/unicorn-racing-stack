#!/usr/bin/env python3
"""
test_static_reopt.py — standalone verification for the width-modulation + re-optimization core.

Phase-1 test (no perception, no ROS graph): inject synthetic static obstacles, run the core,
and compare the clean vs obstacle-aware racelines both numerically (est. lap time) and visually
(saved PNG). Does NOT touch any gb_optimizer node.

Run inside the `unicorn` env (needs the vendored optimizer + tph):

    python3 test_static_reopt.py \
        --map ifac --config SIM \
        --obs "3.0,1.0,0.25;  -2.0,4.5,0.20"

Obstacles are `x,y,r` triples in the map frame, separated by ';'. With no --obs a couple of
obstacles are auto-placed on the clean raceline for a quick smoke test.
"""

import argparse
import os
import sys

import numpy as np

# resolve repo paths so the script runs from source without a colcon install
_THIS = os.path.dirname(os.path.abspath(__file__))
_PKG_PY = os.path.abspath(os.path.join(_THIS, "..", "gb_optimizer"))
_STACK_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _PKG_PY not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..")))  # for `gb_optimizer` package
if _PKG_PY not in sys.path:
    sys.path.insert(0, _PKG_PY)

from gb_optimizer.static_reopt_core import (  # noqa: E402
    Obstacle,
    ModulationParams,
    load_reftrack,
    reoptimize,
    reoptimize_with_obstacles,
)


def _parse_obstacles(spec: str):
    obs = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y, r = (float(v) for v in chunk.split(","))
        obs.append(Obstacle(x=x, y=y, r=r))
    return obs


def _auto_obstacles(clean_traj):
    """Place two obstacles right on the clean raceline (quick smoke test)."""
    n = len(clean_traj)
    picks = [n // 4, (2 * n) // 3]
    return [Obstacle(x=float(clean_traj[i, 1]), y=float(clean_traj[i, 2]), r=0.15) for i in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="ifac", help="map folder under stack_master/maps/")
    ap.add_argument("--config", default="SIM", help="config folder under stack_master/config/ (SIM|CAR)")
    ap.add_argument("--obs", default="", help="'x,y,r; x,y,r' obstacles in map frame; empty = auto")
    ap.add_argument("--safety-width", type=float, default=0.8)
    ap.add_argument("--obs-margin", type=float, default=0.05)
    ap.add_argument("--no-sp", action="store_true", help="skip the shortest-path (overtaking) line")
    ap.add_argument("--out", default="", help="output PNG path (default: scratchpad/static_reopt.png)")
    args = ap.parse_args()

    maps_dir = os.path.join(_STACK_ROOT, "stack_master", "maps", args.map)
    input_path = os.path.join(_STACK_ROOT, "stack_master", "config", args.config)
    reftrack_csv = os.path.join(maps_dir, "centerline.csv")

    print(f"[test] reftrack : {reftrack_csv}")
    print(f"[test] config   : {input_path}")
    reftrack = load_reftrack(reftrack_csv)
    print(f"[test] reftrack points: {reftrack.shape[0]}")

    # 1) clean baseline
    print("[test] optimizing CLEAN raceline (mincurv_iqp) ...")
    clean_traj, clean_br, clean_bl, clean_est = reoptimize(
        reftrack, input_path, "mincurv_iqp", args.safety_width, "map_centerline"
    )
    print(f"[test] CLEAN est lap time: {clean_est:.3f}s, max v: {np.max(clean_traj[:,5]):.2f} m/s")

    # 2) obstacles
    obstacles = _parse_obstacles(args.obs) if args.obs else _auto_obstacles(clean_traj)
    print(f"[test] obstacles ({len(obstacles)}):")
    for o in obstacles:
        print(f"         ({o.x:.2f}, {o.y:.2f}) r={o.r:.2f}")

    params = ModulationParams(obs_margin=args.obs_margin)
    try:
        res = reoptimize_with_obstacles(
            reftrack, obstacles, input_path,
            params=params,
            safety_width=args.safety_width,
            safety_width_sp=args.safety_width,
            compute_sp=not args.no_sp,
        )
    except Exception as e:
        # a genuinely blocking obstacle can make the QP infeasible; report the modulation
        # diagnostics so we can see WHY (production node will catch this and fall back).
        from gb_optimizer.static_reopt_core import modulate_widths
        _, rep = modulate_widths(reftrack, obstacles, params,
                                 min_total_width=args.safety_width)
        print(f"[test] OPTIMIZER FAILED: {type(e).__name__}: {str(e)[:80]}")
        print(f"[test] modulation: affected={rep.n_affected}/{rep.n_stations}, "
              f"infeasible={rep.n_infeasible}, min_halfwidth={rep.min_halfwidth_seen:.3f}m, "
              f"sides={rep.obstacle_sides}")
        print("[test] -> obstacle likely blocks the track on the chosen side; "
              "reactive layer must handle it.")
        return
    rep = res["report"]
    obs_traj, obs_br, obs_bl, obs_est = res["main"]
    print(f"[test] OBSTACLE est lap time: {obs_est:.3f}s, max v: {np.max(obs_traj[:,5]):.2f} m/s")
    print(f"[test] modulation: affected={rep.n_affected}/{rep.n_stations}, "
          f"infeasible={rep.n_infeasible}, min_halfwidth={rep.min_halfwidth_seen:.3f}m, "
          f"sides={rep.obstacle_sides}")
    if rep.n_infeasible:
        print(f"[test] WARNING: {rep.n_infeasible} infeasible stations at idx {rep.infeasible_s_idx[:10]}...")
    print(f"[test] lap-time delta (obstacle - clean): {obs_est - clean_est:+.3f}s")

    # 3) plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[test] matplotlib unavailable ({e}); skipping plot")
        return

    out = args.out or os.path.join(
        os.environ.get("CLAUDE_SCRATCHPAD", "/tmp"), "static_reopt.png"
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(clean_br[:, 0], clean_br[:, 1], color="0.6", lw=0.8)
    ax.plot(clean_bl[:, 0], clean_bl[:, 1], color="0.6", lw=0.8, label="track bounds")
    ax.plot(clean_traj[:, 1], clean_traj[:, 2], "b-", lw=1.6, label=f"clean ({clean_est:.2f}s)")
    ax.plot(obs_traj[:, 1], obs_traj[:, 2], "r-", lw=1.6, label=f"obstacle-aware ({obs_est:.2f}s)")
    ax.plot(res["reftrack_mod"][:, 0], res["reftrack_mod"][:, 1], "g.", ms=1.5, label="modulated ref")
    if not args.no_sp and "sp" in res:
        sp_traj = res["sp"][0]
        ax.plot(sp_traj[:, 1], sp_traj[:, 2], "m--", lw=1.0, label=f"obstacle SP ({res['sp'][3]:.2f}s)")
    for o in obstacles:
        ax.add_patch(plt.Circle((o.x, o.y), o.r, color="k", alpha=0.35))
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"static re-opt: {args.map} — {len(obstacles)} obstacle(s)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[test] saved plot -> {out}")


if __name__ == "__main__":
    main()
