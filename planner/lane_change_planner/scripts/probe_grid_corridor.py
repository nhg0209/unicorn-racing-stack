#!/usr/bin/env python3
"""Does measuring the corridor from the occupancy grid recover the lanes the SWAPPED
waypoint bounds were rejecting on the f map?

Runs the REAL _grid_corridor / _lane_bounds / _choose_lane against the REAL f map: its
global_waypoints.json AND its f.png occupancy grid, with a GridFilter stand-in that
reproduces GridFilter's own pixel convention and erosion exactly.

A DIAGNOSTIC, not a test: it prints a comparison table and asserts nothing. It lived in test/
under a test_ name, where pytest collected it and ran the whole body -- map loads, erosion and a
few hundred _choose_lane solves -- as a side effect of collection, with no verdict at the end.

  ~/miniforge3/envs/unicorn/bin/python3 planner/lane_change_planner/scripts/probe_grid_corridor.py
"""
import json
import types
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from frenet_conversion.frenet_converter import FrenetConverter

REPO = Path(__file__).resolve().parents[3]
MOD = REPO / "planner/lane_change_planner/lane_change_planner/change_avoidance_node.py"

cav = types.ModuleType("cav")
cav.__dict__["__file__"] = str(MOD)
exec(compile(MOD.read_text(), str(MOD), "exec"), cav.__dict__)
CAV = cav.ChangeAvoidanceNode


class Grid:
    """GridFilter stand-in: same 255=free convention, same erosion, same world_to_pixel."""

    def __init__(self, mapdir, name, kernel_size=3):
        meta = yaml.safe_load((mapdir / f"{name}.yaml").read_text())
        self.resolution = meta["resolution"]
        self.origin = (meta["origin"][0], meta["origin"][1])
        img = np.array(Image.open(mapdir / meta["image"]).convert("L"))
        # map_server: dark = occupied. Match GridFilter (100 -> 0 occupied, else 255 free).
        occ = img < int(255 * (1.0 - meta["occupied_thresh"]))
        base = np.where(occ, 0, 255).astype(np.uint8)
        # ROS OccupancyGrid row 0 is the BOTTOM of the map; a .pgm/.png row 0 is the TOP.
        base = np.flipud(base)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        self.eroded_image = cv2.erode(base, k)


class W:
    __slots__ = ("s_m", "d_left", "d_right", "vx_mps", "kappa_radpm")


class Msg:
    def __init__(self, w): self.wpnts = w


class Log:
    def __init__(self): self.warns = []
    def info(self, *a, **k): pass
    def warn(self, m, **k): self.warns.append(m)
    def error(self, *a, **k): pass


class Fake:
    def get_logger(self): return self._log


for n in ("_lane_bounds", "_solve_profile", "_solve_gentlest", "_choose_lane", "_pass_speed",
          "_meet_s_raw", "_update_meet_s", "_meet_s", "_pass_hold_m", "_lane_horizon",
          "_sdiff", "_lane_profile_at", "_free_mask", "_grid_corridor", "_grid_is_authority"):
    setattr(Fake, n, getattr(CAV, n))


def load(mapname):
    d = json.load(open(REPO / f"stack_master/maps/{mapname}/global_waypoints.json"))
    out = []
    for w in d["global_traj_wpnts_iqp"]["wpnts"]:
        o = W(); o.s_m = w["s_m"]; o.d_left = w["d_left"]; o.d_right = w["d_right"]
        o.vx_mps = w["vx_mps"]; o.kappa_radpm = w["kappa_radpm"]; out.append(o)
    return out


def mk(wpnts, conv, grid, s, gap, opp_vs, trust_grid):
    f = Fake()
    f._log = Log()
    f.scaled_wpnts_msg = Msg(wpnts); f.scaled_max_idx = len(wpnts)
    f.scaled_max_s = float(wpnts[-1].s_m + (wpnts[1].s_m - wpnts[0].s_m))
    f.scaled_delta_s = float(wpnts[1].s_m - wpnts[0].s_m)
    f.converter = conv; f.map_filter = grid
    f.current_s, f.current_d, f.current_vs = float(s), 0.0, opp_vs
    f.hold_horizon_m = 22.0; f.commit_horizon_m = 30.0; f.lane_commit = True
    f.lane_solve_ds = 0.25; f.lane_max_slope = 0.35; f.lane_max_slope_close = 0.55
    f.pass_overlap_m = 0.55; f.pass_hold_max_m = 12.0; f.pass_lead_max_s = 5.0
    f.pass_gap_m = 1.2; f.pass_behind_m = 5.0; f.meet_s_slew_m = 0.15
    f.width_car = 0.30; f.spline_bound_mindist = 0.12; f.lane_smooth_pad_m = 0.01
    f.sep_margin_m = 0.42; f.veh_length_m = 0.55
    f.use_prediction = False; f.opp_pred_slack_m = 0.20
    f.trust_grid_bounds = trust_grid
    f.grid_scan_max = 3.0; f.grid_scan_step = 0.05; f.bounds_warn_m = 0.5
    f._opp_d_used = {1: 0.0, -1: 0.0}; f._lane_dbg = ""
    f.side = None; f.lane_sign = 0; f.lane_s = f.lane_d = None; f.lane_offset_cur = 0.0
    f.meet_s = None; f.commit_meet_s = None; f.commit_target_d = None
    f.target = dict(id=1, s=(s + gap) % f.scaled_max_s, d=0.0, vs=opp_vs, vd=0.0,
                    size=0.30, last_seen=0.0)
    return f


mapdir = REPO / "stack_master/maps/f"
wpnts = load("f")
xy = np.array([[0.0, 0.0]])
gj = json.load(open(mapdir / "global_waypoints.json"))["global_traj_wpnts_iqp"]["wpnts"]
gxy = np.array([[w["x_m"], w["y_m"]] for w in gj])
conv = FrenetConverter(gxy[:, 0], gxy[:, 1])
grid = Grid(mapdir, "f")

# sanity: the raceline itself must read FREE in the eroded grid
probe = Fake(); probe._log = Log(); probe.map_filter = grid
free = probe._free_mask(gxy)
print(f"sanity: raceline points inside eroded free space: {100*free.mean():.1f}%  "
      f"({'OK' if free.mean() > 0.95 else 'GRID CONVENTION WRONG -- results below are meaningless'})")

kappa = np.array([abs(w.kappa_radpm) for w in wpnts])
straight = np.argsort(kappa)[:len(kappa) // 6]
corner = np.argsort(kappa)[-len(kappa) // 6:]

# corridor comparison at a few anchors
print("\ncorridor at sample s (waypoint bounds vs measured):")
for j in list(straight[:3]) + list(corner[:3]):
    s = wpnts[j].s_m
    f = mk(wpnts, conv, grid, s, 5.0, 1.5, True)
    g = f._grid_corridor(s)
    print(f"  s={s:6.2f}  wpnt[dl={wpnts[j].d_left:5.2f} dr={wpnts[j].d_right:5.2f}]"
          f"  ->  grid[d_lo={g[0]:+5.2f} d_hi={g[1]:+5.2f}] width={g[1]-g[0]:.2f}"
          if g else f"  s={s:6.2f}  grid corridor None")

print("\nlane-fit rate, waypoint bounds vs measured corridor:")
for label, idxs in (("STRAIGHT", straight), ("CORNER", corner)):
    for opp_vs in (1.5, 3.0):
        row = {}
        for trust in (False, True):
            ok = 0; n = 0
            for j in idxs[:: max(1, len(idxs) // 30)]:
                f = mk(wpnts, conv, grid, wpnts[j].s_m, 5.0, opp_vs, trust)
                n += 1
                if f._choose_lane():
                    ok += 1
            row[trust] = (ok, n)
        (o0, n0), (o1, n1) = row[False], row[True]
        print(f"  {label:8s} opp_vs={opp_vs:.1f}  waypoint-bounds {o0:2d}/{n0:2d}"
              f"   ->  grid-corridor {o1:2d}/{n1:2d}   ({o1-o0:+d})")

f = mk(wpnts, conv, grid, wpnts[straight[0]].s_m, 5.0, 1.5, True)
f._choose_lane()
print(f"\nswap self-report fired: {any('SWAPPED' in w for w in f._log.warns)}")
for w in f._log.warns[:1]:
    print("  " + w.replace("\n", " ")[:160])
