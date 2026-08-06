#!/usr/bin/env python3
"""How EARLY, and how OFTEN, the static planner can plan around a box on the real maps.

Drives the real do_spline against the real map -- global_waypoints.json for the raceline and the
corridor, <map>.png eroded twice for the sampling grid and the body floor -- with one box on the
raceline at each station, and asks:

  FEASIBILITY  at what fraction of the stations does a path come out, per {forward gap} x
               {lateral tracking error}
  EARLINESS    how many of the CORNER stations (|kappa| > 0.8) can be planned at ANY distance,
               and how far out. This is what the driver feels: until a plan exists the car is
               committed to TRAILING, and a plan that only appears two metres from the box arrives
               after the braking decision. The COUNT is what is gated -- see MIN_CORNERS_PLANNABLE
               for why the mean distance cannot fail.
  SPEED        the maneuver's curvature-limited speed cap, sqrt(a_lat_max / peak|kappa|). The ramp
               ladder buys feasibility with curvature, so this is the price side of the trade and
               it is gated, not just reported.
  SUPERSET     the ladder's rung 0 is the geometry the main pass already tried, so turning it on
               can only ADD feasible cells. A cell that goes feasible -> infeasible is a bug.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_feasibility.py --check

(the conda env: the system python3 has no trajectory_planning_helpers)
"""
import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MOD = REPO / "planner/spliner/spliner/static_avoidance_node.py"

# --- gates -------------------------------------------------------------------------------------
STATION_STRIDE = 3                 # 123 stations on ifac
GAPS = [12.0, 8.0, 4.0, 2.0]       # forward distance from the car to the box
FEASIBLE_GATE_GAPS = (12.0, 8.0)   # the two the floor below is asserted at
MIN_FEASIBLE = 95                  # of the 123, at those gaps, for every tracking error
# HOW MANY of the corner stations can be planned at all, NOT the mean distance at which they can.
# The mean is degenerate as a gate: the harness drives the planner at a fixed 3 m/s, so the
# lookahead is max(lookahead_min, 1.5*3) = lookahead_min exactly, and a box further away than that
# is not in the planner's horizon at all. Every station that is plannable is therefore plannable at
# precisely lookahead_min and the mean over them is exactly that number -- it read 15.00 with 22 of
# 33 stations plannable and would still read 15.00 with one. The count is the quantity that moves.
MIN_CORNERS_PLANNABLE = 22         # of the 33 ifac stations with |kappa| > CORNER_KAPPA
# The corner grid the ramp ladder is FOR, and the cost of running it. The ladder is a re-plan per
# rung, so its budget trades corner feasibility against loop time, and the loop period is 50 ms:
# a planner that overruns it feeds the state machine's own staleness gate. Both sides are gated.
CORNER_GAPS = (12.0, 8.0, 4.0)     # 33 stations x 3 = 99 cells
SHIPPED_LADDER_MS = 20.0           # static_avoidance_params.yaml: ramp_search_max_ms
MIN_CORNER_CELLS = 46
MAX_LOOP_P95_MS = 40.0
MIN_SPEED_CAP = 2.50               # [m/s] mean over the feasible cells
A_LAT_MAX = 6.0
CORNER_KAPPA = 0.8
EARLY_GAPS = [round(0.5 * k, 2) for k in range(1, 41)]     # 0.5 .. 20.0 m


class _Grid:
    """GridFilter stand-in: GridFilter's own 255=free convention, erosion and pixel mapping."""

    def __init__(self, mapdir, name, kernel_size):
        import cv2
        import yaml
        from PIL import Image
        meta = yaml.safe_load((mapdir / f"{name}.yaml").read_text())
        self.resolution = meta["resolution"]
        self.origin = (meta["origin"][0], meta["origin"][1])
        img = np.array(Image.open(mapdir / meta["image"]).convert("L"))
        occ = img < int(255 * (1.0 - meta["occupied_thresh"]))
        base = np.flipud(np.where(occ, 0, 255).astype(np.uint8))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        self.eroded_image = cv2.erode(base, k)

    def is_point_inside(self, x, y):
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        if px < 0 or py < 0 or px >= self.eroded_image.shape[1] or py >= self.eroded_image.shape[0]:
            return False
        return self.eroded_image[py, px] == 255


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    warning = warn
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass


def load_node_module():
    san = types.ModuleType("san")
    san.__dict__["__file__"] = str(MOD)
    exec(compile(MOD.read_text(), str(MOD), "exec"), san.__dict__)
    return san


def load_map(name):
    mapdir = REPO / "stack_master/maps" / name
    wp = json.load(open(mapdir / "global_waypoints.json"))["global_traj_wpnts_iqp"]["wpnts"]
    return mapdir, wp


class Harness:
    """Everything that does not change between cells, built once."""

    def __init__(self, name):
        from frenet_conversion.frenet_converter import FrenetConverter
        self.san = load_node_module()
        self.name = name
        self.mapdir, self.wp = load_map(name)
        xy = np.array([[w["x_m"], w["y_m"]] for w in self.wp])
        self.converter = FrenetConverter(xy[:, 0], xy[:, 1])
        self.map_filter = _Grid(self.mapdir, name, 3)
        self.body_filter = _Grid(self.mapdir, name, 7)
        self.L = float(self.wp[-1]["s_m"] + (self.wp[1]["s_m"] - self.wp[0]["s_m"]))
        self.s_arr = np.array([w["s_m"] for w in self.wp])
        self.gbw = [types.SimpleNamespace(
            x_m=w["x_m"], y_m=w["y_m"], s_m=w["s_m"], d_left=w["d_left"], d_right=w["d_right"],
            vx_mps=w["vx_mps"], kappa_radpm=w["kappa_radpm"]) for w in self.wp]
        self.stations = list(range(0, len(self.wp) - 1, STATION_STRIDE))
        self.corners = [i for i in range(len(self.wp) - 1)
                        if abs(self.wp[i]["kappa_radpm"]) > CORNER_KAPPA]

    def _node(self, cur_d, ladder):
        san = self.san
        n = san.ObstacleSpliner.__new__(san.ObstacleSpliner)
        n.name = "static_avoidance_planner"
        n.get_logger = lambda: _Log()
        stamp = san.OTWpntArray().header.stamp
        n.get_clock = lambda: types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=0, to_msg=lambda: stamp))
        n.converter = self.converter
        n.map_filter, n.body_filter, n.body_kernel_size = self.map_filter, self.body_filter, 7
        n.width_car, n.safety_margin, n.safety_margin_d = 0.30, 0.15, 0.15
        n.wall_margin = 0.10
        n.lookahead_min, n.lookahead_k = 15.0, 1.5
        n.n_d_samples, n.sample_gaps, n.max_weave = 10, True, 3
        n.knot_merge_s_m = 0.4
        n.ramp_len, n.return_len, n.ramp_len_min_m = 4.5, 4.5, 2.5
        n.ramp_search_enable = bool(ladder)
        n.ramp_search_entry_m = [3.15, 2.5, 2.0, 1.5, 1.0]
        n.ramp_search_exit_m = [4.5, 2.5, 1.5]
        n.ramp_search_max_ms = 1e6               # offline: judge the ladder, not the machine
        n.obs_gather_extra_m, n.commit_drop_on_new_obstacle = 4.5, True
        # tail_m from static_avoidance_params.yaml, not the node default: a harness that plans a
        # metre of tail the car never gets is not measuring the shipped planner.
        n.tail_m, n.apex_bulge, n.preramp_len_m = 0.0, 0.05, 3.0
        n.kappa_add_max, n.kappa_abs_max = 2.0, 5.5
        n.a_lat_max, n.a_long_max, n.a_long_accel = A_LAT_MAX, 4.0, 3.0
        n.w_d, n.w_k, n.w_c, n.w_obs, n.obs_sigma = 1.0, 0.1, 5.0, 2.0, 0.5
        n.shift_min, n.shift_buffer, n.hold_after = 1.0, 0.5, 0.5
        n._d_end_prev = 0.0
        n._last_pub = None    # last path handed to the controller (handover blend)
        n.use_grid_check, n.trust_grid_bounds = True, True
        n.grid_scan_max, n.grid_scan_step, n.bounds_warn_m = 3.0, 0.05, 0.5
        n.clear_gate_enable, n.clear_margin_m, n.clear_hyst_m = True, 0.10, 0.03
        n.clear_max_cur_d, n.clear_latch_ttl_s = 0.15, 10.0
        n._clear_latch, n._line_clear = {}, False
        n.commit_enable, n._committed = False, None
        n.squeeze_enable, n.squeeze_steps = True, 2
        n.squeeze_safety_floor_m, n.squeeze_wall_floor_m = 0.05, 0.08
        n.squeeze_max_speed_mps = 3.0
        n.relax_hold_s, n._relax_until = 2.0, 0.0
        n.obs_memory_sec, n._mem_cands_obs, n._mem_cands_time = 0.5, [], None
        n._near_zero_since, n._moving_since = {}, {}
        n.static_near_zero_mps, n.static_promote_sec = 0.15, 0.5
        n.static_demote_mps, n.static_demote_sec = 0.35, 0.3
        n.reframe_warn_m, n._emit_markers = 0.05, False
        n._publish_feasible = lambda ok: None
        n._candidate_markers = lambda *a, **k: san.MarkerArray()
        n._store_commit = lambda *a, **k: None
        n.gb_max_s, n.gb_max_idx = self.L, len(self.wp)
        n.cur_vs, n.cur_d = 3.0, cur_d
        return n

    def cell(self, i, gap, cur_d, ladder=True, max_ms=None):
        """One plan. None, or (peak |d|, peak |kappa|, squeezed).

        `max_ms` overrides the ladder's time budget. The feasibility grid runs it UNBUDGETED on
        purpose -- it is there to judge the ladder, not the machine -- but any measurement of what
        a cycle COSTS has to use the budget that actually ships.
        """
        n = self._node(cur_d, ladder)
        if max_ms is not None:
            n.ramp_search_max_ms = float(max_ms)
        s_obs = self.wp[i]["s_m"]
        n.cur_s = (s_obs - gap) % self.L
        resp = self.converter.get_cartesian(np.array([n.cur_s]), np.array([cur_d]))
        pxy = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)[0]
        n.cur_x, n.cur_y = float(pxy[0]), float(pxy[1])
        j = int(np.argmin(np.abs(self.s_arr - n.cur_s)))
        j2 = (j + 1) % (len(self.wp) - 1)
        n.cur_yaw = float(np.arctan2(self.wp[j2]["y_m"] - self.wp[j]["y_m"],
                                     self.wp[j2]["x_m"] - self.wp[j]["x_m"]))
        w = self.wp[i]
        n.obstacles = [types.SimpleNamespace(
            id=1, s_start=(s_obs - 0.15) % self.L, s_end=(s_obs + 0.15) % self.L,
            s_center=s_obs, d_center=0.0, d_right=-0.15, d_left=0.15, size=0.30, vs=0.0, vd=0.0,
            is_static=True, is_visible=True, x_m=w["x_m"], y_m=w["y_m"], is_actually_a_gap=False)]
        n._promoted = {1}
        try:
            res = n.do_spline(self.gbw)
        except Exception:
            return None
        if res is None or res[0] is None or not len(res[0].wpnts):
            return None
        pts = res[0].wpnts
        return (max(abs(x.d_m) for x in pts),
                max(abs(x.kappa_radpm) for x in pts),
                res[0].ot_line == "squeeze")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="ifac")
    ap.add_argument("--check", action="store_true", help="exit 1 unless every gate passes")
    ap.add_argument("--gaps", type=float, nargs="*", default=GAPS)
    a = ap.parse_args()

    H = Harness(a.map)
    fails = []
    # check_track_bounds.py --all exits 1 today: map f ships with d_left/d_right SWAPPED on 402
    # stations against 0 correct, in all four of its waypoint sets. The corridor, the sampled
    # terminal offsets and the obstacle keep-out sides in this sweep all come from those bounds,
    # so a run on that map describes a mirrored track rather than this code. ifac is clean.
    if a.map in ("f",):
        print(f"  !! WARNING: map {a.map} ships with d_left/d_right SWAPPED "
              f"(check_track_bounds.py --all). These numbers are not evidence about the planner.")

    # --- feasibility + superset ------------------------------------------------------------
    cells = {}
    for ladder in (False, True):
        for cur_d in (0.0, 0.3):
            for gap in a.gaps:
                for i in H.stations:
                    cells[(ladder, cur_d, gap, i)] = H.cell(i, gap, cur_d, ladder)
    print(f"=== feasibility | map {a.map} | {len(H.stations)} stations (stride {STATION_STRIDE}) ===")
    print("  cur_d \\ gap | " + " | ".join(f"{g:5.1f} m" for g in a.gaps))
    for cur_d in (0.0, 0.3):
        row = []
        for gap in a.gaps:
            n_ok = sum(1 for i in H.stations if cells[(True, cur_d, gap, i)])
            row.append(f"{n_ok:4d}/{len(H.stations):<3d}")
            if gap in FEASIBLE_GATE_GAPS and n_ok < MIN_FEASIBLE:
                fails.append(f"feasible {n_ok} of {len(H.stations)} at gap {gap:.0f} m, "
                             f"cur_d {cur_d:.1f} (need {MIN_FEASIBLE})")
        print(f"     {cur_d:5.2f}    | " + " | ".join(row))
    lost = [k for k in cells if k[0] and not cells[k] and cells[(False,) + k[1:]]]
    gained = [k for k in cells if k[0] and cells[k] and not cells[(False,) + k[1:]]]
    print(f"ramp ladder: +{len(gained)} cells, -{len(lost)} cells "
          f"(rung 0 is the geometry the main pass already tried, so -N must be 0)")
    if lost:
        fails.append(f"the ramp ladder LOST {len(lost)} feasible cells: {lost[:5]}")

    # --- speed -----------------------------------------------------------------------------
    # over the WHOLE grid, every gap: a ladder that shortens a ramp does it wherever the
    # geometry is tight, and the tight cells are the near ones
    caps = [math.sqrt(A_LAT_MAX / max(v[1], 1e-3)) for k, v in cells.items() if k[0] and v]
    kap = [v[1] for k, v in cells.items() if k[0] and v]
    mean_cap = float(np.mean(caps)) if caps else 0.0
    print(f"=== speed | mean peak|kappa| {np.mean(kap):.3f} -> mean cap "
          f"sqrt(a_lat/kappa) = {mean_cap:.3f} m/s over {len(caps)} feasible cells ===")
    if mean_cap < MIN_SPEED_CAP:
        fails.append(f"mean speed cap {mean_cap:.3f} m/s < {MIN_SPEED_CAP} "
                     f"(the ladder is escaping into ramps that are too short)")

    # --- corners, and what the ladder costs to get them --------------------------------------
    t_cell = []
    n_corner_ok = 0
    for i in H.corners:
        for gap in CORNER_GAPS:
            t0 = time.perf_counter()
            got = H.cell(i, gap, 0.0, True, max_ms=SHIPPED_LADDER_MS)
            t_cell.append((time.perf_counter() - t0) * 1e3)
            n_corner_ok += 1 if got else 0
    n_cells = len(H.corners) * len(CORNER_GAPS)
    p95 = float(np.percentile(t_cell, 95)) if t_cell else 0.0
    print(f"=== corners | {n_corner_ok}/{n_cells} feasible (gate {MIN_CORNER_CELLS}) | planner "
          f"loop p50 {np.percentile(t_cell, 50):.1f} ms p95 {p95:.1f} ms "
          f"(gate {MAX_LOOP_P95_MS}, period 50) ===")
    if n_corner_ok < MIN_CORNER_CELLS:
        fails.append(f"{n_corner_ok} of {n_cells} corner cells feasible < {MIN_CORNER_CELLS}")
    if p95 > MAX_LOOP_P95_MS:
        fails.append(f"planner loop p95 {p95:.1f} ms > {MAX_LOOP_P95_MS} (period 50 ms)")

    # --- earliness -------------------------------------------------------------------------
    first = {}
    for i in H.corners:
        first[i] = None
        for g in reversed(EARLY_GAPS):
            if H.cell(i, g, 0.0, True) is not None:
                first[i] = g
                break
    got = [v for v in first.values() if v is not None]
    mean_early = float(np.mean(got)) if got else 0.0
    # the mean over ALL corner stations, unplannable counted as zero: unlike the mean over the
    # plannable ones it is not pinned to the lookahead, so it is worth printing
    mean_all = float(np.mean([v if v is not None else 0.0 for v in first.values()]))
    print(f"=== earliness | {len(H.corners)} corner stations (|kappa| > {CORNER_KAPPA}) | "
          f"{len(got)} ever plannable (gate {MIN_CORNERS_PLANNABLE}) | greatest-plannable "
          f"distance {mean_early:.2f} m over those, {mean_all:.2f} m over all ===")
    if len(got) < MIN_CORNERS_PLANNABLE:
        fails.append(f"only {len(got)} of {len(H.corners)} corner stations are plannable at any "
                     f"distance (need {MIN_CORNERS_PLANNABLE})")

    if fails:
        print("\nFAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1 if a.check else 0
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
