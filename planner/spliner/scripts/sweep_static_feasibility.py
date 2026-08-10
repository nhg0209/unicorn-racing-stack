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
# THE TWO CORNER GATES ARE FRACTIONS OF WHAT THE MAP HAS, NOT COUNTS.
#
# They were absolute: MIN_CORNER_CELLS = 46 and MIN_CORNERS_PLANNABLE = 22, both set against
# ifac, which has 33 stations over CORNER_KAPPA and therefore 33 * len(CORNER_GAPS) = 99 corner
# cells. On a map with fewer corners the same integers are not a weaker or stronger gate, they are
# an IMPOSSIBLE one: ifac_0807 has 12 such stations and 36 cells, so 46 could never be reached and
# the map was a permanent FAIL for owning a flatter layout. A gate no map can pass is not
# measuring the planner.
#
# The fractions are read straight off the numbers that were already there -- 46/99 and 22/33 --
# so on ifac the thresholds come out at exactly 46 and 22 and the gate is bit-identical to the one
# it replaces. Nothing was chosen to make a second map pass; what the second map then does is a
# measurement. (It passes: 18 cells against ceil(36 * 46/99) = 17, and 8 stations against
# ceil(12 * 2/3) = 8 -- the second by a margin of zero, which is worth knowing and is not worth
# hiding by rounding the fraction down.)
#
# CEIL, not round: rounding down would quietly weaken the fraction on every map whose count is not
# a multiple of the denominator.
#
# A caveat the fraction cannot fix: with 12 corner stations one station is 8.3 % of the gate, so
# on such a map this check is coarse. That is a property of the map, not of the threshold.
MIN_CORNER_CELL_FRAC = 46.0 / 99.0
# CORNER HEADROOM, IN METRES, REPLACING THE "ever plannable" COUNT.
#
# That count was gated at 22/33 on ifac and, as a fraction, at 8/12 on ifac_0807 -- which the map
# met with a margin of exactly ZERO, one station from failing. Widening the fraction would have
# been tuning to pass, so the axis was measured instead, and the count turns out to be the wrong
# quantity for two reasons:
#
#   IT IS QUANTISED. On a 12-corner map one station is 8.3 % of the gate, so the verdict can flip
#   on a single cell of a time-budgeted ladder that is not reproducible run to run (the corner
#   cell count itself reads 56 or 57 of 99 on ifac depending on the run).
#
#   IT IS A COUNT, AND THE QUESTION IS A DISTANCE. "Plannable at some distance" cannot tell a
#   corner with half a metre of room from one that is a centimetre from impossible; both are 1.
#   Measured on ifac, the stations it counts span headroom from 0.0 to 10.8 cm and the count is
#   the same number either way. It also drifts from the distances the car plans at: on ifac_0807
#   8 stations are "ever plannable" but only 6 at any distance in CORNER_GAPS (12, 8, 4 m).
#   ("ever plannable" additionally includes 15.0 m = lookahead_min, where the box sits exactly on
#   the horizon; that is the easiest sample in EARLY_GAPS and every ever-plannable station on both
#   maps is plannable there.)
#
# What replaces it is continuous and is about the track: for each corner station, how much margin
# would have to be given up for a corridor to exist at all (sweep_static_corridor.min_violation,
# delta off safety_margin_d and wall_margin together, half_car untouched). Metres, not counts, so
# a station flipping moves the statistic by millimetres rather than by 8 %.
#
# THRESHOLDS ARE DERIVED FROM ifac AS IT STANDS, not chosen to make anything pass. ifac measures
# p90 = 7.7 cm and max = 10.8 cm over its 33 corners today. The gates are set 1.3 and 2.2 cm above
# those so ordinary variation does not flip them, and the derivation is here so the next person
# can see the slack is deliberate and how big it is. ifac_0807 then reads 4.9 / 6.6 cm -- it
# passes, and it passes with room, which is the difference between a gate and a coin toss.
CORNER_HEADROOM_P90_M = 0.090      # ifac today: 0.077
CORNER_HEADROOM_MAX_M = 0.130      # ifac today: 0.108
# The corner grid the ramp ladder is FOR, and the cost of running it. The ladder is a re-plan per
# rung, so its budget trades corner feasibility against loop time, and the loop period is 50 ms:
# a planner that overruns it feeds the state machine's own staleness gate. Both sides are gated.
CORNER_GAPS = (12.0, 8.0, 4.0)     # 33 stations x 3 = 99 cells on ifac
SHIPPED_LADDER_MS = 20.0           # static_avoidance_params.yaml: ramp_search_max_ms
MAX_LOOP_P95_MS = 40.0
# WHAT THE LADDER COSTS, AS GEOMETRY. This gate exists to catch a ramp search that buys
# feasibility by bending the path harder than it should -- a shape question. It used to be
# expressed as a SPEED, mean sqrt(a_lat_max / |kappa|) >= 2.50 m/s, which folds the vehicle's
# lateral budget into a threshold about geometry: the same paths score 2.630 m/s at a_lat 6.0,
# 2.278 at 4.5 and 3.037 at 8.0 while not one waypoint moves. The repo's two vehicle configs
# disagree by 78 % on that number (CAR ay_max 4.5, SIM 8.0), so the gate moved with the config
# and reported it as a planner change. Same disease as G2 measuring SPATIAL and REPLAN as one
# number (see sweep_avoidance_stability).
#
# So the gate is now the geometric half alone, mean sqrt(1 / |kappa|), and the threshold is the
# old one with the vehicle divided back out -- an EXACT re-expression, not a re-tune:
#
#     MIN_BEND = MIN_SPEED_CAP / sqrt(A_LAT_MAX) = 2.50 / sqrt(6.0) = 1.02062
#
# so today's verdict is bit-for-bit the verdict it replaces (measured 1.0737 against 1.0206, the
# same margin as 2.630 against 2.50) and it stops moving when the vehicle file does. The speed is
# still PRINTED, with the a_lat_max it was computed at, because what the driver feels is a speed.
# Units are sqrt(m): sqrt(1/kappa) is the square root of a radius. Ugly, and honest.
MIN_BEND = 2.50 / math.sqrt(6.0)   # [sqrt(m)] mean sqrt(1/|kappa|) over the feasible cells
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
    """The waypoint list EXACTLY as the node receives it on /global_waypoints.

    global_republisher reads this same array out of global_waypoints.json and publishes it
    unmodified, so the duplicate closing point is KEPT here: the file repeats point 0
    byte-for-byte (x, y, d_left, d_right, kappa all identical, only id and s_m advance) to close
    the loop, and that row reaches the node. Dropping it -- as the offline reopt scripts do,
    because a solver cannot take a repeated station -- would give this harness a different lap
    length and a different index modulus from the planner it is supposed to be measuring.

    What the duplicate settles is the LAP LENGTH. wpnts[-1].s_m is the arc length of point 0 gone
    all the way round, i.e. the lap; it is NOT the last distinct station, and adding a spacing to
    it (as this harness used to) overshoots the lap by exactly one waypoint.
    """
    mapdir = REPO / "stack_master/maps" / name
    wp = json.load(open(mapdir / "global_waypoints.json"))["global_traj_wpnts_iqp"]["wpnts"]
    return mapdir, wp


def load_params():
    """The SHIPPED planner parameters, read from the yaml race.launch.xml attaches.

    Hand-copied literals drift, and had: this harness ran wall_margin 0.10 against the shipped
    0.15, knot_merge_s_m 0.4 against 1.2 and kappa_add_max 2.0 against 5.0. Those are not small
    -- the first two decide how much corridor a candidate may use and whether two nearby boxes
    get one apex or two, and the third is the curvature gate itself. A harness whose whole claim
    is that it drives the real do_spline has to drive it with the real numbers.
    """
    import yaml
    p = yaml.safe_load((REPO / "stack_master/config/static_avoidance_params.yaml").read_text())
    return p["static_avoidance_planner"]["ros__parameters"]


class Harness:
    """Everything that does not change between cells, built once."""

    def __init__(self, name):
        from frenet_conversion.frenet_converter import FrenetConverter
        self.san = load_node_module()
        self.name = name
        self.mapdir, self.wp = load_map(name)
        self.P = load_params()
        xy = np.array([[w["x_m"], w["y_m"]] for w in self.wp])
        # psi as well, exactly as ObstacleSpliner.initialize_converter builds it. do_spline never
        # calls the one method that reads it (get_frenet_velocities), so this changes no number
        # today -- it removes a way for the two frames to diverge later.
        self.converter = FrenetConverter(xy[:, 0], xy[:, 1],
                                         np.array([w["psi_rad"] for w in self.wp]))
        self.map_filter = _Grid(self.mapdir, name, int(self.P["kernel_size"]))
        self.body_filter = _Grid(self.mapdir, name, int(self.P["body_kernel_size"]))
        # gb_cb: gb_max_s = wpnts[-1].s_m. NOT that plus a spacing -- the last row IS point 0 come
        # round again (see load_map), so its s already is the lap. The old +ds put every `% L` in
        # this harness one waypoint out of phase with the planner's.
        self.L = float(self.wp[-1]["s_m"])
        self.s_arr = np.array([w["s_m"] for w in self.wp])
        self.gbw = [types.SimpleNamespace(
            x_m=w["x_m"], y_m=w["y_m"], s_m=w["s_m"], d_left=w["d_left"], d_right=w["d_right"],
            vx_mps=w["vx_mps"], kappa_radpm=w["kappa_radpm"]) for w in self.wp]
        # len(wp) - 1: the last row is the repeated point 0, so it is not a distinct station
        self.stations = list(range(0, len(self.wp) - 1, STATION_STRIDE))
        self.corners = [i for i in range(len(self.wp) - 1)
                        if abs(self.wp[i]["kappa_radpm"]) > CORNER_KAPPA]
        self.last_exc = None     # cell() swallows exceptions to keep a sweep going; keep the last
        self.cur_vs = 3.0        # see _node: this is the squeeze gate as well as the lookahead
        self.force_relax = False  # emulate a live /planner/avoidance/relax request
        # d(s) generator, or None for whatever the yaml ships. An OVERRIDE, not a second source:
        # every other planner value still comes from the yaml, so a sweep that sets nothing here
        # measures exactly the shipped configuration.
        self.plan_method = None
        self.pin_apex = None
        self.qp_ladder = None

    def _node(self, cur_d, ladder):
        san = self.san
        n = san.ObstacleSpliner.__new__(san.ObstacleSpliner)
        n.name = "static_avoidance_planner"
        n.get_logger = lambda: _Log()
        stamp = san.OTWpntArray().header.stamp
        n.get_clock = lambda: types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=0, to_msg=lambda: stamp))
        n.converter = self.converter
        n.map_filter, n.body_filter = self.map_filter, self.body_filter
        # Every declared parameter, straight from the shipped yaml, under the node's own name.
        # dyn_param_cb is a name -> same-named attribute assignment throughout (the one exception,
        # the deprecated kappa_max alias, is not in the yaml), so this reproduces the startup sync.
        for k, v in self.P.items():
            if k in ("measure", "from_bag", "kernel_size", "body_kernel_size"):
                continue                  # not planning geometry / already spent on the two grids
            setattr(n, k, v)
        n.body_kernel_size = int(self.P["body_kernel_size"])
        # --- the only deliberate departures from the shipped configuration, each for a reason ---
        n.ramp_search_enable = bool(ladder)      # the superset check needs the ladder off as well
        # corridor_qp skips the ladder BY ITS OWN RULE (it opens no cell there and costs the cycle
        # 2.4x), so the switch above does not reach it. Deliberately NOT tied to that switch: the
        # default sweep has to measure the SHIPPED configuration, and tying them made the default
        # run measure a forced-on ladder instead -- a sweep that reports on a setting nobody runs.
        # `qp_ladder` overrides it, which is how the claim stays re-testable.
        if self.qp_ladder is not None:
            n.corridor_qp_ramp_ladder = bool(self.qp_ladder)
        n.ramp_search_max_ms = 1e6               # offline: judge the ladder, not the machine
        n.commit_enable = False                  # one-shot cells: nothing to reuse, nothing to store
        n.obs_memory_sec = 0.5                   # node default; not a declared parameter
        n._d_end_prev = 0.0
        n._last_pub = None    # last path handed to the controller (handover blend)
        n._clear_latch, n._line_clear = {}, False
        n._committed = None
        # relax_cb sets this to now + relax_hold_s; the clock stub reads 0, so any positive value
        # is a live request. This is what the state machine raises after static_deadlock_timeout_s
        # of standstill, and it overrides the squeeze's enable flag and speed gate (never the
        # floors) -- so it is the only way to see the squeeze at racing speed.
        n._relax_until = 1e18 if self.force_relax else 0.0
        n._mem_cands_obs, n._mem_cands_time = [], None
        n._near_zero_since, n._moving_since = {}, {}
        n._emit_markers = False
        n._publish_feasible = lambda ok: None
        n._candidate_markers = lambda *a, **k: san.MarkerArray()
        n._store_commit = lambda *a, **k: None
        # gb_cb, verbatim: the lap length and the index modulus are the LAST row's own fields.
        # gb_max_idx was len(wp) here, one more than the id the node reads, so `idx % gb_max_idx`
        # never wrapped to 0 -- it landed on the duplicate row instead.
        n.gb_max_s, n.gb_max_idx = self.L, int(self.wp[-1]["id"])
        # CUR_VS IS A GATE, NOT JUST A LOOKAHEAD. 3.0 was chosen so the speed-proportional
        # lookahead sits at lookahead_min, and it silently disabled the squeeze pass for every
        # measurement ever taken with this harness: _squeeze_schedule returns [] when
        # `v >= squeeze_max_speed_mps`, and squeeze_max_speed_mps is 3.0, so the reduced-margin
        # retry -- the one mechanism in the planner whose entire purpose is to avoid a TRAILING
        # standstill -- never ran once. Kept at 3.0 as the DEFAULT so every number this campaign
        # has already reported stays reproducible, and made settable so the other side of that
        # threshold can be measured rather than assumed.
        n.cur_vs, n.cur_d = self.cur_vs, cur_d
        if self.plan_method is not None:
            n.static_plan_method = str(self.plan_method)
        if self.pin_apex is not None:
            n.corridor_qp_pin_apex = bool(self.pin_apex)
        return n

    OBS_HALF = 0.15                       # [m] box half-extent, in s and in d

    def make_node(self, i, gap, cur_d, ladder=True, max_ms=None, boxes=((0.0, 0.0),)):
        """A node posed at `gap` metres before station `i`, with `boxes` in front of it.

        `boxes` is [(ds, d_center), ...]: metres PAST the station in s, and the box centre's
        lateral offset from the raceline. The default is the single centred box the feasibility
        grid has always used, so that grid is unchanged.
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
        h = self.OBS_HALF
        s_arr = np.array([(s_obs + ds) % self.L for ds, _dd in boxes])
        d_arr = np.array([dd for _ds, dd in boxes])
        bxy = self.converter.get_cartesian(s_arr, d_arr)
        bxy = (bxy.T if bxy.ndim == 2 else bxy).reshape(-1, 2)
        n.obstacles = [types.SimpleNamespace(
            id=k + 1, s_start=(s_c - h) % self.L, s_end=(s_c + h) % self.L, s_center=s_c,
            d_center=float(d_c), d_right=float(d_c) - h, d_left=float(d_c) + h, size=2 * h,
            vs=0.0, vd=0.0, is_static=True, is_visible=True,
            x_m=float(bxy[k, 0]), y_m=float(bxy[k, 1]), is_actually_a_gap=False)
            for k, (s_c, d_c) in enumerate(zip(s_arr, d_arr))]
        n._promoted = {o.id for o in n.obstacles}
        return n

    def cell(self, i, gap, cur_d, ladder=True, max_ms=None):
        """One plan. None, or (peak |d|, peak |kappa|, squeezed).

        `max_ms` overrides the ladder's time budget. The feasibility grid runs it UNBUDGETED on
        purpose -- it is there to judge the ladder, not the machine -- but any measurement of what
        a cycle COSTS has to use the budget that actually ships.
        """
        n = self.make_node(i, gap, cur_d, ladder, max_ms)
        try:
            res = n.do_spline(self.gbw)
        except Exception as e:
            self.last_exc = e
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
    # The A_LAT_MAX-vs-planner consistency check that used to be here is GONE because the
    # constant it compared is gone: the reported speed now reads H.P["a_lat_max"] directly, so the
    # two cannot disagree. A guard that exists to catch a copy is worth less than not copying.
    # MEASURED PER MAP, not looked up. This was `if a.map in ("f",)` -- a hardcoded literal, so
    # ifac_0807 arrived with the same defect and the warning stayed silent for it. The corridor,
    # the sampled terminal offsets and the obstacle keep-out sides all come from those bounds, so
    # a run on such a map describes a mirrored track rather than this code.
    # This block used `os` and the module never imported it, so every run raised NameError into a
    # bare `except: pass` and the warning has never once been printed -- on any map, including the
    # ones it was added for. Rebuilt on REPO (a Path that is already correct) so there is nothing
    # left to forget to import, and the except now SAYS when the check could not run: a silent
    # swallow is how a check that never ran passes for a check that found nothing.
    try:
        sys.path.insert(0, str(REPO / "planner" / "gb_optimizer"))
        from gb_optimizer import track_bounds as _tb
        _d = REPO / "stack_master/maps" / a.map
        _wp = json.load(open(_d / "global_waypoints.json"))["global_traj_wpnts_iqp"]["wpnts"]
        _v, _ok, _sw, _ = _tb.verdict([[w["x_m"], w["y_m"]] for w in _wp],
                                      [w["d_right"] for w in _wp], [w["d_left"] for w in _wp],
                                      str(_d), a.map)
        if _v == "SWAPPED":
            print(f"  !! WARNING: map {a.map} ships with d_left/d_right SWAPPED ({_sw} stations "
                  f"against {_ok}). These numbers are not evidence about the planner.")
    except Exception as e:
        print(f"  (track-bounds check unavailable: {type(e).__name__}: {e})")

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
    kap = [v[1] for k, v in cells.items() if k[0] and v]
    bend = [math.sqrt(1.0 / max(v[1], 1e-3)) for k, v in cells.items() if k[0] and v]
    mean_bend = float(np.mean(bend)) if bend else 0.0
    # The speed is the same number times sqrt(a_lat_max) -- reported, never gated, and printed
    # with the a_lat it came from so it can never again be read as if it were vehicle-independent.
    # Taken from the node's own parameter rather than a constant here: that constant was a NINTH
    # copy of the vehicle's lateral budget, and it disagreed with both configs.
    a_lat = float(H.P["a_lat_max"])
    print(f"=== bend | mean peak|kappa| {np.mean(kap):.3f} -> mean sqrt(1/|kappa|) = "
          f"{mean_bend:.4f} sqrt(m) over {len(bend)} feasible cells (gate {MIN_BEND:.4f}) ===")
    print(f"    at the planner's a_lat_max = {a_lat:.2f} that is a mean speed cap of "
          f"{mean_bend * math.sqrt(a_lat):.3f} m/s   [reported, NOT gated]")
    if mean_bend < MIN_BEND:
        fails.append(f"mean sqrt(1/|kappa|) {mean_bend:.4f} < {MIN_BEND:.4f} sqrt(m) "
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
    min_corner_cells = math.ceil(MIN_CORNER_CELL_FRAC * n_cells)
    p95 = float(np.percentile(t_cell, 95)) if t_cell else 0.0
    print(f"=== corners | {n_corner_ok}/{n_cells} feasible (gate {min_corner_cells} = "
          f"ceil({MIN_CORNER_CELL_FRAC:.4f} x {n_cells})) | planner "
          f"loop p50 {np.percentile(t_cell, 50):.1f} ms p95 {p95:.1f} ms "
          f"(gate {MAX_LOOP_P95_MS}, period 50) ===")
    if n_corner_ok < min_corner_cells:
        fails.append(f"{n_corner_ok} of {n_cells} corner cells feasible < {min_corner_cells}")
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
    # REPORTED, NOT GATED -- see CORNER_HEADROOM_*: "ever plannable" is satisfied at lookahead_min
    # by construction, so it describes the horizon rather than the corner.
    n_gap_ok = sum(1 for i in H.corners
                   if any(H.cell(i, g, 0.0, True) for g in CORNER_GAPS))
    print(f"=== earliness | {len(H.corners)} corner stations (|kappa| > {CORNER_KAPPA}) | "
          f"{len(got)} ever plannable (reported, not gated; {n_gap_ok} of them at a distance in "
          f"CORNER_GAPS) | greatest-plannable distance {mean_early:.2f} m over those, "
          f"{mean_all:.2f} m over all ===")

    # --- corner headroom: how many centimetres short is the tightest corner? ------------------
    from sweep_static_corridor import Corridor          # local: it imports this module
    C = Corridor(H)
    head = []
    for i in H.corners:
        d = C.min_violation(i, 12.0, 0.0, ((0.0, 0.0),))
        head.append(0.15 if d is None else d)           # None = not reachable at any legal margin
    p90 = float(np.percentile(head, 90))
    hmax = float(np.max(head))
    print(f"=== corner headroom | margin that would have to be given up for a corridor to exist, "
          f"over {len(head)} corners | median {100*np.median(head):.1f} cm | "
          f"p90 {100*p90:.1f} cm (gate {100*CORNER_HEADROOM_P90_M:.1f}) | "
          f"max {100*hmax:.1f} cm (gate {100*CORNER_HEADROOM_MAX_M:.1f}) ===")
    if p90 > CORNER_HEADROOM_P90_M:
        fails.append(f"corner headroom p90 {100*p90:.1f} cm > {100*CORNER_HEADROOM_P90_M:.1f} "
                     f"(nine corners in ten must be within that of a usable corridor)")
    if hmax > CORNER_HEADROOM_MAX_M:
        fails.append(f"worst corner headroom {100*hmax:.1f} cm > "
                     f"{100*CORNER_HEADROOM_MAX_M:.1f}")

    if fails:
        print("\nFAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1 if a.check else 0
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
