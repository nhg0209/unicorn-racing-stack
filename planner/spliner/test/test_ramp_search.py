#!/usr/bin/env python3
"""The two halves of "the avoidance starts far too late in a narrow corner".

1. The adaptive ramp scan read the corridor at the WRONG STATION. scan_s is path-local -- distance
   from the car -- and _grid_corridor_batch treats its argument as an absolute station. Same bug
   6b8112e fixed for knot_cor, left behind in the ramp scan. The sampled points slid along the
   track as the car approached, so the ramp was shortened, or not, by a corridor belonging
   somewhere else entirely.

2. The 4.5 m entry ramp is laid across the pinch in a narrow corner, and every candidate dies on
   it. Nothing shortens the ramp except the corridor scan agreeing to, so the path only appears
   once the gap itself has fallen below the ramp length -- i.e. with the car right on top of the
   box. The ramp pair is now a SEARCH DIMENSION, retried longest-first when every candidate at
   today's geometry was rejected.

Run (after sourcing the workspace):
  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/test/test_ramp_search.py
"""
import types
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parents[1] / "spliner" / "static_avoidance_node.py"
san = types.ModuleType("san")
san.__dict__["__file__"] = str(MOD)
exec(compile(MOD.read_text(), str(MOD), "exec"), san.__dict__)
ObstacleSpliner = san.ObstacleSpliner

TRACK_LEN = 60.0
WPNT_DIST = 0.1
HALF_WIDTH = 1.2


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    warning = warn
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass


class _Clock:
    def now(self):
        return types.SimpleNamespace(nanoseconds=0,
                                     to_msg=lambda: san.OTWpntArray().header.stamp)


class _Converter:
    """Straight raceline along +x; +d is +y."""
    @staticmethod
    def get_cartesian(s, d):
        return np.vstack([np.asarray(s, float) % TRACK_LEN, np.asarray(d, float)])

    @staticmethod
    def get_e_psi(x, y, yaw):
        return 0.0


class _PinchGrid:
    """Eroded-map stand-in: free within +-w_pinch of the raceline over [lo, hi), +-w_open elsewhere.

    A short pinch a fixed distance BEFORE the box is the geometry the ladder exists for. The
    offset a ramp carries at a fixed station falls with the ramp length, so shortening the ramp is
    what gets the path through -- and below ramp_len_min_m the adaptive fit cannot go.
    """

    def __init__(self, lo, hi, w_pinch, w_open, resolution=0.05):
        self.resolution = resolution
        self.origin = (-1.0, -3.0)
        h = int(6.0 / resolution)
        w = int((TRACK_LEN + 2.0) / resolution)
        y = (np.arange(h) + 0.5) * resolution + self.origin[1]
        x = (np.arange(w) + 0.5) * resolution + self.origin[0]
        pinch = (x >= lo) & (x < hi)
        free = np.where(pinch[None, :], np.abs(y)[:, None] <= w_pinch,
                        np.abs(y)[:, None] <= w_open)
        self.eroded_image = np.where(free, 255, 0).astype(np.uint8)

    def is_point_inside(self, x, y):
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        if px < 0 or py < 0 or px >= self.eroded_image.shape[1] or py >= self.eroded_image.shape[0]:
            return False
        return self.eroded_image[py, px] == 255


class _BandGrid:
    """Eroded-map stand-in: free within +-w_a of the raceline before `split`, +-w_b after it.

    The whole point is that the corridor DIFFERS BY STATION, so reading it at the wrong s gives a
    different answer -- which is the defect under test.
    """

    def __init__(self, split, w_a, w_b, resolution=0.05):
        self.resolution = resolution
        self.origin = (-1.0, -3.0)
        h = int(6.0 / resolution)
        w = int((TRACK_LEN + 2.0) / resolution)
        y = (np.arange(h) + 0.5) * resolution + self.origin[1]
        x = (np.arange(w) + 0.5) * resolution + self.origin[0]
        a = (np.abs(y)[:, None] <= w_a) & (x[None, :] < split)
        b = (np.abs(y)[:, None] <= w_b) & (x[None, :] >= split)
        self.eroded_image = np.where(a | b, 255, 0).astype(np.uint8)

    def is_point_inside(self, x, y):
        """Verbatim GridFilter.is_point_inside -- _path_off_track calls it per point."""
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        if px < 0 or py < 0 or px >= self.eroded_image.shape[1] or py >= self.eroded_image.shape[0]:
            return False
        return self.eroded_image[py, px] == 255


def gb_wpnts():
    return [types.SimpleNamespace(x_m=i * WPNT_DIST, y_m=0.0, s_m=i * WPNT_DIST,
                                  d_left=HALF_WIDTH, d_right=HALF_WIDTH,
                                  vx_mps=4.0, kappa_radpm=0.0)
            for i in range(int(TRACK_LEN / WPNT_DIST))]


def box(s, d, half_d=0.15, half_s=0.15, oid=None):
    return types.SimpleNamespace(
        id=oid if oid is not None else int(s * 10), s_start=(s - half_s) % TRACK_LEN,
        s_end=(s + half_s) % TRACK_LEN, s_center=s, d_center=d,
        d_right=d - half_d, d_left=d + half_d, size=2 * half_d, vs=0.0, vd=0.0,
        is_static=True, is_visible=True, x_m=s, y_m=d, is_actually_a_gap=False)


def planner(obstacles, cur_s, cur_d=0.0):
    n = ObstacleSpliner.__new__(ObstacleSpliner)
    n.name = "static_avoidance_planner"
    n.get_logger = lambda: _Log()
    n.get_clock = lambda: _Clock()
    n.converter = _Converter()
    n.cur_s, n.cur_d, n.cur_vs = cur_s, cur_d, 3.0
    n.cur_x, n.cur_y, n.cur_yaw = cur_s, cur_d, 0.0
    n.gb_max_s, n.gb_max_idx = TRACK_LEN, int(TRACK_LEN / WPNT_DIST)
    n.obstacles = obstacles
    n.width_car, n.safety_margin, n.wall_margin = 0.30, 0.15, 0.10
    n.safety_margin_d = n.safety_margin
    n.lookahead_min, n.lookahead_k = 15.0, 1.5
    n.n_d_samples, n.sample_gaps = 10, True
    n.max_weave = 3
    n.knot_merge_s_m = 0.4
    n.ramp_len, n.return_len, n.tail_m = 4.5, 4.5, 0.0
    n.ramp_len_min_m = 2.5
    n.ramp_search_enable = True
    # THE LADDER IS A SAMPLE MECHANISM, so this file has to say so now that it is not the default.
    # do_spline gates the ladder on `static_plan_method != "corridor_qp" or corridor_qp_ramp_ladder`
    # -- it was measured to open exactly ZERO cells under corridor_qp over 8484 (ifac) and 9462
    # (ifac_0807) race cells while costing the cycle p95 111 -> 45 ms, so the corridor path skips
    # it. This fixture used to inherit "sample" from the class default and get the ladder for free;
    # the default is corridor_qp now, and a test about the ladder must name the method that has one.
    n.static_plan_method = "sample"
    n.ramp_search_entry_m = [3.15, 2.5, 2.0, 1.5, 1.0]
    n.ramp_search_exit_m = [4.5, 2.5, 1.5]
    n.ramp_search_max_ms = 1000.0            # offline: never cut the ladder short on time
    n.apex_bulge, n.preramp_len_m = 0.05, 3.0
    n.kappa_add_max, n.kappa_abs_max = 5.0, 5.5
    n.a_lat_max, n.a_long_max, n.a_long_accel = 6.0, 4.0, 3.0
    n.w_d, n.w_k, n.w_c, n.w_obs, n.obs_sigma = 1.0, 0.1, 5.0, 2.0, 0.5
    n.shift_min, n.shift_buffer, n.hold_after = 1.0, 0.5, 0.5
    n._d_end_prev = 0.0
    n._last_pub = None    # last path handed to the controller (handover blend)
    n.use_grid_check, n.trust_grid_bounds = False, False
    n.grid_scan_max, n.grid_scan_step, n.bounds_warn_m = 3.0, 0.05, 0.5
    n.clear_gate_enable, n.clear_margin_m, n.clear_hyst_m = False, 0.10, 0.03
    n.clear_max_cur_d, n.clear_latch_ttl_s = 0.15, 10.0
    n._clear_latch, n._line_clear = {}, False
    n.commit_enable, n._committed = False, None
    n.squeeze_enable = False
    n.squeeze_steps, n.squeeze_safety_floor_m = 2, 0.05
    n.squeeze_wall_floor_m, n.squeeze_max_speed_mps = 0.08, 3.0
    n.relax_hold_s, n._relax_until = 2.0, 0.0
    n.obs_memory_sec, n._mem_cands_obs, n._mem_cands_time = 0.5, [], None
    n._promoted = {o.id for o in obstacles}
    n._near_zero_since, n._moving_since = {}, {}
    n.static_near_zero_mps, n.static_promote_sec = 0.15, 0.5
    n.static_demote_mps, n.static_demote_sec = 0.35, 0.3
    n.reframe_warn_m = 0.05
    n.obs_gather_extra_m = 4.5
    n.commit_drop_on_new_obstacle = True
    n._emit_markers = False
    n.feasible = []
    n._publish_feasible = lambda ok: n.feasible.append(ok)
    n._candidate_markers = lambda *a, **k: san.MarkerArray()
    n._store_commit = lambda *a, **k: None
    return n


def hump_start(w, cur_s):
    """Path-local station at which the published offset leaves the raceline."""
    # the quintic leaves the raceline with zero value AND zero slope, so anything but a tiny
    # threshold reads the ramp as shorter than it is
    off = [x for x in w.wpnts if abs(x.d_m) > 1e-6]
    if not off:
        return None
    return min((x.s_m - cur_s) % TRACK_LEN for x in off)


def test_the_ramp_scan_reads_the_real_stations():
    # The map is NARROW before s = 20 and WIDE after it. The car sits at s = 25 with a box 8 m
    # ahead, so the ramp's real span (28.5 .. 37.5) is entirely in the wide part and the full 4.5 m
    # entry ramp fits. Its PATH-LOCAL span (3.5 .. 12.5) is entirely in the narrow part, where no
    # ramp length can carry the offset -- so a scan that reads path-local stations falls to the
    # bottom of its ladder and opens the hump 2 m late.
    n = planner([box(33.0, -0.35, oid=1)], cur_s=25.0)
    n.trust_grid_bounds = True
    n.map_filter = _BandGrid(split=20.0, w_a=0.20, w_b=1.20)
    real = n._grid_corridor(33.0, wall_margin=n.wall_margin)
    phantom = n._grid_corridor(8.0, wall_margin=n.wall_margin)
    assert real is not None and phantom is not None
    assert (real[1] - real[0]) - (phantom[1] - phantom[0]) > 1.0, \
        "the harness must make the real and the path-local read genuinely differ"
    w, _m = n.do_spline(gb_wpnts())
    assert w.wpnts, "the box must be plannable at all"
    start = hump_start(w, 25.0)
    r_in = 8.0 - start
    assert r_in > 4.0, (
        f"the hump opened {r_in:.2f} m before the box, not the full {n.ramp_len:.1f} m: the ramp "
        f"scan is reading the corridor {n.cur_s:.0f} m away from where the ramp actually is")
    print(f"PASS the ramp scan reads its real stations (entry ramp {r_in:.2f} m, "
          f"corridor there {real[1]-real[0]:.2f} m wide vs {phantom[1]-phantom[0]:.2f} m "
          f"at the path-local station)")


def test_the_ladder_finds_a_path_the_full_ramp_cannot_fit():
    # A box in the MIDDLE of the track (pass offset 0.45 m either way) with a 0.4 m pinch one
    # metre before it. The offset a ramp is carrying at a fixed station falls with the ramp
    # length, so the pinch is passable -- but only by a ramp shorter than ramp_len_min_m, which is
    # the floor the adaptive fit cannot go below. It returns 2.5 m, the path puts 0.31 m of offset
    # into a 0.20 m gap, and every candidate is rejected.
    def run(enable):
        n = planner([box(33.0, 0.0, oid=1)], cur_s=25.0)
        n.trust_grid_bounds, n.use_grid_check = True, True
        n.ramp_search_enable = enable
        n.map_filter = n.body_filter = _PinchGrid(lo=31.6, hi=32.0, w_pinch=0.20, w_open=1.20)
        n.body_kernel_size = 7
        w, _m = n.do_spline(gb_wpnts())
        return w

    assert not run(False).wpnts, (
        "the harness must reproduce the failure: at today's ramp geometry, floored at "
        "ramp_len_min_m, every candidate is rejected here")
    w = run(True)
    assert w.wpnts, "the ladder must find the shorter ramp that fits the pinch"
    start = hump_start(w, 25.0)
    assert 8.0 - start < 2.5, "the rung that passed must be one the adaptive fit could not reach"
    print(f"PASS the ladder plans where the full ramp cannot (entry ramp {8.0 - start:.2f} m, "
          f"below the {2.5:.1f} m adaptive floor; no path at all without it)")


def test_the_ladder_is_a_last_resort_and_never_shortens_a_ramp_that_fits():
    # rung 0 is today's geometry: whatever the main pass accepts must come out unchanged, or the
    # ladder is buying feasibility with curvature it did not have to spend.
    def run(enable):
        n = planner([box(33.0, -0.35, oid=1)], cur_s=25.0)
        n.ramp_search_enable = enable
        w, _m = n.do_spline(gb_wpnts())
        return w

    a, b = run(False), run(True)
    assert a.wpnts and b.wpnts
    assert [round(x.d_m, 9) for x in a.wpnts] == [round(x.d_m, 9) for x in b.wpnts], \
        "an open corridor must plan identically with and without the ladder"
    print("PASS where the full ramp already fits, the ladder changes nothing")


if __name__ == "__main__":
    test_the_ramp_scan_reads_the_real_stations()
    test_the_ladder_finds_a_path_the_full_ramp_cannot_fit()
    test_the_ladder_is_a_last_resort_and_never_shortens_a_ramp_that_fits()
    print("ALL PASS")
