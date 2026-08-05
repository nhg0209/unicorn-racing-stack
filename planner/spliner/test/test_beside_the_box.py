#!/usr/bin/env python3
"""Does the static planner keep publishing a path while the car is BESIDE a box?

The reported symptom was "the local waypoints go empty next to an obstacle". The waypoint array
itself has no holes -- the planner was publishing an EMPTY one, because every candidate was
rejected for a reason that had nothing to do with the geometry being impassable:

  the knot station of a box the car is already level with was clipped to a floor of 0.3 m ahead,
  the 0.4 m merge window that follows then swallowed the NEXT box's knot,
  and obs_ok went on enforcing that second box's keep-out against a path never shaped for it.

With the boxes 0.5 m apart there was no recovery at all. Run (after sourcing the workspace):
  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/test/test_beside_the_box.py
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
    n.lookahead_min, n.lookahead_k = 15.0, 1.5
    n.n_d_samples, n.sample_gaps = 10, True
    n.max_weave = 3
    n.knot_merge_s_m = 0.4
    n.ramp_len, n.return_len, n.tail_m = 4.5, 4.5, 0.0
    n.ramp_len_min_m = 2.5
    n.apex_bulge, n.preramp_len_m = 0.05, 3.0
    n.kappa_add_max, n.kappa_abs_max = 5.0, 5.5
    n.a_lat_max, n.a_long_max, n.a_long_accel = 6.0, 4.0, 3.0
    n.w_d, n.w_k, n.w_c, n.w_obs, n.obs_sigma = 1.0, 0.1, 5.0, 2.0, 0.5
    n.shift_min, n.shift_buffer, n.hold_after = 1.0, 0.5, 0.5
    n._d_end_prev = 0.0
    n.use_grid_check, n.trust_grid_bounds = False, False
    n.grid_scan_max, n.grid_scan_step, n.bounds_warn_m = 3.0, 0.05, 0.5
    n.clear_gate_enable, n.clear_margin_m, n.clear_hyst_m = True, 0.10, 0.03
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
    n._emit_markers = False
    n.feasible = []
    n._publish_feasible = lambda ok: n.feasible.append(ok)
    n._candidate_markers = lambda *a, **k: san.MarkerArray()
    n._store_commit = lambda *a, **k: None
    return n


def run(obstacles, cur_s):
    """One planning cycle: returns (n_wpnts, log messages)."""
    n = planner(obstacles, cur_s)
    msgs = []

    class _Cap:
        def info(self, m, **k): msgs.append(str(m))
        def warn(self, m, **k): msgs.append(str(m))
        warning = warn
        def error(self, m, **k): msgs.append(str(m))
        def debug(self, *a, **k): pass
    n.get_logger = lambda: _Cap()
    w, _m = n.do_spline(gb_wpnts())
    return len(w.wpnts), msgs


def unknotted(msgs):
    """How many obstacles ahead were left WITHOUT a knot (0 if the diagnostic never fired)."""
    for m in msgs:
        if "got NO knot" in m:
            return int(m.split("] ")[1].split(" of ")[0])
    return 0


def test_a_blocking_box_keeps_its_knot_while_the_car_is_beside_the_first():
    # THE mechanism. The knot station of a box the car is level with was clipped to a floor of
    # 0.3 m ahead; the 0.4 m merge window that follows then swallowed the NEXT box's knot, while
    # obs_ok went on enforcing that second box's keep-out against a path never shaped for it.
    # Boxes 0.5 m apart, car drawing level with the first one.
    obs = [box(8.0, -0.35, oid=1), box(8.5, -0.35, oid=2)]
    for cur_s in (7.85, 7.90, 7.95):
        _n, msgs = run(obs, cur_s)
        assert unknotted(msgs) == 0, (
            f"at s={cur_s} a blocking box was left without a knot while obs_ok still enforces it; "
            f"that is the state in which every candidate is rejected")
    print("PASS both blocking boxes keep a knot while the car draws level with the first")


def test_a_box_behind_the_car_does_not_pull_a_knot_forward():
    # The gap must be SIGNED. `(s_center - cur_s) % L` for a centre a hand's breadth BEHIND the car
    # is L - 0.1, which the old clip turned into `lookahead` -- a knot fifteen metres ahead for a
    # box beside the mirror, which then merge-swallowed everything genuinely close.
    near = box(8.0, -0.35, oid=1)
    far = box(12.0, -0.35, oid=2)
    _n, msgs = run([near, far], cur_s=8.05)       # first box just behind the centre
    assert unknotted(msgs) == 0, "a box behind the car must not cost the box ahead its knot"
    print("PASS a box just behind the car does not pull a knot to the lookahead edge")


class _PhantomGrid:
    """Eroded-map stand-in that is FREE only in a band, and only over part of the track.

    The point is that the corridor differs by station: reading it at the wrong s gives a different
    answer, which is exactly the defect under test. Free within +-`w_near` of the raceline for
    s < `split`, and +-`w_far` beyond it.
    """

    def __init__(self, split=20.0, w_near=1.2, w_far=0.42, resolution=0.05):
        self.resolution = resolution
        self.origin = (-1.0, -3.0)
        h = int(6.0 / resolution)
        w = int((TRACK_LEN + 2.0) / resolution)
        y = (np.arange(h) + 0.5) * resolution + self.origin[1]
        x = (np.arange(w) + 0.5) * resolution + self.origin[0]
        near = (np.abs(y)[:, None] <= w_near) & (x[None, :] < split)
        far = (np.abs(y)[:, None] <= w_far) & (x[None, :] >= split)
        self.eroded_image = np.where(near | far, 255, 0).astype(np.uint8)


def test_the_second_knot_reads_its_own_corridor():
    # knots[i][0] is the distance from the CAR; _corridor_at handed it to _grid_corridor as an
    # ABSOLUTE station. So every knot after the first read its corridor at a phantom point
    # elsewhere on the track -- and the apex bulge was then clipped to that phantom corridor.
    #
    # Here the map is wide up to s = 20 and narrow after it. Two boxes sit in the WIDE part, and
    # the car is close enough that the s-local station of the second knot (~1.6 m) falls in the
    # wide part too -- so a correct read gives the wide corridor for both. The regression is
    # visible the other way round: with the car at s = 18.5, the second knot's s-local value is
    # ~1.6 while its true station is ~20.1, i.e. across the split.
    n = planner([box(20.1, -0.35, oid=1), box(21.1, -0.35, oid=2)], cur_s=18.5)
    n.trust_grid_bounds, n.use_grid_check = True, False
    n.grid_scan_max, n.grid_scan_step = 3.0, 0.05
    n.map_filter = _PhantomGrid(split=20.0)
    # the second knot's TRUE station is past the split (narrow); its s-local value is not
    true_cor = n._grid_corridor(21.1, wall_margin=n.wall_margin)
    phantom_cor = n._grid_corridor(21.1 - 18.5, wall_margin=n.wall_margin)
    assert true_cor is not None and phantom_cor is not None
    assert abs((true_cor[1] - true_cor[0]) - (phantom_cor[1] - phantom_cor[0])) > 0.5, \
        "the harness must make the two reads genuinely differ"
    w, _m = n.do_spline(gb_wpnts())
    assert w.wpnts, "the pair must still be plannable"
    # the peak must respect the corridor at the SECOND box's own station, not the phantom one
    peak = max(abs(x.d_m) for x in w.wpnts)
    assert peak <= max(abs(true_cor[0]), abs(true_cor[1])) + 1e-6, \
        f"peak |d| {peak:.3f} exceeds the corridor at the knot's real station {true_cor}"
    print(f"PASS the second knot reads the corridor at its own station "
          f"(true {true_cor[1]-true_cor[0]:.2f} m wide vs phantom "
          f"{phantom_cor[1]-phantom_cor[0]:.2f} m)")


def test_a_path_exists_on_the_approach():
    # Where the geometry is solvable, a path must come out. Closer than that the car is already
    # inside the box's INFLATED keep-out (obs_margin on top of the box), which no re-plan can
    # undo -- that range is covered by the committed path, not by re-planning.
    for label, obs in (("one box", [box(8.0, -0.35, oid=1)]),
                       ("0.5 m apart", [box(8.0, -0.35, oid=1), box(8.5, -0.35, oid=2)]),
                       ("1.0 m apart", [box(8.0, -0.35, oid=1), box(9.0, -0.35, oid=2)])):
        empty = [round(6.0 + 0.1 * k, 2) for k in range(0, 14)
                 if run(obs, 6.0 + 0.1 * k)[0] == 0]
        assert not empty, f"{label}: no path on the approach at s = {empty}"
    print("PASS a path exists at every approach station from 2.0 m out to 0.7 m")


def test_a_box_already_behind_the_car_keeps_its_keep_out():
    # Dropping the KNOT must not drop the OBSTACLE: a box the car is level with still contributes
    # its keep-out to obs_ok, so the planner may not route through it just because it is past.
    o = box(8.0, -0.35, oid=1)
    n = planner([o], cur_s=8.0)
    w, _m = n.do_spline(gb_wpnts())
    if w.wpnts:                                   # still shaped: check it does not cut the box
        span = 0.15 + 0.30                        # half the box in s + obs_margin
        for x in w.wpnts:
            ds = abs(((x.s_m - o.s_center + TRACK_LEN / 2) % TRACK_LEN) - TRACK_LEN / 2)
            if ds <= span:
                assert not (min(o.d_right, o.d_left) - 0.30 <= x.d_m
                            <= max(o.d_right, o.d_left) + 0.30), \
                    f"the path cut through a box it had merely passed (s={x.s_m:.2f})"
    print("PASS a box the car is level with keeps its keep-out even without a knot")


if __name__ == "__main__":
    test_a_blocking_box_keeps_its_knot_while_the_car_is_beside_the_first()
    test_a_box_behind_the_car_does_not_pull_a_knot_forward()
    test_the_second_knot_reads_its_own_corridor()
    test_a_path_exists_on_the_approach()
    test_a_box_already_behind_the_car_keeps_its_keep_out()
    print("ALL PASS")
