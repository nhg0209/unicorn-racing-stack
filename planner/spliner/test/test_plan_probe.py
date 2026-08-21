#!/usr/bin/env python3
"""A probe plan must change nothing, or the diagnostic becomes the bug.

`_log_plan`'s counterfactual line -- "would the other d(s) generator have found a path here?" -- is
answered by RE-RUNNING do_spline with the method swapped. That is the right way to ask (a second
copy of the gate stack would drift from the real one) and the dangerous way to ask: the second call
runs the same code that commits the path, records the last publication, and publishes the
feasibility verdict the state machine gates on. A probe that leaked any of those would make turning
the log ON change how the car drives, which is the one thing a diagnostic may never do.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/test/test_plan_probe.py
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


def box(s, d, half_d=0.15, half_s=0.15):
    return types.SimpleNamespace(
        id=int(s * 10), s_start=(s - half_s) % TRACK_LEN, s_end=(s + half_s) % TRACK_LEN,
        s_center=s, d_center=d, d_right=d - half_d, d_left=d + half_d, size=2 * half_d,
        vs=0.0, vd=0.0, is_static=True, is_visible=True, x_m=s, y_m=d, is_actually_a_gap=False)


def planner(obstacles, cur_s, cur_d=0.0, method="sample"):
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
    n.max_weave, n.knot_merge_s_m = 3, 0.4
    n.ramp_len, n.return_len, n.tail_m = 4.5, 4.5, 0.0
    n.ramp_len_min_m = 2.5
    n.ramp_search_enable = True
    n.ramp_search_entry_m = [3.15, 2.5, 2.0, 1.5, 1.0]
    n.ramp_search_exit_m = [4.5, 2.5, 1.5]
    n.ramp_search_max_ms = 1000.0
    n.apex_bulge, n.preramp_len_m = 0.05, 3.0
    n.kappa_add_max, n.kappa_abs_max = 5.0, 5.5
    n.a_lat_max, n.a_long_max, n.a_long_accel = 6.0, 4.0, 3.0
    n.w_d, n.w_k, n.w_c, n.w_obs, n.obs_sigma = 1.0, 0.1, 5.0, 2.0, 0.5
    n.shift_min, n.shift_buffer, n.hold_after = 1.0, 0.5, 0.5
    n._d_end_prev = 0.0
    n._last_pub = None
    n.use_grid_check, n.trust_grid_bounds = False, False
    n.grid_scan_max, n.grid_scan_step, n.bounds_warn_m = 3.0, 0.05, 0.5
    n.clear_gate_enable, n.clear_margin_m, n.clear_hyst_m = False, 0.10, 0.03
    n.clear_max_cur_d, n.clear_latch_ttl_s = 0.15, 10.0
    n._clear_latch, n._line_clear = {}, False
    n.commit_enable, n._committed = True, None
    n.squeeze_enable = False
    n.squeeze_steps, n.squeeze_safety_floor_m = 2, 0.05
    n.squeeze_wall_floor_m, n.squeeze_max_speed_mps = 0.08, 3.0
    n.relax_hold_s, n._relax_until = 2.0, 0.0
    n.obs_memory_sec, n._mem_cands_obs, n._mem_cands_time = 0.5, [], None
    n._promoted = {o.id for o in obstacles}
    n._near_zero_since, n._moving_since = {}, {}
    n.static_near_zero_mps, n.static_promote_sec = 0.15, 0.5
    n.static_demote_mps, n.static_demote_sec = 0.35, 0.3
    n.reframe_warn_m, n.obs_gather_extra_m = 0.05, 4.5
    n.commit_drop_on_new_obstacle = True
    n._emit_markers = False
    n.static_plan_method = method
    n.feasible, n.commits, n.published = [], [], []
    n._publish_feasible = lambda ok: n.feasible.append(ok)
    n._candidate_markers = lambda *a, **k: san.MarkerArray()
    n._store_commit = lambda *a, **k: n.commits.append(1)
    n._note_published = lambda s, d: n.published.append(1)
    return n


def _pts(res):
    return 0 if (res is None or res[0] is None) else len(res[0].wpnts)


def test_a_probe_plans_the_same_path():
    """Whatever else it does not do, it has to answer the same question."""
    for method in ("sample", "corridor_qp"):
        real = planner([box(20.0, 0.0)], cur_s=12.0, method=method)
        pro = planner([box(20.0, 0.0)], cur_s=12.0, method=method)
        a = real.do_spline(gb_wpnts())
        b = pro.do_spline(gb_wpnts(), probe=True)
        assert _pts(a) > 0, method
        assert _pts(b) == _pts(a), (method, _pts(a), _pts(b))
        da = np.array([w.d_m for w in a[0].wpnts])
        db = np.array([w.d_m for w in b[0].wpnts])
        assert np.array_equal(da, db), method
    print("PASS a probe returns the same path as the call it stands in for, both methods")


def test_a_probe_records_nothing():
    for method in ("sample", "corridor_qp"):
        n = planner([box(20.0, 0.0)], cur_s=12.0, method=method)
        res = n.do_spline(gb_wpnts(), probe=True)
        assert _pts(res) > 0, method
        assert n.feasible == [], (method, n.feasible)      # the SM's gate must not move
        assert n.commits == [], (method, n.commits)        # nothing to follow next cycle
        assert n.published == [], (method, n.published)    # no handover blend anchor
        assert n._d_end_prev == 0.0, (method, n._d_end_prev)   # no anti-chatter memory
        assert n._committed is None, method
    print("PASS a probe publishes no verdict, stores no commit, and remembers no choice")


def test_a_probe_never_reports_infeasible():
    """A refusing probe returns None, not the empty publication that carries feasible=False.

    This is the leak that would matter most: the state machine drops out of avoidance on that
    edge, so a probe that published it would make the diagnostic itself cause a TRAILING.
    """
    n = planner([box(20.0, 0.0, half_d=1.15)], cur_s=12.0)   # a box across the whole corridor
    res = n.do_spline(gb_wpnts(), probe=True)
    assert res is None, res
    assert n.feasible == [], n.feasible
    m = planner([box(20.0, 0.0, half_d=1.15)], cur_s=12.0)   # the same cell, for real
    m.do_spline(gb_wpnts())
    assert m.feasible == [False], m.feasible
    print("PASS a refusing probe returns None where the real call publishes feasible=False")


def test_the_counterfactual_swap_runs_both_generators():
    """What _log_plan does: swap the method, probe, put it back."""
    n = planner([box(20.0, 0.0)], cur_s=12.0, method="corridor_qp")
    real = n.do_spline(gb_wpnts())
    assert _pts(real) > 0
    before = n.static_plan_method
    n.static_plan_method = "sample"
    alt = n.do_spline(gb_wpnts(), probe=True)
    n.static_plan_method = before
    assert n.static_plan_method == "corridor_qp"
    assert _pts(alt) > 0
    d_qp = np.array([w.d_m for w in real[0].wpnts])
    d_sa = np.array([w.d_m for w in alt[0].wpnts])
    assert not np.array_equal(d_qp, d_sa), "the two generators produced the same array"
    print(f"PASS the swap yields two different shapes (max |d_qp - d_sample| = "
          f"{np.max(np.abs(d_qp - d_sa)):.3f} m) and restores the method")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"ALL PASS ({len(fns)} checks)")
