#!/usr/bin/env python3
"""Does a fresh plan hand over from the path the controller is already tracking?

A fresh plan replaces the current reference in one cycle with geometry anchored at the CAR. The
pre-ramp decays cur_d, but cur_d is not the previous reference -- it is the previous reference
minus a tracking lag. So every commit release (past-end, box moved, new obstacle, slice no longer
clear) handed the controller a step the size of that lag, at the point it steers for.

The blend matches the last published value AT THE CAR and fades it out over commit_reanchor_len_m
with the same smootherstep _reanchor_commit uses. Past the fade the new plan is untouched. And
because a blend is geometry the planner CHOSE, it answers to the same body/corridor gates as any
candidate -- if it fails one, the unblended plan is published instead.

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/test/test_handover_blend.py
"""
import types
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parents[1] / "spliner" / "static_avoidance_node.py"
san = types.ModuleType("san")
san.__dict__["__file__"] = str(MOD)
exec(compile(MOD.read_text(), str(MOD), "exec"), san.__dict__)
ObstacleSpliner = san.ObstacleSpliner

TRACK_LEN = 120.0
WPNT_DIST = 0.1
HALF_WIDTH = 1.2


class _Log:
    def __init__(self): self.msgs = []
    def info(self, m, **k): self.msgs.append(str(m))
    def warn(self, m, **k): self.msgs.append(str(m))
    warning = warn
    def error(self, m, **k): self.msgs.append(str(m))
    def debug(self, *a, **k): pass


class _Converter:
    @staticmethod
    def get_cartesian(s, d):
        return np.vstack([np.asarray(s, float) % TRACK_LEN, np.asarray(d, float)])

    @staticmethod
    def get_e_psi(x, y, yaw):
        return 0.0


class _WallGrid:
    """Free only within +-`half` of the raceline, so an offset blend leaves the drivable area."""

    def __init__(self, half, resolution=0.05):
        self.resolution = resolution
        self.origin = (-1.0, -3.0)
        h, w = int(6.0 / resolution), int((TRACK_LEN + 2.0) / resolution)
        y = (np.arange(h) + 0.5) * resolution + self.origin[1]
        self.eroded_image = np.where((np.abs(y) <= half)[:, None] * np.ones((1, w), bool),
                                     255, 0).astype(np.uint8)

    def is_point_inside(self, x, y):
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


def box(s, d, oid, half_d=0.15, half_s=0.15):
    return types.SimpleNamespace(
        id=oid, s_start=(s - half_s) % TRACK_LEN, s_end=(s + half_s) % TRACK_LEN, s_center=s,
        d_center=d, d_right=d - half_d, d_left=d + half_d, size=2 * half_d, vs=0.0, vd=0.0,
        is_static=True, is_visible=True, x_m=s, y_m=d, is_actually_a_gap=False)


def planner(obstacles, cur_s=8.0, cur_d=0.0, **over):
    n = ObstacleSpliner.__new__(ObstacleSpliner)
    n.name = "static_avoidance_planner"
    log = _Log()
    n.get_logger = lambda: log
    n.log = log
    n._t = 100.0
    stamp = san.OTWpntArray().header.stamp
    n.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=int(n._t * 1e9), to_msg=lambda: stamp))
    n.converter = _Converter()
    n.cur_s, n.cur_d, n.cur_vs = cur_s, cur_d, 3.0
    n.cur_x, n.cur_y, n.cur_yaw = cur_s, cur_d, 0.0
    n.gb_max_s, n.gb_max_idx = TRACK_LEN, int(TRACK_LEN / WPNT_DIST)
    n.obstacles = obstacles
    n.width_car, n.safety_margin, n.wall_margin = 0.30, 0.15, 0.10
    n.safety_margin_d = n.safety_margin
    n.lookahead_min, n.lookahead_k = 15.0, 1.5
    n.n_d_samples, n.sample_gaps, n.max_weave = 10, True, 3
    n.knot_merge_s_m = 0.4
    n.ramp_len, n.return_len, n.tail_m = 4.5, 4.5, 1.0
    n.ramp_len_min_m = 2.5
    n.ramp_search_enable = True
    n.ramp_search_entry_m = [3.15, 2.5, 2.0, 1.5, 1.0]
    n.ramp_search_exit_m = [4.5, 2.5, 1.5]
    n.ramp_search_max_ms = 1000.0
    n.obs_gather_extra_m = 4.5
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
    n.commit_enable, n._committed = False, None
    n.commit_dev_max, n.commit_reanchor_len_m, n.commit_reanchor_max_m = 0.6, 2.0, 1.0
    n.commit_obs_ds, n.commit_obs_dd = 0.75, 0.40
    n.commit_drop_on_new_obstacle = True
    n.squeeze_enable = False
    n.squeeze_steps, n.squeeze_safety_floor_m = 2, 0.05
    n.squeeze_wall_floor_m, n.squeeze_max_speed_mps = 0.08, 3.0
    n.relax_hold_s, n._relax_until = 2.0, 0.0
    n.obs_memory_sec, n._mem_cands_obs, n._mem_cands_time = 0.5, [], None
    n._promoted = {o.id for o in obstacles}
    n._near_zero_since, n._moving_since = {}, {}
    n.static_near_zero_mps, n.static_promote_sec = 0.15, 0.5
    n.static_demote_mps, n.static_demote_sec = 0.35, 0.3
    n.reframe_warn_m, n._emit_markers = 0.05, False
    n.body_filter = types.SimpleNamespace(eroded_image=None)
    n.map_filter = types.SimpleNamespace(eroded_image=None)
    n._publish_feasible = lambda ok: None
    n._candidate_markers = lambda *a, **k: san.MarkerArray()
    for k, v in over.items():
        setattr(n, k, v)
    return n


def seed_previous(n, offset, span=6.0):
    """Pretend the last published path carried `offset` of lateral offset around the car."""
    s = (n.cur_s + np.arange(0.0, span, WPNT_DIST)) % TRACK_LEN
    d = np.full(len(s), float(offset))
    n._last_pub = (s, d, n.get_clock().now().nanoseconds * 1e-9)


def d_at(w, cur_s, ahead):
    ds = [((x.s_m - cur_s) % TRACK_LEN) for x in w.wpnts]
    k = int(np.argmin([abs(v - ahead) for v in ds]))
    return float(w.wpnts[k].d_m)


def test_a_fresh_plan_starts_where_the_previous_one_was():
    # The step the controller feels is between successive published REFERENCES, not between the
    # car and the new plan -- the car is a tracking lag behind whatever it was last given.
    obs = [box(20.0, -0.35, 1)]
    plain = planner(obs)
    w0, _m = plain.do_spline(gb_wpnts())
    assert w0.wpnts
    base_at_car = d_at(w0, plain.cur_s, 0.0)

    n = planner(obs)
    seed_previous(n, offset=0.30)
    w, _m = n.do_spline(gb_wpnts())
    assert w.wpnts, "the blend must not cost the path"
    got_car = d_at(w, n.cur_s, 0.0)
    step = abs(got_car - 0.30)
    assert step < 0.02, (
        f"the published path starts at {got_car:+.3f} m against a previous reference of +0.300: "
        f"a {abs(got_car - 0.30):.3f} m step handed straight to the controller "
        f"(unblended start was {base_at_car:+.3f})")
    # ...and past the fade it is the plan that was chosen, untouched
    far = 2.0 + n.commit_reanchor_len_m
    assert abs(d_at(w, n.cur_s, far) - d_at(w0, plain.cur_s, far)) < 1e-6, \
        "past the fade the blend must change nothing"
    print(f"PASS a fresh plan starts at the previous reference ({got_car:+.3f} vs +0.300 m, "
          f"unblended {base_at_car:+.3f}) and is untouched past {n.commit_reanchor_len_m:.1f} m")


def test_the_blend_fades_over_the_reanchor_length():
    obs = [box(20.0, -0.35, 1)]
    n = planner(obs)
    seed_previous(n, offset=0.30)
    w, _m = n.do_spline(gb_wpnts())
    prof = [(a, d_at(w, n.cur_s, a)) for a in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)]
    ds = [d for _a, d in prof]
    assert ds[0] > ds[-1], "the fade must be monotone away from the previous offset"
    assert abs(ds[-1]) < 0.02, f"the offset must be gone by {n.commit_reanchor_len_m + 0.5} m: {prof}"
    print("PASS the handover fades out over commit_reanchor_len_m: "
          + ", ".join(f"{a:.1f}m {d:+.3f}" for a, d in prof))


def test_a_blend_that_leaves_the_track_is_refused():
    # G9. A blend is geometry the planner CHOSE, so it answers to the same gates as a candidate.
    # Free only within +-0.20 m here, so matching a +0.30 m previous reference puts the car body
    # outside; the unblended plan must be published instead.
    obs = [box(20.0, -0.35, 1)]
    plain = planner(obs)
    w0, _m = plain.do_spline(gb_wpnts())
    n = planner(obs, use_grid_check=True)
    n.map_filter = n.body_filter = _WallGrid(half=0.20)
    n.body_kernel_size = 7
    seed_previous(n, offset=0.30)
    w, _m = n.do_spline(gb_wpnts())
    assert w.wpnts, "refusing the blend must not cost the path"
    assert any("handover blend REFUSED" in m for m in n.log.msgs), \
        f"the blend should have been refused: {[m[:80] for m in n.log.msgs][:3]}"
    assert abs(d_at(w, n.cur_s, 0.0) - d_at(w0, plain.cur_s, 0.0)) < 1e-6, \
        "a refused blend must publish exactly the unblended plan"
    print("PASS a blend that leaves the drivable area is refused, and the unblended plan is published")


def test_a_stale_previous_publication_is_not_blended_onto():
    # Matching a reference the controller stopped holding half a second ago is inventing
    # continuity that does not exist.
    obs = [box(20.0, -0.35, 1)]
    n = planner(obs)
    seed_previous(n, offset=0.30)
    n._t += 1.0                                   # one second later
    plain = planner(obs)
    w0, _m = plain.do_spline(gb_wpnts())
    w, _m = n.do_spline(gb_wpnts())
    assert abs(d_at(w, n.cur_s, 0.0) - d_at(w0, plain.cur_s, 0.0)) < 1e-6, \
        "a stale publication must not be blended onto"
    print("PASS a publication older than the blend window is ignored")


if __name__ == "__main__":
    test_a_fresh_plan_starts_where_the_previous_one_was()
    test_the_blend_fades_over_the_reanchor_length()
    test_a_blend_that_leaves_the_track_is_refused()
    test_a_stale_previous_publication_is_not_blended_onto()
    print("ALL PASS")
