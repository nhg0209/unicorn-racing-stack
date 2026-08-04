#!/usr/bin/env python3
"""End-to-end test of the static planner on CLUSTERED obstacles.

Drives the real do_spline on a straight synthetic track with a real corridor, and asks the only
question that matters for the reported failure: does a passable cluster produce a path, or
feasible=False and a TRAILING deadlock?

Four mechanisms could each turn a passable cluster into "every candidate rejected", and they
compound:
  * max_weave capped the number of APEXES while obs_ok enforces every box in the lookahead, so an
    un-knotted third box was flown through by the return ramp;
  * apex_bulge pushed the peak away from d=0 instead of away from the OBSTACLE, moving it into the
    keep-out whenever a box sits off-centre in d;
  * obstacles the line already clears consumed knot slots the blocking ones needed;
  * the terminal-offset samples were measured against the nearest box alone, so the free lane
    between two boxes was sampled only by luck.

Run (after sourcing the workspace):
  python3 planner/spliner/test/test_obstacle_cluster.py
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
HALF_WIDTH = 1.2          # drivable half-width each side of the straight raceline


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def warning(self, *a, **k): pass


class _Clock:
    def now(self): return types.SimpleNamespace(
        nanoseconds=0, to_msg=lambda: san.OTWpntArray().header.stamp)


class _Converter:
    """Straight raceline along +x; +d is +y."""
    @staticmethod
    def get_cartesian(s, d):
        return np.vstack([np.asarray(s, float) % TRACK_LEN, np.asarray(d, float)])

    @staticmethod
    def get_e_psi(x, y, yaw):
        return 0.0


def gb_wpnts(half_width=HALF_WIDTH):
    return [types.SimpleNamespace(x_m=i * WPNT_DIST, y_m=0.0, s_m=i * WPNT_DIST,
                                  d_left=half_width, d_right=half_width,
                                  vx_mps=4.0, kappa_radpm=0.0)
            for i in range(int(TRACK_LEN / WPNT_DIST))]


def box(s, d, half_d=0.15, half_s=0.35, oid=None):
    return types.SimpleNamespace(
        id=oid if oid is not None else int(s * 10), s_start=s - half_s, s_end=s + half_s,
        s_center=s, d_center=d, d_right=d - half_d, d_left=d + half_d,
        size=2 * half_d, vs=0.0, vd=0.0, is_static=True, is_visible=True,
        x_m=s, y_m=d, is_actually_a_gap=False)


def planner(obstacles, cur_d=0.0, cur_s=0.0):
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
    n.ramp_len, n.return_len, n.tail_m = 4.5, 4.5, 0.0
    n.ramp_len_min_m = 2.5       # adaptive-ramp floor; == ramp_len would disable the shortening
    n.apex_bulge, n.preramp_len_m = 0.10, 3.0
    n.kappa_add_max, n.kappa_abs_max = 5.0, 5.5
    n.a_lat_max, n.a_long_max, n.a_long_accel = 6.0, 4.0, 3.0
    n.w_d, n.w_k, n.w_c, n.w_obs, n.obs_sigma = 1.0, 0.1, 5.0, 2.0, 0.5
    n._d_end_prev = 0.0
    n.use_grid_check, n.trust_grid_bounds = False, False
    n.bounds_warn_m = 0.5
    n.clear_gate_enable, n.clear_margin_m, n.clear_hyst_m = True, 0.10, 0.03
    n.clear_max_cur_d, n.clear_latch_ttl_s = 0.15, 10.0
    n._clear_latch, n._line_clear = {}, False
    n.commit_enable, n._committed = False, None
    n.squeeze_enable = False
    n.squeeze_steps, n.squeeze_safety_floor_m = 2, 0.05
    n.squeeze_wall_floor_m, n.squeeze_max_speed_mps = 0.05, 3.0
    n.relax_hold_s, n._relax_until = 2.0, 0.0
    n.obs_memory_sec, n._mem_cands_obs, n._mem_cands_time = 0.5, [], None
    n._promoted = {o.id for o in obstacles}
    n._near_zero_since, n._moving_since = {}, {}
    n.static_near_zero_mps, n.static_promote_sec = 0.15, 0.5
    n.static_demote_mps, n.static_demote_sec = 0.35, 0.3
    n._emit_markers = False
    n.feasible = []
    n._publish_feasible = lambda ok: n.feasible.append(ok)
    n._candidate_markers = lambda *a, **k: san.MarkerArray()
    n._store_commit = lambda *a, **k: None
    return n


def plan(n, half_width=HALF_WIDTH):
    wpnts, _m = n.do_spline(gb_wpnts(half_width))
    return wpnts


def peak_d(wpnts):
    return max((abs(w.d_m) for w in wpnts.wpnts), default=0.0)


def clears_all(wpnts, obstacles, obs_margin=0.30):
    """Does the published path stay out of every inflated keep-out box?"""
    for o in obstacles:
        for w in wpnts.wpnts:
            ds = abs(((w.s_m - o.s_center + TRACK_LEN / 2) % TRACK_LEN) - TRACK_LEN / 2)
            if ds <= (o.s_end - o.s_start) / 2 + obs_margin:
                if min(o.d_right, o.d_left) - obs_margin <= w.d_m <= max(o.d_right, o.d_left) + obs_margin:
                    return False
    return True


def test_two_boxes_same_side_2m_apart():
    # (a) Both boxes on the SAME side, 2 m apart -- the path should hold one offset across both
    # rather than weaving back to the raceline between them.
    obs = [box(8.0, -0.35, oid=1), box(10.0, -0.35, oid=2)]
    n = planner(obs)
    w = plan(n)
    assert w.wpnts, "a passable same-side pair must produce a path"
    assert clears_all(w, obs), "the published path must clear both boxes"
    # d over the stretch spanning both boxes must stay on one side and not return to 0 between them
    between = [x.d_m for x in w.wpnts if 8.0 <= x.s_m <= 10.0]
    assert between and min(between) > 0.05, \
        f"the offset must be HELD across the pair, got min |d| = {min(between):.3f}"
    assert all(v > 0 for v in between), "and stay on one side"
    print(f"PASS same-side pair 2 m apart: offset held (min {min(between):.2f}, peak {peak_d(w):.2f})")


def test_three_boxes_2_to_4m_apart_is_feasible():
    # (b) THE deadlock case: three boxes inside the lookahead. With max_weave 2 the third got no
    # knot, contributed its keep-out to obs_ok anyway, and killed every candidate.
    obs = [box(8.0, -0.30, oid=1), box(10.0, 0.30, oid=2), box(14.0, -0.30, oid=3)]
    n = planner(obs)
    w = plan(n)
    assert w.wpnts, "three passable boxes must produce a path, not feasible=False"
    assert n.feasible and n.feasible[-1] is True
    assert clears_all(w, obs), "the published path must clear all three boxes"
    print(f"PASS three boxes 2-4 m apart: feasible, peak |d| = {peak_d(w):.2f}")


def test_offcentre_box_bulges_away_from_it():
    # (c) A box off-centre in d, as everything reads on a swapped re-opt line. The apex must end up
    # on the far side of the box from the raceline, and the bulge must WIDEN that gap rather than
    # narrow it. d_center = +0.25 is off-centre and still blocks (its keep-out spans d = 0).
    o = box(9.0, 0.25, oid=1)
    n = planner([o])
    w = plan(n)
    assert w.wpnts, "an off-centre blocking box must be passable"
    assert clears_all(w, [o])
    apex = max(w.wpnts, key=lambda x: abs(x.d_m - o.d_center))
    gap = abs(apex.d_m - o.d_center)
    assert gap >= 0.30 + 0.10 - 1e-6, \
        f"apex must clear the box by keep-out + bulge, got {gap:.3f}"
    print(f"PASS off-centre box (d_c={o.d_center:+.2f}): apex at {apex.d_m:+.2f}, gap {gap:.2f}")


def test_bulge_rule_is_relative_to_the_obstacle():
    # The rule itself, including the case the end-to-end path can no longer reach: a box far enough
    # off-centre that a pass offset lands on the same side of d=0 as the box. The old rule
    # (further from d=0) moved the apex TOWARD such a box. That case is now skipped as
    # already-cleared before it ever gets a knot, so this pins the rule directly -- it must not
    # silently regress if the skip is ever loosened.
    n = planner([])
    n.apex_bulge = 0.10
    for da, d_c in ((0.55, 0.0), (-0.55, 0.0), (0.95, 0.50), (-0.05, -0.50), (0.05, 0.50)):
        b = n._bulge_away_from(da, types.SimpleNamespace(d_center=d_c))
        assert abs((da + b) - d_c) > abs(da - d_c), \
            f"bulge moved the apex toward the box: da={da:+.2f} d_c={d_c:+.2f} -> {da + b:+.2f}"
    # the old rule fails exactly the last case
    da, d_c = 0.05, 0.50
    old_peak = da + np.sign(da) * 0.10
    assert abs(old_peak - d_c) < abs(da - d_c), "harness must reproduce the old rule's error"
    # degenerate: apex exactly on the box centreline keeps the old sign
    assert n._bulge_away_from(0.4, types.SimpleNamespace(d_center=0.4)) > 0
    print("PASS the bulge is measured from the obstacle, not from d = 0")


class BodyGrid:
    """A GridFilter-shaped body image for this synthetic track: free only where |y| <= limit.

    The harness converter maps (s, d) -> (x = s, y = d), so this is "the car body fits only within
    +-limit of the raceline" — the same question _path_body_unsafe asks of the real eroded map.
    """

    def __init__(self, limit, res=0.05):
        self.resolution = res
        self.origin = (-1.0, -3.0)
        w = int((TRACK_LEN + 2.0) / res)
        h = int(6.0 / res)
        y = (np.arange(h) + 0.5) * res + self.origin[1]
        self.eroded_image = np.where(np.abs(y)[:, None] <= limit, 255, 0).astype(np.uint8) \
            * np.ones((1, w), dtype=np.uint8)


def test_squeeze_cannot_relax_the_body_floor():
    # The squeeze pass gives up safety_margin and wall_margin -- clearance ON TOP of the car's own
    # width. The body floor IS that width, so relaxing it would not buy a tighter pass, it would
    # buy a collision. This pins that the squeeze retry is checked against the same body image as
    # the full-margin pass: the section below is passable ONLY at reduced margins, and stops being
    # passable at all once the body floor says the car does not fit there.
    o = box(9.0, -0.20, oid=1)                   # keep-out [-0.65, +0.25] at the design margin
    def narrow_planner():
        n = planner([o])
        n.wall_margin, n.squeeze_enable, n.cur_vs = 0.10, True, 1.0
        return n
    n = narrow_planner()
    w = plan(n, 0.45)                            # too tight at the design margins
    assert w.wpnts and w.ot_line == "squeeze", \
        "the harness must reproduce a section that only the squeeze pass can pass"
    peak = max(abs(x.d_m) for x in w.wpnts)
    n2 = narrow_planner()
    n2.body_filter = BodyGrid(limit=peak - 0.02)  # the squeezed line does not fit the body floor
    w2 = plan(n2, 0.45)
    assert not w2.wpnts and n2.feasible[-1] is False, \
        "the squeeze pass must NOT be able to publish a path the car body does not fit on"
    # ...and the floor is what did it: widen the floor past the squeezed line and it returns
    n3 = narrow_planner()
    n3.body_filter = BodyGrid(limit=peak + 0.10)
    w3 = plan(n3, 0.45)
    assert w3.wpnts and w3.ot_line == "squeeze", "a body floor with room must not block the pass"
    print(f"PASS the squeeze pass cannot relax the body floor (squeezed peak {peak:.2f} m)")


def test_apex_bulge_is_clipped_to_the_corridor():
    # apex_bulge pushes the peak FURTHER from the box than the pass offset needs -- a deliberate
    # extra swing. On a wide track that is free; on a narrow one it put the peak at the sampling
    # limit + apex_bulge, i.e. past the corridor with ZERO terminal reserve (and NEGATIVE under
    # squeeze, which lowers the limit the bulge is measured from). The corridor test then killed
    # the candidate outright, so the bulge cost the pass exactly where the pass was scarcest.
    # Clipped, the candidate survives at the corridor edge and the box-clearance test still has
    # the final say.
    half_car, wall = 0.15, 0.05
    o = box(9.0, -0.20, oid=1)                   # keep-out [-0.65, +0.25] at obs_margin 0.30
    # ROOMY: the clip must be inert. The peak is the pass offset plus the full bulge.
    n = planner([o]); n.wall_margin = wall
    w = plan(n, 1.20)
    peak_wide = max(abs(x.d_m) for x in w.wpnts)
    assert w.wpnts and clears_all(w, [o])
    assert peak_wide > 0.25 + 1e-3, "on a wide track the bulge must still be applied in full"
    # NARROW: 0.48 m half-width leaves a car-centre corridor of +-0.28 m, and the unclipped peak
    # (pass offset 0.25 + bulge 0.10 = 0.35) is past even the waypoint-bounds limit of 0.33 --
    # every candidate was rejected and the planner conceded TRAILING on a box it can pass.
    n2 = planner([o]); n2.wall_margin = wall
    w2 = plan(n2, 0.48)
    limit = 0.48 - half_car - wall               # 0.28 m, the corridor limit for a car CENTRE
    unclipped = 0.25 + n2.apex_bulge
    assert unclipped > 0.48 - half_car, "the harness must reproduce the over-bulge"
    assert w2.wpnts, "the clipped candidate must survive where the unclipped one was rejected"
    assert n2.feasible and n2.feasible[-1] is True
    peak = max(abs(x.d_m) for x in w2.wpnts)
    assert peak <= limit + 1e-6, f"peak |d| {peak:.3f} exceeds the corridor limit {limit:.3f}"
    assert peak > limit - 0.02, f"the clip must bite here, peak {peak:.3f} vs limit {limit:.3f}"
    assert clears_all(w2, [o]), "…and the clipped path must still clear the box"
    print(f"PASS bulge clip: wide peak {peak_wide:.2f} (bulge intact), narrow peak {peak:.2f} "
          f"= corridor limit {limit:.2f} (unclipped {unclipped:.2f} was rejected)")


def test_already_cleared_box_does_not_spend_a_knot():
    # A box the line already clears must not consume one of the three knot slots, or a genuinely
    # blocking fourth box goes un-knotted and obs_ok rejects everything.
    cleared = box(6.0, 0.95, oid=1)            # keep-out [0.50, 1.40]: d=0 is outside it
    blocking = [box(9.0, -0.25, oid=2), box(11.5, 0.25, oid=3), box(14.0, -0.25, oid=4)]
    n = planner([cleared] + blocking)
    w = plan(n)
    assert w.wpnts, "the already-cleared box must not cost the blocking ones their knots"
    assert clears_all(w, [cleared] + blocking)
    print(f"PASS an already-cleared box leaves the knot slots to the blocking boxes")


if __name__ == "__main__":
    test_two_boxes_same_side_2m_apart()
    test_three_boxes_2_to_4m_apart_is_feasible()
    test_offcentre_box_bulges_away_from_it()
    test_bulge_rule_is_relative_to_the_obstacle()
    test_apex_bulge_is_clipped_to_the_corridor()
    test_squeeze_cannot_relax_the_body_floor()
    test_already_cleared_box_does_not_spend_a_knot()
    print("ALL PASS")
