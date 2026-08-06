#!/usr/bin/env python3
"""Why the planner does not react to the SECOND of two boxes.

When box 1 is planned, box 2 is still outside the lookahead: it gets neither a knot (it is not in
the knot list) nor a keep-out (obs_ok only sees the lookahead set). The resulting box-1-only path
is then COMMITTED -- frozen and republished verbatim -- and "box 2 came into view" was not among
the release conditions. So the planner published feasible=True at a box it had never planned
around, flipped when the car was right on top of it, and the state machine dropped the overtake
static_feasible_lost_sec later, which is later still.

Two independent halves, both needed:
  the gather horizon must reach as far as the PATH does, so obs_ok cannot certify a path that
    runs its exit ramp through a box (a path that fails its own re-check the moment that box
    enters the lookahead),
  and the commit must be released when a qualifying box it was never planned around enters the
    lookahead, so that box gets an apex while there is still room to make one.

Run (after sourcing the workspace):
  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/test/test_commit_horizon.py
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
LOOKAHEAD = 15.0


class _Log:
    def __init__(self): self.msgs = []
    def info(self, m, **k): self.msgs.append(str(m))
    def warn(self, m, **k): self.msgs.append(str(m))
    warning = warn
    def error(self, m, **k): self.msgs.append(str(m))
    def debug(self, *a, **k): pass


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


def box(s, d, oid, half_d=0.15, half_s=0.15):
    return types.SimpleNamespace(
        id=oid, s_start=(s - half_s) % TRACK_LEN, s_end=(s + half_s) % TRACK_LEN, s_center=s,
        d_center=d, d_right=d - half_d, d_left=d + half_d, size=2 * half_d, vs=0.0, vd=0.0,
        is_static=True, is_visible=True, x_m=s, y_m=d, is_actually_a_gap=False)


class Planner:
    """One node driven over a sequence of car stations, with the real commit machinery."""

    def __init__(self, obstacles, **over):
        n = ObstacleSpliner.__new__(ObstacleSpliner)
        self.n = n
        self.log = _Log()
        n.name = "static_avoidance_planner"
        n.get_logger = lambda: self.log
        stamp = san.OTWpntArray().header.stamp
        n.get_clock = lambda: types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=int(self.t * 1e9), to_msg=lambda: stamp))
        self.t = 0.0
        n.converter = _Converter()
        n.cur_d, n.cur_vs = 0.0, 3.0
        n.cur_yaw = 0.0
        n.gb_max_s, n.gb_max_idx = TRACK_LEN, int(TRACK_LEN / WPNT_DIST)
        n.obstacles = obstacles
        n.width_car, n.safety_margin, n.wall_margin = 0.30, 0.15, 0.10
        n.safety_margin_d = n.safety_margin
        n.lookahead_min, n.lookahead_k = LOOKAHEAD, 1.5
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
        n.use_grid_check, n.trust_grid_bounds = False, False
        n.grid_scan_max, n.grid_scan_step, n.bounds_warn_m = 3.0, 0.05, 0.5
        n.clear_gate_enable, n.clear_margin_m, n.clear_hyst_m = False, 0.10, 0.03
        n.clear_max_cur_d, n.clear_latch_ttl_s = 0.15, 10.0
        n._clear_latch, n._line_clear = {}, False
        n.commit_enable, n._committed = True, None
        n.commit_drop_on_new_obstacle = True
        n.commit_dev_max, n.commit_reanchor_len_m, n.commit_reanchor_max_m = 0.6, 2.0, 1.0
        n.commit_obs_ds, n.commit_obs_dd = 0.75, 0.40
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
        self.feasible = []
        n._publish_feasible = lambda ok: self.feasible.append(bool(ok))
        n._candidate_markers = lambda *a, **k: san.MarkerArray()
        for k, v in over.items():
            setattr(n, k, v)

    def step(self, cur_s, dt=0.05):
        self.t += dt
        n = self.n
        n.cur_s, n.cur_x, n.cur_y = cur_s, cur_s, n.cur_d
        w, _m = n.do_spline(gb_wpnts())
        return w

    def committed_ids(self):
        c = self.n._committed
        return set() if c is None else {oid for (oid, _s, _d) in c['obs']}


def knot_ids(p):
    """Obstacle ids the LAST plan actually shaped the path around (i.e. got a knot)."""
    c = p.n._committed
    return set() if c is None else {oid for (oid, _s, _d) in c['obs']}


def test_a_box_that_comes_into_reach_releases_the_commit():
    # A box from BEYOND the gather horizon -- the one case the gather cannot cover, because there
    # is nothing to gather yet. Box 2 sits 12 m behind box 1, outside the 19.5 m gather horizon
    # when the plan around box 1 is frozen, and comes into the lookahead while that plan is still
    # being republished. The commit must be released by THIS condition rather than by the safety
    # re-check, which fires later, once the frozen geometry is already violated, and publishes
    # feasible=False -- the state machine's cue to abandon the overtake.
    S2 = 32.0
    for drop in (True, False):
        p = Planner([box(20.0, -0.35, 1), box(S2, -0.35, 2)], commit_drop_on_new_obstacle=drop)
        p.step(8.0)
        assert p.committed_ids() == {1}, p.committed_ids()
        fired = first_false = planned = None
        for k in range(1, 100):                      # 0.1 m steps, past box 2 entering
            p.log.msgs.clear()
            p.feasible.clear()
            cur_s = 8.0 + 0.1 * k
            p.step(cur_s)
            if first_false is None and False in p.feasible:
                first_false = cur_s
            if planned is None and 2 in p.committed_ids():
                planned = cur_s
            if fired is None and any("came into the" in m and "never" in m for m in p.log.msgs):
                fired = cur_s
        if drop:
            assert fired is not None, "box 2 came into reach and the commit was not released"
            gap = S2 - fired
            assert gap > 10.0, f"box 2 was only reacted to at {gap:.2f} m"
            assert planned is not None and planned <= fired + 1e-9, \
                "the release must be followed by a plan that includes box 2, in the same cycle"
            assert first_false is None, (
                f"feasible=False at s={first_false:.2f} (box 2 {S2 - first_false:.2f} m ahead): "
                f"the release is supposed to make the re-plan happen BEFORE anything is violated")
            print(f"PASS a box coming into reach releases the commit and is planned around "
                  f"in the same cycle ({gap:.2f} m ahead, no feasible=False at all)")
        else:
            assert fired is None, "the harness must reproduce the failure: no release without it"
            assert planned is None, (
                "...and without it box 2 gets no apex at all while the frozen plan is being "
                "republished -- it waits for the commit to expire, or for the safety re-check to "
                "fail, whichever comes first")


def test_the_release_does_not_fire_on_the_boxes_it_was_planned_around():
    # The commit exists to stop the planner re-solving from a moving car every cycle. A release
    # that fired on the boxes it was planned around would be a re-plan every cycle.
    p = Planner([box(20.0, -0.35, 1), box(25.0, -0.35, 2)])
    p.step(8.0)
    assert {1, 2} <= p.committed_ids(), p.committed_ids()
    n_before = len(p.log.msgs)
    for k in range(1, 40):
        p.step(8.0 + 0.1 * k)
    released = [m for m in p.log.msgs[n_before:] if "came into the" in m]
    assert not released, f"the commit was released without a new box: {released[:2]}"
    print("PASS the release does not fire on the obstacles the commit was planned around")


def test_the_gather_horizon_reaches_as_far_as_the_path():
    # obs_ok used to be filtered by the LOOKAHEAD while the path itself runs return_len + tail_m
    # past the last knot, so a candidate could be certified while its own exit ramp went through a
    # box just outside it. That path is not merely optimistic -- it is one the planner itself
    # rejects: _commit_slice_clear re-checks the frozen slice against EVERY live obstacle, so it
    # failed on the very next cycle, published feasible=False, was re-planned identically, and
    # flapped at 20 Hz.
    for spacing in (3.5, 5.0):
        p = Planner([box(20.0, -0.35, 1), box(20.0 + spacing, -0.35, 2)])
        n = p.n
        n.cur_s, n.cur_x, n.cur_y = 8.0, 8.0, 0.0
        lookahead = max(n.lookahead_min, n.lookahead_k * n.cur_vs)
        s2 = 20.0 + spacing
        assert s2 - 8.0 > lookahead, "box 2 must start OUTSIDE the lookahead to mean anything"
        assert s2 - 8.0 <= lookahead + n.obs_gather_extra_m, "...and inside the gather horizon"
        w = p.step(8.0)
        assert w.wpnts, f"spacing {spacing}: a path must still come out"
        obs_margin_d = n.width_car / 2.0 + n.safety_margin_d
        lo, hi = -0.35 - 0.15 - obs_margin_d, -0.35 + 0.15 + obs_margin_d
        cut = [x for x in w.wpnts
               if abs(((x.s_m - s2 + TRACK_LEN / 2) % TRACK_LEN) - TRACK_LEN / 2) <= 0.15 + 0.29
               and lo <= x.d_m <= hi]
        assert not cut, (f"spacing {spacing}: the published path runs through box 2 at "
                         f"{[round(x.s_m, 2) for x in cut][:4]} -- obs_ok never saw it")
    print("PASS the gather horizon covers the exit ramp, so obs_ok sees the box the path reaches")


def test_the_commit_records_every_box_the_release_will_ask_about():
    # THE bug behind "the planner re-plans every single cycle". The release condition compares the
    # commit's own obs list against everything inside the GATHER horizon, so the commit has to
    # record that same set. It recorded obs_enforce instead -- the boxes the path was shaped
    # around -- and a box that found no free max_weave slot is in the first set and not the
    # second. It therefore read as "never planned around" on the very next cycle, every cycle,
    # for as long as it stayed in the horizon: the commit was released as fast as it was made.
    # Run log, four boxes: 64 fresh plans in two laps, with runs of 10 and 9 consecutive cycles.
    near = [box(20.0, -0.35, 1), box(22.0, -0.35, 2), box(24.0, -0.35, 3)]
    far = box(26.0, -0.35, 4)                       # inside the gather horizon, no slot left
    p = Planner(near + [far], max_weave=3)
    p.step(9.0)
    n = p.n
    gather = (max(n.lookahead_min, n.lookahead_k * n.cur_vs) + n.obs_gather_extra_m)
    reachable = {int(o.id) for _g, o in n._gather_obstacles_ahead(n.obstacles, gather)}
    assert 4 in reachable, "the fixture must put a box in the gather band with no slot for it"
    stored = p.committed_ids()
    missing = sorted(reachable - stored)
    assert not missing, (
        f"the commit does not record {missing}, which the release will then see as a box it was "
        f"never planned around -- on this cycle and every cycle after it")
    # ...and that is exactly what the release asks, so it must not fire on a fresh commit
    p.log.msgs.clear()
    p.step(9.05)
    assert not any("came into the" in m for m in p.log.msgs), \
        f"the commit was released one cycle after it was made: {p.log.msgs[:2]}"
    print(f"PASS a fresh commit records every box inside the gather horizon ({sorted(stored)})")


def test_a_box_past_the_lookahead_takes_only_a_LEFTOVER_weave_slot():
    # Enforcing a box without shaping around it rejects every candidate, so a box in the extended
    # band must be able to take a knot -- but never one a box in the driving horizon needed. The
    # list is sorted by gap, so the near boxes are considered first; with max_weave full, the
    # far one gets nothing.
    near = [box(20.0, -0.35, 1), box(22.0, -0.35, 2), box(24.0, -0.35, 3)]
    far = box(26.0, -0.35, 4)                       # inside the gather horizon, outside max_weave
    p = Planner(near + [far], max_weave=3)
    p.step(9.0)
    ids = p.committed_ids()
    assert {1, 2, 3} <= ids, f"the boxes in the lookahead must be planned around first: {ids}"
    p2 = Planner([box(20.0, -0.35, 1), far], max_weave=3)
    p2.step(9.0)
    assert 4 in p2.committed_ids(), "with slots to spare the far box must be planned around"
    print("PASS a box past the lookahead takes a weave slot only when one is left over")


def test_a_box_is_planned_around_as_soon_as_it_is_reachable():
    # The release and the gather now use the SAME horizon, so "a box the plan never knew about"
    # and "a box the next plan would shape around" are the same set. The moment box 2 is inside
    # the gather horizon AND the planner is publishing, it must be in the plan -- not one lookahead
    # later, when the frozen path finally expires.
    for spacing in (1.5, 2.5, 3.5, 5.0, 7.0, 9.0):
        p = Planner([box(20.0, -0.35, 1), box(20.0 + spacing, -0.35, 2)])
        s2 = 20.0 + spacing
        gather = (max(p.n.lookahead_min, p.n.lookahead_k * p.n.cur_vs) + p.n.obs_gather_extra_m)
        active = got = None
        for k in range(0, 200):
            cur_s = 2.0 + 0.1 * k
            w = p.step(cur_s)
            if active is None and w.wpnts and (s2 - cur_s) <= gather:
                active = cur_s
            if got is None and 2 in p.committed_ids():
                got = cur_s
            if got is not None:
                break
        assert active is not None and got is not None, f"spacing {spacing}: box 2 never planned for"
        assert got - active <= 1.0, (
            f"spacing {spacing}: box 2 was inside the gather horizon and the planner was active "
            f"for {got - active:.2f} m before it was planned around")
    print("PASS a reachable box is planned around in the first cycle it is reachable "
          "(spacings 1.5-9.0 m, 0.00 m of travel)")


def test_no_infeasible_cycle_while_the_second_box_is_still_ahead():
    # The symptom, end to end: feasible must not flip to False while there is still room to plan.
    for spacing in (1.5, 2.5, 3.5, 5.0, 7.0, 9.0):
        p = Planner([box(20.0, -0.35, 1), box(20.0 + spacing, -0.35, 2)])
        bad = []
        for k in range(0, 110):                       # 8.0 -> 19.0 m, 0.1 m steps
            cur_s = 8.0 + 0.1 * k
            p.feasible.clear()
            p.step(cur_s)
            gap2 = (20.0 + spacing) - cur_s
            if gap2 > 1.0 and False in p.feasible:
                bad.append(round(gap2, 2))
        assert not bad, (f"spacing {spacing} m: feasible=False published with box 2 still "
                         f"{max(bad):.2f} m ahead (all: {bad[:6]})")
    print("PASS no infeasible cycle at spacings 1.5-9.0 m while box 2 is more than 1 m ahead")


if __name__ == "__main__":
    test_a_box_that_comes_into_reach_releases_the_commit()
    test_the_release_does_not_fire_on_the_boxes_it_was_planned_around()
    test_the_gather_horizon_reaches_as_far_as_the_path()
    test_the_commit_records_every_box_the_release_will_ask_about()
    test_a_box_past_the_lookahead_takes_only_a_LEFTOVER_weave_slot()
    test_a_box_is_planned_around_as_soon_as_it_is_reachable()
    test_no_infeasible_cycle_while_the_second_box_is_still_ahead()
    print("ALL PASS")
