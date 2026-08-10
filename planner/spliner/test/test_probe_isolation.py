#!/usr/bin/env python3
"""A probe plan is one NOBODY ACTS ON. Prove it cannot drop the live commit.

do_spline(probe=True) exists so _log_plan can answer "would the other d(s) generator have found a
path here?" with the real pipeline instead of a second copy of the gate stack. Its docstring
promises "no retries, no commit, no feasibility verdict, no marker and no memory of the choice",
and the write paths were guarded for _d_end_prev, _store_commit, _note_published and
_publish_feasible -- but not for the `if not knots:` branch, whose `self._committed = None` sat
ahead of the `return None if retry else _empty()` that ends a probe.

The probe is the ONLY caller that can reach that line holding a commit. It runs with retry=True,
which does two things at once: it skips the committed-path reuse (so the probe plans rather than
returning early) and it skips the clear gate (so the branch that gate normally pre-empts becomes
reachable). The squeeze and ramp-ladder retries also set retry=True, but they only run after the
top-level pass has already been through this point.

WHY THIS NEVER FIRED, which is the part worth recording. Measured over six scenarios x 292 driving
cycles (two d(s) generators x boxes at d = 0.00 / +0.30 / -0.30, ifac, the real commit machinery):
zero occurrences. Reaching `not knots` needs every box ahead to be cleared by the followed line at
obs_margin_d = width_car/2 + safety_margin_d = 0.30 m, and anything cleared at 0.30 is also cleared
at the clear gate's own threshold -- width_car/2 + clear_margin_m (+ clear_hyst_m) = 0.25-0.28 m --
so on the real pass the gate idles and drops the commit itself one branch earlier. The two
thresholds would have to cross for this to become reachable in a real cycle. That is a coincidence
of two independently tuned margins, not a guarantee, and it is why the fix is a guard rather than a
comment saying it cannot happen.

Run (after sourcing the workspace):
  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/test/test_probe_isolation.py
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
HALF_WIDTH = 1.6
LOOKAHEAD = 15.0
# A commit object the planner never reads on this path: under probe the reuse branch is skipped,
# and with commit_enable False it is not entered at all. Identity is all the assertions need.
SENTINEL = {"obs": [(1, 20.0, 0.6)], "marker": "do not touch"}


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


def planner(obstacles, **over):
    n = ObstacleSpliner.__new__(ObstacleSpliner)
    log = _Log()
    n.name = "static_avoidance_planner"
    n.get_logger = lambda: log
    stamp = san.OTWpntArray().header.stamp
    n.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=0, to_msg=lambda: stamp))
    n.converter = _Converter()
    n.cur_s, n.cur_d, n.cur_vs = 8.0, 0.0, 3.0
    n.cur_x, n.cur_y, n.cur_yaw = 8.0, 0.0, 0.0
    n.gb_max_s, n.gb_max_idx = TRACK_LEN, int(TRACK_LEN / WPNT_DIST)
    n.obstacles = obstacles
    n.width_car, n.safety_margin, n.wall_margin = 0.30, 0.15, 0.10
    n.safety_margin_d = n.safety_margin
    n.lookahead_min, n.lookahead_k = LOOKAHEAD, 1.5
    n.n_d_samples, n.sample_gaps, n.max_weave = 10, True, 3
    n.knot_merge_s_m = 0.4
    n.ramp_len, n.return_len, n.tail_m = 4.5, 4.5, 0.0
    n.ramp_len_min_m = 2.5
    n.ramp_search_enable = False
    n.ramp_search_entry_m, n.ramp_search_exit_m = [], []
    n.ramp_search_max_ms = 1000.0
    n.obs_gather_extra_m = 4.5
    n.apex_bulge, n.preramp_len_m = 0.05, 3.0
    n.kappa_add_max, n.kappa_abs_max = 5.0, 5.5
    n.a_lat_max, n.a_long_max, n.a_long_accel = 6.0, 4.0, 3.0
    n.w_d, n.w_k, n.w_c, n.w_obs, n.obs_sigma = 1.0, 0.1, 5.0, 2.0, 0.5
    n.shift_min, n.shift_buffer, n.hold_after = 1.0, 0.5, 0.5
    n._d_end_prev, n._last_pub = 0.0, None
    n.use_grid_check, n.trust_grid_bounds = False, False
    n.grid_scan_max, n.grid_scan_step, n.bounds_warn_m = 3.0, 0.05, 0.5
    # ON, deliberately: the point is that a probe skips it and so reaches the branch under test.
    n.clear_gate_enable, n.clear_margin_m, n.clear_hyst_m = True, 0.10, 0.03
    n.clear_max_cur_d, n.clear_latch_ttl_s = 0.15, 10.0
    n._clear_latch, n._line_clear = {}, False
    n.commit_enable, n._committed = False, None
    n.commit_drop_on_new_obstacle = True
    n.commit_replan_gap_m = 7.0
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
    n.static_plan_method = "sample"
    n.body_filter = types.SimpleNamespace(eroded_image=None)
    n.map_filter = types.SimpleNamespace(eroded_image=None)
    n._publish_feasible = lambda ok: None
    n._candidate_markers = lambda *a, **k: san.MarkerArray()
    n._store_commit = lambda *a, **k: None
    for k, v in over.items():
        setattr(n, k, v)
    return n, log


# A box the FOLLOWED line already clears at obs_margin_d (= 0.30): its near edge is at
# 0.60 - 0.15 = 0.45, and 0.45 - 0.30 > 0. It is still inside the lookahead and still visible, so
# it reaches obs_ahead and the knot loop skips it -- which is what empties `knots`.
CLEARED_BOX = 0.60


def _reaches_the_branch(log):
    return any("already cleared by the" in m for m in log.msgs)


def test_a_probe_does_not_drop_the_live_commit():
    n, log = planner([box(20.0, CLEARED_BOX, 1)])
    n._committed = SENTINEL
    out = n.do_spline(gb_wpnts(), probe=True)
    assert _reaches_the_branch(log), (
        "the probe did not reach the `not knots` branch, so this test is no longer testing "
        f"anything -- check obs_margin_d against CLEARED_BOX. log: {log.msgs}")
    assert out is None, "a probe must return None, not an empty publication"
    assert n._committed is SENTINEL, (
        "the probe dropped the live commit: the next real cycle would re-plan from scratch and "
        "lay fresh geometry under a moving car, caused by a diagnostic")
    print("PASS a probe reaching `not knots` leaves the commit alone")


def test_a_real_pass_with_the_gate_on_never_gets_there():
    """The asymmetry that kept this latent, asserted rather than described.

    Same box, same margins, probe=False: the clear gate fires FIRST and idles, releasing the commit
    from its own branch one step earlier. That is why zero of 292 driving cycles ever executed the
    line under test -- not because the line is safe, but because the only caller that skips the
    gate is the only caller that must not act.
    """
    n, log = planner([box(20.0, CLEARED_BOX, 1)])
    n._committed = SENTINEL
    n.do_spline(gb_wpnts(), probe=False)
    assert any("planner idle" in m for m in log.msgs), (
        f"the clear gate did not idle; the two thresholds may have crossed: {log.msgs}")
    assert not _reaches_the_branch(log), (
        "a real pass now reaches the `not knots` branch. The clear gate's threshold "
        "(width_car/2 + clear_margin_m + clear_hyst_m) has crossed the knot loop's "
        "(width_car/2 + safety_margin_d), so this is no longer latent -- which is what the guard "
        "under test is for.")
    print("PASS with the clear gate on, a real pass idles before the branch is reachable")


def test_a_real_pass_still_drops_it_where_the_branch_IS_reachable():
    """The guard must be about the probe and nothing else.

    With the clear gate off -- the condition the branch's own comment names as the only way a real
    pass arrives here -- the commit is released exactly as before. Without this the fix could have
    been "stop clearing the commit here", which would strand a frozen path once every box ahead is
    cleared by the followed line.
    """
    n, log = planner([box(20.0, CLEARED_BOX, 1)], clear_gate_enable=False)
    n._committed = SENTINEL
    out = n.do_spline(gb_wpnts(), probe=False)
    assert _reaches_the_branch(log), f"the real pass did not reach the branch: {log.msgs}"
    assert n._committed is None, "a real pass must still release the commit here"
    assert out is not None and not len(out[0].wpnts), "a real pass returns an EMPTY publication"
    print("PASS a real pass reaching the branch still releases the commit")


def test_the_probe_is_the_only_caller_that_can_arrive_holding_one():
    """The reuse branch is what would otherwise have returned before this point.

    Under retry=True the committed-path reuse is skipped, which is the whole reason a probe can
    arrive here with `_committed` set. Asserted so that a future change making the reuse branch
    run under retry -- which would be a different bug -- is caught here rather than in sim.
    """
    n, log = planner([box(20.0, CLEARED_BOX, 1)], commit_enable=True)
    called = []
    n._reuse_committed = lambda *a, **k: called.append(1)
    n._committed = SENTINEL
    n.do_spline(gb_wpnts(), probe=True)
    assert not called, "a probe must not consult the committed path"
    assert n._committed is SENTINEL
    print("PASS a probe neither reads nor writes the commit")


if __name__ == "__main__":
    test_a_probe_does_not_drop_the_live_commit()
    test_a_real_pass_with_the_gate_on_never_gets_there()
    test_a_real_pass_still_drops_it_where_the_branch_IS_reachable()
    test_the_probe_is_the_only_caller_that_can_arrive_holding_one()
    print("ALL PASS")
