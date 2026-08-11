#!/usr/bin/env python3
"""Harness test of StateMachine.update_velocity's profile cache and per-path lateral limit.

calc_vel_profile is the expensive call in this node, and a COMMITTED planner path is
geometrically frozen -- the planner republishes the same points at 20 Hz. Re-solving each one is
what put the single-threaded executor 0.3-0.5 s behind, and the freshness gate built on top of
that lag is what then blocked the OVERTAKE commit, so the cost was a lost maneuver, not just
lost time.

A wrong cache is worse than no cache (frozen speed commands), so this pins the invalidation:
  1. identical geometry + speed -> solved once,
  2. changed geometry -> re-solved,
  3. v_start drifting past the quantum -> re-solved (the profile is solved FROM the current
     speed, so it is not a pure function of the points),
  4. one slot per source -- static and dynamic both publish at 20 Hz and must not evict
     each other,
  5. v_cap and ay_max are part of the key, and ay_max really does replace the ggv's lateral
     column for that path only.

Run (after sourcing the workspace):
  python3 state_machine/test/test_velocity_cache.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_machine import state_machine_node as smn                # noqa: E402
from state_machine.state_machine_node import StateMachine          # noqa: E402


class _Log:
    def warn(self, *a, **k): pass
    def info(self, *a, **k): pass


class _Solver:
    """Stand-in for calc_vel_profile that records how often (and with what ggv) it was called.

    It HONOURS v_start and v_max, because that is the contract the caller now relies on: the cap
    is handed to the solver as a bound rather than applied to its output, so a stub that ignores
    v_max would make the cap look like it had stopped working.
    """
    def __init__(self):
        self.calls = 0
        self.ggv_seen = []
        self.vmax_seen = []

    def __call__(self, **kw):
        self.calls += 1
        self.ggv_seen.append(np.array(kw["ggv"], copy=True))
        self.vmax_seen.append(kw.get("v_max"))
        top = min(5.0, float(kw.get("v_max") or 5.0))
        start = float(kw.get("v_start") if kw.get("v_start") is not None else top)
        return np.linspace(start, top, len(kw["kappa"]))


def path(n=10, kappa=0.1, dx=0.1):
    """OTWpntArray stand-in: n points on a straight line with constant curvature."""
    wp = [types.SimpleNamespace(x_m=i * dx, y_m=0.0, s_m=i * dx, kappa_radpm=kappa,
                                vx_mps=0.0, ax_mps2=0.0) for i in range(n)]
    return types.SimpleNamespace(wpnts=wp)


def sm(cur_vs=3.0):
    n = StateMachine.__new__(StateMachine)
    n.name = "state_machine"
    n.get_logger = lambda: _Log()
    n.cur_vs = cur_vs
    n.wpnt_dist = 0.1
    n._vel_cache = {}
    n.vel_cache_quant_mps = 0.25
    n.vel_cache_sig_pts = 40       # tail points that identify the geometry (re-slice invariant)
    n.ggv = np.array([[0.0, 5.0, 4.5], [10.0, 5.0, 4.5]])
    n.ax_max_machines = np.array([[0.0, 5.0], [10.0, 5.0]])
    n.b_ax_max_machines = np.array([[0.0, 5.0], [10.0, 5.0]])
    n.gb_wpnts = types.SimpleNamespace(
        wpnts=[types.SimpleNamespace(vx_mps=6.0) for _ in range(400)])
    n.pars = {"veh_params": {"dragcoeff": 0.0136, "mass": 3.5, "v_max": 7.0},
              "vel_calc_opts": {"dyn_model_exp": 1.0, "vel_profile_conv_filt_window": None}}
    return n


def test_frozen_path_is_solved_once():
    n, solver = sm(), _Solver()
    smn.calc_vel_profile = solver
    for _ in range(20):                              # one second of 20 Hz republishes
        n.update_velocity(path(), cache_key="static")
    assert solver.calls == 1, f"a frozen path must be solved once, was {solver.calls}"
    print("PASS a geometrically frozen path is solved once, not once per republish")


def test_geometry_change_invalidates():
    n, solver = sm(), _Solver()
    smn.calc_vel_profile = solver
    n.update_velocity(path(kappa=0.1), cache_key="static")
    n.update_velocity(path(kappa=0.9), cache_key="static")   # a real re-plan
    assert solver.calls == 2, solver.calls
    print("PASS changed geometry re-solves")


def test_speed_drift_invalidates_past_the_quantum():
    n, solver = sm(cur_vs=3.0), _Solver()
    smn.calc_vel_profile = solver
    n.update_velocity(path(), cache_key="static")
    n.cur_vs = 3.05                                   # inside the quantum
    n.update_velocity(path(), cache_key="static")
    assert solver.calls == 1, "sub-quantum speed drift must reuse the profile"
    n.cur_vs = 3.6                                    # past it
    n.update_velocity(path(), cache_key="static")
    assert solver.calls == 2, "the profile is solved FROM the speed; a real change must re-solve"
    print("PASS v_start drift past the quantum re-solves, inside it does not")


def test_sources_do_not_evict_each_other():
    n, solver = sm(), _Solver()
    smn.calc_vel_profile = solver
    for _ in range(5):                                # both planners publishing at 20 Hz
        n.update_velocity(path(), cache_key="static")
        n.update_velocity(path(), cache_key="dynamic")
    assert solver.calls == 2, f"one solve per source, was {solver.calls}"
    print("PASS static and dynamic paths get their own cache slot")


def test_a_forward_reslice_still_hits_the_cache():
    # THE reason the cache measured 0%. A committed path is frozen, but the planner republishes
    # only the part still ahead of the car, so the array loses points from the FRONT -- at 3 m/s a
    # 0.1 m station is passed every 34 ms against this node's 50 ms cycle. Hashing the whole
    # array's float64 bytes therefore missed every cycle, and the 0.3-0.5 s of executor lag the
    # cache exists to remove was still in the log.
    n, solver = sm(), _Solver()
    smn.calc_vel_profile = solver
    full = path(n=60)
    n.update_velocity(full, cache_key="static")
    assert solver.calls == 1
    for k in range(1, 12):                       # the car eats one station per cycle
        msg = types.SimpleNamespace(wpnts=[w for w in path(n=60).wpnts[k:]])
        n.update_velocity(msg, cache_key="static")
        n.cur_vs = msg.wpnts[0].vx_mps           # ...and drives the reference it was given
    assert solver.calls == 1, (
        f"a forward re-slice of a frozen path re-solved {solver.calls - 1} times: the key is not "
        f"invariant to the head being consumed")
    print("PASS a frozen path re-sliced forward 11 times is solved once")


def test_a_reslice_whose_head_left_the_car_behind_re_solves():
    # The head is validated rather than keyed: a cached profile is reused only if, sliced to the
    # current window, it starts within one quantum of the speed the car is actually doing.
    n, solver = sm(cur_vs=3.0), _Solver()
    smn.calc_vel_profile = solver
    n.update_velocity(path(n=60), cache_key="static")
    assert solver.calls == 1
    n.cur_vs = 3.0 + 4.0 * n.vel_cache_quant_mps       # the car is now far off the cached head
    n.update_velocity(types.SimpleNamespace(wpnts=path(n=60).wpnts[3:]), cache_key="static")
    assert solver.calls == 2, "a profile whose head no longer matches the car must be re-solved"
    print("PASS a cached profile is dropped once its head no longer matches the car's speed")


def test_a_changed_tail_is_a_different_path():
    n, solver = sm(), _Solver()
    smn.calc_vel_profile = solver
    n.update_velocity(path(n=60), cache_key="static")
    other = path(n=60)
    other.wpnts[-1].kappa_radpm += 0.5                 # a genuinely different geometry
    n.update_velocity(other, cache_key="static")
    assert solver.calls == 2, "a path with a different tail must re-solve"
    print("PASS a different tail geometry is a different path")


def test_cap_and_ay_max_are_honoured_and_keyed():
    n, solver = sm(), _Solver()
    smn.calc_vel_profile = solver
    msg = path()
    n.update_velocity(msg, cache_key="static")
    assert abs(msg.wpnts[-1].vx_mps - 5.0) < 1e-9
    # the cap applies, and it is part of the key (so it cannot be served a stale uncapped profile)
    msg2 = path()
    n.update_velocity(msg2, v_cap=2.5, cache_key="static")
    assert solver.calls == 2, "a different cap must re-solve"
    assert abs(solver.vmax_seen[-1] - 2.5) < 1e-9, \
        f"the cap must reach the solver as a bound, not be applied to its output: {solver.vmax_seen}"
    assert abs(msg2.wpnts[-1].vx_mps - 2.5) < 1e-9, msg2.wpnts[-1].vx_mps
    # ...and THE HEAD OF THE PROFILE IS THE CAR'S CURRENT SPEED. Applied as np.minimum to the
    # solved profile the cap rewrote vx[0] -- the solver's own v_start -- from cur_vs to the cap,
    # so the reference began with a step the controller's slew limiter then took ~300 ms to serve.
    assert abs(msg2.wpnts[0].vx_mps - n.cur_vs) < 1e-9, (
        f"the profile starts at {msg2.wpnts[0].vx_mps:.3f} against a current speed of "
        f"{n.cur_vs:.3f}: that difference is a step in the reference")
    # ay_max replaces the ggv's lateral column for this path only
    n.update_velocity(path(), ay_max=5.0, cache_key="static")
    assert np.allclose(solver.ggv_seen[-1][:, 2], 5.0), solver.ggv_seen[-1]
    assert np.allclose(n.ggv[:, 2], 4.5), "the node's own ggv must not be mutated"
    assert np.allclose(solver.ggv_seen[0][:, 2], 4.5), "...and the default path still uses it"
    print("PASS v_cap and ay_max are applied and keyed; the global ggv is left alone")


if __name__ == "__main__":
    test_frozen_path_is_solved_once()
    test_geometry_change_invalidates()
    test_speed_drift_invalidates_past_the_quantum()
    test_sources_do_not_evict_each_other()
    test_a_forward_reslice_still_hits_the_cache()
    test_a_reslice_whose_head_left_the_car_behind_re_solves()
    test_a_changed_tail_is_a_different_path()
    test_cap_and_ay_max_are_honoured_and_keyed()
    print("ALL PASS")
