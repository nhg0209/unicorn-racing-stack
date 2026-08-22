#!/usr/bin/env python3
"""The four propagation modes, extracted from the deleted spliner_node, still run and still wrap.

This guards a file NOTHING CALLS. That is the point: an unwired reference implementation with no
test is indistinguishable from dead code the next time someone reads the directory, and the whole
reason obs_propagation.py exists is that the migration item outlived the node it lived in.

Two of the assertions are about bugs the original had and this extraction fixes -- `heuristic`
raising UnboundLocalError, and `adaptive_velheuristic` dividing by zero at a standstill. They are
tested because "all four modes run" was not true before, and a reader has no way to know which of
the four were ever executed.

Run (after sourcing the workspace):
  ~/miniforge3/envs/unicorn/bin/python3 planner/lane_change_planner/test/test_obs_propagation.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lane_change_planner.obs_propagation import (        # noqa: E402
    MODES, ego_vx_at, predict_obs_movement)

TRACK_LEN = 40.0


def obs(s_center=5.0, d_center=0.4, vs=1.0, vd=0.2, half_s=0.15, half_d=0.15):
    return types.SimpleNamespace(
        s_start=(s_center - half_s) % TRACK_LEN, s_center=s_center,
        s_end=(s_center + half_s) % TRACK_LEN,
        d_left=d_center + half_d, d_center=d_center, d_right=d_center - half_d,
        vs=vs, vd=vd)


def test_every_mode_runs():
    """Including the two that could not, before."""
    for mode in MODES:
        out = predict_obs_movement(obs(), cur_s=0.0, gb_max_s=TRACK_LEN, ego_vx=4.0, mode=mode)
        assert out is not None, mode
        for f in ("s_start", "s_center", "s_end", "d_left", "d_center", "d_right"):
            v = getattr(out, f)
            assert v == v and abs(v) < 1e6, f"{mode}: {f} = {v}"
    print(f"PASS all {len(MODES)} modes run: {', '.join(MODES)}")


def test_a_standstill_ego_does_not_divide_by_zero():
    """adaptive_velheuristic's rel_speed is (1 - scaler) * ego_speed, unguarded in the original."""
    out = predict_obs_movement(obs(), cur_s=0.0, gb_max_s=TRACK_LEN, ego_vx=0.0,
                               mode="adaptive_velheuristic")
    assert out.s_center == out.s_center, "NaN from a zero closing speed"
    print("PASS adaptive_velheuristic survives a stationary ego")


def test_s_wraps_at_the_lap():
    """A box near the seam must come back inside [0, gb_max_s), on all three s fields."""
    out = predict_obs_movement(obs(s_center=TRACK_LEN - 0.5, vs=6.0), cur_s=TRACK_LEN - 2.0,
                               gb_max_s=TRACK_LEN, ego_vx=8.0, mode="constant",
                               fixed_pred_time=2.0)
    for f in ("s_start", "s_center", "s_end"):
        v = getattr(out, f)
        assert 0.0 <= v < TRACK_LEN, f"{f} = {v} is outside the lap"
    print("PASS s wraps to the lap length on every s field")


def test_the_input_is_not_mutated():
    """The original moved the obstacle in place and returned it; callers shared the mutation."""
    o = obs()
    before = (o.s_center, o.d_center)
    predict_obs_movement(o, cur_s=0.0, gb_max_s=TRACK_LEN, ego_vx=4.0, mode="adaptive")
    assert (o.s_center, o.d_center) == before, "predict_obs_movement mutated its argument"
    print("PASS the caller's obstacle is left alone")


def test_beyond_the_horizon_nothing_moves():
    o = obs(s_center=25.0)
    out = predict_obs_movement(o, cur_s=0.0, gb_max_s=TRACK_LEN, ego_vx=4.0, mode="adaptive")
    assert (out.s_center, out.d_center) == (o.s_center, o.d_center)
    print("PASS an obstacle past the horizon is returned unpropagated")


def test_ego_vx_at_uses_the_real_spacing():
    """The defect this helper exists for: the original hardcoded int(cur_s * 10).

    At a 0.5 m spacing, station 4 is s = 2.0 m. The old expression would have asked for index 20.
    """
    wpnts = [types.SimpleNamespace(vx_mps=float(i)) for i in range(40)]
    assert ego_vx_at(2.0, wpnts, 0.5) == 4.0
    assert ego_vx_at(2.0, wpnts, 0.1) == 20.0        # what the original always did
    assert ego_vx_at(2.0, [], 0.5) == 0.0
    print("PASS ego_vx_at indexes by wpnt_dist, not by a hardcoded 0.1 m")


def test_an_unknown_mode_is_refused():
    """A fallthrough branch is how reopt_method silently selected a solver that took minutes."""
    try:
        predict_obs_movement(obs(), cur_s=0.0, gb_max_s=TRACK_LEN, mode="adptive")
    except ValueError:
        print("PASS an unknown mode raises instead of falling through")
        return
    raise AssertionError("an unknown mode was accepted")


if __name__ == "__main__":
    test_every_mode_runs()
    test_a_standstill_ego_does_not_divide_by_zero()
    test_s_wraps_at_the_lap()
    test_the_input_is_not_mutated()
    test_beyond_the_horizon_nothing_moves()
    test_ego_vx_at_uses_the_real_spacing()
    test_an_unknown_mode_is_refused()
    print("ALL PASS")
