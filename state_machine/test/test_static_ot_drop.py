#!/usr/bin/env python3
"""Harness test of what happens when a static OVERTAKE is DROPPED.

Binds the REAL StateMachine methods (and the REAL ObstacleTransition) onto a bare object with
hand-set state. What it pins down:
  1. While the car is still out on the avoidance hump, TRAILING keeps the avoidance geometry as
     its reference instead of snapping back to the raw raceline -- which, for a STATIC obstacle,
     is the line that runs into it.
  2. That reference switch does not cost TRAILING its target: get_farthest_target answers for an
     OVERTAKE source too, so the gap PID still has something to brake for.
  3. The two sustain terms are symmetric: availability blips are debounced over the same window
     as feasibility blips, so the noisier term cannot decide the exit on its own.
  4. A drop starts a re-entry cooldown, so the drop/re-enter pair cannot oscillate at the
     message rate.

Run (after sourcing the workspace):
  python3 state_machine/test/test_static_ot_drop.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_machine.state_machine_node import StateMachine        # noqa: E402
from state_machine.states_types import StateType                 # noqa: E402
from state_machine import state_transitions                      # noqa: E402


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def error(self, *a, **k): pass


def wd(is_init=True, on_spline=True, closest_target=None):
    """WaypointData stand-in with just the fields the code under test reads."""
    return types.SimpleNamespace(
        is_init=is_init,
        array=np.zeros((5, 3)) if is_init else None,
        list=[types.SimpleNamespace(s_m=float(i)) for i in range(5)] if is_init else [],
        closest_target=closest_target, closest_gap=1.0,
        latest_threshold=1.0,
        on_spline_front_horizon_thres_m=0.5, on_spline_min_dist_thres_m=0.85,
        _on_spline=on_spline)


def sm(cur_d=0.5, t=100.0):
    n = StateMachine.__new__(StateMachine)
    n.name = "state_machine"
    n.get_logger = lambda: _Log()
    n.now_sec = lambda: n._t
    n._t = t
    n.cur_d = cur_d
    n.recovery_exit_d_m = 0.2
    n.cur_static_avoidance_wpnts = wd()
    n.cur_gb_wpnts = wd()
    n.cur_avoidance_wpnts = wd()
    n.cur_recovery_wpnts = wd()
    n.cur_start_wpnts = wd()
    # _check_on_spline is exercised in its own right elsewhere; here it is the harness knob that
    # says whether the car is still on the cached path.
    n._check_on_spline = lambda w: bool(getattr(w, "_on_spline", False))
    n.static_avoidance_feasible = True
    n._static_feasible_t = t
    n._static_feasible_true_t = t
    n.static_feasible_stale_sec = 0.5
    n.static_feasible_lost_sec = 0.4
    n._static_avail_true_t = t
    n.static_avail_lost_sec = 0.4
    n._static_ot_cooldown_until = None
    n.static_ot_reentry_cooldown_sec = 0.3
    n.static_overtaking_mode = True
    n.static_avoidance_wpnts = types.SimpleNamespace(
        wpnts=[1, 2, 3],
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=int(t), nanosec=0)))
    return n


def test_trailing_holds_the_avoidance_reference_while_off_the_line():
    n = sm(cur_d=0.5)
    assert n._hold_static_avoidance_reference() is True
    # ...and the transition acts on it
    n._check_free_frenet = lambda w: False
    n._check_line_lost = lambda: False
    n._check_overtaking_mode = lambda: False
    n._check_static_overtaking_mode = lambda: False
    state, src = state_transitions.ObstacleTransition(n, close_to_raceline=False)
    assert state == StateType.TRAILING and src == StateType.OVERTAKE, (state, src)
    print("PASS TRAILING keeps the avoidance geometry while the car is still out on it")


def test_reference_returns_to_the_raceline_when_appropriate():
    # back near the line -> the raceline is the right reference again
    n = sm(cur_d=0.1)
    assert n._hold_static_avoidance_reference() is False
    # off the line but no longer on the cached path -> likewise
    n = sm(cur_d=0.5)
    n.cur_static_avoidance_wpnts._on_spline = False
    assert n._hold_static_avoidance_reference() is False
    # off the line, path never cached -> likewise (and no crash)
    n = sm(cur_d=0.5)
    n.cur_static_avoidance_wpnts = wd(is_init=False)
    assert n._hold_static_avoidance_reference() is False
    print("PASS the hold is conservative: it releases as soon as either condition fails")


def test_trailing_keeps_its_target_across_the_reference_switch():
    n = sm(cur_d=0.5)
    obs = types.SimpleNamespace(id=4)
    n.cur_static_avoidance_wpnts.closest_target = obs
    tgt, src = n.get_farthest_target(StateType.OVERTAKE)
    assert tgt == [obs] and src == StateType.OVERTAKE, (tgt, src)
    # falls back to the global line's target when the avoidance cache has none
    n.cur_static_avoidance_wpnts.closest_target = None
    n.cur_gb_wpnts.closest_target = obs
    assert n.get_farthest_target(StateType.OVERTAKE)[0] == [obs]
    # and still reports nothing when there genuinely is nothing
    n.cur_gb_wpnts.closest_target = None
    assert n.get_farthest_target(StateType.OVERTAKE)[0] == []
    print("PASS the gap PID keeps its target when the reference switches to the avoidance path")


def test_availability_blip_is_debounced_like_feasibility():
    n = sm()
    n._check_availability = lambda a, b: False        # one late publish
    assert n._check_overtaking_mode_sustainability() is True, "a blip must not drop OVERTAKE"
    assert n._static_ot_cooldown_until is None
    n._t += n.static_avail_lost_sec + 0.01            # sustained unavailability
    assert n._check_overtaking_mode_sustainability() is False
    assert n._static_ot_cooldown_until is not None, "a real drop arms the cooldown"
    print("PASS availability blips are debounced over the same window as feasibility blips")


def test_drop_arms_a_reentry_cooldown():
    n = sm()
    n._check_availability = lambda a, b: True
    n._check_latest_wpnts = lambda a, b: True
    n.current_position = [0.0, 0.0, 0.0]
    n.cur_s = 0.0
    n.track_length = 40.0
    # feasibility lost -> drop + cooldown
    n._t += n.static_feasible_lost_sec + 0.01
    assert n._check_overtaking_mode_sustainability() is False
    assert n._check_static_overtaking_mode() is False, "re-entry must be refused during cooldown"
    n._t += n.static_ot_reentry_cooldown_sec + 0.01
    n._static_feasible_t = n._t
    n._static_feasible_true_t = n._t
    assert n._check_static_overtaking_mode() is True, "...and allowed once it expires"
    print("PASS a drop arms a short re-entry cooldown that then expires")


if __name__ == "__main__":
    test_trailing_holds_the_avoidance_reference_while_off_the_line()
    test_reference_returns_to_the_raceline_when_appropriate()
    test_trailing_keeps_its_target_across_the_reference_switch()
    test_availability_blip_is_debounced_like_feasibility()
    test_drop_arms_a_reentry_cooldown()
    print("ALL PASS")
