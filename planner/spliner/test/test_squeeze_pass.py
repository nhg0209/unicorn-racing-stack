#!/usr/bin/env python3
"""Harness test of the static planner's SQUEEZE PASS (reduced-margin retry).

Loads the REAL static_avoidance_node module and binds the REAL methods onto a bare object. What
it pins down:
  1. The schedule steps from the design margins DOWN to the floors, widest attempt first, so a
     section that only needs a couple of centimetres is driven with a couple of centimetres.
  2. It is empty above squeeze_max_speed_mps -- trading clearance for motion is only defensible
     where a mis-clearance is survivable; at racing speed "no candidate" still means TRAILING.
  3. It is empty when disabled, and when the design margins are already at/below the floors
     (nothing left to give), so the caller falls through to feasible=False as before.
  4. The margin a path was SOLVED at is committed with it, and the squeeze marking survives
     commitment -- both are needed or the maneuver tears itself down on its first reuse: the
     re-check would judge a squeeze path by the full design margin and drop it, and the SM's
     speed cap would apply for exactly one cycle.

Run (after sourcing the workspace):
  python3 planner/spliner/test/test_squeeze_pass.py
"""
import types
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parents[1] / "spliner" / "static_avoidance_node.py"

src = MOD.read_text()
san = types.ModuleType("san")
san.__dict__["__file__"] = str(MOD)
exec(compile(src, str(MOD), "exec"), san.__dict__)
ObstacleSpliner = san.ObstacleSpliner


def node(cur_vs=0.0, safety=0.15, wall=0.10, enable=True, steps=2,
         s_floor=0.05, w_floor=0.05, v_max=3.0):
    n = ObstacleSpliner.__new__(ObstacleSpliner)
    n.cur_vs = cur_vs
    n.safety_margin = safety
    n.wall_margin = wall
    n.squeeze_enable = enable
    n.squeeze_steps = steps
    n.squeeze_safety_floor_m = s_floor
    n.squeeze_wall_floor_m = w_floor
    n.squeeze_max_speed_mps = v_max
    return n


def test_schedule_steps_down_to_the_floor():
    sched = node(cur_vs=0.0)._squeeze_schedule()
    assert len(sched) == 2, sched
    # widest first, monotonically tightening, landing exactly on the floors
    assert sched[0][0] > sched[1][0] and sched[0][1] > sched[1][1]
    assert abs(sched[-1][0] - 0.05) < 1e-9 and abs(sched[-1][1] - 0.05) < 1e-9
    assert sched[0][0] < 0.15 and sched[0][1] < 0.10, "the first attempt must already reduce"
    print(f"PASS schedule steps {[(round(a,3), round(b,3)) for a, b in sched]} down to the floor")


def test_schedule_respects_the_speed_gate():
    assert node(cur_vs=2.9)._squeeze_schedule(), "below the gate the pass is offered"
    assert node(cur_vs=3.0)._squeeze_schedule() == [], "at the gate it is not"
    assert node(cur_vs=6.0)._squeeze_schedule() == [], "and certainly not at racing speed"
    print("PASS the squeeze pass is offered only below squeeze_max_speed_mps")


def test_schedule_disabled_and_already_at_floor():
    assert node(cur_vs=0.0, enable=False)._squeeze_schedule() == []
    assert node(cur_vs=0.0, safety=0.05, wall=0.05)._squeeze_schedule() == [], \
        "already at the floor -> nothing to give, fall through to feasible=False"
    assert node(cur_vs=0.0, safety=0.02, wall=0.02)._squeeze_schedule() == [], \
        "below the floor -> the schedule must never RAISE a margin"
    print("PASS schedule is empty when disabled or already at/below the floor")


def test_commit_records_the_margin_and_the_marking():
    n = node(cur_vs=0.0)
    obs = [types.SimpleNamespace(id=3, s_center=5.0, d_center=0.2)]
    arr = np.zeros(4)
    n._store_commit(obs, arr, arr, np.zeros((4, 2)), arr, arr, arr,
                    obs_margin=0.20, squeeze=True)
    assert n._committed['obs_margin'] == 0.20, "the re-check must ask the question the path answers"
    assert n._committed['squeeze'] is True
    # ...and the marking is re-emitted on every republish of the committed path, not just the first
    stamp = san.OTWpntArray().header.stamp
    n.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(to_msg=lambda: stamp))
    msg = n._commit_to_msg(n._committed, slice(0, 4))
    assert msg.ot_line == "squeeze", "the SM speed cap keys off this on every cycle of the maneuver"
    # a normal commit must NOT be marked
    n._store_commit(obs, arr, arr, np.zeros((4, 2)), arr, arr, arr, obs_margin=0.30, squeeze=False)
    assert n._commit_to_msg(n._committed, slice(0, 4)).ot_line == ""
    print("PASS the solved margin and the squeeze marking are committed with the path")


if __name__ == "__main__":
    test_schedule_steps_down_to_the_floor()
    test_schedule_respects_the_speed_gate()
    test_schedule_disabled_and_already_at_floor()
    test_commit_records_the_margin_and_the_marking()
    print("ALL PASS")
