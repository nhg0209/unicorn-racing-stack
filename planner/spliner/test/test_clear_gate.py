#!/usr/bin/env python3
"""Harness test of the static planner's raceline-CLEAR gate.

Loads the REAL static_avoidance_node module and binds the REAL gate methods onto a bare object
with hand-set state. What it pins down:
  1. The clearance is EVALUATED whatever |cur_d| is -- the bug was that it was not, so with the
     documented ~0.5 m tracking error the gate never ran on the swapped line and every pass
     re-avoided a box the global line already cleared.
  2. Entry needs clear_hyst_m more than staying does (anti-flap), and the latch that grants the
     cheaper stay threshold is per OBSTACLE ID.
  3. That latch survives the obstacle leaving the lookahead -- a box exits the horizon on every
     approach, and resetting there cost one re-triggered maneuver per pass.
  4. |cur_d| still means something: standing down while the car is out on an excursion CANCELS
     it, so it must be earned at the entry margin and can never ride on a latch.
  5. A real keep-out violation drops the latch; stale latches are pruned by TTL.

Run (after sourcing the workspace):
  python3 planner/spliner/test/test_clear_gate.py
"""
import sys
import types
from pathlib import Path

MOD = Path(__file__).resolve().parents[1] / "spliner" / "static_avoidance_node.py"

# exec the module rather than importing the package, so no ROS node is constructed
src = MOD.read_text()
san = types.ModuleType("san")
san.__dict__["__file__"] = str(MOD)
exec(compile(src, str(MOD), "exec"), san.__dict__)
ObstacleSpliner = san.ObstacleSpliner

HALF_CAR = 0.15          # width_car / 2
BASE = HALF_CAR + 0.10   # + clear_margin_m  -> 0.25 m, the STAY threshold
HYST = 0.03              # clear_hyst_m      -> 0.28 m, the ENTRY threshold


class _Log:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    warn = warning


class _Clock:
    def __init__(self): self.t = 100.0
    def now(self): return types.SimpleNamespace(nanoseconds=int(self.t * 1e9))


def gate_node(cur_d=0.0):
    n = ObstacleSpliner.__new__(ObstacleSpliner)
    n.name = "static_avoidance_planner"
    n._clock = _Clock()
    n.get_clock = lambda: n._clock
    n.get_logger = lambda: _Log()
    n.cur_d = cur_d
    n.clear_margin_m = 0.10
    n.clear_hyst_m = HYST
    n.clear_max_cur_d = 0.15
    n.clear_latch_ttl_s = 10.0
    n._clear_latch = {}
    return n


def box(oid, edge):
    """Obstacle whose NEAR edge sits `edge` metres to the +d side of the followed line."""
    return types.SimpleNamespace(id=oid, d_right=edge, d_left=edge + 0.30)


def test_evaluation_is_not_gated_on_cur_d():
    # THE regression. The car is 0.5 m off the line (documented steady-state tracking error,
    # 3x clear_max_cur_d) and the line clears the box comfortably. The gate must say so.
    n = gate_node(cur_d=0.5)
    assert n._eval_clear_gate([box(1, 0.60)], HALF_CAR) is True
    print("PASS clearance is evaluated at |cur_d| far above clear_max_cur_d")


def test_entry_needs_more_than_stay():
    # Edge at 0.26: above the 0.25 stay threshold, below the 0.28 entry threshold.
    n = gate_node(cur_d=0.0)
    assert n._eval_clear_gate([box(1, 0.26)], HALF_CAR) is False, "fresh must need entry margin"
    # ...and above entry it latches, after which the stay threshold applies.
    n = gate_node(cur_d=0.0)
    assert n._eval_clear_gate([box(1, 0.30)], HALF_CAR) is True
    assert 1 in n._clear_latch
    assert n._eval_clear_gate([box(1, 0.26)], HALF_CAR) is True, "latched must keep stay margin"
    print("PASS entry threshold exceeds stay threshold, and the latch grants stay")


def test_latch_survives_lookahead_exit():
    # The obstacle leaves the horizon (no obstacles ahead -> only a prune runs) and comes back.
    # Its latch must still be there, or it re-earns idle at the entry margin once per pass.
    n = gate_node(cur_d=0.0)
    assert n._eval_clear_gate([box(7, 0.30)], HALF_CAR) is True
    n._clock.t += 2.0
    n._prune_clear_latch()                       # what do_spline does with nothing ahead
    assert 7 in n._clear_latch, "a latch must outlive the lookahead"
    assert n._eval_clear_gate([box(7, 0.26)], HALF_CAR) is True
    print("PASS the clear latch survives the obstacle leaving the lookahead")


def test_latch_is_per_obstacle():
    # One cleared box must not vouch for another that is genuinely on the line.
    n = gate_node(cur_d=0.0)
    assert n._eval_clear_gate([box(1, 0.30)], HALF_CAR) is True
    assert n._eval_clear_gate([box(1, 0.26), box(2, 0.26)], HALF_CAR) is False, \
        "the un-latched box 2 must still need the entry margin"
    assert 1 in n._clear_latch and 2 not in n._clear_latch
    print("PASS the latch is per obstacle id, not global")


def test_cancel_needs_the_entry_margin():
    # Latched and clear at the stay margin, but the car is out on an excursion: standing down
    # would cancel it, so the entry margin is required and the latch does not help.
    n = gate_node(cur_d=0.0)
    assert n._eval_clear_gate([box(1, 0.30)], HALF_CAR) is True
    n.cur_d = 0.40                                # |cur_d| >= clear_max_cur_d
    assert n._eval_clear_gate([box(1, 0.26)], HALF_CAR) is False, \
        "a cancel may not ride on a latch"
    assert n._eval_clear_gate([box(1, 0.30)], HALF_CAR) is True, \
        "...but a genuine entry-margin clearance still cancels"
    print("PASS cancelling an excursion is earned at the entry margin, never on a latch")


def test_violation_drops_the_latch():
    n = gate_node(cur_d=0.0)
    assert n._eval_clear_gate([box(1, 0.30)], HALF_CAR) is True
    assert n._eval_clear_gate([box(1, 0.10)], HALF_CAR) is False   # box on the line
    assert 1 not in n._clear_latch, "a real keep-out violation must drop the latch"
    assert n._eval_clear_gate([box(1, 0.26)], HALF_CAR) is False, "and idle must be re-earned"
    print("PASS a real violation drops the latch and idle must be re-earned")


def test_ttl_prunes_stale_latches():
    n = gate_node(cur_d=0.0)
    assert n._eval_clear_gate([box(1, 0.30)], HALF_CAR) is True
    n._clock.t += n.clear_latch_ttl_s + 1.0
    n._prune_clear_latch()
    assert not n._clear_latch, "stale latches must not accumulate over a session"
    print("PASS stale latches are pruned by TTL")


if __name__ == "__main__":
    test_evaluation_is_not_gated_on_cur_d()
    test_entry_needs_more_than_stay()
    test_latch_survives_lookahead_exit()
    test_latch_is_per_obstacle()
    test_cancel_needs_the_entry_margin()
    test_violation_drops_the_latch()
    test_ttl_prunes_stale_latches()
    print("ALL PASS")
