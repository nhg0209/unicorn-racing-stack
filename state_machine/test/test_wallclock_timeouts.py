#!/usr/bin/env python3
"""The state machine's timeouts are wall-clock, and that change was behaviour-neutral.

Three timeouts used to be counted in LOOP CYCLES against `timeout_sec * rate_hz`:
the static-trailing deadlock, the FTG timer, and the OVERTAKE ttl latch. That is correct only
while the loop actually achieves `rate_hz`. It does on the car -- bags rosbag2_2026_08_19-22_38_39
and -22_42_37 measure /state_machine at 79.6 / 79.8 Hz, dt p50 12.50 ms, p95 13.4 ms, zero rate
warnings -- so this was never a live bug. It was a coupling: RViz point clouds, bag recording or one
more node slow the loop, and every one of those timeouts silently stretches by the same factor,
with nothing in the log saying so.

What this file pins down, for each of the three:

  1. At 80 Hz the new wall-clock code fires on EXACTLY the same cycle as the cycle-counting code it
     replaced. The old logic is re-implemented here (`_legacy_*`) as a reference model, so the
     comparison survives the original being deleted.
  2. At a synthetic 40 Hz -- the loop degraded, `rate: 80` still in the yaml -- the legacy model
     fires at 2x the wall-clock time it was configured for, and the new code fires at the
     configured time regardless.

The fake clock counts INTEGER NANOSECONDS and divides by 1e9 on read. Accumulating a float `dt`
instead puts ~1e-13 s of drift on the comparison, which lands exactly on the `<=` at the timeout
boundary and shifts the firing cycle by one -- an artefact of the harness, not of the code.

Run (after sourcing the workspace):
  python3 state_machine/test/test_wallclock_timeouts.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_machine.state_machine_node import StateMachine        # noqa: E402
from state_machine.states_types import StateType                 # noqa: E402
from state_machine import state_transitions                      # noqa: E402

CONFIGURED_RATE_HZ = 80          # what state_machine_params.yaml says
DT80 = 1.0 / CONFIGURED_RATE_HZ
DT40 = 1.0 / 40.0
TOL_S = 1e-9


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def error(self, *a, **k): pass


def _base(dt):
    """Bare node with an exact fake clock that advances one loop period per `tick()`."""
    n = StateMachine.__new__(StateMachine)
    n.name = "state_machine"
    n.get_logger = lambda: _Log()
    n._ns = 0
    n._dt_ns = round(dt * 1e9)
    n.now_sec = lambda: n._ns / 1e9
    n.tick = lambda: setattr(n, "_ns", n._ns + n._dt_ns)
    return n


# --------------------------------------------------------------------------------------------- #
# 1. STATIC TRAILING DEADLOCK                                                                     #
# --------------------------------------------------------------------------------------------- #
def _deadlock_node(dt, timeout_s):
    n = _base(dt)
    n.cur_state = StateType.TRAILING
    n.cur_vs = 0.0
    n.cur_s = 0.0
    n.track_length = 40.0
    n.static_deadlock_speed_mps = 0.3
    n.static_deadlock_timeout_s = timeout_s
    n._static_deadlock_t0 = None
    n._relax_sent_t = None
    n.relax_repeat_sec = 2.0
    n.static_avoidance_feasible = False
    n.relax_pub = types.SimpleNamespace(publish=lambda msg: None)
    n.obstacles_in_interest = [types.SimpleNamespace(id=7, is_static=True, s_start=5.0)]
    return n


def _legacy_deadlock(cycles, timeout_s, rate_hz, dt):
    """Pre-change logic: `counter += 1` per stalled cycle, fire once counter > timeout * rate.

    Returns (cycle index it first fired on, wall-clock seconds that had elapsed by then)."""
    counter = 0
    for i in range(cycles):
        counter += 1
        if counter > timeout_s * rate_hz:
            return i, counter * dt
    return None, None


def _run_deadlock(dt, cycles, timeout_s=1.5):
    """Drive the REAL _check_static_trailing_deadlock. Same return shape as _legacy_deadlock."""
    n = _deadlock_node(dt, timeout_s)
    n.cur_vs = 3.0                        # one non-stalled cycle -> establishes the reset baseline
    n._check_static_trailing_deadlock()
    t0 = n.now_sec()
    n.tick()
    n.cur_vs = 0.0
    for i in range(cycles):
        if n._check_static_trailing_deadlock():
            return i, n.now_sec() - t0
        n.tick()
    return None, None


def test_deadlock_is_behaviour_neutral_at_80hz():
    new_cycle, new_t = _run_deadlock(DT80, cycles=400)
    old_cycle, old_t = _legacy_deadlock(400, 1.5, CONFIGURED_RATE_HZ, DT80)
    assert new_cycle == old_cycle == 120, (new_cycle, old_cycle)
    assert abs(new_t - old_t) < TOL_S, (new_t, old_t)
    print(f"PASS deadlock  80 Hz: both fire on stalled cycle {new_cycle} (t = {new_t:.4f} s)")


def test_deadlock_does_not_stretch_at_40hz():
    new_cycle, new_t = _run_deadlock(DT40, cycles=400)
    old_cycle, old_t = _legacy_deadlock(400, 1.5, CONFIGURED_RATE_HZ, DT40)
    assert abs(new_t - 1.5) <= DT40 + TOL_S, f"new fired at {new_t:.3f} s, wanted 1.5 s"
    assert abs(old_t - 3.0) <= DT40 + TOL_S, f"legacy fired at {old_t:.3f} s, wanted 3.0 s"
    assert new_cycle < old_cycle
    print(f"PASS deadlock  40 Hz: legacy fires at {old_t:.3f} s (2x the configured 1.5), "
          f"wall-clock at {new_t:.3f} s")


# --------------------------------------------------------------------------------------------- #
# 2. FTG TIMER                                                                                    #
# --------------------------------------------------------------------------------------------- #
def _ftg_node(dt, ftg_timer_sec):
    n = _base(dt)
    n.cur_state = StateType.TRAILING
    n.cur_vs = 0.0
    n.ftg_speed_mps = 1.0
    n.ftg_timer_sec = ftg_timer_sec
    n.ftg_disabled = False
    n._ftg_slow_t0 = None
    n._check_static_trailing_deadlock = lambda: False   # has its own test above
    return n


def _legacy_ftg(cycles, ftg_timer_sec, rate_hz, dt):
    counter = 0
    for i in range(cycles):
        counter += 1
        if counter > ftg_timer_sec * rate_hz:
            return i, counter * dt
    return None, None


def _run_ftg(dt, cycles, ftg_timer_sec=3.0):
    n = _ftg_node(dt, ftg_timer_sec)
    n.cur_vs = 3.0                        # one fast cycle -> establishes the reset baseline
    n._check_ftg()
    t0 = n.now_sec()
    n.tick()
    n.cur_vs = 0.0
    for i in range(cycles):
        if n._check_ftg():
            return i, n.now_sec() - t0
        n.tick()
    return None, None


def test_ftg_is_behaviour_neutral_at_80hz():
    new_cycle, new_t = _run_ftg(DT80, cycles=600)
    old_cycle, old_t = _legacy_ftg(600, 3.0, CONFIGURED_RATE_HZ, DT80)
    assert new_cycle == old_cycle == 240, (new_cycle, old_cycle)
    assert abs(new_t - old_t) < TOL_S, (new_t, old_t)
    print(f"PASS ftg       80 Hz: both fire on slow cycle {new_cycle} (t = {new_t:.4f} s)")


def test_ftg_does_not_stretch_at_40hz():
    new_cycle, new_t = _run_ftg(DT40, cycles=600)
    old_cycle, old_t = _legacy_ftg(600, 3.0, CONFIGURED_RATE_HZ, DT40)
    assert abs(new_t - 3.0) <= DT40 + TOL_S, f"new fired at {new_t:.3f} s, wanted 3.0 s"
    assert abs(old_t - 6.0) <= DT40 + TOL_S, f"legacy fired at {old_t:.3f} s, wanted 6.0 s"
    assert new_cycle < old_cycle
    print(f"PASS ftg       40 Hz: legacy fires at {old_t:.3f} s (2x the configured 3.0), "
          f"wall-clock at {new_t:.3f} s")


# --------------------------------------------------------------------------------------------- #
# 3. OVERTAKE TTL LATCH                                                                           #
# --------------------------------------------------------------------------------------------- #
def _ot_node(dt, ttl_sec):
    n = _base(dt)
    n.cur_state = StateType.OVERTAKE
    n.overtaking_ttl_sec = ttl_sec
    n.overtaking_ttl_elapsed_sec = 0.0
    n._overtake_ttl_t0 = None
    n._check_overtaking_mode_sustainability = lambda: True
    n._check_enemy_in_front = lambda: False
    # the fallthrough leg (GlobalTrackingTransition, no obstacles, on the line) -> GB_TRACK
    n.cur_obstacles_in_interest = []
    n.recovery_exit_d_m = 0.2
    n._check_close_to_raceline = lambda *a: True
    n._check_close_to_raceline_heading = lambda *a: True
    return n


def _legacy_ot(cycles, ttl_sec, rate_hz, dt):
    """Pre-change latch: an int cycle counter, read by the transition one cycle after it is set."""
    threshold = int(ttl_sec * rate_hz)
    count, state = 0, StateType.OVERTAKE
    for i in range(cycles):
        prev = state
        state = StateType.OVERTAKE if count < threshold else StateType.GB_TRACK
        if state != StateType.OVERTAKE:
            return i, count * dt          # the latch value that failed the test, in seconds
        if prev == StateType.OVERTAKE:
            count += 1                    # no enemy in front, by construction
        else:
            count = 0
    return None, None


def _run_ot(dt, cycles, ttl_sec=3.0):
    n = _ot_node(dt, ttl_sec)
    # one reset cycle first (entering OVERTAKE), matching the legacy counter's zero
    n._update_overtake_ttl(StateType.GB_TRACK, StateType.OVERTAKE)
    n.tick()
    for i in range(cycles):
        prev = n.cur_state
        state, _src = state_transitions.OvertakingTransition(n)
        if state != StateType.OVERTAKE:
            return i, n.overtaking_ttl_elapsed_sec
        n._update_overtake_ttl(prev, state)
        n.cur_state = state
        n.tick()
    return None, None


def test_overtake_ttl_is_behaviour_neutral_at_80hz():
    new_cycle, new_t = _run_ot(DT80, cycles=600)
    old_cycle, old_t = _legacy_ot(600, 3.0, CONFIGURED_RATE_HZ, DT80)
    assert new_cycle == old_cycle == 240, (new_cycle, old_cycle)
    assert abs(new_t - old_t) < TOL_S, (new_t, old_t)
    print(f"PASS ot ttl    80 Hz: both drop OVERTAKE on cycle {new_cycle} (latch = {new_t:.4f} s)")


def test_overtake_ttl_does_not_stretch_at_40hz():
    new_cycle, new_t = _run_ot(DT40, cycles=600)
    old_cycle, old_t = _legacy_ot(600, 3.0, CONFIGURED_RATE_HZ, DT40)
    assert abs(new_t - 3.0) <= DT40 + TOL_S, f"new dropped at {new_t:.3f} s, wanted 3.0 s"
    assert abs(old_t - 6.0) <= DT40 + TOL_S, f"legacy dropped at {old_t:.3f} s, wanted 6.0 s"
    assert new_cycle < old_cycle
    print(f"PASS ot ttl    40 Hz: legacy drops at {old_t:.3f} s (2x the configured 3.0), "
          f"wall-clock at {new_t:.3f} s")


def test_overtake_ttl_resets_on_enemy_in_front():
    """The latch still measures the CURRENT no-enemy run, not total time-in-OVERTAKE."""
    n = _ot_node(DT80, 3.0)
    n._update_overtake_ttl(StateType.GB_TRACK, StateType.OVERTAKE)
    for _ in range(200):                  # 2.5 s of no-enemy time -- short of the 3.0 s ttl
        n.tick()
        n._update_overtake_ttl(StateType.OVERTAKE, StateType.OVERTAKE)
    assert abs(n.overtaking_ttl_elapsed_sec - 2.5) < TOL_S, n.overtaking_ttl_elapsed_sec
    n._check_enemy_in_front = lambda: True
    n.tick()
    n._update_overtake_ttl(StateType.OVERTAKE, StateType.OVERTAKE)
    assert n.overtaking_ttl_elapsed_sec == 0.0, n.overtaking_ttl_elapsed_sec
    print("PASS ot ttl          : an enemy reappearing in front zeroes the latch, as the counter did")


# --------------------------------------------------------------------------------------------- #
# 4. NO TIMEOUT IS DERIVED FROM rate_hz ANY MORE                                                  #
# --------------------------------------------------------------------------------------------- #
def test_rate_hz_is_only_the_timer_period():
    """Regression guard: `rate_hz` may set the timer period and feed the rate monitor, and that is
    all. Any new `* self.rate_hz` is a timeout that will stretch when the loop slows down."""
    allowed = ("1.0 / self.rate_hz", "nominal_hz=self.rate_hz", "self.rate_hz = self.params.rate_hz")
    src = (Path(__file__).resolve().parents[1] / "state_machine" / "state_machine_node.py").read_text()
    offenders = [ln.strip() for ln in src.splitlines()
                 if "rate_hz" in ln and not any(a in ln for a in allowed)]
    assert not offenders, "rate_hz used outside the timer period / rate monitor:\n  " + "\n  ".join(offenders)
    print("PASS rate_hz         : appears only as the timer period and the rate-monitor nominal")


if __name__ == "__main__":
    test_deadlock_is_behaviour_neutral_at_80hz()
    test_deadlock_does_not_stretch_at_40hz()
    test_ftg_is_behaviour_neutral_at_80hz()
    test_ftg_does_not_stretch_at_40hz()
    test_overtake_ttl_is_behaviour_neutral_at_80hz()
    test_overtake_ttl_does_not_stretch_at_40hz()
    test_overtake_ttl_resets_on_enemy_in_front()
    test_rate_hz_is_only_the_timer_period()
    print("ALL PASS")
