#!/usr/bin/env python3
"""Harness test of the lane-change phase machine (IDLE -> ENTRY -> HOLD -> CLOSE).

Loads the REAL change_avoidance_node module, binds the REAL phase-step
methods onto a bare object with hand-set state, and drives them. The point is to prove:
  1. The committed lane is solved ONCE and HOLD never re-solves it.
  2. Only the explicit staleness triggers re-plan it, each with a logged reason.
  3. ENTRY's completion test is meaningful (it is, only because the lane is frozen).
  4. _safe_to_abort() is the single abort predicate and is the conservative one.
Run (after sourcing the workspace):
  python3 planner/lane_change_planner/test/test_phase_machine.py
"""
import sys
import types
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parents[1] / "lane_change_planner" / "change_avoidance_node.py"


# No stubs needed: every dependency is installed and importable once the workspace is
# sourced. We exec the module rather than importing the package so no ROS node is constructed.
src = MOD.read_text()
cav = types.ModuleType("cav")
cav.__dict__["__file__"] = str(MOD)
exec(compile(src, str(MOD), "exec"), cav.__dict__)


# ---------------------------------------------------------------- fake node
class Log:
    def __init__(self):
        self.lines = []

    def info(self, m, **k):
        self.lines.append(("info", m))

    def warn(self, m, **k):
        self.lines.append(("warn", m))

    def error(self, m, **k):
        self.lines.append(("error", m))


class Fake:
    """Bare object carrying just the state the phase steps touch."""

    def __init__(self):
        self._log = Log()
        self.L = 100.0
        self.scaled_max_s = self.L
        self.phase = cav.PHASE_IDLE
        self.target = None
        self.current_s, self.current_d, self.current_vs = 10.0, 0.0, 4.0
        self.meet_s = None
        self.lane_s = None
        self.lane_d = None
        self.lane_sign = 0
        self.side = None
        self.lane_offset_cur = 0.0
        self.close_s = None
        self.close_frozen = False
        self.pass_cnt = 0
        self.clear_fail_cnt = 0
        self.blocked_since = None
        self.idle_until = 0.0
        self._dt = 0.05
        self.obs_all, self.obs_dynamic = [], []
        # tunables
        self.pass_gap_m = 1.2
        self.pass_hyst_s = 0.3
        self.engage_gap_m = 5.0
        self.target_lost_s = 1.0
        self.reengage_block_s = 1.0
        self.rate_hz = 20.0
        self.hold_clear_check = True
        self.hold_clear_fail_s = 0.15
        self.lane_commit = True
        self.entry_join_tol_m = 0.15
        self.commit_dev_max_m = 0.35
        self.commit_meet_ds_m = 1.5
        self.commit_obs_dd_m = 0.40
        self.commit_horizon_m = 30.0
        self.commit_meet_s = None
        self.commit_target_d = None
        self.engage_gap_m = 5.0
        self.solve_calls = 0
        self.lane_s = np.arange(0.0, 30.0, 0.25) + 10.0
        self.lane_d = np.zeros_like(self.lane_s)
        self.meet_raw = 15.0
        self.close_arm_m = 1.0
        self._sep_monitor_m = 0.37
        # call recorders
        self.published, self.built, self.closed = [], [], []
        self.build_returns = {"d_arr": np.zeros(220), "s_lin": np.arange(220) * 0.1, "cs_u": 0.0, "ce": 0.0}
        self.publish_ok = True
        self.clear_ok = True
        self.blocked = False

    def get_logger(self):
        return self._log

    def now_sec(self):
        return self.t

    # stubs for collaborators the phase steps call
    def _update_offset(self, dt):
        pass

    def _build_path(self, closing=False):
        self.built.append(closing)
        return self.build_returns

    def _publish_path(self, path):
        self.published.append(path)
        return self.publish_ok

    def _check_lane_clear_vs_target(self, path):
        return self.clear_ok

    def _path_blocked_ahead(self, path, now):
        return self.blocked

    def _visualize_phase(self):
        pass

    def _clear_markers(self):
        pass

    def _step_close(self, now):
        self.closed.append(now)

    # collaborators the committed-lane logic calls
    def _choose_lane(self, keep_side=False):
        self.solve_calls += 1
        self.commit_meet_s = self.meet_raw
        self.commit_target_d = self.target['d'] if self.target else 0.0
        return True

    def _meet_s_raw(self):
        return self.meet_raw

    def _lane_at(self, s):
        return 0.0


# bind the REAL methods under test
for name in ("_to_idle", "_passed", "_safe_to_abort", "_step_hold", "_arm_close", "_sdiff",
             "_target_lost_for", "_commit_stale", "_abort_checks", "_step_entry"):
    setattr(Fake, name, getattr(cav.ChangeAvoidanceNode, name))


def mk(**kw):
    f = Fake()
    f.t = 100.0
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def tgt(s, vs=2.0, last_seen=100.0, d=0.0):
    return dict(id=1, s=s, d=d, vs=vs, vd=0.0, size=0.3, last_seen=last_seen)


FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


print("=== phase constants ===")
check("PHASE_OPEN is gone", not hasattr(cav, "PHASE_OPEN"))
check("phases are IDLE/ENTRY/HOLD/CLOSE",
      (cav.PHASE_IDLE, cav.PHASE_ENTRY, cav.PHASE_HOLD, cav.PHASE_CLOSE)
      == ("IDLE", "ENTRY", "HOLD", "CLOSE"))
check("_step_open is gone", not hasattr(cav.ChangeAvoidanceNode, "_step_open"))

print("\n=== _safe_to_abort: single, conservative predicate ===")
# pass_gap_m + 0.5 = 1.7 -> abort only when more than 1.7 m BEHIND
f = mk(target=tgt(s=12.0), current_s=10.0)          # 2.0 m behind
check("2.0 m behind -> safe to abort", f._safe_to_abort() is True)
f = mk(target=tgt(s=11.0), current_s=10.0)          # 1.0 m behind
check("1.0 m behind -> NOT safe (old width_car*2+0.6=1.2 would also refuse)", f._safe_to_abort() is False)
f = mk(target=tgt(s=11.4), current_s=10.0)          # 1.4 m behind: old -1.2 allowed, new -1.7 refuses
check("1.4 m behind -> NOT safe (conservative wins over the old 1.2 m rule)",
      f._safe_to_abort() is False)
f = mk(target=None)
check("no target -> safe", f._safe_to_abort() is True)

print("\n=== HOLD: normal cycle publishes ===")
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0), current_s=10.0)
f._step_hold(f.t)
check("stays in HOLD", f.phase == cav.PHASE_HOLD)
check("built a non-closing path", f.built == [False])
check("published once", len(f.published) == 1)

print("\n=== HOLD: target pulled away (inherited from deleted OPEN) ===")
# rel < -(engage_gap_m + 4.0) = -9.0  -> target 10 m ahead
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=20.0), current_s=10.0)
f._step_hold(f.t)
check("-> IDLE", f.phase == cav.PHASE_IDLE)
check("reason logged", any("target pulled away" in m for _, m in f._log.lines),
      str(f._log.lines))
check("did NOT publish", f.published == [])
check("re-engage blocked for reengage_block_s", f.idle_until == f.t + 1.0)

print("\n=== HOLD: pass detected -> CLOSE after pass_hyst_s ===")
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=8.0, vs=2.0), current_s=10.0, current_vs=4.0)
need = max(int(0.3 * 20), 1)
for i in range(need):
    f._step_hold(f.t)
    f.t += 0.05
check(f"CLOSE armed after {need} cycles", f.phase == cav.PHASE_CLOSE, f"phase={f.phase}")
check("_step_close called in the same cycle (no publish gap)", len(f.closed) == 1)

print("\n=== HOLD: target gone for 3*target_lost_s -> CLOSE ===")
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0, last_seen=100.0), current_s=10.0)
f.t = 100.0 + 3.5
f._step_hold(f.t)
check("-> CLOSE", f.phase == cav.PHASE_CLOSE)
check("reason 'target gone'", any("target gone" in m for _, m in f._log.lines))

print("\n=== HOLD: infeasible path -> IDLE ===")
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0), current_s=10.0, publish_ok=False)
f._step_hold(f.t)
check("-> IDLE", f.phase == cav.PHASE_IDLE)
check("reason 'path infeasible'", any("path infeasible" in m for _, m in f._log.lines))

print("\n=== HOLD: blocked ahead uses _safe_to_abort, not the old -1.2 m ===")
# 1.4 m behind: the OLD rule (-1.2) would abort here, the new one must NOT
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=11.4), current_s=10.0, blocked=True)
f._step_hold(f.t)
check("blocked but only 1.4 m behind -> keeps holding (does not steer back alongside)",
      f.phase == cav.PHASE_HOLD, f"phase={f.phase}")
# 2.5 m behind -> abort allowed
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=12.5), current_s=10.0, blocked=True)
f._step_hold(f.t)
check("blocked and 2.5 m behind -> IDLE", f.phase == cav.PHASE_IDLE)
check("reason 'blocked, dropping back'", any("blocked, dropping back" in m for _, m in f._log.lines))

print("\n=== HOLD: per-cycle target-clearance re-verification ===")
need_clear = max(int(0.15 * 20), 1)
# clearance fails while safely behind (2.5 m) -> abort after the dwell, not before
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=12.5), current_s=10.0, clear_ok=False)
for i in range(need_clear - 1):
    f._step_hold(f.t)
check(f"no abort before the {need_clear}-cycle dwell", f.phase == cav.PHASE_HOLD,
      f"phase={f.phase} cnt={f.clear_fail_cnt}")
check("kept publishing while debouncing", len(f.published) == need_clear - 1)
f._step_hold(f.t)
check("aborts on the dwell cycle", f.phase == cav.PHASE_IDLE, f"phase={f.phase}")
check("reason 'lane no longer clears target'",
      any("lane no longer clears target" in m for _, m in f._log.lines))
check("the failing lane was NOT published",
      len(f.published) == need_clear - 1, f"published={len(f.published)}")

# clearance recovers before the dwell expires -> counter resets, no abort
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=12.5), current_s=10.0, clear_ok=False)
f._step_hold(f.t)
f._step_hold(f.t)
f.clear_ok = True
f._step_hold(f.t)
check("counter resets when clearance recovers", f.clear_fail_cnt == 0)
check("still HOLD", f.phase == cav.PHASE_HOLD)

# clearance fails while ALONGSIDE (1.0 m behind) -> hold the lane, do not steer back
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=11.0), current_s=10.0, clear_ok=False)
for i in range(need_clear + 3):
    f._step_hold(f.t)
check("alongside: never aborts", f.phase == cav.PHASE_HOLD, f"phase={f.phase}")
check("alongside: warns instead", any(lvl == "warn" and "too close alongside" in m
                                      for lvl, m in f._log.lines))
check("alongside: keeps publishing the lane", len(f.published) == need_clear + 3)

# toggle off -> old behaviour, never checked
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=12.5), current_s=10.0, clear_ok=False,
       hold_clear_check=False)
for i in range(need_clear + 3):
    f._step_hold(f.t)
check("hold_clear_check=False reproduces the old behaviour", f.phase == cav.PHASE_HOLD)
check("hold_clear_check=False leaves the counter untouched", f.clear_fail_cnt == 0)

print("\n=== dwell counts derive from rate_hz, not a hardcoded 20 ===")
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=8.0, vs=2.0), current_s=10.0, current_vs=4.0,
       rate_hz=40.0)
for i in range(max(int(0.3 * 40), 1) - 1):
    f._step_hold(f.t)
check("at 40 Hz the pass dwell needs 12 cycles, not 6", f.phase == cav.PHASE_HOLD,
      f"phase={f.phase} pass_cnt={f.pass_cnt}")
f._step_hold(f.t)
check("...and fires on the 12th", f.phase == cav.PHASE_CLOSE, f"phase={f.phase}")

print("\n=== committed lane: HOLD must not re-solve ===")
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0), current_s=10.0)
f.commit_meet_s, f.commit_target_d = 15.0, 0.0
lane_before = f.lane_d.copy()
for i in range(20):
    f._step_hold(f.t)
check("20 HOLD cycles trigger 0 lane solves", f.solve_calls == 0, f"solves={f.solve_calls}")
check("committed lane bytes unchanged", np.array_equal(f.lane_d, lane_before))
check("still publishing every cycle", len(f.published) == 20)

print("\n=== committed lane: explicit re-plan triggers ===")
# (a) opponent drifts laterally beyond commit_obs_dd_m
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0, d=0.0), current_s=10.0)
f.commit_meet_s, f.commit_target_d = 15.0, 0.0
f._step_hold(f.t)
check("no re-plan while the opponent holds its line", f.solve_calls == 0)
f.target['d'] = 0.5                      # > commit_obs_dd_m = 0.40
f._step_hold(f.t)
check("opponent lateral drift -> exactly one re-plan", f.solve_calls == 1)
check("re-plan reason logged", any("HOLD re-plan" in m and "laterally" in m
                                   for _, m in f._log.lines), str(f._log.lines[-2:]))
# (b) meeting point drifts
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0), current_s=10.0)
f.commit_meet_s, f.commit_target_d = 15.0, 0.0
f.meet_raw = 17.0                        # 2.0 m > commit_meet_ds_m = 1.5
f._step_hold(f.t)
check("meeting-point drift -> re-plan", f.solve_calls == 1)
check("reason names the drift", any("meeting point drifted" in m for _, m in f._log.lines))
# (c) car falls off the committed lane -> re-plan AND drop back to ENTRY
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0), current_s=10.0, current_d=0.5)
f.commit_meet_s, f.commit_target_d = 15.0, 0.0
f._step_hold(f.t)
check("car off the lane -> re-plan", f.solve_calls == 1)
check("...and re-enters ENTRY to blend back on", f.phase == cav.PHASE_ENTRY, f"phase={f.phase}")

print("\n=== ENTRY: completion test is meaningful (frozen lane) ===")
f = mk(phase=cav.PHASE_ENTRY, target=tgt(s=13.0), current_s=10.0, current_d=0.5)
f.commit_meet_s, f.commit_target_d = 15.0, 0.0
f._step_entry(f.t)
check("0.50 m off the lane -> stays in ENTRY", f.phase == cav.PHASE_ENTRY)
check("ENTRY publishes", len(f.published) == 1)
f.current_d = 0.10                        # < entry_join_tol_m = 0.15
f._step_entry(f.t)
check("0.10 m off the lane -> ENTRY complete", f.phase == cav.PHASE_HOLD, f"phase={f.phase}")
check("transition logged", any("ENTRY -> HOLD (on lane" in m for _, m in f._log.lines))
check("ENTRY never re-solved a valid commit", f.solve_calls == 0)

print("\n=== ENTRY aborts while safely behind ===")
f = mk(phase=cav.PHASE_ENTRY, target=tgt(s=12.5), current_s=10.0, current_d=0.5, clear_ok=False)
f.commit_meet_s, f.commit_target_d = 15.0, 0.0
f._step_entry(f.t)
check("entry lane not clearing target -> IDLE", f.phase == cav.PHASE_IDLE)
check("reason logged", any("entry lane does not clear the target" in m for _, m in f._log.lines))
check("failing entry path never published", f.published == [])

print("\n=== lane_commit=False restores the legacy per-cycle re-solve ===")
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0), current_s=10.0, lane_commit=False)
f.offset_calls = 0
f._update_offset = lambda dt: setattr(f, "offset_calls", f.offset_calls + 1)
for i in range(5):
    f._step_hold(f.t)
check("legacy mode re-solves every cycle", f.offset_calls == 5, f"calls={f.offset_calls}")
check("legacy mode never uses the commit triggers", f.solve_calls == 0)

print("\n=== _to_idle resets the maneuver state ===")
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0), lane_s=np.zeros(3), lane_d=np.zeros(3),
       close_s=5.0, close_frozen=True, pass_cnt=4, clear_fail_cnt=3, blocked_since=1.0,
       meet_s=42.0)
f.commit_meet_s, f.commit_target_d = 15.0, 0.0
f._to_idle(f.t, "test")
check("phase IDLE", f.phase == cav.PHASE_IDLE)
check("target/meet_s/lane cleared",
      f.target is None and f.meet_s is None and f.lane_s is None and f.lane_d is None)
check("counters reset", f.pass_cnt == 0 and f.clear_fail_cnt == 0 and f.blocked_since is None)
check("close state cleared", f.close_s is None and f.close_frozen is False)
check("commit refs cleared", f.commit_meet_s is None and f.commit_target_d is None)

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
