#!/usr/bin/env python3
"""Harness test of the lane-change phase machine (IDLE -> HOLD -> CLOSE).

Loads the REAL change_avoidance_node module, binds the REAL phase-step
methods onto a bare object with hand-set state, and drives them. The point is to prove:
  1. IDLE -> HOLD happens directly, and CLOSE/IDLE transitions still work.
  2. The OPEN guards that were deleted were either unreachable or are still covered.
  3. _safe_to_abort() is the single abort predicate and is the conservative one.
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


# bind the REAL methods under test
for name in ("_to_idle", "_passed", "_safe_to_abort", "_step_hold", "_arm_close", "_sdiff",
             "_target_lost_for"):
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
check("phases are IDLE/HOLD/CLOSE",
      (cav.PHASE_IDLE, cav.PHASE_HOLD, cav.PHASE_CLOSE) == ("IDLE", "HOLD", "CLOSE"))
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

print("\n=== _to_idle resets the maneuver state ===")
f = mk(phase=cav.PHASE_HOLD, target=tgt(s=13.0), lane_s=np.zeros(3), lane_d=np.zeros(3),
       close_s=5.0, close_frozen=True, pass_cnt=4, clear_fail_cnt=3, blocked_since=1.0,
       meet_s=42.0)
f._to_idle(f.t, "test")
check("phase IDLE", f.phase == cav.PHASE_IDLE)
check("target/meet_s/lane cleared",
      f.target is None and f.meet_s is None and f.lane_s is None and f.lane_d is None)
check("counters reset", f.pass_cnt == 0 and f.clear_fail_cnt == 0 and f.blocked_since is None)
check("close state cleared", f.close_s is None and f.close_frozen is False)

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
