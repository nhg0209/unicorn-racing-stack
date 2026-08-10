#!/usr/bin/env python3
"""Can the lane-change planner engage on something that is not moving?

It could, and it did. In a sim run with NO opponent vehicle at all (opponent_vehicle has_opp=False,
four static boxes) the planner engaged twice on stationary boxes:

  382.981  IDLE -> ENTRY (engage): target id=2  side=left  offset=0.68  v_opp=0.0
  391.819  IDLE -> ENTRY (engage): target id=18 side=right offset=0.58  v_opp=0.2

Two independent reasons it got there, and BOTH have to be closed by the same test:

  1. `obs_cb` splits on `not o.is_static`, and that is the tracker's position-persistence verdict,
     which is False on every freshly created track (staticFlag stays None until min_nb_meas). So
     "not is_static" also means "just appeared", and a parked box arrives as a dynamic obstacle.
  2. The one speed condition in the gate, `engage_min_closing_mps`, is a CLOSING speed:
     ego - o.vs is at its LARGEST when the obstacle is standing still. A parked box does not
     merely slip past that gate, it passes it maximally. No test of absolute speed existed.

At 391.819 the static planner was avoiding the SAME box the other way (390.464 avoid LEFT
d_end=+1.00 vs lane_change side=right offset=0.58). The SM adopted the dynamic path at 393.680, the
reference stepped 1.58 m, the steering saturated 15 cycles running (394.590-394.993) and the car hit
the wall (396.092 iTTC, 398.201 respawn).

The fix mirrors static_avoidance_node._track_near_zero from the other side, so the two planners
cannot both claim one obstacle. The last two checks here are the ones that matter for racing: a real
opponent must still be overtaken, and the mirror must stay a mirror.

  ~/miniforge3/envs/unicorn/bin/python3 planner/lane_change_planner/test/test_engage_gate.py
"""
import re
import sys
import types
from pathlib import Path

import yaml

MOD = Path(__file__).resolve().parents[1] / "lane_change_planner" / "change_avoidance_node.py"
REPO = Path(__file__).resolve().parents[3]

src = MOD.read_text()
cav = types.ModuleType("cav")
cav.__dict__["__file__"] = str(MOD)
exec(compile(src, str(MOD), "exec"), cav.__dict__)


class Log:
    def __init__(self):
        self.lines = []

    def info(self, m, **k):
        self.lines.append(m)

    warn = error = info


class Fake:
    """Only the state the engage gate and the moving-tracker touch."""

    def __init__(self):
        self._log = Log()
        self.t = 100.0
        self.scaled_max_s = 100.0
        self.current_s, self.current_d, self.current_vs = 10.0, 0.0, 4.0
        self.obs_all, self.obs_dynamic = [], []
        self._moving_since = {}
        # the shipped gate values (lane_change_params.yaml)
        self.engage_gap_m = 8.0
        self.engage_gap_min_m = 1.5
        self.obs_traj_tresh = 1.0
        self.engage_min_closing_mps = -0.5
        self.engage_min_vs_mps = 0.35
        self.engage_moving_s = 0.3

    def get_logger(self):
        return self._log

    def now_sec(self):
        return self.t


for name in ("_pick_engage_target", "_track_moving", "_moving_for", "_is_overtakable",
             "_engage_reject_reason", "_sdiff"):
    setattr(Fake, name, getattr(cav.ChangeAvoidanceNode, name))


def obs(oid, s, vs, d=0.0, vd=0.0, is_static=False, size=0.3, visible=True):
    return types.SimpleNamespace(id=oid, s_center=s, d_center=d, vs=vs, vd=vd, size=size,
                                 is_static=is_static, is_visible=visible)


def feed(f, obstacles, secs, dt=0.05):
    """Run the tracker over `secs` of cycles at 20 Hz, as obs_cb would."""
    n = max(1, int(round(secs / dt)))
    for _ in range(n):
        f.t += dt
        f.obs_all = list(obstacles)
        f.obs_dynamic = [o for o in obstacles if not o.is_static]
        f._track_moving(f.obs_dynamic)
    return f


FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def case_the_two_engagements_from_the_crash_run_are_refused():
    # id=2 at v_opp=0.0 and id=18 at v_opp=0.2, both 3-4 m ahead, exactly as logged
    for oid, v in ((2, 0.0), (18, 0.2)):
        f = feed(Fake(), [obs(oid, 13.5, v)], secs=5.0)     # 5 s of standing still: no doubt left
        tgt = f._pick_engage_target()
        check(f"crash run: id={oid} v_opp={v} is refused", tgt is None,
              f"engaged anyway: {tgt}")
        why = f._engage_reject_reason()
        check(f"crash run: id={oid} refusal is named in the diagnostic",
              "moving_for" in why and "NO" in why, why)


def case_a_real_opponent_is_still_overtaken():
    """THE CASE THAT MUST NOT REGRESS. A racing opponent, and a slow one."""
    for v in (0.5, 1.0, 2.0, 4.0):
        f = feed(Fake(), [obs(7, 13.5, v)], secs=1.0)
        check(f"a real opponent at {v} m/s is still engageable",
              f._pick_engage_target() is not None, f._engage_reject_reason())


def case_the_dwell_is_what_separates_parked_from_just_seen():
    # A moving opponent is not engageable on its FIRST frame -- speed alone cannot tell it from a
    # freshly created track whose KF has not spun up -- and becomes engageable once it holds.
    f = feed(Fake(), [obs(7, 13.5, 2.0)], secs=0.05)
    check("first frame of a moving obstacle is not yet a target",
          f._pick_engage_target() is None)
    f = feed(f, [obs(7, 13.5, 2.0)], secs=0.30)
    check("...and it is one after engage_moving_s", f._pick_engage_target() is not None)


def case_a_noise_spike_does_not_arm_the_gate():
    """A parked box whose tracked speed brushes past the floor on one frame must not become a
    target -- the reading is continuous, so a drop back under resets it."""
    o_park, o_spike = obs(3, 13.5, 0.0), obs(3, 13.5, 0.9)
    f = Fake()
    for _ in range(8):                      # 0.4 s of alternating spike / zero
        feed(f, [o_spike], secs=0.05)
        feed(f, [o_park], secs=0.05)
    check("an alternating noise spike never accumulates the dwell",
          f._pick_engage_target() is None, f"moving_for={f._moving_for(o_park):.2f}")


def case_vd_counts_as_motion_too():
    # a car crossing the raceline is moving even when its vs matches the ego's
    f = feed(Fake(), [obs(9, 13.5, 0.0, vd=1.2)], secs=1.0)
    check("lateral motion alone makes an obstacle engageable",
          f._pick_engage_target() is not None, f._engage_reject_reason())


def case_the_bookkeeping_does_not_grow_without_bound():
    f = feed(Fake(), [obs(i, 13.5, 2.0) for i in range(5)], secs=1.0)
    check("all five tracks are remembered while present", len(f._moving_since) == 5)
    f = feed(f, [obs(0, 13.5, 2.0)], secs=0.05)
    check("ids the tracker stopped publishing are dropped", set(f._moving_since) == {0},
          str(set(f._moving_since)))


def case_the_two_planners_cannot_both_claim_one_obstacle():
    """The mirror, asserted against the OTHER planner's shipped yaml rather than a copy of it.

    static_avoidance treats a dynamic-flagged obstacle as STATIC below static_near_zero_mps; this
    planner may engage one above engage_min_vs_mps. Overlap = a static keep-out one way and a lane
    change the other way, on the same box. A gap resolves to TRAILING and is deliberate.
    """
    lc = yaml.safe_load((REPO / "stack_master" / "config" / "lane_change_params.yaml").read_text())
    sa = yaml.safe_load((REPO / "stack_master" / "config"
                         / "static_avoidance_params.yaml").read_text())
    engage_v = float(lc["planner_change"]["ros__parameters"]["engage_min_vs_mps"])
    near_zero = float(sa["static_avoidance_planner"]["ros__parameters"]["static_near_zero_mps"])
    check("engage_min_vs_mps >= static_near_zero_mps (no shared obstacle)",
          engage_v >= near_zero, f"{engage_v} < {near_zero}")
    # ...and the node's own in-code seed must agree, since `ros2 run` with no yaml gets that one
    seed = re.search(r"self\.engage_min_vs_mps = ([\d.]+)", src)
    check("the node's in-code default agrees with the yaml",
          seed is not None and abs(float(seed.group(1)) - engage_v) < 1e-9,
          f"seed={seed.group(1) if seed else None} yaml={engage_v}")


def case_a_static_flagged_obstacle_was_never_a_candidate():
    # the pre-existing filter still does its half of the job
    f = feed(Fake(), [obs(4, 13.5, 2.0, is_static=True)], secs=1.0)
    check("is_static=True is not in obs_dynamic at all", f._pick_engage_target() is None)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


def case_engage_gate():
    """pytest entry point: the whole file is one gate."""
    assert main() == 0, FAILS


if __name__ == "__main__":
    sys.exit(main())
