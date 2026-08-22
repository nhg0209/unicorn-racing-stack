#!/usr/bin/env python3
"""Can target_velocity address the whole raceline, or only the first 10 % of it?

The index was `int((x[0] * 10) % track_length)`: a STATION index taken modulo a LENGTH IN METRES.
On ifac (385 waypoints over 38.4 m) that only ever produces 0..38, so a track anywhere past the
first ~3.8 m of the lap was handed the raceline speed of a corner it is not in -- and every s
beyond that aliases onto the same 39 entries. The `* 10` is a hardcoded 0.1 m spacing, the same
defect GlobalTracking was fixed for and obs_propagation carries a note about.

This is DEAD code today: useTargetVel is hardcoded False, so the P_vs term never runs. Fixing the
arithmetic is not enabling it, and the last check here pins that it stays off.

  ~/miniforge3/envs/unicorn/bin/python3 perception/scripts/test_target_velocity_index.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MOD = REPO / "perception" / "scripts" / "multi_tracking.py"

# The module builds ROS objects at import; load it as source and pull only what we need.
src = MOD.read_text()
ns = {"__name__": "mt", "__file__": str(MOD)}
try:
    exec(compile(src, str(MOD), "exec"), ns)
except Exception as exc:                      # rclpy/msg imports are fine, node construction is not
    if not ns.get("Opponent_state"):
        raise SystemExit(f"could not load multi_tracking: {exc}")
Opponent_state = ns["Opponent_state"]
wpnt_spacing = ns["wpnt_spacing"]

RATIO = 1.0


class Wpnt:
    def __init__(self, i, dist):
        self.s_m = i * dist
        self.vx_mps = float(i)            # vx == index, so the returned speed NAMES the index used


class Kf:
    def __init__(self, s):
        self.x = [s, 0.0, 0.0, 0.0]


def state(s, n=385, dist=0.1):
    """An Opponent_state carrying s, against a line of n waypoints at `dist` spacing."""
    wpnts = [Wpnt(i, dist) for i in range(n)]
    Opponent_state.waypoints = wpnts
    Opponent_state.wpnt_dist = wpnt_spacing(wpnts)
    Opponent_state.track_length = wpnts[-1].s_m
    Opponent_state.ratio_to_glob_path = RATIO
    o = Opponent_state.__new__(Opponent_state)
    o.dynamic_kf = Kf(s)
    return o


def old_index(s, n=385, dist=0.1):
    """What the code did before, for the direction of the change to be visible."""
    return int((s * 10) % ((n - 1) * dist))


def test_a_the_end_of_the_track_indexes_the_last_waypoints():
    # ifac: 385 stations, 0.1 m apart, track_length 38.4 m. 38.4 is also the float trap:
    # 38.4 // 0.1 is 383.0, so truncation loses the last station.
    for s, want in ((38.4, 384), (38.37, 384), (38.32, 383), (38.0, 380), (20.0, 200), (0.0, 0)):
        got = state(s).target_velocity()
        assert got == want, f"s={s}: index {got}, expected {want}"
        assert old_index(s) <= 38, f"old formula reached {old_index(s)}; update this test"
    # ...and normalize_s's negative representation of "just before the seam" belongs at the END
    got = state(-0.12).target_velocity()
    assert got == 384, f"s=-0.12 (0.12 m before the seam) indexed {got}, expected 384"
    assert old_index(-0.12) == 37, f"the old formula sent it to waypoint {old_index(-0.12)}"


def test_b_every_waypoint_is_reachable():
    n, dist = 385, 0.1
    reached = {int(state(i * dist, n, dist).target_velocity()) for i in range(n)}
    assert reached == set(range(n)), (
        f"only {len(reached)} of {n} waypoints are addressable, max {max(reached)}")
    old = {old_index(i * dist, n, dist) for i in range(n)}
    assert len(old) == 39, f"the old formula reached {len(old)} of {n}; update this test"


def test_c_the_spacing_is_measured_not_assumed():
    # a map published at 0.2 m: the same s must land on half the index, which `* 10` cannot do
    n, dist = 200, 0.2
    for s, want in ((0.0, 0), (10.0, 50), (39.8, 199)):
        got = state(s, n, dist).target_velocity()
        assert got == want, f"{dist} m spacing, s={s}: index {got}, expected {want}"
    assert wpnt_spacing([Wpnt(i, dist) for i in range(n)]) == dist
    assert wpnt_spacing([]) is None and wpnt_spacing(None) is None


def test_d_no_global_line_yet_zeroes_the_term_instead_of_indexing_nothing():
    o = state(10.0)
    Opponent_state.waypoints, Opponent_state.wpnt_dist = None, None
    o.dynamic_kf.x[1] = 3.7
    # P_vs * (target - x[1]) must come out 0, not raise and not command a phantom speed
    assert o.target_velocity() == 3.7


def test_e_usetargetvel_is_still_off():
    # the fix is arithmetic; enabling the term is a separate decision and is NOT taken here
    assert "self.useTargetVel = False" in src, "useTargetVel is no longer initialised False"
    assert "useTargetVel = True" not in src.replace(
        "tracked_obstacle.dynamic_state.useTargetVel = True", ""), (
        "something now sets useTargetVel True outside the ttl-death branch")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
