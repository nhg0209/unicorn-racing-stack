#!/usr/bin/env python3
"""Does the static/dynamic vote survive a stationary box's centroid wobble WITHOUT letting a
moving opponent through?

The veto that decides whether a track may vote static was a SPEED: net displacement over the
measurement window, divided by the window's duration. On a young window (min_nb_meas / rate ~
0.15 s) that divisor is small enough that a stationary 0.3-0.6 m box whose centroid shifts 0.06 m
-- which it does routinely on approach, as the visible face changes -- reads 0.40 m/s and is
vetoed. Since the tracker recreates its tracks around every lap, that misreading arrived once per
approach: the box spent its first frames classified dynamic, the static planner dropped it, and
the state machine fell out of the overtake into TRAILING.

The floor must not let a real opponent vote static. At 2 m/s a car covers 0.30 m in the same
window, so it stays vetoed -- that is what this file checks, in both directions.

  ~/miniforge3/envs/unicorn/bin/python3 perception/scripts/test_static_classification.py
"""
import math
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MOD = REPO / "perception" / "scripts" / "multi_tracking.py"

# The module builds ROS objects at import; load it as source and pull only what we need.
src = MOD.read_text()
ns = {"__name__": "mt", "__file__": str(MOD)}
try:
    exec(compile(src, str(MOD), "exec"), ns)
except Exception as exc:                      # rclpy/msg imports are fine, node construction is not
    if not ns.get("ObstacleSD"):
        raise SystemExit(f"could not load multi_tracking: {exc}")
ObstacleSD = ns["ObstacleSD"]

RATE = 40.0
MIN_NB = 6
TRACK_LEN = 36.6


def configure(net_floor=0.12, speed_max=0.4):
    ObstacleSD.rate = RATE
    ObstacleSD.min_nb_meas = MIN_NB
    ObstacleSD.static_speed_max = speed_max
    ObstacleSD.static_net_floor = net_floor
    ObstacleSD.min_std = 0.16
    ObstacleSD.max_std = 0.2
    ObstacleSD.demote_speed = 1.0


def track(s_series, d_series):
    """An ObstacleSD carrying a given measurement window, without touching ROS."""
    o = ObstacleSD.__new__(ObstacleSD)
    o.measurments_s = list(s_series)
    o.measurments_d = list(d_series)
    o.measurment_times = [i / RATE for i in range(len(s_series))]
    o.nb_meas = len(s_series) + MIN_NB      # past min_nb_meas, so isStatic actually votes
    o.mean = (sum(s_series) / len(s_series), sum(d_series) / len(d_series))
    o.static_count = 0
    o.total_count = 0
    o.staticFlag = None
    o.near_dynamic = False
    o.size = 0.4
    return o


def votes_static(o, n=12):
    for _ in range(n):
        o.isStatic(TRACK_LEN)
    return bool(o.staticFlag)


def test_a_stationary_box_with_a_wobbling_centroid_votes_static():
    # 0.06 m of centroid shift over a 6-sample (0.125 s) window = 0.48 m/s by the old speed test,
    # over the 0.40 m/s veto -> could not vote static. Its NET displacement is 0.06 m, well under
    # the 0.12 m floor.
    configure()
    vetoed = []
    for net in (0.06, 0.08, 0.10):
        s = [10.0 + net * i / (MIN_NB - 1) for i in range(MIN_NB)]
        o = track(s, [0.0] * MIN_NB)
        win_v = o.window_speed(TRACK_LEN)
        assert win_v > 0.4, f"the harness must reproduce the speed veto (got {win_v:.2f} m/s)"
        vetoed.append(win_v)
        assert votes_static(o), f"a stationary box wobbling {net:.2f} m must vote static"
    print(f"PASS a stationary box wobbling 0.06-0.10 m votes static "
          f"(the old speed test read {min(vetoed):.2f}-{max(vetoed):.2f} m/s, all vetoed)")


def test_a_moving_opponent_is_still_vetoed():
    # THE regression this floor could have caused. 1.0-4.0 m/s over the same window.
    configure()
    for v in (1.0, 1.5, 2.0, 3.0, 4.0):
        win_t = (MIN_NB - 1) / RATE
        net = v * win_t
        s = [10.0 + net * i / (MIN_NB - 1) for i in range(MIN_NB)]
        o = track(s, [0.0] * MIN_NB)
        assert net > 0.12, f"{v} m/s must move further than the floor (got {net:.3f} m)"
        assert not votes_static(o), f"an opponent at {v:.1f} m/s must NOT vote static"
    print("PASS a moving opponent at 1.0-4.0 m/s is still vetoed (net 0.16-0.63 m > 0.12 floor)")


def test_the_floor_is_where_it_says_it_is():
    # Either side of 0.12 m, on the same window.
    configure()
    for net, want in ((0.115, True), (0.125, False)):
        s = [10.0 + net * i / (MIN_NB - 1) for i in range(MIN_NB)]
        o = track(s, [0.0] * MIN_NB)
        assert votes_static(o) is want, f"net {net:.3f} m should vote static={want}"
    print("PASS the boundary sits at the configured 0.12 m of net displacement")


def test_a_long_window_still_uses_the_speed_limit():
    # The floor is a FLOOR, not a replacement: over a long window the speed limit is the larger of
    # the two and still governs, so a slow crawl is not laundered into "static" by taking longer.
    configure()
    n = 60                                     # 1.5 s of window
    win_t = (n - 1) / RATE
    net = 0.35 * win_t                         # 0.35 m/s: under the 0.40 veto -> static
    s = [10.0 + net * i / (n - 1) for i in range(n)]
    assert net > 0.12, "this window's net is past the floor, so the SPEED must be what decides"
    assert votes_static(track(s, [0.0] * n)), "0.35 m/s over a long window is still static"
    net2 = 0.60 * win_t                        # 0.60 m/s: over the veto -> dynamic
    s2 = [10.0 + net2 * i / (n - 1) for i in range(n)]
    assert not votes_static(track(s2, [0.0] * n)), "0.60 m/s must stay vetoed"
    print("PASS over a long window the speed limit still governs (0.35 static, 0.60 vetoed)")


if __name__ == "__main__":
    test_a_stationary_box_with_a_wobbling_centroid_votes_static()
    test_a_moving_opponent_is_still_vetoed()
    test_the_floor_is_where_it_says_it_is()
    test_a_long_window_still_uses_the_speed_limit()
    print("ALL PASS")
