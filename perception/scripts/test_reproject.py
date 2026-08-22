#!/usr/bin/env python3
"""multi_tracking: frenet frame re-projection across a static_reopt line swap.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest perception/scripts/test_reproject.py -q

THE BUG THIS PINS. The converter was built once and never rebuilt, while the bridge upstream
rebuilds on every geometry change. From the first swap this node stored (s,d) measured in the NEW
frame and reconstructed x_m,y_m through the OLD converter. Measured on bag bias_0818_2322: kiss
held box2 at (1.69,0.19) and box3 at (5.93,1.18) all run; this node published them 0.67 m and
0.73 m away between t=20 s and t=40 s, both moving and both recovering at the same instant. Four
physical boxes became seven tracked objects.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frenet_conversion.frenet_converter import FrenetConverter   # noqa: E402
import multi_tracking as MT                                       # noqa: E402


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def line(y_off=0.0, bump_at=None, bump=0.0, n=401, length=40.0):
    """A straight line along +x, optionally with a LOCAL lateral bump — the shape a re-optimized
    line actually has. A uniform offset would be re-projectable with one ds/dd; a local one is
    exactly what makes the per-point round trip necessary."""
    x = np.linspace(0.0, length, n)
    y = np.full_like(x, y_off)
    if bump_at is not None:
        y = y + bump * np.exp(-0.5 * ((x - bump_at) / 2.0) ** 2)
    return x, y


def node(conv, track_length=40.0):
    n = MT.StaticDynamic.__new__(MT.StaticDynamic)
    n.converter = conv
    n.track_length = track_length
    n.tracked_obstacles = []
    n.get_logger = lambda: _Log()
    return n


def track(s, d, hist=None):
    t = MT.ObstacleSD.__new__(MT.ObstacleSD)
    t.measurments_s = list(hist[0]) if hist else [s]
    t.measurments_d = list(hist[1]) if hist else [d]
    t.mean = [s, d]
    t.dynamic_state = None
    return t


# =======================================================================================


def test_a_swap_moves_sd_so_that_map_position_is_preserved():
    """THE contract: (s,d) changes, map (x,y) does not."""
    x0, y0 = line()
    x1, y1 = line(bump_at=12.0, bump=0.5)        # local hump, like a re-optimized line
    old, new = FrenetConverter(x0, y0), FrenetConverter(x1, y1)
    n = node(old)
    t = track(12.0, 0.30)
    before = old.get_cartesian(t.mean[0], t.mean[1])
    n.tracked_obstacles = [t]
    n.converter = new
    n._reproject_tracks(old, 40.0)
    after = new.get_cartesian(t.mean[0], t.mean[1])
    assert math.hypot(after[0] - before[0], after[1] - before[1]) < 0.02, (before, after)
    assert abs(t.mean[1] - 0.30) > 0.05, \
        "d did not change at all -- the frames were identical, so this proves nothing"
    print(f"PASS map position preserved ({before[0]:.3f},{before[1]:.3f}) -> "
          f"({after[0]:.3f},{after[1]:.3f}); d {0.30:.2f} -> {t.mean[1]:.2f}")


def test_the_history_is_reprojected_point_by_point_not_by_one_offset():
    """A LOCAL hump means the frame difference is a function of s. One ds/dd taken at the centre
    is right there and wrong at the ends of the same track's history."""
    x0, y0 = line()
    x1, y1 = line(bump_at=12.0, bump=0.6)
    old, new = FrenetConverter(x0, y0), FrenetConverter(x1, y1)
    hs = [8.0, 12.0, 16.0]                       # spread across and outside the hump
    n = node(old); t = track(12.0, 0.0, hist=(hs, [0.0] * 3)); n.tracked_obstacles = [t]
    xy_before = [old.get_cartesian(s, 0.0) for s in hs]
    n.converter = new
    n._reproject_tracks(old, 40.0)
    for (bx, by), s_new, d_new in zip(xy_before, t.measurments_s, t.measurments_d):
        ax, ay = new.get_cartesian(s_new, d_new)
        assert math.hypot(ax - bx, ay - by) < 0.02, ((bx, by), (ax, ay))
    shifts = [abs(a - b) for a, b in zip(t.measurments_d, [0.0] * 3)]
    assert max(shifts) - min(shifts) > 0.05, (
        f"every history point moved by the same amount ({shifts}) -- a uniform offset would have "
        f"passed this, so the per-point round trip is not being exercised")
    print(f"PASS history re-projected per point; d shifts differ across it: "
          f"{[round(v,3) for v in shifts]}")


def test_reprojection_is_idempotent():
    """The bridge and this node rebuild on different messages. Re-projecting an already-correct
    (s,d) must be a no-op, so the one frame where they disagree costs nothing."""
    x0, y0 = line()
    x1, y1 = line(bump_at=12.0, bump=0.5)
    old, new = FrenetConverter(x0, y0), FrenetConverter(x1, y1)
    n = node(old); t = track(12.0, 0.30); n.tracked_obstacles = [t]
    n.converter = new
    n._reproject_tracks(old, 40.0)
    once = (t.mean[0], t.mean[1])
    n._reproject_tracks(new, 40.0)               # same frame twice
    assert abs(t.mean[0] - once[0]) < 0.01 and abs(t.mean[1] - once[1]) < 0.01, (once, t.mean)
    print(f"PASS re-projecting into the frame already held is a no-op ({once[0]:.3f},{once[1]:.3f})")


def test_two_boxes_065_m_apart_stay_two_boxes():
    """THE OPPOSITE FAILURE. box0/box1 in bias_0818_2322 sit 0.65 m apart. Whatever the swap does
    to the frame, the two must not land on top of each other."""
    x0, y0 = line()
    x1, y1 = line(bump_at=13.0, bump=0.5)
    old, new = FrenetConverter(x0, y0), FrenetConverter(x1, y1)
    a_xy = old.get_cartesian(13.2, -0.64)
    b_xy = old.get_cartesian(13.26, 0.0)
    sep_before = math.hypot(a_xy[0] - b_xy[0], a_xy[1] - b_xy[1])
    n = node(old)
    ta, tb = track(13.2, -0.64), track(13.26, 0.0)
    n.tracked_obstacles = [ta, tb]
    n.converter = new
    n._reproject_tracks(old, 40.0)
    a2 = new.get_cartesian(ta.mean[0], ta.mean[1])
    b2 = new.get_cartesian(tb.mean[0], tb.mean[1])
    sep_after = math.hypot(a2[0] - b2[0], a2[1] - b2[1])
    assert abs(sep_after - sep_before) < 0.05, (sep_before, sep_after)
    assert sep_after > 0.5, f"the two boxes closed to {sep_after:.3f} m"
    print(f"PASS 0.65 m separation survives the swap: {sep_before:.3f} -> {sep_after:.3f} m")


def test_the_kf_position_moves_and_its_position_variance_grows():
    x0, y0 = line()
    x1, y1 = line(bump_at=12.0, bump=0.5)
    old, new = FrenetConverter(x0, y0), FrenetConverter(x1, y1)

    class KF:
        x = [12.0, 1.5, 0.30, 0.1]
        P = [[0.01, 0, 0, 0], [0, 0.02, 0, 0], [0, 0, 0.01, 0], [0, 0, 0, 0.02]]

    class DS:
        isInitialised = True
        dynamic_kf = KF()

    n = node(old); t = track(12.0, 0.30); t.dynamic_state = DS(); n.tracked_obstacles = [t]
    p_before = (KF.P[0][0], KF.P[2][2])
    v_before = (KF.x[1], KF.x[3])
    n.converter = new
    n._reproject_tracks(old, 40.0)
    assert abs(t.dynamic_state.dynamic_kf.x[2] - 0.30) > 0.05, "the KF position did not move"
    assert t.dynamic_state.dynamic_kf.P[0][0] > p_before[0], "s variance was not inflated"
    assert t.dynamic_state.dynamic_kf.P[2][2] > p_before[1], "d variance was not inflated"
    assert (t.dynamic_state.dynamic_kf.x[1], t.dynamic_state.dynamic_kf.x[3]) == v_before, \
        "velocity was rotated by an angle this function never measured"
    print(f"PASS KF position re-projected, position variance inflated "
          f"{p_before[0]:.4f} -> {t.dynamic_state.dynamic_kf.P[0][0]:.4f}, velocity untouched")


if __name__ == "__main__":
    for fn in (test_a_swap_moves_sd_so_that_map_position_is_preserved,
               test_the_history_is_reprojected_point_by_point_not_by_one_offset,
               test_reprojection_is_idempotent,
               test_two_boxes_065_m_apart_stay_two_boxes,
               test_the_kf_position_moves_and_its_position_variance_grows):
        fn()
