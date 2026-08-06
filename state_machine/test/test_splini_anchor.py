#!/usr/bin/env python3
"""Harness test of where the published avoidance window STARTS.

The reported symptom is an avoidance spline that begins at a position the car has already driven
past. Two independent mechanisms could produce it, and the old code had a branch for each:

  1. a free XY argmin over the whole path snaps to the wrong branch wherever the path runs close
     to itself -- and an avoidance path deliberately does, since its entry and exit ramps lie on
     the same raceline a few metres apart;
  2. the splice-from-start branch pinned the window to index 0 while the car was within
     splice_start_dist_m of it, which stayed true long after the car had passed that point because
     the SM adopts the republished path 0.3-0.5 s late.

Run (after sourcing the workspace):
  python3 state_machine/test/test_splini_anchor.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_machine.state_machine_node import StateMachine        # noqa: E402

TRACK_LEN = 40.0


def wd(xy, s, d=None):
    """WaypointData stand-in: array columns are [x, y, s, d]."""
    xy = np.asarray(xy, float)
    s = np.asarray(s, float)
    d = np.zeros(len(s)) if d is None else np.asarray(d, float)
    return types.SimpleNamespace(
        array=np.column_stack([xy[:, 0], xy[:, 1], s, d]),
        list=[types.SimpleNamespace(s_m=float(v), x_m=float(a), y_m=float(b))
              for v, a, b in zip(s, xy[:, 0], xy[:, 1])])


def sm(cur_s, pos):
    n = StateMachine.__new__(StateMachine)
    n.track_length = TRACK_LEN
    n.max_s = TRACK_LEN
    n.cur_s = cur_s
    n.current_position = [pos[0], pos[1], 0.0]
    return n


def hairpin_path():
    """An out-and-back path: the outbound leg (s 8..14) and the return leg (s 26..32) run 0.4 m
    apart in XY. A global argmin from a car on the outbound leg can land on the return leg."""
    s_out = np.arange(8.0, 14.0, 0.1)
    s_ret = np.arange(26.0, 32.0, 0.1)
    xy_out = np.column_stack([s_out - 8.0, np.zeros(len(s_out))])
    xy_ret = np.column_stack([(32.0 - s_ret), np.full(len(s_ret), 0.4)])
    return wd(np.vstack([xy_out, xy_ret]), np.concatenate([s_out, s_ret]))


def test_s_window_beats_a_self_close_path():
    p = hairpin_path()
    # car at s=11 on the OUTBOUND leg, x=3.0, but sitting 0.25 m toward the return leg laterally,
    # so the nearest point in raw XY belongs to the RETURN leg 15 m away in s.
    n = sm(cur_s=11.0, pos=(3.0, 0.25))
    raw = int(np.argmin(np.linalg.norm(p.array[:, 0:2] - np.array([3.0, 0.25]), axis=1)))
    assert p.array[raw, 2] > 20.0, "the harness must reproduce the wrong-branch argmin"
    idx = n._splini_anchor_index(p)
    assert 8.0 <= p.array[idx, 2] <= 14.0, \
        f"the s-window must keep the anchor on the car's own leg, got s={p.array[idx, 2]:.1f}"
    print("PASS the s-window keeps the anchor off a far branch the path runs close to")


def test_back_margin_is_small_and_clamped():
    s = np.arange(5.0, 15.0, 0.1)
    p = wd(np.column_stack([s, np.zeros(len(s))]), s)
    n = sm(cur_s=10.0, pos=(10.0, 0.0))
    idx = n._splini_anchor_index(p)
    nearest = int(np.argmin(np.abs(p.array[:, 2] - 10.0)))
    assert idx == nearest - 2, f"expected a 2-point back margin, got {nearest - idx}"
    assert abs(p.array[nearest, 2] - p.array[idx, 2]) <= 0.3, \
        "the back margin must stay inside ~0.3 m, not strand the window behind the car"
    # ...and it cannot go negative at the very start of the path
    n0 = sm(cur_s=5.0, pos=(5.0, 0.0))
    assert n0._splini_anchor_index(p) == 0
    print("PASS the back margin is 2 points, bounded, and clamped at the path start")


def test_anchor_follows_the_car_not_the_path_start():
    # THE regression: the old splice branch pinned the window to index 0 for as long as the car was
    # within splice_start_dist_m of the path start. With the SM adopting 0.3-0.5 s late, that kept
    # the first published point metres behind the car for the whole approach.
    s = np.arange(5.0, 20.0, 0.1)
    p = wd(np.column_stack([s, np.zeros(len(s))]), s)
    firsts = []
    for car_s in (5.2, 6.0, 8.0, 12.0):
        n = sm(cur_s=car_s, pos=(car_s, 0.0))
        idx = n._splini_anchor_index(p)
        firsts.append(p.array[idx, 2] - car_s)
    assert all(-0.35 <= f <= 0.0 for f in firsts), \
        f"the window must track the car, never lag behind it: {['%+.2f' % f for f in firsts]}"
    assert firsts == sorted(firsts, key=abs) or max(map(abs, firsts)) < 0.35
    print(f"PASS the anchor tracks the car ({', '.join('%+.2f m' % f for f in firsts)})")


def test_s_window_is_wrap_aware():
    # a path straddling the seam: s runs 38, 39, 0, 1, ...
    s = np.concatenate([np.arange(38.0, 40.0, 0.1), np.arange(0.0, 2.0, 0.1)])
    x = np.arange(len(s)) * 0.1
    p = wd(np.column_stack([x, np.zeros(len(s))]), s)
    n = sm(cur_s=39.5, pos=(1.5, 0.0))
    idx = n._splini_anchor_index(p)
    assert abs(((p.array[idx, 2] - 39.5 + 20.0) % 40.0) - 20.0) <= 0.35, \
        f"anchor must be beside the car across the seam, got s={p.array[idx, 2]:.2f}"
    print("PASS the s-window is wrap-aware across the start/finish seam")


def test_the_splice_joins_by_position_not_by_s():
    # The two arrays are only lined up by s when they are parameterised by the SAME line, and they
    # are not whenever static_reopt has swapped one: the cached avoidance path carries the s of the
    # line it was planned on, cur_gb_wpnts comes from the swapped one, and an obstacle-aware line
    # has a different arc length (measured +0.95 m per lap). The run log shows what that does to
    # the join: 0.595 m of step against a 0.1 m spacing, plus two of 0.303 m.
    x = np.arange(0.0, 40.0, 0.1)
    gb = wd(np.column_stack([x, np.zeros(len(x))]), x)          # global: s == x
    n = sm(cur_s=10.0, pos=(10.0, 0.0))
    n.cur_gb_wpnts = gb
    n.wpnt_dist = 0.1
    # the avoidance path ENDS at x = 12.0, but its s was stamped on a line 0.6 m longer
    tail = types.SimpleNamespace(x_m=12.0, y_m=0.0, s_m=12.6)
    # what the s question answers, correctly, about the wrong parameterisation
    s_idx = int(np.searchsorted(gb.array[:, 2], tail.s_m, side="right"))
    s_step = float(np.hypot(gb.array[s_idx, 0] - tail.x_m, gb.array[s_idx, 1] - tail.y_m))
    assert s_step > 1.5 * n.wpnt_dist, \
        f"the harness must reproduce the failure: the s-spliced join only steps {s_step:.3f} m"
    idx = n._splice_index(tail, "avoidance")
    step = float(np.hypot(gb.array[idx, 0] - tail.x_m, gb.array[idx, 1] - tail.y_m))
    assert step <= 1.5 * n.wpnt_dist, (
        f"the join steps {step:.3f} m: the padding was spliced by an s that belongs to a different "
        f"line (the s answer starts at x={gb.array[s_idx, 0]:.2f} for a path ending at "
        f"x={tail.x_m:.2f})")
    # ...and it continues FORWARD from the tail, never on top of it
    assert gb.array[idx, 0] > tail.x_m - 1e-9, "the padding restarted behind the path's end"
    print(f"PASS the splice joins by position: {step:.3f} m of step where the s answer gives "
          f"{s_step:.3f} m (budget {1.5 * n.wpnt_dist:.3f} m)")


def test_the_splice_still_works_without_geometry():
    # A tail with no x/y (an older cached array) must not crash the splice.
    x = np.arange(0.0, 40.0, 0.1)
    n = sm(cur_s=10.0, pos=(10.0, 0.0))
    n.cur_gb_wpnts = wd(np.column_stack([x, np.zeros(len(x))]), x)
    n.wpnt_dist = 0.1
    idx = n._splice_index(types.SimpleNamespace(s_m=12.0), "avoidance")
    assert 118 <= idx <= 122, idx
    print("PASS a tail with no geometry falls back to the s question")


if __name__ == "__main__":
    test_the_splice_joins_by_position_not_by_s()
    test_the_splice_still_works_without_geometry()
    test_s_window_beats_a_self_close_path()
    test_back_margin_is_small_and_clamped()
    test_anchor_follows_the_car_not_the_path_start()
    test_s_window_is_wrap_aware()
    print("ALL PASS")
