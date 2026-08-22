#!/usr/bin/env python3
"""Left/right must come from the centerline's geometry, not from contour winding or a flag.

THE DEFECT THIS PINS. global_planner_node labelled the bounds by picking the outer contour by
LENGTH and comparing its local traversal direction to the start pose, then relied on
dist_to_bounds swapping its return when reverse_mapping was set. Contour winding is cv2's, not the
driver's; the centerline is flipped up to twice before the decision; and nothing tied the two
together. Maps f (402 stations against 0) and ifac_0807 (461 against 1) shipped with d_right and
d_left exchanged, which mirrors every corridor and makes a box on a straight read as unavoidable.

These tests do NOT re-run the generator -- it needs a live map-editor session. They exercise the
side-assignment function the generator now calls, on the real map images and centerlines, and
assert the two properties the old decision lacked: it agrees with the geometry on every sampled
station, and it FOLLOWS the centerline when the centerline is reversed.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "planner", "gb_optimizer"))

from gb_optimizer.track_bounds import (assign_sides, load_contours,  # noqa: E402
                                       score, truth_left_right, verdict)

MAPS = os.path.join(REPO, "stack_master", "maps")


def _centerline(name):
    import csv
    with open(os.path.join(MAPS, name, "centerline.csv")) as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].lstrip().startswith("#")]
    try:
        float(rows[0][0])
    except ValueError:
        rows = rows[1:]
    a = np.array([[float(c) for c in r[:4]] for r in rows])
    return a[:, :2], a[:, 2], a[:, 3]          # pts, w_right, w_left


def _maps_with_a_centerline():
    return [m for m in sorted(os.listdir(MAPS))
            if os.path.isfile(os.path.join(MAPS, m, "centerline.csv"))
            and os.path.isfile(os.path.join(MAPS, m, f"{m}.yaml"))]


@pytest.mark.parametrize("name", _maps_with_a_centerline())
def test_assignment_matches_the_geometry_on_every_station(name):
    """The function the generator now uses must agree with the truth at every sampled station,
    on every shipped map -- including the two whose stored labels are wrong."""
    pts, _wr, _wl = _centerline(name)
    a, b = load_contours(os.path.join(MAPS, name), name)
    _br, bl = assign_sides(pts, a, b)
    checked = wrong = 0
    for i in range(0, len(pts), max(1, len(pts) // 120)):
        t = truth_left_right(pts, i, a, b)
        if t is None:
            continue
        checked += 1
        d_left = float(np.min(np.hypot(bl[:, 0] - pts[i][0], bl[:, 1] - pts[i][1])))
        if abs(d_left - t[0]) > 1e-9:
            wrong += 1
    assert checked > 50, f"{name}: only {checked} usable stations"
    assert wrong == 0, f"{name}: {wrong}/{checked} stations labelled against the geometry"


@pytest.mark.parametrize("name", _maps_with_a_centerline())
def test_reversing_the_centerline_swaps_the_sides(name):
    """Drive the track the other way and left becomes right. The old decision could not do this:
    it read the CONTOUR's direction, which does not change when the car turns around."""
    pts, _wr, _wl = _centerline(name)
    a, b = load_contours(os.path.join(MAPS, name), name)
    r_fwd, l_fwd = assign_sides(pts, a, b)
    r_rev, l_rev = assign_sides(pts[::-1].copy(), a, b)
    assert np.shares_memory(r_fwd, l_rev) or np.array_equal(r_fwd, l_rev), \
        f"{name}: reversing the centerline did not swap the sides"
    assert np.array_equal(l_fwd, r_rev)


def test_the_known_bad_map_is_still_detected():
    """f still ships with d_left/d_right disagreeing with its geometry; ifac and ifac_0807 agree.

    ifac_0807 USED TO BE in the swapped list and is not any more -- it was regenerated (127
    swapped stations against 0 at the commit this test was written, 0 against 124 now). That is
    the map being fixed, not the detector breaking, so the expectation moves with it rather than
    the test being deleted: f is what still exercises the positive case, and ifac_0807 has joined
    ifac in exercising the negative one. If f ever reads 'ok' without being regenerated, THAT is
    the detector breaking.
    """
    seen = {}
    for name in ("ifac", "f", "ifac_0807"):
        d = os.path.join(MAPS, name)
        if not os.path.isdir(d):
            continue
        pts, wr, wl = _centerline(name)
        a, b = load_contours(d, name)
        n_ok, n_sw, _amb = score(pts, wl, wr, a, b)
        seen[name] = "SWAPPED" if n_sw > n_ok else "ok"
    for good in ("ifac", "ifac_0807"):
        if good in seen:
            assert seen[good] == "ok", f"{good} reads {seen[good]} -- regressed, or detector broken"
    if "f" in seen:
        assert seen["f"] == "SWAPPED", "f reads ok -- regenerated, or detector broken"


def test_verdict_never_raises_on_a_bad_map_dir():
    """The runtime callers use verdict() as a startup diagnostic; it must degrade, not throw."""
    assert verdict([[0.0, 0.0]], [1.0], [1.0], "/nonexistent", "nope")[0] == "unknown"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
