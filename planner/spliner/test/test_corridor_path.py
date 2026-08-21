#!/usr/bin/env python3
"""The corridor QP, on its own: does it stay in the box, leave from where it was told, and is its
smoothness operator the OPEN one?

Every one of these is a property the node cannot check for itself -- by the time do_spline sees the
profile it is one array among ten and the only verdict left is pass/fail on the gates. The failure
this file exists for is the one round 5 named: reusing the periodic second difference on an open
segment, which is silent (the QP still solves) and wrong (it bends the segment to join its own two
ends).

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/test/test_corridor_path.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "planner" / "gb_optimizer"))

from spliner.corridor_path import (            # noqa: E402
    cut_keepout, open_second_difference, solve_corridor_path,
)

DS = 0.0997                                    # the maps' publishing spacing


def _run(n=121, d0=0.0, dp0=0.0, box=(-0.15, 0.30), left=True, bulge=0.05, w_dev=0.0,
         max_vars=60, half=1.2, dpp0=None):
    s = np.arange(n) * DS
    lo, hi = np.full(n, -half), np.full(n, half)
    m = (s > s[n // 2] - 0.5) & (s < s[n // 2] + 0.5)
    cut_keepout(lo, hi, m, box[0], box[1], left, bulge)
    lo[-1] = hi[-1] = 0.0
    lo[-2] = hi[-2] = 0.0
    d = solve_corridor_path(s, lo, hi, d0, dp0, w_dev, max_vars, dpp0)
    return s, lo, hi, m, d


def test_the_operator_is_open_not_periodic():
    """The two wrap rows, priced.

    A ramp profile -- 0 at one end, 0.4 at the other, monotone between -- is what an entry ramp IS.
    The periodic operator charges it for the step between its last station and its first; the open
    one does not, and the ratio is not a rounding difference.
    """
    from gb_optimizer.closed_reopt import _second_difference
    n, ds = 46, 0.1
    t = np.linspace(0.0, 1.0, n)
    ramp = 0.4 * (t ** 3) * (10.0 + t * (-15.0 + 6.0 * t))       # smootherstep, 0 -> 0.4
    s = np.arange(n) * ds
    D_open = open_second_difference(s)
    D_closed = _second_difference(n, ds)
    e_open = float(np.sum((D_open @ ramp) ** 2))
    e_closed = float(np.sum((D_closed @ ramp) ** 2))
    assert D_open.shape == (n - 2, n), D_open.shape
    assert np.allclose(D_open, D_closed[1:-1]), "the interior stencil must be the SAME one"
    assert e_closed > 100.0 * e_open, (e_closed, e_open)
    print(f"PASS open operator scores {e_open:.3f} on a ramp where the periodic one scores "
          f"{e_closed:.1f} ({e_closed / e_open:.0f}x) -- the two wrap rows")


def test_it_stays_inside_the_corridor():
    s, lo, hi, m, d = _run()
    assert d is not None
    worst = float(np.max(np.maximum(d - hi, lo - d)))
    assert worst <= 1e-6, worst
    print(f"PASS inside the corridor at every one of {len(s)} stations (worst {worst:.2e} m)")


def test_the_keepout_forces_the_apex_without_pinning_it():
    """Nothing tells the solver where the apex is. The keep-out is in the bounds, so the profile
    has to clear the box, and minimum bending puts it exactly at the edge the bulge asks for."""
    _s, _lo, _hi, m, d = _run(box=(-0.15, 0.30), left=True, bulge=0.05)
    assert d is not None
    assert float(np.min(d[m])) >= 0.35 - 1e-6, float(np.min(d[m]))
    _s, _lo, _hi, m2, d2 = _run(box=(-0.30, 0.15), left=False, bulge=0.05)
    assert d2 is not None
    assert float(np.max(d2[m2])) <= -0.35 + 1e-6, float(np.max(d2[m2]))
    print(f"PASS the corridor alone puts the apex at {float(np.min(d[m])):+.3f} (left) and "
          f"{float(np.max(d2[m2])):+.3f} (right) -- box edge + bulge, unpinned")


def test_the_start_pin_is_C1():
    for d0, dp0 in ((0.0, 0.0), (0.22, 0.0), (0.22, -0.08), (-0.30, 0.12)):
        s, _lo, _hi, _m, d = _run(d0=d0, dp0=dp0)
        assert d is not None, (d0, dp0)
        assert abs(d[0] - d0) < 1e-6, (d0, d[0])
        assert abs((d[1] - d[0]) / DS - dp0) < 1e-6, (dp0, (d[1] - d[0]) / DS)
    print("PASS d(s0) and the one-sided d'(s0) are the values handed in, over four start states")


def test_the_C2_start_pin_kills_the_curvature_step_at_the_seam():
    """C1 leaves the solver free to start bending at the first station it owns, and it does.

    The measured quantity is the one the sweep gates: |d[2] - 2 d[1] + d[0]| / ds, the |d'| step a
    controller reads off the published points at the junction with the pre-ramp.
    """
    def _tight(apex_frac, d0, dp0, dpp0):
        """The apex close to the start -- a SHORT entry ramp, which is what a race profile keeps
        producing and what makes the difference visible at all. With the apex in the middle the
        solver has room to ease in and C1 alone already reads 0.006."""
        n = 121
        s = np.arange(n) * DS
        lo, hi = np.full(n, -1.2), np.full(n, 1.2)
        j = int(n * apex_frac)
        cut_keepout(lo, hi, (s > s[j] - 0.5) & (s < s[j] + 0.5), -0.15, 0.30, True, 0.05)
        lo[-1] = hi[-1] = 0.0
        lo[-2] = hi[-2] = 0.0
        d = solve_corridor_path(s, lo, hi, d0, dp0, 0.0, 60, dpp0)
        assert d is not None
        return abs(float(d[2] - 2 * d[1] + d[0])) / DS

    worst_c1 = worst_c2 = 0.0
    for d0, dp0 in ((0.0, 0.0), (0.10, -0.03), (-0.10, 0.05)):
        worst_c1 = max(worst_c1, _tight(0.15, d0, dp0, None))
        worst_c2 = max(worst_c2, _tight(0.15, d0, dp0, 0.0))
    assert worst_c1 > 0.05, worst_c1          # C1 alone really does exceed the gate
    assert worst_c2 < 1e-6, worst_c2          # continuing the curvature removes the step outright
    # And when the prefix IS curved, the pin continues THAT rather than forcing a flat start.
    for q in (0.4, -0.6):
        assert abs(_tight(0.15, 0.1, -0.03, q) - abs(q) * DS) < 1e-6, q
    print(f"PASS the |d'| step at a short-ramp seam is {worst_c1:.4f} with the slope pin alone and "
          f"{worst_c2:.1e} once the curvature is continued too")


def test_the_terminal_pin_lands_on_the_raceline():
    """The caller collapses the last two stations onto 0; that is what makes the join to the d = 0
    tail C1 in the same difference the seam is measured with."""
    _s, _lo, _hi, _m, d = _run(d0=0.3, dp0=0.05)
    assert d is not None
    assert abs(d[-1]) < 1e-5 and abs(d[-2]) < 1e-5, (d[-1], d[-2])
    print(f"PASS ends at d = {d[-1]:+.2e} with d' = {(d[-1] - d[-2]) / DS:+.2e}")


def test_a_pin_between_control_points_is_still_reachable():
    """The failure the control grid had to be built around.

    A degenerate box at a station that is NOT a control point asks a cubic to hit one value to
    within a micrometre between its knots; quadprog reports the whole set inconsistent rather than
    'your basis is too coarse'. With 60 controls over 210 stations that was 13 % of solves.
    """
    n = 211
    s = np.arange(n) * DS
    lo, hi = np.full(n, -1.2), np.full(n, 1.2)
    for j in (57, 58, 59, 104, 105):                     # deliberately off the even control grid
        lo[j] = hi[j] = 0.20
    lo[-1] = hi[-1] = 0.0
    lo[-2] = hi[-2] = 0.0
    d = solve_corridor_path(s, lo, hi, 0.0, 0.0, 0.0, 60)
    assert d is not None, "a pin between two control points must not read as an empty corridor"
    assert max(abs(d[j] - 0.20) for j in (57, 58, 59, 104, 105)) < 1e-4, d[57]
    print("PASS five pins off the control grid are all hit at 60 controls")


def test_an_empty_corridor_is_None_not_a_violation():
    n = 61
    s = np.arange(n) * DS
    lo, hi = np.full(n, 0.4), np.full(n, 0.2)            # hi < lo everywhere
    assert solve_corridor_path(s, lo, hi, 0.0, 0.0) is None
    s2 = np.arange(3) * DS                               # too short to bend
    assert solve_corridor_path(s2, np.zeros(3), np.ones(3), 0.0, 0.0) is None
    print("PASS an empty corridor and a run too short both return None")


def test_the_decimation_never_costs_feasibility():
    """Fewer control points may cost shape; it must never cost a station's bound."""
    for mv in (8, 20, 40, 60, 10 ** 6):
        s, lo, hi, _m, d = _run(n=211, d0=0.2, dp0=0.03, max_vars=mv)
        assert d is not None, mv
        worst = float(np.max(np.maximum(d - hi, lo - d)))
        # 1e-6 is the solver's own box slack (it widens both sides by that much so a pin is an
        # equality it can represent), so the bound to assert is that slack and not zero.
        assert worst <= 1.1e-6, (mv, worst)
    print("PASS bounds hold at every station for 8, 20, 40, 60 and unlimited control points")


def test_bending_falls_as_the_basis_is_freed():
    """A sanity check on the objective itself: more freedom cannot bend MORE."""
    s, lo, hi, _m, _d = _run(n=211, d0=0.2, dp0=0.03)
    D2 = open_second_difference(s)
    seen = []
    for mv in (8, 20, 60, 10 ** 6):
        _s, _lo, _hi, _m2, d = _run(n=211, d0=0.2, dp0=0.03, max_vars=mv)
        e = float(np.sum((D2 @ d) ** 2))
        if seen:
            assert e <= seen[-1] * 1.02 + 1e-9, (mv, e, seen[-1])
        seen.append(e)
    # The numbers are worth reading, not just the inequality: 8 controls over 211 stations costs
    # 0.4 % of bending on a shape like this. The basis is not where the answer is decided; the
    # corridor is. What the control count buys back is the FEASIBILITY of tight pins, which is why
    # _ctrl_indices forces them in rather than the count being raised.
    print("PASS bending is monotone in the control count: " +
          " -> ".join(f"{e:.5g}" for e in seen))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"ALL PASS ({len(fns)} checks)")
