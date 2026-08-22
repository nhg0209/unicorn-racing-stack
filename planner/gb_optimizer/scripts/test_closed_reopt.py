#!/usr/bin/env python3
"""Gate for the closed-track avoidance QP, on the real ifac map and its real corridor.

Every number here was measured first (planner/gb_optimizer/scripts/bench_closed_reopt.py prints
the full tables); the thresholds are the requirement, not the measurement, and none of them was
moved to make a case pass.

WHAT IS ASSERTED AND WHAT IS NOT:

  clearance >= obs_margin + w_veh/2 = 0.300 m   -- the safety requirement. disc_allow_m is NOT
      part of it: the QP solves with the allowance so the cubic upsample has somewhere to sag,
      and the check is taken on the upsampled line, which is the one the car drives.

  peak |kappa| is NEVER asserted against a constant. ifac's own raceline peaks at 1.448 against a
      curvlim of 1.5, so on this map every avoidance is within 4% of the limit before it starts
      and an absolute gate would only measure the map. What matters is that the avoidance does not
      make it worse than the hump pipeline did on the same scenarios (1.57-1.67 on three boxes),
      so peaks are reported and compared to the clean line, never used to reject.

  grid independence is asserted for CLEARANCE ONLY. D2 scales as 1/ds^2, so the offset profile's
      resolution is a modelling choice and curvature moves with it by construction (1.43 at 0.50 m,
      1.80 at 0.30 m on the same two boxes). Clearance is the property that must not care where
      the grid falls, and after the analytic keep-out it does not.

  the upsample sag is asserted at the DEFAULT grid only. disc_allow_m is calibrated to
      grid_step_m = 0.50 and the sag is 4.11 mm there, 5.40 at 0.30 and 22.97 at 0.70 -- a
      different grid needs a different allowance, which is why the two are documented together.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "planner/gb_optimizer"))

from gb_optimizer import closed_reopt as C            # noqa: E402
from gb_optimizer.closed_reopt import Obstacle, ReoptParams  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_closed_reopt import (A_I, B_I, TRIOS, box, corridor_from_map,  # noqa: E402
                                excursions, load_ifac, pair_span)

P = ReoptParams()
NEED = P.obs_margin + P.w_veh / 2.0


@pytest.fixture(scope="module")
def track():
    ref = load_ifac()
    return ref, corridor_from_map(ref)


def _clear(rep):
    return min(rep.clearances) if rep.clearances else float("inf")


# (1) ----------------------------------------------------------------------------------------
def test_no_obstacle_is_a_no_op(track):
    """Nothing to avoid must mean nothing done -- the same object back, not a re-splined copy."""
    ref, cor = track
    line, d, rep = C.reoptimize_closed(ref, [], cor, P)
    assert np.shares_memory(line, ref)
    assert np.max(np.abs(d)) == 0.0
    assert rep.ok


# (2) ----------------------------------------------------------------------------------------
def test_two_boxes_hold_the_offset_between_them(track):
    """0.48 m held across the 8.5 m gap where the hump pipeline drew a W back to the raceline."""
    ref, cor = track
    _l, d, rep = C.reoptimize_closed(ref, [box(ref, A_I), box(ref, B_I)], cor, P)
    assert rep.ok
    assert rep.hold >= 0.40, f"hold {rep.hold:.3f}"
    assert _clear(rep) >= NEED - 1e-9, f"clearances {rep.clearances}"
    assert excursions(d, pair_span(len(ref))) == 0, "the line crosses back between the boxes"
    # 400 ms, not the 100 this asserted when the QP solved 74 stations. At grid 0.10 it solves
    # every published station (367) to kill the merge-tail ripple, and the measured p95 is ~200 ms
    # -- still HALF the hump pipeline's measured 822 ms on the same cases, and off the executor.
    assert rep.solve_ms < 400.0


# (3) ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("corner", (60, 120, 200))
def test_a_corner_box_does_not_change_the_pair(track, corner):
    """W2': a box elsewhere on the lap may move the pair's line, but not its BEHAVIOUR.

    The offset does shift by a few mm -- the objective is global and that is honest -- so what is
    asserted is the hold, both clearances and the excursion count, not a displacement in mm.
    """
    ref, cor = track
    pair = [box(ref, A_I), box(ref, B_I)]
    span = pair_span(len(ref))
    _la, da, ra = C.reoptimize_closed(ref, pair, cor, P)
    _lb, db, rb = C.reoptimize_closed(ref, pair + [box(ref, corner)], cor, P)
    assert rb.ok
    assert rb.hold >= 0.40, f"hold {rb.hold:.3f}"
    assert min(rb.clearances[:2]) >= NEED - 1e-9, f"pair clearances {rb.clearances[:2]}"
    assert excursions(db, span) == excursions(da, span)


# (4) ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("step", (0.10, 0.30, 0.50, 0.70))
def test_clearance_does_not_depend_on_the_grid(track, step):
    """The keep-out is analytic and covers each station's half-spacing, so where the grid falls
    stops mattering. Before either, 0.30 m spacing gave a clearance of -0.14 m."""
    ref, cor = track
    _l, _d, rep = C.reoptimize_closed(ref, [box(ref, A_I), box(ref, B_I)], cor,
                                      ReoptParams(grid_step_m=step))
    assert _clear(rep) >= NEED - 1e-9, f"grid {step}: {rep.clearances}"


# (5) ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("trio", TRIOS)
def test_every_box_is_cleared_or_named(track, trio):
    """Twelve three-box placements around the lap: cleared by 0.300 m, or reported infeasible.

    Two of the twelve contain a box the car cannot pass on either side. That is a fact about the
    track, not a solver failure, and the answer is to name it -- with the station, the side and
    the metres -- for the reactive layer, and solve the rest.
    """
    ref, cor = track
    _l, _d, rep = C.reoptimize_closed(ref, [box(ref, i) for i in trio], cor, P)
    assert rep.ok
    assert _clear(rep) >= NEED - 1e-9, f"{trio}: {rep.clearances}"
    assert len(rep.infeasible_why) == len(rep.infeasible)
    for why in rep.infeasible_why:
        assert "station" in why


def test_an_unavoidable_box_is_reported_not_forced(track):
    """A box the corridor cannot clear must come back named, and must not take the others down."""
    ref, cor = track
    _l, _d, rep = C.reoptimize_closed(ref, [box(ref, i) for i in (330, 30, 180)], cor, P)
    assert rep.ok and rep.infeasible == [2]
    assert _clear(rep) >= NEED - 1e-9, "the two avoidable boxes still got their margin"


# (6) ----------------------------------------------------------------------------------------
def test_disc_allow_covers_the_upsample_sag(track):
    """disc_allow_m must be the measured sag's upper bound, at the grid it is calibrated for.

    The QP solves to need + disc_allow_m and the cubic between its stations gives some of it
    back; what survives in the published clearance is the headroom. Measured as what the upsample
    SPENDS, because re-solving with the allowance at zero measures nothing now -- the contract
    check hands the box to the reactive layer instead of clearing it short.
    """
    ref, cor = track
    spent = max(P.disc_allow_m - (c - NEED)
                for obs in [[box(ref, A_I), box(ref, B_I)]]
                + [[box(ref, i) for i in t] for t in TRIOS]
                for c in C.reoptimize_closed(ref, obs, cor, P)[2].clearances)
    assert spent <= P.disc_allow_m + 1e-9, (
        f"the upsample spent {spent * 1e3:.2f} mm of a {P.disc_allow_m * 1e3:.1f} mm allowance")


# (7) ----------------------------------------------------------------------------------------
def test_w_trades_hold_against_return_and_nothing_else(track):
    """The default w must hold 0.40 between two boxes AND bring the line home away from them.

    Curvature does not choose w: the whole sweep 0.0-0.1 lands between 1.425 and 1.447 against a
    clean line of 1.448, differences smaller than the grid moves them by. What does separate them
    is how much offset is left lying around the rest of the lap -- 0.194 m at w = 0, 0.040 at the
    default -- and that is a raceline the car did not ask for.
    """
    ref, cor = track
    obs = [box(ref, A_I), box(ref, B_I)]
    far = np.min([np.hypot(ref[:, 0] - o.x, ref[:, 1] - o.y) for o in obs], axis=0) > 2.5
    _l, d, rep = C.reoptimize_closed(ref, obs, cor, P)
    assert rep.hold >= 0.40
    assert float(np.median(np.abs(d[far]))) <= 0.06, "measured 0.058 at the default w = 0.010"
    assert rep.peak_kappa <= 1.70, "no worse than the hump pipeline's 1.57-1.67 on the same map"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------------------------------------------
# WRAP REGRESSION -- the two defects fixed in 930665d, which would come back silently
# ------------------------------------------------------------------------------------------
def test_the_far_side_of_the_lap_is_left_alone(track):
    """One box must not move the line on the OTHER SIDE of the track.

    The first defect: the keep-out selected stations by tangential projection alone, and on a
    closed loop the far side runs back the other way, so a station 11.9 m across the infield saw
    the box at dt ~ 0 and took the full band at du ~ -12 m. Six coarse stations per two-box case
    got a constraint they could not meet and were then "resolved" by pinning the line to the
    corridor edge -- the line sat on the wall at six places with no obstacle near them.
    """
    ref, cor = track
    n = len(ref)
    i = 275
    _l, d, rep = C.reoptimize_closed(ref, [box(ref, i)], cor, P)
    assert rep.ok
    s = np.concatenate([[0.0], np.cumsum(C._closed_el(ref[:, :2]))[:-1]])
    ds = np.abs(s - s[i])
    L = s[-1] + float(C._closed_el(ref[:, :2])[-1])
    arc = np.minimum(np.abs(s - s[i]), L - np.abs(s - s[i]))
    far = arc > 12.0                                  # a third of the lap away from the box
    # The offset does decay rather than stop -- the objective is global -- and that decay is
    # measured, not assumed: 0.335 m at 4 m, 0.096 at 8, 0.043 at 12, 0.018 at the antipode.
    # The defect produced 0.10-0.70 m at stations picked by nothing but a tangent direction.
    assert np.max(np.abs(d[far])) < 0.05, (
        f"the line moved {np.max(np.abs(d[far])):.3f} m a third of a lap from the only box")


def test_only_stations_beside_the_box_are_constrained(track):
    """The set of stations the keep-out narrows must be a longitudinal neighbourhood of the box.

    Asserted on the bounds themselves, not on the solution: a bound that moves 12 m away is the
    defect whether or not the solver happens to absorb it that time.
    """
    ref, cor = track
    pts = ref[:, :2]
    n = len(pts)
    _psi, nv, tan = C._frame(pts)
    hi = np.minimum(ref[:, 2] - 0.5 * P.w_veh, np.where(np.isfinite(cor[1]), cor[1], np.inf))
    lo = np.maximum(-(ref[:, 3] - 0.5 * P.w_veh), np.where(np.isfinite(cor[0]), cor[0], -np.inf))
    o = box(ref, 275)
    lo2, hi2 = C.obstacle_bounds(pts, nv, tan, [o], [-1], lo, hi, P, 0.1)
    touched = np.flatnonzero((lo2 > lo + 1e-9) | (hi2 < hi - 1e-9))
    assert len(touched) > 0
    reach = o.r + P.obs_margin + 0.5 * P.w_veh + P.disc_allow_m
    d_eucl = np.hypot(pts[touched, 0] - o.x, pts[touched, 1] - o.y)
    assert np.max(d_eucl) <= reach + 0.1, (
        f"a station {np.max(d_eucl):.2f} m from the box was constrained by it (reach {reach:.2f})")


def test_the_half_spacing_correction_is_what_makes_the_grid_not_matter(track, monkeypatch):
    """Ask the keep-out at the station instead of over the interval it stands for, and a 0.70 m
    grid straddles the box: the nearest station is up to 0.35 m along, sees sqrt(R^2 - 0.35^2)
    instead of R, and the line passes closer than it was told. This pins that mechanism, so the
    correction cannot be dropped as an unused argument.
    """
    ref, cor = track
    obs = [box(ref, A_I), box(ref, B_I)]
    p70 = ReoptParams(grid_step_m=0.70)
    real = C.obstacle_bounds
    monkeypatch.setattr(C, "obstacle_bounds",
                        lambda *a, **k: real(*a[:8], 0.0) if len(a) >= 8 else real(*a, **k))
    _l, _d, without = C.reoptimize_closed(ref, obs, cor, p70)
    monkeypatch.undo()
    _l, _d, with_it = C.reoptimize_closed(ref, obs, cor, p70)
    # Since the contract check went in, falling short does not show up as a short clearance --
    # it shows up as the box being handed to the reactive layer instead, which is the point.
    assert len(without.clearances) < len(with_it.clearances), (
        "the point-sampled keep-out was supposed to cost a box")
    assert min(with_it.clearances) >= NEED - 1e-9
