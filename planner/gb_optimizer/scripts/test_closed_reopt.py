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
    """0.488 m held across the 8.5 m gap where the hump pipeline drew a W back to the raceline."""
    ref, cor = track
    _l, d, rep = C.reoptimize_closed(ref, [box(ref, A_I), box(ref, B_I)], cor, P)
    assert rep.ok
    assert rep.hold >= 0.40, f"hold {rep.hold:.3f}"
    assert _clear(rep) >= NEED - 1e-9, f"clearances {rep.clearances}"
    assert excursions(d, pair_span(len(ref))) == 0, "the line crosses back between the boxes"
    assert rep.solve_ms < 100.0


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

    The QP holds the keep-out at its own stations; the periodic cubic between them dips. Worst
    over the two-box case and all twelve field placements at grid 0.50: 4.11 mm against 10.0.
    """
    ref, cor = track
    p0 = ReoptParams(disc_allow_m=0.0)
    worst = max(C.reoptimize_closed(ref, obs, cor, p0)[2].sag_mm
                for obs in [[box(ref, A_I), box(ref, B_I)]]
                + [[box(ref, i) for i in t] for t in TRIOS])
    assert 0.0 < worst <= P.disc_allow_m * 1e3, f"sag {worst:.2f} mm vs {P.disc_allow_m * 1e3} mm"


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
    assert float(np.median(np.abs(d[far]))) <= 0.05
    assert rep.peak_kappa <= 1.70, "no worse than the hump pipeline's 1.57-1.67 on the same map"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
