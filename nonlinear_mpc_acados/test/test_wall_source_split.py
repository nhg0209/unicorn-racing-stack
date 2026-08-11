"""Corridor walls must come from the CENTERLINE's d_left/d_right, not the
raceline's. In raceline mode (track_source=raceline) the IQP raceline wpnts
carry unreliable d_left/d_right (PIPELINE.md 2026-05-30: reported 0.5 vs real
1.25), which collapses the MPC corridor and kills lateral avoidance room.

build_track_from_wpnts(ref_wpnts, wall_wpnts=centerline) must place the corridor
walls at the TRUE wall position (centerline point + (d - WALL_MARGIN)·centerline
normal), resampled onto the raceline stations, while keeping the raceline as the
reference line.

Run:
    cd src/unicorn-racing-stack/nonlinear_mpc_acados
    python3 -m pytest test/test_wall_source_split.py -v
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from nonlinear_mpc_acados.track_loader import build_track_from_wpnts

WALL_MARGIN = 0.18  # mirrors track_loader._WALL_MARGIN


def _straight_centerline(n=80, length=8.0, d=1.0):
    """Centerline along +x at y=0, heading psi=0. Correct walls at y=±d."""
    wp = []
    for i in range(n):
        x = length * i / (n - 1)
        wp.append(SimpleNamespace(
            x_m=x, y_m=0.0,
            psi_rad=0.0, psi_centerline_rad=0.0,
            d_left=d, d_right=d, vx_mps=3.0, kappa_radpm=0.0))
    return wp


def _raceline_offset(n=60, length=8.0, y_off=0.4, bad_d=0.2):
    """A reference line PARALLEL to the centerline but shifted +y by y_off,
    carrying BAD d_left/d_right. Same heading psi=0. If walls were built from
    THIS line's d (the bug), they'd land at y_off±bad_d — wrong."""
    wp = []
    for i in range(n):
        x = length * i / (n - 1)
        wp.append(SimpleNamespace(
            x_m=x, y_m=y_off,
            psi_rad=0.0, psi_centerline_rad=0.0,
            d_left=bad_d, d_right=bad_d, vx_mps=3.0, kappa_radpm=0.0))
    return wp


class TestWallSourceSplit(unittest.TestCase):
    def _eval_walls_at(self, td, s):
        lx = float(td.left_lut_x(s)); ly = float(td.left_lut_y(s))
        rx = float(td.right_lut_x(s)); ry = float(td.right_lut_y(s))
        return (lx, ly), (rx, ry)

    def test_walls_from_centerline_not_raceline(self):
        center = _straight_centerline(d=1.0)
        race = _raceline_offset(y_off=0.4, bad_d=0.2)
        td = build_track_from_wpnts(
            race, wall_wpnts=center, default_v=3.0,
            inflation_factor=0.0, corridor_half_width=0.0)

        # Reference line is the raceline (shifted to y=0.4).
        s_mid = float(td.element_arc_lengths_orig[-1]) * 0.5
        cx, cy = float(td.center_lut_x(s_mid)), float(td.center_lut_y(s_mid))
        self.assertAlmostEqual(cy, 0.4, delta=0.05,
                               msg="reference line must follow the raceline (y=0.4)")

        # Walls must be at the TRUE centerline walls y=±(1.0-margin), NOT at the
        # raceline's bad d (which would give y≈0.4±0.2 → left 0.6, right 0.2).
        (lx, ly), (rx, ry) = self._eval_walls_at(td, s_mid)
        self.assertAlmostEqual(ly, +(1.0 - WALL_MARGIN), delta=0.06,
                               msg=f"left wall should be true centerline wall, got y={ly:.3f}")
        self.assertAlmostEqual(ry, -(1.0 - WALL_MARGIN), delta=0.06,
                               msg=f"right wall should be true centerline wall, got y={ry:.3f}")

    def test_backward_compatible_without_wall_wpnts(self):
        """No wall_wpnts → behaviour unchanged: walls from the ref wpnts' own d."""
        center = _straight_centerline(d=1.0)
        td = build_track_from_wpnts(
            center, default_v=3.0, inflation_factor=0.0, corridor_half_width=0.0)
        s_mid = float(td.element_arc_lengths_orig[-1]) * 0.5
        (lx, ly), (rx, ry) = self._eval_walls_at(td, s_mid)
        self.assertAlmostEqual(ly, +(1.0 - WALL_MARGIN), delta=0.06)
        self.assertAlmostEqual(ry, -(1.0 - WALL_MARGIN), delta=0.06)


if __name__ == '__main__':
    unittest.main()
