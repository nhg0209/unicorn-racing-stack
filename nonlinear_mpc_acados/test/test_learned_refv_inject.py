"""learned_speeds_for_wpnts — 학습값을 wpnt 누적s 에 매칭, 빈 구간은 analytic 유지.

Run:
    cd nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest \
        test/test_learned_refv_inject.py -v
"""
from __future__ import annotations
import unittest
from types import SimpleNamespace
import numpy as np

from nonlinear_mpc_acados.track_loader import learned_speeds_for_wpnts


def _wpnts(xy, vx):
    return [SimpleNamespace(x_m=x, y_m=y, vx_mps=v) for (x, y), v in zip(xy, vx)]


class TestInject(unittest.TestCase):
    def test_override_within_tol_else_keep(self):
        # 직선 4점 (간격 1m), 누적 s=[0,1,2,3]
        xy = [(0, 0), (1, 0), (2, 0), (3, 0)]
        wp = _wpnts(xy, [4.0, 4.0, 4.0, 4.0])
        learned_s = np.array([0.0, 1.0])      # s=0,1 만 학습됨
        learned_v = np.array([6.0, 6.5])
        out = learned_speeds_for_wpnts(wp, learned_s, learned_v, L=4.0, tol=0.3)
        self.assertAlmostEqual(out[0], 6.0)   # 학습값
        self.assertAlmostEqual(out[1], 6.5)   # 학습값
        self.assertAlmostEqual(out[2], 4.0)   # tol 밖 → analytic 유지
        self.assertAlmostEqual(out[3], 4.0)

    def test_empty_learned_keeps_all(self):
        xy = [(0, 0), (1, 0)]
        wp = _wpnts(xy, [4.0, 5.0])
        out = learned_speeds_for_wpnts(wp, np.array([]), np.array([]), L=2.0, tol=0.3)
        np.testing.assert_allclose(out, [4.0, 5.0])

    def test_circular_wraparound_seam(self):
        # 루프 길이 L=4. s≈L-0.1(=3.9) 학습 bin 은 wpnt[0](s=0)와 원형거리 0.1m
        # → seam 넘어 override 돼야. wpnt[3](s=3)와는 0.9m → 유지.
        xy = [(0, 0), (1, 0), (2, 0), (3, 0)]
        wp = _wpnts(xy, [4.0, 4.0, 4.0, 4.0])
        out = learned_speeds_for_wpnts(wp, np.array([3.9]), np.array([7.0]), L=4.0, tol=0.3)
        self.assertAlmostEqual(out[0], 7.0)   # s=0 ↔ 3.9 원형거리 0.1 ≤ tol
        self.assertAlmostEqual(out[3], 4.0)   # s=3 ↔ 3.9 거리 0.9 > tol

    def test_tol_boundary_inclusive(self):
        # 거리가 정확히 tol 이면 override 포함(<=).
        xy = [(0, 0), (1, 0)]
        wp = _wpnts(xy, [4.0, 4.0])
        out = learned_speeds_for_wpnts(wp, np.array([0.3]), np.array([9.0]), L=10.0, tol=0.3)
        self.assertAlmostEqual(out[0], 9.0)   # s=0 ↔ 0.3, 거리=tol → 포함
        self.assertAlmostEqual(out[1], 4.0)   # s=1 ↔ 0.3, 거리=0.7 → 제외


if __name__ == "__main__":
    unittest.main()
