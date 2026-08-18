"""learn_refv 순수함수 — 그립여유 상향/한계 유지 + 안전 clamp 검증.

Run:
    cd nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest \
        test/test_learn_refv.py -v
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import learn_refv as lr  # noqa: E402

A_LAT_LIM = 6.5
MAX_SPEED = 10.0


class TestLearnProfile(unittest.TestCase):
    def test_raise_when_grip_margin(self):
        # g=a_lat/lim < 0.8 인 bin → base+delta (그립캡 미만이므로)
        base = np.array([4.0]); vx = np.array([4.0])
        a_lat = np.array([0.5 * A_LAT_LIM])      # g≈0.5 → 상향
        kap = np.array([0.01])                    # 거의 직선 → grip_cap 큼
        new, st = lr.learn_profile(base, vx, a_lat, kap, A_LAT_LIM, MAX_SPEED,
                                   delta=0.3, smooth_win=1)
        self.assertAlmostEqual(new[0], 4.3, places=6)
        self.assertLessEqual(st, MAX_SPEED)

    def test_hold_lower_when_at_limit(self):
        base = np.array([5.0]); vx = np.array([4.2])
        a_lat = np.array([0.98 * A_LAT_LIM])     # g>0.95 → min(base, vx_med)
        kap = np.array([0.01])
        new, _ = lr.learn_profile(base, vx, a_lat, kap, A_LAT_LIM, MAX_SPEED, smooth_win=1)
        self.assertAlmostEqual(new[0], 4.2, places=6)

    def test_grip_cap_never_exceeded(self):
        # 코너(큰 κ) 에서 상향해도 √(lim/κ) 캡 못 넘음
        base = np.array([6.0]); vx = np.array([3.0])
        a_lat = np.array([0.1 * A_LAT_LIM])      # g 작음 → 상향 시도
        kap = np.array([1.0])                     # grip_cap=√(6.5/1)=2.55
        new, _ = lr.learn_profile(base, vx, a_lat, kap, A_LAT_LIM, MAX_SPEED,
                                  delta=0.3, smooth_win=1)
        self.assertLessEqual(new[0], np.sqrt(A_LAT_LIM / 1.0) + 1e-9)

    def test_nan_bins_preserved(self):
        base = np.array([4.0, np.nan]); vx = np.array([4.0, np.nan])
        a_lat = np.array([0.5 * A_LAT_LIM, np.nan]); kap = np.array([0.01, np.nan])
        new, _ = lr.learn_profile(base, vx, a_lat, kap, A_LAT_LIM, MAX_SPEED, smooth_win=1)
        self.assertTrue(np.isnan(new[1]))

    def test_bin_laps_clean_filter(self):
        # 비feasible / 저속 샘플은 제외
        s = np.array([0.1, 0.2, 0.3, 0.4])
        vx = np.array([4.0, 4.0, 0.2, 4.0])               # idx2 저속
        kap = np.array([0.01, 0.01, 0.01, 0.01])
        feas = np.array([1, 1, 1, 0])                      # idx3 비feasible
        act = np.array([1, 1, 1, 1]); refv = np.full(4, 5.0)
        c, vm, kp, al, base, cnt = lr.bin_laps(s, vx, kap, feas, act, refv,
                                               L=10.0, bin_width=10.0, min_vx=1.0,
                                               min_count=1)
        self.assertEqual(int(cnt[0]), 2)                  # 4개 중 2개만 clean


if __name__ == "__main__":
    unittest.main()
