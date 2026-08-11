"""eval_gp_accuracy (B) 노이즈 지표 — std / lag-1 자기상관 계산 검증.

Run:
    cd nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest \
        test/test_eval_gp_noise.py -v
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_gp_accuracy as ev  # noqa: E402


class TestNoiseMetrics(unittest.TestCase):
    def test_std_matches_numpy(self):
        res = np.random.RandomState(0).randn(500, 3)
        m = ev.noise_metrics(res)
        np.testing.assert_allclose(m["std"], res.std(axis=0), rtol=1e-6)

    def test_ac1_smooth_vs_noise(self):
        # 매끄러운 신호(누적합)는 ac1≈1, 백색잡음은 ac1≈0.
        smooth = np.cumsum(np.random.RandomState(1).randn(1000)) * 0.01
        white = np.random.RandomState(2).randn(1000)
        res = np.stack([smooth, white, smooth], axis=1)
        m = ev.noise_metrics(res)
        self.assertGreater(m["ac1"][0], 0.9)   # smooth
        self.assertLess(abs(m["ac1"][1]), 0.2)  # white


if __name__ == "__main__":
    unittest.main()
