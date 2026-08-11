"""extract_residuals --source ekf — 유한차분 대신 로깅 EKF/명령 컬럼 사용 검증.

Run:
    cd nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest \
        test/test_extract_residuals_ekf.py -v
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ 는 패키지가 아니므로 경로 삽입 후 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_residuals as er  # noqa: E402


def _synthetic_df(n=8):
    """상수 dt=DT, feasible=1, vx>min_vx. car_x/car_y 의 유한차분 vy 와
    vy_ekf 가 명백히 다르도록 설정 → ekf 경로가 컬럼을 쓰는지 판별."""
    DT = er.DT
    t = np.arange(n) * DT
    return pd.DataFrame({
        "t": t,
        "v_actual": np.full(n, 2.0),       # vx
        "steer_cmd": np.full(n, 0.05),     # delta
        "car_x": np.arange(n) * 0.5,       # 유한차분 vx_world=12.5 (vy_ekf 와 무관)
        "car_y": np.arange(n) * 0.3,
        "car_yaw": np.full(n, 0.0),
        "current_s": np.arange(n) * 0.08,
        "mpcc_active": np.ones(n),
        "feasible": np.ones(n),
        "vy_ekf": np.full(n, 0.01),        # 작은 횡속도
        "r_ekf": np.full(n, 0.10),         # 작은 yaw rate
        "v_cmd": np.full(n, 1.0),          # a_x 명령
    })


class TestExtractEkf(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(er.MPC_LOGS)  # 사용 안 함; 파일은 tmp 로 직접 작성
        self.csv = Path(__file__).resolve().parent / "_tmp_ekf.csv"
        _synthetic_df().to_csv(self.csv, index=False)

    def tearDown(self):
        if self.csv.exists():
            self.csv.unlink()

    def test_shapes(self):
        X, Y = er.process_csv(self.csv, min_vx=0.5, source="ekf")
        self.assertEqual(X.shape[1], 5)
        self.assertEqual(Y.shape[1], 3)
        self.assertGreater(X.shape[0], 0)

    def test_uses_ekf_columns_not_finite_diff(self):
        # GP input 의 vy(=2번째 열) 는 vy_ekf(0.01) 여야 한다.
        # 위치 유한차분이면 0.01 이 절대 안 나옴(car_y diff 기반 → 수 m/s).
        X, _ = er.process_csv(self.csv, min_vx=0.5, source="ekf")
        np.testing.assert_allclose(X[:, 1], 0.01, atol=1e-9)  # vy
        np.testing.assert_allclose(X[:, 2], 0.10, atol=1e-9)  # r
        np.testing.assert_allclose(X[:, 4], 1.00, atol=1e-9)  # a_x = v_cmd

    def test_residual_matches_euler_step(self):
        # 손계산: 한 내부 샘플의 residual == actual_kp1 - euler_step(state_k,u_k,dt)[3:6]
        df = _synthetic_df()
        DT = er.DT
        k = 2
        delta_prev = df["steer_cmd"].to_numpy()[k - 1]
        state_k = np.array([df["car_x"][k], df["car_y"][k], df["car_yaw"][k],
                            df["v_actual"][k], df["vy_ekf"][k], df["r_ekf"][k],
                            df["current_s"][k], delta_prev])
        u_k = np.array([df["v_cmd"][k], df["steer_cmd"][k], df["v_actual"][k]])
        x_pred = er.euler_step(state_k, u_k, DT)
        actual_kp1 = np.array([df["v_actual"][k + 1], df["vy_ekf"][k + 1], df["r_ekf"][k + 1]])
        expected_res = actual_kp1 - x_pred[3:6]

        X, Y = er.process_csv(self.csv, min_vx=0.5, source="ekf")
        # 첫 유효 샘플이 k=1 이므로 k=2 는 인덱스 1
        np.testing.assert_allclose(Y[1], expected_res, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
