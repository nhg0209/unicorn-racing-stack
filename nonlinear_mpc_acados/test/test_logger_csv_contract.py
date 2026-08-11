"""mpc_debug_logger CSV 계약 — GP 필수 컬럼 + atomic 필드 동기 보증.

extract_residuals.py 가 이름으로 읽는 컬럼이 헤더에 있어야 하고,
mpc_node /mpc_debug 와 동기되는 DBG_FIELDS 길이가 atomic 2필드 추가 후 25 여야 한다.

Run:
    cd nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest \
        test/test_logger_csv_contract.py -v
"""
from __future__ import annotations
import unittest

from nonlinear_mpc_acados.mpc_debug_logger import DBG_FIELDS, csv_header

# extract_residuals.py 의 needed 집합 (GP 학습 필수 입력)
GP_REQUIRED = {"t", "v_actual", "steer_cmd", "car_x", "car_y", "car_yaw",
               "current_s", "mpcc_active", "feasible"}


class TestLoggerCsvContract(unittest.TestCase):
    def test_dbg_fields_has_atomic_fields(self):
        self.assertIn("t_ctrl", DBG_FIELDS)
        self.assertIn("feasible_msg", DBG_FIELDS)

    def test_dbg_fields_length_synced_with_mpc_node(self):
        # /mpc_debug 배열과 동기 — 1단계 atomic 2 + 2단계 EKF vy/r 2 추가 후 27
        self.assertEqual(len(DBG_FIELDS), 27)

    def test_dbg_fields_has_ekf_velocity(self):
        self.assertIn("vy_ekf", DBG_FIELDS)
        self.assertIn("r_ekf", DBG_FIELDS)

    def test_header_has_gp_required_columns(self):
        missing = GP_REQUIRED - set(csv_header())
        self.assertEqual(missing, set(), f"GP 필수 컬럼 누락: {missing}")


if __name__ == "__main__":
    unittest.main()
