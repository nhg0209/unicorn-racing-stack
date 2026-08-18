"""지연보상 테스트 — x0를 τ초 앞으로 명목동역학 전파 (실측 130ms 보상용)."""
import numpy as np
import pytest

from nonlinear_mpc_acados.mpc_core.lmpc.nominal_dynamics import latency_compensate_x0


def test_zero_tau_identity():
    x0 = np.array([1.0, 2.0, 0.1, 3.0, 0.0, 0.0, 5.0])
    out = latency_compensate_x0(x0, np.array([0.0, 0.0, 3.0]), 0.0)
    assert np.allclose(out, x0)


def test_straight_advance():
    # 직진 3 m/s, 130ms → x가 약 0.39m 전진, 나머지 거의 불변
    x0 = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 10.0])
    u = np.array([0.0, 0.0, 3.0])
    out = latency_compensate_x0(x0, u, 0.13)
    assert out[0] == pytest.approx(0.39, abs=0.02)
    assert abs(out[1]) < 0.02 and abs(out[2]) < 0.02
    assert out[3] == pytest.approx(3.0, abs=0.05)
    assert out[6] == pytest.approx(10.0 + 3.0 * 0.13, abs=0.02)  # s는 p_v로 전진


def test_accel_applied():
    x0 = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    u = np.array([2.0, 0.0, 3.0])
    out = latency_compensate_x0(x0, u, 0.13)
    assert out[3] == pytest.approx(3.0 + 2.0 * 0.13, abs=0.03)


def test_shape_is_7():
    x0 = np.zeros(7); x0[3] = 2.0
    out = latency_compensate_x0(x0, np.array([0.0, 0.1, 2.0]), 0.13)
    assert out.shape == (7,)


def test_column_shaped_control_regression():
    # 2026-07-03 실차 사고 재현 케이스: solve가 (3,1) 열벡터 반환 → TypeError로
    # 제어루프 사망. ravel 코어스로 수용해야 함.
    x0 = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 10.0])
    u_col = np.array([[0.5], [0.1], [3.0]])
    out = latency_compensate_x0(x0, u_col, 0.13)
    assert out.shape == (7,)
    assert out[0] == pytest.approx(0.39, abs=0.03)
