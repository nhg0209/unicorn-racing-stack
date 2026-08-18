"""H10 회귀 테스트 — SS ψ를 solver의 unwrapped ψ 브랜치로 리베이스.

배경(2026-07-03): solver 내부 ψ는 랩마다 ±2π 누적(unwrap)되는데 SS 교사 ψ는
wrapped 로 패킹돼, joint-α anchor 의 ψ 잔차가 랩 k에서 2πk → opti≈(2πk)² 발산
(실차 k=2/4/7에서 156/610/1866 재현). 수정 = pack 시 rebase_angles_to_ref.
"""
import math

import numpy as np
import pytest

from nonlinear_mpc_acados.mpc_core.acados_kinematic import rebase_angles_to_ref


def test_identity_when_same_branch():
    a = np.array([0.1, -0.5, 3.0])
    out = rebase_angles_to_ref(a, 0.0)
    assert np.allclose(out, a)


def test_two_laps_unwrapped_ref():
    # 랩 2회 후 solver ψ ≈ 원래 ψ + 4π. wrapped SS ψ=0.3 은 4π+0.3 으로 와야 함.
    ref = 0.25 + 4.0 * math.pi
    out = rebase_angles_to_ref(np.array([0.3]), ref)
    assert out[0] == pytest.approx(0.3 + 4.0 * math.pi)
    assert abs(out[0] - ref) <= math.pi


def test_negative_unwrap():
    ref = -0.1 - 6.0 * math.pi   # 반대 방향 3랩
    out = rebase_angles_to_ref(np.array([0.2, -0.4]), ref)
    assert np.all(np.abs(out - ref) <= math.pi + 1e-9)
    # 2π 격자 위에서만 이동 (각도 의미 보존)
    assert np.allclose((out - np.array([0.2, -0.4])) % (2.0 * math.pi), 0.0, atol=1e-9)


def test_residual_bounded_random():
    rng = np.random.default_rng(7)
    for _ in range(50):
        ref = float(rng.uniform(-40, 40))
        a = rng.uniform(-math.pi, math.pi, size=10)
        out = rebase_angles_to_ref(a, ref)
        assert np.all(np.abs(out - ref) <= math.pi + 1e-9)
