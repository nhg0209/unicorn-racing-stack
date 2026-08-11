"""속도명령 슬루 리미터 테스트 — 발행 drive.speed 의 1-사이클 스파이크 제거.

배경(2026-07-03 실차): v_plan(브레이크 정직 가드)이 RTI plan 지터를 그대로
전달해 132회/172s 의 >0.8 m/s 스텝 발생 (지속 median 100ms — 물리적으로
무의미한 저크). 물리 한계(제동 4.3, 가속 ~6)를 넘는 스텝만 잘라낸다.
"""
import pytest

from nonlinear_mpc_acados.mpc_core.model_policy import slew_limit_speed


def test_within_limits_passthrough():
    # 25ms 사이클, down 5.0 → 최대 하강 0.125
    assert slew_limit_speed(3.0, 2.95, 0.025, up_rate=8.0, down_rate=5.0) == pytest.approx(2.95)
    assert slew_limit_speed(3.0, 3.1, 0.025, up_rate=8.0, down_rate=5.0) == pytest.approx(3.1)


def test_down_spike_clamped():
    # 3.0 → 0.5 (한 사이클) = -100 m/s² 급 스파이크 → -5 m/s² 로 제한
    out = slew_limit_speed(3.0, 0.5, 0.025, up_rate=8.0, down_rate=5.0)
    assert out == pytest.approx(3.0 - 5.0 * 0.025)


def test_up_spike_clamped():
    out = slew_limit_speed(2.0, 3.5, 0.025, up_rate=8.0, down_rate=5.0)
    assert out == pytest.approx(2.0 + 8.0 * 0.025)


def test_honest_brake_passes():
    # 물리 한계 내 제동(-4.3 m/s²)은 그대로 통과
    tgt = 3.0 - 4.3 * 0.025
    assert slew_limit_speed(3.0, tgt, 0.025, up_rate=8.0, down_rate=5.0) == pytest.approx(tgt)


def test_sustained_brake_reaches_target():
    # 스파이크가 아니라 지속 감속이면 여러 사이클에 걸쳐 목표 도달
    v = 3.0
    for _ in range(40):  # 1초
        v = slew_limit_speed(v, 0.0, 0.025, up_rate=8.0, down_rate=5.0)
    assert v == pytest.approx(0.0, abs=1e-9)


def test_bad_dt_passthrough():
    # dt<=0 (첫 사이클/클럭 점프) 은 무제한 통과 (가드)
    assert slew_limit_speed(3.0, 0.5, 0.0, up_rate=8.0, down_rate=5.0) == 0.5
