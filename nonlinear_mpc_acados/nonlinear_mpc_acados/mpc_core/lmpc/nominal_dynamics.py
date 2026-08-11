"""Numpy mirror of acados f_expl (8-state Pacejka tanh blend) + one-step
velocity residual. Single source of truth for B4' error regression.

Constants copied verbatim from scripts/extract_residuals.py:33-45. That script
keeps its own duplicate mirror; Task 5's cross-check guards against drift.
"""
import numpy as np

L_WB = 0.307
# 2026-07-28: acados_kinematic.py:119-124 배포값과 동기 (기존 값은 6개 상수 모두
# 어긋나 있었음: m+2%, Iz+23%, lf+2%, lr-15%, h_cg-81%, mu 별도 아래 참조).
M = 3.47
IZ = 0.04712
LF = 0.15875
LR = 0.17145
H_CG = 0.074
# dyn_mu 는 yaml(config/ddrx_unified_params.yaml: dyn_mu: 0.70)로 런타임 변경되는
# 값이라 yaml이 진실이며, 여기 값은 (actuation_latency_s=0 이라 현재 미사용인)
# 오프라인 근사치일 뿐이다.
MU = 0.70
BF = BR = 10.0
DF = DR = 1.0
G = 9.81
DT = 0.04
V_B = 0.5
V_S = 0.3


def f_dynamic(state, u):
    _, _, psi, vx, vy, r, _, delta_prev = state
    a_x, delta, p_v = u
    vx_safe = max(vx, 1e-3)
    alpha_f = np.arctan2(-vy - LF * r, vx_safe) + delta
    alpha_r = np.arctan2(-vy + LR * r, vx_safe)
    F_zf = M * (-a_x * H_CG + G * LR) / L_WB
    F_zr = M * (a_x * H_CG + G * LF) / L_WB
    F_yf = MU * DF * F_zf * np.tanh(BF * alpha_f)
    F_yr = MU * DR * F_zr * np.tanh(BR * alpha_r)
    return np.array([
        vx * np.cos(psi) - vy * np.sin(psi),
        vx * np.sin(psi) + vy * np.cos(psi),
        r,
        a_x + (-F_yf * np.sin(delta)) / M + vy * r,
        (F_yr + F_yf * np.cos(delta)) / M - vx * r,
        (F_yf * LF * np.cos(delta) - F_yr * LR) / IZ,
        p_v,
        (delta - delta_prev) / DT,
    ])


def f_kinematic(state, u):
    _, _, psi, vx, vy, r, _, delta_prev = state
    a_x, delta, p_v = u
    beta_kin = np.arctan(LR * np.tan(delta) / L_WB)
    vy_tgt = vx * np.tan(beta_kin)
    r_tgt = (vx / L_WB) * np.tan(delta) * np.cos(beta_kin)
    tau_kin = 0.05
    return np.array([
        vx * np.cos(psi + beta_kin),
        vx * np.sin(psi + beta_kin),
        r_tgt,
        a_x,
        (vy_tgt - vy) / tau_kin,
        (r_tgt - r) / tau_kin,
        p_v,
        (delta - delta_prev) / DT,
    ])


def f_expl(state, u):
    vx = state[3]
    w = 0.5 * (1.0 + np.tanh((vx - V_B) / V_S))
    return w * f_dynamic(state, u) + (1.0 - w) * f_kinematic(state, u)


def predict_next(state, u, dt):
    """One Euler step of the nominal blended dynamics. Returns (8,)."""
    return np.asarray(state, float) + dt * f_expl(np.asarray(state, float),
                                                  np.asarray(u, float))


def velocity_residual(state, u, next_state, dt):
    """actual_next[vx,vy,r] - nominal_predicted[vx,vy,r]. Returns (3,)."""
    pred = predict_next(state, u, dt)
    return np.asarray(next_state, float)[3:6] - pred[3:6]


def latency_compensate_x0(x0_7, u_last, tau, sub_dt=0.01):
    """x0(7-dim)를 τ초 앞으로 명목 동역학 전파 — 액추에이션 지연 보상.

    2026-07-03 실측: 조향명령→차체 요 응답 130ms (소프트웨어 20ms + 물리 110ms).
    solve 는 '지금' 상태 기준 u[0] 를 내지만 차는 τ 뒤에야 반응 → x0 를 τ만큼
    미리 전진시켜 solve 하면 u[0] 가 '적용 시점' 상태의 최적이 된다.
    u_last = 직전 적용 컨트롤 [a_x, δ, p_v]. sub_dt Euler substep (0.13/0.01=13회).
    """
    u_last = np.ravel(np.asarray(u_last, dtype=float))[:3]  # (3,1)/DM/list 모두 수용
    s = np.concatenate([np.asarray(x0_7, float).ravel(), [float(u_last[1])]])
    t = 0.0
    tau = float(tau)
    while t < tau - 1e-9:
        dt = min(float(sub_dt), tau - t)
        s = predict_next(s, u_last, dt)
        t += dt
    return s[:7]
