"""Backend-agnostic model/LMPC policy helpers.

Kept dependency-free (no casadi / acados_template) so the ROS node can import
it regardless of which solver backend (acados or ipopt) is selected — importing
it must never pull in a heavy/optional solver dependency.
"""
from __future__ import annotations


# ─── Grip single source (2026-06-10 friction-ellipse-mu spec) ───────────────
G_GRAV = 9.81
# Solver longitudinal brake limit [m/s²]. MUST stay in sync with
# acados_kinematic lbu[0] (which imports this const — single source).
# 2026-06-15: 실차 CSV(mpc_20260615_171130) 분석 — 전진 감속 시 실제 감속 ≈
# 0.7~1.0 m/s²(코스팅; speed PID가 양수 저속 명령에 brake 전류 안 씀=능동제동 미실행).
# 실험: -4.0(낙관)→코너 늦게 박음 / -1.0(정직코스팅)→트위스티 맵서 전구간 과감속 크롤.
# 코스팅만으론 5 m/s 불가(둘 중 하나로 수렴). 근본해결=VESC Speed PID로 능동제동 살리기.
# 그때까지는 -3.0 기준(검증된 baseline, max_speed 4 에서 안 박힘) 유지.
# 단일소스: acados_kinematic lbu[0] = 솔버 a_x 하한 + 캡 _bf(=|A_MIN|/a_lat) 둘 다.
A_MIN_DYN = -7.0  # 2026-07-03 2차 상향: 실측 순간최대 감속 -7.08 근거 (명령단 brake_anticip 5.5 가드)


def grip_a_lat_limit(mu, ellipse_frac=0.95):
    """Physical lateral-accel ceiling a_lim = μ·g·η (η = ellipse headroom)."""
    return float(mu) * G_GRAV * float(ellipse_frac)


def clamp_a_lat_to_grip(a_lat_safe, mu, ellipse_frac=0.95):
    """Clamp a requested a_lat_safe to the physical μ·g·η ceiling.

    Returns (effective_a_lat, clamped). BO/yaml can request any a_lat — the
    speed profile must never be built on grip the tire cannot deliver
    (mu=0.6 BO-best non-reproduction root cause, 2026-06-09/10).
    """
    lim = grip_a_lat_limit(mu, ellipse_frac)
    a = float(a_lat_safe)
    return (min(a, lim), a > lim)


# ─── Avoidance side decision (2026-06-11 window-aware) ──────────────────────
def decide_side_window(e_c_obs, w_left, w_right,
                       w_car_safe=0.21, margin=0.1):
    """Window-aware avoidance side decision (pure, numpy/casadi-free).

    Replaces the single-point top-2 boundary-distance compare whose
    centerline tie always returned -1 ("always avoids down" bug,
    2026-06-11). Looks at the corridor room the detour tube actually
    drives through: per window sample s_k, the gap between the obstacle's
    lateral line and each labeled boundary.

    e_c_obs  — obstacle lateral offset (solver e_c sign convention).
    w_left/w_right — signed e_c projections of the labeled left/right
        boundary at each window sample (sin_t·Δx − cos_t·Δy). The labels'
        sign flips with track orientation (CW/CCW — see the corridor
        smooth-max/min in acados_kinematic), so gaps use |w − e_c_obs|,
        orientation-agnostic.

    Returns +1 (pass on labeled-left side) or -1 (labeled-right side):
      1. one side's bottleneck < w_car_safe → the other side
      2. both blocked → larger bottleneck (less-bad, was unconditional -1)
      3. bottleneck gap differs > margin → larger bottleneck
      4. else mean room differs > margin → larger mean (the "tie at the
         obstacle but one side opens downstream" case)
      5. true tie → -1 (legacy default, deterministic)
    """
    if len(w_left) == 0 or len(w_right) == 0:
        return -1
    e = float(e_c_obs)
    gap_l = [abs(float(w) - e) for w in w_left]
    gap_r = [abs(float(w) - e) for w in w_right]
    min_l, min_r = min(gap_l), min(gap_r)
    left_blocked = min_l < w_car_safe
    right_blocked = min_r < w_car_safe
    if left_blocked and not right_blocked:
        return -1
    if right_blocked and not left_blocked:
        return +1
    if left_blocked and right_blocked:
        return +1 if min_l > min_r else -1
    if abs(min_l - min_r) > margin:
        return +1 if min_l > min_r else -1
    mean_l = sum(gap_l) / len(gap_l)
    mean_r = sum(gap_r) / len(gap_r)
    if abs(mean_l - mean_r) > margin:
        return +1 if mean_l > mean_r else -1
    return -1


def slew_limit_speed(prev: float, target: float, dt: float,
                     up_rate: float = 8.0, down_rate: float = 5.0) -> float:
    """발행 drive.speed 슬루 리미터 — 물리 한계를 넘는 1-사이클 스텝 제거.

    2026-07-03: v_plan(브레이크 정직 가드 + brake-distance cap)이 SQP_RTI plan
    지터를 그대로 전달해 >0.8 m/s 스텝이 46회/분 발생 (지속 median 100ms).
    실측 제동한계 4.3 m/s², 가속 ~6 m/s² — 그보다 빠른 명령 변화는 차가 따라갈
    수 없어 정보가 아니라 저크다. down_rate=5.0 은 정직한 풀브레이크(4.3)를
    통과시키면서 스파이크만 자른다. dt<=0(첫 사이클/클럭 점프) 은 무제한 통과.
    stuck-release 후진(-0.5 즉시)은 호출측에서 바이패스할 것.
    """
    if dt <= 0.0:
        return float(target)
    lo = float(prev) - float(down_rate) * float(dt)
    hi = float(prev) + float(up_rate) * float(dt)
    return float(min(hi, max(lo, float(target))))
