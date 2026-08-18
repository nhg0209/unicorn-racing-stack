"""OppTracker — 단일 상대차 Frenet 알파-베타 트래커 (0723E).

"지금 보이는 점"이 아니라 "그 상대"라는 객체를 추적한다:
  predict(관성 전진, 가림 중에도) → associate(게이트 내 감지=같은 대상) → update.
이걸로 대체되는 밴드에이드: 이벤트 속도추정, 점프 리셋, 홀드 로직,
신선도/속도 조건 (0722z, 0723v/x/B/C/D).
"""
import math


class OppTracker:
    ALPHA = 0.45          # 위치 보정 게인
    BETA = 0.35           # 속도 보정 게인
    GATE_D = 0.8          # 횡방향 연관 게이트 [m]
    DROP_AFTER = 2.5      # 이만큼 감지 없으면 트랙 소멸 [s]
    DYN_V = 0.8           # 동적 확정 속도 문턱 [m/s]
    DYN_SUSTAIN = 0.5     # 동적 확정 지속 요구 [s]

    def __init__(self, track_len: float):
        self.L = float(track_len)
        self.reset()

    def reset(self):
        self.active = False
        self.s = None
        self.d = 0.0
        self.vs = 0.0
        self._t_pred = None       # 마지막 predict 시각
        self._t_upd = None        # 마지막 감지 갱신 시각
        self._dyn_since = None
        self._low_since = None
        self.hits = 0

    # ── 시간 전파 (매 사이클, 감지 없어도) ─────────────────────────
    def predict(self, now: float):
        if not self.active:
            return
        if now - self._t_upd > self.DROP_AFTER:
            self.reset()
            return
        dt = now - self._t_pred
        if dt > 0:
            self.s = (self.s + max(0.0, self.vs) * dt) % self.L
            self._t_pred = now

    # ── 감지 반영 ──────────────────────────────────────────────────
    def update(self, detections_sd, now: float):
        """detections_sd: [(s, d), ...] — 정적 확정 셀을 제외한 감지들."""
        self.predict(now)
        if not detections_sd:
            return False
        if not self.active:
            s, d = detections_sd[0]
            self.active = True
            self.s, self.d, self.vs = float(s), float(d), 0.0
            self._t_pred = self._t_upd = now
            self._dyn_since = None
            self.hits = 1
            return True
        # 연관: 예측 위치 근방 (s 게이트는 미갱신 시간·속도 비례로 확장)
        dt_upd = max(1e-3, now - self._t_upd)
        gate_s = 0.6 + (abs(self.vs) + 1.0) * min(dt_upd, 1.5)
        best = None; best_score = None
        for s, d in detections_sd:
            ds = (float(s) - self.s + 0.5 * self.L) % self.L - 0.5 * self.L
            dd = float(d) - self.d
            if abs(ds) < gate_s and abs(dd) < self.GATE_D:
                score = abs(ds) + 1.5 * abs(dd)   # 횡편차 가중 (클러터 낚임 방지)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (ds, float(s), float(d))
        if best is None:
            return False
        ds, s_m, d_m = best
        # 알파-베타 갱신.
        # 2026-08-05 혁신(innovation) 가속 클램프: 상대차의 물리 가속 한계
        # (A_OPP_MAX)로 vs 변화율을 묶는다. 종전엔 ds/dt_upd 를 그대로 먹어서,
        # 정지 장애물이 두 클러스터(가까운 면/먼 면)로 잡혀 연관이 면 사이를
        # 오가면 ds≈0.3 이 dt 0.04 s 에 들어와 vs += 0.35·(0.3/0.04) = ±2.6
        # → 즉시 클램프 포화(-1.0) → F1b 지속조건까지 통과 → LATCHED
        # (bag 15_34/15_47: vs=-0.81~-1.00, 13건 전부 이 패턴).
        # 실상대는 |가속| ≤ A_OPP_MAX 라 추적 성능 불변; 면 플립은 ±0.24/사이클
        # 진동으로 갇혀 DYN_V(0.8) 지속 돌파가 불가능해진다.
        A_OPP_MAX = 6.0   # [m/s²] 상대차 최대 가속 가정
        dvs = self.BETA * (ds / dt_upd)
        cap = A_OPP_MAX * dt_upd
        self.vs += max(-cap, min(cap, dvs))
        self.vs = max(-1.0, min(self.vs, 10.0))
        self.s = (self.s + self.ALPHA * ds) % self.L
        self.d += self.ALPHA * (d_m - self.d)
        self._t_upd = now
        self.hits += 1
        # 동적 확정 상태.
        # 2026-07-31: 확정 전/후로 비대칭을 뒤집었다. 이전에는 미확정 상태에서도
        # 스파이크 1회로 _dyn_since 가 서고, 해제만 0.7s 지속을 요구했다. 그래서
        # vs 가 오르내리면 _low_since 가 매번 리셋돼 해제가 성립하지 않고, 0.5s 뒤
        # 그대로 확정됐다 — 정지 장애물이 상대차로 잡히던 경로(bag 17_35).
        #   미확정: |vs| 가 한 번이라도 문턱 아래로 가면 즉시 리셋
        #           → 연속 DYN_SUSTAIN 동안 초과해야 확정 (스파이크 면역)
        #   확정됨: 기존대로 저속 0.7s 지속을 요구해야 해제
        #           (가림 후 재보정 시 vs 순간 dip 으로 매랩 OFF/재래치하던 것 방지)
        # 실제로 움직이는 상대는 vs 가 연속 유지되므로 판정이 달라지지 않는다.
        if abs(self.vs) > self.DYN_V:
            if self._dyn_since is None:
                self._dyn_since = now
            self._low_since = None
        else:
            if getattr(self, '_low_since', None) is None:
                self._low_since = now
            if self.confirmed_dynamic(now):
                if now - self._low_since > 0.7:
                    self._dyn_since = None
            else:
                self._dyn_since = None
        return True

    # ── 상태 조회 ──────────────────────────────────────────────────
    def confirmed_dynamic(self, now: float) -> bool:
        # 2026-08-05 신선도 게이트: 확정 **진입**은 "지금도 검출이 이어지는"
        # 대상만. 정적 장애물은 재입장 첫 0.25 s 만 트래커에 들어오는데, 그
        # 창에서 vs 가 오르면 코스팅 중에도 타이머가 차서 확정되던 구멍
        # (bag 16_01: LATCHED 4건).
        # 2026-08-05b 스티키: 신선도를 **유지 조건에도** 걸었더니 실상대가
        # 코너에서 1~2 s 가려질 때마다 확정이 풀려 래치 플래핑 55건
        # (bag 16_20). 진입에만 걸고, 한 번 확정되면 F1b 의 원래 해제
        # (저속 0.7 s 지속 → _dyn_since 리셋) 또는 트랙 소멸로만 풀린다.
        if not (self.active and self._dyn_since is not None
                and now - self._dyn_since >= self.DYN_SUSTAIN):
            self._confirmed_latch = False
            return False
        if getattr(self, '_confirmed_latch', False):
            return True                      # 유지: 가림 코스팅 관용
        if self._t_upd is not None and now - self._t_upd < 0.3:
            self._confirmed_latch = True     # 진입: 신선한 검출 필수
            return True
        return False

    def coasting(self, now: float) -> bool:
        return self.active and (now - self._t_upd) > 0.2

    def xy(self, rs, rx, ry):
        """현재 (s,d) → 맵 좌표 (raceline 배열 필요)."""
        import numpy as np
        j = int(np.argmin(np.abs(rs - (self.s % self.L))))
        j2 = (j + 1) % len(rs)
        tx, ty = rx[j2] - rx[j], ry[j2] - ry[j]
        n = math.hypot(tx, ty) + 1e-9
        return (float(rx[j] + self.d * ty / n),
                float(ry[j] - self.d * tx / n))
