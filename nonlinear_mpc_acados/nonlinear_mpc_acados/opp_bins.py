"""OppBins — 상대차 s-bin 통계 학습기 (0723 GP-bins 설계, torch 의존성 0).

트랙 진행거리 s를 0.5m bin으로 나눠 상대의 횡위치 d(s)·속도 vs(s)를
Welford 러닝 평균/분산으로 축적한다. FSDP sparse-GP의 본질(라인+속도 프로파일
+불확실성)을 learned_refv 패턴으로 대체. 세션 메모리 전용(상대는 매번 다름).
"""
import numpy as np


class OppBins:
    def __init__(self, track_len: float, bin_size: float = 0.5):
        self.L = float(track_len)
        self.bs = float(bin_size)
        self.nb = max(1, int(round(self.L / self.bs)))
        self.n = np.zeros(self.nb, dtype=int)
        self.d_mean = np.zeros(self.nb)
        self.d_M2 = np.zeros(self.nb)
        self.v_mean = np.zeros(self.nb)
        self.v_M2 = np.zeros(self.nb)
        self.lap_seen = np.zeros(self.nb, dtype=int)   # 서로 다른 랩에서 본 횟수
        self._last_lap = np.full(self.nb, -1, dtype=int)
        self._lap_idx = 0
        self._prev_s = None

    def _bin(self, s: float) -> int:
        return int((s % self.L) / self.bs) % self.nb

    def add(self, s: float, d: float, vs: float) -> None:
        """상대 관측 1건 추가 (s, 횡위치 d, 진행속도 vs)."""
        if self._prev_s is not None and (self._prev_s - s) > 0.5 * self.L:
            self._lap_idx += 1   # raw s 급감 = 랩 경계 통과
        self._prev_s = s
        i = self._bin(s)
        self.n[i] += 1
        for mean, M2, x in ((self.d_mean, self.d_M2, d),
                            (self.v_mean, self.v_M2, vs)):
            delta = x - mean[i]
            mean[i] += delta / self.n[i]
            M2[i] += delta * (x - mean[i])
        if self._last_lap[i] != self._lap_idx:
            self._last_lap[i] = self._lap_idx
            self.lap_seen[i] += 1

    # ── 신뢰도 게이트 ────────────────────────────────────────────
    def coverage(self) -> float:
        """관측된 bin 중 2랩 이상에서 본 bin 비율."""
        seen = self.lap_seen >= 1
        if not seen.any():
            return 0.0
        return float((self.lap_seen[seen] >= 2).mean())

    def v_sigma_median(self) -> float:
        m = self.n >= 3
        if not m.any():
            return float('inf')
        return float(np.median(np.sqrt(self.v_M2[m] / self.n[m])))

    def ready(self, cov_thr: float = 0.8, sig_thr: float = 0.5) -> bool:
        return self.coverage() >= cov_thr and self.v_sigma_median() <= sig_thr

    def decay(self) -> None:
        """게이트 OFF 복귀 시: 분산 신뢰 축소(재학습 유도) — 평균은 유지."""
        self.lap_seen = np.minimum(self.lap_seen, 1)
        self.d_M2 *= 2.0   # 분산 부풀림 → ready 재충족에 새 관측 필요
        self.v_M2 *= 2.0

    # ── 예측 조회 ───────────────────────────────────────────────
    def query(self, s: float):
        """(d, vs) 학습값. 관측 부족 bin은 None."""
        i = self._bin(s)
        if self.n[i] < 3:
            return None
        return float(self.d_mean[i]), float(self.v_mean[i])

    def predict_xy(self, rs, rx, ry, s0: float, times, fallback_v: float):
        """per-stage 예측 위치: 학습 vs(s) 적분 전진 + 학습 라인 d(s) 오프셋.
        미학습 구간은 fallback_v 등속 + d=0 (raceline)."""
        out = []
        s_cur = float(s0)
        t_prev = 0.0
        for t in times:
            dt = float(t) - t_prev
            t_prev = float(t)
            q = self.query(s_cur)
            v = q[1] if q is not None else float(fallback_v)
            s_cur = (s_cur + max(0.0, v) * dt) % self.L
            q2 = self.query(s_cur)
            d = q2[0] if q2 is not None else 0.0
            j = int(np.argmin(np.abs(rs - s_cur)))
            j2 = (j + 1) % len(rs)
            tx, ty = rx[j2] - rx[j], ry[j2] - ry[j]
            nrm = float(np.hypot(tx, ty)) + 1e-9
            # +e_c 축 = (sin, -cos) = (ty, -tx)/n  (solver 관례와 동일)
            out.append((float(rx[j] + d * ty / nrm),
                        float(ry[j] - d * tx / nrm)))
        return out
