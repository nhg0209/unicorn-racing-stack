"""speed_profile_learner.py — 온라인 랩단위 ref_v/a_lat 스케일 학습 (bin 테이블).

v2 (2026-07-24, 방향 B): 실증-엔벨로프 추종(v1)이 랩타임을 악화시켜 성과
기반 hill-climb 로 교체. 하한 = 1.0 (baseline) — 학습이 스케일을 baseline
아래로 내리지 않는다. 랩타임 프록시(`len(buf)·ctrl_dt`)가 기존 best 대비
`lt_tol` 이상 악화되면 직전 상향을 되돌리고 `raise_cooldown_laps` 랩 동안
재시도를 쉰다.

바운드 (스펙 §3):
  0) hill-climb: 상향은 현 목표를 실제로 추종 중(+코너는 grip 활용 중)인
     bin에만, 랩타임이 나빠지면 즉시 되돌림 (regression revert)
  1) 하한 고정: base·1.0 (baseline) 아래로는 절대 안 내림
  2) ILC: 랩당 스케일 변화 |Δ| ≤ step_delta
  3) 마찰 천장: base·scale ≤ √(μ·g/κ)
실패 랩: 사고 s ± incident_margin_m 구간을 마지막 성공 스냅샷으로 롤백.

랩 종료(s 큰 폭 하락 > lap_jump)·텔레포트(원형 |Δs| > teleport_jump)·
스톨(vx < stall_vx 가 stall_cycles 연속)을 자체 감지 — ROS 무의존.

스펙: docs/superpowers/specs/2026-07-24-refv-online-learning-design.md

min_samples 기본값 참고: 랩당 bin당 샘플수 ≈ bin_width·ctrl_rate/vx. 25 Hz
제어주기·bin_width=0.5 m·주행속도 4~8 m/s 기준 ≈ 0.5·25/(4~8) ≈ 1.6~3.1개
뿐이라, 기본값을 5로 두면 대부분의 bin이 표본 부족으로 영구히 학습되지
않는다. 그래서 기본값은 2 — 안전 상한은 min_samples가 아니라 step_delta·
마찰천장·feasibility pass·rollback 3+1층 바운드가 담당한다.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import numpy as np

from ...track_loader import velocity_feasibility_pass

G = 9.81
_FMT_VERSION = 1


class SpeedProfileLearner:
    def __init__(self, L, kappa_of_s, base_refv_of_s, *,
                 ctrl_dt=0.04, bin_width=0.5, step_delta=0.05,
                 refv_scale_min=1.0, refv_scale_max=1.5,
                 alat_scale_min=1.0, alat_scale_max=1.3,
                 a_lat_base=6.0, mu=1.0, min_samples=2,
                 a_brake=3.0, smooth_win=3, hist_laps=4,
                 incident_margin_m=3.0,
                 stall_vx=0.3, stall_cycles=25,
                 lap_jump=5.0, teleport_jump=3.0,
                 g_lo=0.8, kappa_corner=0.05,
                 lt_tol=0.02, raise_cooldown_laps=2,
                 min_margin_m=0.35, straight_kappa=0.02):
        self.L = float(L)
        self.bin_width = float(bin_width)
        self.nb = max(1, int(np.ceil(self.L / self.bin_width)))
        self.centers = (np.arange(self.nb) + 0.5) * self.bin_width
        self.kappa = np.array([max(0.0, float(kappa_of_s(s % self.L)))
                               for s in self.centers])
        self.base = np.array([max(0.1, float(base_refv_of_s(s % self.L)))
                              for s in self.centers])
        self.ctrl_dt = float(ctrl_dt)
        self.step_delta = float(step_delta)
        self.refv_lim = (float(refv_scale_min), float(refv_scale_max))
        self.alat_lim = (float(alat_scale_min), float(alat_scale_max))
        self.a_lat_base = float(a_lat_base)
        self.mu = float(mu)
        self.min_samples = int(min_samples)
        self.a_brake = float(a_brake)
        self.smooth_win = int(smooth_win)
        self.incident_margin_m = float(incident_margin_m)
        self.stall_vx = float(stall_vx)
        self.stall_cycles = int(stall_cycles)
        self.lap_jump = float(lap_jump)
        self.teleport_jump = float(teleport_jump)
        self.g_lo = float(g_lo)
        self.kappa_corner = float(kappa_corner)
        self.lt_tol = float(lt_tol)
        self.raise_cooldown_laps = int(raise_cooldown_laps)
        # 방향성 벽여유 게이트 (task-8, 2026-07-27): ifac s≈14 실측 벽충돌 —
        # 그립기반 게이트(위 refv_gate/alat_gate)는 저곡률 구간(κ≈0.1,
        # a_lat 여유)을 "안전"으로 오판하지만 실제로는 커브 바깥쪽(=드리프트
        # 방향) 벽까지 0.31 m 뿐이었다. min_margin_m 미만인 드리프트-쪽 벽
        # 여유 bin 은 상향에서 제외 (하향은 건드리지 않음).
        self.min_margin_m = float(min_margin_m)
        self.straight_kappa = float(straight_kappa)

        self.refv_scale = np.ones(self.nb)
        self.alat_scale = np.ones(self.nb)
        self._good_refv = self.refv_scale.copy()
        self._good_alat = self.alat_scale.copy()
        self._vx_hist = [deque(maxlen=hist_laps) for _ in range(self.nb)]
        self._alat_hist = [deque(maxlen=hist_laps) for _ in range(self.nb)]
        self.a_acc = self.a_brake          # 실측 가속 엔벨로프 EMA (보수 초기값)
        self.laps_learned = 0
        self.best_lt = None                # clean 랩 최소 랩타임(프록시)
        self._last_action = None           # 'raise' | None
        self._cooldown = 0                 # 상향 재시도 쉬는 랩 수

        self._buf_s: list[float] = []
        self._buf_vx: list[float] = []
        self._buf_alat: list[float] = []
        self._buf_margin_left: list[float] = []
        self._buf_margin_right: list[float] = []
        self._buf_kappa_signed: list[float] = []
        self._incident_s: list[float] = []
        self._prev_s: float | None = None
        self._stall_cnt = 0

    # ── 조회 ────────────────────────────────────────────────
    def _bin_of(self, s: float) -> int:
        return min(self.nb - 1, int((s % self.L) / self.bin_width))

    def refv_scale_at(self, s: float) -> float:
        return float(self.refv_scale[self._bin_of(s)])

    def alat_scale_at(self, s: float) -> float:
        return float(self.alat_scale[self._bin_of(s)])

    # ── 기록 + 자체 이벤트 감지 ──────────────────────────────
    def record(self, s, vx, a_lat, incident=False,
               margin_left=None, margin_right=None, kappa_signed=None):
        s_m = float(s) % self.L
        vx = float(vx)
        event = None
        if self._prev_s is not None:
            drop = self._prev_s - s_m
            circ = (s_m - self._prev_s + 0.5 * self.L) % self.L - 0.5 * self.L
            if drop > self.lap_jump:                 # 랩 rollover (logger 규약)
                event = self._finish_lap()
            elif abs(circ) > self.teleport_jump:     # 리스폰/텔레포트
                self._incident_s.append(self._prev_s)
        if vx < self.stall_vx:
            self._stall_cnt += 1
            if self._stall_cnt == self.stall_cycles:
                self._incident_s.append(s_m)
        else:
            self._stall_cnt = 0
        if incident:
            self._incident_s.append(s_m)
        self._buf_s.append(s_m)
        self._buf_vx.append(vx)
        self._buf_alat.append(abs(float(a_lat)))
        self._buf_margin_left.append(float(margin_left) if margin_left is not None else np.nan)
        self._buf_margin_right.append(float(margin_right) if margin_right is not None else np.nan)
        self._buf_kappa_signed.append(float(kappa_signed) if kappa_signed is not None else np.nan)
        self._prev_s = s_m
        return event

    # ── 랩 처리 ────────────────────────────────────────────
    def _finish_lap(self):
        lt = len(self._buf_s) * self.ctrl_dt   # 랩타임 프록시 (25Hz 샘플수)
        try:
            if self._buf_s:
                idx = np.clip((np.asarray(self._buf_s) / self.bin_width)
                              .astype(int), 0, self.nb - 1)
                coverage = len(np.unique(idx)) / self.nb
            else:
                coverage = 0.0
            if coverage < 0.9:
                # 트랙 일부만 도는 부분 랩(세션 첫 랩이 중간에서 시작하는 등)
                # — best_lt/학습/쿨다운 어느 것도 건드리지 않고 그냥 버린다.
                return 'lap_partial'
            clean = not self._incident_s
            if clean:
                self._learn(lt)
                self.laps_learned += 1
                ev = 'lap_clean'
            else:
                self._rollback()
                self._cooldown = self.raise_cooldown_laps
                ev = 'lap_dirty'
        finally:
            self._buf_s.clear(); self._buf_vx.clear(); self._buf_alat.clear()
            self._buf_margin_left.clear(); self._buf_margin_right.clear()
            self._buf_kappa_signed.clear()
            self._incident_s = []
            self._stall_cnt = 0
        return ev

    def _learn(self, lt):
        s = np.asarray(self._buf_s); vx = np.asarray(self._buf_vx)
        alat = np.asarray(self._buf_alat)
        mleft = np.asarray(self._buf_margin_left)
        mright = np.asarray(self._buf_margin_right)
        ksig = np.asarray(self._buf_kappa_signed)
        # 실측 가속 엔벨로프 (양의 dv/dt p75, EMA) — 직선 forward pass 용
        dv = np.diff(vx) / self.ctrl_dt
        pos = dv[(dv > 0.2) & (dv < 15.0)]
        if pos.size >= 10:
            self.a_acc = float(np.clip(0.7 * self.a_acc + 0.3 * np.percentile(pos, 75),
                                       0.5, 10.0))
        # bin 집계(이번 랩 vx/|a_lat| p90) → 히스토리 push (유지) + 이번 랩 값 보관
        idx = np.clip((s / self.bin_width).astype(int), 0, self.nb - 1)
        v90 = np.full(self.nb, np.nan)
        alat90 = np.full(self.nb, np.nan)
        for b in range(self.nb):
            m = idx == b
            if int(m.sum()) >= self.min_samples:
                v90[b] = float(np.percentile(vx[m], 90))
                alat90[b] = float(np.percentile(alat[m], 90))
                self._vx_hist[b].append(v90[b])
                self._alat_hist[b].append(alat90[b])

        # ── 방향성 벽여유 (task-8) — 커브 "바깥쪽"(=드리프트 방향) 벽여유만
        # 골라 per-sample → per-bin 최솟값(최악 케이스)으로 집계. 좌회전
        # (kappa_signed>+straight_kappa)이면 바깥쪽=오른쪽 → margin_right,
        # 우회전(<-straight_kappa)이면 바깥쪽=왼쪽 → margin_left, 직선이면
        # min(양쪽) — 어느 쪽이든 여유 부족하면 상향 보류. kappa_signed 가
        # NaN(미제공)인 샘플은 비교식이 전부 False 라 직선 분기(else)로
        # 빠지고, margin 도 NaN 이면 그대로 NaN → 데이터 없음(하위호환).
        # 2026-08-03: 커브에서도 min(양쪽) — 바깥쪽만 보던 방향성 게이트의
        # 사각지대 수정. 실측(bag 20_24): s23.2 좌회전 구간에서 게이트가
        # 바깥(우측, 여유 0.85+)만 보고 상향 허용 → scale 1.394 → **안쪽
        # (에이펙스, 좌측) 클립 2회**. 안쪽 여유(≈0.32)를 봤다면 동결됐다.
        # 에이펙스 클립은 바깥 드리프트만큼 흔한 실패 모드다.
        drift_margin = np.minimum(mleft, mright)
        _ = ksig  # kappa_signed 는 게이트에 더는 안 쓰지만 record 서명 유지
        margin_bin = np.full(self.nb, np.nan)
        for b in range(self.nb):
            m = idx == b
            if np.any(m):
                vals = drift_margin[m]
                if np.any(~np.isnan(vals)):
                    margin_bin[b] = float(np.nanmin(vals))
        # 게이트: 데이터 없음(NaN)=차단 안 함(기존 동작 보존), 있으면
        # min_margin_m 이상일 때만 상향 허용. 하향은 절대 건드리지 않는다.
        margin_ok = np.isnan(margin_bin) | (margin_bin >= self.min_margin_m)

        # ── 성과(랩타임) 판정 ────────────────────────────────
        if (self._last_action == 'raise' and self.best_lt is not None
                and lt > self.best_lt * (1.0 + self.lt_tol)):
            # 악화 → 상향 직전 스냅샷으로 전체 복원, best 갱신 안 함
            self.refv_scale = self._good_refv.copy()
            self.alat_scale = self._good_alat.copy()
            self._cooldown = self.raise_cooldown_laps
            self._last_action = None
            return

        self.best_lt = lt if self.best_lt is None else min(self.best_lt, lt)

        if self._cooldown > 0:
            self._cooldown -= 1
            return

        # ── 상향 시도 ────────────────────────────────────────
        # ★ 스냅샷은 이번 랩 갱신 "이전" 값 — dirty 랩 롤백/랩타임 악화 복원이
        # 마지막 상향을 실제로 되돌리게 하기 위함.
        prev_refv = self.refv_scale.copy()
        prev_alat = self.alat_scale.copy()
        self._good_refv = prev_refv
        self._good_alat = prev_alat

        have = ~np.isnan(v90)
        # refv: 현 목표를 실제로 추종 중이고(+ 드리프트쪽 벽여유 충분한) bin만 상향
        refv_gate = have & (v90 >= 0.9 * self.base * prev_refv) & margin_ok
        # alat: 코너(κ>임계) 이면서 cap 이 의미있게 쓰이는(활용률≥0.6) +
        # 드리프트쪽 벽여유 충분한 bin만 상향
        alat_gate = (have & (self.kappa > self.kappa_corner)
                     & (alat90 >= 0.6 * self.a_lat_base * prev_alat)
                     & margin_ok)

        d = self.step_delta
        self.refv_scale = prev_refv.copy()
        self.refv_scale[refv_gate] += d
        self.alat_scale = prev_alat.copy()
        self.alat_scale[alat_gate] += d
        # 스무딩 (이동평균, ~1.5 m)
        if self.smooth_win > 1:
            k = np.ones(self.smooth_win) / self.smooth_win
            pad = self.smooth_win // 2
            for arr in (self.refv_scale, self.alat_scale):
                ext = np.concatenate([arr[-pad:], arr, arr[:pad]])
                arr[:] = np.convolve(ext, k, mode='valid')[:self.nb]
        # 마찰 천장: base·scale ≤ √(μ·g/κ)
        with np.errstate(divide='ignore'):
            v_ceil = np.where(self.kappa > 1e-3,
                              np.sqrt(self.mu * G / np.maximum(self.kappa, 1e-9)),
                              np.inf)
        self.refv_scale = np.minimum(self.refv_scale, v_ceil / self.base)
        # brake/accel feasibility pass (Task 1 공유 함수)
        v = self.base * self.refv_scale
        ds = np.full(self.nb, self.bin_width)
        v = velocity_feasibility_pass(v, ds, a_brake=self.a_brake,
                                      a_accel=self.a_acc)
        self.refv_scale = v / self.base
        # ILC 상한 재클램프: 스무딩(과 위의 마찰천장/feasibility pass)은 bin별로
        # 독립적으로 값을 미는 게 아니라 이웃과 평균/전파하므로, rollback이나
        # 천장이 만든 gap 옆 bin이 랩당 step_delta보다 더 "상향"될 수 있다.
        # 하향은 안전 방향(천장·feasibility·rollback이 강제로 눌러야 하므로)
        # 이라 제한하지 않고, 상향만 이번 랩 시작 값(prev_*) 기준 +step_delta로
        # 다시 눌러준다 — 비대칭 클램프.
        self.refv_scale = np.minimum(self.refv_scale, prev_refv + self.step_delta)
        self.alat_scale = np.minimum(self.alat_scale, prev_alat + self.step_delta)
        # 2차 안전 클램프
        self.refv_scale = np.clip(self.refv_scale, *self.refv_lim)
        self.alat_scale = np.clip(self.alat_scale, *self.alat_lim)
        if not (np.all(np.isfinite(self.refv_scale))
                and np.all(np.isfinite(self.alat_scale))):
            self.refv_scale = prev_refv
            self.alat_scale = prev_alat
            return
        self._last_action = 'raise'

    def _rollback(self):
        # AIMD: additive increase, on-incident decrease. 단순 복원(_good_*)은
        # 이미 마진값인 _good_*을 되돌려도 다음 사고를 못 막는다(no-op 재발) →
        # 사고 구간은 (직전값, last-good) 중 작은 쪽보다 한 단계(-step_delta)
        # 더 내리고, floor(=refv_lim[0]/alat_lim[0], 기본 1.0)에서 멈춘다.
        # _good_*도 낮춘 값으로 갱신해 이후 클린랩 상향이 낮아진 지점부터
        # 재시작하고, 나중의 랩타임-악화 복원이 이 구간을 다시 마진값으로
        # 되돌리지 못하게 한다.
        nbins = int(np.ceil(self.incident_margin_m / self.bin_width))
        refv_floor = self.refv_lim[0]
        alat_floor = self.alat_lim[0]
        for s_i in self._incident_s:
            b = self._bin_of(s_i)
            for k in range(b - nbins, b + nbins + 1):
                kk = k % self.nb
                lo_refv = min(self._good_refv[kk], self.refv_scale[kk]) - self.step_delta
                lo_alat = min(self._good_alat[kk], self.alat_scale[kk]) - self.step_delta
                self.refv_scale[kk] = max(refv_floor, lo_refv)
                self.alat_scale[kk] = max(alat_floor, lo_alat)
                self._good_refv[kk] = self.refv_scale[kk]
                self._good_alat[kk] = self.alat_scale[kk]

    # ── persistence ─────────────────────────────────────────
    def save(self, path):
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, version=_FMT_VERSION, L=self.L, bin_width=self.bin_width,
                 refv_scale=self.refv_scale, alat_scale=self.alat_scale,
                 a_acc=self.a_acc, laps_learned=self.laps_learned)

    def load(self, path) -> bool:
        p = Path(path).expanduser()
        if not p.is_file():
            return False
        try:
            d = np.load(p, allow_pickle=False)
            if (abs(float(d['L']) - self.L) > 0.5
                    or abs(float(d['bin_width']) - self.bin_width) > 1e-6
                    or d['refv_scale'].shape != (self.nb,)
                    or d['alat_scale'].shape != (self.nb,)):
                return False
            rs = np.asarray(d['refv_scale'], float)
            al = np.asarray(d['alat_scale'], float)
            a_acc_in = float(d['a_acc'])
            if not (np.all(np.isfinite(rs)) and np.all(np.isfinite(al))
                    and np.isfinite(a_acc_in)):
                return False
            self.refv_scale = np.clip(rs, *self.refv_lim)
            self.alat_scale = np.clip(al, *self.alat_lim)
            self._good_refv = self.refv_scale.copy()
            self._good_alat = self.alat_scale.copy()
            self.a_acc = float(np.clip(a_acc_in, 0.5, 10.0))
            self.laps_learned = int(d['laps_learned'])
            return True
        except Exception:
            return False
