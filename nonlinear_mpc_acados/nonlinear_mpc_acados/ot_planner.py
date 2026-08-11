"""추월 창 계산 + 추월 상태기계 (순수 로직, ROS/numpy 무의존).

2026-08-05 추월 재설계 (docs/superpowers/specs/2026-08-05-overtake-redesign-design.md):
구 _ot_feasible 은 D_need≈16.7 m 연속 조건을 요구했으나 실트랙(ifac) 최장
후보 구간이 5.6 m 라 통과율 0%. 창(기하 사전계산) + 상태기계로 교체.
"""
from dataclasses import dataclass


@dataclass
class Window:
    s_start: float
    s_end: float          # 랩어랩 창은 s_end > L (length 불변식 유지)
    side: int             # +1 = e_c 증가(room_up) 쪽 — side-decide 규약 동일
    v_min: float          # 창 내 ref_v 최소 (판정의 보수적 ego 속도)

    @property
    def length(self):
        return self.s_end - self.s_start


def _mk_window(i0, i1, ok, n, ds, opp_vs, adv_min, catch_dist):
    """창 조건 검사 및 Window 객체 생성 (길이·이득·속도 검증)."""
    v_min = min(ok[j % n][2] for j in range(i0, i1 + 1))
    adv = v_min - opp_vs
    if adv < adv_min:
        return []
    need = v_min * catch_dist / adv
    length = (i1 - i0 + 1) * ds
    if length < need:
        return []
    return [Window(i0 * ds, (i1 + 1) * ds, ok[i0 % n][1], v_min)]


def build_windows(sample, L, ds=0.1, opp_vs_assumed=2.5, keepout=0.35,
                  r_car=0.25, wall_margin=0.10, kappa_max=0.8,
                  adv_min=0.5, catch_dist=2.3):
    """트랙 1회 스캔으로 추월 창 목록을 만든다 (기동 시 1회).

    sample(s) -> (room_up, room_dn, abs_kappa, ref_v).
    셀 조건: max(room) ≥ keepout + r_car + wall_margin AND |κ| ≤ kappa_max.
    창 조건: adv = v_min − opp_vs_assumed ≥ adv_min AND
             length ≥ v_min · catch_dist / adv  (옆지르기 필요거리).
    """
    need_room = keepout + r_car + wall_margin
    n = int(round(L / ds))
    ok = []
    for i in range(n):
        ru, rd, k, v = sample(i * ds)
        ok.append((max(ru, rd) >= need_room and abs(k) <= kappa_max,
                   +1 if ru >= rd else -1, v))
    # 연속 run 추출 (랩 경계 병합: 시작점이 ok 면 마지막 run 과 잇는다)
    runs, cur = [], None
    for i in range(n):
        good, side, v = ok[i]
        if good and cur is None:
            cur = [i, i, side, v]
        elif good:
            cur[1] = i
            cur[3] = min(cur[3], v)
        elif cur is not None:
            runs.append(cur); cur = None
    if cur is not None:
        runs.append(cur)
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        first = runs.pop(0)
        runs[-1][1] = first[1] + n            # 랩어랩: s_end > L 로 표현
        runs[-1][3] = min(runs[-1][3], first[3])
    out = []
    for i0, i1, _side, v_min in runs:
        # side 는 창 전체의 다수결이 아니라 "최소 여유가 남는 쪽" — 셀별 side 가
        # 갈리면 보수적으로 각 side 의 최소 room 을 비교해야 하나, 실트랙에서
        # 창 내 side 반전은 관측되지 않아(직선/단일 곡선) 첫 셀 side 를 쓰고
        # 반전 시 창을 쪼갠다.
        sides = {ok[j % n][1] for j in range(i0, i1 + 1)}
        if len(sides) > 1:
            # side 반전 지점에서 분할
            split = i0
            for j in range(i0, i1 + 1):
                if ok[j % n][1] != ok[split % n][1]:
                    out.extend(_mk_window(split, j - 1, ok, n, ds,
                               opp_vs_assumed, adv_min, catch_dist))
                    split = j
            out.extend(_mk_window(split, i1, ok, n, ds,
                       opp_vs_assumed, adv_min, catch_dist))
            continue
        out.extend(_mk_window(i0, i1, ok, n, ds,
                   opp_vs_assumed, adv_min, catch_dist))
    return out


@dataclass
class Decision:
    state: str
    gap_target: object   # float | None (None = 속도캡 해제)
    r_safe: float
    kick_side: object    # int | None
    log: object          # str | None


class OtPlanner:
    """추월 상태기계. 모든 상태 출구가 _to() 하나를 지나 r_safe 복원 누락을
    구조적으로 차단한다 (스펙 4.4)."""

    COMMIT_GRACE = 2.0  # 유지검사 전용 여유 — 창끝(s9.5)+2m는 폭 1.8~2.1·κ≤0.62로 물리 무해.
                        # 진입 게이트는 grace 0. ot_live8: 조기포기 5건(gap 1.1~1.5) 복구

    def __init__(self, windows, L, keepout_r_safe=0.10, launch_gap=1.5,
                 pass_margin=0.8, cooldown=5.0, commit_gap=4.0,
                 approach_dist=8.0, abort_clear=1.0, base_gap=2.5,
                 r_safe_normal=0.3, approach_fail_cooldown=1.0,
                 adv_min_rt=0.3, engage_gap=6.0, stall_v=1.0, stall_t=0.4):
        self.ws = list(windows)
        self.L = float(L)
        self.p = dict(keepout_r_safe=keepout_r_safe, launch_gap=launch_gap,
                      pass_margin=pass_margin, cooldown=cooldown,
                      commit_gap=commit_gap, approach_dist=approach_dist,
                      abort_clear=abort_clear, base_gap=base_gap,
                      r_safe_normal=r_safe_normal,
                      approach_fail_cooldown=approach_fail_cooldown,
                      adv_min_rt=adv_min_rt, engage_gap=engage_gap,
                      stall_v=stall_v, stall_t=stall_t)
        self.state = 'TRAIL'
        self.win = None            # APPROACH/COMMIT 대상 창
        self.cool_until = -1e9     # ABORT/APPROACH-실패 쿨다운 만료 시각
        # r_safe 0.10(keepout 선축소) 적용 중 여부. APPROACH 진입에서 True,
        # 상태 무관 일반 규칙(아래 update() 상단)으로 지연 복원.
        self.r_reduced = False
        # G2(최종리뷰) + APPROACH 확장(ot_live7): COMMIT/APPROACH 중
        # ego_v < stall_v 연속 시각 시작점. None=저속 아님. APPROACH/COMMIT
        # 진입(_to)마다 리셋 — 이전 상태의 저속 이력이 새 상태에 새지 않게
        # 한다 (APPROACH 저속이 방금 진입한 COMMIT 을 오염시키는 것 방지 등).
        self._stall_since = None
        self._log = None

    # ---- 내부 유틸 ----
    def _gap(self, ego_s, opp_s):
        return (opp_s - ego_s + self.L / 2.0) % self.L - self.L / 2.0

    def _in_window(self, w, s):
        return ((s - w.s_start) % self.L) < w.length

    def _next_window(self, ego_s):
        best, bd = None, 1e18
        for w in self.ws:
            d = (w.s_start - ego_s) % self.L
            if self._in_window(w, ego_s):
                d = 0.0
            if d < bd:
                best, bd = w, d
        return best, bd

    def _feasible(self, w, opp_vs, opp_active, sigma_ok):
        """창 진입/유지 가능성 판정.

        2026-08-05 평형 커밋 재설계: 예측 산술(t_arr/t_rem/need/remaining/
        arrival-gap) 전부 삭제. 활성 재검증(ot_live3) 결과 창 내 조건
        need≤remaining 이 창의 93% 를 잠식해 진입 직후 판정을 죽였고,
        간격 압축 전제 자체가 닭-달걀(압축 수단이 COMMIT 안에 잠겨
        있어 APPROACH 중엔 압축이 일어나지 않음) 이었다. 새 정의는
        "상대가 활성이고 트래킹이 붙어 있고 런타임 상대속도 우위가
        adv_min_rt 이상"이면 feasible — 옆지르기 자체의 성사는 상태기계
        (engage_gap/commit_gap 게이트)와 최적화(자연 발생 회피)에 맡긴다.
        """
        if not (opp_active and sigma_ok) or w is None:
            return False
        return (w.v_min - opp_vs) >= self.p['adv_min_rt']

    def _pass_completable(self, w, ego_s, gap, opp_vs, grace=0.0):
        """현재 gap 에서 창 잔여 안에 앞지름(gap→−pass_margin)이 끝나는가.
        need = v_min·(gap+pass_margin)/adv — 커밋 중 ego≈v_min 가정의 보수 추정.
        (6c 가 삭제한 정적 검사의 동적 복원 — ot_live4: 불완결 커밋 21회 전부
        창끝 ABORT + 쐐기 3회. 불가능하면 커밋 안 하는 것이 완성형 판정.)
        grace: 유지검사(COMMIT)에만 허용 — 진입 게이트는 grace=0(기본값)."""
        adv = w.v_min - opp_vs
        if adv < self.p['adv_min_rt']:
            return False
        remaining = w.length - ((ego_s - w.s_start) % self.L)
        need = w.v_min * (gap + self.p['pass_margin']) / adv
        return need <= remaining + grace

    def _to(self, state, log=None):
        # 유일한 전이 통로. r_reduced 를 True 로 켜는 곳은 APPROACH 진입
        # 뿐(축소 시작) — 복원(False)은 어떤 전이도 직접 하지 않고 update()
        # 상단의 상태 무관 일반 규칙(APPROACH/COMMIT 아닌 동안 |gap|>=
        # abort_clear)에 전부 위임한다. COMMIT→DONE 도 예외 없이 이 규칙을
        # 탄다 — 즉시 복원은 abort_clear 게이트를 우회하는 취약점이었다.
        # 같은 사이클 내 연쇄 전이 시 로그 누적 (덮어쓰기 X)
        if state in ('COMMIT', 'APPROACH'):
            # G2 + APPROACH 확장: 새 COMMIT/APPROACH 마다 저속 타이머 리셋 —
            # 직전 상태(특히 COMMIT 진입 직전 APPROACH)의 저속 이력이 새
            # 상태의 stall 판정을 오염시키지 않게 한다.
            self._stall_since = None
        if log:
            self._log = ((self._log + ' ') if self._log else '') + log
        self.state = state

    # ---- 매 주기 ----
    def update(self, now, ego_s, ego_v, opp_s, opp_vs,
               opp_active, sigma_ok, qp_ok):
        # ego_v: G2(최종리뷰)에서 COMMIT 진행정체(쐐기) 판정에 다시 쓰인다.
        # opp_s: None 허용 — 트래커 비활성/부재 사이클(G3, 최종리뷰). gap
        # 계산이 불가하므로 아래에서 opp_s is None 을 상태별로 명시 처리.
        self._log = None
        kick = None
        gap = None if opp_s is None else self._gap(ego_s, opp_s)
        P = self.p

        # 상태 무관 일반 복원 규칙 (2026-08-05): APPROACH/COMMIT 이 아닌
        # 동안(=상대와 나란히 붙어 있을 위험이 낮아진 뒤) gap 이
        # abort_clear 이상 벌어지면 keepout 을 복원한다. 옛 코드는 ABORT
        # 전용이었으나 APPROACH→TRAIL 이탈도 나란히 있는 채로 일어나므로
        # 같은 규칙을 태워야 keepout 급팽창을 구조적으로 막을 수 있다.
        # (이번 사이클 진입 시점의 state 기준 — 아래에서 같은 사이클 내
        # 전이가 일어나도 그 판단 자체엔 영향 없다.)
        # G3: opp_s 부재 시 나란히 붙어 있을 위험 자체가 없으므로 gap 조건을
        # 참으로 간주 — keepout 급팽창 위험 없이 복원을 허용한다.
        _gap_clear = True if gap is None else abs(gap) >= P['abort_clear']
        if self.r_reduced and self.state not in ('APPROACH', 'COMMIT') \
                and _gap_clear:
            self.r_reduced = False
            self._log = (self._log or '') + ' [OT] r_safe 복원'

        # ABORT 처리를 먼저 — ABORT→TRAIL 전이 후 같은 사이클에서 TRAIL 재처리
        if self.state == 'ABORT':
            if (not self.r_reduced) and now >= self.cool_until:
                self._to('TRAIL', "[OT] ABORT→TRAIL (쿨다운 종료)")

        if self.state == 'TRAIL':
            w, d = self._next_window(ego_s)
            if (opp_s is not None and now >= self.cool_until
                    and d <= P['approach_dist']
                    and self._feasible(w, opp_vs, opp_active, sigma_ok)
                    and 0.0 < gap <= P['engage_gap']):
                self.win = w
                self.r_reduced = True
                self._to('APPROACH',
                         f"[OT] TRAIL→APPROACH window s{w.s_start:.1f}~"
                         f"{w.s_end:.1f} side={w.side:+d} gap={gap:.1f}")

        elif self.state == 'APPROACH':
            w = self.win
            # G2 확장(ot_live7 +337.5s): APPROACH 도 COMMIT 과 동일한
            # 진행정체(쐐기) 검출을 받는다 — keepout 축소(0.35)+간격목표 1.5
            # 로 상대에 바짝 붙는 상태라 STUCK 텔레포트에 노출된다. 단
            # 이탈은 ABORT 가 아니라 TRAIL(실패 쿨다운) — 간격목표가 base_gap
            # 로 복귀해 자연히 물러나면 충분, ABORT 의 긴 쿨다운(5s)은 과함.
            if ego_v < P['stall_v']:
                if self._stall_since is None:
                    self._stall_since = now
            else:
                self._stall_since = None
            if opp_s is None:
                # G3: 상대 소실 = 판정 자체가 불가 — 판정 상실과 동일하게
                # TRAIL 이탈 (실패 전용 쿨다운으로 재진입 채터 억제).
                self.cool_until = now + P['approach_fail_cooldown']
                self._to('TRAIL', "[OT] APPROACH→TRAIL (상대 소실)")
            else:
                d = (w.s_start - ego_s) % self.L
                past = (ego_s - (w.s_end % self.L)) % self.L < self.L / 2.0 \
                    and not self._in_window(w, ego_s) and d > P['approach_dist']
                if not self._feasible(w, opp_vs, opp_active, sigma_ok) \
                        or past or gap > P['engage_gap']:
                    # 실측 ot_live1: 1 s 간격 재진입 채터 관측 — APPROACH 실패
                    # 전용 쿨다운(ABORT 의 5.0 과 별개, 기본 1.0)으로 keepout
                    # 0.55↔0.35(정상↔선축소) 플래핑을 억제한다. 2026-08-05:
                    # gap 예측 기반 이탈은 삭제 — 창 지나침/판정 상실/engage_gap
                    # 이탈(정지 상태 등)만 남는다.
                    self.cool_until = now + P['approach_fail_cooldown']
                    self._to('TRAIL', f"[OT] APPROACH→TRAIL (판정 상실) gap={gap:.1f}")
                elif self._stall_since is not None \
                        and (now - self._stall_since) >= P['stall_t']:
                    self.cool_until = now + P['approach_fail_cooldown']
                    self._to('TRAIL', f"[OT] APPROACH→TRAIL (stall) gap={gap:.1f}")
                elif self._in_window(w, ego_s) and 0.0 < gap <= P['commit_gap'] \
                        and self._pass_completable(w, ego_s, gap, opp_vs):
                    kick = w.side
                    self._to('COMMIT',
                             f"[OT] APPROACH→COMMIT window s{w.s_start:.1f}~"
                             f"{w.s_end:.1f} side={w.side:+d} gap={gap:.1f}")

        elif self.state == 'COMMIT':
            w = self.win
            # G2: 진행정체(쐐기) ABORT — ego_v < stall_v 가 stall_t 이상
            # 연속이면 진행이 죽은 것으로 보고 텔레포트/STUCK 구출에 앞서
            # ABORT 한다 (원칙 4: 창 이탈 없이 정지해 있는 채로 방치 금지).
            if ego_v < P['stall_v']:
                if self._stall_since is None:
                    self._stall_since = now
            else:
                self._stall_since = None
            if opp_s is None:
                # G3: 기존 not opp_active ABORT 경로 재사용 — 상대 소실도
                # 같은 사유(추종 대상 상실)로 취급.
                self.cool_until = now + P['cooldown']
                self._to('ABORT', "[OT] COMMIT→ABORT (상대 소실)")
            elif gap <= -P['pass_margin']:
                # r_reduced 는 여기서 건드리지 않는다 — DONE 은 상태 전이만
                # 하고 복원은 상단 일반 규칙(|gap|>=abort_clear)에 위임.
                # 2026-08-05 리뷰: 즉시 복원은 abort_clear 게이트를 우회해
                # pass_margin(라이브 파라미터) 이 작을 때 근접 keepout
                # 급팽창을 허용하는 취약점이었다.
                self._to('TRAIL', f"[OT] COMMIT→DONE gap={gap:.1f}")
            elif (not qp_ok) or (not opp_active) \
                    or not self._in_window(w, ego_s):
                self.cool_until = now + P['cooldown']
                self._to('ABORT',
                         f"[OT] COMMIT→ABORT gap={gap:.1f} qp_ok={qp_ok} "
                         f"opp={opp_active}")
            elif self._stall_since is not None \
                    and (now - self._stall_since) >= P['stall_t']:
                self.cool_until = now + P['cooldown']
                self._to('ABORT', f"[OT] COMMIT→ABORT gap={gap:.1f} stall")
            elif gap > 0.0 and not self._pass_completable(w, ego_s, gap, opp_vs,
                                                         grace=self.COMMIT_GRACE):
                # gap<=0 (이미 옆지름 진행 중)이면 need 산식이 의미를 잃으므로
                # 건너뛴다 — DONE(gap<=-pass_margin) 판정이 항상 이 앞에서
                # 먼저 걸린다. 커밋 후 상대가 가속해 완주 불가능해지면 조기
                # 복귀(ot_live4: 불완결 커밋 반복 방지).
                self.cool_until = now + P['cooldown']
                self._to('ABORT', f"[OT] COMMIT→ABORT gap={gap:.1f} infeasible")

        r_safe = P['keepout_r_safe'] if self.r_reduced else P['r_safe_normal']
        gap_t = {'TRAIL': P['base_gap'], 'APPROACH': P['launch_gap'],
                 'COMMIT': None, 'ABORT': P['base_gap']}[self.state]
        return Decision(self.state, gap_t, r_safe, kick, self._log)
