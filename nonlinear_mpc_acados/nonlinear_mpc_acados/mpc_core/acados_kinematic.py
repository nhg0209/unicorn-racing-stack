#!/usr/bin/env python3
"""acados port of EVO-MPCC base — 4-state Cartesian kinematic, EXTERNAL cost.

Identical model to Nonlinear_MPC.py (IPOPT) so the ROS node can swap
backends without re-tuning. Difference is the solver: SQP_RTI + HPIPM
runs in 1-5 ms per cycle vs IPOPT's 18-70 ms.

State (4):  [x, y, psi, s]
Input (3):  [v, delta, p]   (direct controls, NOT rates)

Per-stage parameters (model.p), set each cycle from the ROS node:
  [obs_dmin, obs_x, obs_y, side_pref,
   q_cte, q_lag, q_d_delta, R_safe, M_slack,
   left_x, left_y, right_x, right_y]

Track / centerline / boundaries supplied as CasADi spline interpolants
(self.center_lut_x(s) etc.) — same interface the IPOPT version uses, set
via set_track_data() before setup_MPC().
"""
import os
import math
import time
import numpy as np
import casadi as ca
import scipy.linalg
from ._ros_compat import NullLogger, monotonic_now, yaw_to_quat
from .model_policy import (A_MIN_DYN, clamp_a_lat_to_grip, decide_side_window,
                           grip_a_lat_limit)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


#: 슬랙 벌점 행 순서 — acados h 벡터의 slacked 행(idxsh)과 1:1 대응.
_SLACK_ROWS = ('obstacle', 'corridor', 'corridor', 'a_lat', 'ellipse',
               'obstacle2', 'obstacle3')

#: config/mpc_tuning.yaml 이 없거나 깨졌을 때 쓰는 기본값 (파일과 같은 값).
_SLACK_FALLBACK = {
    'obstacle':  {'zl': 200.0, 'zu': 40.0, 'Zl': 120.0, 'Zu': 30.0},
    'corridor':  {'zl': 200.0, 'zu': 20.0, 'Zl': 120.0, 'Zu': 15.0},
    'a_lat':     {'zl': 50.0,  'zu': 50.0, 'Zl': 15.0,  'Zu': 15.0},
    'ellipse':   {'zl': 50.0,  'zu': 50.0, 'Zl': 15.0,  'Zu': 15.0},
    'obstacle2': {'zl': 200.0, 'zu': 40.0, 'Zl': 120.0, 'Zu': 30.0},
    'obstacle3': {'zl': 200.0, 'zu': 40.0, 'Zl': 120.0, 'Zu': 30.0},
}


#: cost_weights / obstacle 섹션 기본값 (mpc_tuning.yaml 과 같은 값).
_COST_FALLBACK = {
    'q_cte': 15.0, 'q_lag': 80.0, 'q_psi': 10.0, 'q_v': 12.0, 'q_dd': 5.0,
    'q_p': 1.0, 'q_side': 3.0, 'q_d_rate': 80.0, 'q_dv': 15.0,
    'terminal_cte': 5.0, 'terminal_lag': 5.0, 'terminal_psi': 4.0,
    'terminal_v': 3.0,
}
_OBS_FALLBACK = {
    'big_m': 5.0, 'sigma_gate': 4.5, 'sigma_side': 0.5, 'line_attenuation': 0.70,
}
_COR_FALLBACK = {
    'r_car_straight': 0.24, 'kappa_ref': 0.5,
}
_AVOID_FALLBACK = {
    'release_fade_end': 3.0,
    'commit_assoc_radius': 0.6,
    'commit_follow_alpha': 0.3,
    'commit_miss_release_s': 0.3,
}

def _tuning_path():
    """mpc_tuning.yaml 경로. 소스트리 → ament share 순으로 찾는다.

    2026-07-29: 노드는 build/ 트리에서 실행된다(egg-link). build/config 에는
    빌드 시점에 만들어진 심링크만 있어서 새 파일은 없다 — __file__ 상대경로만
    쓰면 조용히 폴백값으로 돌아 "파일을 고쳤는데 안 먹는" 함정이 된다.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.normpath(os.path.join(here, '..', '..', 'config', 'mpc_tuning.yaml'))]
    try:
        from ament_index_python.packages import get_package_share_directory
        cands.append(os.path.realpath(os.path.join(
            get_package_share_directory('nonlinear_mpc_acados'),
            'config', 'mpc_tuning.yaml')))
    except Exception:  # noqa: BLE001
        pass
    # build/ 에서 돌 때: 같은 config 디렉터리의 다른 파일이 소스로 심링크돼
    # 있으므로 그 실경로 옆에서 찾는다.
    sib = os.path.normpath(os.path.join(here, '..', '..', 'config',
                                        'ddrx_unified_params.yaml'))
    if os.path.islink(sib) or os.path.exists(sib):
        cands.append(os.path.join(os.path.dirname(os.path.realpath(sib)),
                                  'mpc_tuning.yaml'))
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


_TUNING_PATH = _tuning_path()


def load_tuning(log=None):
    """config/mpc_tuning.yaml → {'slack': {...}, 'cost': {...}, 'obs': {...}}.

    제어 거동을 실제로 정하는 값들이라 코드에 박아두지 않고 파일로 뺐다
    (2026-07-29). 파일이 없거나 키가 빠지면 그 항목만 기본값으로 폴백한다 —
    튜닝 파일 하나 때문에 제어 노드가 못 뜨는 일은 없어야 한다.
    """
    cfg = {}
    try:
        import yaml
        with open(_TUNING_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        if log:
            log.info(f"[tuning] {_TUNING_PATH} 로드")
    except Exception as e:  # noqa: BLE001 — 없으면 기본값으로 계속
        if log:
            log.warn(f"[tuning] {_TUNING_PATH} 로드 실패({e}) — 코드 기본값 사용")
    sl = cfg.get('slack') or {}
    slack = {k: np.array([float((sl.get(row) or {}).get(k, _SLACK_FALLBACK[row][k]))
                          for row in _SLACK_ROWS])
             for k in ('zl', 'zu', 'Zl', 'Zu')}
    cw = cfg.get('cost_weights') or {}
    cost = {k: float(cw.get(k, d)) for k, d in _COST_FALLBACK.items()}
    ob = cfg.get('obstacle') or {}
    obs = {k: float(ob.get(k, d)) for k, d in _OBS_FALLBACK.items()}
    co = cfg.get('corridor') or {}
    cor = {k: float(co.get(k, d)) for k, d in _COR_FALLBACK.items()}
    av = cfg.get('avoidance') or {}
    avoid = {k: float(av.get(k, d)) for k, d in _AVOID_FALLBACK.items()}
    if log:
        log.info(f"[tuning] corridor slack zl={slack['zl'][1]:.0f}/Zl={slack['Zl'][1]:.0f}, "
                 f"q_cte={cost['q_cte']:.1f} q_lag={cost['q_lag']:.1f}, "
                 f"BIG_OBS={obs['big_m']:.1f} SIGMA_OBS={obs['sigma_gate']:.1f}, "
                 f"R_car(직선)={cor['r_car_straight']:.2f}@κ_ref={cor['kappa_ref']:.2f}")
    return {'slack': slack, 'cost': cost, 'obs': obs, 'corridor': cor,
            'avoid': avoid}


def codegen_paths(use_dynamic, nx_solver, dyn_mu=None):
    """Return (export_dir, json_path) keyed to the OCP structure.

    Switching kinematic<->dynamic (2026-06-10 unified layout: BOTH nx=8 — the
    dyn/kin tag alone separates f_kin vs blended-Pacejka codegens) changes the
    generated solver. A FIXED export dir made acados re-use the previous
    codegen and silently run the WRONG model — the recurring "stale codegen"
    trap. Keying the dir/json to (model, nx) makes that axis safe (and same
    config warm-reuses its build instead of regenerating). Tags: dyn8 / kin8.

    NOTE: only the model/nx axis is keyed. Other codegen-time knobs —
    N_horizon, dT, speed_target, lookahead_m, baked cost weights — are NOT in
    the tag, so changing only those reuses the same dir. Callers that vary them
    (BO/sweep harnesses) must still `rm -rf ~/.acados_codegen/*`
    between runs to force regeneration.
    dyn_mu IS keyed (2026-06-10).
    """
    tag = f"{'dyn' if use_dynamic else 'kin'}{int(nx_solver)}"
    if dyn_mu is not None:
        # μ is baked into the tanh tire + friction-ellipse codegen — key it
        # so switching dyn_mu can never reuse a stale build (mu0p600 etc.).
        tag += f"_mu{float(dyn_mu):.3f}".replace('.', 'p')
    # 2026-07-16: /tmp -> 홈 (macOS 재부팅마다 /tmp 소실 = 매번 풀빌드 수 분).
    _base = os.path.expanduser("~/.acados_codegen")
    os.makedirs(_base, exist_ok=True)
    return (f"{_base}/acados_codegen_evompcc_{tag}",
            f"{_base}/acados_ocp_evompcc_{tag}.json")


def rebase_angles_to_ref(angles, ref):
    """2π-주기 각 배열을 ref 근방 브랜치로 리베이스 (반환 각 − ref ∈ [−π, π]).

    H10 (2026-07-03): solve()는 ψ를 unwrap(랩당 ±2π 누적)하는데 LMPC SS 교사의
    ψ는 wrapped 로 패킹됐다. joint-α anchor 의 ψ 잔차는 raw 차이라(2π-주기 아님)
    랩 k에서 2πk 로 자라 opti≈(2πk)² 발산 (실차 k=2/4/7 → 156/610/1866, 오프라인
    재현 1% 이내). SS ψ를 pack 시 현재 unwrapped ψ 브랜치로 옮기면 해소.
    """
    a = np.asarray(angles, dtype=float)
    return a + 2.0 * np.pi * np.round((float(ref) - a) / (2.0 * np.pi))


class MPC:
    """acados backend mirroring the IPOPT EVO-MPCC interface."""

    def __init__(self, cost_type, system_model, logger=None):
        self._log = logger if logger is not None else NullLogger("[MPC-acados]")
        self.arch = "acados_evompcc"

        # Sizing — set in set_initial_params()
        self.N = None
        self.dT = None
        self.n_states = 4
        self.n_controls = 3
        # Wheelbase from yaml
        self.L = 0.307

        # ──────────────── Dynamic model (Phase 1: toggle + params) ────────
        # Default OFF: kinematic bicycle (current working configuration).
        # Switch via env var or class override; full dynamic adapter
        # comes online in Phase 2 (cost/output rewiring).
        self.use_dynamic = False
        # nlp_solver_iters (2026-06-19): 1 → SQP_RTI (default, real-time stable).
        # >1 → SQP with CAPPED max_iter + MERIT_BACKTRACKING globalization, to
        # escape the documented "1-iter RTI can't rebuild the dual active set"
        # low-speed trap WITHOUT the unbounded-step startup divergence that
        # uncapped multi-iter SQP hit. Set onto self before setup_MPC (ROS param).
        self.nlp_solver_iters = 1
        # B4' error-dynamics regression: affine velocity correction added to
        # f_expl rows [vx,vy,r]. Default OFF → p[76:79]=0 → baseline f_expl.
        self._err_regr = False
        self._e_corr = np.zeros(3)   # filled per-cycle by mpc_node (B4'.3)
        # ── Phase D closed-form CasADi GP residual (dynamics-only) ─────────
        # Independent of the l4acados `use_gp_residual` path. Default OFF.
        # Set by mpc_node from the `use_gp_casadi` ROS param BEFORE setup_MPC.
        self.use_gp_casadi = False
        self.gp_casadi_ckpt = '~/bo_results/gp_residual_realvy.pt'
        self.gp_casadi_train_data = '~/bo_results/gp_train_data_realvy.pt'
        # Vehicle dynamics — values match `stack_master/config/SIM/
        # SIM_pacejka.yaml` so the MPC's internal model is identical to
        # what the f110-simulator's std_kinematics::update_pacejka()
        # actually integrates. Identical params → MPC predictions match
        # simulator behaviour to numerical precision.
        # 2026-05-28 #17: align EXACTLY with gym sim_params.yaml (was 6 params
        # off — m+2%, Iz+23%★, lf+2%, lr-15%★, h_cg-81%★, mu-5%). This
        # mismatch caused MPCC to predict slower yaw than gym actually
        # delivered → corner over-rotate → outside slip into wall.
        self.dyn_m    = 3.47      # mass [kg]                  (sim: m)
        self.dyn_Iz   = 0.04712   # yaw inertia [kg·m²]        (sim: I)
        self.dyn_lf   = 0.15875   # CG to front axle [m]        (sim: lf)
        self.dyn_lr   = 0.17145   # CG to rear axle [m]         (sim: lr)
        self.dyn_h_cg = 0.074     # CG height [m] (load transfer) (sim: h)
        self.dyn_mu   = 1.0489    # friction coefficient        (sim: mu)
        self.ellipse_frac = 0.95  # friction ellipse headroom η (a_lim = μ·g·η)
        # Linear tire stiffness — sim-matched (f110-simulator params.yaml).
        # F_y = μ · C_S · F_z · α (linear in slip angle).
        # F1Tenth realistic; dynamic effects are inherently small at
        # this scale + speed → looks kinematic-like. Pure visual
        # similarity to kinematic is EXPECTED for low-speed F1Tenth.
        self.dyn_Csf = 4.718
        self.dyn_Csr = 5.4562
        # Tire model selector. Three options:
        #   'linear'  : F_y = μ·C_S·F_z·α                — fastest, no saturation
        #   'tanh'    : F_y = μ·D·F_z·tanh(B·α)          — smooth saturation
        #   'pacejka' : F_y = μ·D·F_z·sin(C·atan(B·α−…)) — most accurate, unstable in RTI
        self.dyn_tire_model = 'linear'
        # Legacy flag kept for backward compat (read-only proxy).
        self.dyn_use_pacejka = (self.dyn_tire_model == 'pacejka')
        # STANDARD F1Tenth Pacejka (literature: TUM/AMZ/Liniger):
        #   B (stiffness): 10  — typical
        #   C (shape):     1.4 — magic number for tire curves
        #   D (peak):      1.0 — peak friction coefficient
        #   E (curvature): 0   — symmetric
        # Numerically stable for SQP_RTI single-iter (C ≈ 1.4 keeps
        # sin·atan derivative well-conditioned across operating range).
        self.dyn_Bf, self.dyn_Cf, self.dyn_Df, self.dyn_Ef = (
            10.0, 1.4, 1.0, 0.0)
        self.dyn_Br, self.dyn_Cr, self.dyn_Dr, self.dyn_Er = (
            10.0, 1.4, 1.0, 0.0)
        # Hybrid blend tuned for HIGH-SPEED DYNAMIC mode:
        # vx ≥ 0.5 m/s → essentially 100% Pacejka dynamic.
        # Below 0.5 (startup / stop only) blend toward kinematic so the
        # atan2-slip singularity is well-conditioned. ICRA / racing usage:
        # car spends ≪1% of time below 0.5 m/s, so this is effectively
        # always dynamic.
        #   vx=0.0: w_std = 0.07  (93% kinematic — startup only)
        #   vx=0.5: w_std = 0.50  (50/50 — passing through this is brief)
        #   vx=1.0: w_std = 0.95  (95% dynamic)
        #   vx=2.0: w_std = 1.00  (~100% dynamic)
        self.dyn_v_b   = 0.5       # 2026-06-03 0.3→0.5 (ifac_mpcc 검증값, ICRA·타이트)
        self.dyn_v_s   = 0.3       # 2026-06-03 0.1→0.3 (블렌드 폭↑=tanh 부드럽게, stiff Hessian 방지)
        self.dyn_v_min = 0.2       # below: kinematic-dominated [m/s]
        self.dyn_a_max = 7.5       # max accel (matches sim max_accel)
        # Singularity epsilon for atan2 denominator. 1.0 m/s = robust for
        # SQP_RTI (proven stable on ICRA + F maps). Smaller (0.5) gives
        # better low-speed accuracy but causes IPM step collapse on
        # certain map start states. Stability over fidelity.
        # 2026-05-29: lowered 1.0 -> 0.5 for better low-speed yaw-rate
        # fidelity. cold_start_vx_floor=2.0 feeds solver vx>=2.0 at startup,
        # which should avoid the singular start regime that previously
        # caused IPM collapse. Verified empirically.
        self.dyn_v_eps = 1.0       # 2026-06-03 0.5→1.0 (ifac: 0.5는 IPM step collapse, 1.0 안정)

        # Bounds
        self.v_max = 6.0
        # R3 — decouple max_speed: v_max stays the HARD CAP (vx ubx, ubu, per-stage
        # caps). These two are set independently (None → derive from v_max so
        # behaviour is unchanged when unset) so raising the cap does NOT make the
        # progress target more aggressive or lengthen the κ-lookahead (the things
        # that caused the high-speed trajectory shake).
        self.speed_target = None   # q_p progress reward target (None → v_max)
        self.lookahead_m  = None   # ref_v κ-lookahead window [m] (None → max(6, v_max²/6))
        self.v_min = 0.5
        self.theta_max = 0.4
        self.theta_min = -0.4
        # C-STEER: hard steering-RATE limit [rad/s]. Gym servo slew is 3.2, but
        # the MPC is constrained tighter (≈paper Kim&Park 2025 uses 1.0) so it
        # plans SMOOTH, servo-feasible steering → kills the documented sign-flip
        # hunting at root. Enforced as an OCP h-constraint on BOTH stage 0
        # (published δ) and stages 1..N-1.
        self.sv_max = 2.0
        self.s_min, self.s_max = 0.0, 1e3
        self.p_min, self.p_max = 0.0, 6.0

        self.is_ot = False
        self.vheid = {}
        self.param = {}

        # Track splines (set via set_track_data)
        self.center_lut_x = None
        self.center_lut_y = None
        self.center_lut_dx = None
        self.center_lut_dy = None
        self.right_lut_x = None
        self.right_lut_y = None
        self.left_lut_x = None
        self.left_lut_y = None
        self.element_arc_lengths = None
        self.arc_lengths_orig_l = None
        self.path_length = None
        self.ref_v = None    # CasADi interpolant ref_v(s)

        # Live (rqt-tunable) weights — same names the IPOPT MPC uses.
        # R_safe=0.3 (was 0.8 → 0.4 → 0.3): with corridor h-constraint now
        # also active (R_CAR=0.10 from each wall, ≈0.35 m usable each side
        # of centerline), the 0.4 m keepout combined with a 0.4 m D_DETOUR
        # forced the car to a position 0.535 m off-center to clear an
        # on-centerline obstacle — outside the corridor. Solver couldn't
        # satisfy both → braked to 2.9 m/s every lap to soften slack.
        # 0.3 m matches car_R 0.135 + obs_marker 0.135 + 0.03 m margin and
        # keeps the required offset (0.435 m) inside the corridor.
        # ── 좁은 틈 통과 문턱 (2026-08-22) ────────────────────────────
        # 0723 에 만든 narrow-pass 예외가 문턱 0.70 으로 하드코딩돼 있었는데,
        # 이후 R_safe(0.40)+R_car(0.30) 이 정확히 0.70 이 되면서 주 판정과
        # 값이 같아져 **예외가 완전히 무효화**됐다 (좁은 틈 = 항상 side 0 =
        # BLOCK-STOP). 대회 규정은 "틈 50cm 이상이면 통과"이므로 물리 최소로
        # 되돌린다: 장애물중심→벽 r ≥ r_obs(0.15) + [차반폭0.15+여유0.05]
        # + [차반폭0.15+벽여유0.05] = 0.55.  50cm 틈이면 r=0.65 라 통과.
        self.narrow_room_live   = 0.55   # [m] 이 이상이면 서행 통과 허용
        self.narrow_pass_v_live = 1.5    # [m/s] 통과 중 속도 상한 (0=제한없음)
        self.narrow_pass_hold   = 1.5    # [s]  판정 후 속도상한 유지 시간
        self._narrow_pass_until = 0.0
        self.R_safe_live    = 0.3
        # 2026-08-06 최종리뷰 G1: OT 상태기계가 obstacle 제약(p_arr[7])만
        # 축소하기 위한 전용 채널. None=비활성(R_safe_live 그대로 사용).
        # R_safe_live 자체는 side-decide 문턱(w_car_safe)/_full_keep 이
        # 함께 읽으므로 OT 코드가 더 이상 건드리지 않는다 — rqt 라이브
        # 튜닝 보존 + 0.10 축소가 side-decide 를 붕괴시키던 결함 제거.
        self.R_safe_obs_live = None
        self.a_lat_safe_live   = 6.0    # rqt: curvature-speed cap headroom
        self.D_apex_live       = 0.22   # rqt: apex-bias lateral target offset.
                                        # Pulls e_c toward inside of corner.
                                        # 0 = no apex bias (pure centerline).
                                        # 0.4 = aggressive racing line.
        self.R_car_live        = 0.0    # rqt: corridor + obs h-constraint margin
                                        # (0.0 = wall touchable for racing line)
        self.commit_dist_live  = 10.0   # rqt: obstacle commit trigger distance
        self.cost_spike_thr_live = 500.0  # rqt: fallback threshold
        self.alpha_steer_live  = 0.6    # rqt: steer EMA blend (Python only)
        # 0724 온라인 학습 스케일 조회 함수 (mpc_node 가 set; None=학습 off)
        self._refv_scale_fn = None
        self._alat_scale_fn = None
        # Baked codegen-time cost weights (must match q_*_def in
        # setup_MPC). Used to convert rqt-set absolute weights into the
        # scale multiplier pushed via p_sym slots. ALL major cost weights
        # are scale-wired so BO / rqt can sweep without acados rebuild.
        self.q_cte_scale_live   = 1.0
        self.q_lag_scale_live   = 1.0
        self.q_psi_scale_live   = 1.0
        self.q_v_scale_live     = 1.0
        self.q_dd_scale_live    = 1.0
        self.q_p_scale_live     = 1.0
        self.q_drate_scale_live = 1.0
        self.q_dv_scale_live    = 1.0   # 2026-05-27 #8 — a_x penalty
        # Static cost weights (set via param dict)
        self.q_v       = 15.0
        self.q_dv      = 15.0
        self.q_mu      = 0.05
        self.q_vp_proj = 60.0   # progress reward gamma
        self.progress_gamma = 0.0   # B1 선형 진행보상 -gamma*p_v (0=무효). mpc_node가 codegen-time set

        # Solver storage
        self.solver = None
        self.X0 = None
        self.u0 = None
        self.WARM_START = False

        # Lap tracking — keep solver-internal s unbounded so the dynamics
        # constraint ṡ = p_v never sees a wrap discontinuity. Spline lookup
        # in cost uses s_periodic = s − L·floor(s/L) for [0, L) wrap.
        self.lap_count = 0
        self._last_sensor_s = None
        # Same persistent-offset trick for yaw (mirrors lap_count for s).
        # Without this, sensor yaw ∈ [−π, π] wraps every full revolution
        # and the cycle-by-cycle "shift initial_state[2] ± 2π" detection
        # fires repeatedly in the wrap zone, each time triggering multi-
        # iter SQP and disrupting the solve → cost spike (observed: 974
        # at the exact wrap moment + ~5 cycles of elevated cost).
        # Tracking yaw_offset as multiples of 2π keeps the solver-internal
        # ψ monotonically continuous, so no wrap event ever reaches the
        # solver.
        self._yaw_offset = 0.0
        self._last_sensor_yaw = None

        # Debug — same names IPOPT version uses, ROS node reads these
        self.dbg_n_obs_input = 0
        self.dbg_sel_dmin = float('inf')
        self.dbg_sel_x = float('inf')
        self.dbg_sel_y = float('inf')
        self.dbg_side_pref = 0.0
        self.dbg_solver_status = ""
        self.boundary_hook = None  # callable(shifted_points: list[(x,y)]) or None

    # ------------------------------------------------------------------
    # Setters (mirror IPOPT MPC interface so node code reuses)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_time_steps(mode: str, dT: float, N: int) -> list[float]:
        """multi-dt time_steps grid 생성.
        - uniform   : [dT] * N
        - pyramidal : 가까이 sharp control + 중간 medium + 멀리 long planning.
                      N=80, dT=0.04 가정 시 [0.04]*20 + [0.06]*30 + [0.10]*30 = 5.6s.
                      N 다른 경우 비율 조정.
        """
        mode = str(mode).lower()
        if mode == 'pyramidal':
            n1 = max(1, N // 4)         # 가까이 (25%) dt = dT
            n3 = max(1, (N * 3) // 8)   # 멀리   (37.5%) dt = dT*2.5
            n2 = N - n1 - n3            # 중간 (37.5%) dt = dT*1.5
            return [dT] * n1 + [dT * 1.5] * n2 + [dT * 2.5] * n3
        # default uniform
        return [dT] * N

    def set_initial_params(self, param, vheid, is_ot):
        self.vheid = vheid
        self.param = param
        self.dT = param['dT']
        self.N = param['N']
        # multi-dt time_steps grid (pyramidal: 가까이 sharp + 멀리 long planning).
        # uniform 이면 [dT]*N. pyramidal 이면 [dT]*20 + [dT*1.5]*30 + [dT*2.5]*30 (N=80 가정).
        self.time_steps_mode = param.get('time_steps_mode', 'uniform')
        self.time_steps = self._build_time_steps(self.time_steps_mode, self.dT, self.N)
        self.L = vheid.get('l_wb', param.get('L', 0.307))
        self.theta_max = param['theta_max']
        self.theta_min = -self.theta_max
        self.v_max = param['v_max']
        self.v_min = 0.5
        self.x_min, self.x_max = param['x_min'], param['x_max']
        self.y_min, self.y_max = param['y_min'], param['y_max']
        self.psi_min, self.psi_max = param['psi_min'], param['psi_max']
        self.s_min, self.s_max = param['s_min'], param['s_max']
        self.p_min, self.p_max = param['p_min'], param['p_max']
        self.is_ot = is_ot
        # Param overrides for static weights (LIVE ones come from defaults below)
        self.q_v       = param.get('mpc_v_track',     self.q_v)
        self.q_dv      = param.get('mpc_w_accel',     self.q_dv)
        self.q_mu      = param.get('q_mu',            self.q_mu)
        self.q_vp_proj = param.get('mpc_vp_project',  self.q_vp_proj)

    def a_lat_safe_eff(self):
        """a_lat_safe_live clamped to physical grip μ·g·η.

        BO/yaml/rqt 가 어떤 값을 요청하든 속도 프로필·vcap·a_lat 제약이
        타이어가 못 내는 그립 위에 세워지지 않게 하는 단일 게이트.
        Live-safe: 호출 시점의 a_lat_safe_live/dyn_mu 로 매번 계산.
        """
        eff, clamped = clamp_a_lat_to_grip(
            self.a_lat_safe_live, self.dyn_mu, self.ellipse_frac)
        if clamped and not getattr(self, '_alat_clamp_warned', False):
            self._alat_clamp_warned = True
            self._log.warning(
                f"[MPC-acados] a_lat_safe_live {float(self.a_lat_safe_live):.2f} > "
                f"μgη {eff:.2f} (mu={float(self.dyn_mu):.3f}) → clamped. "
                "BO 탐색 상한이 물리 그립에 묶임 (의도된 동작).")
        return eff

    def set_track_data(self, c_x, c_y, c_dx, c_dy, r_x, r_y, l_x, l_y,
                       element_arc_lengths, original_arc_length_total, ref_v):
        self.center_lut_x, self.center_lut_y = c_x, c_y
        self.center_lut_dx, self.center_lut_dy = c_dx, c_dy
        self.right_lut_x, self.right_lut_y = r_x, r_y
        self.left_lut_x, self.left_lut_y = l_x, l_y
        self.element_arc_lengths = element_arc_lengths
        self.arc_lengths_orig_l = original_arc_length_total
        self.path_length = original_arc_length_total
        self.ref_v = ref_v

        # Precompute curvature κ(s) on a fine grid for runtime v-cap lookup.
        # κ = dθ/ds where θ = atan2(dy/ds, dx/ds). Sharp corners (hairpin)
        # have |κ| > 2 rad/m on f1tenth tracks; pure-Cartesian MPCC's
        # contour/lag cost geometry breaks there, so we cap v per stage by
        # v ≤ sqrt(a_lat_safe / |κ|) to keep the QP well-posed.
        self.kappa_ds = 0.05
        L = float(original_arc_length_total)
        s_grid = np.arange(0.0, L, self.kappa_ds)
        dx_g = np.array([float(c_dx(float(s))) for s in s_grid])
        dy_g = np.array([float(c_dy(float(s))) for s in s_grid])
        theta_g = np.unwrap(np.arctan2(dy_g, dx_g))
        self.kappa_grid = np.gradient(theta_g, self.kappa_ds)
        self._log.info("[MPC-acados] kappa grid: |k|_max=%.2f rad/m, |k|_p95=%.2f",
                      float(np.max(np.abs(self.kappa_grid))),
                      float(np.percentile(np.abs(self.kappa_grid), 95)))

        # Make |κ| available to CasADi as a linear interpolant — enables
        # per-stage curvature-aware speed cap inside the cost. Track is
        # closed-loop so append the s=0 value at s=L for clean wrap-around.
        # FORWARD-LOOKING: instead of |κ(s)|, use max(|κ(s')| for s' ∈ [s,
        # s+LOOKAHEAD]). At s=60 (just before the upper-left U-turn) the
        # raw |κ| is only 0.04 (v_kin=11 m/s, cap inactive), but at s=63
        # the peak |κ|=0.71 (v_kin=2.9 m/s). MPC's horizon (~2.5 m at
        # v=5) barely reaches the peak so per-stage cap fires too late
        # to brake. Forward-max over LOOKAHEAD = 4 m makes the cap kick
        # in 4 m before any sharp corner, giving MPC enough time to slow.
        abs_k_arr = np.abs(self.kappa_grid)
        # Gaussian-ish smoothing of |κ| profile before forward-max. Without
        # it, |κ| has high-freq variation from spline derivative noise →
        # ref_v_cap jitters cycle-to-cycle as car advances, manifesting as
        # speed surge after corner exit. 11-tap rolling avg ≈ 0.5 m smooth.
        smooth_win = 11
        kernel = np.ones(smooth_win) / float(smooth_win)
        abs_k_arr = np.convolve(abs_k_arr, kernel, mode='same')
        # 2026-06-02 속도비례: 고속일수록 코너 전 제동거리↑ 필요. v=8 면
        # 8→2.3 감속에 ~10m. per-stage hard cap 이 u[0] 을 √(a_lat/κ_fwd) 로
        # 묶으므로, forward-max 가 코너를 이만큼 일찍 봐야 cap 이 제때 내려감.
        LOOKAHEAD_M = float(self.lookahead_m) if self.lookahead_m else max(6.0, float(self.v_max) ** 2 / 6.0)   # R3: fixed window decouples shake from cap
        n_look = int(LOOKAHEAD_M / self.kappa_ds)
        n_grid = len(abs_k_arr)
        # R4 (refv_smoothing.distance_tapered_forward_max) was tried here to remove
        # the ref_v step when a corner crosses the window edge. On this ultra-tight
        # map the forward window is SATURATED (a corner is always within ~6 m →
        # p95≈max), so there is no edge step to smooth and R4 is a no-op here; it
        # only helps open maps with sparse corners + long straights. baseline uses
        # the plain hard max. (2026-08-18: 그 모듈 mpc_core/refv_smoothing.py 는
        # import 0회라 삭제. 개방형 맵에서 필요해지면 커밋 e7e2e52 에서 복구:
        #   git show e7e2e52:nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/refv_smoothing.py)
        # 2026-06-08 (사용자 통찰): plain forward-MAX 가 완만(저-κ) 구간서도 멀리
        # 있는 tight 코너 κ로 ref_v 를 눌러 가속을 막음(twisty 맵서 p95≈max 포화).
        # → brake-distance-aware 등가 κ 로 교체: 거리 d 만큼 떨어진 코너는 제동거리
        # 가 있으니 지금은 빨라도 됨 → 등가-κ 를 거리로 할인.
        #   v_allow(s)² = v_corner(s')² + 2·a_brake·d  (a_brake=a_lat 가정)
        #   = a_lat/κ(s') + 2·a_lat·d = a_lat·(1/κ' + 2d)
        #   κ_eq = a_lat/v_allow² = 1/(1/κ' + 2d) = κ'/(1+2d·κ')   ← a_lat 소거
        # κ_eq[d=0]=κ'(local, 코너서 full), d↑ → κ_eq→0(제한 풀림). window 안
        # 모든 upcoming 코너 중 가장 binding(max κ_eq) 채택 → 완만 구간 가속 허용,
        # tight 코너는 제때 감속.
        _ds = float(self.kappa_ds)
        # bf = a_brake/a_lat (κ_eq 유도식 그대로). 고정 0.7은 a_lat_safe=5 에서
        # 제동 3.5 m/s² 가정 = 솔버 한계(3.0) 초과 낙관 — 솔버 값으로 일관화.
        _bf = min(1.0, abs(A_MIN_DYN) / max(1e-3, self.a_lat_safe_eff()))
        abs_k_fwd = np.empty(n_grid, dtype=float)
        for i in range(n_grid):
            j = min(i + n_look, n_grid)
            seg = abs_k_arr[i:j]
            dist = np.arange(len(seg)) * _ds
            k_eq = seg / (1.0 + 2.0 * _bf * dist * seg)   # brake-aware equivalent κ (margin)
            abs_k_fwd[i] = float(k_eq.max()) if len(seg) else float(abs_k_arr[i])
        self._log.info("[MPC-acados] brake-aware |κ_eq| (lookahead=%.1f m): max=%.3f p95=%.3f "
                      "(was plain forward-max — 사용자 통찰: 완만구간 가속)",
                      LOOKAHEAD_M,
                      float(abs_k_fwd.max()),
                      float(np.percentile(abs_k_fwd, 95)))
        if self.use_dynamic:
            self._log.info("[MPC-acados] MODEL = DYNAMIC (%s tire) (n_states=8, "
                          "n_controls=3, LM=1.0, RTI 1-iter)",
                          self.dyn_tire_model)
        else:
            self._log.info("[MPC-acados] MODEL = KINEMATIC-UNIFIED 8-state (f_kin, "
                          "n_states=8, n_controls=3 [a_x, δ, p_v], LM=0.2)")
        s_grid_ext = list(s_grid) + [L]
        abs_k_ext  = abs_k_fwd.tolist() + [abs_k_fwd[0]]
        self.abs_kappa_lut = ca.interpolant(
            'abs_kappa_lut', 'linear', [s_grid_ext], abs_k_ext
        )

        # Signed κ LUT for apex-biased lateral reference. Used by the cost
        # to shift the e_c target toward the INSIDE of the upcoming corner,
        # so centerline tracking produces racing-line behaviour. Sign:
        # κ_signed > 0 = left turn (CCW) → inside is LEFT.
        #
        # Profile design — emulate the global IQP raceline's behaviour
        # where straights aren't strictly tracked: the car drifts
        # diagonally toward the inside of the next corner during the
        # approach straight, hits apex, and decays back to centerline
        # gradually on the exit. Without this, centerline mode produces
        # straight-on-straight then sharp corner attack — slow.
        #
        # Construction:
        #  1. Gaussian kernel σ=0.8 m (3σ ≈ 2.4 m support) — covers a
        #     racing-line-like wide influence zone around each apex.
        #  2. Forward-shift output by 0.6 m (np.roll on closed loop).
        #     At s on the approach straight, the LUT returns the
        #     smoothed κ at s+0.6 m → bias kicks in BEFORE apex.
        # Net bias profile (peak-normalised) around an apex:
        #   apex-2.5 m: 0.10  (drift starts on approach straight)
        #   apex-1.5 m: 0.40  (clear diagonal on straight)
        #   apex-0.6 m: 1.00  (peak — slight before geometric apex)
        #   apex      : 0.75
        #   apex+0.5 m: 0.30  (sustain on exit)
        #   apex+1.5 m: 0.05  (almost back to centerline)
        sigma_m_apex   = 0.8
        forward_bias_m = 0.6
        half_idx = int(3 * sigma_m_apex / self.kappa_ds)
        kx = np.arange(-half_idx, half_idx + 1) * self.kappa_ds
        ker_apex = np.exp(-kx * kx / (2.0 * sigma_m_apex * sigma_m_apex))
        ker_apex /= ker_apex.sum()
        signed_k_smooth = np.convolve(self.kappa_grid, ker_apex, mode='same')
        forward_bias_idx = int(forward_bias_m / self.kappa_ds)
        signed_k_arr = np.roll(signed_k_smooth, -forward_bias_idx)
        signed_k_ext = signed_k_arr.tolist() + [signed_k_arr[0]]
        self.signed_kappa_lut = ca.interpolant(
            'signed_kappa_lut', 'linear', [s_grid_ext], signed_k_ext
        )

    # ------------------------------------------------------------------
    # Dynamic-bicycle model builder (Phase 1: helper, NOT YET WIRED)
    # ------------------------------------------------------------------
    def _build_dynamic_model(self):
        """Build the Pacejka single-track dynamic model expressions.

        Mirrors `f110-simulator/src/std_kinematics.cpp::update_pacejka`
        verbatim so MPC predictions match what the simulator integrates
        (assuming MPC and sim are configured with the same SIM_pacejka
        parameters — which `__init__` does by default).

        Returns a dict (intentionally NOT installed into `setup_MPC` yet).
        Phase 2 wiring will:
          - replace the kinematic `x` / `u` / `f_expl` with these
          - swap the cost residual `v − ref_v` → `vx − ref_v`
          - swap the h-constraint `a_lat = v² tan(δ)/L` → `a_lat = vx · r`
          - change `solve_step` initial state to include vx, vy, r
          - change `solve_step` output to use vx[1] as v_cmd
        """
        # ── States (8): x, y, ψ, vx, vy, r, s, δ_prev ─────────────────
        x_pos      = ca.SX.sym('x_pos')
        y_pos      = ca.SX.sym('y_pos')
        psi        = ca.SX.sym('psi')
        vx         = ca.SX.sym('vx')
        vy         = ca.SX.sym('vy')
        r_yaw      = ca.SX.sym('r_yaw')
        s          = ca.SX.sym('s')
        delta_prev = ca.SX.sym('delta_prev')
        x = ca.vertcat(x_pos, y_pos, psi, vx, vy, r_yaw, s, delta_prev)

        # ── Inputs (3): a_x, δ, p_v ───────────────────────────────────
        # Replaces kinematic's `v` with `a_x` (longitudinal accel).
        # F1Tenth's ackermann_drive accepts `drive.speed` (m/s); we'll
        # send `vx[1]` (next-stage predicted vx) as v_cmd in Phase 2.
        a_x   = ca.SX.sym('a_x')
        delta = ca.SX.sym('delta')
        p_v   = ca.SX.sym('p_v')
        u = ca.vertcat(a_x, delta, p_v)

        xdot = ca.SX.sym('xdot', 8)

        # ── Pacejka dynamic equations ─────────────────────────────────
        # Slip angles. Singularity at v_x → 0 in atan2 denominator: use
        # ca.fmax(vx, ε) so derivatives stay finite. Kinematic blend
        # below dyn_v_b further regularises (low-speed mode).
        vx_safe = ca.fmax(vx, self.dyn_v_eps)
        # sim convention: alpha_f = atan2(-vy - lf·r, vx) + delta
        alpha_f = ca.atan2(-vy - self.dyn_lf * r_yaw, vx_safe) + delta
        alpha_r = ca.atan2(-vy + self.dyn_lr * r_yaw, vx_safe)

        # Vertical loads with longitudinal load transfer
        # (sim: F_zf = m · (-a_x · h_cg + g · l_r) / L)
        g_const = 9.81
        L_wb = self.dyn_lf + self.dyn_lr
        F_zf = self.dyn_m * (-a_x * self.dyn_h_cg + g_const * self.dyn_lr) / L_wb
        F_zr = self.dyn_m * ( a_x * self.dyn_h_cg + g_const * self.dyn_lf) / L_wb

        # Tire force model selector — see __init__ self.dyn_tire_model.
        if self.dyn_tire_model == 'pacejka':
            # Full Pacejka Magic Formula. NOTE: SQP_RTI single iter
            # struggles with this — sin·atan derivative variation
            # between cycles → MINSTEP cascades. Use only with multi-
            # iter solve enabled (self.dyn_multi_iter).
            pacejka_arg_f = (self.dyn_Bf * alpha_f
                             - self.dyn_Ef * (self.dyn_Bf * alpha_f
                                              - ca.atan(self.dyn_Bf * alpha_f)))
            pacejka_arg_r = (self.dyn_Br * alpha_r
                             - self.dyn_Er * (self.dyn_Br * alpha_r
                                              - ca.atan(self.dyn_Br * alpha_r)))
            F_yf = (self.dyn_mu * self.dyn_Df * F_zf
                    * ca.sin(self.dyn_Cf * ca.atan(pacejka_arg_f)))
            F_yr = (self.dyn_mu * self.dyn_Dr * F_zr
                    * ca.sin(self.dyn_Cr * ca.atan(pacejka_arg_r)))
        elif self.dyn_tire_model == 'tanh':
            # tanh tire = SIMPLIFIED Pacejka. saturation built-in but
            # derivative is well-behaved everywhere (sech², bounded).
            # F_y = μ·D·F_z·tanh(B·α). Reaches saturation D·F_z at α≫0,
            # linear for small α with slope D·F_z·μ·B.
            # Used in: Hewing/Liniger ETH, Carrau et al, several
            # racing MPC papers. Sweet spot of accuracy + numerics.
            F_yf = self.dyn_mu * self.dyn_Df * F_zf * ca.tanh(self.dyn_Bf * alpha_f)
            F_yr = self.dyn_mu * self.dyn_Dr * F_zr * ca.tanh(self.dyn_Br * alpha_r)
        else:  # 'linear'
            # F_y = μ · C_S · F_z · α. No saturation.
            F_yf = self.dyn_mu * self.dyn_Csf * F_zf * alpha_f
            F_yr = self.dyn_mu * self.dyn_Csr * F_zr * alpha_r

        # Dynamic state derivatives (sim std_kinematics.cpp:82-88).
        # F_xf = 0 (RWD); F_xr is folded into a_x (sim's `accel` arg).
        f_dyn = ca.vertcat(
            vx * ca.cos(psi) - vy * ca.sin(psi),                         # ẋ
            vx * ca.sin(psi) + vy * ca.cos(psi),                         # ẏ
            r_yaw,                                                        # ψ̇
            a_x + (1.0 / self.dyn_m) * (-F_yf * ca.sin(delta)) + vy * r_yaw,
            (1.0 / self.dyn_m) * (F_yr + F_yf * ca.cos(delta)) - vx * r_yaw,
            (1.0 / self.dyn_Iz) * (F_yf * self.dyn_lf * ca.cos(delta)
                                    - F_yr * self.dyn_lr),
            p_v,
            (delta - delta_prev) / self.dT,
        )

        # ── Kinematic equations (for low-vx blend) ────────────────────
        # When vx ≪ v_b, the dynamic atan2 and small lateral forces
        # produce noisy derivatives. Blend toward kinematic single-track
        # so prediction stays well-conditioned at start-up / near-stop.
        # Kinematic regime forces vy and r toward their kinematic-
        # consistent values via fast first-order decay (τ_kin small).
        beta_kin   = ca.atan(self.dyn_lr * ca.tan(delta) / L_wb)
        vy_target  = vx * ca.tan(beta_kin)
        r_target   = (vx / L_wb) * ca.tan(delta) * ca.cos(beta_kin)
        tau_kin    = 0.05  # rapid relaxation toward kinematic (50 ms)
        f_kin = ca.vertcat(
            vx * ca.cos(psi + beta_kin),                                  # ẋ
            vx * ca.sin(psi + beta_kin),                                  # ẏ
            r_target,                                                      # ψ̇ (set by kin geom)
            a_x,                                                           # v̇x
            (vy_target - vy) / tau_kin,                                    # v̇y → kin
            (r_target  - r_yaw) / tau_kin,                                 # ṙ   → kin
            p_v,
            (delta - delta_prev) / self.dT,
        )

        # Smooth blend: w_std rises from 0 to 1 across vx ∈ [v_b−v_s, v_b+v_s]
        # (sim: w_std = 0.5·(1 + tanh((vx − v_b)/v_s)))
        # 2026-06-10 unified layout: KINEMATIC mode = this same 8-state model
        # with f_expl = f_kin at ALL speeds (w_std ≡ 0). Only the vehicle
        # model changes between modes — state/input layout, costs, bounds,
        # LMPC, GP/B4' residual hooks and BO search space stay IDENTICAL.
        if self.use_dynamic:
            w_std  = 0.5 * (1.0 + ca.tanh((vx - self.dyn_v_b) / self.dyn_v_s))
            f_expl = w_std * f_dyn + (1.0 - w_std) * f_kin
        else:
            w_std  = ca.SX(0.0)
            f_expl = f_kin   # pure kinematic single-track, 8-state

        # Real lateral acceleration (replaces v² tan(δ)/L kinematic form).
        a_lat_expr = vx * r_yaw

        return dict(
            x=x, u=u, xdot=xdot, f_expl=f_expl,
            x_pos=x_pos, y_pos=y_pos, psi=psi,
            vx=vx, vy=vy, r=r_yaw, s=s, delta_prev=delta_prev,
            a_x=a_x, delta=delta, p_v=p_v,
            F_yf=F_yf, F_yr=F_yr, F_zf=F_zf, F_zr=F_zr,
            alpha_f=alpha_f, alpha_r=alpha_r,
            f_dyn=f_dyn, f_kin=f_kin, w_std=w_std,
            a_lat_expr=a_lat_expr,
        )

    # ------------------------------------------------------------------
    # OCP setup
    # ------------------------------------------------------------------
    def setup_MPC(self):
        # 2026-07-29: 코드에 박혀 있던 튜닝 상수를 config/mpc_tuning.yaml 로 뺐다.
        # cost/obstacle 값은 심볼릭 모델·W 행렬에 구워지므로 바꾸면 acados 가
        # 재생성된다(codegen 해시가 달라짐). slack 은 재시작만으로 반영된다.
        _tune = load_tuning(self._log)
        _cw, _ob = _tune['cost'], _tune['obs']
        # 곡률 비례 R_car 용 (solve 루프에서 사용 — 런타임 파라미터)
        self._rcar_straight = _tune['corridor']['r_car_straight']
        self._rcar_kref = max(1e-6, _tune['corridor']['kappa_ref'])
        # 커밋 해제 페이드 끝점 [m] — 1.5 보다 커야 한다(1.5 가 페이드 시작).
        self.RELEASE_FADE_END = max(1.6, _tune['avoid']['release_fade_end'])
        # F3 커밋 추종 (실차 검출 지터 흡수 — 해제 검사부 주석 참조)
        self.COMMIT_ASSOC_R = _tune['avoid']['commit_assoc_radius']
        self.COMMIT_FOLLOW_ALPHA = _tune['avoid']['commit_follow_alpha']
        self.COMMIT_MISS_S = _tune['avoid']['commit_miss_release_s']

        ocp = AcadosOcp()
        model_ac = AcadosModel()
        model_ac.name = 'mpcc_evompcc_acados'

        # ---- States: x, y, psi, s, delta_prev ----
        # delta_prev is augmented to enable a per-stage steer-rate cost
        # (residual = delta - delta_prev). NONLINEAR_LS y(x,u) can only
        # see one stage at a time, so we shadow the previous applied δ
        # as a state. Dynamics: δ_prev_dot = (δ − δ_prev)/dt → after one
        # Euler step, δ_prev_new = δ_old (the control just applied).
        # Without rate cost the solver picks bimodal δ patterns
        # ([+0.22, +0.08, +0.22, +0.08, ...] alternating) that satisfy
        # the same average heading change with similar |δ|² total but
        # produce a visibly wavy prediction line.
        # ── Unified 8-state layout (2026-06-10) ─────────────────────────
        # BOTH modes: 8 states [x, y, ψ, vx, vy, r, s, δ_prev], 3 inputs
        # [a_x, δ, p_v]. Only the vehicle model (f_expl) differs:
        #   dynamic   : Pacejka/tanh tire forces, kinematic blend at low vx
        #   kinematic : pure kinematic single-track (f_kin, no tire forces)
        # Everything downstream — costs (incl. vterm + q_dv), bounds, LMPC
        # safe-set (slot 3 = vx), joint-α, GP/B4' residual rows [3,4,5],
        # warm-start, BO 9D search space — is shared with NO model branch.
        dyn = self._build_dynamic_model()
        u          = dyn['u']
        x        = dyn['x']
        f_expl   = dyn['f_expl']
        xdot     = dyn['xdot']
        nx       = 8
        x_         = dyn['x_pos']
        y_         = dyn['y_pos']
        psi        = dyn['psi']
        s          = dyn['s']
        delta_prev = dyn['delta_prev']
        delta      = dyn['delta']
        p_v        = dyn['p_v']
        vx_for_cost  = dyn['vx']
        a_lat_expr   = dyn['a_lat_expr']
        a_x_input    = dyn['a_x']
        nu = 3
        # ── Phase D (closed-form CasADi GP residual) ──────────────────
        # Adds the offline-trained sparse-GP posterior MEAN (as a pure
        # CasADi expression) to the dynamics ONLY. Independent of the
        # l4acados `use_gp_residual` path (which rebuilds the OCP). The
        # cost / W / p_sym are left UNTOUCHED — the GP is mapped through
        # B_d to the velocity-state rows [vx, vy, r] = indices 3,4,5 of
        # the 8-state vector [x, y, psi, vx, vy, r, s, delta_prev].
        # Works for BOTH models (unified layout) — train the GP per-model
        # (kinematic GP learns kinematic-model residuals).
        # Default OFF (self.use_gp_casadi); guarded so any failure falls
        # back to the plain f_expl.
        if getattr(self, 'use_gp_casadi', False) and getattr(self, '_err_regr', False):
            self._log.warn(
                "[MPC-acados] use_gp_casadi AND _err_regr both set — skipping "
                "GP residual (e_corr error-regression takes precedence; both add "
                "to f_expl velocity rows = double correction).")
        if getattr(self, 'use_gp_casadi', False) and not getattr(self, '_err_regr', False):
            try:
                from .gp_casadi_residual import (
                    load_gp_casadi_params, build_casadi_residual,
                )
                gp_ckpt = os.path.expanduser(
                    getattr(self, 'gp_casadi_ckpt',
                            '~/bo_results/gp_residual_realvy.pt'))
                gp_train = os.path.expanduser(
                    getattr(self, 'gp_casadi_train_data',
                            '~/bo_results/gp_train_data_realvy.pt'))
                gp_p = load_gp_casadi_params(gp_ckpt, gp_train)
                # SAME symbols already in scope (unified-model build).
                gp_z = ca.vertcat(dyn['vx'], dyn['vy'], dyn['r'],
                                  dyn['delta'], dyn['a_x'])
                mu = build_casadi_residual(gp_z, gp_p)
                # 0812 GP 저속 게이트 — GP 는 vx>1.5 데이터로만 학습돼 정지·저속에서
                # 평균회귀(요가속 편향 주입 → 출발 조향 스윙, 0811 아젠다 ⑤①).
                # 게이트: vx 1.0 중심 시그모이드 — 정지 0, 1.5 에서 ~0.95, 학습영역 1.0.
                # 후진(vx<0)도 자연히 0 (GP 학습영역 밖).
                # (0813: nuc14 트리에서 역이식 — 두 트리 통합)
                _gp_gate = 0.5 * (1 + ca.tanh((dyn["vx"] - 1.0) / 0.3))
                mu = _gp_gate * mu
                # B_d: 3 GP outputs -> state rows vx(3), vy(4), r(5).
                # Pad residual to f_expl width: 8 (plain) or 18 (joint-α
                # LMPC appends 10 α-states AFTER the 8 physical, so rows
                # 3/4/5 = vx/vy/r in both; α rows get 0 residual).
                _nx_full = f_expl.shape[0]
                f_expl = f_expl + ca.vertcat(
                    0, 0, 0, mu[0], mu[1], mu[2], *([0] * (_nx_full - 6)))
                self._log.info(
                    "[MPC-acados] Phase D CasADi-GP residual ACTIVE "
                    "(M=%d inducing, ckpt=%s) — added to f_expl rows "
                    "[vx,vy,r]", gp_p.M, gp_ckpt)
            except Exception as _gp_e:
                self._log.warn(
                    "[MPC-acados] Phase D CasADi-GP residual FAILED "
                    "(%s) — falling back to plain f_expl", _gp_e)
        # B3: split physical state dim (ROS interface) from solver state dim.
        # joint-α: solver carries nx=8+K but the ROS-facing physical state is 8.
        self._nx_solver = nx
        self.n_states   = 8
        self.n_controls = nu

        # ---- Per-cycle parameters (constant across all stages) ----
        # idx: 0=obs_dmin, 1=obs_x, 2=obs_y, 3=side_pref,
        #      4=q_cte, 5=q_lag, 6=q_d_delta, 7=R_safe, 8=M_slack,
        #      9=e_c_obs (obstacle's lateral offset from centerline, Frenet)
        # 10 reserved (unused).
        # The half-plane uses e_c_obs (Frenet) instead of a tangent-anchored
        # Cartesian normal. The previous tangent-anchored form (obs_tan_sin/
        # cos) was stable per cycle but ignored track curvature — over a
        # curving section the prediction had to choose between following
        # the curve (e_c roughly tracks centerline) and respecting a
        # straight-line half-plane that rotated relative to the curve, so
        # prediction shot off-track. Frenet-frame e_c naturally follows
        # the curve, and a Cartesian-distance proximity gate disables the
        # constraint far from the obstacle.
        # Slot map (per-cycle constants):
        #   0  obs_dmin (debug)        9  e_c_obs (Frenet obstacle offset)
        #   1  obs_x                   10 a_lat_safe   (rqt)
        #   2  obs_y                   11 D_apex       (rqt)
        #   3  side_pref               12 q_psi_scale  (rqt)
        #   4  D_DETOUR    (rqt)       13 q_v_scale    (rqt)
        #   5  R_CAR       (rqt)       14 q_dd_scale   (rqt)
        #   6  q_cte_scale (rqt)       15 q_p_scale    (rqt)
        #   7  R_safe      (rqt)       16 q_drate_scale(rqt)
        #   8  q_lag_scale (rqt)       17 q_dv_scale   (rqt) — 2026-05-27 #8
        # 2026-05-28 #18 LMPC slots (per-cycle constants — terminal cost only):
        #   18..57  : K=10 × (x*, y*, ψ*, vx*) — SS nearest 점들 (4 floats each)
        #   58..67  : K=10 × Q* — cost-to-go (step-count to lap end)
        #   68      : lmpc_w     (default 0 → LMPC OFF; backward-compat 보존)
        #   69      : lmpc_alpha (distance penalty in softmin)
        #   70      : lmpc_beta  (softmin sharpness; reviewer 권장 0.05)
        #   71      : lmpc_reg_w (regularization 가중치 — best SS 점 attractor)
        # Per-stage (4):
        #   72 left_x  73 left_y  74 right_x  75 right_y
        # B4' error regression (3, const across horizon):
        #   76 e_corr_vx  77 e_corr_vy  78 e_corr_r
        # P2 Stage-B (0722o) 2번째 장애물 슬롯 (4):
        #   79 obs2_x  80 obs2_y  81 side2  82 e_c_obs2
        #   부재 시 sentinel(x=1e6, side2=0) → gate2=0 → h_obs2 항등만족 = 기존과 동일
        # 0723b 3번째 장애물 슬롯 (4): 83 obs3_x  84 obs3_y  85 side3  86 e_c_obs3
        #   (obs2와 동일 sentinel 규약. Stage-B가 전방 최근접 2개를 obs2/obs3에 채움 —
        #    3개 동시 시야에서 3번째가 무제약이던 한계 제거.)
        # 0723u A2-확장 (1): 87 obs1_mode — 0=반평면(정적, 기존), 1=유클리드 원형
        #   (동적 상대: per-stage 예측 위치 keepout → 추종/추월이 최적화에서 자연 발생)
        n_p_const = 18 + 54 + 3 + 4 + 4 + 1 + 2   # ... + obs1_mode + refv/alat scale
        n_p_stage = 4             # 4 corridor (per-stage)
        n_p_total = n_p_const + n_p_stage   # 90
        K_LMPC = 10
        p_sym = ca.SX.sym('p_sym', n_p_total)
        e_corr_sym = p_sym[76:79]   # B4' velocity-row correction (vx,vy,r)
        # P2 Stage-B: 2번째 장애물 (부재=sentinel)
        obs2_x = p_sym[79]; obs2_y = p_sym[80]
        side2_pref = p_sym[81]; e_c_obs2_p = p_sym[82]
        # 0723b obs3 슬롯
        obs3_x = p_sym[83]; obs3_y = p_sym[84]
        side3_pref = p_sym[85]; e_c_obs3_p = p_sym[86]
        obs1_mode = p_sym[87]   # 0723u: 0=반평면 / 1=원형(동적)
        # 0724 온라인 학습 속도 프로파일 (docs/superpowers/specs/2026-07-24-refv-*)
        # per-stage 스케일: 기본 1.0 → 기존 식과 동일. mpc_node 가 학습기
        # (SpeedProfileLearner) 조회값을 stage 별로 주입.
        refv_scale_p = p_sym[88]   # ref_v_raw 배율
        alat_scale_p = p_sym[89]   # a_lat_safe 배율 (κ-cap 상향/하향)
        obs_dmin = p_sym[0]; obs_x = p_sym[1]; obs_y = p_sym[2]
        side_pref = p_sym[3]
        D_detour_p     = p_sym[4]   # side-cost detour offset (rqt)
        R_car_p        = p_sym[5]   # corridor + obs h margin (rqt)
        q_cte_scale_p  = p_sym[6]   # × q_cte_def (rqt)
        R_safe_p       = p_sym[7]
        q_lag_scale_p  = p_sym[8]   # × q_lag_def (rqt)
        e_c_obs_p      = p_sym[9]
        a_lat_safe_p   = p_sym[10]  # curvature speed-cap headroom (rqt)
        D_apex_p       = p_sym[11]  # apex-bias offset (rqt) — ACTIVE: drives e_c_ref (~L888)
        q_psi_scale_p   = p_sym[12]  # × q_psi_def
        q_v_scale_p     = p_sym[13]  # × q_v_def
        q_dd_scale_p    = p_sym[14]  # × q_dd_def
        q_p_scale_p     = p_sym[15]  # × q_p_def
        q_drate_scale_p = p_sym[16]  # × q_d_rate_def
        q_dv_scale_p    = p_sym[17]  # × q_dv_def (longitudinal accel a_x penalty)
        # LMPC SS slots: 50 floats packed as (K, 4) state + (K,) cost-to-go.
        # ss_states[k] = (x*, y*, ψ*, vx*); padding 점들의 Q* = 1e6 → exp(-β·1e6) ≈ 0.
        lmpc_ss_states = ca.reshape(p_sym[18:18 + K_LMPC * 4], 4, K_LMPC)  # 4 x K
        lmpc_ss_Q      = p_sym[18 + K_LMPC * 4 : 18 + K_LMPC * 5]            # (K,)
        lmpc_w_p     = p_sym[68]
        lmpc_alpha_p = p_sym[69]
        lmpc_beta_p  = p_sym[70]
        lmpc_reg_w_p = p_sym[71]
        left_x  = p_sym[72]; left_y  = p_sym[73]
        right_x = p_sym[74]; right_y = p_sym[75]

        # ---- Reference geometry ----
        # s_periodic = s − L·floor(s/L) ∈ [0, L). The solver-internal s
        # state grows unboundedly across laps (dynamics ṡ = p_v stays
        # smooth — no wrap discontinuity → no lap-rollover ACADOS_MINSTEP).
        # ca.floor's derivative is 0, so ∂s_periodic/∂s = 1 throughout,
        # which means the cost gradient w.r.t. s is correct everywhere
        # except exactly at multiples of L (a measure-zero set; for a
        # closed-loop track the spline values at s=L⁻ and s=0⁺ are nearly
        # equal anyway, so the cost-value jump is tiny).
        L_track = float(self.path_length)
        s_periodic = s - L_track * ca.floor(s / L_track)
        ref_x  = self.center_lut_x(s_periodic)
        ref_y  = self.center_lut_y(s_periodic)
        dxt    = self.center_lut_dx(s_periodic)
        dyt    = self.center_lut_dy(s_periodic)
        # Compute sin/cos of the centerline tangent DIRECTLY from dxt, dyt
        # — never form t_angle = atan2(dyt, dxt) as an intermediate
        # variable. atan2 has a branch cut at the −x axis: when the
        # centerline tangent crosses (dxt<0, dyt=0±ε) — which happens on
        # any leftward straight or curve in trackf at s≈56.7 — atan2
        # jumps from +π to −π even though the geometric direction is
        # smooth. That value-jump in t_angle, even with downstream
        # atan2(sin·,cos·) wrap-safety in yaw_err, creates a ~43k cost
        # spike at exactly that s every lap (observed on trackf at
        # (6-7, 18) wrap location). Forming sin/cos directly via
        # normalization is smooth across the jump.
        norm_t = ca.sqrt(dxt * dxt + dyt * dyt + 1e-12)
        sin_t = dyt / norm_t
        cos_t = dxt / norm_t

        # Contour (lateral) and lag (along-track) errors
        e_c = sin_t * (x_ - ref_x) - cos_t * (y_ - ref_y)
        e_l = -cos_t * (x_ - ref_x) - sin_t * (y_ - ref_y)
        # Reference velocity at s, capped by kinematic curvature limit
        # v ≤ √(a_lat_safe / |κ(s)|). Without this cap the centerline
        # ref_v=6 m/s applies even at the hairpins (κ_max≈0.84) where
        # the kinematically feasible v is sqrt(8/0.84) ≈ 3.08 m/s. MPC
        # was forced to choose between (a) tracking ref_v=6 → saturated
        # steer + a_lat slack absorbing huge violation → wall-stuck, or
        # (b) braking only on the q_v term, which competes with q_lag and
        # never wins enough. With the cap, ref_v becomes ~3 m/s in tight
        # corners → MPC naturally slows for the apex; the rest of the
        # cost machinery just tracks the new (kinematically-safe) target.
        # The hard a_lat backstop is now derived as a_lat_safe + 1.0 (see
        # a_lat_max below), so it sits just above this soft cap and is rarely
        # binding — tracking ref_v=√(a_lat_safe/κ) yields steady a_lat≈a_lat_safe,
        # which stays under the backstop (no per-corner slack spike).
        # Smooth fmin via 0.5·(a+b−sqrt((a−b)²+ε)) — differentiable
        # everywhere, kink rounded over ~0.03 m/s.
        ref_v_raw  = self.ref_v(s_periodic) * refv_scale_p
        kappa_at_s = self.abs_kappa_lut(s_periodic)
        v_kin_max  = ca.sqrt(a_lat_safe_p * alat_scale_p / (kappa_at_s + 1e-3))
        EPS_VM     = 1e-3
        # CiMPCC g(κ) 원복 (alpha=2 도 너무 강함, mpc 가 vx=0 local min 갇힘).
        # Heilmeier CSV ref_v 이미 곡률+brake 포함 → 충분.
        diff_v     = ref_v_raw - v_kin_max
        _rv1       = 0.5 * (ref_v_raw + v_kin_max
                            - ca.sqrt(diff_v * diff_v + EPS_VM))
        # ref_v = Heilmeier CSV profile smooth-min'd with the curvature cap
        # √(a_lat_safe/κ).  (A width-aware cap was tried 2026-06-02 but was
        # neutralized — K_WIDTH=100 made the smooth-min always pick _rv1 — and
        # removed 2026-06-03: it solved a non-existent width problem (map min
        # width 0.89 m, not 0.36 m) and only ever slowed the car.)
        ref_v_expr = _rv1

        # ---- Distance² to obstacle (used by h constraint only) ----
        d2 = (x_ - obs_x) ** 2 + (y_ - obs_y) ** 2

        # Heading error — wrap to [-π, π] via atan2(sin, cos) of the
        # difference. Use the angle-subtraction identities so the inputs
        # to atan2 are computed from sin_t, cos_t (smooth) rather than
        # going through t_angle (would have the same branch-cut spike):
        #   sin(ψ − t) = sin(ψ)·cos(t) − cos(ψ)·sin(t)
        #   cos(ψ − t) = cos(ψ)·cos(t) + sin(ψ)·sin(t)
        sin_psi = ca.sin(psi); cos_psi = ca.cos(psi)
        sin_diff = sin_psi * cos_t - cos_psi * sin_t
        cos_diff = cos_psi * cos_t + sin_psi * sin_t
        yaw_err = ca.atan2(sin_diff, cos_diff)

        # ---- NONLINEAR_LS cost form ----
        # y(x,u) = [e_c, e_l, yaw_err, v − ref_v, δ, p − p_max, side_term]
        # The side_term re-introduces obstacle-side preference but in a
        # PSD form (multiplied by sqrt(proximity)·|side_pref|). When no
        # obstacle (sentinel) → proximity≈0 → cost contribution≈0.
        # σ=1.3 (was 0.6 → 1.0 → 1.3; 1.5 caused cost=3337 IPM blowup with
        # the old q_side=25, but with q_side=12 and D_DETOUR_SIDE=0.25
        # there's headroom). Earlier engagement so prediction starts to
        # bend laterally before the obstacle is in the dynamics horizon —
        # avoids the "pred line stuck against obstacle, brake at the last
        # moment" pattern (pred_endpoint within R_safe of obs in 14% of
        # cycles in the σ=1.0 run). Proximity table:
        #   d=2  m: σ=1.0→0.14   σ=1.3→0.31
        #   d=3  m: σ=1.0→0.01   σ=1.3→0.07
        #   d=4  m: σ=1.0→3e-4   σ=1.3→0.008
        sigma_side = _ob['sigma_side']  # (mpc_tuning.yaml) 원래 0.5 # 1.3 → 1.0 → 0.7 → 0.5. Tighter side cost
                          # window so MPC barely detours unless very close
                          # to obstacle. Proximity at d=1m: 0.14, at
                          # d=0.7m: 0.37 — engagement starts at ~1 m.
        proximity_side = ca.exp(-d2 / (2.0 * sigma_side * sigma_side))
        # 0.25 m (was 0.4 → 0.25 to pair with R_safe 0.3 and corridor
        # h-constraint). With the soft corridor active, a 0.4 m detour put
        # the cost minimum at the corridor edge → lateral push fighting the
        # corridor push, and MPC braked to release the tension. 0.25 m
        # places the detour line at e_c=±0.25 well inside the corridor
        # (±0.35 usable), so side cost and corridor cost don't compete.
        # Required clearance from obstacle still met: detour 0.25 + R_safe
        # 0.3 (with car edge buffer) = 0.55 m closest approach center-to-center.
        # D_detour_p (=p_sym[4]) is extracted above but never referenced here —
        # side_term is hardcoded off, so the slot is unused. Kept as a fixed
        # 0.0 placeholder (2026-07-28 대청소: 상위 rqt live-param 제거) purely
        # to preserve p_sym index numbering — do NOT remove the slot itself.
        # |side_pref| ∈ {0, 1}; smooth approx avoids non-differentiable |·|
        # 1e-3 (was 1e-9): keeps sqrt's gradient bounded near 0 — with
        # 1e-9, ∂sqrt/∂x at x=0 is ~16k, which makes the GN Hessian
        # spike when proximity → 0 and overshoots IPM step.
        # B: side_term off (obstacle avoidance, VPMPCC 없음). residual 자리는
        # 유지 (codegen W 호환).
        side_term = ca.SX(0.0)
        # Adaptive lane-tracking attenuation (mirrors IPOPT MPCC.py technique).
        # Multiplies the e_c/e_l cost by 1 - 0.95·exp(-d²/2σ²): centerline
        # tracking is normal far away, fades to ~5% at obstacle center.
        # Without this, q_cte (pull to centerline) and q_side (push to detour
        # line) fight each other near the obstacle → MPC brakes hard to
        # release the tension and outputs jerky control. With attenuation,
        # the centerline pull naturally relaxes when needed, so detour is
        # "free" near obstacle and tracking resumes once past. NONLINEAR_LS
        # form: cost = q · residual², so multiplying residual by sqrt(att)
        # multiplies cost by att.
        # σ_atten = 1.0 (Gaussian width):
        #   d=∞    → att=1.00 (normal)
        #   d=2 m  → att≈0.87
        #   d=1 m  → att≈0.42 (cost halved)
        #   d=0.5  → att≈0.16
        #   d=0    → att≈0.05 (tracking off)
        # B (VPMPCC simplify): obstacle attenuation + κ-attenuation off.
        # VPMPCC 는 corner 에서도 centerline tracking 유지.
        # Restored (avoidance Option 2): lane-tracking attenuation near the
        # selected obstacle, mirrors ifac_mpcc ipopt_kinematic.py:202. Fades
        # the e_c/e_l contouring cost to ~5% at the obstacle center so the
        # detour is "free" and not fighting the centerline pull. Gated by d²
        # to the obstacle (line 837): no obstacle → sentinel → d²≈1e12 →
        # proximity_atten=0 → attenuation=1.0 (Phase-B-identical).
        # 0722y σ_atten 1.0→4.0: 게이트 σ4.5와 동조. σ1.0은 q_cte(라인당김)가
        # 장애물 1.5m 밖에서 87~100% 유지 → h_obs slack이 못 이겨 계획이
        # 0.4~0.55에서 정체(전 bag 공통, "0.3~0.45 접촉"의 진짜 구조 원인).
        # side_term은 코드상 0(꺼짐) — detour는 오로지 h_obs vs q_cte 균형.
        sigma_atten     = 4.0
        proximity_atten = ca.exp(-d2 / (2.0 * sigma_atten * sigma_atten))
        # 0723l 0.95→0.70: 장애물 옆 라인 복원력을 30% 남김. 완전 감쇠(5%)는
        # U자 정점에서 조향 지연발 다이브(-0.18→-0.40, bag 14_07 FAIL 3/6)를
        # 되당길 힘이 없음. h_obs 경계는 hard(zl 2000)라 회피 폭은 불변 —
        # 경계 밖 오버슛에만 복원력 작용. ★codegen 재생성 필요.
        # 0723v: 원형(동적) 모드에선 감쇠 안 함 — 추종 시 "자기 라인 유지 + 감속"
        # (bag 15_33: 상대 근처 |e_c| 0.23/0.51로 라인이탈하던 것 차단). 추월은
        # 라인이탈 비용을 이기는 진행이득이 있을 때만 = "확실할 때만"의 최적화적 표현.
        attenuation     = 1.0 - _ob['line_attenuation'] * proximity_atten * (1.0 - obs1_mode)
        att_kappa   = ca.SX(1.0)
        sqrt_att    = ca.sqrt(attenuation * att_kappa + 1e-6)
        # 8th residual: δ − δ_prev → steer-rate cost. Penalises stage-to-
        # stage δ change directly, which kills within-prediction zigzag.
        # Cost weights q_cte_def, q_lag_def in W are baked at codegen,
        # but multiplied by q_cte_scale_p / q_lag_scale_p at the residual
        # level so the rqt slider can change the EFFECTIVE weight live:
        #   cost = W_baked · (sqrt(scale_p) · sqrt(att) · e_c)²
        #        = W_baked · scale_p · att · e_c²
        # default scale=1.0 → effective = baked.
        sqrt_q_cte_scale   = ca.sqrt(q_cte_scale_p   + 1e-9)
        sqrt_q_lag_scale   = ca.sqrt(q_lag_scale_p   + 1e-9)
        sqrt_q_psi_scale   = ca.sqrt(q_psi_scale_p   + 1e-9)
        sqrt_q_v_scale     = ca.sqrt(q_v_scale_p     + 1e-9)
        sqrt_q_dd_scale    = ca.sqrt(q_dd_scale_p    + 1e-9)
        sqrt_q_p_scale     = ca.sqrt(q_p_scale_p     + 1e-9)
        sqrt_q_drate_scale = ca.sqrt(q_drate_scale_p + 1e-9)
        sqrt_q_dv_scale    = ca.sqrt(q_dv_scale_p    + 1e-9)

        # ── Apex-biased lateral reference (2026-06-01, 해결3 복원) ──
        # e_c_ref=0 (pure centerline) made the cost BLIND to the racing line:
        # progress = centerline-arc-rate (capped at v_max) gives zero reward
        # for apex-cutting, so low q_cte / wide corridor (tested) did NOT make
        # the car cut apexes. The fix is to bias the lateral REFERENCE toward
        # the inside of each corner (κ-dependent) so the contouring cost itself
        # pulls the car onto an in-out-in apex line.
        #   e_c sign: e_c<0 = left, >0 = right of centerline.
        #   left turn (κ>0) → apex on the left → bias e_c_ref<0 = -sign(κ)·D.
        # signed_kappa_lut is the apex-kernel-smoothed signed κ (bias kicks in
        # BEFORE the geometric apex, decays on exit — see the kernel comments).
        # tanh(·/κ_ref) saturates so |e_c_ref| ≤ D_apex on real corners and ≈0
        # on straights. D_apex_p (=D_apex_live param) tunes depth; 0 = off.
        _signed_k = self.signed_kappa_lut(s_periodic)
        e_c_ref = -D_apex_p * ca.tanh(_signed_k / 0.20)

        # 9th residual: q_dv · a_x (longitudinal accel penalty).
        # 2026-05-27 review #8 — VPMPCC q_Δv 와 동일 정신. 이전엔 baked weight
        # self.q_dv=15 만 정의되고 cost 에 미연결 (ghost weight) → 연결.
        _spd_tgt = float(self.speed_target) if self.speed_target else float(self.v_max)  # R3: progress target (decoupled from hard cap)
        y_expr   = ca.vertcat(sqrt_q_cte_scale   * sqrt_att * (e_c - e_c_ref),
                              sqrt_q_lag_scale   * sqrt_att * e_l,
                              sqrt_q_psi_scale   * yaw_err,
                              sqrt_q_v_scale     * (vx_for_cost - ref_v_expr),
                              # curvature feedforward (2026-06-19, ref-confirmed):
                              # pull delta toward atan(L*kappa) — the steer the
                              # curvature needs — not toward 0. Fixes apex under-
                              # steer (all ref MPCCs lack this). δ_ff=0 on straights.
                              sqrt_q_dd_scale    * (delta - 1.15 * ca.atan(self.L * kappa_at_s * ca.sign(_signed_k))),
                              sqrt_q_p_scale     * (p_v - _spd_tgt),
                              side_term,
                              sqrt_q_drate_scale * (delta - delta_prev),
                              sqrt_q_dv_scale    * a_x_input,
                              p_v)   # [9] 선형 진행보상용 (W=0, psi linear)
        # ── LMPC terminal cost addition (Rosolia 2018 §IV.B, simplified) ──
        # 2026-05-28 #18: terminal value-function approximation via softmin
        # over Sampled Safe Set (K=10 nearest historical states).
        #
        # softmin_β(g_i) = -1/β · log Σ_i exp(-β · g_i)
        #   where g_i = Q_i + α · d_i² ,  d_i² = weighted_L2(x_N - x*_i)²
        #
        # Reviewer 2026-05-28: NONLINEAR_LS 의 squared(=cost = w·softmin²) 가
        # 의미상 어긋나지만 EXTERNAL 로 가면 acados 구조 변경 큼. 절충:
        # softmin 자체가 항상 양수가 아니라도 squared 형태 OK (위치 = SS min
        # 으로 동일 수렴). w_lmpc=0 (default) → 영향 0.
        #
        # x_N 의 weighted L2 to ss_states[k] (4-dim subset: px, py, ψ, vx).
        # vx 는 weight 0.3 (vx mismatch 영향 작게 — warm_transfer #4-B 와 짝).
        W_lmpc_diag = ca.diag(ca.SX([1.0, 1.0, 1.5, 0.3]))   # px, py, ψ, vx
        # x_N 의 해당 4 dim: state = [x, y, psi, vx, vy, r, s, delta_prev]
        # 변수명은 우리 acados 스코프 기준 (x_, y_, psi, vx_for_cost).
        # 2026-06-10 unified layout: kinematic 도 8-state (vx = state[3]) →
        # terminal cost 에 vx 사용 가능. 양 모드 동일.
        x_N_4 = ca.vertcat(x_, y_, psi, vx_for_cost)
        # ── min-time terminal: proximity-weighted cost-to-go (NLS-compatible) ──
        # 2026-05-30: the single nearest-point attractor (ss[:,0], CONSTANT
        # Q_best) only TRACKED the nearest stored state (its corner speed
        # included) → never minimized time-to-go → lap time DRIFTED up after
        # ~10 laps (not Rosolia-monotonic; measured). Replace with a smooth
        # softmax over the K safe-set points: terminal cost = proximity-weighted
        # cost-to-go  cog(x_N) = Σ_j w_j·Q_j ,  w_j = softmax_j(-β·d_j²).
        # Minimizing √(lmpc_w·cog) pulls x_N toward LOW cost-to-go (far-along)
        # states that are also REACHABLE (near) → forward progress / min-time.
        # Padding points (Q=1e6, far) get w≈0. Single smooth scalar residual →
        # stays in NONLINEAR_LS (no EXTERNAL_COST); a small reg toward the
        # nearest point keeps the Gauss-Newton Hessian positive-definite (the
        # 2026-05-28 log-sum-exp softmin's −1/β·log instability is avoided).
        _d2_cols = []
        for _j in range(K_LMPC):
            _dj = x_N_4 - lmpc_ss_states[:, _j]
            _d2_cols.append(ca.sum1(_dj * (W_lmpc_diag @ _dj)))
        d2_vec = ca.vertcat(*_d2_cols)                      # (K,1)
        _neg = -lmpc_beta_p * d2_vec
        _w_un = ca.exp(_neg - ca.mmax(_neg))                # numerically stable softmax
        _w_sm = _w_un / (ca.sum1(_w_un) + 1e-12)
        cog = ca.sum1(_w_sm * lmpc_ss_Q)                    # proximity-weighted cost-to-go
        d2_best = d2_vec[0]                                 # nearest-point reg (conditioning)
        lmpc_residual = ca.sqrt(lmpc_w_p + 1e-12) * ca.sqrt(
            cog + (lmpc_alpha_p + lmpc_reg_w_p) * d2_best + 1e-6
        )

        # analytic terminal cost-to-go: land horizon-end vx on the global optimal
        # speed profile ref_v_expr(s_N). Removes N-dependency (brakes for corners
        # beyond the horizon). Reuses BO-tunable q_v scale; baked terminal emphasis x3.
        # 2026-06-10 unified layout: vx is a state in BOTH modes → vterm everywhere.
        vterm_res = sqrt_q_v_scale * (vx_for_cost - ref_v_expr)
        y_expr_e = ca.vertcat(e_c, e_l, yaw_err, vterm_res, lmpc_residual)

        # ---- Constraints (h) — minimal set for stable SQP_RTI ----
        # 1) obstacle half-plane (replaces 2026-05-06 the non-convex annulus
        #    d²−R²≥0). The annulus form is bistable: slack absorbs the
        #    "diving through obstacle" path as a local optimum, so the
        #    predicted trajectory ends *inside* the keepout (observed:
        #    pred_endpoint within R_safe of obs in 14% of obstacle-active
        #    cycles → "경로 라인이 장애물에 막힘" symptom + last-moment
        #    brake/burst). Half-plane: project (car − obs) onto the
        #    right-perpendicular of the centerline tangent at the predicted
        #    stage, then require side_pref · projection ≥ R_safe + R_CAR.
        #    Convex, one-sided, prediction physically cannot cross to the
        #    other side. side_pref ∈ {-1, 0, +1} from decide_side_pref;
        #    when side_pref ≈ 0 (no obstacle / sentinel), a |side_pref|-
        #    gated big-M term keeps the constraint trivially satisfied.
        # 2) corridor lateral bound — added 2026-05-06 to prevent sim-wall
        #    stuck. Project corridor boundary points onto the same right-
        #    perpendicular as e_c, then bound e_c. Smooth max/min so it
        #    works for CW/CCW orientation and avoids the kink that caused
        #    HPIPM S_MINSTEP at narrow stages.
        # 3) lateral accel limit (kinematic).
        # R_CAR now read from p_sym[5] (rqt-tunable). Default value in
        # self.R_car_live = 0.02. Live tunable per-cycle. Used by both
        # h_obs (obstacle margin) and h_corridor (wall margin).
        R_CAR = R_car_p
        # ---- (1) obstacle half-plane (Frenet + proximity-gated) ----
        # Frenet form: side_pref · (e_c − e_c_obs) ≥ R + R_CAR. Naturally
        # curve-aware because e_c at each predicted stage is measured
        # against the centerline at THAT stage's s. No tangent rotation
        # issue.
        # Proximity gate: gate = |side_pref| · exp(−d²/2σ²). Disables the
        # constraint smoothly when far from the obstacle (d>3 m → gate≈0)
        # or when no obstacle is selected (side_pref≈0). Big-M on the
        # complementary side keeps the constraint trivially satisfied
        # outside the active zone.
        SIGMA_OBS = _ob['sigma_gate']   # (mpc_tuning.yaml) 원래 4.5 # 0722f: 2.5→4.5. σ2.5는 commit(6-10m)에도 게이트≈0.06으로
                          # 제약이 잠들어, 계획이 2.5m 안에서 0.9m를 급횡이동해야
                          # 했음(동역학 불가→slack→0.5m 접촉, bag 15_04 계측).
                          # 4.5: 5m서 게이트 0.54, 3m서 0.80 — 점진 detour 개시.
                          # (2026-06-29 was 2.5; 원래 1.0)
        prox_obs  = ca.exp(-d2 / (2.0 * SIGMA_OBS * SIGMA_OBS))
        abs_side  = ca.sqrt(side_pref * side_pref + 1e-3)
        gate      = abs_side * prox_obs           # ≈ 1 only when active
        # 0722z BIG-M 누수 픽스 50→5: h = gate·raw + (1−gate)·BIG 구조에서
        # gate가 근접에서도 1이 아니라(σ4.5, d=0.42 → 0.996) (1−gate)·50 ≈ +0.2가
        # 위반을 공짜로 허용 — 계획 평형이 정확히 keepout−0.2에 앉음 (bag 20_37
        # OBSDBG 실증: d=0.41서 raw=−0.23, sl0=0 = solver 눈엔 무위반).
        # 몇 주간 "간격 0.3~0.45 접촉"의 구조 원인. R_safe/zl 튜닝이 안 먹힌 이유.
        # 5.0: d=0.42 누수 0.02m(무해) / d=5m 허용 4.3m(원거리 OFF 기능 유지).
        BIG_OBS   = _ob['big_m']   # (mpc_tuning.yaml) 원래 5.0
        h_obs_hp = (gate * (side_pref * (e_c - e_c_obs_p) - (R_safe_p + R_CAR))
                    + (1.0 - gate) * BIG_OBS)
        # 0723u 원형 모드 (동적 상대, A2-확장): 스테이지별 예측 위치 (obs_x,obs_y)
        # = pred_obs[k] 중심의 유클리드 keepout. 종+횡 모두 척력 → 상대가 빠르면
        # 감속(추종), 느리면 우회(추월)가 최적화에서 자연 발생 — 상태머신 없음.
        # 부재 시 sentinel(1e6) → d² 거대 → 항상 만족 (게이트 불필요).
        _R_dyn = R_safe_p + R_CAR
        h_obs_circ = d2 - _R_dyn * _R_dyn
        h_obs = (1.0 - obs1_mode) * h_obs_hp + obs1_mode * h_obs_circ
        # ---- (1b) P2 Stage-B: 2번째 장애물 half-plane (동일 형식) ----
        # #1 release 전에도 #2가 제약에 존재 → 가림 쌍/연속 장애물을 계획이
        # 하나의 연속 곡선으로 처리 (release 대기 1.5m 갭 제거).
        d2_2 = (x_ - obs2_x) ** 2 + (y_ - obs2_y) ** 2
        prox_obs2 = ca.exp(-d2_2 / (2.0 * SIGMA_OBS * SIGMA_OBS))
        abs_side2 = ca.sqrt(side2_pref * side2_pref + 1e-3)
        gate2 = abs_side2 * prox_obs2
        h_obs2 = (gate2 * (side2_pref * (e_c - e_c_obs2_p) - (R_safe_p + R_CAR))
                  + (1.0 - gate2) * BIG_OBS)
        # ---- (1c) 0723b: 3번째 장애물 half-plane (동일 형식) ----
        d2_3 = (x_ - obs3_x) ** 2 + (y_ - obs3_y) ** 2
        prox_obs3 = ca.exp(-d2_3 / (2.0 * SIGMA_OBS * SIGMA_OBS))
        abs_side3 = ca.sqrt(side3_pref * side3_pref + 1e-3)
        gate3 = abs_side3 * prox_obs3
        h_obs3 = (gate3 * (side3_pref * (e_c - e_c_obs3_p) - (R_safe_p + R_CAR))
                  + (1.0 - gate3) * BIG_OBS)
        # ---- (2) corridor ----
        w_left  = sin_t * (left_x - ref_x) - cos_t * (left_y - ref_y)
        w_right = sin_t * (right_x - ref_x) - cos_t * (right_y - ref_y)
        EPS_MM    = 1e-4
        diff_lr   = w_left - w_right
        sqrt_diff = ca.sqrt(diff_lr * diff_lr + EPS_MM)
        upper_lat = 0.5 * (w_left + w_right + sqrt_diff) - R_CAR
        lower_lat = 0.5 * (w_left + w_right - sqrt_diff) + R_CAR
        h_corridor_top = upper_lat - e_c
        h_corridor_bot = e_c - lower_lat
        # ---- (3) a_lat ----
        # Kinematic: a_lat = v² tan(δ)/L (geometric centripetal acc).
        # Dynamic: a_lat = vx · r (true lateral acceleration). Both
        # are computed in the model branch above into `a_lat_expr`.
        a_lat = a_lat_expr
        # ---- (4) friction ellipse (2026-06-10 spec) ----
        # (a_x/a_lim)² + (a_lat/a_lim)² ≤ 1, a_lim = μ·g·η. u[0]=a_x 는 unified
        # 입력 레이아웃에서 양 모드 공통. 제동·가속 중 가용 a_lat 이 자동 감소
        # (combined-slip). 고그립(μ=1.0489)에선 a_lim≈9.8 → 거의 안 물림 = 기존
        # 동작 보존; 저그립(μ=0.6)에선 a_lim≈5.6 = 실질 한계.
        _a_lim = max(1e-3, grip_a_lat_limit(self.dyn_mu, self.ellipse_frac))
        h_ellipse = (u[0] / _a_lim) ** 2 + (a_lat / _a_lim) ** 2
        # C-STEER (2026-06-19): hard steering-rate row |(δ−δ_prev)/dT| ≤ sv_max.
        # δ_prev is the augmented state (= last-applied steer); bounded as a
        # path h-constraint (NOT slacked) it forces servo-feasible steering and
        # kills the cycle-to-cycle hunting that the soft q_drate cost could not.
        steer_rate = (u[1] - delta_prev) / self.dT
        model_ac.con_h_expr = ca.vertcat(h_obs, h_corridor_top, h_corridor_bot, a_lat, h_ellipse, h_obs2, h_obs3, steer_rate)
        # C-STEER node-0: path con_h above covers stages 1..N-1; the PUBLISHED
        # control is u_0 (stage 0). A stage-0 h-constraint with ONLY the steer-
        # rate (state-only rows would be constant at the fixed x0 → drop them)
        # rate-limits the published δ_0 → kills the full-lock launch swerve.
        model_ac.con_h_expr_0 = ca.vertcat(steer_rate)

        # ---- Compose model ----
        # B4' error-dynamics regression: add affine velocity correction to the
        # blended dynamic f_expl. nx-wide via explicit rows [3,4,5]=[vx,vy,r] so
        # it composes with joint-α (nx=18) and plain (nx=8) alike. Gated: when
        # _err_regr is False the slots stay 0 → exact baseline f_expl.
        if self._err_regr:
            f_expl = f_expl + ca.vertcat(
                ca.SX.zeros(3), e_corr_sym[0], e_corr_sym[1], e_corr_sym[2],
                ca.SX.zeros(f_expl.shape[0] - 6))
        model_ac.f_impl_expr = xdot - f_expl
        model_ac.f_expl_expr = f_expl
        model_ac.x = x
        model_ac.xdot = xdot
        model_ac.u = u
        model_ac.z = ca.vertcat([])
        model_ac.p = p_sym
        ocp.model = model_ac
        ocp.dims.np = n_p_total

        # ---- Cost: NONLINEAR_LS ----
        # W is fixed at codegen; LIVE tuning of q_cte/q_lag/q_d_delta done by
        # multiplying inside y_expr if needed later. For now, use defaults.
        # Tuned for kinematic limits: too large q_cte/q_psi forces tighter
        # tracking than dynamics can follow at corners → car stalls or
        # bounces off the line.
        # 2026-05-27: baked weights 강화 — 박힘 ↓ 위해 centerline 추종 강.
        # BO 가 scale 0.3~5 학습 → effective q_cte=4.5~75, q_lag=24~400, etc.
        # local minima 회피 위해 baked 는 paper VPMPCC 의 중간 값.
        q_cte_def     = _cw['q_cte']    # 9→15 (centerline lateral 추종 강)
        q_lag_def     = _cw['q_lag']    # 45→80 (along-track 강)
        q_psi_def     = _cw['q_psi']
        q_v_def       = _cw['q_v']
        q_dd_def      = _cw['q_dd']
                                # × 20 stages now totals ~48 vs q_lag 60,
                                # so solver feels significant cost for
                                # large |δ|. Without this the solver was
                                # cycle-to-cycle picking +0.4 / −0.1
                                # alternating (65% sign-flip rate observed)
                                # because both gave similar predicted cost
                                # — flat surface. Higher q_d_delta forces
                                # the solver to commit to moderate δ
                                # spread across the horizon, killing the
                                # prediction-line trembling.
        q_p_def       = _cw['q_p']     # progress pull 약화 (4→1) — 보수적 시작, BO 가 scale 학습
        q_side_def    = _cw['q_side']     # very soft hint (was 12 → 3, mirrors
                                # IPOPT MPCC.py W_SIDE=3). With the new
                                # attenuation that fades q_cte/q_lag near
                                # the obstacle, side cost no longer needs
                                # to fight q_cte for the detour — it's
                                # just a gentle nudge. q_side=12 was too
                                # much: it produced cost spikes when the
                                # car was forced to choose between the
                                # detour line and the half-plane edge.
        q_d_rate_def  = _cw['q_d_rate']    # 원복 (S2 100 으로 안 효과)
                                # 30 → 50 → 80. Strong rate cost minimises
                                # corner→straight transition wobble.
        q_dv_def      = _cw['q_dv']    # 2026-05-27 #8: longitudinal accel a_x penalty.
                                # 이전엔 self.q_dv = 15 만 정의되고 cost 연결
                                # 안 됨 → 9th residual 로 연결.
        ny   = 10  # 9 + p_v[9] (B1 선형 진행보상용, W=0)
        # terminal: vx cost-to-go residual (vterm) before lmpc_residual — BOTH modes
        # (2026-06-10 unified layout).
        ny_e = 5
        # ---- Cost: CONVEX_OVER_NONLINEAR (Phase B0, 2026-06-04) ----
        # Migrated from NONLINEAR_LS. With ψ(r) = ½·rᵀWr and r = y_expr this
        # reproduces the LS cost EXACTLY (same Gauss-Newton Hessian), but the
        # CONL form lets later phases add convex non-LS terms — e.g. a linear
        # progress reward −γ·p_v (B1) that pure NLS cannot express.
        W_mat = np.diag([q_cte_def, q_lag_def, q_psi_def,
                         q_v_def, q_dd_def, q_p_def, q_side_def,
                         q_d_rate_def, q_dv_def, 0.0])   # [9]=p_v: 제곱기여 0
        # W_e: vx terminal cost-to-go (q_v_def*3) before the LMPC residual
        # (last entry 1.0, since residual already carries sqrt(lmpc_w)).
        # 2026-06-10 unified layout: same in both modes.
        W_e_diag = [q_cte_def * _cw['terminal_cte'], q_lag_def * _cw['terminal_lag'],
                    q_psi_def * _cw['terminal_psi'], q_v_def * _cw['terminal_v'], 1.0]
        W_e_mat = np.diag(W_e_diag)
        ocp.cost.cost_type   = 'CONVEX_OVER_NONLINEAR'
        ocp.cost.cost_type_0 = 'CONVEX_OVER_NONLINEAR'
        ocp.cost.cost_type_e = 'CONVEX_OVER_NONLINEAR'
        # residual r(x,u) — identical to the NLS y_expr (live q_*_scale already
        # baked inside via sqrt(q_*_scale) factors → effective w = W·q_scale).
        ocp.model.cost_y_expr   = y_expr
        ocp.model.cost_y_expr_0 = y_expr
        ocp.model.cost_y_expr_e = y_expr_e
        # outer convex ψ(r) = ½ rᵀ W r, written in a fresh symbolic r.
        _r_conl   = ca.SX.sym('r_conl',   ny)
        _r_conl_e = ca.SX.sym('r_conl_e', ny_e)
        ocp.model.cost_r_in_psi_expr   = _r_conl
        ocp.model.cost_r_in_psi_expr_0 = _r_conl
        ocp.model.cost_r_in_psi_expr_e = _r_conl_e
        # B1 선형 진행보상: -gamma*p_v (r[9]). gamma=0이면 기존과 동일. CONL이라 GN Hessian 불변(PSD).
        _prog = float(getattr(self, "progress_gamma", 0.0))
        ocp.model.cost_psi_expr   = 0.5 * ca.mtimes([_r_conl.T,   ca.DM(W_mat),   _r_conl]) - _prog * _r_conl[9]
        ocp.model.cost_psi_expr_0 = 0.5 * ca.mtimes([_r_conl.T,   ca.DM(W_mat),   _r_conl]) - _prog * _r_conl[9]
        ocp.model.cost_psi_expr_e = 0.5 * ca.mtimes([_r_conl_e.T, ca.DM(W_e_mat), _r_conl_e])
        ocp.cost.yref   = np.zeros(ny)
        ocp.cost.yref_0 = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        # ---- Dimensions / horizon ----
        ocp.solver_options.N_horizon = self.N
        # multi-dt time_steps (pyramidal grid 면 가까이 sharp, 멀리 long planning).
        # tf = sum(time_steps), acados 가 stage 별 dt 자동 적용.
        ocp.solver_options.time_steps = np.array(self.time_steps, dtype=float)
        ocp.solver_options.tf = float(np.sum(self.time_steps))
        Tf = ocp.solver_options.tf

        # ---- Initial state placeholder ----
        ocp.constraints.x0 = np.zeros(nx)

        # ---- Input bounds ----
        # 2026-06-10 unified layout: u = [a_x, δ, p_v] in BOTH modes — bounds on
        # accel/decel from sim (max_accel=7.51, max_decel=8.26). v_max enforced
        # as a STATE bound on vx instead.
        ocp.constraints.idxbu = np.array([0, 1, 2])
        # a_min 줄임: -8.26 → -3.0. 강한 brake (8m/s²) 는 실제로 거의
        # 발생 안 하는데 솔버가 vx<1 singular 영역에서 "어떻게든 멈춰"
        # 라며 -8.26 출력 → 다음 cycle vx 음수 예측 → Pacejka 망가짐
        # → ACADOS_MINSTEP 캐스케이드. -3.0 이면 충분히 감속 가능하고
        # 솔버가 미친 brake 못 함.
        a_min_dyn = A_MIN_DYN   # 솔버 제동한계 — model_policy 단일소스 (ref_v a_long 과 동기)
        a_max_dyn = 10.0   # 2026-07-03 2차: 8→10 (모터 실가속 ~2.5라 장식 상한, 사용자 결정)
                           # grip ~4 m/s². 7.51 (sim max_accel) 은 saturated 되어
                           # solver 가 "instant fix" 권장 → traj 예측 unrealistic →
                           # zigzag + over-shoot. 4 가 realistic, 차도 추종 가능.
        ocp.constraints.lbu = np.array([a_min_dyn, self.theta_min, 0.0])
        ocp.constraints.ubu = np.array([a_max_dyn, self.theta_max, self.p_max])
        # Kept for the per-stage kinematic p_v grip cap in solve() — per-stage
        # solver.set(k,"ubu",…) must pass the FULL bound vector, so remember
        # the codegen-time values to copy-and-modify slot 2 only.
        self._ubu_codegen = np.array([a_max_dyn, self.theta_max, self.p_max])
        # vx lower bound 0: 절대 후진 안 함. -1.0 허용 시 tire model
        # 의 vx<0 영역 (실제로는 vx²=0.01 이라도 sign(vx)·F 가 발산)
        # 에서 솔버 폭주 가능. 0 으로 강제하면 reverse 자체 차단.
        ocp.constraints.idxbx = np.array([3, 4, 5])
        ocp.constraints.lbx   = np.array([0.0, -10.0, -20.0])
        ocp.constraints.ubx   = np.array([self.v_max + 0.5, 10.0, 20.0])   # 작은 margin (0.5 m/s). v_max 까지 빡빡하게 강제하면 solver IPM 발산 → 12m teleport. 원래 +2 로 풀어줌.

        # ---- h bounds (only h_obs + a_lat now) ----
        # uh[0] HUGE: obstacle absence is encoded by setting obs_x/y to a
        # sentinel ~1e6 m, which makes h_obs = d² ≈ 1e12. With BIG=1e3
        # that's a 10⁹-magnitude constraint violation every cycle, slack
        # cost explodes, and HPIPM produces NaN step directions →
        # ACADOS_MINSTEP. With uh=1e15 the trivial case is well within
        # bounds.
        # a_lat_max = HARD backstop on actual lateral accel (vx·r). It must
        # sit ABOVE the soft κ-cap a_lat_safe, else it binds every corner:
        # tracking ref_v=√(a_lat_safe/κ) gives steady-corner a_lat = vx²κ =
        # a_lat_safe, so a_lat_max < a_lat_safe (the old 8 vs deploy-9 case)
        # silently slacks the constraint on every apex. Derive it from the
        # codegen-time a_lat_safe_live (pushed from yaml before setup_MPC)
        # with a +1.0 backstop margin, floored at the historical 8.0.
        #   NOTE: a full 12 was tried and reverted (hairpin trembling — cost
        #   surface too flat at high κ → IPM zigzag). +1.0 keeps the cap a
        #   true backstop (just above the soft cap) without re-entering that
        #   flat region; the steer-output EMA filter further damps any wobble.
        a_lat_max = max(8.0, float(self.a_lat_safe_eff()) + 1.0)
        self._log.info(f"[MPC-acados] a_lat hard cap = {a_lat_max:.2f} (a_lat_safe={a_lat_max - 1.0:.2f} + 1.0 backstop)")
        # h order: [h_obs, corr_top, corr_bot, a_lat, ellipse, h_obs2, h_obs3, steer-rate]
        # ellipse 행: 0 ≤ h ≤ 1 (h 는 제곱합이라 하한 0 은 자연 충족 — 무해).
        ocp.constraints.lh = np.array([0.0, 0.0, 0.0, -a_lat_max, 0.0, 0.0, 0.0, -self.sv_max])  # +h_obs2(5)+h_obs3(6), steer-rate (hard)
        ocp.constraints.uh = np.array([1e15, 1e15, 1e15, a_lat_max, 1.0, 1e15, 1e15, self.sv_max])
        # C-STEER stage-0 bounds: steer-rate only, hard (not slacked).
        ocp.constraints.lh_0 = np.array([-self.sv_max])
        ocp.constraints.uh_0 = np.array([self.sv_max])
        # Slack on first five — corridor, obstacle, a_lat, ellipse can be
        # transiently violated. Slack absorbs without triggering cascade.
        ocp.constraints.idxsh = np.array([0, 1, 2, 3, 4, 5, 6])   # +h_obs2(5)+h_obs3(6); Σα 비slack
        ns = 7
        ocp.constraints.lsh = np.zeros(ns)
        ocp.constraints.ush = np.zeros(ns)
        # Per-constraint slack tuning — second reduction. Quadratic Zl was
        # the dominant source of cost spikes (slack=2m × Zl=500 = 2000
        # per stage); reducing 500→80 capped spikes at ~6k. Reducing
        # further to 30 caps them at ~2-3k while keeping a bounded but
        # still-meaningful push back to the constraint via the linear zl
        # term. Linear zl is bumped slightly so the gradient at small
        # violations stays informative (push = zl + 2·Zl·s; with smaller
        # Zl, the linear term is what the optimizer feels for s < 0.5m).
        #   idx 0 (h_obs):       zl=40,  Zl=30   (was 30/80)
        #   idx 1,2 (corridor):  zl=20,  Zl=15   (was 15/30)
        #   idx 3 (a_lat):       zl=50,  Zl=15   (was 50/30)
        #   idx 4 (ellipse):     zl=50,  Zl=15   (a_lat 행과 동일 — spec §2)
        # (2026-06-02 corridor 6배 강화 시도 → 역효과 5.1접촉/랩: 강한 corridor
        #  push 가 반대편 클립 유발. 원복.)
        # 2026-07-29 재시도 (corridor 20/15 → 200/120, 장애물 행과 동급):
        #   근거 — 존 없는 bag 15_07 에서 48초 8충돌, 접촉 지점의 실여유가
        #   s20~22 에서 -0.043 m. 코리도 한계는 벽-R_car(0.30) 인데 차는
        #   벽-0.155 까지 감 = 0.145 m 위반. 그 비용이 zl+2·Zl·s = 20+2·15·0.145
        #   ≈ 24 로 진행이득에 밀림. 장애물 행(200/120)의 1/10 이라 벽이 항상
        #   최약 탈출구였다. 존 4개로 트랙 58% 를 덮던 것은 이 약한 기본값을
        #   구간마다 손으로 메운 반창고.
        #   ⚠ 2026-06-02 실패(반대편 클립)를 반드시 재확인할 것 — 좌/우 접촉을
        #   따로 세고 랩 수를 충분히 볼 것. 악화되면 이 블록만 되돌리면 된다.
        # 2026-07-29: 값 자체는 config/mpc_tuning.yaml 에서 읽는다 (수정 편의).
        _slack = _tune['slack']
        ocp.cost.zl = _slack['zl']
        ocp.cost.zu = _slack['zu']
        ocp.cost.Zl = _slack['Zl']
        ocp.cost.Zu = _slack['Zu']
        # ── 2026-07-09 국소 corridor 하드닝 ────────────────────────────
        # 스텁(s≈21.6) 같은 지점에서 soft corridor 가 진행이득에 뚫리는 문제:
        # slack 기본값을 저장해두고 solve() 가 per-stage 로 존 안에서만 corridor
        # 행(1,2) 벌금을 corridor_hard_zones 의 factor 배로 cost_set (codegen 불필요).
        # 위 ocp.cost.* 와 같은 출처(mpc_tuning.yaml)를 쓴다 — 두 곳이 어긋나면
        # 존 진입/이탈 때 벌점이 튀므로 반드시 같은 dict 에서 복사한다.
        self._slack_def = {k: v.copy() for k, v in _slack.items()}
        self._slack_last = {}          # stage → 마지막 적용 factor (전환시에만 cost_set)
        # 2026-08-19: 4원소로 확장 — slack_factor(코리도 벌금 배율) 와
        # qcte_factor(그 존의 q_cte 배율) 는 서로 독립 노브다.
        self.corridor_hard_zones = []  # [(s0, s1, slack_factor, qcte_factor)] — mpc_node 가 채움

        # ---- Initial parameter values (overridden every cycle) ----
        ocp.parameter_values = np.zeros(n_p_total)
        ocp.parameter_values[4]  = 0.0                      # D_DETOUR slot: dead (side_term hardcoded 0), kept for index stability
        ocp.parameter_values[5]  = self.R_car_live          # R_CAR
        ocp.parameter_values[6]  = self.q_cte_scale_live    # q_cte scale
        ocp.parameter_values[7]  = self.R_safe_live         # R_safe
        ocp.parameter_values[8]  = self.q_lag_scale_live    # q_lag scale
        ocp.parameter_values[10] = self.a_lat_safe_eff()    # A_LAT_SAFE (μ-clamped)
        ocp.parameter_values[11] = self.D_apex_live         # D_apex
        ocp.parameter_values[12] = self.q_psi_scale_live    # q_psi scale
        ocp.parameter_values[13] = self.q_v_scale_live      # q_v scale
        ocp.parameter_values[14] = self.q_dd_scale_live     # q_dd scale (steer reg)
        ocp.parameter_values[15] = self.q_p_scale_live      # q_p scale (progress)
        ocp.parameter_values[16] = self.q_drate_scale_live  # q_d_rate scale
        ocp.parameter_values[17] = self.q_dv_scale_live     # q_dv scale (a_x)
        ocp.parameter_values[88] = 1.0   # refv_scale (학습 off 기본)
        ocp.parameter_values[89] = 1.0   # alat_scale

        # ---- Solver options ----
        # NONLINEAR_LS form makes the Gauss-Newton Hessian = J^T·W·J
        # automatically PSD. SQP_RTI works again. PROJECT regularize is
        # an extra safety net (acados forum recommendation).
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # SQP_RTI single-iter for BOTH kinematic and dynamic.
        # Multi-iter SQP was diverging at startup (~15 iters too much
        # freedom when warm-start far from optimum → IPM step grows
        # unbounded). RTI = 1 SQP iter per cycle, naturally stable.
        # Pacejka's higher-order derivatives are tamed by strong LM (3.0)
        # below — same approach as kinematic, just more regularization.
        #
        # nlp_solver_iters>1 (2026-06-19): use AS-RTI (advanced-step RTI), NOT
        # full SQP. Full SQP+globalization hit Maximum_Iterations every cycle →
        # the node's RTI-oriented logic reset the warm-start each time → 0.5 m/s
        # crawl + STUCK (tested 06-19). AS-RTI keeps SQP_RTI semantics (SUCCESS
        # status, real-time, warm-start preserved) while doing _nlp_iters
        # STANDARD_RTI iterations per cycle — so it escapes the 1-iter active-set
        # trap without the divergence (uncapped SQP) OR the reset-thrash (capped
        # SQP). as_rti_level: 4 = STANDARD_RTI (each AS iter = one full RTI).
        _nlp_iters = int(getattr(self, 'nlp_solver_iters', 1))
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        if _nlp_iters > 1:
            ocp.solver_options.as_rti_level = 4          # STANDARD_RTI
            ocp.solver_options.as_rti_iter  = _nlp_iters
        self._log.info("[MPC-acados] nlp_solver=SQP_RTI as_rti_iter=%d (nlp_solver_iters=%d)"
                       % (int(getattr(ocp.solver_options, 'as_rti_iter', 1)), _nlp_iters))
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # PROJECT (CONVEXIFY caused hairpin trembling — Hessian rebuilt
        # slightly differently each cycle, IPM step direction wobbled
        # across multiple near-equal optima in tight corners).
        # PROJECT just clips negative eigenvalues to small positive,
        # leaving block structure stable cycle-to-cycle → deterministic
        # step direction.
        ocp.solver_options.regularize_method = 'PROJECT'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.print_level = 0
        # Back to 100 — 200 was an exploration knob that combined with
        # CONVEXIFY/loose a_lat caused hairpin instability (more iters
        # = more chance to wander between near-equal optima). 100 is
        # sufficient when cost surface is well-conditioned.
        ocp.solver_options.qp_solver_iter_max = 100
        # LM for dynamic linear tire. Balanced: strong enough for IPM
        # stability with slip derivatives, loose enough that cost weights can
        # actually steer the solution (not over-damped). Overridable via the
        # `lm_dynamic` ROS param (set onto self before setup_MPC) so it can be
        # swept / BO-tuned without code edits.
        _lm_dyn = float(getattr(self, 'lm_dynamic', 1.0))
        # NOTE(2026-06-10 unified layout): kept mode-dependent ON PURPOSE — this
        # is a NUMERIC conditioning knob, not an algorithmic difference. f_kin has
        # no Pacejka slip derivatives, so it needs far less damping (0.2).
        ocp.solver_options.levenberg_marquardt = _lm_dyn if self.use_dynamic else 0.2
        self._log.info("[MPC-acados] levenberg_marquardt = %.3f (use_dynamic=%s)"
                       % (ocp.solver_options.levenberg_marquardt, self.use_dynamic))

        # Codegen + build — dir/json keyed to (model, nx) so kinematic<->dynamic
        # toggles never reuse a stale codegen (see codegen_paths()).
        ocp.code_export_directory, json_path = codegen_paths(
            self.use_dynamic, self._nx_solver,
            dyn_mu=float(self.dyn_mu))
        # ── 2026-07-16 warm-reuse: AcadosOcpSolver 기본값은 매 시작 풀 재생성
        # (generate=True/build=True). GP가 동역학에 구워진 뒤 생성 단계가 수 분 →
        # dylib+json+스탬프(GP ckpt 경로·mtime)가 일치하면 로드만 한다.
        # GP ckpt 변경 → 스탬프 불일치 → 자동 풀 재생성. 그 외 codegen-baked
        # 노브(N/dT/speed_target/lookahead/dyn_mu 외 가중치) 변경 시엔 기존 관례대로
        # rm -rf ~/.acados_codegen/* 필수 (codegen_paths NOTE 참조).
        _stamp_path = json_path + ".stamp"
        # 0818: use_gp_casadi=false 인데 ckpt 경로가 남아 있으면 스탬프가 같아
        # GP 구운 솔버를 warm-reuse 하던 함정 — 플래그가 꺼지면 서명을 nogp 로.
        _gp_ck = (os.path.expanduser(getattr(self, 'gp_casadi_ckpt', '') or '')
                  if getattr(self, 'use_gp_casadi', False) else '')
        try:
            _gp_sig = "%s:%d" % (_gp_ck, int(os.path.getmtime(_gp_ck))) if _gp_ck and os.path.isfile(_gp_ck) else "nogp"
        except OSError:
            _gp_sig = "nogp"
        # 0724: p/nx 차원도 스탬프에 포함 — 차원 바뀐 codegen 을 조용히
        # warm-reuse 하는 함정 방지 (오늘 np 88→90 미감지 실사고).
        _gp_sig += ":np%d:nx%d" % (int(ocp.dims.np), int(self._nx_solver))
        # 0729: mpc_tuning.yaml 의 codegen-baked 값(cost_weights/obstacle)도
        # 스탬프에 포함 — 이전엔 가중치를 바꿔도 스탬프가 못 보고 옛 솔버를
        # warm-reuse 해서 "yaml 을 고쳤는데 그대로"가 됐다 (rm -rf 수동 관례).
        # slack/corridor 는 런타임 반영이라 제외 (불필요한 재생성 방지).
        _gp_sig += ":tw%s" % ",".join(
            "%g" % _cw[k] for k in sorted(_cw))
        _gp_sig += ":ob%s" % ",".join(
            "%g" % _ob[k] for k in sorted(_ob))
        _lib_ok = os.path.isdir(ocp.code_export_directory) and any(
            f.startswith("libacados_ocp_solver") for f in os.listdir(ocp.code_export_directory))             if os.path.isdir(ocp.code_export_directory) else False
        _stamp_ok = False
        try:
            _stamp_ok = open(_stamp_path).read().strip() == _gp_sig
        except OSError:
            pass
        _reuse = _lib_ok and os.path.isfile(json_path) and _stamp_ok
        if _reuse:
            self._log.info("[MPC-acados] warm-reusing codegen %s (rm -rf ~/.acados_codegen/* 로 강제 재생성)",
                           ocp.code_export_directory)
        else:
            self._log.info("[MPC-acados] generating solver (GP 포함 수 분 걸림, 최초/GP변경시)...")
        self.solver = AcadosOcpSolver(ocp, json_file=json_path,
                                      generate=not _reuse, build=not _reuse)
        if not _reuse:
            try:
                open(_stamp_path, "w").write(_gp_sig)
            except OSError:
                pass
        # GP residual wrap (gp_residual_wrapper.wrap_solver_with_gp) needs the
        # AcadosOcp object — setup_MPC built it as a local, so stash it.
        self.ocp = ocp
        self._log.info("[MPC-acados] solver ready")

        # Stash dim info for solve()
        self._n_p_const = n_p_const
        self._n_p_total = n_p_total

        # Storage
        self.X0 = np.zeros((self.N + 1, getattr(self, '_nx_solver', self.n_states)))
        self.u0 = np.zeros((self.N, self.n_controls))

    # ------------------------------------------------------------------
    # Bound construction helpers (mirror IPOPT solve())
    # ------------------------------------------------------------------
    def _obstacle_frenet(self, ox, oy):
        """Obstacle's Frenet coords (s_obs, e_c_obs) at its nearest centerline
        point. Returns (0.0, 0.0) for sentinel. Computed ONCE when an
        obstacle is committed (see solve()'s commit-once logic), not every
        cycle, so the brute-force search is one-shot per obstacle pass.
        """
        if ox > 1e5 or oy > 1e5 or self.kappa_grid is None:
            return 0.0, 0.0
        # 0722q numpy 그리드 캐시: 기존 브루트포스(1541점×CasADi 콜2)는 호출당
        # 10-20ms — obs2 스캔이 매 사이클 호출하며 solve 8.5→30ms 폭증의 진범
        # (bag 16_30: 감지0 구간 9.4ms vs 감지시 30.6ms). 1회 샘플링 후 argmin.
        if getattr(self, '_cgrid_x', None) is None:
            n0 = len(self.kappa_grid)
            self._cgrid_s = np.arange(n0) * self.kappa_ds
            self._cgrid_x = np.array([float(self.center_lut_x(s)) for s in self._cgrid_s])
            self._cgrid_y = np.array([float(self.center_lut_y(s)) for s in self._cgrid_s])
        _i = int(np.argmin((self._cgrid_x - ox) ** 2 + (self._cgrid_y - oy) ** 2))
        best_s = float(self._cgrid_s[_i])
        rxc = float(self._cgrid_x[_i])
        ryc = float(self._cgrid_y[_i])
        dxt = float(self.center_lut_dx(best_s))
        dyt = float(self.center_lut_dy(best_s))
        nrm = math.sqrt(dxt * dxt + dyt * dyt) + 1e-9
        sin_t_obs = dyt / nrm
        cos_t_obs = dxt / nrm
        e_c_obs = sin_t_obs * (ox - rxc) - cos_t_obs * (oy - ryc)
        return best_s, e_c_obs

    def get_path_constraints_points(self, prev_soln):
        """Sample left/right boundary at each predicted stage's s.
        2026-06-10 unified layout: BOTH modes are 8-state → s = col 6
        (vx/vy/r 가 3/4/5 자리).
        """
        right_points = np.zeros((self.N, 2))
        left_points = np.zeros((self.N, 2))
        idx_s = 6
        # ── 0813 라이브 벽마진 존 (rqt: mzone_*_live) ─────────────────
        # track_loader 상수존은 기동 시 1회 굽지만, 여기는 매 사이클 벽을
        # 샘플링하므로 여기서 당기면 rqt 로 즉시 반영된다. A-B 캡슐 안의
        # "좌벽" 샘플점만 센터 방향으로 extra(램프) 만큼 이동 — 솔버 코리도
        # 전용이라 side-decide/뷰어의 원 LUT 는 그대로(보수 방향 불일치 없음).
        _mz_ex = float(getattr(self, 'mzone_extra_live', 0.0))
        _mz_on = _mz_ex > 1e-4
        if _mz_on:
            _mzA = np.array([float(getattr(self, 'mzone_ax_live', 0.0)),
                             float(getattr(self, 'mzone_ay_live', 0.0))])
            _mzB = np.array([float(getattr(self, 'mzone_bx_live', 0.0)),
                             float(getattr(self, 'mzone_by_live', 0.0))])
            _mzab = _mzB - _mzA
            _mzab2 = float(_mzab @ _mzab) + 1e-9
            _mzrf = float(getattr(self, 'mzone_rfull_live', 0.9))
            _mzrz = max(float(getattr(self, 'mzone_rzero_live', 1.8)), _mzrf + 1e-3)
        for k in range(1, self.N + 1):
            sk = float(prev_soln[k, idx_s]) % self.path_length
            right_points[k - 1, :] = np.array([self.right_lut_x(sk),
                                               self.right_lut_y(sk)],
                                              dtype=object).squeeze()
            lp = np.array([self.left_lut_x(sk),
                           self.left_lut_y(sk)], dtype=object).squeeze().astype(float)
            if _mz_on:
                _t = min(1.0, max(0.0, float((lp - _mzA) @ _mzab) / _mzab2))
                _dseg = float(np.hypot(*(lp - (_mzA + _t * _mzab))))
                _w = min(1.0, max(0.0, (_mzrz - _dseg) / (_mzrz - _mzrf)))
                if _w > 0.0:
                    _cp = np.array([float(self.center_lut_x(sk)),
                                    float(self.center_lut_y(sk))])
                    _dir = _cp - lp
                    _n = float(np.hypot(*_dir))
                    if _n > 1e-6:
                        lp = lp + (_mz_ex * _w / _n) * _dir
            left_points[k - 1, :] = lp
        return right_points, left_points

    def _kappa_at(self, s):
        """O(1) curvature lookup at arclength s (modulo path_length).
        Kept for diagnostics — not used to tighten constraints anymore."""
        if self.kappa_grid is None:
            return 0.0
        sw = float(s) % self.path_length
        idx = int(sw / self.kappa_ds)
        if idx < 0:
            idx = 0
        elif idx >= len(self.kappa_grid):
            idx = len(self.kappa_grid) - 1
        return abs(float(self.kappa_grid[idx]))

    def select_front_obstacle(self, curr_x, curr_y, curr_yaw,
                              obstacles, D_max=20.0, D_min_trig=10.0,
                              ang_max=2.0 * math.pi / 3.0):
        """Pick the closest obstacle in front of the car, or sentinel
        [1e6, 1e6, 1e6] when none qualify.

        D_max bumped 12→20 so curving tracks where obstacle is on next
        segment (Euclidean farther than path-distance) still trigger.
        ang_max bumped π/2→2π/3 (90°→120°) so obstacles slightly to the
        side on tight curves aren't filtered out before the car has
        rotated to face them.
        """
        if obstacles is None or len(obstacles) == 0:
            return [1e6, 1e6, 1e6]
        best = [1e6, 1e6, 1e6]
        cx, cy = curr_x + math.cos(curr_yaw) * 0.05, curr_y + math.sin(curr_yaw) * 0.05
        for ob in obstacles:
            if hasattr(ob, '__len__') and len(ob) >= 2:
                ox, oy = float(ob[0]), float(ob[1])
            else:
                continue
            dx, dy = ox - cx, oy - cy
            d = math.hypot(dx, dy)
            if d > D_max:
                continue
            # in front (with looser cone)?
            ang = math.atan2(dy, dx) - curr_yaw
            while ang > math.pi: ang -= 2 * math.pi
            while ang < -math.pi: ang += 2 * math.pi
            if abs(ang) > ang_max:
                continue
            if d < best[0]:
                best = [d, ox, oy]
        return best

    def decide_side_pref(self, s_obs, e_c_obs, obs_xy=None,
                         extra_up=0.0, extra_dn=0.0,
                         win_back=1.0, win_fwd=2.0, ds=0.25, margin=0.1):
        """Window-aware side decision (2026-06-11). Replaces the
        single-point top-2 boundary-distance compare whose centerline tie
        always returned -1 ("always avoids down" bug). Samples the labeled
        left/right corridor room along s ∈ [s_obs−win_back, s_obs+win_fwd]
        — the detour tube the car actually drives — and lets
        decide_side_window pick the side whose bottleneck (then mean room)
        is larger. Failure-driven flip (_side_history) stays as the rare
        fallback for what the map can't predict. One-shot per obstacle
        commit, so the ~13×6 LUT evals are negligible.
        """
        L = float(self.path_length)
        n = max(2, int(round((win_back + win_fwd) / ds)) + 1)
        w_l = np.empty(n)
        w_r = np.empty(n)
        for i in range(n):
            sk = (float(s_obs) - win_back + i * ds) % L
            cx = float(self.center_lut_x(sk)); cy = float(self.center_lut_y(sk))
            dxt = float(self.center_lut_dx(sk)); dyt = float(self.center_lut_dy(sk))
            nrm = math.hypot(dxt, dyt) + 1e-9
            sin_t, cos_t = dyt / nrm, dxt / nrm
            w_l[i] = (sin_t * (float(self.left_lut_x(sk)) - cx)
                      - cos_t * (float(self.left_lut_y(sk)) - cy))
            w_r[i] = (sin_t * (float(self.right_lut_x(sk)) - cx)
                      - cos_t * (float(self.right_lut_y(sk)) - cy))
        # 2026-07-21 side 판정 임계 = 실제 solver keepout(R_safe+R_car)에 맞춤.
        # (기존 기본 0.21은 실제 필요폭 0.60보다 작아, 못 지나갈 좁은 쪽을
        #  통과가능으로 오판 → 좁은 쪽으로 회피 시도 → 벽·장애물 사이 STALL.
        #  bag 16_22 s13: obs e_c=-0.50, 오른쪽 gap 0.46<0.60인데 0.21로 통과판정.)
        w_car_safe = float(self.R_safe_live) + float(self.R_car_live)
        # ── P1 (2026-07-22): 점유맵 실측 판정 — 첫 시도 정답 ──────────────
        # w-LUT 기반 room이 실벽과 불일치(bag 0721: "불가"쪽 통과성공/"가능"쪽
        # STUCK) → mpc_node가 주입한 map_raycast로 장애물 위치에서 solver e_c축
        # (∇e_c=(sin_t,−cos_t)) 양방향 실벽거리를 직접 측정. 부호=e_c축 그대로.
        _rc = getattr(self, 'map_raycast', None)
        if _rc is not None:
            sk0 = float(s_obs) % L
            dxt = float(self.center_lut_dx(sk0)); dyt = float(self.center_lut_dy(sk0))
            nrm = math.hypot(dxt, dyt) + 1e-9
            sin_t, cos_t = dyt / nrm, dxt / nrm
            # 0722c 원점 = raw 장애물 xy (LUT 재구성 오차 제거); 없으면 재구성 폴백
            if obs_xy is not None:
                obx, oby = float(obs_xy[0]), float(obs_xy[1])
            else:
                obx = float(self.center_lut_x(sk0)) + float(e_c_obs) * sin_t
                oby = float(self.center_lut_y(sk0)) - float(e_c_obs) * cos_t
            # 0722f 스왑 원복: bag 15_04 LUT계측으로 확정 — 솔버 벽밴드(wl/wr)와
            # 물리실측이 원래 방향(+e_c=(sin,−cos))에서 일치. 0722c 스왑은
            # stop-guard 감속을 "통과"로 오독한 데서 나온 오류였음.
            r_up = _rc(obx, oby, sin_t, -cos_t)    # solver +e_c 방향 실벽거리
            r_dn = _rc(obx, oby, -sin_t, cos_t)    # solver −e_c 방향
            if r_up is not None and r_dn is not None:
                # 0722b: 통과 필요 room = keepout(0.6, 장애물↔차중심) + R_car(벽↔차중심).
                # 0.6만 보면 "장애물은 피하고 벽에 박는" 쪽 허용 (bag 14_18 s0.1:
                # room 0.68 쪽 선택→0.70 벌리다 벽 STALL. 헤어핀 불안정의 실체).
                w_car_safe = w_car_safe + float(self.R_car_live)
                # 0722g 헤어핀 보너스: 곡률 안쪽 통과시 호 수축으로 유클리드 간극
                # ≈ e_c간극×(1−e/R) — R1.7·e0.9면 실제 0.47m (bag 15_10 접촉
                # 0.29-0.4 일치). 곡률 비례로 필요 room 상향 (κ0.6→+0.9).
                _ik = int((sk0 / self.kappa_ds)) % len(self.kappa_grid)
                _ko = abs(float(self.kappa_grid[_ik]))
                # 0723e κ보너스 ½ 재축소 (0.75κ→0.4κ, cap 0.45→0.25): 코너는 공간이
                # 제일 없는 곳인데 보너스가 keepout을 최대 +0.45 부풀려 좁은 코너
                # 통과 자체를 막음 (bag 13_21: s7.0 keepout 1.19 / s22.4 1.35 → 벽충돌).
                w_car_safe = w_car_safe + min(0.25, 0.4 * _ko)
                # 0722h: 30s 막힌쪽 제외를 여기(room 검사와 함께)서 — 이전엔 commit
                # 블록이 room 검사 없이 반대쪽 강제 → 재시도가 좁은쪽 돌진(더 깊숙히).
                _key = round(float(s_obs) % L * 2.0) / 2.0
                _stuck = set()
                _rsm = getattr(self, '_recent_stuck', {}).get(_key)
                # 0722l TTL 30→8s: 30s는 랩(~17s)을 넘겨 다음 랩 fresh까지 차단
                # (bag 15_52 t=41.7: room 2.13인데 side=0 정지루프). 8s=한 접근 스케일.
                if _rsm is not None and time.monotonic() - _rsm[1] < 8.0:
                    _stuck = _rsm[0]
                # P2 Stage-A: 이웃 병합 — 그 side 최외곽 멤버까지의 폭(extra)만큼
                # 요구 room 가산 (병합 장애물의 실질 반폭 확장)
                ok_up = (r_up - float(extra_up) >= w_car_safe) and (+1 not in _stuck)
                ok_dn = (r_dn - float(extra_dn) >= w_car_safe) and (-1 not in _stuck)
                # ── 2026-08-13 달성가능 keepout 기반 side 선택 (hall bag2 실측) ──
                # raw room 비교는 두 문제를 낳는다: ①센터 장애물(e_c≈0)에선 검출
                # 노이즈 ±5cm 가 side 를 결정(랩마다 반전), ②좁은 쪽이 낙점되면
                # 아래 _keep_eff 의 max(0.40,…) 플로어가 코리도 한계(_cor_reach)를
                # 넘어 "장애물 vs 코리도 soft 모순"이 부활 → 실통과 간격 0.26~0.29
                # (요구 0.30 미달 = 스침, hall_0813 bag2 7/15회). 선택 단계에서
                # 코리도 한계까지 반영한 달성가능 keepout 을 양쪽 계산해 큰 쪽을
                # 고른다 — 좁은 쪽은 아예 선택되지 않는다.
                _kb_o = min(0.15, 0.25 * _ko)
                # ── 0813 σ-비례 keepout: 트래커 불확실성(σ)만큼 넓게 피한다.
                # 방금 본 물체(σ 큼)=넓게, 여러 랩 본 물체(σ 수렴 3cm)=타이트.
                _sig_b = 0.0
                _hints = getattr(self, 'obs_sigma_hints', None)
                if _hints and obs_xy is not None:
                    for _hx, _hy, _hs in _hints:
                        if (_hx - obx) ** 2 + (_hy - oby) ** 2 < 0.36:
                            _sig_b = min(0.20, 2.0 * float(_hs))
                            break
                _full_keep = (float(self.R_safe_live) + _kb_o + _sig_b
                              + float(self.R_car_live))
                _wlx = float(self.left_lut_x(sk0)); _wly = float(self.left_lut_y(sk0))
                _wrx = float(self.right_lut_x(sk0)); _wry = float(self.right_lut_y(sk0))
                _cx = float(self.center_lut_x(sk0)); _cy = float(self.center_lut_y(sk0))
                _wl = sin_t * (_wlx - _cx) - cos_t * (_wly - _cy)
                _wr = sin_t * (_wrx - _cx) - cos_t * (_wry - _cy)
                _hi = max(_wl, _wr); _lo = min(_wl, _wr)
                _cr_up = (_hi - float(self.R_car_live)) - float(e_c_obs)
                _cr_dn = float(e_c_obs) - (_lo + float(self.R_car_live))
                _ach_up = min(_full_keep, (r_up - float(extra_up)) - 0.32, _cr_up - 0.12)
                _ach_dn = min(_full_keep, (r_dn - float(extra_dn)) - 0.32, _cr_dn - 0.12)
                # ── 0813 D6 side 기억: 같은 물체(0.6m)는 이전에 고른 side 승계.
                # 검출이 부호를 넘나들어도 랩마다 방향이 안 바뀐다 (동전던지기
                # 소멸). 그 side 가 여전히 ok/ach 양수일 때만 — 막히면 재결정.
                if not hasattr(self, '_side_mem'):
                    self._side_mem = {}
                _smk = (round(obx * 2.0) / 2.0, round(oby * 2.0) / 2.0)
                _sm = self._side_mem.get(_smk)
                _now_sm = time.monotonic()
                if _sm is not None and _now_sm - _sm[1] > 20.0:
                    _sm = None
                if ok_up and ok_dn:
                    if _sm is not None and (
                            (_sm[0] > 0 and _ach_up > 0.0)
                            or (_sm[0] < 0 and _ach_dn > 0.0)):
                        _side = _sm[0]
                    else:
                        _side = +1 if _ach_up >= _ach_dn else -1
                elif ok_up:
                    _side = +1
                elif ok_dn:
                    _side = -1
                else:
                    # 0723c narrow-pass: 풀 요구(0.9+κ)는 미달이어도 물리 최소
                    # (NARROW_ROOM=0.60 = 장애물↔차중심 0.38 + 차↔벽 0.22)만
                    # 되면 넓은 쪽으로 서행 통과 — "항상 성공" (bag 13_01: 직선
                    # 3번째 room 0.7-0.8이 요구 0.9 미달로 매랩 BLOCK-STOP되던 것).
                    # 0723d 문턱 0.60→0.70: 실측 최소폭 = 장애물쪽 0.31(반폭 합)
                    # + 벽쪽 코너 0.33 ≈ 0.68 (bag 13_10 산수. gym 직사각형 판정
                    # 픽스와 세트 — 그 전 원형 0.44 기준으론 0.77도 불가였음).
                    # 0822: 하드코딩 0.70 → narrow_room_live. 0.70 은 주 판정
                    # w_car_safe 와 값이 같아 예외가 죽어 있었다 (위 주석 참조).
                    _nr = float(getattr(self, 'narrow_room_live', 0.55))
                    nk_up = (r_up - float(extra_up) >= _nr) and (+1 not in _stuck)
                    nk_dn = (r_dn - float(extra_dn) >= _nr) and (-1 not in _stuck)
                    if nk_up and nk_dn:
                        _side = +1 if (r_up - extra_up) >= (r_dn - extra_dn) else -1
                    elif nk_up:
                        _side = +1
                    elif nk_dn:
                        _side = -1
                    else:
                        _side = 0    # 물리적으로도 불가 → BLOCK-STOP
                    if _side != 0:
                        # 좁은 틈 통과 확정 — 추종오차가 그대로 접촉이 되는
                        # 구간이라 속도를 낮춘다 (mpc_node 가 이 시각을 본다).
                        self._narrow_pass_until = (
                            time.monotonic()
                            + float(getattr(self, 'narrow_pass_hold', 1.5)))
                        self._log.info_throttle(
                            1.0,
                            "[MPC] 좁은틈 통과 side=%+d (r_up=%.2f r_dn=%.2f "
                            "문턱=%.2f, 주판정 %.2f 미달)",
                            _side, r_up - float(extra_up), r_dn - float(extra_dn),
                            _nr, w_car_safe)
                # 0723c keepout 공간맞춤(shift): 선택 side의 실측 room에 keepout이
                # 안 들어가면 그 차이만큼 solver 경계를 안쪽으로 — h_obs vs corridor
                # soft 모순(횡마비→정면 STUCK, bag 13_01 s14.5 3/6랩)의 원천 제거.
                # 적용은 e_c_obs를 pass쪽으로 shift (p만 바뀜, codegen 불필요).
                # LUT 벽 e_c (shift clamp + 계측 로그 겸용 — 0723e에서 앞으로 이동)
                _wlx = float(self.left_lut_x(sk0)); _wly = float(self.left_lut_y(sk0))
                _wrx = float(self.right_lut_x(sk0)); _wry = float(self.right_lut_y(sk0))
                _cx = float(self.center_lut_x(sk0)); _cy = float(self.center_lut_y(sk0))
                _wl = sin_t * (_wlx - _cx) - cos_t * (_wly - _cy)
                _wr = sin_t * (_wrx - _cx) - cos_t * (_wry - _cy)
                if _side != 0:
                    _room_s = (r_up - float(extra_up)) if _side > 0 else (r_dn - float(extra_dn))
                    _kb_o = min(0.15, 0.25 * _ko)   # 0723e 동조
                    _full_keep = float(self.R_safe_live) + _kb_o + float(self.R_car_live)
                    # 0723d 벽마진 0.32 (차 코너 도달거리).
                    # 0723e LUT 코리도 한계 추가: 요구 경계가 solver LUT 벽을 넘으면
                    # obstacle slack(2000)이 corridor slack(80)을 눌러 계획이 벽 관통
                    # (bag 13_21: s7.0 경계 +0.87 = LUT wr +0.86 → WALL 충돌,
                    # s22.4 경계 -0.60 < wl -0.48 → WALL 충돌). 경계를 코리도
                    # (벽 − R_car) 안으로 강제.
                    _hi = max(_wl, _wr); _lo = min(_wl, _wr)
                    if _side > 0:
                        _cor_reach = (_hi - float(self.R_car_live)) - float(e_c_obs)
                    else:
                        _cor_reach = float(e_c_obs) - (_lo + float(self.R_car_live))
                    # 2026-08-07 통과밴드 0.12 (bag 16_40_59, s35.1 STUCK×2):
                    # _cor_reach 에 경계를 딱 맞추면 "요구 경계 == 코리도 한계"
                    # 가 되어 차 중심의 허용 밴드가 정확히 0(한 점) — 6.8 m/s
                    # 접근 후 횡이동을 못 하고 0.34 m 정면 낑김. 실측: e_obs
                    # +0.17, 벽 −0.74 → keepout 0.55 요구 e≤−0.39 vs 코리도
                    # 한계 −0.39. 경계를 한계보다 0.12 안쪽으로 당겨 유한 폭을
                    # 보장한다 (넓은 곳은 _cor_reach 가 안 물려 무영향).
                    # 0813 D3 수정: 플로어 0.40 이 _cor_reach 캡을 되엎으면
                    # "요구 경계 > 코리도 한계" 모순이 부활해 h_obs(2000) vs
                    # 코리도 슬랙 힘겨루기로 계획이 압축된다 (kftrk1 심 실측:
                    # 센터 장애물 2곳에서 13/15 STUCK, 실차 hall bag2 0.26~0.29
                    # 스침). 플로어 자체를 코리도 도달가능치로 캡 — 좁은 곳은
                    # 얇게라도 "모순 없이" 통과하고, 얇은 만큼은 감속(통과속도
                    # d_keep 티어)이 보상한다.
                    _keep_eff = max(min(0.40, _cor_reach - 0.12),
                                    min(_full_keep, _room_s - 0.32,
                                        _cor_reach - 0.12))
                    _shift = _full_keep - _keep_eff
                    # (0723r tight-pass 속도 티어는 0723s에서 원복 — 흐름 우선)
                    _nd = getattr(self, '_narrow_shift', None)
                    if _nd is None:
                        _nd = {}
                        self._narrow_shift = _nd
                    _nkey = (round(obx * 2.0) / 2.0, round(oby * 2.0) / 2.0)
                    if _shift > 1e-3:
                        # 0818b: keep_eff 도 저장 — 하류 감속 티어가 '얼마나 남았나'로 판단
                        _nd[_nkey] = (_shift, time.monotonic(), float(_keep_eff))
                    else:
                        _nd.pop(_nkey, None)
                    if len(_nd) > 32:
                        _nnow = time.monotonic()
                        for _k in [k for k, v in _nd.items() if _nnow - v[1] > 10.0]:
                            _nd.pop(_k, None)
                # (0722e 계측 _wl/_wr는 0723e에서 shift clamp용으로 위로 이동)
                _shift_log = 0.0
                if _side != 0:
                    _shift_log = float(_shift)
                    # 0813 D6: side 기억 갱신 (20s TTL, 사용 시 연장)
                    self._side_mem[_smk] = (_side, _now_sm)
                    if len(self._side_mem) > 64:
                        self._side_mem = {
                            k: v for k, v in self._side_mem.items()
                            if _now_sm - v[1] < 20.0}
                self._log.info(
                    "[MPC] side-decide(MAP) s_obs=%.1f e_c=%+.2f room_up=%.2f "
                    "room_dn=%.2f keepout=%.2f -> side=%+d shift=%.2f | LUT벽 e_c: wl=%+.2f wr=%+.2f"
                    " | ach u/d=%.2f/%.2f",
                    s_obs, e_c_obs, r_up, r_dn, w_car_safe, _side, _shift_log, _wl, _wr,
                    _ach_up, _ach_dn)
                return _side
        # 2026-07-21e 라벨 완전 배제 — e_c 축 직접 판정 (첫 시도 정답이 목표).
        # 좌/우 "라벨"(w_l/w_r)은 트랙 방향·경계추출 이음새에 따라 뒤집혀 어떤
        # 라벨→부호 매핑도 전 구간 일관 불가 (bag 20_19/20_25: 같은 보정이 s51.5는
        # 3/3 성공, s18.3은 반대). 대신 샘플별 상한 hi=max(wl,wr)/하한 lo=min(wl,wr)
        # 로 코리도 span을 라벨 무관하게 재구성 → +e_c쪽 여유 vs −e_c쪽 여유 비교.
        # 반환 부호 = e_c 방향 그 자체 (solver h_obs와 동일 축) — 매핑 단계 없음.
        e = float(e_c_obs)
        hi = [max(float(a), float(b)) for a, b in zip(w_l, w_r)]
        lo = [min(float(a), float(b)) for a, b in zip(w_l, w_r)]
        room_up = min(hi) - e          # +e_c 방향 병목 여유
        room_dn = e - max(lo)          # −e_c 방향 병목 여유
        if room_up < w_car_safe and room_dn < w_car_safe:
            _side = 0                  # 양측 불가 → BLOCK-STOP
        elif room_up < w_car_safe:
            _side = -1
        elif room_dn < w_car_safe:
            _side = +1
        else:
            _side = +1 if room_up >= room_dn else -1
        self._log.info(
            "[MPC] side-decide s_obs=%.1f e_c=%+.2f room_up=%.2f room_dn=%.2f "
            "keepout=%.2f -> side=%+d",
            s_obs, e_c_obs, room_up, room_dn, w_car_safe, _side)
        return _side

    def _narrow_shifted(self, ox, oy, side, e_obs):
        """0723c: narrow-pass keepout shift 적용 — decide가 저장한 shift만큼
        e_c_obs를 pass쪽으로 옮겨 solver 요구 경계를 실측 room 안으로 клamp.
        해당 장애물에 shift가 없거나(공간 충분) 오래됐으면(10s) 원값 그대로."""
        _nd = getattr(self, '_narrow_shift', None)
        if not _nd or side == 0:
            return e_obs
        _it = _nd.get((round(float(ox) * 2.0) / 2.0, round(float(oy) * 2.0) / 2.0))
        if _it is None or time.monotonic() - _it[1] > 10.0:
            return e_obs
        return float(e_obs) - float(side) * _it[0]

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, initial_state, obstacles, pred_obs=None):
        """initial_state: numpy length 7 from ROS node = [x, y, psi, vx, vy, r, s].
        We extend internally to 8: append delta_prev = the steer command we
        sent last cycle (initialised to 0). 2026-06-10 unified layout: SAME
        in both kinematic and dynamic modes.
        pred_obs: optional (N+1, 2) per-stage predicted obstacle (x, y). None →
        constant current position for every stage (legacy behaviour).
        """
        L = float(self.path_length)

        # Augment state with the previous applied δ for steer-rate cost.
        # Node sends 7 → expand to 8 (append δ_prev). target = self.n_states.
        if len(initial_state) == self.n_states - 1:
            last_delta = float(getattr(self, '_last_delta_applied', 0.0))
            initial_state = np.append(np.asarray(initial_state, dtype=float),
                                       last_delta)

        # State-vector index helpers — unified 8-state layout in BOTH modes:
        # [x, y, ψ, vx, vy, r, s, δ_prev].
        IDX_S = 6
        IDX_DELTA_PREV = 7

        # ---- Unwrap sensor s into solver-internal monotonic coordinate ----
        # Node passes current_s ∈ [0, L). We track lap_count internally and
        # produce solver_s = sensor_s + lap_count·L which grows unboundedly
        # across laps. Dynamics ṡ = p_v then stays smooth — no wrap
        # discontinuity ever reaches the QP, so no lap-rollover MINSTEP and
        # no cost spike at lap boundaries.
        sensor_s = float(initial_state[IDX_S])
        if self._last_sensor_s is not None:
            d = sensor_s - self._last_sensor_s
            if d < -L / 2.0:
                self.lap_count += 1   # forward wrap (typical: 86 → 0)
            elif d > L / 2.0:
                self.lap_count -= 1   # reverse (rare: backward sensor jump)
        self._last_sensor_s = sensor_s
        initial_state[IDX_S] = sensor_s + self.lap_count * L
        s0 = float(initial_state[IDX_S])

        # ---- ψ unwrap via persistent offset (same trick as lap_count for s) ----
        # Sensor yaw ∈ [−π, π]; we accumulate ±2π each time the sensor
        # wraps and pass `sensor_yaw + yaw_offset` to the solver. The
        # solver's internal ψ then grows monotonically — no wrap event
        # ever reaches the QP, no multi-iter SQP fixup needed, no
        # repeated wrap-detection at every cycle in the wrap zone (which
        # was the underlying cause of the 974-cost spike at s≈56 every
        # lap). Cost (yaw_err via sin/cos identities) and dynamics (cos/
        # sin of ψ) are 2π-periodic, so unbounded ψ is harmless.
        sensor_yaw = float(initial_state[2])
        if self._last_sensor_yaw is not None:
            d_yaw = sensor_yaw - self._last_sensor_yaw
            if d_yaw < -math.pi:
                self._yaw_offset += 2.0 * math.pi
            elif d_yaw > math.pi:
                self._yaw_offset -= 2.0 * math.pi
        self._last_sensor_yaw = sensor_yaw
        initial_state[2] = sensor_yaw + self._yaw_offset
        psi_unwrapped = False  # legacy flag — nothing to do, kept for the
                               # multi-iter trigger below (always False now)

        # First-cycle warm start — dynamics-feasible forward rollout from
        # the actual car state. Previously we placed X0[1..N] on the
        # centerline regardless of where the car was, so the X0[0]→X0[1]
        # jump violated dynamics by up to ~0.5 m laterally. acados then
        # could not make any progress and reported ACADOS_MINSTEP at QP
        # iter 1. Rolling forward with the same controls used in u0 keeps
        # X0[k+1] = X0[k] + dt·f(X0[k], u_k), satisfying dynamics by
        # construction.
        if not self.WARM_START:
            # Cold-start path fires after stuck/spike fallback or obstacle
            # commit/release. solver.reset() here is critical for the
            # stuck case: stuck override sets u_seq=zeros, the solver's
            # dual variables then carry a bias "u=0 minimizes the QP",
            # and the next cycle's 1-iter SQP_RTI can't escape — the car
            # stays at v=0 forever even though cold-rollout pushes a
            # reasonable u0=seed_v warm-start. Reset zeros the duals so
            # the rollout actually drives the solution. Note: ψ-wrap
            # doesn't go through this path (it sets X0 directly without
            # WARM_START=False), so the Plan-B "no reset on wrap" stays.
            try:
                self.solver.reset()
            except Exception:
                pass
            try:
                seed_v = max(float(self.ref_v(s0 % L)) * 0.5, 1.0)
            except Exception:
                seed_v = max(self.v_max * 0.4, 1.0)
            self.u0 = np.zeros((self.N, self.n_controls))
            # Warm-start (unified 8-state, both modes): u = [a_x, δ, p_v].
            # Hold a_x = 0 (no acceleration during warm-start) so vx stays
            # at the car's current velocity. No ramp = no input bound risk.
            # Solver's first iter then finds the correct a_x. p_v set to
            # seed_v so s grows during warm-start. Skip slip dynamics —
            # vy, r held constant so X0 stays well-conditioned.
            self.u0[:, 0] = 0.0      # a_x (no acceleration in seed)
            self.u0[:, 1] = 0.0      # delta (overwritten per-stage below, curvature-aware)
            self.u0[:, 2] = seed_v   # p_v (input, can be set free)
            self.X0[0, :self.n_states] = initial_state
            # δ=0 (straight-line) seed is fine off a straight but on a corner
            # it sends the rollout well outside the corridor before SQP-RTI's
            # single iteration/cycle can pull it back in (measured: κ=0.5 →
            # 92 cm off at 2 m ahead, 206 cm off at 3.2 m ahead, vs a track
            # half-width of only 0.5-1.1 m) → QP infeasible → MINSTEP → first
            # command corrupt → wall. This path also fires after STUCK
            # recovery and obstacle commit/release mid-corner (see block
            # comment above), so it's not just a start-line issue.
            # Fix: seed steer from the raceline curvature at each rolled-out
            # s, δ_k = atan(L_wb·κ(s_k)) — the steer the curvature actually
            # needs, not 0. self.kappa_grid is SIGNED (κ=dθ/ds, θ=unwrap(
            # atan2(dy,dx)) — see its construction ~line 484); the cost's own
            # curvature feed-forward term (line ~1149,
            # atan(self.L·kappa_at_s·sign(_signed_k))) reconstructs this same
            # signed quantity from abs_kappa_lut × sign(signed_kappa_lut), so
            # atan(self.L·κ) directly (no extra sign flip) is the confirmed
            # convention. self.L here is wheelbase (0.307 m, set at __init__
            # line 192/428) — NOT the local `L` (=self.path_length, track
            # length) used a few lines above for `s0 % L`; the two must not
            # be confused.
            _n_kg = len(self.kappa_grid)
            _steer_lim = 0.4  # rad — no self.max_steer/delta_max attr in this class
            for k in range(self.N):
                xk = self.X0[k, :]
                uk = self.u0[k, :]
                _sk = float(xk[IDX_S]) % L
                _idx = int(_sk / self.kappa_ds) % _n_kg
                _kappa_k = float(self.kappa_grid[_idx])
                delta_ = float(np.clip(np.arctan(self.L * _kappa_k), -_steer_lim, _steer_lim))
                uk[1] = delta_
                p_ = uk[2]
                psi_ = xk[2]
                vx_  = xk[3]            # vx stays constant in seed
                delta_prev_ = xk[7]
                dx_dt   = vx_ * np.cos(psi_)
                dy_dt   = vx_ * np.sin(psi_)
                dpsi_dt = (vx_ / self.L) * np.tan(delta_)
                ds_dt   = p_
                ddprev  = (delta_ - delta_prev_) / self.dT
                self.X0[k + 1, :self.n_states] = xk[:self.n_states] + self.dT * np.array(
                    [dx_dt, dy_dt, dpsi_dt, 0.0, 0.0, 0.0,
                     ds_dt, ddprev])
                self.X0[k + 1, self.n_states:] = self.X0[k, self.n_states:]  # α constant over horizon
            for k in range(self.N + 1):
                self.solver.set(k, "x", self.X0[k, :])
            for k in range(self.N):
                self.solver.set(k, "u", self.u0[k, :])
            self.WARM_START = True
            # Flag for multi-iter SQP — solver.reset() above zeroed dual
            # variables; without this flag the main solve below runs only
            # 1 SQP_RTI iteration which can't rebuild the dual active set
            # → returns near-zero (or otherwise corrupt) controls →
            # stuck-at-zero loop after every stuck recovery / fallback.
            self._just_cold_started = True

        # (No s-wrap fixup needed — solver-internal s is monotonic.)

        # ---- Per-cycle parameters ----
        self._obs_blocked_d = None   # BLOCK-STOP: 매 사이클 리셋 (커밋 성공/장애물 소실시 자동 해제)
        sel = self.select_front_obstacle(initial_state[0], initial_state[1],
                                         initial_state[2], obstacles)
        self.dbg_n_obs_input = len(obstacles) if obstacles is not None else 0
        self.dbg_sel_dmin, self.dbg_sel_x, self.dbg_sel_y = sel

        # ---- Commit-once obstacle / side decision ----
        # Once we engage an obstacle, FREEZE the chosen avoidance side and
        # the obstacle Frenet coords until the car has driven past it
        # longitudinally. Without this, decide_side_pref / select_front_obs
        # may flip mid-pass (e.g., as the car turns into the obstacle the
        # corridor sample distances change), and the half-plane direction
        # would suddenly reverse → prediction "bursts" the other way.
        # Stored: (ox, oy, s_obs, side_pref, e_c_obs).
        sel_is_real = sel[1] < 1e5
        if not hasattr(self, '_committed_obs'):
            self._committed_obs = None
        # 커밋 해제 페이드 배수 (1.0 = 통상, 0.0 = 해제 직전). 아래에서 갱신.
        self._release_fade = 1.0

        # Release current commitment if (a) car drove past it in s, (b) it
        # is no longer in the obstacle list, or (c) we drifted very far.
        if self._committed_obs is not None:
            cox, coy, cs_obs, cside, ce_c = self._committed_obs
            s_car = float(initial_state[IDX_S]) % L
            delta_s = (s_car - cs_obs) % L
            car_to_obs = math.hypot(initial_state[0] - cox,
                                     initial_state[1] - coy)
            # ── 2026-07-31 F3: 커밋이 검출 지터를 따라가게 (실차 필수) ──
            # 이전에는 "커밋 좌표 0.3 m 이내 검출 존재"로 유지를 판정했는데,
            # 실차 라이다 검출은 같은 물체가 프레임마다 0.1~0.3 m 흔들린다
            # (bag 14_31: 물체 1개가 0.1 m 격자 수십 셀에 퍼짐). 게이트를
            # 벗어나는 순간 "committed obs removed" 로 즉시 해제되어, 실차
            # 3개 백 연속 side_pref 0% — 회피가 발동 자체를 못 했다.
            #   ① 연관: 반경 COMMIT_ASSOC_R 안 최근접 검출 (반경만 넓히면
            #      옆 물체를 오인하므로 EMA 추종과 세트)
            #   ② 추종: 커밋 좌표를 EMA 로 검출 쪽으로 끌어당기고 s_obs/제약선도
            #      같이 이동 (제약이 실제 물체를 따라감). side 는 절대 재계산
            #      안 함 — 회피 방향 일관성의 기초.
            #   ③ 해제: 연속 COMMIT_MISS_S 초 미검출일 때만 (검출률 62% 백도
            #      있었음 — 한 프레임 빠짐으로 끊으면 안 됨)
            _assoc = None
            if obstacles is not None and len(obstacles) > 0:
                # 0813 속도비례 종방향 게이트: 고속에서 정적 장애물 검출이 진행방향으로
                # v·지연 만큼 밀려 보인다(bag 16_52 v6.7: 0.7s 에 s +3m). 등방 0.6 이면
                # 새 물체로 오인→커밋 플래핑. 횡은 0.6 유지(옆 물체 보호), 종만 확장.
                ASSOC_V_GAIN = 0.20          # [s] 6.7m/s → 종 반경 0.6+1.34 ≈ 1.9m
                _v_now = abs(float(initial_state[3]))
                _r_lat = self.COMMIT_ASSOC_R
                _r_lon = self.COMMIT_ASSOC_R + ASSOC_V_GAIN * _v_now
                _psi = float(initial_state[2])
                _cpsi, _spsi = math.cos(_psi), math.sin(_psi)
                _bd = None
                for op in obstacles:
                    _dx, _dy = op[0] - cox, op[1] - coy
                    _lon = _dx * _cpsi + _dy * _spsi      # 차 진행방향 성분
                    _lat = -_dx * _spsi + _dy * _cpsi
                    _n2 = (_lon / _r_lon) ** 2 + (_lat / _r_lat) ** 2   # 타원 정규화 거리²
                    if _n2 < 1.0 and (_bd is None or _n2 < _bd):
                        _bd = _n2
                        _assoc = (float(op[0]), float(op[1]))
            _mono = monotonic_now()
            if _assoc is not None:
                _a = self.COMMIT_FOLLOW_ALPHA
                _nx = cox + _a * (_assoc[0] - cox)
                _ny = coy + _a * (_assoc[1] - coy)
                # 제약선(e_eff)은 클러스터 병합값이라 절대치 재계산 대신
                # 기준 e_c 의 변화량만큼 평행이동한다.
                _s_old, _e_old = self._obstacle_frenet(cox, coy)
                _s_new, _e_new = self._obstacle_frenet(_nx, _ny)
                cox, coy = _nx, _ny
                cs_obs = _s_new
                ce_c = ce_c + (_e_new - _e_old)
                self._committed_obs = (cox, coy, cs_obs, cside, ce_c)
                self._commit_seen_t = _mono
            # ── 0821 커밋 중 병합 갱신 (연속 2개 회피) ────────────────────
            # 커밋 시점 병합(2389행)은 그 순간 보이는 이웃만 반영한다. #2 가 #1
            # 뒤에 가려져 있으면 누락되고, 회피 중 튀어나와 재커밋→접촉 (0722j
            # 주석의 미해결 케이스). 커밋을 유지한 채(side 고정=플래핑 없음)
            # 제약선만 최외곽으로 확장한다. 확장 전용(축소 없음)이라 이미 확보한
            # 회피 폭은 줄지 않고, 캡으로 무한 확장(→infeasible)을 막는다.
            if obstacles is not None and cside != 0:
                _L_m = float(self.path_length)
                _e_ext = ce_c
                for _ob in obstacles:
                    if not (hasattr(_ob, '__len__') and len(_ob) >= 2):
                        continue
                    _ox3, _oy3 = float(_ob[0]), float(_ob[1])
                    if (_ox3 - cox) ** 2 + (_oy3 - coy) ** 2 < 0.01:
                        continue                      # 커밋 물체 자신
                    _sj3, _ej3 = self._obstacle_frenet(_ox3, _oy3)
                    _dsj3 = (_sj3 - cs_obs + 0.5 * _L_m) % _L_m - 0.5 * _L_m
                    if -1.0 <= _dsj3 <= 2.5:          # 커밋 창(+2.0)보다 살짝 넓게
                        _e_ext = (max(_e_ext, _ej3) if cside > 0
                                  else min(_e_ext, _ej3))
                _MERGE_CAP = 0.60                     # [m] 최대 확장 (트랙폭 보호)
                _e_ext = (min(_e_ext, ce_c + _MERGE_CAP) if cside > 0
                          else max(_e_ext, ce_c - _MERGE_CAP))
                if abs(_e_ext - ce_c) > 1e-3:
                    self._log.warn_throttle(
                        1.0, "[MPC] 커밋 중 병합 확장 e_eff %+.2f→%+.2f (side %+d)",
                        ce_c, _e_ext, cside)
                    ce_c = _e_ext
                    self._committed_obs = (cox, coy, cs_obs, cside, ce_c)
            obs_still_present = (
                _assoc is not None
                or (getattr(self, '_commit_seen_t', None) is not None
                    and _mono - self._commit_seen_t < self.COMMIT_MISS_S))
            # (0722u 조기 release 0.3m 실험은 원복 — #1 옆구리에서 보호가 걷혀
            #  즉시 #2에 박고, 멀리 있는 다음 단일까지 obs2로 잡혀 단일도 회귀.
            #  bag 17_24: t=6.2 release Δs0.36 → 1.2s 뒤 STUCK 실증.)
            # 0723o release 콜드리셋 제거: 제약 해제는 relaxation이라 warm 해가
            # 그대로 유효 — 리셋하면 통과 직후마다 계획 점프 (장애물 6개 = 랩당
            # 6회 킥). bag 14_28 조향 출렁임(1.4-2.5Hz, 장애물 스케일)의 주요원.
            # 커밋 측 콜드리셋(0722v, 로컬미니멈 탈출)은 유지.
            # 2026-07-31 계단 해제 → 페이드. Δs>1.5 에서 한 사이클에
            # side_pref ±1 → 0 으로 끊으면 h_obs 제약이 즉시 사라져 계획이 라인으로
            # 급복귀하고, 그 관성이 반대쪽 오버슛으로 나온다 (bag 17_35: d +0.63 →
            # -0.22, 0.85 m 왕복 = 눈에 보이는 "헤딩 흔들림"의 실체. 조향 자체는
            # 오히려 장애물 구간이 더 작다 — |δ| 0.05 vs 장애물 없는 구간 0.17).
            # side_pref 는 p_sym[3] 으로 연속값을 받고 h_obs 가 |side_pref| 게이트로
            # 부드럽게 꺼지도록 이미 설계돼 있다 — 계단으로 쓰던 것을 바로잡는다.
            # 0723o 가 콜드리셋을 없앤 것과 같은 취지(통과 직후 계획 점프 제거).
            self._release_fade = 1.0
            if 1.5 < delta_s < 0.5 * L:
                if delta_s >= self.RELEASE_FADE_END:
                    self._log.info("[MPC] passed committed obs (Δs=%.2f m) — release", delta_s)
                    self._committed_obs = None
                else:
                    self._release_fade = float(
                        (self.RELEASE_FADE_END - delta_s)
                        / (self.RELEASE_FADE_END - 1.5))
            elif not obs_still_present:
                self._log.info("[MPC] committed obs removed — release")
                self._committed_obs = None
            elif car_to_obs > 12.0:
                # Drifted way past; safety release.
                self._committed_obs = None

        # Track NEW commit this cycle so we can:
        #   (a) push a stronger detour prior into X0 (option 2)
        #   (b) run extra SQP iters before the main solve (option 3)
        just_committed = False

        # Side-decision cache (hysteresis). Maps obstacle position →
        # last committed side. Re-using the same side for the same
        # obstacle prevents the "sometimes goes up, sometimes down" flip
        # symptom — once a side is chosen for an obstacle, stick with it
        # for all subsequent passes.
        if not hasattr(self, '_side_history'):
            self._side_history = {}

        # Commit on first engagement. Trigger distance is rqt-tunable
        # via self.commit_dist_live (default 10 m). Larger = engage
        # detour earlier (smoother but more conservative); smaller =
        # late commit (snappy but riskier).
        # 2026-07-21 속도비례 commit: 고정 6m는 7m/s에서 반응시간 0.86s뿐 —
        # 감속+횡이동 물리적으로 부족 → keepout 안(0.43m)까지 파고들어 충돌
        # (bag 19_48: 실패 5건 전부 진입 v4.8-7.0, side는 정상 ±1, BLOCK-STOP 0).
        # commit_eff = max(commit_dist, v·t_react). t_react=1.5s: 7m/s→10.5m,
        # 저속(<4m/s)은 기존 commit_dist 그대로 (민첩성 보존).
        _v_now = abs(float(initial_state[3]))
        _commit_eff = max(float(self.commit_dist_live),
                          _v_now * float(getattr(self, 'commit_treact_live', 1.5)))
        # 0723t P3 오버테이크 게이트: node가 "추종 모드"로 판단하면(동적 상대 +
        # 속도우위 부족) 신규 커밋 억제 — 진행 중이던 커밋은 자연 완료(히스테리시스).
        if (self._committed_obs is None and sel_is_real
                and float(sel[0]) < _commit_eff
                and not getattr(self, 'suppress_commit', False)):
            s_obs_new, e_c_obs_new = self._obstacle_frenet(sel[1], sel[2])
            s_car_now = float(initial_state[IDX_S]) % L
            delta_s_now = (s_car_now - s_obs_new) % L
            # In-front zone (see prior comment block).
            obs_is_ahead = (delta_s_now < 1.5) or (delta_s_now > 0.5 * L)
            if obs_is_ahead:
                # 0722d 단순화: 캐시/flip학습/blocked 3종 제거 — fresh(MAP 레이캐스트)가
                # 이제 신뢰 지오메트리라 매번 fresh 판정. (옛 캐시가 정답 fresh를
                # 좁은쪽으로 덮던 것이 bag 14_46 "좁은틈 진입"의 실체: t=0.0 fresh -1
                # 직후 t=0.1 stale cache +1 override.) 남긴 것 하나: 이번 접근에서
                # 방금 막힌 side 30초 기억 → 같은 쪽 재시도 루프만 방지.
                key = round(s_obs_new * 2.0) / 2.0
                # ── P2 Stage-A (0722i): 이웃 장애물을 side별 추가 요구폭으로 반영 ──
                # detour 창(Δs −1~+5m) 안 다른 장애물의 e_c가 sel보다 위/아래로
                # 벗어난 만큼(extra) 그 side 필요 room이 늘고, 커밋 제약선(e_c_obs)도
                # 최외곽 멤버로 이동 = 클러스터·1.3m 슬라럼·회피경로상 장애물 통합.
                # 장애물 1개면 extra=0, e_eff=e_obs → P1과 바이트 동일 (불침범).
                e_hi = e_lo = float(e_c_obs_new)
                if obstacles is not None:
                    for _ob in obstacles:
                        if not (hasattr(_ob, '__len__') and len(_ob) >= 2):
                            continue
                        _ox2, _oy2 = float(_ob[0]), float(_ob[1])
                        if (_ox2 - sel[1]) ** 2 + (_oy2 - sel[2]) ** 2 < 0.01:
                            continue          # sel 자신
                        _sj, _ej = self._obstacle_frenet(_ox2, _oy2)
                        _dsj = (_sj - s_obs_new + 0.5 * L) % L - 0.5 * L
                        # 0722o 창 +5→+2m: 분리형(2~9m)은 이제 obs2 슬롯이 별도
                        # 제약으로 담당 — 병합과 이중제약(반대 side 충돌) 방지.
                        if -1.0 <= _dsj <= 2.0:
                            e_hi = max(e_hi, float(_ej))
                            e_lo = min(e_lo, float(_ej))
                extra_up = e_hi - float(e_c_obs_new)
                extra_dn = float(e_c_obs_new) - e_lo
                sp_new = self.decide_side_pref(s_obs_new, e_c_obs_new,
                                               obs_xy=(sel[1], sel[2]),
                                               extra_up=extra_up,
                                               extra_dn=extra_dn)
                # (0722h: 막힌쪽 제외는 decide 내부에서 room 검사와 함께 처리 —
                #  여기 있던 room-무검사 반대쪽 강제가 "재시도 때 더 깊숙히"의 범인)
                if sp_new != 0:
                    # 커밋 제약선 = 선택 side의 최외곽 멤버 (병합 half-plane)
                    _e_eff = e_hi if sp_new > 0 else e_lo
                    self._committed_obs = (
                        float(sel[1]), float(sel[2]),
                        s_obs_new, int(sp_new), float(_e_eff))
                    self._commit_seen_t = monotonic_now()   # F3 추종 기준시각
                    self._log.info(
                        "[MPC] committed obs=(%.2f,%.2f) s_obs=%.2f e_c_obs=%+.2f side=%+d"
                        " (병합 e_eff=%+.2f, extra u/d=%.2f/%.2f)",
                        sel[1], sel[2], s_obs_new, e_c_obs_new, sp_new,
                        _e_eff, extra_up, extra_dn)
                    # 0722v 콜드리셋 복구: 0722s(웜 유지)가 오판 — RTI가 직선 이전해에
                    # 들러붙어 커밋 후 detour가 아예 안 생김 (bag 17_28: s31 등 6건
                    # 전부 횡이동 0으로 정면 접촉, 이전 통과 지점 포함). 콜드리셋+X0
                    # 프라이어 = 계획을 옆으로 점프시키는 의도된 킥 (원저자 설계).
                    self.WARM_START = False  # rebuild with frozen side
                    just_committed = True
                else:
                    # 2026-07-16 BLOCK-STOP: 양쪽 다 공간부족(side=0)이면 기존엔
                    # 장애물 비용이 통째로 꺼져(게이트=|side|) 감속 없이 관통 시도
                    # (U커브 s19.3 장애물 벽충돌, bag 18:54 실증). 막힘 플래그를
                    # 세워 노드 명령단이 장애물 앞 정지하도록.
                    self._obs_blocked_d = float(sel[0])
                    self._log.warn_throttle(1.0,
                        "[MPC] obstacle BLOCKED both sides (d=%.2f m) — stop guard",
                        float(sel[0]))

        # Use committed values; if no commitment, fall back to sentinel.
        if self._committed_obs is not None:
            cox, coy, cs_obs, cside, ce_c = self._committed_obs
            sel = [math.hypot(initial_state[0] - cox,
                              initial_state[1] - coy), cox, coy]
            # 0731 페이드: 통과 후 Δs 1.5~3.0 m 구간에서 ±1 → 0 으로 선형 감소.
            # 장애물이 없으면 이 분기 자체에 안 들어오므로 단일 레이싱은 불변.
            side_pref = cside * self._release_fade
            e_c_obs_val = self._narrow_shifted(cox, coy, cside, ce_c)  # 0723c
            _s_obs1 = float(cs_obs)   # 0723g s-창 게이트용
        else:
            sel = [1e6, 1e6, 1e6]
            side_pref = 0
            e_c_obs_val = 0.0
            _s_obs1 = None
            # ── 2026-08-22 v2: 선택기·커밋과 무관하게 "경로 위" 장애물은 무조건 제약 ──
            # v1(sel_is_real 요구)은 select_front_obstacle 이 센티넬을 내면(실차 32.9%)
            # 발동하지 않았다. 실차 실측(맥 14:38): 2 m 내 제약활성 100% 인 장애물만
            # 무접촉, 8~28% 인 것들은 전부 접촉/스침 — 활성률과 접촉이 완전 상관.
            # 규칙: 라인 횡거리가 (차반폭 0.175 + 장애물반폭 0.15 + 여유 0.10) 이내이고
            #       전방 6 m 안이면, 커밋 슬롯 상태와 무관하게 가장 가까운 것을 제약한다.
            try:
                _s_car_b = float(initial_state[IDX_S]) % L
                _band = 0.175 + 0.15 + 0.10
                _best = None
                for _ob in (obstacles or []):
                    if not (hasattr(_ob, '__len__') and len(_ob) >= 2):
                        continue
                    _ox, _oy = float(_ob[0]), float(_ob[1])
                    _so, _eo = self._obstacle_frenet(_ox, _oy)
                    _dsb = (_so - _s_car_b) % L
                    if _dsb > 0.5 * L:
                        _dsb -= L
                    if -0.5 <= _dsb <= 6.0 and abs(float(_eo)) <= _band:
                        if _best is None or _dsb < _best[0]:
                            _best = (_dsb, _ox, _oy, _so, _eo)
                if _best is not None:
                    _dd = math.hypot(initial_state[0] - _best[1],
                                     initial_state[1] - _best[2])
                    _sd1 = self.decide_side_pref(_best[3], _best[4],
                                                 obs_xy=(_best[1], _best[2]))
                    if _sd1 != 0:
                        sel = [_dd, _best[1], _best[2]]
                        side_pref = float(_sd1)
                        e_c_obs_val = float(self._narrow_shifted(
                            _best[1], _best[2], _sd1, _best[4]))
                        _s_obs1 = float(_best[3])
                        self._log.info_throttle(
                            2.0, "[MPC] 경로상 장애물 강제제약 (d=%.2f Δs=%.2f e_c=%+.2f side=%+d)",
                            _dd, _best[0], float(_best[4]), int(_sd1))
                    else:
                        # 피할 방향이 없다 → 감속/정지. 노드 명령단이 이 값을 보고
                        # 장애물 앞에서 멈춘다(BLOCK-STOP). "멈추고 생각하기".
                        self._obs_blocked_d = float(_dd)
                        self._log.warn_throttle(
                            1.0, "[MPC] 경로상 장애물인데 피할 쪽 없음 (d=%.2f) — 정지가드", _dd)
            except Exception as _e1x:
                self._log.warn_throttle(5.0, "[MPC] 경로상 제약 실패: %s", _e1x)
        self.dbg_side_pref = float(side_pref)
        # ── P2 Stage-B (0722o): 2번째 장애물 슬롯 — 매 사이클 fresh ──────
        # sel(#1)·committed 외 장애물 중 차 전방 Δs∈(0.3,9m) 최근접 1개.
        # side2 = P1 맵실측 fresh (커밋 상태기계 없음 — 순수 예고 제약).
        # #1 release 전에 #2가 이미 제약에 존재 → 가림쌍 1.5m 갭 소멸,
        # 계획이 #1→#2를 연속 곡선으로.
        self._obs2 = None
        self._obs3 = None
        try:
            _s_car2 = float(initial_state[IDX_S]) % L
            _cands = []
            for _ob in (obstacles or []):
                if not (hasattr(_ob, '__len__') and len(_ob) >= 2):
                    continue
                _bx, _by = float(_ob[0]), float(_ob[1])
                if sel_is_real and (_bx - float(sel[1])) ** 2 + (_by - float(sel[2])) ** 2 < 0.25:
                    continue                      # #1(sel/committed) 자신 제외
                _s2o, _e2o = self._obstacle_frenet(_bx, _by)
                _ds2 = (_s2o - _s_car2) % L
                if 0.3 <= _ds2 <= 9.0:
                    _cands.append((_ds2, _bx, _by, _s2o, _e2o))
            # 0723b: 전방 최근접 2개 → obs2/obs3 슬롯 (기존 1개 → 3동시 시야에서
            # 3번째 무제약이던 한계 제거). 0.5m 이내 중복 후보는 하나로.
            _cands.sort(key=lambda c: c[0])
            _picked = []
            for _c in _cands:
                if any((_c[1] - _p[1]) ** 2 + (_c[2] - _p[2]) ** 2 < 0.25 for _p in _picked):
                    continue
                _picked.append(_c)
                if len(_picked) == 2:
                    break
            # 0722p 스로틀 유지: decide(맵 raycast) 재판정은 후보별 0.25s 캐시.
            _now2 = time.monotonic()
            _cache = getattr(self, '_obs23_cache', {})
            _slots = []
            for _ds2, _bx, _by, _s2o, _e2o in _picked:
                _key = (round(_bx / 0.3), round(_by / 0.3))
                _hit = _cache.get(_key)
                if _hit is not None and _now2 - _hit[0] < 0.25:
                    _sd2 = _hit[1]
                else:
                    _sd2 = self.decide_side_pref(_s2o, _e2o, obs_xy=(_bx, _by))
                    _cache[_key] = (_now2, _sd2)
                if _sd2 != 0:
                    _slots.append((_bx, _by, float(_sd2),
                                   float(self._narrow_shifted(_bx, _by, _sd2, _e2o)),
                                   float(_s2o)))  # 0723c shift / 0723g s-창용 s
            if len(_cache) > 32:
                _cache = {k: v for k, v in _cache.items() if _now2 - v[0] < 1.0}
            self._obs23_cache = _cache
            if _slots:
                self._obs2 = _slots[0]
            if len(_slots) > 1:
                self._obs3 = _slots[1]
        except Exception as _e2exc:
            self._log.warn_throttle(5.0, "[MPC] obs2/3 select skip: %s", _e2exc)
        self.dbg_sel_dmin, self.dbg_sel_x, self.dbg_sel_y = sel

        # Avoidance prior on warm-start X0 — Option 2: at the just-committed
        # cycle, push HARDER and FARTHER. Without this strong push, X0 is
        # the previous (no-obstacle) prediction; the half-plane constraint
        # is suddenly active and slack absorbs a 1-2 m violation → cost
        # spike. With the strong commit-time push, X0 already curves around
        # the obstacle, so the half-plane is near-satisfied from cycle one.
        # After commit (subsequent cycles), the regular gentler push
        # maintains the detour shape as the car approaches.
        # 0723H 추월 킥 (동적/원형 모드): node가 요청 시 1회 — 추종 로컬미니멈
        # 탈출용 X0 옆 점프. 커밋 킥과 동일 형식, 원형 제약이 간격은 계속 보장.
        _otk = getattr(self, '_ot_kick', None)
        if _otk is not None:
            self._ot_kick = None
            _ks, _kx, _ky = _otk
            for k in range(1, self.N + 1):
                xk, yk, psik = self.X0[k, 0], self.X0[k, 1], self.X0[k, 2]
                d2k = (xk - _kx) ** 2 + (yk - _ky) ** 2
                w = math.exp(-d2k / (2.0 * 4.0))
                nx_, ny_ = -math.sin(psik), math.cos(psik)
                self.X0[k, 0] += _ks * 0.5 * w * nx_
                self.X0[k, 1] += _ks * 0.5 * w * ny_
            for k in range(self.N + 1):
                self.solver.set(k, "x", self.X0[k, :])
        if side_pref != 0:
            if just_committed:
                D_PRIOR     = 0.40   # bigger lateral kick at commit
                SIGMA_SQ    = 4.0    # σ=2.0 — wider Gaussian, more stages
                # 0812: 8.0→20.0. 커밋은 max(6, 1.5·v) 거리에서 나는데(7m/s=10.5m)
                # 킥은 8m 안에서만 → v>5.3 이면 영영 미발동. 킥 없으면 RTI 가 직선
                # 이전해에 들러붙어 detour 0 (0722v 실증). 커밋 시 항상 킥.
                TRIGGER_D   = 20.0   # always push at commit
            else:
                # 0723j 0.30→0.10 → 0723k 0.10→0.0: 지속 푸시는 매 사이클
                # "이전 해 + D·w" 가산이라 수렴해를 경계 밖으로 계속 밀어냄.
                # U자 실측(bag 14_01): PASS 랩 e_c -0.21~-0.27 vs FAIL 랩
                # -0.30~-0.41 — 통과 밴드 폭이 0.21m뿐이라 0.10 가산도 치명.
                # 하드제약+shift가 경계를 지키므로 지속 푸시는 완전 제거,
                # 커밋 킥(0.40, 로컬미니멈 탈출)만 유지.
                D_PRIOR     = 0.0
                SIGMA_SQ    = 1.0
                TRIGGER_D   = 0.0   # 지속 푸시 비활성 (TRIGGER 0 = 발동 안 함)
            if float(sel[0]) < TRIGGER_D:
                ox, oy = sel[1], sel[2]
                for k in range(1, self.N + 1):
                    xk, yk, psik = self.X0[k, 0], self.X0[k, 1], self.X0[k, 2]
                    d2k = (xk - ox) ** 2 + (yk - oy) ** 2
                    w = math.exp(-d2k / (2.0 * SIGMA_SQ))
                    nx_, ny_ = -math.sin(psik), math.cos(psik)
                    self.X0[k, 0] += side_pref * D_PRIOR * w * nx_
                    self.X0[k, 1] += side_pref * D_PRIOR * w * ny_
                for k in range(self.N + 1):
                    self.solver.set(k, "x", self.X0[k, :])

        # Per-stage parameter array
        # X0[k, IDX_S] is unbounded (solver-internal monotonic s), so always
        # wrap with % L for spline lookups (corridor / track boundary).
        _alat_eff = float(self.a_lat_safe_eff())   # 사이클당 1회 — stage 루프 불변
        for k in range(self.N + 1):
            sk = float(self.X0[k, IDX_S]) % L
            # 0724 학습 스케일 (기본 1.0). _vcap(HARD cap)에도 alat 스케일을
            # 반영해야 cost 의 상향된 코너 목표를 HARD cap 이 막지 않는다.
            _rsc = _asc = 1.0
            if self._refv_scale_fn is not None:
                try:
                    _rsc = float(self._refv_scale_fn(sk))
                except Exception:
                    _rsc = 1.0
            if self._alat_scale_fn is not None:
                try:
                    _asc = float(self._alat_scale_fn(sk))
                except Exception:
                    _asc = 1.0
            try:
                lx = float(self.left_lut_x(sk));  ly = float(self.left_lut_y(sk))
                rx = float(self.right_lut_x(sk)); ry = float(self.right_lut_y(sk))
            except Exception:
                lx = ly = rx = ry = 0.0
            # ── ★ 2026-06-02 per-stage HARD corner-speed cap (kinematic) ──
            # 진단: 곡률 soft-cost(ref_v) 와 slack 된 a_lat 제약은 q_p(progress)
            # 에 밀려 코너 진입속도를 강제 못 함 → R=0.87m 코너에 v=3~5 진입 →
            # 필요 a_lat=v²/κ⁻¹ ≫ 한계 → understeer 로 벽 클립/충돌(맵은 안 좁음).
            # → 각 예측 stage 의 속도 컨트롤 u[0] 상한을 v ≤ √(a_lat_safe/|κ(s_k)|)
            # 으로 HARD 제약. 직선(|κ|→0)은 v_max 그대로(빠름), 코너만 물리적
            # grip 속도로 강제(깨끗). PP 가 암묵적으로 하는 grip-limited racing.
            # |κ| 는 forward-max LUT(lookahead 6m) → 코너 6m 전부터 감속 시작.
            # s_k 는 이전 solve 의 warm-start 궤적(self.X0)에서 — 약간의 오차는
            # 보수적 floor 로 흡수. 후진(stuck-recover)은 mpc_node 가 cmd.speed
            # 로 직접 처리 → u[0] lbu=0 이므로 이 cap 과 무간섭.
            try:
                _absk = float(self.abs_kappa_lut(sk))
                _vcap = math.sqrt(_alat_eff * _asc / (_absk + 1e-3))
                # floor 1.0: κ-스파이크나 cold warm-start 에서 v→0 stall 방지
                # (가장 타이트한 코너도 √(6/0.84)=2.67 라 floor 는 평소 불활성).
                _vcap = min(float(self.v_max), max(_vcap, 1.0))
                # unified 8-state (both modes): vx 는 STATE idx 3. HARD ubx[vx]
                # cap 은 vx 가 못 따라잡으면(a_min=-3 제동한계) QP infeasible →
                # MINSTEP 28→95 폭증(2026-06-03 측정). 대신 GENEROUS margin(×1.6)
                # 으로 극단 과속만 막고, 코너속도는 soft ref_v+a_lat 가 담당.
                # 이래야 slip-aware 추종을 살리면서 MINSTEP 안 늘림.
                # k<N only: the terminal node has no state box (nbx_e=0), so
                # setting ubx there throws (was 61×/run log spam at k=N).
                if 1 <= k < self.N:
                    _vcap_dyn = min(float(self.v_max) + 0.5, _vcap * 1.6)
                    _ub = [_vcap_dyn, 10.0, 20.0]
                    self.solver.set(k, "ubx", np.array(_ub))
                # ── kinematic-only p_v grip cap (mode-dependent NUMERIC knob,
                # same family as the LM damping — NOT an algorithmic difference;
                # constraint STRUCTURE stays identical in both modes). f_kin has
                # no tire physics, so soft ref_v + slacked a_lat lose to the
                # progress reward at corner entry (2026-06-02 diagnosis; 2026-06-10
                # final2 bring-up: deterministic crash at the same 3 corners every
                # lap). The old 5-state kin capped the speed INPUT hard at ×1.0;
                # in the unified layout the speed-intent input is p_v (u[2]) —
                # cap it to the κ-grip speed. INPUT bound → always feasible
                # (a hard ubx[vx] ×1.0 measurably spiked MINSTEP 28→95), and the
                # ×1.6 ubx stays untouched in both modes. Dynamic mode skips
                # this: the tire model self-limits, validated clean at ×1.6.
                if (not self.use_dynamic) and k < self.N:
                    _ubu = self._ubu_codegen.copy()
                    _ubu[2] = min(_ubu[2], _vcap)
                    self.solver.set(k, "ubu", _ubu)
            except Exception as _e:
                # Was `pass` — that hid the joint-α ubx dimension bug for weeks.
                # Log (throttled) so a per-stage constraint failure is visible.
                self._log.warn_throttle(2.0,
                    "[MPC-acados] per-stage bound set failed (k=%d): %s", k, _e)
            # 2026-05-28 #18 LMPC: p_arr 길이 22 → 76. Reviewer #★1 confirmed —
            # codegen 길이가 76 이면 22 길이 set 은 dimension throw. p_arr 전체
            # 76 으로 짜되, LMPC slots (18..71) 은 attr 가 있으면 사용, 없으면 0
            # (default → lmpc_w=0 → LMPC term 0 → 기존 동작 보존).
            p_arr = np.zeros(self._n_p_total, dtype=float)
            # 0..17 — 기존 constants
            p_arr[0]  = float(sel[0])               # dmin (debug)
            if pred_obs is not None and k < len(pred_obs):
                p_arr[1] = float(pred_obs[k][0])    # obs_x (predicted @ stage k)
                p_arr[2] = float(pred_obs[k][1])    # obs_y
            else:
                p_arr[1] = float(sel[1])            # obs_x (legacy: current pos)
                p_arr[2] = float(sel[2])            # obs_y
            # 0723g s-창 게이트: h_obs 게이트가 유클리드 거리라 헤어핀에선 반대편
            # 레그(직선거리 1.5~2m)의 스테이지까지 제약돼 계획이 엉뚱한 곳에서
            # 밀림 (bag 13_35: s22.4 장애물, 계획 Max d -0.82 부풀며 내벽 충돌 3/4랩).
            # 스테이지 s가 장애물 s ±2.0m 밖이면 그 스테이지는 sentinel → 게이트 0.
            # 접근 셰이핑은 장애물 ±2m 스테이지가 담당하므로 직선 거동 불변.
            # 0723i 비대칭 창: 통과쪽(+)은 0.9m(장애물반경 0.15+차 반길이 0.29+여유)만.
            # +2.0 대칭창은 통과 후에도 e_c 요구+atten 감쇠를 s+2m까지 유지 —
            # 헤어핀 출구처럼 하류 벽이 좁아지는 곳(-e_c 벽 0.95→0.50, s23.2-23.7
            # 실측)에서 계획이 바깥에 눌린 채 출구 진입 → 전면코너 벽접촉 (bag
            # 13_50: 5/6랩, 계획-추종 오차 3cm인데 계획 자체가 벽으로). 창 밖은
            # sentinel → 게이트·atten 동시 해제 = 즉시 라인 복귀.
            if _s_obs1 is not None:
                _dso1 = (sk - _s_obs1 + 0.5 * L) % L - 0.5 * L
                if _dso1 < -2.0 or _dso1 > 0.9:
                    p_arr[1] = 1e6; p_arr[2] = 1e6
            p_arr[3]  = float(side_pref)
            p_arr[4]  = 0.0  # D_DETOUR slot: dead (side_term hardcoded 0), kept for index stability
            # 2026-07-29 곡률 비례 R_car (0722g R_safe 와 같은 패턴):
            #   직선 r_car_straight ↔ 코너 R_car_live 를 |κ|/κ_ref 로 보간.
            # 상수 0.30 은 직선에서 레이스라인을 코리도 한계 9cm 앞까지 몰아
            # 솔버가 물러서는 "직선 S자"를 만들었다 (실측 필요 반치수: 직선
            # p95 0.245 / 헤어핀 0.256 — mpc_tuning.yaml corridor 절 참고).
            # min() 가드: R_car_live 를 0.24 밑으로 내리면 그 값이 전 구간 상한.
            _rcl = float(self.R_car_live)
            _rcs = min(self._rcar_straight, _rcl)
            _ik5 = int((sk / self.kappa_ds)) % len(self.kappa_grid)
            _kf5 = min(1.0, abs(float(self.kappa_grid[_ik5])) / self._rcar_kref)
            p_arr[5]  = _rcs + (_rcl - _rcs) * _kf5
            # 2026-07-21 헤어핀 q_cte 존 부스트: corridor_hard_zones 안 스테이지는
            # q_cte_scale × qcte_zone_factor (기본 1.0=off, live). 헤어핀 라인추종
            # 타이트닝 — slack 하드닝과 같은 존을 공유 (존=문제 s구간이므로).
            # 2026-07-21b 회피 우선: committed 장애물 ±3m 스테이지는 부스트 제외 —
            # 존 안 라인 정중앙 장애물에서 q_cte×2.5가 라인으로 당겨 detour 못 벌리고
            # v=0 갇힘 (bag 20_10: CONTACT 전부 존 안). LMPC 회피중 off와 같은 원리.
            _avoid_s = (float(self._committed_obs[2])
                        if getattr(self, '_committed_obs', None) is not None else None)
            _near_obs = (_avoid_s is not None
                         and min((sk - _avoid_s) % L, (_avoid_s - sk) % L) < 3.0)
            # 2026-08-19: 존별 q_cte 배율. 전역 qcte_zone_factor 하나로는
            # "시케인은 풀고 출구코너는 조이는" 조합이 불가능했다 (배율이 하나뿐).
            # 우선순위: 존 자신의 배율 > 전역 폴백. 폴백을 남겨야 존별 값이 없던
            # 시절 yaml 이 무수정으로 변경 전과 동일하게 동작한다.
            # 존 안이어도 배율이 1.0 이면 break 하지 않는다 — slack 전용 존이
            # 뒤쪽 q_cte 존을 가려버리면 안 되기 때문.
            _qzf = float(getattr(self, 'qcte_zone_factor', 1.0))
            _qf = 1.0
            if not _near_obs:
                for _z0, _z1, _f, _zq in (getattr(self, 'corridor_hard_zones', None) or ()):
                    if _z0 <= sk <= _z1:
                        _cand = _zq if _zq != 1.0 else _qzf
                        if _cand != 1.0:
                            _qf = _cand
                            break
            p_arr[6]  = float(self.q_cte_scale_live) * _qf
            # 0722g: 스테이지 곡률 비례 R_safe 상향 — 헤어핀에서 solver가 실제로
            # 더 멀리 벌리도록 (호 수축 보상; decide 임계와 동일 원리, 파라미터라
            # codegen 불필요). 직선(κ≈0)은 기존값 그대로.
            # 0722w κ게인 ½: 풀게인은 중간곡률 스팟(s31/s70)에서 요구가 코리도
            # 허용을 수cm 초과 → soft모순 → 횡거동 마비 → 정면접촉 (bag 17_33:
            # s31 요구+1.05 vs 상한+1.03). 통과/실패가 이 부등식으로 정확히 갈림.
            # 0723e κ게인 재축소 0.5κ→0.25κ, cap 0.3→0.15 (decide 0.4κ/0.25와 동조):
            # 코너 keepout 과대 → 좁은 코너 접근경로가 코리도 밖 → 벽충돌 (bag 13_21).
            _ikk = int((sk / self.kappa_ds)) % len(self.kappa_grid)
            # 2026-08-06 최종리뷰 G1: obstacle 제약 전용 R_safe — OT 축소는
            # R_safe_obs_live(비-None 시)로만 반영, R_safe_live 는 무변경.
            _rs_obs = self.R_safe_obs_live
            _base_rs = float(_rs_obs) if _rs_obs is not None else float(self.R_safe_live)
            p_arr[7]  = _base_rs + min(0.15, 0.25 * abs(float(self.kappa_grid[_ikk])))
            p_arr[8]  = float(self.q_lag_scale_live)
            p_arr[9]  = e_c_obs_val
            p_arr[10] = _alat_eff
            p_arr[11] = float(self.D_apex_live)
            p_arr[12] = float(self.q_psi_scale_live)
            # ── 2026-07-16 직선 전용 q_v 부스트 (κ-게이트, per-stage) ──
            # 진단: 고속존 ref-cmd +1.79 / cmd-act -0.02 (bag 0709-19:11) —
            # 차는 명령을 따라오는데 명령이 ref 를 안 쫓음 = 전역 q_v 가
            # 코너 타협값이라 직선에서 낮음. 코너캡과 같은 abs_kappa_lut 로
            # 직선 stage 만 q_v ×factor (1.0=off, codegen 불필요, live).
            _qv = float(self.q_v_scale_live)
            _sf = float(getattr(self, "straight_qv_factor", 1.0))
            if _sf > 1.0:
                try:
                    if float(self.abs_kappa_lut(sk)) < float(getattr(self, "straight_kappa_thr", 0.15)):
                        _qv *= _sf
                except Exception:
                    pass
            p_arr[13] = _qv
            p_arr[14] = float(self.q_dd_scale_live)
            p_arr[15] = float(self.q_p_scale_live)
            p_arr[16] = float(self.q_drate_scale_live)
            p_arr[17] = float(self.q_dv_scale_live)
            # 18..67 — LMPC SS slots (caller fills via self._lmpc_ss_states / _lmpc_ss_Q
            # 외부에서 set_lmpc_params() 로 갱신). 미설정 시 zeros + lmpc_w=0 으로 무시.
            ss_states = getattr(self, '_lmpc_ss_states', None)
            ss_Q      = getattr(self, '_lmpc_ss_Q', None)
            if ss_states is not None and ss_Q is not None:
                # H10 fix (2026-07-03): SS ψ(행 2)를 solver 의 unwrapped ψ 브랜치로
                # 리베이스 — anchor ψ 잔차가 2πk 로 자라는 발산 차단. initial_state[2]
                # 는 이 시점에 이미 yaw_offset 이 더해진 unwrapped 값.
                ss_states = np.array(ss_states, dtype=float, copy=True)
                ss_states[2, :] = rebase_angles_to_ref(ss_states[2, :],
                                                       initial_state[2])
                # ss_states (4, 10) F-order pack, ss_Q (10,)
                p_arr[18:58] = ss_states.flatten(order='F')  # column-major: [s0_x, s0_y, s0_psi, s0_vx, s1_x, ...]
                p_arr[58:68] = ss_Q
            else:
                # padding Q = 1e6 (reviewer: exp(-β·1e6) ≈ 0 자연 무시)
                p_arr[58:68] = 1e6
            # 68..71 — LMPC scalars (default OFF)
            # 2026-06-11: LMPC term cost gated OFF while committed to an
            # obstacle. Safe set 랩들은 장애물 없이 기록된 라인 — cost-to-go 가
            # 터미널 상태를 장애물 관통 라인으로 끌어당겨 half-plane 회피와
            # 싸운다 (관측: LMPC on + 회피 중 wedge/stuck). side_pref!=0 인
            # 동안만 0; 통과 후 release 되면 원래 weight 복귀.
            _lmpc_w = float(getattr(self, 'lmpc_w_live', 0.0))
            if side_pref != 0:
                _lmpc_w = 0.0
            p_arr[68] = _lmpc_w
            p_arr[69] = float(getattr(self, 'lmpc_alpha_live', 1.0))
            p_arr[70] = float(getattr(self, 'lmpc_beta_live', 0.05))  # reviewer 권장
            p_arr[71] = float(getattr(self, 'lmpc_reg_w_live', 0.001))
            # 72..75 — corridor (per-stage)
            p_arr[72] = lx; p_arr[73] = ly
            p_arr[74] = rx; p_arr[75] = ry
            # 79..82 — P2 Stage-B obs2 (부재 sentinel: x=1e6, side2=0)
            # 0723g: 슬롯 튜플 5번째 = 장애물 s — 스테이지 s-창(±2m) 밖이면 sentinel.
            _o2 = getattr(self, '_obs2', None)
            if _o2 is not None and (len(_o2) < 5 or
                    -2.0 <= ((sk - _o2[4] + 0.5 * L) % L - 0.5 * L) <= 0.9):
                p_arr[79] = _o2[0]; p_arr[80] = _o2[1]
                p_arr[81] = _o2[2]; p_arr[82] = _o2[3]
            else:
                p_arr[79] = 1e6; p_arr[80] = 1e6
                p_arr[81] = 0.0; p_arr[82] = 0.0
            # 83..86 — 0723b obs3 (동일 sentinel 규약)
            _o3 = getattr(self, '_obs3', None)
            if _o3 is not None and (len(_o3) < 5 or
                    -2.0 <= ((sk - _o3[4] + 0.5 * L) % L - 0.5 * L) <= 0.9):
                p_arr[83] = _o3[0]; p_arr[84] = _o3[1]
                p_arr[85] = _o3[2]; p_arr[86] = _o3[3]
            else:
                p_arr[83] = 1e6; p_arr[84] = 1e6
                p_arr[85] = 0.0; p_arr[86] = 0.0
            # 76..78 — B4' e_corr (const across horizon)
            p_arr[76] = float(self._e_corr[0])
            p_arr[77] = float(self._e_corr[1])
            p_arr[78] = float(self._e_corr[2])
            # 87 — 0723u obs1 모드 (node가 동적 상대 감지 시 1)
            p_arr[87] = float(getattr(self, 'obs1_mode', 0.0))
            # 88/89 — 0724 온라인 학습 속도 프로파일 스케일 (기본 1.0)
            p_arr[88] = _rsc
            p_arr[89] = _asc
            self.solver.set(k, "p", p_arr)
            # ── 국소 corridor slack 하드닝 (스텁 존 등) ──
            if 1 <= k < self.N and getattr(self, '_slack_def', None) is not None:
                _fac = 1.0
                # 0721: 경계 계단(1.0↔f 점프)이 존 경계 s24-25/s34-35 조향 진동 유발
                # → 존 밖 1m 램프로 완만화 (존 안은 풀 강도 유지). 0.25 양자화로 cost_set 캐시 보존.
                # 0721b 회피 우선: committed 장애물 ±3m 스테이지는 하드닝 제외 (위 q_cte
                # 부스트와 동일 원리 — 존 slack ×5-6이 detour 공간을 막아 v=0 갇힘).
                _RAMP = 1.0
                # _near_obs면 존 목록을 비워 _fac=1.0 → 아래 기존 cost_set 경로가
                # 자연히 기본 slack으로 복원 (별도 리셋 불필요).
                for _z0, _z1, _f, _ in (() if _near_obs else (getattr(self, 'corridor_hard_zones', None) or ())):
                    if (_z0 - _RAMP) <= sk <= (_z1 + _RAMP):
                        _w = min(1.0, (sk - (_z0 - _RAMP)) / _RAMP, ((_z1 + _RAMP) - sk) / _RAMP)
                        # 0729 완화 존(factor<1) 지원: max() 는 <1 을 삼켜버린다.
                        # '1에서 더 멀리 벗어난 쪽'을 채택 — 강화/완화 존이 겹치면
                        # 효과가 큰 쪽이 이긴다.
                        _cand = 1.0 + (float(_f) - 1.0) * max(0.0, _w)
                        if abs(_cand - 1.0) > abs(_fac - 1.0):
                            _fac = _cand
                # 0722r near-obs corridor 하한 ×4: 0721b의 완전해제(×1)는 obs행이
                # ×10(2000)이 된 지금 밸런스 붕괴 — 두 장애물 사이 낑기면 벽(20/15)이
                # 최약 탈출구 → 계획이 벽 관통 (bag 16_35: 벽너머 계획 4%, 피크 34/41).
                # ×4(80/60)는 detour 공간은 유지하되 벽이 편법이 되지 않는 수준.
                if _near_obs:
                    # 2026-08-05: 4.0 → 1.5. ×4 는 코리도 기본 20/15 시절 보정
                    # (0722r, 의도 80/60). 기본이 60/400 이 된 지금 ×4 는 240/1600
                    # — 장애물 부스트(2000/7000)와 양쪽 절벽을 만들어 직선 회피 중
                    # 채터링(칼날 균형)의 원인 (bag 16_01 실측: s34.9 회피 흔들림).
                    # 60/400 자체가 이미 "벽=탈출구" 를 막는 세기라 소폭만 남긴다.
                    _fac = 1.5
                _fac = round(_fac * 4.0) / 4.0
                # 0722l 커밋 중 장애물 slack 행(0) ×10: 계획이 keepout을 slack으로
                # 사던 것 차단 (bag 15_52: plan이 5.3m/s로 violation 0.5m 유지한 채
                # 접촉 — zl200·0.5=비용 100이 progress에 밀림). 장애물 ±3m 스테이지만.
                _fobs = 10.0 if _near_obs else 1.0
                _lkey = (_fac, _fobs)
                # 0729 기본값 None: 존이 하나도 없어도(=_lkey (1,1)) 첫 사이클에
                # 한 번은 cost_set 이 돌게 한다. warm-reuse(generate=False) 로 옛
                # 슬랙값이 생성 C 에 구워져 있으면 _slack_def 변경이 무시되던 문제.
                if self._slack_last.get(k) != _lkey:
                    for _key, _base in self._slack_def.items():
                        _v = _base.copy()
                        _v[0] *= _fobs                    # 장애물 행
                        _v[5] *= _fobs                    # 장애물2 행 (Stage-B)
                        _v[6] *= _fobs                    # 장애물3 행 (0723b)
                        _v[1] *= _fac; _v[2] *= _fac      # corridor top/bot 행
                        self.solver.cost_set(k, _key, _v)
                    self._slack_last[k] = _lkey

        # ---- Stage 0 init state via tightened bounds ----
        self.solver.set(0, "lbx", initial_state)
        self.solver.set(0, "ubx", initial_state)

        # Option 3: at "transient" cycles (new commit, ψ-wrap, just-cold-
        # started after a stuck/spike fallback), the cost surface and/or
        # dual active set change abruptly. SQP_RTI's single iter cannot
        # converge duals from cold (or near-cold) state on those cycles
        # → corrupt control output (near-zero, saturated, or oscillating).
        # Run 2 extra solve() calls before the main solve so SQP has
        # converged. Pure cycles (~99%) skip and do single-iter as before.
        # Cost: ~6 ms extra on transient cycles, well within 25 ms budget.
        # Critical for breaking the stuck-at-zero loop: stuck override
        # produces u=0; next cycle's cold-start reset() zeros duals; if
        # we don't multi-iter here, 1-iter from cold dual returns ~0
        # again → infinite stuck.
        cold_started = getattr(self, '_just_cold_started', False)
        # Multi-iter pre-solve ONLY at transient events (commit, ψ-wrap,
        # cold-start). Per-cycle multi-iter caused oscillating predictions
        # because the second iter found a slightly different optimum
        # than the first → trajectory zigzag visible in RViz.
        if just_committed or psi_unwrapped or cold_started:
            for _ in range(2):
                try:
                    self.solver.solve()
                except Exception:
                    break
        self._just_cold_started = False

        # ---- Solve ----
        try:
            status = self.solver.solve()
            self.dbg_solver_status = self._status_to_string(status)
            traj = np.array([self.solver.get(k, "x") for k in range(self.N + 1)])
            u_seq = np.array([self.solver.get(k, "u") for k in range(self.N)])
            # (0722 OBSDBG 임시 계측은 BIG_OBS 누수 확진 후 제거 — 0723 정리)
            # status: 0=OK, 1=NaN/Failure, 2=MaxIter, 3=MinStep, 4=QP_Failure.
            # 3/4 at the hairpin produces a corrupt warm-start that locks
            # subsequent QPs into a 36-iter limit cycle (observed s≈48m
            # cascade). Forcing WARM_START=False rebuilds X0/u0 from the
            # current car state via Euler rollout next cycle.
            # 2026-06-08: include 2 (MaxIter) — a QP that hit the iteration limit
            # without converging was being accepted as a good solve, applying a
            # half-converged control and seeding a bad warm-start.
            bad_status = status in (1, 2, 3, 4)
            has_nan = np.isnan(traj).any() or np.isnan(u_seq).any()
            if bad_status or has_nan:
                self._log.warn_throttle(1.0,
                    "[MPC-acados] status=%s nan=%s — reset warm-start",
                    self.dbg_solver_status, has_nan)
                self.WARM_START = False
                # Safe fallback control: gentle slow-down, hold heading.
                # Avoids feeding the ROS node a corrupt v_cmd while next
                # cycle re-seeds.
                try:
                    seed_v = max(float(self.ref_v(s0 % L)) * 0.3, 0.5)
                except Exception:
                    seed_v = 1.0
                # joint-α: solver expects _nx_solver-wide x (8 phys + K α). A bare
                # 8-wide tile here makes self.X0 8-wide → next cycle's set(k,"x")
                # raises "dimension 18 vs 8". Build at solver width, α uniform.
                _nxs = getattr(self, '_nx_solver', self.n_states)
                traj = np.zeros((self.N + 1, _nxs))
                traj[:, :self.n_states] = initial_state
                if _nxs > self.n_states:
                    traj[:, self.n_states:] = 1.0 / max(1, _nxs - self.n_states)
                u_seq = np.zeros((self.N, self.n_controls))
                # u[0] = a_x (unified, both modes) — small positive accel to
                # keep the car coasting forward. p_v = seed_v.
                u_seq[:, 0] = 0.5
                # Also seed predicted vx so the output speed is sensible.
                traj[:, 3] = seed_v
                u_seq[:, 2] = seed_v
        except Exception as e:
            self._log.warn_throttle(2.0, "[MPC-acados] solver exception %s — reset", str(e))
            self.WARM_START = False
            _nxs = getattr(self, '_nx_solver', self.n_states)
            traj = np.zeros((self.N + 1, _nxs))
            traj[:, :self.n_states] = initial_state
            if _nxs > self.n_states:
                traj[:, self.n_states:] = 1.0 / max(1, _nxs - self.n_states)
            u_seq = np.zeros((self.N, self.n_controls))
            self.dbg_solver_status = "exception"

        try:
            opti_value = float(self.solver.get_cost())
        except Exception:
            opti_value = float('nan')

        # Cost-spike fallback — when the QP "succeeded" but cost is far
        # above the steady-state level, the solution is untrustworthy
        # (typically saturated v_max + steer=±0.4 from a corrupt warm-
        # start at obstacle commit / ψ-wrap). Output gentle ref_v·0.3
        # straight forward + warm-start rebuild.
        cost_spike = ((not np.isnan(opti_value))
                      and opti_value > self.cost_spike_thr_live)

        # Stuck detection (option D) — cost-spike fallback alone misses
        # the case where the QP returns saturated controls with cost
        # below threshold (observed: cost ≈ 270, vcmd=6, steer=+0.4 at
        # s=60 lap 4, car wedged into wall, fallback never engages).
        # Estimate v_actual from sensed position diff; if it stays below
        # 0.1 m/s while we keep commanding > 2 m/s for several cycles,
        # the car is physically stuck even though the optimizer thinks
        # everything is fine. Override with v=0/steer=0 to release the
        # wall contact, then rebuild warm-start next cycle.
        now_t = monotonic_now()
        v_est = 0.0
        # disp = raw per-cycle displacement [m]. v_est(=disp/dtm) 는 wall-clock dtm
        # 노이즈(timing hiccup)로 작은 disp 인데도 작게 나와 stuck 오발동(2026-05-29
        # forensic: 2/23 false-fire). disp 자체로도 게이트 → 진짜 위치동결만 stuck.
        # prior pos 없으면 1e9 (첫 cycle 미발동).
        disp = 1e9
        if hasattr(self, '_pos_for_v') and self._pos_for_v is not None:
            px, py, pt = self._pos_for_v
            dtm = max(now_t - pt, 1e-3)
            disp = math.hypot(initial_state[0] - px, initial_state[1] - py)
            v_est = disp / dtm
        self._pos_for_v = (float(initial_state[0]),
                            float(initial_state[1]), now_t)
        # Agent R-round3 Fix 1: persistent STUCK release for ~0.5s (20 cycles)
        # with reverse cmd. 이전 1-cycle release 는 cmd=0 후 다음 cycle 의
        # _v_cmd_for_stuck=0 → count reset → release 종료 → 다시 forward cmd →
        # wall contact 안 풀려 infinite loop. release_remaining 카운터로 강제 지속.
        v_cmd_prev = getattr(self, '_v_cmd_for_stuck', 0.0)
        self._stuck_release_remaining = getattr(self, '_stuck_release_remaining', 0)
        # 2026-05-27: strict `> 2.0` 가 fallback 의 seed_v=2.0 과 정확히 일치 → 미발동.
        # 1.5 로 완화: fallback v_floor (=2.0) 보다 낮으면 발동.
        # 2026-05-28 #15: v=6 sim 박힘 시 MPC "자포자기 모드" 진입 → vcmd=0.02 같이 작은 양수 →
        # `> 1.5` 미발동. STUCK 감지 못함 → 영원히 박힘 (1시간 누적 lap_count=144 false).
        # → `> 0.0` 으로 완화 (의도 cmd=0 정확과 자포자기 vcmd≠0 구분).
        # 2026-05-29 startup/respawn grace: stuck-release 의 후진(-0.5)이 출발·리스폰
        # 직후 오발동 (정지→가속 빈 구간을 박힘으로 착각 → "살짝 뒤로 감"). 차가 한 번
        # 이라도 실제로 움직인 뒤(_has_moved)에만 stuck 판정. 리스폰 점프(v_est 가
        # 차 최대속도보다 비현실적으로 큼)는 위치 텔레포트 → latch 리셋 + 그 cycle 무시.
        teleport = v_est > 20.0
        if teleport:
            self._has_moved = False
            # respawn/teleport mid-release: abort the release so the car does
            # not sit frozen right after being moved to a fresh pose.
            self._stuck_release_remaining = 0
        # 2026-07-28: _has_moved 래치를 위치차분(v_est)에서 실측 바퀴속도로 교체.
        # v_est 는 dT=0.04 라 한 사이클 2cm 만 튀어도 0.5 를 넘는데, 바로 아래
        # stuck 조건의 disp 문턱도 2cm 라 경계가 겹쳤다. 위치추정이 한 번만
        # 튀면 래치가 영구히 켜지고 이후 출발 전 정지구간이 STUCK 으로 잡혀
        # "살짝 뒤로 갔다 출발"이 간헐 발생했다(시뮬·실차 양쪽 관측).
        # initial_state[3] = vx (odom twist, VESC 엔코더) 는 위치추정 노이즈와
        # 무관하고 정지 중엔 0 이라 이 오탐 경로가 닫힌다.
        # ⚠ 전제: odom 이 들어와야 vx 가 채워진다. mpc_node._current_state_4 는
        # pose 는 _last_pose, vx 는 _last_odom 으로 따로 채우고 odom 이 없으면
        # vx=0 폴백이라, "pose 만 오고 odom 은 안 오는" 구성에서는 이 래치가
        # 영영 안 켜져 진짜 STUCK 도 못 잡는다. 현재 어떤 launch 도
        # /car_state/pose 를 발행하지 않아(전 백엔드가 /car_state/odom 만 채움)
        # 도달 불가지만, 그 토픽을 되살릴 때는 이 지점을 재검토할 것.
        # ★ 2026-08-03: 래치 ON 은 mpc_node(solve 호출 직전, floor 적용 전
        # raw_vx 기준)로 이동 — 단일 출처. 여기서 initial_state[3] 을 보면
        # cold_start_vx_floor(2.0) 주입값이라 정지 상태에서도 첫 solve 에
        # 래치가 켜져 engage 전 STUCK→텔레포트가 반복됐다 (bag 14_58: engage
        # 전 odom vx 전부 0.0000 인데 stuck-recover 4회). teleport 시 False
        # 리셋(위)은 유지한다.
        if (v_est < 0.1 and disp < 0.02 and v_cmd_prev > 0.0
                and getattr(self, '_has_moved', False)
                and self._stuck_release_remaining == 0):
            self._stuck_count = getattr(self, '_stuck_count', 0) + 1
        else:
            self._stuck_count = 0
        if self._stuck_count > 10:                  # ~0.25 s detect
            self._stuck_release_remaining = 20      # ~0.5 s forced release (ceiling)
            self._stuck_count = 0
            # 2026-06-11 failure-driven side flip: STUCK while committed to an
            # obstacle (and near it) = the cached side choice FAILED. Without
            # this, _side_history replays the same losing side every lap
            # (관측: 같은 장애물에서 매 랩 100% stuck→reverse 반복). Flip the
            # cached side and release — the commit logic re-engages next cycle
            # with the opposite side, so the car retries the other way within
            # the same pass.
            if getattr(self, '_committed_obs', None) is not None:
                cox, coy, cs_obs, cside, ce_c = self._committed_obs
                d_obs = math.hypot(initial_state[0] - cox,
                                   initial_state[1] - coy)
                if d_obs < 5.0:
                    # 0722d: 영구 학습(flip캐시/blocked) 제거 — 방금 막힌 side만
                    # 30초 기억(자동 소멸). 재커밋 시 그 side 제외, 양쪽 다 기억되면
                    # side=0 → BLOCK-STOP. 오염이 랩을 넘어 살아남지 않음.
                    key = round(float(cs_obs) * 2.0) / 2.0
                    if not hasattr(self, '_recent_stuck'):
                        self._recent_stuck = {}
                    _sides, _ = self._recent_stuck.get(key, (set(), 0.0))
                    _sides.add(int(cside))
                    self._recent_stuck[key] = (_sides, time.monotonic())
                    self._log.warn(
                        "[MPC] STUCK during avoidance (d_obs=%.1f m) obs %s side %+d — "
                        "release (막힌쪽 30s 기억: %s)", d_obs, key, cside, sorted(_sides))
                    self._committed_obs = None
                    self.WARM_START = False
        # 2026-07-28: 후진 거리 기반 조기종료 제거. 해제 중 명령이 후진(-0.5)
        # 에서 정지(0.0)로 바뀌어 차가 뒤로 물러나지 않으므로 back_dist 가
        # 문턱을 넘을 수 없다(도달 불가). 해제는 20 사이클 카운트다운으로만
        # 끝난다.
        is_stuck = self._stuck_release_remaining > 0
        if is_stuck:
            self._stuck_release_remaining -= 1
        self._stuck_release_active = False          # set True below if stuck

        if cost_spike or is_stuck:
            if is_stuck:
                self._log.warn_throttle(1.0,
                    "[MPC] STUCK release n=%d (v_est=%.2f, last_vcmd=%.2f) — hold-stop",
                    self._stuck_release_remaining, v_est, v_cmd_prev)
                u_seq = np.zeros((self.N, self.n_controls))
                u_seq[:, 0] = 0.0      # 0820 후진 제거 — 정지 유지
                traj[:, 3] = 0.0       # (이전 후진 -0.5)
                self._stuck_release_active = True
                # Steer EMA reset so old zigzag steer doesn't persist into release
                if hasattr(self, '_steer_filt'):
                    self._steer_filt = 0.0
            else:
                self._log.warn_throttle(1.0,
                    "[MPC] cost %.0f > %.0f — safe fallback",
                    opti_value, self.cost_spike_thr_live)
                # 2026-05-28 fix: v_floor 가 항상 ~2 → v=6 운전 중 fallback 후
                # 차가 2 로 강제 감속 → 다음 cycle 또 cost spike → 무한 cycle.
                # v_max 의 절반 사용 → fallback 후 적당 속도 유지 → 곧 회복 가능.
                # v=6 → v_floor=3.0, v=8 → v_floor=4.0.
                v_floor = max(0.5 * self.v_max, v_est + 1.0)
                try:
                    seed_v = max(float(self.ref_v(s0 % L)) * 0.5, v_floor)
                except Exception:
                    seed_v = v_floor
                u_seq = np.zeros((self.N, self.n_controls))
                u_seq[:, 0] = 1.5          # a_x — stronger accel (unified)
                traj[:, 3] = seed_v        # set predicted vx for output
                u_seq[:, 1] = 0.0          # delta
                u_seq[:, 2] = seed_v       # p_v
            self.WARM_START = False
        # Stash this cycle's commanded v for next-cycle stuck-check.
        # Unified: ROS node sends a traj-derived vx as cmd.drive.speed, so the
        # stuck detector must compare actual v against THAT value, not
        # traj[1, 3] (which at vx≈1.7 + 0.04·0.5 ≈ 1.73 is always < 2.0 so the
        # detector never fires — Agent R2 fix 2026-05-26).
        self._v_cmd_for_stuck = float(traj[-1, 3]) if traj.shape[0] > 1 else 0.0

        # Steer-output EMA filter — corrects cycle-to-cycle steer trembling
        # observed at curves/hairpin (65% of cycles had steer sign flips:
        # +0.40 → −0.10 → +0.40 oscillating). Cost surface is ~flat over
        # multiple δ values that all achieve a similar predicted ψ change,
        # so the IPM picks different corners each cycle. Smoothing the
        # output via EMA absorbs the cycle-to-cycle jitter without
        # changing what the solver sees (warm-start is unaffected). Only
        # δ is filtered — v is left alone so braking and accel stay snappy.
        # α = 0.6 — moderate smoothing (60% new sample, 40% previous);
        # racing-friendly response while killing the +/− oscillation.
        # Agent R-round3 Fix 3: adaptive α — vx<1.5 (Pacejka linear singularity 영역)
        # 에서 cost surface 가 δ 에 대해 거의 flat → solver 가 ±0.15 zigzag pick.
        # 그 영역만 α 줄여 heavy smoothing. vx 충분 (≥1.5) 이면 alpha_steer_live 그대로.
        vx_now = float(initial_state[3])   # unified: vx = state[3] in both modes
        scale = min(1.0, max(0.2, vx_now / 1.5))    # vx=0 → 0.2, vx≥1.5 → 1.0
        alpha = self.alpha_steer_live * scale
        new_steer = float(u_seq[0, 1])
        prev_filt = getattr(self, '_steer_filt', new_steer)
        filt_steer = alpha * new_steer + (1.0 - alpha) * prev_filt
        # Defensive clip: EMA of two in-bound values is in-bound by induction,
        # but a numerical overshoot or a prev_filt seeded from a different
        # bound regime must never push the actuator past the physical limit.
        filt_steer = min(self.theta_max, max(self.theta_min, filt_steer))
        self._steer_filt = filt_steer
        u_seq[0, 1] = filt_steer

        # Track applied δ for next cycle's state-augmentation (delta_prev).
        # We use the FILTERED δ (what actually goes to the actuator) so the
        # rate-cost residual is consistent with the physical command stream.
        self._last_delta_applied = filt_steer

        self.X0 = traj
        self.u0 = u_seq

        # Visualize boundary along predicted s
        try:
            self._publish_boundary(traj)
        except Exception:
            pass

        # Return shapes that match IPOPT MPC: (first_control DM, traj, u_seq, opti)
        return ca.DM(u_seq[0, :]), traj, u_seq, opti_value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _status_to_string(status):
        return {
            0: 'Solve_Succeeded',
            1: 'Failure',
            2: 'Maximum_Iterations_Exceeded',
            3: 'Minimum_Step_Size_Reached',
            4: 'QP_Failure',
        }.get(status, f'status_{status}')

    def _publish_boundary(self, traj):
        """Visualize the EFFECTIVE drivable corridor (track boundary minus
        R_CAR margin), so the user sees what region the MPC actually
        permits the car to occupy. Lowering R_CAR via rqt → dots move
        outward (closer to wall); raising → dots move inward.
        """
        if self.boundary_hook is None:
            return
        right_pts, left_pts = self.get_path_constraints_points(traj)
        # Shift each boundary point toward the centerline by R_CAR. We
        # approximate the inward normal at boundary_i as (center_i - boundary_i)
        # normalized, where center_i is the centerline point at the same s.
        L = float(self.path_length)
        margin = float(self.R_car_live)
        IDX_S_LOCAL = 6   # unified 8-state layout (both modes)
        shifted = []
        for k in range(traj.shape[0] - 1):
            sk = float(traj[k + 1, IDX_S_LOCAL]) % L
            try:
                cx_s = float(self.center_lut_x(sk))
                cy_s = float(self.center_lut_y(sk))
            except Exception:
                cx_s = 0.0; cy_s = 0.0
            for px, py in (right_pts[k], left_pts[k]):
                vx = cx_s - px; vy = cy_s - py
                d  = math.hypot(vx, vy) + 1e-9
                shifted.append((px + margin * vx / d, py + margin * vy / d))
        self.boundary_hook(shifted)

    def heading(self, yaw):
        """yaw → (x, y, z, w) tuple. Caller wraps into geometry_msgs/Quaternion."""
        return yaw_to_quat(yaw)

    def init_mpc_start_conditions(self):
        self.X0 = np.zeros((self.N + 1, getattr(self, '_nx_solver', self.n_states)))
        self.u0 = np.zeros((self.N, self.n_controls))
