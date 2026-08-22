# MPCC 파라미터 튜닝 안내 (맥 실차 기준)

작성 2026-08-18 · 대상 `nonlinear_mpc_acados` · 값은 이 날짜의 맥(miniui-Macmini) 실값

이 문서 하나만 보고 혼자 튜닝할 수 있게 쓴 것이다.
"어느 파일을 고치나 → 언제 반영되나 → 증상별로 뭘 만지나 → 각 값이 뭔가" 순서다.

---

## 1. 고치는 파일은 3개뿐

| 파일 | 무엇이 들어있나 | 반영 방법 |
|---|---|---|
| `config/ddrx_unified_params.yaml` | ROS 파라미터 79개. 평소 튜닝은 거의 다 여기 | 대부분 주행 중 변경 가능 (§2) |
| `config/mpc_tuning.yaml` | 슬랙 벌점 · codegen 가중치 · 장애물 상수 | 재시작 또는 acados 자동 재생성 |
| `launch/mpcc_unicorn.launch.xml` | 토픽 배선 · 노드 on/off | 재시작 |

그 외 파일은 튜닝 대상이 아니다.

### ⚠ launch 가 yaml 을 덮어쓰는 6개
아래는 `mpcc_unicorn.launch.xml` 이 `<param>` 으로 직접 박아서, **yaml 에서 고쳐도 안 먹는다.**
바꾸려면 launch 를 고치거나 launch 인자로 넘겨야 한다.

```
odom_topic_name      cmd_vel_topic_name   track_source
track_name           use_gp_residual      enable_sim_reset
```

---

## 2. 반영 시점 — 3가지밖에 없다

### (A) 즉시 반영 — 주행 중에도 바뀐다  ★52개
매 제어 주기(25 Hz)마다 다시 읽거나 파라미터 콜백으로 밀어넣는다.

```bash
ros2 param set /mpc_node q_p_scale_live 5.0
ros2 param get /mpc_node q_p_scale_live      # 확인
```

주행 중 A/B 를 이걸로 한다. 마음에 드는 값을 찾았으면 **yaml 에도 반드시 옮겨 적어야** 다음 launch 에 남는다.
(또는 `ros2 param set /mpc_node save_params true` → 노드가 현재 런타임 값을 yaml 에 써넣고 스스로 false 로 돌아간다.)

### (B) 재시작하면 반영 — 27개
`mpc_node` 를 껐다 켜면 된다. codegen 재생성은 필요 없다.

### (C) acados 재생성이 필요 — 솔버에 구워진 값
| 어떻게 처리되나 | 해당 값 |
|---|---|
| **자동 감지됨** (바꾸면 알아서 재빌드) | `use_dynamic`, `dyn_mu`, GP ckpt 경로/mtime, `mpc_tuning.yaml` 의 `cost_weights`·`obstacle` 절 |
| **자동 감지 안 됨 — 손으로 캐시를 지워야 함** | `N_horizon`, `dT`, `speed_target`, `lookahead_m`, `max_speed`(ubu 상한) |

두 번째 줄이 함정이다. 안 지우면 **yaml 을 고쳤는데 옛 솔버가 그대로 돈다.**

```bash
rm -rf ~/.acados_codegen/*     # 그 다음 relaunch (재생성 수십 초~수 분)
```

> ⚠ 예전 주석·문서에 나오는 `rm -rf /tmp/acados_codegen_evompcc*` 는 **더 이상 맞지 않다.**
> 캐시는 2026-07-16 부터 `~/.acados_codegen/` 이다. /tmp 경로로 지우면 아무것도 안 지워진다.
> (2026-08-18 에 코드·로그·BO 스크립트의 안내 문구를 전부 고쳤다.)

### ⚠ (A)와 (C)를 동시에 가진 3개 — 반쪽만 먹는다
`dyn_mu`, `max_speed`, `ellipse_frac` 은 제어 루프에서도 읽고 codegen 에도 구워진다.
주행 중 `ros2 param set` 하면 **런타임 클램프만 바뀌고 솔버 내부 모델은 옛 값 그대로다.**
진짜로 바꾸려면 yaml 수정 → (dyn_mu 는 자동 재생성 / max_speed 는 캐시 삭제) → relaunch.

---

## 3. 증상 → 만질 손잡이

| 증상 | 1순위 | 2순위 | 비고 |
|---|---|---|---|
| 전체적으로 느리다 | `q_p_scale_live` ↑ | `progress_reward_gamma` ↑ (C) | q_p 는 와이드호(코너 부풀림)의 직접 원인이기도 하다 |
| 직선에서 속도가 안 붙는다 | `straight_qv_factor` ↑ | `speed_cmd_gain` ↑, `accel_preview_s` ↑ | `brake_anticip_d_min` 이 근거리 제동을 빼줘야 효과 남 |
| 코너 진입 과속 → 박힘 | `a_lat_safe_live` ↓ | `dyn_mu` ↓ (C), `brake_anticip_a` ↑ | a_lat 은 μ·g·η 위로 올려도 클램프돼 무효 |
| 벽에 밀린다 / 코리도 뚫림 | `mpc_tuning.yaml` 의 `slack.corridor` ↑ | `corridor_hard*_factor` ↑ (구간 한정) | 존은 반창고다. 슬랙 60/400 이면 대개 존 없이 된다 |
| 좌우로 흔들린다 (위빙·발진) | `q_drate_scale_live` ↑ | `alpha_steer_live` ↓, `actuation_latency_s` 확인 | 맥 0.6 Hz 자기발진 대책이 q_drate 60 + latency 0.12 세트 |
| 라인이 안쪽으로 안 파고든다 | `track_source: raceline` 확인 | `D_apex_live` ↑ | raceline 모드면 D_apex 는 0 이 맞다 |
| 코너에서 라인이 부푼다(와이드) | `q_cte_scale_live` ↑ | `q_p_scale_live` ↓ | SAFE 세트(4.9469 / 6.5)가 이 문제의 복귀점 |
| 회피가 너무 이르다/늦다 | `commit_dist_live` | `commit_treact` | 반응거리는 검출기 `max_range`(8 m)가 지배한다 |
| 회피 폭이 좁다/넓다 | `R_safe_live` | `R_car_live` | 필요 여유 = `R_safe + R_car` |
| 출발이 굼뜨다 | `startup_speed` ↑ | `cold_start_vx_floor` | 옛 3초 PP warmup 을 이 둘이 대체 |
| 솔버가 자꾸 fallback 뜬다 | `cost_spike_thr_live` ↑ | `nlp_solver_iters` (B) | 정상 cost 500~2000, 8000 = 진짜 불안정만 |

---

## 4. 파라미터 전체 (`ddrx_unified_params.yaml`, 79개)

`L` = 주행 중 반영 / `R` = 재시작 / `C` = codegen 재생성 관련

### 4.1 코스트 가중치 배율 — 평소 튜닝의 주무대
BO 가 학습하는 대상이 이 7개다. 기준 가중치는 `mpc_tuning.yaml` 의 `cost_weights`, 여기 값은 그것의 **배율**이다.

| 파라미터 | 현재값 | 무엇을 | 올리면 | 내리면 |
|---|---|---|---|---|
| `q_cte_scale_live` | L 4.9469 | 횡오차(라인 이탈) 벌점 | 라인에 단단히 붙음, 와이드호 억제 | 코너 자유도↑, 부풀기 쉬움 |
| `q_lag_scale_live` | L 0.7195 | 종방향 정렬(진행 지연) 벌점 | 계획 진행에 맞춰 감 | 늘어짐 |
| `q_psi_scale_live` | L 1.2528 | yaw 오차 벌점 | 차체 방향 안정 | 헤딩 흔들림 |
| `q_v_scale_live` | L 2.0 | ref_v 추종 벌점 | 속도 프로파일 충실 | 속도 자유, 코너 타협 |
| `q_dd_scale_live` | L 1.0 | 조향 − 곡률 피드포워드 편차 | 곡률 예측대로 조향 | (BO 대상 아님) |
| `q_p_scale_live` | L 6.5 | 진행(progress) 보상 | **빨라짐**, 대신 코너 부풀림 | 얌전·느림 |
| `q_drate_scale_live` | L 60.0 | 조향 변화율 벌점 | **위빙·발진 감쇠** | 반응 빠름, 발진 위험 |
| `q_dv_scale_live` | L 0.15 | 종가속 a_x 벌점 | 가감속 부드러움 | 0.08 은 a_x 폭주 회귀 이력 있음 |

> 현재 세트는 **SAFE**(실차 기본). FAST(심 BO)는 `q_cte 5.8716 / q_v 2.6627 / q_p 9.6668` 인데
> 빈 트랙 전용 시그니처라 실차 부팅값으로는 쓰지 않는다.
> `q_drate 60` 은 맥 전용 발진 대책이다 — **27.8 로 내리지 말 것.**

### 4.2 속도 · 가감속

| 파라미터 | 현재값 | 설명 |
|---|---|---|
| `max_speed` | L+C 3.0 | 절대 속도 상한 [m/s]. codegen ubu 에도 구워짐 → 진짜로 바꾸려면 캐시 삭제 후 relaunch |
| `speed_target` | R+C 7.5 | 코스트가 지향하는 목표 속도. **캐시 자동 감지 안 됨** |
| `max_speed_p` | R 10.0 | progress 변수 p 의 천장 |
| `progress_reward_gamma` | R+C 5.0 | 선형 진행보상 −γ·p_v. 1.0 은 사실상 무효, 5.0 ≈ +0.6 m/s |
| `vel_scale` | R 1.3 | 트랙 CSV 의 ref_v 전체 배율 |
| `accel_preview_s` | L 1.0 | 가속 예견 창 [s]. 클수록 VESC 를 더 몰아붙임 |
| `speed_cmd_gain` | L 1.5 | 가속 오차 증폭. 1.0 = off |
| `speed_cmd_err_max` | L 2.2 | 명령 ≤ 실속 + 이 값. 직선 서징/슬래밍 방지. 0 = off |
| `slew_up_rate` / `slew_up_rate_straight` | L 4.0 / 4.0 | 속도 명령 상승률 제한 [m/s²] |
| `slew_down_rate` | L 5.0 *(yaml 에 없음)* | 하강률 제한 |
| `slew_gate_settle_s` | L 0.0 | 코너 탈출 후 정착 지연 [s]. 0 = 없음 |
| `startup_speed` | L 3.0 | 정지 출발 시 출력 속도 하한 |
| `cold_start_vx_floor` | L 2.0 | 솔버 x0 의 vx 하한. 저속 ill-conditioning 회피 |

### 4.3 제동 · 코너 진입

| 파라미터 | 현재값 | 설명 |
|---|---|---|
| `brake_anticip_a` | L 5.5 | 코너 선행 제동 강도 [m/s²]. 크면 일찍·강하게 |
| `brake_anticip_d_min` | L 1.0 | 이 거리 미만은 선행제동에서 제외 [m]. 직선 가속 목줄 해제용 |
| `brake_preview_s` | L 0.4 | min-guard 예견 창 [s] |
| `a_lat_safe_live` | L 8.0 | 횡가속 한계. ref_v cap = √(a_lat/κ). **μ·g·η 를 넘기면 자동 클램프돼 무효** |
| `straight_qv_factor` | L 2.0 | 직선 구간만 q_v ×배. 1.0 = off |
| `straight_kappa_thr` | L 0.15 | 직선 판정 곡률 문턱 [1/m] |

### 4.4 라인 · 코리도

| 파라미터 | 현재값 | 설명 |
|---|---|---|
| `track_source` | R `raceline` | `raceline`(/global_waypoints, IQP 최적화선) 또는 `centerline`. **launch 가 덮어씀** |
| `D_apex_live` | L 0.0 | 코너 안쪽 bias. raceline 모드에선 0 이 맞다 |
| `R_car_live` | L 0.3 | 차 반지름 [m]. 코리도·회피 여유 계산의 기준 |
| `mpc_corridor_half_width` | R 0.0 | 고정폭 코리도. 0 = 트랙 경계 사용 |
| `inflation_factor` | R 0.0 | 경계 안쪽 수축량. 0 = raw boundary |
| `lookahead_m` | R+C 2.0 | 전방 창 [m]. **캐시 자동 감지 안 됨** |
| `extend_part` | R 2 | 루프 겹침 (2 → +50%) |

### 4.5 코리도 하드닝 존 (구간 한정 벌점 배수)

`corridor_hard{,2..6}_s0 / _s1 / _factor` — s0~s1 구간에서 코리도 슬랙 벌점을 factor 배.
**전부 즉시 반영(L)이라 주행 중 A/B 가능.**

| 존 | s0 | s1 | factor | 되돌리기 값 |
|---|---|---|---|---|
| 1 | 19.5 | 28.0 | 1.0 (off) | 4.0 |
| 2 | 8.5 | 14.5 | 1.0 (off) | 4.0 |
| 3 | 5.0 | 7.5 | 1.0 (off) | 4.5 |
| 4 | 0.5 | 2.5 | 1.0 (off) | 3.0 |
| 5, 6 | 0.0 | 0.0 | 1.0 | (zone-save 가 자동 추가한 빈 슬롯) |

`qcte_zone_factor` (L, 1.0) 은 존 안에서 q_cte 를 추가로 ×배.

> 2026-08-12 부로 존은 전부 off 다. 슬랙 60/400 + 폭비례 마진 0.5 + 벽마진 캡슐존이
> 들어오면서 존의 존재 이유(약한 기본 슬랙을 때우는 반창고)가 사라졌기 때문이다.
> 벽에 밀리면 존을 켜기 전에 `mpc_tuning.yaml` 의 `slack.corridor` 부터 본다.

`zone_edit_enable`(L, false) 을 켜면 RViz 에서 존 경계를 드래그로 편집할 수 있고,
`~/save_zones` 서비스로 yaml 에 저장된다.

### 4.6 회피 · 추월

| 파라미터 | 현재값 | 설명 |
|---|---|---|
| `R_safe_live` | L 0.5 | 장애물 반지름 + 안전 margin [m] |
| `commit_dist_live` | L 6.0 | 회피 commit 거리 [m]. 클수록 일찍 |
| `commit_treact` | L 2.0 | 반응 시간 [s]. 고속에서 거리로 환산됨 |
| `obs_pass_speed` | L 3.3 *(yaml 에 없음)* | 통과 시 속도 |
| `overtake_enabled` | L **false** *(yaml 에 없음)* | 추월 로직 on/off |
| `ot_engage_gap` / `ot_commit_gap` / `ot_launch_gap` | L 6.0 / 4.0 / 1.5 *(yaml 에 없음)* | 추월 단계별 간격 [m] |
| `ot_keepout` / `ot_pass_margin` / `ot_cooldown` | L 0.35 / 0.8 / 5.0 *(yaml 에 없음)* | 측방 여유·통과 마진·재시도 대기 |
| `use_opp_prediction` | L true *(yaml 에 없음)* | 상대차 예측 사용 |

> **회피가 늦으면 MPCC 가 아니라 검출부터 본다.** 맥 실차의 장애물 입력은
> `/kiss_loc/obstacle_poses` — kiss_icp 의 BEV 검출기다. 손잡이는 MPCC 가 아니라
> `stack_master/config/kiss.yaml` 의 `detect_*` 절에 있다:
> `detect_z_min/max` 0.2/0.50 (검출 z 밴드), `detect_eps` 0.3 (DBSCAN 반경),
> `detect_min_samples` 2, `detect_track_dist_min` 0.1 (트랙 밖 제거), `max_range` 70.0.
> 물체가 낮거나 스캔이 성글면 z 밴드와 min_samples 부터 의심한다.
>
> 패키지 안의 `scan_obstacle_detector`(`max_range` 8 m)는 2D 라이다용이라
> **이 launch 에서는 안 뜬다** — 심/노트북 경로다. 여기 값을 고쳐도 맥 실차는 안 바뀐다.

### 4.7 모델 · 솔버

| 파라미터 | 현재값 | 설명 |
|---|---|---|
| `use_dynamic` | R+C true | true = 8-state 동역학(타이어 슬립), false = 운동학 f_kin. **자동 재생성됨** |
| `dyn_tire_model` | R `tanh` | `linear` / `tanh` / `pacejka` |
| `dyn_mu` | L+C 0.65 | 모델 타이어 μ. **자동 재생성됨**. 주행 중 set 은 반쪽만 먹음 |
| `ellipse_frac` | L+C 0.95 | 마찰원 여유 η (a_lim = μ·g·η) |
| `lm_dynamic` | R 3.0 | acados LM 정규화. 크면 QP 실패↓ |
| `nlp_solver_iters` | R 1 | 1 = SQP_RTI(실시간). >1 = SQP + globalization |
| `N_horizon` | R+C 40 | 예측 구간 길이. **캐시 자동 감지 안 됨** |
| `dT` | R+C 0.04 | 첫 스텝 간격 [s] (25 Hz). **캐시 자동 감지 안 됨** |
| `mpc_max_steering` | R 0.4 | 조향 계획 한계 [rad]. 실차 서보 실효 한계는 약 0.39 |
| `vehicle.l_wb` | R 0.307 | 휠베이스 [m] |
| `cost_spike_thr_live` | L 8000.0 | 이 코스트 넘으면 fallback. 정상 주행은 500~2000 |
| `alpha_steer_live` | L 1.0 | 조향 EMA. 1.0 = EMA 없음 (지연이 발진을 키워서 해제됨) |
| `actuation_latency_s` | L 0.12 | 서보 지연 보상 [s]. 맥 실측 120 ms |

### 4.8 GP 잔차 · 학습 ref_v

| 파라미터 | 현재값 | 설명 |
|---|---|---|
| `use_gp_casadi` | R+C true | 학습된 GP 사후평균을 동역학에 CasADi 로 구움 |
| `gp_casadi_ckpt` | R+C `~/bo_results/gp_residual_mac_hall0813.pt` | ckpt 경로·mtime 이 스탬프에 들어감 → 바꾸면 **자동 재생성** |
| `gp_casadi_train_data` | R `~/bo_results/res_mac_merged.pt` | 학습 데이터 |
| `use_learned_refv` | R false | 학습 ref_v 주입 (현재 npz 가 구맵 것이라 off) |
| `learned_refv_path` | R `.../maps/ifac/learned_refv_0716.npz` | |
| `use_refv_learning` / `refv_*` 9개 | R *(yaml 에 없음)* | 온라인 ref_v 학습. `use_refv_learning` 이 false 라 전부 비활성 |
| `sector_refv_spec` | R `''` *(yaml 에 없음)* | 섹터별 ref_v 배율 스펙 |

### 4.9 토픽 · 경로 (평소 안 건드림)

`odom_topic_name` `/car_state/odom` · `localized_pose_topic_name` `/car_state/pose` ·
`cmd_vel_topic_name` `/vesc/high_level/ackermann_cmd` · `track_name` `f` · `track_dir` `''`
— 전부 R. 앞의 3개와 `track_name` 은 **launch 가 덮어쓴다.**

---

## 5. yaml 에 없는 파라미터 30개

아래는 코드에 기본값만 있고 yaml 엔 안 적혀 있다. **동작은 하고 있고, `ros2 param set` 도 된다.**
튜닝하려면 yaml 에 줄을 추가하면 그때부터 관리 대상이 된다.

```
alat_scale_max 1.3        alat_scale_min 1.0        enable_sim_reset true(launch=false)
obs_pass_speed 3.3        overtake_enabled false    ot_shadow true
ot_bins_decay_on_lost false  ot_commit_gap 4.0      ot_cooldown 5.0
ot_engage_gap 6.0         ot_keepout 0.35           ot_launch_gap 1.5
ot_pass_margin 0.8        publish_zone_markers true save_params false
refv_learn_path ''        refv_lt_tol 0.02          refv_min_margin_m 0.35
refv_min_samples 2        refv_raise_cooldown 2     refv_scale_max 1.5
refv_scale_min 1.0        refv_step_delta 0.05      sector_refv_spec ''
slew_down_rate 5.0        use_opp_prediction true   use_refv_learning false
zone_edit_enable false    zone_yaml_path ''
```

전체 목록은 언제든 실행 중 노드에서 확인할 수 있다:

```bash
ros2 param dump /mpc_node        # 지금 노드가 가진 값 전부
ros2 param list /mpc_node        # 이름만
```

---

## 6. 함정 모음 (겪고 나서 적은 것들)

1. **캐시 경로**: `~/.acados_codegen/` 이다. `/tmp/...` 안내는 옛날 것.
2. **`max_speed` / `speed_target` / `N_horizon` / `dT` / `lookahead_m`** 는 바꿔도 자동 재생성되지 않는다. 캐시를 손으로 지워야 한다.
3. **launch `<param>` 이 yaml 을 이긴다.** §1 의 6개.
4. **`ros2 param set` 으로 찾은 값은 yaml 에 옮겨 적어야 남는다.** (`save_params true` 로 자동 저장 가능)
5. **`a_lat_safe_live` 를 μ·g·η 위로 올려도 무효** — 노드가 클램프하고 로그로 알려준다.
6. **`dyn_mu` / `max_speed` / `ellipse_frac` 은 주행 중 set 이 반쪽만 먹는다** (§2 참고).
7. **q_drate 60 은 맥 전용 발진 대책** — 심 BO 값(27.8)을 그대로 실차에 가져오면 위빙이 돌아온다.
8. **선언 안 된 키를 yaml 에 써도 ROS 는 조용히 무시한다.** 오타가 나도 에러가 안 난다.
   2026-08-18 에 그런 무효 키 13개를 제거했다 (`q_cte_live` 처럼 핵심처럼 보이는 것 포함).
   새 키를 넣었는데 안 먹으면 `ros2 param list` 에 있는지부터 확인할 것.

---

## 7. 참고

- 슬랙·codegen 가중치·장애물 상수의 상세 설명은 `config/mpc_tuning.yaml` 헤더 주석에 있다 (잘 정리돼 있음).
- 파이프라인 전반은 `nonlinear_mpc_acados/mpc_core/PIPELINE.md`.
- 2026-08-18 정리 내역: 커밋 `19717de`, 그 직전 복원지점 `e7e2e52`,
  삭제한 `.bak` 86개는 `~/mpcc_bak_archive_20260818.tar.gz`.
