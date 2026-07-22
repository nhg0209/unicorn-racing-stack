# State Machine & Spliner — 디버깅 / 튜닝 가이드

대상: `state_machine`, `spliner`(static/dynamic avoidance), `recovery_spliner`.
전제: 전 스택은 **Frenet(s, d)** 좌표 위에서 동작하며, 판단 입력은 `/car_state/odom_frenet`,
장애물은 `/tracking/obstacles`(동적) / `/tracking/raw_obstacles`(정적·미분류)에서 온다.

> ⚠️ 2026-06-29 P0 패치 반영분: `_check_close_to_raceline_heading`가 이제 **실제 헤딩정렬**을
> 검사한다(이전엔 `cur_d`를 라디안과 잘못 비교 → 사실상 무력). 그 결과 `RECOVERY/OVERTAKE →
> GB_TRACK` 복귀가 이전보다 엄격하다. 라인복귀가 안 되면 8장(주의)부터 보라.

---

## 0. 디버깅 원칙 — 항상 입력→상단 순서로

상태머신/스플라이너 거동이 이상하면 **위쪽(판단)부터 의심하지 말고 아래(입력)부터** 확인한다.
대부분의 "상태가 안 바뀐다 / 추월을 안 한다"는 사실 입력 단계 문제다.

```
1) 위치/속도   /car_state/odom_frenet  (s, d, vs 정상인가? d 부호/seam?)
2) 감지        /detect/raw_obstacles    (박스가 잡히나? 크기/위치 맞나?)
3) 분류/추적   /tracking/obstacles      (동적으로 분류되나? vs 값?)
4) 경로 생성   /planner/avoidance/*     (회피 wpnt가 나오나? 비어있나?)
5) 판단        /state_machine, /behavior_strategy (상태 전이/타겟)
6) 추종        /local_waypoints         (제어가 받는 최종 경로)
```

---

## 1. 진단 토픽 / 마커 치트시트

### state_machine
| 토픽 | 타입 | 용도 |
|---|---|---|
| `/state_machine` | String | **현재 상태 문자열** (가장 먼저 볼 것) |
| `/state_marker` | Marker | 상태 색구슬: 파랑=GB_TRACK, 빨강=OVERTAKE, 노랑=TRAILING, 자홍=ATTACK, 흰색=FTGONLY, 초록=RECOVERY |
| `/behavior_strategy` | BehaviorStrategy | local_wpnts + state + overtaking/trailing 타겟 (제어 입력) |
| `/local_waypoints` | WpntArray | 최종 추종 경로 |
| `/local_waypoints/markers` | MarkerArray | 위 경로 시각화(z=속도) |
| `/state_machine/trailing_target` | Marker | 노란 구 = 따라가는 대상 |
| `/state_machine/overtaking_target` | Marker | 파란 구 = 추월 대상 |
| `/ot_section_check` | Bool | 현재 추월 허용 섹터인가 |
| `/emergency_marker` | Marker | 배터리 저전압 경고 |
| `/state_machine/latency` | Float32 | 루프 주파수(measure:=true 시) |

### perception (detect + tracking)
| 토픽 | 용도 |
|---|---|
| `/detect/raw_obstacles` | detect가 매 프레임 내는 박스(분류 전) |
| `/detect/breakpoints_markers` | 클러스터 시작/끝 점 |
| `/detect/obstacles_markers_new` | 빨간 큐브 = 피팅된 박스 |
| `/detect/on_track_points` | 트랙 내부로 인정된 스캔점(measure 시) — GridFilter 검증용 |
| `/tracking/obstacles` | **동적 장애물**(is_static=False, vs 포함) |
| `/tracking/raw_obstacles` | 정적·미분류 |
| `/tracking/static_dynamic_marker_pub` | 색: 빨강=미분류, 초록=정적, 파랑=동적 |

### planner (avoidance / recovery)
| 토픽 | 용도 |
|---|---|
| `/planner/avoidance/otwpnts` | 동적 회피 경로(race.launch에선 lane_change_planner가 발행) |
| `/planner/avoidance/markers_sqp` | (lane_change) 동적 회피 경로 시각화 |
| `/spline_sample_points` | (lane_change) 샘플점 + 바운드 검사 |
| `/planner/avoidance/merger` | (lane_change) 회피→글로벌 블렌딩 구간 [s_end, 회피끝 s] |
| `/planner/avoidance/static_otwpnts` | 정적 회피 경로(static_avoidance_node) |
| `/planner/avoidance/markers` | 회피 경로 시각화 |
| `/planner/avoidance/considered_OBS` | 지금 회피 기준 잡은 장애물 |
| `/planner/avoidance/propagated_obs` | 예측 전파된 장애물 위치 |
| `/planner/recovery/wpnts` | 레이스라인 복귀 경로 |
| `/planner/avoidance/spline_samples` | 스플라인 샘플점(바운드 검사 결과) |

---

## 2. 빠른 헬스체크 (실행 직후 1분)

```bash
ros2 topic echo /state_machine --once                 # 상태 살아있나
ros2 topic hz /car_state/odom_frenet                  # ~80Hz 나오나
ros2 topic hz /tracking/obstacles                     # 동적 장애물 흐르나
ros2 topic echo /tracking/obstacles --field obstacles # vs / is_static 확인
ros2 topic hz /planner/avoidance/static_otwpnts       # 회피 경로 발행되나
ros2 topic echo /behavior_strategy --field state      # 판단 결과
ros2 param dump /state_machine                        # 적용된 파라미터 확인
```
RViz에서 `/state_marker`, `/tracking/static_dynamic_marker_pub`,
`/planner/avoidance/markers`, `/local_waypoints/markers`를 켜두면 대부분 눈으로 잡힌다.

---

## 3. 증상 기반 디버깅

| 증상 | 1순위 확인 | 흔한 원인 / 조치 |
|---|---|---|
| 장애물이 있는데 상태가 계속 GB_TRACK | `/tracking/obstacles` 비었는지 | 분류가 정적으로 빠짐 → tracking `max_std/min_std/min_nb_meas`. 또는 `interest_horizon_m`/`gb_horizon_m` 너무 짧음 |
| 추월을 절대 안 함 | `/ot_section_check` (false?) | `ot_sectors.yaml`에 추월 허용 섹터 없음. 또는 `_check_overtaking_mode` 미충족(회피경로 비었거나 free 아님) |
| 정적 장애물 추월만 안 됨 | `/planner/avoidance/static_feasible` 수신/신선도 | 플래너 죽음/미배선이면 게이트 fail-closed(0.5s stale). static_OT check 로그의 feasible/latest 항목 확인 |
| 회피경로가 비어서 나옴(otwpnts empty) | `/planner/avoidance/markers` | danger_flag: 트랙바운드에 너무 근접 → `spline_bound_mindist`↓ 또는 `evasion_dist`↓. 또는 라인 안 타서 쪽전환 불가 |
| 상태가 GB↔OVERTAKE 깜빡임(채터링) | `/state_machine` 빠르게 토글 | 히스테리시스 부족 → `overtaking_ttl_sec`↑, `splini_hyst_timer_sec`↑. OT 캐시 만료(2s) 확인 |
| 추월 후 라인 복귀 안 함(RECOVERY 고착) | heading 정렬 여부 | P0 패치로 heading 게이트 활성화됨 → `recovery_planner.yaml`의 `on_spline_*`, 헤딩 정렬 20°. 8장 참고 |
| TRAILING인데 멈춰버림 | `cur_vs`, FTG 카운터 로그 | `ftg_active=true`면 저속 지속 시 FTGONLY로 빠짐. 의도면 OK, 아니면 `ftg_active=false` |
| 회피가 너무 늦음/급함 | `/planner/avoidance/propagated_obs` | 예측 전파 부족 → `fixed_pred_time`↑. 너무 일찍이면 ↓ |
| 동적 장애물 vs가 튄다/0 | `/tracking/obstacles` vs 필드 | EKF 미수렴 → tracking `var_pub`(확신 임계), `process_var_*`. `vs_reset` 이하면 정적 강등됨 |
| 박스가 트랙밖 벽을 잡음 | `/detect/on_track_points` | GridFilter erosion 부족 → detect `filter_kernel_size`↑, `boundaries_inflation`↑ |
| 박스가 너무 잘게 쪼개짐/합쳐짐 | `/detect/breakpoints_markers` | `new_cluster_threshold_m`, `lambda_deg`, `sigma` |

---

## 4. state_machine 튜닝

라이브 변경: `ros2 param set /state_machine <name> <val>` (대부분 즉시 반영, on-set 콜백).
영구 저장: `state_machine_params.yaml` 수정 또는 rqt `save_params` 버튼(→ 같은 yaml에 기록).

### 기하 / 호라이즌
| 파라미터 | 기본 | 의미 | ↑ 하면 | 권장 시작 |
|---|---|---|---|---|
| `gb_ego_width_m` | 0.4 | "라인 위" 판정 횡거리(`_check_close_to_raceline`) | GB_TRACK 유지 잘됨/복귀 관대 | 차폭+여유 |
| `gb_horizon_m` | 15.0 | 전방 적("enemy in front") 탐지 거리 | 추월 더 일찍 고려 | 12~18 |
| `interest_horizon_m` | 20.0 | 관심 장애물 윈도우(s gap) | 더 먼 장애물도 판단 반영 | 15~25 |
| `overtaking_horizon_m` | 6.9 | 추월 모드 관련 호라이즌 | — | 그대로 |
| `emergency_break_horizon` | 0.5 | 비상정지 호라이즌 | — | 그대로 |

### 회피 free 판정 (막힘 민감도)
| 파라미터 | 기본 | 의미 | ↑ 하면 |
|---|---|---|---|
| `lateral_width_gb_m` | 0.3 | GB 경로 free 판정 횡폭 | 더 쉽게 "막힘" → 추월/트레일링 잦아짐 |
| `lateral_width_ot_m` | 0.3 | 추월 경로 free 판정 횡폭 | 추월 중 더 보수적 |
> 실제 free 검사는 `free_dist < lateral_width_m × scaling_factor`. `scaling_factor`는
> `gap / free_scaling_reference_distance_m`로 0~1 clip(멀수록 관대). 후자는 planner yaml에 있음.

### 추월 결정 임계 (P0에서 파라미터화)
| 파라미터 | 기본 | 의미 | 튜닝 |
|---|---|---|---|
| `static_ot_speed_mps` | (제거됨) | 정적 추월 속도 가드 — 라이브 설정에서 10.0으로 비활성이었고, 진입 속도는 회피 경로의 slow-in 프로파일이 담당하므로 삭제 | — |
| `getting_closer_rel_vel_mps` | -0.5 | (ego−상대) s속도 ≥ 이 값이면 "접근중" | 더 보수적이면 ↑(0), 관대하면 ↓ |

### 히스테리시스 / 채터링 억제
| 파라미터 | 기본 | 의미 |
|---|---|---|
| `overtaking_ttl_sec` | 3.0 | 추월 모드 이탈 전 유지 시간(프레임 카운트) |
| `splini_hyst_timer_sec` | 0.7 | 추월 쪽(좌/우) 전환 최소 간격 |
| `splini_ttl` / `pred_splini_ttl` | 2.0 | 회피경로 캐시 수명 (planner에 따라 택1) |
> 그 외 OT 캐시는 코드상 미사용 시 2초 후 강제 만료(stale 경로로 깜빡임 방지) — 고정.

### FTG (비상 회피)
| 파라미터 | 기본 | 의미 |
|---|---|---|
| `ftg_active` | false | FTG 탈출 활성화 |
| `ftg_timer_sec` | 3.0 | TRAILING 저속이 이만큼 지속되면 FTGONLY |
| `ftg_speed_mps` | 0.1 | 이 속도 미만을 "저속"으로 카운트 |
> 평상시 false 권장. 막혀서 못 빠지는 트랙에서만 켜라.

### 강제/안전
`force_GBTRACK`(추월 끔), `use_force_trailing`(충돌예측 강제 트레일링),
`volt_threshold`(11.0, 저전압 경고), `timetrials_only`(true면 장애물 무시).

---

## 5. 스플라인 생성 튜닝

> race.launch 기준: **정적**=`static_avoidance_node`(`/static_otwpnts`, 5.1),
> **동적**=`lane_change_planner`/`change_avoidance_node`(`/otwpnts`, 5.3), 복귀=`recovery_spliner`(5.2).
> 라이브 변경: `ros2 param set /<node_name> <name> <val>`.

### 5.1 apex 스플라인 핵심 (static_avoidance / spliner)
| 파라미터 | 기본 | 범위 | 의미 / 방향 |
|---|---|---|---|
| `evasion_dist` | 0.6 | 0.25~1.25 | 장애물에서 apex까지 횡거리. ↑=크게 비켜감(안전, 느림), ↓=아슬하게 |
| `spline_bound_mindist` | 0.30 | 0.05~1.0 | 트랙바운드 최소 여유. 이내로 붙으면 회피 포기(danger). ↓=공격적, ↑=잘 포기 |
| `obs_traj_tresh` | 1.0 | 0.1~1.5 | 레이스라인에서 이 횡거리 내 장애물만 회피 대상. ↓=라인 위 장애물만 |
| `pre_apex_dist0/1/2` | 4/3/2 | 0.5~8 | apex 이전 스플라인 제어점 s거리(클수록 완만) |
| `post_apex_dist0/1/2` | 4.5/5/5.5 | 0.5~12 | apex 이후 복귀 거리(클수록 완만 복귀) |
| `spline_scale` | 0.8 | 0.5~2.0 | 전체 스플라인 스케일 |
| `kd_obs_pred` | 1.0 | 0.1~10 | 예측 시 d 복원 게인(adaptive/heuristic 모드) |
| `fixed_pred_time` | 0.15 | 0~1.0 | constant 모드 예측 전파 시간[s]. ↑=상대 미래위치로 더 일찍 회피 |
| `post_min_dist`/`post_max_dist`/`post_sampling_dist` | 1.5/5.0/5.0 | — | 복귀 구간 샘플링 거리 |
| `kernel_size` | 8 | 1~20 | (정적) 바운드 검사 erosion |

튜닝 직관:
- **추월이 트랙밖으로 새거나 자주 포기** → `evasion_dist`↓, `spline_bound_mindist`↓
- **회피가 너무 과격/멀미** → `pre/post_apex_dist`↑(완만), `spline_scale`↑
- **회피 타이밍 늦음** → `fixed_pred_time`↑ (단 너무 크면 헛회피)
- 코드상 **코너 바깥쪽 추월이면 자동 1.75× 완만화 + 속도 0.9×** (하드코딩, 의도된 동작)

### 5.2 recovery_spliner
| 파라미터 | 기본 | 의미 |
|---|---|---|
| `spline_scale` | 0.8 | 접선 합류 곡률 스케일 |
| `smooth_len` | 1.0 | 현재 위치를 헤딩방향으로 미리 당겨 합류를 부드럽게(`find_tangent_idx`) |
| `n_loc_wpnts` | 80 | 생성 포인트 수 |
> recovery 경로는 상태머신에서 `vel_planner_safety_factor=0.5`로 보수적 속도 재계산됨.

### 5.3 동적 추월 — lane_change_planner (change_avoidance_node) **[2026-07 lane-hold 재작성]**

race.launch의 **동적 추월 기본 플래너**. 예전의 "장애물 스냅샷 구간에 cosine ramp" 방식은
상대가 움직이면 경로가 매 사이클 재앵커링되며 차가 휘청거렸다. 현재는 **페이즈 기반
lane-hold 추월**로 동작한다 (설정: `stack_master/config/lane_change_params.yaml`):

- **IDLE**: 갭이 `engage_gap_m`(5.0)으로 좁혀질 때까지 플래너는 **침묵** — 접근은
  TRAILING(컨트롤러 gap PID, `0.25·v+1.55m`에서 안정화) 또는 레이스라인 주행이 제
  속도로 담당. 좁혀지면 추월섹터(`/ot_section_check`) 확인 후 타깃 래칭 + 사이드 1회
  선택(진행 중 재투표 없음 → 좌우 플래핑 제거). **레인 오프셋은 상대의 실제 횡위치에서
  역산**: 상대 중심에서 `size/2+sep_margin_m` 이상(=SM free-check 통과 보장),
  `lane_offset_m`은 하한, 벽(회랑−차폭/2−`spline_bound_mindist`)이 상한.
- **OPEN**: 현재 차량 위치(헤딩 e_psi 반영)에서 quintic 블렌드로 **센터라인 평행 레인**
  (센터라인 d(s) 테이블 + 부호×오프셋)에 진입. 타깃 바로 뒤에서 engage하면 블렌드가
  **타깃 도달 전에 끝나도록 캡**(미완성 블렌드가 free-check를 깨는 것 방지). 레인 도달 시 HOLD.
- **HOLD**: 상대 위치와 무관하게 **레인을 계속 추종**(경로가 상대를 쫓아다니지 않음).
  상대가 레인 쪽으로 파고들면 오프셋이 슬루(`offset_slew_mps`) 제한으로 **자동 확장**
  (벽 캡) — 접근 중 SM free-check가 깨져 OVERTAKE→TRAILING을 반복하며 "스플라인만
  따라가다 느려지던" 루프의 해결책. 타깃은 id+최근접 s로 연속 추적, 미검출 시 EKF
  속도로 coast. 랩 시임을 고려한 signed Δs가 `pass_gap_m` 이상 & 상대가 다시
  빨라지지 않음이 `pass_hyst_s` 지속되면 CLOSE.
- **CLOSE**: `close_arm_m` 앞에서 시작하는 **래칭된 복귀 램프**로 레이스라인 복귀.
  차가 아직 레인 위면 램프 시작점이 차 앞으로 슬라이드(상태머신이 언제 채택해도 횡 스텝 0).
  복귀 회랑에 장애물이 있으면 시작점을 그 뒤로 미룬다. 상대가 재추월하면 HOLD로 복귀.
- 경로는 매 사이클 차량 앵커로 재발행하지만 **레인 기하는 고정**이라 SM 캐시 스플라이싱과
  무충돌. 경로 길이 `hold_horizon_m`(22m) > SM `interest_horizon_m`(20m) 유지 필수
  (상대가 경로 끝을 넘으면 SM free-check가 즉시 실패하는 구조 때문).
- 벽 처리: 레인을 트랙 회랑(`d_left/right - 차폭/2 - spline_bound_mindist`)으로 클립,
  `GridFilter`(erosion 3)로 최종 검사. 속도는 계속 vx=0으로 발행 — SM `update_velocity`가
  곡률 기반으로 재계산(변경 없음).

**주요 파라미터** (라이브: `ros2 param set /planner_change <name> <val>`,
영구: `stack_master/config/lane_change_params.yaml`):
| 파라미터 | 기본 | 의미 / 방향 |
|---|---|---|
| `engage_gap_m` | 5.0 | 이 갭까지는 TRAILING이 제 속도로 접근(플래너 침묵). 트레일링 유지거리(0.25·v+1.55≈2~3m)보다 크고 SM 커밋 게이트(10m)보다 작아야 함 |
| `lane_offset_m` | 0.35 | 레인 오프셋의 **하한**(실제 오프셋은 상대 위치에서 역산·자동 확장) |
| `sep_margin_m` | 0.50 | 장애물 half-size 초과 요구 이격. **SM의 0.4(ego/2+lateral_width) 미만 금지** |
| `offset_slew_mps` | 0.6 | 오프셋 실시간 확장/축소 속도. 상대가 요동치면 ↑ |
| `pass_gap_m` / `pass_hyst_s` | 1.2 / 0.3 | 추월 완료 판정 리드/지속시간. 복귀가 성급하면 ↑ |
| `open/close_ramp_min_m`, `*_time_s` | 3.0 / 0.8 | 램프 길이 = max(min, t·v). 과격하면 ↑ |
| `close_arm_m` | 1.0 | 복귀 램프가 차 앞에서 시작하는 거리 |
| `hold_horizon_m` | 22.0 | 발행 경로 길이(20 미만으로 줄이지 말 것) |
| `target_lost_s` | 1.0 | 타깃 coast 허용 시간(폐색 강건성) |
| `obs_traj_tresh` | 1.0 | 래칭 시 |상대 d − ego d| 필터 |

튜닝 직관:
- **추월을 아예 시작 안 함** → ① `ot_sectors.yaml`의 `ot_flag` 확인(§6, 최다 원인)
  ② 갭이 `engage_gap_m`까지 안 좁혀짐: 트레일링 유지거리(`trailing_gap`/`trailing_vel_gain`)가
  engage_gap보다 크지 않은지 확인 ③ 레인 선택 실패(양쪽 벽 한계 초과): 로그에서
  `IDLE -> OPEN`이 아예 없으면 `sep_margin_m`↓(0.42 미만 금지) 또는 트랙이 물리적으로 좁은 것
- **접근 중 OVERTAKE↔TRAILING 반복(따라가다 느려짐)** → 오프셋 자동확장이 벽에 캡된 경우.
  `spline_bound_mindist`↓로 벽 여유를 조금 양보하거나 그 구간 추월 포기(섹터 제외)
- **복귀가 늦음/오래 레인에 머묾** → `pass_gap_m`↓, SM `dynamic_avoidance_planner.yaml`
  `hyst_timer_sec`(1.5) 확인 — CLOSE 채택 지연의 상한이다
- **진입이 과격** → `open_ramp_time_s`↑ (단 타깃 직전 engage면 갭 캡이 우선); **복귀가 과격** → `close_ramp_time_s`↑
- **좌우 플래핑이 다시 보이면** 페이즈 로그(`[LaneChange] IDLE -> OPEN ...`) 확인:
  래칭 후엔 사이드가 안 바뀌는 게 정상, 바뀐다면 abort→재engage 반복이므로 abort 원인
  (`lane no longer viable`/`blocked`) 로그를 볼 것
> 주의: 이 노드는 `merger`만 내고 `fail_trailing`은 발행하지 않는다(상태머신은 토픽을 구독하지만
> 기본 race.launch에선 publisher 없음 → 항상 False). 정적/동적이 동시에 후보면 상태머신이
> `_check_static_overtaking_mode`(저속) vs `_check_overtaking_mode`(동적)로 택일한다.
> RViz 디버그: `/planner/avoidance/lanes`(두 레인), `/planner/avoidance/markers_sqp`(경로+페이즈 텍스트).

동반 노드 `waypoint_updater`(`update_waypoints`)는 `/global_waypoints_updated`를 발행한다
(현 플래너는 미사용, 다른 플래너·기록용으로 유지).

---

## 6. 섹터 설정 (맵별, 코드 아님)

`stack_master/maps/<map>/`:
- `ot_sectors.yaml` — `Overtaking_sectorN.ot_flag=true`인 구간에서만 추월 허용
  (`ot_sector_begin` = 추월 시작 마진). 이게 없으면 `/ot_section_check`가 항상 false → 추월 0.
- `speed_scaling.yaml` — `SectorN.only_FTG=true`인 구간은 무조건 FTGONLY (위험 구간 안전화).
> 둘 다 `sector_tuner` 노드로 런타임 변경 가능하며 상태머신이 ParameterEvent로 실시간 반영.
> "추월을 아예 안 한다"의 가장 흔한 원인이 **추월 섹터 미설정**이다.

---

## 7. 권장 튜닝 워크플로우

1. **타임트라이얼 먼저**: `timetrials_only:=true` 또는 장애물 없이 글로벌 라인/속도부터 안정화.
2. **감지/분류 검증**: 가상상대(sim) 또는 박스를 두고 `/tracking/obstacles`에서 **동적으로
   분류되고 vs가 맞는지** 확인. 안 되면 tracking `max_std/min_std/min_nb_meas/var_pub`.
3. **섹터 설정**: 추월할 구간에 `ot_flag=true`, 위험구간 `only_FTG=true`.
4. **추월 발동 여부**: `gb_horizon_m`/`interest_horizon_m`로 "언제 고려", `lateral_width_*`로
   "얼마나 쉽게 막힘 판정". `/state_machine`이 OVERTAKE로 들어가는지.
5. **회피 형상**: spliner `evasion_dist`/`spline_bound_mindist`/`*_apex_dist`로 경로 다듬기.
   `/planner/avoidance/markers`로 눈으로 확인.
6. **안정화**: 채터링 보이면 `overtaking_ttl_sec`/`splini_hyst_timer_sec`↑.
7. **복귀**: 추월 후 RECOVERY→GB_TRACK 매끄러운지. heading 정렬/`on_spline_*` 조정.
8. **sim→real**: `racecar_version`(SIM/CAR) 차량동역학(ggv) 다름. real에서는 보수적으로
   (`evasion_dist`↑, `spline_bound_mindist`↑) 시작 후 점진 공격화.

---

## 8. P0 패치 관련 주의 (heading 게이트)

`_check_close_to_raceline_heading(20)`가 이제 **헤딩오차 < 20°** 를 실제로 본다(wrap 정규화 포함).
- 영향: `RECOVERY/OVERTAKE/TRAILING/FTGONLY/START → GB_TRACK` 전환의
  `close_to_raceline(0.05) × heading(20)` 게이트.
- 증상이 "복귀가 너무 안 된다"면 일시적으로 임계를 키워 확인:
  코드에서 호출부 `_check_close_to_raceline_heading(20)`의 인자를 30~40으로 올려보고,
  맞으면 그 값으로 확정(아직 파라미터화 안 됨 — 필요하면 P1에서 yaml로 뺄 것).
- 복귀가 의도대로 엄격해진 것이면 정상. 이전엔 헤딩이 틀어져도 라인 근처면 GB로 튀었었다.

---

## 9. measure 모드 & 로그

- `measure:=true`로 각 노드 latency 토픽 활성화(`/state_machine/latency`,
  `/detect/latency`, `/tracking/latency`, `/planner/avoidance/latency`).
  루프 주파수가 목표(state 80Hz, detect/tracking 40Hz, spliner 20Hz) 대비 떨어지면 CPU/DDS 의심.
- `from_bag:=true`로 bag 재생 디버깅(파라미터 콜백 비활성 등 일부 노드 동작 변경).
- 상태/카운터 로그는 노드 stdout(`output="screen"`)에 throttle로 찍힌다(FTG 카운터, free False 사유 등).

---

## 부록 — 빠른 참조

- 상태 enum: GB_TRACK / TRAILING / OVERTAKE / FTGONLY / RECOVERY / START / LOSTLINE / ATTACK
- 루프 레이트: state_machine 80Hz, detect/tracking 40Hz, spliner 20Hz, carstate 80Hz
- free 검사식: `free_dist = 횡거리 − 장애물크기/2 − 차폭/2`,
  `막힘 ⇔ free_dist < lateral_width_m × clip(gap/free_scaling_reference_distance_m, 0, 1)`
- 동적 장애물 free 검사는 예측궤적의 **TTC~TT0 시간창**만 본다.
