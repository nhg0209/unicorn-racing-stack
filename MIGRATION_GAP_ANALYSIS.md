# 스택 현황 & 잔여 갭 분석 (2026-07-22 현행화)

> **이 문서의 2026-06-25 버전은 폐기되었다** (내용은 git 이력 참조). 당시 "치명적 통합 결함"
> (perception Frenet 미기입 / 컨트롤러 단순 PP / 트래킹 스텁)은 7월 커밋들로 **전부 해소**되었고,
> 이 버전은 2026-07-22 전수 코드 분석 + 예측 파이프라인 배선 감사 기준으로
> **실제로 남아있는 갭**만 담는다. (판정 기호: 🔴 미작동/끊김, 🟡 부분, ⬛ 의도적 제외)

---

## 1. 이전 버전 치명 결함 — 해소 확인 (요약)

| 6/25 지적 | 현재 상태 | 근거 |
|---|---|---|
| perception이 Frenet(s,d)을 안 채워 실차 planning 전멸 | ✅ 해소 | `perception/src/detect.cpp` 자체 FrenetConverter로 s/d 기입, `kiss_obstacle_bridge.py`(Livox 3D 경로)도 동일. virtual_perception 없이 실차 동작 |
| 트래킹이 1프레임 스텁 | ✅ 해소 | `perception/scripts/multi_tracking.py` — Frenet EKF `[s,vs,d,vd]`, NN 데이터연관+TTL, 정적/동적 분류(강등 가드·near-dynamic 억제 포함) |
| 컨트롤러가 단순 PP 6파라미터 | ✅ 해소 | `controller_manager` = L1/PP + heading PID + trailing 갭 PID + future position + 속도/조향 스케일링 + AEB + FTG 멀티플렉싱 (`controller/controller/combined/src/Controller.py`). 대안 `pp_heading_controller`(마찰원 PP)도 추가 |
| GP 예측 미가동 | ✅ 가동 | `gp_traj_predictor` 3노드가 race.launch.xml:152-159에서 기동 (단, **활용도는 §2 참조**) |
| dyn reconfigure 미작동 | ✅ 대부분 해소 | detect / tracking / controller_manager / state_machine / sector·ot tuner 전부 런타임 콜백 + save_yaml 보유. 플래너 4종만 저장 미지원(§3) |
| 섹터 live 연동 끊김 | ✅ 해소 | SM이 ParameterEventHandler로 sector/ot tuner 변경 실시간 반영 (`state_machine_node.py:490-530`) |

---

## 2. 예측 파이프라인 — 배선/활용 갭 (2026-07-22 감사) 🔴

생산: `opp_prediction.py`가 `/opponent_prediction/obstacles_pred`(PredictionArray, **200스텝 × dt 0.02s = 4s** 호라이즌, `pred_s/pred_d`만 기입)를 발행.
상대가 자기 라인 위(±0.25m)이고 반 랩 이상 관측되면 **GP 라인 추종 전파**, 아니면 **등속 전파 + force_trailing** 브랜치.

### ✅ 2026-07-22 수리 완료 (P0 배선 수리)

| # | 갭 | 수리 내용 |
|---|---|---|
| 2-1 | GP 예측이 상대가 **0.6 m 이내**일 때만 발행(그 외 빈 배열) | 근접 게이트 제거 — 상대가 자기 라인 위면 **상시 발행**. begin=현재 위치, end=호라이즌 끝 (`opp_prediction.py` GP 브랜치) |
| 2-2 | force_trailing 삼중 사망(토픽 불일치+미소비+`not` 반전) | SM 구독을 `/opponent_prediction/force_trailing`으로 정정, `not` 제거, `_check_overtaking_mode` **진입 게이트**로 소비(지속성 게이트는 미적용 — 상대 옆 이탈 방지). `use_force_trailing`(기본 false) 옵트인 |
| 2-3 | SM TTC 윈도우 dt 불일치(0.05 vs 생산 0.02 → 시간창 40%만 검사) | `pred_dt = 0.02` 정합 (`_check_free_frenet`) |
| 2-6 | `/mpc_controller/ego_prediction` 완전 사망(publisher 없음+저장만) | SM 구독/콜백/변수 제거 |
| 2-7 | 루프 페이싱 파손(spin/sleep 도달 불가, 마커 frenet 서비스 200회/사이클) | `spin_once` + wall-clock 10 Hz 페이싱, 마커 변환 **1회 배치 호출**(`_pred_markers`), pred 배열 사이클당 1회 발행으로 정리 |

주의(수리의 파급): SM의 동적 free 검사가 이제 **실제로 GP 예측 궤적**을 검사한다(이전엔 예측이 거의 비어
현재위치 폴백이 대부분). OVERTAKE 진입/유지 판정 성향이 바뀌므로 sim 재검증 필요. TTC가 예측 호라이즌
(200×0.02=4 s)을 넘는 장애물은 예측창이 비어 free 취급되는 기존 구조는 유지(진입 게이트 `_check_getting_closer(10 m)`가 상황을 한정).

### 남은 항목

| # | 갭 | 상세 | 근거 |
|---|---|---|---|
| 2-4 | **활성 동적 플래너의 예측 소비 최소** | `change_avoidance_node`(lane_change)는 `/opponent_trajectory` 평균 d(s)만 사이드 선택에 사용(`_opp_d_band`). `obstacles_pred`·GP 불확실성(`d_var/vs_var`)·재가속 예측 미사용 → **P1/P2에서 이식** | `change_avoidance_node.py:199,447-458` |
| 2-5 | `/opponent_prediction/obstacles` 활성 소비자 없음 | 휴면 플래너 2종만 구독 — P2의 입력 후보 | `dynamic_avoidance_node.py:162`, `sqp_avoidance_node.py:95` |
| 2-8 | 트레일링 타겟은 예측 미사용 | 컨트롤러 갭 PID는 SM이 넘긴 **현재 관측** Obstacle(s/vs)만 사용. 예측은 free 판정 bool에만 기여 | `controller_manager.py:281-288`, `state_machine_node.py:957` |
| 2-9 | `/init_opp_trajectory` 서비스 호출자 없음 | ego 라인으로 상대 궤적 시드하는 경로 미사용 | `opp_prediction.py` 서비스 서버 |
| 2-10 | `save_distance_front` 파라미터 무기능화 | 2-1 수리로 게이트 용도 소멸 — 선언만 남음. P1에서 재활용 또는 제거 | `opp_prediction.py:61` |

부가: `/opponent_trajectory`는 상대를 **반 랩** 관측해야 처음 생성되고(`opponent_trajectory.py:268-269`),
그 전까지 `opp_prediction`은 시작 시 `wait_for_message`로 **블로킹**(`opp_prediction.py:281`).
OVERTAKE 중에는 궤적 갱신 동결(`opponent_trajectory.py:60-64`).

---

## 3. 기타 잔여 갭

| 항목 | 상태 | 비고 |
|---|---|---|
| MAP(steering lookup LUT) 조향 | 🔴 미복원 | `TODO/system_identification/steering_lookup`에 차량별 Pacejka LUT csv 보존, COLCON_IGNORE. 현재 L1/PP로 대체 중 |
| 플래너 4종 튜닝값 저장 | 🟡 | static_avoidance/lane_change는 startup yaml 로드 O·save X, recovery/start는 코드 기본값뿐. 라이브 튜닝값이 재시작 시 증발 — sector_tuner 패턴(save_yaml) 복제 필요 |
| `planner/avoidance/fail_trailing` | 🔴 배선 없음 | SM 구독하나 publisher 없음(항상 False). `state_machine_node.py:737` |
| ATTACK 상태 | 🟡 도달 불가 | 어떤 전이도 반환하지 않음 (`state_transitions.py`) |
| state_indicator(LED) / car_to_car_sync | ⬛ 제외 유지 | 정책상 제외 |

---

## 4. 휴면 코드 인벤토리 (재활용 후보)

| 위치 | 내용물 | 재활용 가치 |
|---|---|---|
| `planner/spliner/spliner/spliner_node.py` | 자체 장애물 전파 4모드(`_predict_obs_movement:525-598`): constant / **adaptive**(도달시간 리드 + d의 지수적 라인복귀 이완, `kd_obs_pred`) / adaptive_velheuristic / heuristic | **1순위** — GP 비의존·저비용·우아한 열화. 활성 lane_change에 이식 적합 |
| `planner/sqp_planner/sqp_avoidance_node.py` | ① 예측 장애물 배열을 직접 회피 대상으로 사용(`obs_prediction_cb:202-206`) ② **GP d(s) 프로파일을 동적 장애물 회랑 중심**으로 사용(`sqp_solver:383-395`) ③ SLSQP d-프로파일 최적화 | ②는 이식 가치 높음 (버그 주의: `:392` s[m]를 waypoint 개수로 mod) |
| `planner/spliner_planner/dynamic_avoidance_node.py` | 예측 기반 apex 배치 구조 | 🟡 예측 대입이 `len==20` 게이트(생산자는 200)로 도달 불가(`:309`) — 사실상 사장 |
| `prediction/.../predictor_opponent_trajectory.py` | 상대가 라인 이탈 시 **라인 복귀 곡선 GP**(Matern, `vs_var=69` 센티널) | 런치 시 `/opponent_trajectory` 경합 주의(GP 노드와 동일 토픽) — 아이디어만 이식 권장 |
| `TODO/system_identification/` | steering_lookup(LUT 다수) + id_controller | MAP 조향 복원 시 필수 재료 |
| `state_estimation/state_estimation/` | 구 ETH 융합 레이어(carstate_node) | 현 스택에선 대체 완료 — 삭제 후보 |

---

## 5. 권장 순서 (동적 추월 = 개발 중 전제)

1. **배선 수리(§2-1,2-3)**: GP 브랜치 발행 게이트 완화 + SM dt 0.02 정합 — 코드 몇 줄로 기존 GP 파이프라인이 살아남.
2. **lane_change에 예측 주입(§2-4)**: spliner의 adaptive 전파(§4) 이식 → engage 타이밍/오프셋 선행 확장, sqp의 GP-회랑 아이디어로 pass window 동안 상대 예상 d(s) 기반 오프셋 산정.
3. **죽은 배선 정리(§2-2,2-5,2-6,2-9)**: 쓸 거면 잇고, 안 쓸 거면 구독/발행 제거.
4. 플래너 save_yaml(§3) + opp_prediction 루프 구조 정리(§2-7)는 병행 가능.
