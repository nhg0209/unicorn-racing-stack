# ROS1 → ROS2 마이그레이션 갭 분석 (실제 기능 기준)

> `MIGRATION_LOG.md`는 "빌드/임포트 통과 = 완료"로 보고 있으나, **빌드가 되는 것과 ROS1 기능이
> 그대로 살아있는 것은 다르다.** 이 문서는 `origin/ros1` 원본과 현재 워킹트리(fix/HJ)를
> 코드 레벨로 비교해, **무엇이 실제로 누락/변경되었는지**와 **dynamic reconfigure가 모듈별로
> 진짜 작동하는지**를 정리한다. (수정은 하지 않음. 분석/계획 전용.)

판정 기호: ✅ 충실 포팅 / 🟡 부분(빌드는 되나 기능 누락) / 🔴 다른 구현·미작동 / ⬛ 의도적 제외

---

## 0. 가장 치명적인 통합 결함 (먼저 읽을 것)

### (a) Perception이 Frenet(s,d)을 채우지 않는다 → 실차에서 planning 전체가 굶는다 🔴
- **ROS1**: `detect.py`와 `multi_tracking.py`가 **자체적으로 `FrenetConverter`를 들고** 장애물의
  `s_center/d_center/vs/vd`를 채워서 `/tracking/obstacles`로 발행했다.
  (근거: `origin/ros1:perception/scripts/detect.py:277` `get_frenet(...)`,
  `multi_tracking.py:420,645,675` 등)
- **ROS2 현재**: `detect_ros.py`→`/detections`(센서 프레임 x/y), `tracking_ros.py`→`/tracking/obstacles_raw`
  (map 프레임, **Cartesian만**). Frenet 필드는 0으로 둠
  (`tracking_ros.py:201` 주석 "so the merger can fill Frenet", 필드 세팅 `:214-227`).
- Frenet을 실제로 채우는 노드는 `virtual_perception/tracking_merger.py:91-116` 인데,
  **`virtual_perception`은 SIM 전용 패키지**(`low_level.launch.xml:53-55`는 sim 분기에서만 include).
- 결과: **SIM에서는** detect→track→merger(Frenet 채움)→predictor→planner 가 이어지지만,
  **실차에서는 `tracking_merger`가 안 떠서 `/tracking/obstacles`가 발행되지 않음** →
  `opponent_predictor`, `state_machine`, `spliner/spliner_planner/sqp_planner/lane_change_planner`가
  전부 입력을 못 받아 굶는다. (근거: `headtohead.launch.xml:42-59`, planner 구독 토픽은 §5 참조)

### (b) Controller가 ROS1 컨트롤러가 아니다 (CAC 단순 PP로 대체됨) 🔴
- ROS1 컨트롤러는 MAP(steering lookup 기반) + Pure-Pursuit 하이브리드 + heading PID + trailing(추종) +
  future-position 예측 + 속도/곡률/횡오차 스케일링 + L1 적응형 + AEB + FTG 멀티플렉싱 + 40여개 dyn 파라미터.
- ROS2 현재(`controller_ros.py`,`PP.py`)는 **순수 Pure-Pursuit 1종 + 하드코딩 파라미터 6개**.
  trailing/MAP/future/scaling/AEB/dyn-reconfig 전부 없음. `gapfollow/wallfollow/estop`은 따로 떠있을 뿐
  컨트롤러에 멀티플렉싱되지 않음.
- `steering_lookup`은 `TODO/`에 `COLCON_IGNORE`로 빠져 있고, **ROS2 어디서도 import 안 함**
  (전 트리 grep 결과 0건) → 빌드는 안 깨지지만 **MAP 조향이 통째로 사라진 것**이 확정.

### (c) Perception(detect/track/predict)이 다른 구현으로 대체됨 🔴
- ROS1: 각도점프 클러스터링 + **L-shape 사각 피팅** + Frenet 필터링(detect.py),
  **4-state EKF 다물체 트래킹** + static/dynamic 분류 + TTL/데이터연관(multi_tracking.py),
  별도 패키지의 **GP(가우시안 프로세스) 궤적 예측**(`gp_traj_predictor`).
- ROS2: 단순 데카르트 클러스터링 + AABB(피팅 없음), **칼만 없는 1프레임 트래킹 스텁**
  (`tracking.py`의 `_associate()`는 빈 스텁, 매 프레임 트랙 재생성),
  예측은 **등속(constant-velocity)** 1줄짜리. → ROS1 perception의 핵심 기능 대부분 미구현.

---

## 1. Controller 🔴 (가장 큰 작업)

| 기능 (ROS1) | ROS2 | 상태 | 근거 |
|---|---|---|---|
| MAP 모드 (steering LUT) | 없음 | 🔴 MISSING | `controller_ros.py`에 lookup import 0건 |
| Pure-Pursuit | 있음(단순) | 🟡 CHANGED | `PP.py:92-169` (CAC 스타일) |
| heading PID 보정 | 없음 | 🔴 | ROS1 `Controller.py:544+` |
| trailing(추종) 컨트롤 | 없음 | 🔴 | ROS1 `Controller.py:346` — head-to-head 불가 |
| future position 예측 + IMU 융합 | 없음 | 🔴 | ROS1 `Controller.py:181` |
| 속도 스케일링(곡률/횡오차/heading/가속) 5종 | 없음 | 🔴 | ROS1 `Controller.py:380+` |
| L1 적응형 lookahead | 하드코딩 | 🟡 | ROS2는 `0.5+0.3v` 선형뿐 |
| 조향 rate limit / AEB | 없음 | 🔴 | ROS1 `Controller.py:161,297` |
| FTG 멀티플렉싱(FTGONLY) | 분리됨 | 🔴 | gapfollow가 통합 안 됨 |
| dynamic reconfigure (40+ param) | 없음 | 🔴 | `l1_params_server.py` 미포팅 |
| 시각화 마커 5종 | lookahead만 | 🟡 | |

난이도: **상**. trailing·MAP·future가 head-to-head 레이싱의 핵심.

---

## 2. Perception (detection / tracking / prediction) 🔴

| 항목 | ROS1 | ROS2 | 상태 |
|---|---|---|---|
| 클러스터링 | 각도점프+Frenet인지 | 단순 데카르트 | 🟡 CHANGED |
| L-shape 사각 피팅 | 있음 | AABB only | 🔴 MISSING |
| 출력 좌표계 | Frenet(s,d) | 센서/맵 Cartesian | 🔴 CHANGED (→§0a) |
| 트래킹 | 4-state EKF 다물체 | 1프레임 스텁 | 🔴 MISSING |
| static/dynamic 분류 | std+투표 | 없음 | 🔴 |
| 데이터 연관 / TTL | 있음 | 빈 스텁 | 🔴 |
| 예측 | GP 회귀 | 등속 | 🔴 CHANGED |
| dyn reconfigure(23 param) | server+cfg+yaml | 없음 | 🔴 |

난이도: **상**. 트래킹(EKF) 복원이 안정성에 가장 큰 영향.

---

## 3. State Machine 🟡 (코어는 살아있음)

- 상태 8종(GB_TRACK/TRAILING/OVERTAKE/FTGONLY/RECOVERY/ATTACK/START/LOSTLINE)·전이·행동·vel_planner: ✅ 충실 포팅.
- dyn reconfigure: ✅ **작동** (`state_machine_params.py:241-281` 콜백 + `config/state_machine_params.yaml` 로드).
- 🟡 **state_indicator 노드(LED) 미포팅** (`blink1` 의존; ⬛ 제외 정책과 연동).
- 🟡 **섹터/추월존 live 갱신 불가**: 현재는 init 때 1회만 읽음(`state_machine.py:143-147,415-432` 주석
  "sector tuner 포팅되면 live로"). sector_tuner 자체는 떠 있으나 SM과의 live 연동이 끊김.

---

## 4. Planner (local/overtaking) 🟡

5개 패키지(spliner / spliner_planner / sqp_planner / recovery_spliner / lane_change_planner) 모두
**알고리즘 노드는 포팅됨**. 단:
- 🔴 **`dynamic_*_server.py` 5개 전부 미포팅**, `.cfg` 5개 전부 없음.
- 🟡 dyn reconfigure 런타임 콜백은 노드 inline으로 **있음**(아래 표) — 단 **YAML 저장(save_params)과
  startup YAML 로드가 없음** → 튜닝값이 재부팅 시 사라짐(ROS1은 stack_master/config/*.yaml로 저장했음).
- 의존성(frenet_conversion, grid_filter, tph, ccma)은 포팅 OK.

---

## 5. Dynamic Reconfigure 모듈별 실제 작동 검증 (사용자 핵심 요구)

"파일 존재"가 아니라 **(A)파라미터 선언 (B)런타임 콜백이 실제 반영 (C)YAML 저장 (D)startup YAML 로드**
4가지를 모두 만족해야 "작동"으로 본다.

| 모듈 | (A)선언 | (B)런타임 콜백 | (C)YAML 저장 | (D)YAML 로드 | 종합 | 근거 |
|---|:--:|:--:|:--:|:--:|---|---|
| controller | ✗ | ✗ | ✗ | ✗ | 🔴 BROKEN | `controller_ros.py:44-50` (6개만), 콜백 0건 |
| perception(detect/track) | 일부 | ✗ | ✗ | ✗ | 🔴 BROKEN | `detect_ros.py:58-72`,`tracking_ros.py:58-69` |
| spliner | ✓ | ✓ | ✗ | ✗ | 🟡 PARTIAL | `spliner_node.py:123-165, 168-230` |
| spliner_planner | ✓ | ✓ | ✗ | ✗ | 🟡 PARTIAL | `dynamic_avoidance_node.py:115-122,198-209` |
| sqp_planner | ✓ | ✓ | ✗ | ✗ | 🟡 PARTIAL | `sqp_avoidance_node.py:116-131,162-192` |
| recovery_spliner | ✓ | ✓ | ✗ | ✗ | 🟡 PARTIAL | `recovery_spliner_node.py:152` |
| lane_change_planner | ✓ | ✓ | ✗ | ✗ | 🟡 PARTIAL | `change_avoidance_node.py:171,178-191` |
| state_machine | ✓ | ✓ | (서버 의존) | ✓ | ✅ WORKS | `state_machine_params.py:241-281`, `config/state_machine_params.yaml` |
| sector_tuner | ✓ | ✓ | ✓ | ✓ | ✅ WORKS | `sector_tuner.py:70-157` (save_yaml 포함) |
| overtaking_sector_tuner | ✓ | ✓ | ✓ | ✓ | ✅ WORKS | `ot_interpolator.py:64-179` |

→ **dyn reconfigure 작동 기준 미달: controller·perception(완전 미작동), planner 5종(저장/로드 누락).**
   `sector_tuner`/`overtaking_sector_tuner`가 "골드 스탠다드" 패턴(콜백+save_yaml+yaml 로드) → 나머지가 이걸 따라야 함.

---

## 6. 지원 라이브러리 / 글로벌 플래너

| 항목 | 상태 | 비고 |
|---|---|---|
| grid_filter | ✅ | rospy→rclpy 포팅, 소비자 6곳 `node=self` 패치 완료 |
| polygon_filter | ⬛ EMPTY-SHELL | C++만, ROS1에도 소비자 없음 → 손실 없음 |
| frenet_conversion(+_msgs) | ✅ | lib/msgs 분리 |
| frenet_odom_republisher, vel_planner, lap_analyser, set_pose, random_obstacle_publisher | ✅ | |
| gb_optimizer(글로벌) | ✅ | grid_filter 비의존 확인 |
| steering_lookup | 🔴 IGNORED-NOT-BUILT | `TODO/`에 COLCON_IGNORE. controller가 안 쓰니 빌드는 OK지만 MAP 복원 시 필요 |
| car_to_car_sync | ⬛ 제외 | 멀티카 동기화. 정책상 제외 |
| state_indicator | 🟡/⬛ | LED(blink1) 의존. 제외 정책 검토 |

---

## 7. 권장 단계별 마이그레이션 계획

원칙: 기능 죽이지 않기. **아래에서 위로** — 데이터(perception)가 막히면 위(planner/controller) 검증이
불가능하므로 perception 시임부터.

- **Phase A — Perception Frenet 시임 복구 (최우선, §0a).**
  detect/track이 실차에서도 Frenet(s,d,vs,vd)을 채우도록. (ROS1처럼 perception 내부에서 FrenetConverter를
  쓰게 하거나, tracking_merger의 Frenet 채움 부분을 sim 비의존 노드로 분리.) 이게 풀려야 planner/SM이
  실차에서 산다.
- **Phase B — Tracking(EKF) 복원 (§2).** ROS1 `multi_tracking.py`의 4-state EKF·데이터연관·TTL·
  static/dynamic 분류를 ROS2로 충실 포팅. L-shape 피팅도 detect로 복원.
- **Phase C — Controller 복원 (§1).** ROS1 MAP/PP 하이브리드 + heading PID + trailing + future +
  스케일링 + AEB + FTG 멀티플렉싱. steering_lookup를 `TODO/`에서 정규 패키지로 승격(COLCON_IGNORE 해제).
- **Phase D — Prediction(GP) 복원 (§2).** 등속 → GP 회귀(`gp_traj_predictor`)로. (트래킹이 안정된 뒤.)
- **Phase E — Dynamic reconfigure 완성 (§5).** planner 5종에 save_yaml+startup yaml 로드 추가
  (sector_tuner 패턴 복제), controller·perception에 파라미터 전체 선언 + 콜백 + yaml 신설.
- **Phase F — State machine live 연동 (§3).** sector/추월존 live 갱신 재연결. (옵션) state_indicator.
- **Phase G — 통합/실차 검증.** 각 Phase 후 SIM, 마지막에 실차 토픽 그래프 점검.

각 Phase는 한 기능 영역이라 병렬화 가능하나, **A는 B/C/D/F의 선행조건**이므로 가장 먼저.
