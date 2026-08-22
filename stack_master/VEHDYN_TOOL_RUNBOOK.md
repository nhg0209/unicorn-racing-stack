# 차량 동역학 자동 측정 — 사용 절차

`vehdyn_test_node` + `vehdyn_analyze.py` 사용법.

**이 문서와 `VEHICLE_DYNAMICS_RUNBOOK.md` 의 관계.** 그 문서는 2026-08-12 에 **손으로** 수행한
절차이고, 각 기동이 왜 존재하는지와 도구 없이 재는 법의 기준으로 남습니다. 이 문서는 그
절차의 §2(기동)와 §4(대회장)를 자동화한 도구의 조작법입니다. **왜**가 궁금하면 그쪽,
**어떻게 돌리나**가 궁금하면 이쪽입니다.

---

## 0. 한 줄 요약

| 상황 | 명령 | 시간 |
|---|---|---|
| 대회장, 노면이 바뀜 | `mode:=circle` (기본값) | ~2분 |
| 연습장, 전체 특성화 | `mode:=full` | ~15분 |
| 공간이 거의 없음 | `mode:=current` | ~1분, 신뢰도 낮음 |

산출물은 `config/vehdyn_measured/<타임스탬프>/` 에 **후보로만** 생깁니다.
**라이브 `veh_dyn_info/` 는 이 도구가 절대 쓰지 않습니다.** 사람이 읽고 손으로 복사합니다.

---

## 1. 설치

NUC 으로 옮길 파일:

```
stack_master/config/vehdyn_test_params.yaml
stack_master/launch/vehdyn_test.launch.xml
stack_master/scripts/vehdyn_test_node.py        ★ 필수
stack_master/scripts/vehdyn_analyze.py
stack_master/scripts/test_vehdyn.py             (선택)
stack_master/CMakeLists.txt                     ★ install(PROGRAMS) 에 +1줄
```

```bash
chmod +x stack_master/scripts/vehdyn_*.py stack_master/scripts/test_vehdyn.py
```

**실행 권한을 반드시 확인하십시오.** 설치본이 소스로의 심볼릭 링크라 소스 파일 모드가 그대로
실행됩니다. 644 로 복사되면 launch 가 permission denied 로 죽습니다.

`stack_master` 를 다시 빌드해야 합니다 — `CMakeLists.txt` 가 바뀌었고, config/launch 에 **새
파일**이 생겼으므로 심볼릭 링크 생성을 위해 install 단계가 다시 돌아야 합니다.

`vehdyn_analyze.py` 는 **scipy** 가 필요합니다(`savgol_filter`). NUC 에 없으면 분석만
노트북에서 하십시오 — bag 만 가져오면 됩니다.

---

## 2. 시작 전 확인 — 빠뜨리면 결과가 조용히 틀립니다

### 2.1 기준값이 차의 실제 파일과 같은가

`vehdyn_test_params.yaml` 은 라이브 값을 **참조로 박아두고** 있고, analyzer 는 라이브 csv 를
읽지 않고 이 참조만 씁니다. 어긋나면 `report.md` 의 "live now" 열이 거짓이 되고 **k 가 그만큼
틀립니다.**

```bash
grep -v '^#' stack_master/config/CAR/veh_dyn_info/ggv.csv | head -1
head -2 stack_master/config/CAR/veh_dyn_info/ax_max_machines.csv
head -2 stack_master/config/CAR/veh_dyn_info/b_ax_max_machines.csv
```

읽은 값을 yaml 과 맞춥니다:

```yaml
ggv_ay_max_ref: 5.7          # ggv.csv 3번째 열
ggv_ax_max_ref: 7.0          # ggv.csv 2번째 열
ax_max_machines_ref: 9.5
b_ax_max_machines_ref: 10.0  # 차가 아직 5.0 이면 5.0 으로
```

> `k = ay_측정 / ggv_ay_max_ref` 입니다. 측정 9.05 에서 ref 5.7 이면 k=1.59, ref 7.0 이면
> k=1.29 — 결과 전체가 그만큼 스케일됩니다.

### 2.2 공간과 안전 상자

상자는 **차를 가운데 세워둔 상태의 `/car_state/odom` 좌표계**입니다. 재고 넣으십시오.

```yaml
box_center_x: 0.0
box_center_y: 0.0
box_w: 8.0            # 전체 폭 (반폭 아님)
box_h: 8.0
safety_margin_m: 1.0
straight_len_m: 8.0
area_radius_m: 4.0
v_max_allowed: 9.0
```

### 2.3 `/car_state/odom` 이 나오는가

이 토픽은 `frenet_odom_republisher` 가 내고, **글로벌 라인이 필요하며 `low_level` 만으로는 안
뜹니다.** 없으면 노드는 **fail-closed** 로 아예 움직이지 않습니다.

```bash
ros2 topic hz /car_state/odom
```

안 나오면 맵을 띄우거나, `odom_topic` 을 가진 오도메트리로 바꾸고 **상자 좌표계가 그 프레임에
맞는지** 다시 확인하십시오.

---

## 3. 대회장 절차 — 노면이 바뀌었을 때 (약 2분)

### 3-1. 계획 확인 (차는 움직이지 않습니다)

```bash
ros2 launch stack_master vehdyn_test.launch.xml
```

표가 출력됩니다. **읽고 납득한 뒤에만** 다음으로 갑니다:

```
  dry_run             : True   <-- NOTHING WILL BE PUBLISHED
  safety box          : 8.0 x 8.0 m centred (0.0, 0.0), margin 1.0 m
  v_max_allowed       : 9.0 m/s   slew 3.0 m/s^2

  DERIVED FROM THE SPACE
    straight usable   : 6.00 m -> v_target 7.99 m/s
    circle radius     : 2.85 m -> washout at ~4.47 m/s

   #  maneuver           side   budget[m]  est[s]  note
   0  T2_washout         left        34.5    12.4  washout ramp 1/3
   1  T2_washout         right       34.5    12.4  washout ramp 1/3
   ...
```

확인할 것: **원 반경이 실제 공간에 들어가는가**, **워시아웃 속도가 감당 가능한가**,
**거리 예산이 상자 안에서 말이 되는가**.

### 3-2. 녹화 시작

```bash
ros2 bag record -s mcap -o ~/vehdyn_$(date +%m%d_%H%M) \
  /vesc/sensors/imu/raw /car_state/odom /vesc/odom /vesc/sensors/core \
  /vesc/high_level/ackermann_cmd /vesc/commands/servo/position /joy
```

**소싱을 빠뜨리면 `vesc_msgs` 를 못 찾아 `/vesc/sensors/core` 가 조용히 빠집니다** — 경고만
뜨고 녹화는 계속됩니다. 시작 후 `ros2 bag info` 로 6개 토픽이 다 잡혔는지 확인하십시오.

### 3-3. 실행

```bash
ros2 launch stack_master vehdyn_test.launch.xml dry_run:=false
```

차를 **상자 중앙, 원의 시작 지점**에 두고 시작합니다.

### 3-4. 분석

```bash
python3 stack_master/scripts/vehdyn_analyze.py ~/vehdyn_<타임스탬프>
```

`config/vehdyn_measured/<타임스탬프>/` 에 `ggv.csv`, `ax_max_machines.csv`,
`b_ax_max_machines.csv`, `report.md`, `raw.json` 이 생깁니다.

### circle 모드가 무엇을 하고 무엇을 하지 않는가

- **하는 것**: 워시아웃으로 `ay_max` 를 재고, `k = ay_측정 / ggv_ay_max_ref` 를 구해
  **ggv 의 ax_max 와 ay_max 양쪽에 곱합니다.** 둘 다 타이어 그립이라 노면이 바뀌면 같이
  스케일되는 것이 물리적으로 맞습니다.
- **하지 않는 것**: `ax_max_machines` / `b_ax_max_machines` 는 **건드리지 않고 현재 값을 그대로
  복사**합니다. 모터와 브레이크 성질이라 노면과 무관합니다.

---

## 4. 전체 특성화 — 연습장 (약 15분)

```bash
ros2 launch stack_master vehdyn_test.launch.xml mode:=full long_mode:=oval dry_run:=false
```

`long_mode` 선택:

| 값 | 언제 | 비고 |
|---|---|---|
| `oval` (기본) | 상자만 있을 때 | 8×4 m 에서 랩 20.6 m, 코너 3.7 / 직선 7.0 m/s. **공간 효율이 가장 좋습니다** |
| `shuttle` | 긴 직선이 있을 때 | `straight_len_m` 필요 |
| `circle_accel` | 큰 원만 있을 때 | 타이어 여유를 남긴 채 가속 |

`oval` / `circle_accel` 은 `|a_lat| < ay_ref/2` 인 표본만 종가속으로 채택합니다.

`full` 은 `circle` 이 하는 전부에 더해:
- `ax_max_machines` / `b_ax_max_machines` 를 **속도 구간별 실측**으로 만듭니다
- ggv 의 `ax_max` 를 k 스케일이 아니라 **직접 측정값**으로 씁니다

### T3/T4 는 반드시 등속 안정화 후 계단

`step_settle_s`(기본 2.0초) 동안 `v_washout` 의 70% 로 **등속 유지한 뒤에** 스로틀/제동 계단을
넣습니다. 처음부터 가속하면 `a_x` 와 `a_lat` 이 같이 올라 **포락선이 아니라 램프를 재게 됩니다.**
2026-08-12 시험이 정확히 그렇게 무효가 됐습니다. 이 값을 0 으로 만들지 마십시오.

---

## 5. 조작자 절차 — 기동 사이 PAUSE

기동이 하나 끝나면 노드가 PAUSE 로 들어가 **속도 명령 발행을 멈춥니다.**

1. 콘솔에 다음 기동의 이름과 안내가 뜹니다.
2. **`buttons[4]`(humandrive)를 눌러** 조종 권한을 가져옵니다.
   `simple_mux` 는 스틱을 움직이는 것만으로는 안 넘어옵니다 — 버튼을 눌러야 합니다.
3. 차를 다음 기동의 시작 위치·방향으로 몹니다.
4. **`buttons[7]`(`resume_button`)** 을 눌러 재개합니다.
5. 노드가 현재 pose 를 계획된 시작 pose 와 대조합니다. 벗어나면 **거부하고 다시 안내합니다**:

```
RESUME REFUSED: 1.23 m / 41.2 deg from the planned start pose (tol 0.5 m / 20 deg).
Reposition and press again.
```

허용오차는 `pose_tol_m`(0.5 m) / `pose_tol_deg`(20°). **엉뚱한 위치에서 고속 출발을 막는 유일한
게이트입니다.** 넓히지 마십시오.

완료한 기동은 `progress.json` 에 기록되어, 중간에 끊겨도 그 지점부터 재개됩니다.

> 매 기동 앞에 **정지 5초**(`settle_still_s`)가 들어갑니다. 바이어스 추출과 구간 분할이 여기
> 달려 있습니다. 생략 불가입니다.

---

## 6. 중단 조건 — 네 가지

| 조건 | 동작 |
|---|---|
| **조이 입력** — 스틱이 데드존(0.15)을 넘거나 아무 버튼 | 즉시 정지 |
| **상자 이탈** — `/car_state/odom` 이 상자 밖 | 즉시 정지 |
| **거리 초과** — 기동별 사전 계산 예산 초과 | 즉시 정지 |
| **오도메트리 없음/노후** — 1초 이상 | 즉시 정지 (fail-closed) |

조이는 "권한 요청"이 아니라 **중단**으로 처리합니다. 조작자가 권한을 원하는 것인지 차가
이상해서 패드로 손이 간 것인지 노드는 구분할 수 없기 때문입니다.

속도 명령에는 슬루 제한(`v_slew_mps2`, 기본 3.0)이 걸려 있어 계단 명령이 그대로 나가지 않습니다.

**물리 킬스위치를 항상 손에 두십시오.** 위 네 가지는 소프트웨어이고, 소프트웨어가 죽으면 같이
죽습니다.

---

## 7. `report.md` 읽는 법

### 먼저 볼 것 — 축 검증

```
| corr(imu.x, dv/dt), |a_lat|<2.0 | +0.929 | > 0.8 | PASS |
| corr(imu.y, v*gz)              | +0.934 | > 0.8 | PASS |
```

**FAIL 이면 값이 아예 나오지 않습니다.** IMU 축 배정을 증명하지 못했다는 뜻이고, 그 아래 모든
숫자는 어느 축이 어느 축인지에 대한 추측이 됩니다. 이 검사가 `controller_manager` 가 `-imu.x`
를 종가속으로 쓰는데 실측은 `+imu.x` 인 부호 오류를 잡았습니다.

### 신뢰도 항목

- **정지 표본 수** — 바이어스 추정의 근거. 적으면 경고가 뜹니다
- **종방향 속도 커버리지** — `0–9 m/s` 를 못 덮었으면 그 밖 구간은 측정이 아니라 참조값입니다
- **원 피팅 잔차** — 0.15 m 초과면 궤적이 원이 아니었다는 뜻(보통 속도가 아직 오르는 중).
  **신뢰도 낮음**으로 표시됩니다
- **호 길이** — π(반 바퀴) 미만이면 잔차가 아무리 좋아 보여도 **반경이 구속되지 않습니다.**
  8/12 의 마지막 우회전이 2.23 rad 에 잔차 0.09 — 세트에서 가장 좋아 보이는 피팅인데 18.3
  m/s² 를 냈습니다(실제 ~10). 이 값을 낮춰서 구간을 되살리지 마십시오
- **두 횡방향 방법의 차이** — 가속도계가 보통 1~2 m/s² 높습니다(코너링 롤로 중력이 횡축에
  샙니다). **보수적으로 낮은 쪽을 채택**합니다

### 대회장 재보정 기준값

```
**ay_max measured here = 9.052 m/s^2**
```

**이 숫자를 적어두십시오.** 다음에 노면이 바뀌면 circle 을 돌리고 나누기만 하면 됩니다:
`k = ay_새로 / ay_이번`.

---

## 8. 적용

1. `report.md` 의 숫자가 믿을 만한지 판단합니다. 라이브 값(ggv 7.0/5.7, ax_max_machines 9.5,
   b_ax 10.0)에서 크게 벗어나면 **차가 아니라 측정을 의심하십시오** — 라이브 세트는 지금 잘
   돌고 있는 검증된 값입니다.
2. `config/vehdyn_measured/<ts>/` 의 csv 를 `config/<CAR|SIM>/veh_dyn_info/` 로 **손으로**
   복사합니다.
3. **레이스라인을 재생성합니다.** 차가 따르는 `vx_mps` 는 `maps/<map>/global_waypoints.json`
   안의 **오프라인 산출물**이고, 온라인 경로는 그것을 낮출 수만 있지 올릴 수 없습니다.
   **재생성 전까지는 차가 조금도 빨라지지 않습니다. 스택 재시작으로는 안 됩니다.**
4. `ggv.csv` 의 손으로 쓴 헤더를 **되살려 넣으십시오.** 재생성 도구가 헤더를 지웁니다.

> 예외: 회피 계획 쪽(`static_avoidance_node`, `state_machine`)은 startup 에 veh_dyn 을 읽으므로
> **재시작만으로 즉시 반영됩니다.** 레이스라인만 재생성이 필요합니다.

---

## 9. 문제 해결

| 증상 | 원인 |
|---|---|
| `executable 'vehdyn_test_node.py' not found` | `CMakeLists.txt` 의 `install(PROGRAMS)` 한 줄이 빠졌거나 리빌드 안 함 |
| permission denied | 소스 파일이 644. `chmod +x` |
| 차가 안 움직임, 로그에 `dry_run is TRUE` | 정상. `dry_run:=false` |
| 차가 안 움직임, `no odometry received yet` | `/car_state/odom` 이 없음. §2.3 |
| 즉시 `ABORT: outside safety box` | 상자 좌표가 실제 위치와 다름. §2.2 |
| `RESUME REFUSED` 반복 | 차가 계획 시작점에서 벗어남. 되돌려 놓고 재시도 |
| analyzer 가 `FAILED — axis validation` | IMU 배선/축이 바뀌었거나 bag 에 기동이 없음 |
| `report.md` 의 "live now" 가 차와 다름 | yaml 참조값이 낡음. §2.1 |
| `/vesc/sensors/core` 가 bag 에 없음 | 녹화 전 소싱 누락 |

### 자체 검사

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest stack_master/scripts/test_vehdyn.py -q
```

18개 통과해야 합니다. 그중 2개는 `~/vehdyn_0812_1948` bag 이 있을 때만 돌고, 그것이 **분석부의
회귀 시험**입니다 — 2026-08-12 실측을 재현하는지 확인합니다. bag 은 저장소에 없으므로 없으면
skip 됩니다.

`stack_master/scripts` 는 CLAUDE.md 의 pytest 경로 목록에 없어서 문서화된 명령으로는 수집되지
않습니다. 위 명령을 따로 돌리십시오.
