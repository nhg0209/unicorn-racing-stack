# Raceline 생성 & 섹터 튜닝 사용법 (gb_optimizer + IFAC2026 `_modi` 이식)

unicorn 스택에서 **맵 → raceline → 섹터 → 주행**까지의 전체 워크플로 정리.
gb_optimizer의 흐름을 따르되, IFAC2026의 2단계 최적화(IQP→SP)와 `[map]_modi.png`
경로 유도 방식을 이식한 생성 경로(`create_path.launch.py`)를 포함한다.

> 좌표/단위: 모든 raceline은 `map` 프레임, f110_msgs/WpntArray.
> 맵 폴더: `stack_master/maps/<map>/` (소스 트리. `--symlink-install`이면 install이 여기로 심링크).

---

## 0. 빌드

```bash
cd ~/unicorn_ws
colcon build --symlink-install --packages-select gb_optimizer stack_master
source install/setup.bash
```
- 차량 동역학/최적화 파라미터는 **`stack_master/config/CAR/`** 사용
  (`racecar_f110.ini` + `veh_dyn_info/{ggv,ax_max_machines}.csv`).
- 빌드는 직접 실행 (이 저장소 규칙).

---

## 1. 맵 만들기 (SLAM 매핑) — 기존 흐름, 미변경

```bash
ros2 launch stack_master mapping.launch.xml map:=<맵이름>
```
- cartographer로 SLAM. 트랙을 한 바퀴 돌고 `finish_map`으로 종료하면
  `stack_master/maps/<맵이름>/`에 `<맵이름>.png / .yaml / .pbstream` 저장.
- **online(create_map=true, 매핑 중 라이브 raceline 빌드) 분기는 unicorn에선 사용 안 함.**
  매핑은 맵만 저장하고, raceline은 아래 2~3에서 **offline**으로 생성한다.

---

## 2. (선택) `[맵이름]_modi.png`로 경로 유도

raceline을 원하는 영역으로 유도하고 싶을 때만.

1. `stack_master/maps/<맵이름>/<맵이름>.png`를 복사해 `<맵이름>_modi.png` 생성.
2. 이미지 편집기로 **막고 싶은 영역을 검정(점유)으로 칠한다** = "가상 벽".
3. 끝. (자동 감지됨)

동작 원리 (2-pass):
- `_modi.png`가 있으면 **centerline + 트랙폭(w_tr)** 을 그걸로 추출 → 가상 벽이
  최적화 QP의 편차 제약(`-dev_max ≤ α ≤ dev_max`)을 좁혀 **경로가 그 영역을 우회**.
- **실제 벽 경계는 항상 원본 `<맵이름>.png`에서 별도 추출** → 발행되는
  `d_left`/`d_right` 및 주행 검증(check_traj)은 실제 트랙 기준 (회피/state_machine이
  가상 벽을 실제로 오인하지 않음).
- `<맵이름>.yaml`의 `image:`(로컬라이제이션 맵)는 **건드리지 않는다.**

---

## 3. Raceline 생성

두 가지 경로가 있다. 보통 **A(IFAC 이식)** 사용.

### A. IFAC 이식 옵티마이저 (`_modi` 지원) — 권장

```bash
ros2 launch stack_master create_path.launch.py map:=<맵이름>
```
순차 실행(OnProcessExit):
1. `centerline_extractor` → `centerline.csv` + `boundary_{right,left}.csv`
   (`_modi.png` 있으면 폭은 가상 벽, 경계는 실제 벽)
2. `trajectory_optimizer` → `global_waypoints.json` (IQP 레이싱라인 + SP 최단경로,
   unicorn 스키마. 발행 `d_left/d_right`는 실제 벽 거리로 채움)
3. `global_trajectory_publisher`(json→토픽) + `sector_slicer` + `ot_sector_slicer`

주요 인자:
| arg | 기본 | 설명 |
|---|---|---|
| `map` | (필수) | 맵 폴더명 |
| `racecar_version` | `CAR` | `stack_master/config/<여기>/` 의 ini+veh_dyn 선택 |
| `safety_width_iqp` | `-1` | IQP 안전폭 [m]. **<0 = ini 값 사용**(아래 참고). >0이면 그 값으로 일시 오버라이드 |
| `safety_width_sp` | `-1` | SP 안전폭 [m]. **<0 = ini 값 사용**. >0이면 오버라이드 |
| `reverse` | `false` | 진행방향 반전(CW) |
| `enable_mintime` | `false` | opt_mintime 추가 실행(CasADi 필요) |
| `show_plots` | `false` | 추출 결과 matplotlib 표시(디스플레이 필요) |

**안전폭(width)은 ini에서 관리** — `stack_master/config/<racecar_version>/racecar_f110.ini`:
```ini
[OPTIMIZATION_OPTIONS]
optim_opts_mincurv={"width_opt": 0.6, ...}   # IQP 안전폭 (w_veh)
optim_opts_shortest_path={"width_opt": 0.4}  # SP  안전폭
[GENERAL_OPTIONS]
imp_opts={..., "min_track_width": 0.8, ...}  # prep_track 최소 확장폭
```
- IQP/SP 안전폭의 **단일 소스는 ini의 `optim_opts_*.width_opt`**. 런치/CLI 인자는 >0일 때만 일시 오버라이드.
- 좁은 트랙/공격적 `_modi`에서 `Problem not solvable` 나면 **`optim_opts_mincurv.width_opt`를 낮춘다**(예 0.8→0.6→0.5). `width_opt`가 `min_track_width`(0.8)보다 작아야 IQP에 여유가 생긴다.
- 차폭 `veh_params.width`(0.30)·`curvlim`·`v_max`도 같은 ini.

### B. gb 네이티브 옵티마이저 — `_modi` 미지원 (fallback)

```bash
ros2 launch stack_master raceline_generator.launch.xml map:=<맵이름>
```
gb_optimizer의 `global_planner_node`(watershed+TPH)로 생성. `_modi.png`는 무시된다.

### 3-1. 섹터 분할 (A·B 공통, 인터랙티브)

생성이 끝나면 `sector_slicer`와 `ot_sector_slicer`가 **matplotlib 창**을 띄운다
(디스플레이 필요). 슬라이더로 위치를 옮기며 키로 섹터 경계를 추가 → 닫으면 저장:
- `speed_scaling.yaml` — 속도 스케일 섹터 (sector_slicer)
- `ot_sectors.yaml` — 추월 구간 (ot_sector_slicer)

둘 다 `stack_master/maps/<맵이름>/`에 기록. 처음엔 모든 섹터 `scaling: 1.0`.
섹터를 안 나눠도 전체 1개 섹터로 저장된다(나중에 4-1에서 조정).

생성물 확인:
```bash
ls stack_master/maps/<맵이름>/
# <맵이름>.png .yaml .pbstream  (맵)
# <맵이름>_modi.png             (선택)
# centerline.csv boundary_right.csv boundary_left.csv
# global_waypoints.json         (IQP+SP raceline; 런타임 발행 소스)
# speed_scaling.yaml ot_sectors.yaml
```

---

## 4. 주행 (생성된 raceline 사용)

```bash
# 시뮬
ros2 launch stack_master race.launch.xml map:=<맵이름> sim:=true
# 실차
ros2 launch stack_master race.launch.xml map:=<맵이름> sim:=false
```
`base_system`이 `global_waypoints.json`을 읽어 자동 발행한다:
- `global_trajectory_publisher` → `/global_waypoints`(IQP), `/global_waypoints/shortest_path`(SP),
  `/centerline_waypoints`, `/trackbounds/markers`
- `sector_tuner`(speed_scaling.yaml) → `/global_waypoints_scaled` (state_machine가 추종하는 속도 스케일본)
- `ot_interpolator`(ot_sectors.yaml) → `/global_waypoints/overtaking`

컨트롤러(`controller_manager` 기본 / `use_pp_heading:=true`로 pp_heading)는
state_machine 출력(`/local_waypoints`)을 추종한다.
> raceline 생성을 위해 새 옵티마이저를 다시 돌릴 필요 없음 — json만 있으면 됨.

### 4-1. 런타임 섹터 속도 튜닝

주행 중 섹터별 속도 스케일을 라이브 조정하고, 원하면 yaml에 저장:
```bash
# 예: Sector0를 80%로
ros2 param set /speed_sector_tuner Sector0.scaling 0.8
# 전체 상한
ros2 param set /speed_sector_tuner global_limit 1.0
# speed_scaling.yaml에 저장(맵 폴더로 write-back)
ros2 param set /speed_sector_tuner save_params true
```
(노드명은 base_system에서 `speed_sector_tuner`. `scaling`은 `[0, global_limit]`로 클램프됨.)

---

## 파일 입출력 요약 (맵 1개 기준)

```
stack_master/maps/<맵>/
├── <맵>.yaml / <맵>.png / <맵>.pbstream   [입력] SLAM 산출(매핑)
├── <맵>_modi.png                          [입력·선택] 가상 벽 (경로 유도)
├── centerline.csv                         [중간] x,y,w_tr_right,w_tr_left (폭=_modi)
├── boundary_right.csv / boundary_left.csv [중간] 실제 벽 (d_lr·검증용)
├── global_waypoints.json                  [출력] IQP+SP raceline (런타임 발행 소스)
├── speed_scaling.yaml                     [출력] 속도 스케일 섹터
└── ot_sectors.yaml                        [출력] 추월 구간
```

설정(차량/최적화): `stack_master/config/CAR/racecar_f110.ini`,
`stack_master/config/CAR/veh_dyn_info/{ggv,ax_max_machines}.csv`.

---

## 구조 메모 (어디를 고쳤나)

- 이식 노드: `gb_optimizer/centerline_extractor.py`, `gb_optimizer/trajectory_optimizer.py`
- 번들 TPH: `gb_optimizer/tph/` (IFAC 자기완결 복사. `sys.modules` 별칭이라 gb의
  pip-TPH와 별도 프로세스에서 무충돌)
- 런치: `stack_master/launch/create_path.launch.py` (신규). `raceline_generator.launch.xml`은
  gb 네이티브로 그대로 둠.
- 런타임 발행/섹터/주행 경로는 **전부 기존 그대로** (json 스키마가 동일해 호환).
- 라이브 매핑(create_map=true) 코드는 미수정 — 이식은 **offline 생성 전용**.

---

## 트러블슈팅

| 증상 | 원인/조치 |
|---|---|
| `Centerline CSV not found` | extractor가 먼저 안 돎. `create_path.launch.py`로 실행(순차 보장). |
| `boundary CSVs not found — validation disabled` | 벽 경계 추출 실패(맵 노이즈). morph 후 흰 영역이 닫힌 contour인지 확인. |
| raceline가 벽에 붙음 | `safety_width_iqp` ↑ (예 0.8→1.0). `_modi`로 해당 구간 가상 벽 추가도 가능. |
| 경로가 가상 벽을 안 피함 | `<맵>_modi.png` 파일명/위치 확인(맵 폴더, `_modi` 접미사). 로그의 `Using modified map image` 확인. |
| matplotlib/TkAgg 에러(헤드리스) | extractor는 `show_plots:=false`(기본)면 안전. 섹터 슬라이서는 디스플레이 필요. |
| RViz에서 raceline가 맵과 어긋남 | yaml의 `resolution/origin` 확인. extractor·gb 모두 `cv2.flip`+`pixel*res+origin` 사용. |
| 생성했는데 차가 안 감 | 4의 주행은 별개. mux 키보드 모드 등은 race 사용법 참고(`use_pp_heading`/`keyboard_teleop`). |
