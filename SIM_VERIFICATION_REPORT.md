# Simulation Verification Report — ROS1→ROS2 Migration

Date: 2026-06-20  ·  Workspace: `unicorn_racing_stack` (robostack `ros_env`, ROS2 Jazzy, numpy 2.4)
Map: `f`  ·  Reference: `unicorn-racing-stack-ros1/stack_master/launch/{base_system,headtohead}.launch`

Every migrated autonomy subsystem was verified **running in the gym simulator** with the
virtual opponent. Several nodes shipped as TODO stubs and were implemented; several
migration bugs were fixed. The whole stack now comes up from a **single launch** and the ego
races while the autonomy reacts to the opponent.

---

## 1. Result summary (sequential subsystem verification)

| # | Subsystem | Node(s) | Verdict | Evidence (sim) |
|---|-----------|---------|---------|----------------|
| 0 | Sim baseline | gym_bridge + opponent | ✅ | `/scan`@40 Hz, `/car_state/odom`@80 Hz, map→base_link TF, opponent path-follow traced around track |
| 1 | Global planning | `trajectory_optimizer`, `waypoint_publisher` | ✅ | raceline 345 pts, v 1.48–4.0 m/s, min wall clearance 0.45 m; `/global_waypoints(_scaled,/overtaking)` published |
| 2 | Control | `pp_node` (Pure Pursuit) | ✅ | ego drives **continuous multi-lap, 239 m in 40 s, 0 freezes** |
| 3 | Detection | `detect_node` | ✅ | opponent detected in laser frame at (2.61, −0.26), size 0.26 m |
| 4 | Tracking | `tracking_node` | ✅ | `/tracked_obstacles` stable track of the opponent |
| 5 | State machine | `state_machine` (+`frenet_odom_republisher`) | ✅ | `/state_machine` = GB_TRACK → **TRAILING** when opponent ahead; `/behavior_strategy`, `/local_waypoints` populated |
| 6 | Planning (avoidance) | `spliner_node` | ✅ | `/planner/avoidance/otwpnts`: **0.75 m lateral evasion exactly at the opponent's s** |
| 7 | Integration | `headtohead.launch.xml` | ✅ | **17 nodes from one launch**; ego races 68.7 m no-freeze while track=1 + TRAILING reaction |

---

## 2. How to run (sim)

```bash
export PATH=/home/js/anaconda3/envs/ros_env/bin:$PATH
source /home/js/anaconda3/envs/ros_env/setup.bash
cd /home/js/unicorn_racing_stack && source install/setup.bash

# Full head-to-head autonomy (sensors/sim + opponent + perception + SM + planner + control):
ros2 launch stack_master headtohead.launch.xml map:=f sim:=true

# Just the foundation (sim + opponent + raceline + frenet odom):
ros2 launch stack_master base_system.launch.xml map:=f sim:=true

# Time-trial control only (proven robust multi-lap):
ros2 launch stack_master low_level.launch.xml map:=f sim:=true
ros2 run planner waypoint_publisher --ros-args -p map_name:=f
ros2 run controller pp_node --ros-args --params-file install/stack_master/share/stack_master/config/ppc.yaml
```
Spawn / drive the opponent from the RViz **Sim Control** panel (2D Goal Pose, then Path/FTG),
or headless: publish a `PoseStamped` to `/goal_pose` and a `String` ("path"/"ftg"/"manual") to `/sim/opp_mode`.
Reset the ego with a `PoseWithCovarianceStamped` to **`/sim/initialpose`** (note: the gym uses the
`/sim/`-namespaced topic, not `/initialpose`).

---

## 3. New launch files (ROS1 equivalents)

- `stack_master/launch/base_system.launch.xml` — sim/sensors + opponent + global raceline + Frenet odom.
- `stack_master/launch/headtohead.launch.xml`  — base_system + perception + state machine + spliner + Pure-Pursuit.
  Args: `map`, `sim`, `ot_planner` (default `spliner`), `control_topic`
  (`/local_waypoints` for SM-driven avoidance, or `/global_waypoints` for robust racing).

---

## 4. Stubs implemented (were `raise NotImplementedError` / `return []` / `pass`)

| File | What was implemented |
|------|----------------------|
| `controller/controller/PP.py` | **Pure Pursuit** `_compute()`: pose→yaw, vehicle-frame lookahead (speed-adaptive), curvature→steering, profile speed. |
| `planner/planner/planner/trajectory_optimizer.py` | **Min-curvature raceline `_optimize()`**: resample → normals → numpy linear-solve curvature minimization → smoothing → speed profile. Bounds use the **real boundary CSVs** (the constant centerline widths drove the line into walls). |
| `perception/perception/detect.py` | **Jump-distance LiDAR detector**: polar→Cartesian, jump clustering, AABB, wall-size rejection. |

## 5. Migration bugs fixed

| File | Bug → fix |
|------|-----------|
| `perception/perception/detect_ros.py` | `/scan` subscribed RELIABLE but sim publishes best-effort → **no scans**. Use `qos_profile_sensor_data`. |
| `planner/planner/planner/{waypoint_publisher,trajectory_optimizer}.py` | Package regroup (`planner/planner/planner/…`) broke the fixed `dirname()` map path → **robust `_find_maps_dir()`** (ascend to `stack_master/maps`). |
| `planner/planner/planner/waypoint_publisher.py` | `/global_waypoints_scaled` and `/global_waypoints/overtaking` had **no publisher** → state machine blocked at startup. Now published (identity copies). |
| `race_utils/opponent/opponent/obstacle_merger.py` | Virtual obstacles carried only Cartesian (x,y,θ); planners/SM need **Frenet s/d** → merger now fills s/d via `FrenetConverter` + `/global_waypoints`. |
| `planner/spliner/spliner/spliner_node.py` | `float(size-1 array)` crash under **numpy 2.x** in `get_cartesian([s],[d])` → pass scalars (0-d array). |
| `state_machine/state_machine/state_machine.py` | `track_length` was a constant param → now derived from the global raceline `s_m` (map-agnostic s-wrapping). |
| `controller/controller/PP.py` | Waypoint sub was TRANSIENT_LOCAL → rejected the SM's volatile `/local_waypoints`. Use depth-10 volatile (compatible with both). Added `waypoint_topic`/`odom_topic` params. |

## 6. Topic wiring established (sim)

```
gym ego scan → /scan_raw → scan_augmentor (overlays opponent box) → /scan
  ├─ detect_node → /detections → tracking_node → /tracked_obstacles        (perception pipeline)
opponent_vehicle → /sim/dynamic_obstacles ┐
static_obstacle_manager → /sim/static_obstacles ┼ obstacle_merger → /tracking/obstacles (Cartesian + Frenet)
                                            ┘        ▲ (+ real /tracking/obstacles_raw when present)
/car_state/odom → frenet_odom_republisher → /car_state/odom_frenet
waypoint_publisher → /global_waypoints(_scaled, /overtaking, centerline)
/tracking/obstacles + raceline + odom_frenet → state_machine → /behavior_strategy, /local_waypoints, /state_machine
/tracking/obstacles → spliner_node → /planner/avoidance/otwpnts
/global_waypoints | /local_waypoints → pp_node → /vesc/high_level/ackermann_cmd → simple_mux → gym
```

## 7. Generated data

- `stack_master/maps/f/global_waypoints.csv` — raceline generated by `trajectory_optimizer`
  (`x_m,y_m,w_tr_right_m,w_tr_left_m,psi_rad,kappa_radpm,vx_mps`). Regenerate per map:
  `ros2 run planner trajectory_optimizer --ros-args -p map_name:=<map> -p v_max:=4.0 -p a_lat_max:=4.0`.

## 8. Known items / next steps (not blocking)

1. **Controller following `/local_waypoints`** (true avoidance execution) drives, but can clip a wall at
   one corner where the track passes near itself — the Frenet nearest-point projection jumps to the wrong
   branch, so the local window is built around the wrong `s`. `/global_waypoints` following is robust
   (multi-lap). Fix = Frenet projection with s-continuity tracking; then default `control_topic` to `/local_waypoints`.
2. **Detection is sensor-frame only** (simplified port): it clusters compact returns (opponent box) but also
   surfaces compact wall corners and does not compute Frenet/track-bounds filtering like the ROS1 `detect.cpp`.
   Autonomy consumes the map-frame `/tracking/obstacles` (merger), so this does not block racing.
3. **Not ported / out of scope** (referenced but optional in the ROS2 nodes): MPC `ego_prediction`,
   GP opponent-trajectory prediction, `predictive_spliner` collision/force-trailing topics. Use
   `ot_planner:=spliner` (default in `headtohead.launch.xml`).
4. **`trajectory_optimizer._optimize`** is a clean curvature-minimization (numpy), not the full ROS1
   minimum-curvature QP. Good enough to race; swap in `trajectory_planning_helpers.opt_min_curv` for lap-time.

## 9. Build note

`f110_msgs` / `frenet_conversion_msgs` etc. fail to rebuild with the stale-symlink error
`failed to create symbolic link … Is a directory` after switching to `--symlink-install`. Fix:
`rm -rf build/<pkg> install/<pkg>` then rebuild. All 31 needed packages build clean afterward.

---

# Round 2 — Sequential per-package verification + obstacle endurance (2026-06-20)

Re-verified every package **in order** (drive solo → detect → frenet → behavior/local
waypoints → static avoidance → dynamic → endurance), found and fixed the real root
causes of the "local waypoints appear in the wrong place / car can't even drive" symptom.

## Root-cause fixes
| File | Bug → fix |
|------|-----------|
| `frenet_conversion/frenet_converter.py` | `waypoints_distance_m` hardcoded 0.1 m → `get_approx_s` (index×spacing) seeded the projection on the wrong part of the track, so `/car_state/odom_frenet` d was off by metres on ~half the lap. Now uses the **true cumulative arc-length** at the nearest waypoint + the real mean spacing. → d=0.00 everywhere. |
| `state_machine/states.py` | `GlobalTracking` local-window start `int(cur_s/waypoints_dist)` assumed a uniform 0.1 m raceline; on a 0.25 m raceline the window started **~3.4 m ahead** of the car → PP cut corners into the wall. Now starts at the waypoint whose `s_m` is closest to the car (spacing-independent). → window offset −0.02 m. |
| `state_machine/state_machine.py` | `waypoints_dist`/`track_length` now derived from the actual raceline; `_check_ot_sector` returns **True when no ot_sectors are configured** so OVERTAKE can trigger (otherwise the car only ever trails and stops behind a static obstacle). |
| `gym_bridge.py` | crash-recovery: respawn the ego on the nearest raceline point if it stays wedged against a virtual obstacle (the collision-stop alone dead-locks it). |

## Sequential results (sim, map f, headtohead.launch)
| Step | Result |
|------|--------|
| A3 Frenet | `/car_state/odom_frenet` d=0.00, s-err=0.00 at idx 0/80/160/240/320 (was ±6 m) |
| A4 Solo control | **4 laps, 332 m, max\|d\|=0.32 m, 0 collisions** (PP follows `/local_waypoints`) |
| B Detection | opponent in `/detections` at laser-frame (2.61,−0.26) sz 0.26; tracking stable |
| B Obstacle Frenet | `/tracking/obstacles` opponent s_center=10.7 (truth 10.7), d_center=0.00, err 0.00 m |
| C1/C2 Static avoidance | GB_TRACK→TRAILING→**OVERTAKE**, evades **0.77 m** around an on-line static obstacle, passes it, **0 collisions** |
| **D1 Endurance** | **27 laps, 2086 m, static + dynamic obstacles together, max_collision=0.0 s, 0 recoveries** — OVERTAKE the static + TRAIL the moving opponent every lap, never collides |

Verdict: solo driving and obstacle handling (static + dynamic) are robust over dozens of
laps with zero collisions. State machine cycles GB_TRACK / TRAILING / OVERTAKE correctly.
