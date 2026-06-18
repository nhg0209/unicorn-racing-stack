# UNICORN racing-stack — ROS1→ROS2 Jazzy migration log

Autonomous migration run. Goal: assemble all needed packages into `unicorn-racing-stack`
(single colcon tree), build via `build_packages_on_car.sh` (Docker-verified), and
sim-validate everything except hardware drivers and localization.

## Rules (from user)
- **cartographer**: NOT built — installed via apt (`ros-jazzy-cartographer*`).
- **Excluded** (unnecessary): `blink1`, `car_to_car_sync`, `on_track_sys_id`,
  `gp_traj_predictor`, `slam_tuner`. Also superseded: `f1tenth_simulator` (→ f1tenth_gym_ros),
  `raycast_test` (→ tools/raycaster).
- **Duplicate priority**: 1) CAC, 2) unicorn-ros1, 3) race_stack(ETH).
  Exception: the 3 sim pkgs I actively modified (`f1tenth_gym_ros`, `obstacle_publisher`,
  `stack_master`) keep the **unicorn-racing-stack** versions (raycaster overlay / square obstacles).
- **Architecture**: copy/port everything into `unicorn-racing-stack/`; `creating_autonomous_car`
  stays COLCON_IGNORE'd (pure reference + copy source). `f1tenth_gym` stays pip-editable.

## Source trees
- `creating_autonomous_car` (CAC): ROS2 base (ament). Priority-1 source.
- `unicorn-racing-stack-ros1`: ROS1 (catkin) ORIGINAL — full unicorn feature set to port.
- `race_stack` (ForzaETH): ROS2 (ament) — reference + source for pkgs absent from CAC.

## Package decisions
### ✅ already in unicorn-racing-stack (modified, keep)
stack_master, obstacle_publisher, f1tenth_gym_ros, tools/raycaster, simulator/f1tenth_gym, pitwall

### 🔵 copy from CAC (ROS2, priority-1)
| pkg | CAC path |
|--|--|
| f110_msgs | f110_msgs |
| controller | controller |
| perception | perception |
| planner | planner |
| particle_filter | slam/particle_filter |
| vesc, vesc_ackermann, vesc_driver, vesc_msgs | sensor/vesc/* |
| urg_node | sensor/urg_node |
cartographer* → apt (skip)

### 🟡 from race_stack (ROS2, only where absent from CAC/ros1-unique)
state_estimation, opponent_publisher, map_editor, f110_description, global_planner(?)

### 🔴 PORT from unicorn-ros1 (catkin→ament, priority-2)
steering_lookup, id_controller, gb_optimizer, spliner, spliner_planner, sqp_planner,
recovery_spliner, lane_change_planner, vel_planner, state_machine, frenet_conversion,
frenet_conversion_server, frenet_odom_republisher, grid_filter, polygon_filter,
lap_analyser, sector_tuner, overtaking_sector_tuner, set_pose, random_obstacle_publisher

## Phases
1. [ ] Foundation: copy CAC pkgs → build/verify
2. [ ] race_stack pkgs → build/verify
3. [ ] Port ROS1 pkgs (parallel) → build/verify
4. [ ] Update build_packages_on_car.sh + Docker build test
5. [ ] Sim validation (non-hardware, non-localization)

## Progress log
(appended chronologically below)

### Phase 1 (CAC copies) — DONE
- Build env fix: anaconda python breaks ROS msg-gen (empy). MUST build with clean env:
  `unset CONDA_PREFIX PYTHONHOME; PATH=/usr/bin:...(no conda); -DPython3_EXECUTABLE=/usr/bin/python3`
- Built OK: f110_msgs, controller, perception, planner, particle_filter
- **sudo needs password → NO local apt**. Hardware/apt-dep pkgs deferred to Docker build:
  vesc* (needs ros-jazzy-serial-driver), urg_node (nested vcs), cartographer (apt).
  Local builds use `--packages-ignore vesc_driver vesc_ackermann vesc vesc_msgs urg_node`.
