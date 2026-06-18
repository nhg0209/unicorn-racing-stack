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

### Phase 3 (ROS1 ports) — ALL BUILD OK
- Ported via 6 parallel agents: frenet_conversion(+_msgs split), frenet_conversion_server,
  frenet_odom_republisher, steering_lookup, id_controller, state_machine, sector_tuner,
  overtaking_sector_tuner, lap_analyser, set_pose, random_obstacle_publisher, grid_filter,
  polygon_filter, vel_planner, gb_optimizer, spliner, spliner_planner, sqp_planner,
  recovery_spliner, lane_change_planner.
- Fix: frenet_conversion had rosidl+ament_python in one ament_cmake pkg -> egg target clash.
  SPLIT into frenet_conversion (ament_python class) + frenet_conversion_msgs (ament_cmake srvs).
  frenet_conversion_server now imports frenet_conversion_msgs.srv.
- Fix: unescaped '<' in package.xml <description> (Frenet<->global) -> XML parse error.
- NOTE: ament_python "build" = structural (entry points/setup) only; runtime imports NOT checked.
  Known runtime TODOs: grid_filter Python class still imports rospy; sqp_planner calls removed
  FrenetConverter.get_e_psi; planners need pip libs tph (trajectory_planning_helpers) + ccma.

### Phase 5a (runtime imports) — 24/24 node modules import OK
- Ported grid_filter Python GridFilter rospy->rclpy (takes node=; subscribes via node). Patched 6
  consumers to pass node=self (recovery_spliner, spliner static/start*, spliner_planner, lane_change_planner).
- pip RUNTIME deps (install to ~/.local): trajectory_planning_helpers, ccma (git+ForzaETH/CCMA),
  quadprog, casadi, tqdm, transforms3d.
- **numpy conflict FIX**: pip pulled numpy 2.4.6 into ~/.local, clashing with ROS system numpy 1.26.4
  (dtype ABI + np.Inf removed). Removed ~/.local numpy; rebuilt quadprog from source --no-build-isolation
  against 1.26.4. KEEP system numpy 1.26.4. For Docker: install pip deps WITHOUT upgrading numpy
  (use --no-deps for quadprog/casadi or constrain numpy<2).

### Phase 5b (sim validation) — core nodes start OK
- Sim relaunched after migration build: gym_bridge UP, /scan publishing. OK.
- Bare `ros2 run` smoke (timeout=alive): id_controller OK, lap_analyser OK,
  random_obstacle_publisher OK. sector_tuner needs 'n_sectors' param (config-driven, needs
  launch+map config). frenet_odom exe is `frenet_odom_republisher_node` (test used wrong name; node fine).
- 24/24 node modules import cleanly (gb_optimizer needs apt python3-sklearn/skimage = offline tool).
- NOTE: ported nodes are NOT yet wired into stack_master launch files (CAC's launches use CAC nodes).
  Full autonomy bring-up = user's sequential follow-up. Build + import + individual-node validation done.

### Phase 4 (Docker build test) — IN PROGRESS
- build_packages_on_car.sh rewritten: self-contained (CAC+race_stack+ros1 COLCON_IGNORE'd),
  cartographer via apt, numpy<2 pinned, planner pip deps (tph/ccma/casadi/quadprog/tqdm).
- Running in container `ros2-slam:jazzy` with workspace mounted at /ws (host build/install backed
  up to *_hostbak). osrf/ros:jazzy-desktop pulling in bg for a clean re-test.

## STATUS SUMMARY (migration)
- BUILT (34 pkgs, local + ament): f110_msgs, controller, perception, planner, particle_filter (CAC);
  state_estimation, opponent_publisher, map_editor (race_stack); frenet_conversion(+_msgs),
  frenet_conversion_server, frenet_odom_republisher, steering_lookup, id_controller, state_machine,
  sector_tuner, overtaking_sector_tuner, lap_analyser, set_pose, random_obstacle_publisher,
  grid_filter, polygon_filter, vel_planner, gb_optimizer, spliner, spliner_planner, sqp_planner,
  recovery_spliner, lane_change_planner (ported from ros1); + sim (f1tenth_gym_ros, obstacle_publisher,
  stack_master, tools/raycaster, pitwall).
- HARDWARE (build in Docker w/ apt deps, not sim-validated): vesc*, urg_node.
- apt (not built): cartographer, cartographer_ros.
- EXCLUDED: blink1, car_to_car_sync, on_track_sys_id, gp_traj_predictor, slam_tuner, f1tenth_simulator.
