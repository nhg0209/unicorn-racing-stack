# UNICORN Stack — Working Rules (read first)

## Process
- NEVER run `colcon build` or any build command. The user builds manually. After edits, state which packages need rebuilding.
- NEVER push to `origin` (HMCL-UNIST official). Push only to `personal` remote, and only when asked.
- Commit messages in English, scoped like existing history (`fix(static-avoidance): ...`). User-facing explanations in Korean.
- Do not modify mapping-related code (cartographer configs, mapping_2d.lua, finish_map.py). localization_2d.lua num_subdivisions_per_laser_scan stays 10.
- Plan-first: before editing, list files you will touch and why; keep diffs minimal. Line numbers given in prompts are evidence pointers — re-verify against the actual code before editing.

## Codebase boundaries
- DORMANT — do not modify/launch/"fix": spliner_node, start_spline_node(v1), spliner_planner/*, sqp_planner/*, predictor_opponent_trajectory, dynamic_prediction_server, state_estimation/carstate_node. Only lane_change_planner's update_waypoints is live.
- Single sources of truth: speed scaling = sector_tuner only; /global_waypoints ownership = global_republisher XOR static_reopt_node. Never add another publisher.
- Params: there is no single convention in this subsystem, so check the node before assuming one.
  - static_avoidance_node — stack_master/config/static_avoidance_params.yaml + declare_parameter + a dyn_param_cb branch + the startup sync list. NO save_params. A declare without a branch is silently never applied (planner/spliner/test/test_param_wiring.py guards this).
  - static_reopt_node — parameters come from stack_master/launch/base_system.launch.xml (reopt_* args, forwarded by race.launch.xml), NOT from a yaml, and there is no reconfigure callback: they are read once at construction. check_avoidance_margins.py verifies the two launch files agree.
  - static_obstacle_layer — no yaml and no launch params at all; every value is the declare_parameter default in the node.
  - multi_tracking — opponent_tracker_params.yaml + save_yaml (rqt save button). That save must update the block PER KEY; assigning a fresh dict deletes every key it does not list.
- Check race.launch.xml remaps before renaming topics (/planner/avoidance/otwpnts is remapped per-planner; /planner/avoidance/static_feasible deliberately NOT remapped). Planner gates are fail-closed — preserve that direction.

## Coupled invariants
- Margin chain (line-center → obstacle-EDGE everywhere): SM static GB requirement (gb_ego_width_m/2 + lateral_width_static_gb_m) ≤ planner clear-gate stay (width_car/2 + clear_margin_m) ≤ enforced reopt clearance floor − slack. After ANY margin change run python3 stack_master/scripts/check_avoidance_margins.py (exit 0).
- Curvature/lateral-accel judgments use real geometry (Menger), never the kappa_clean + d''-additive model (proven ~45% low on ifac).
- hold_horizon_m (lane_change) > interest_horizon_m (SM); static planner lookahead ≥ global_tracking max_horizon.

## Verification (offline, no build)
PREREQUISITE for every command below — without it 14 tests fail to collect on `f110_msgs`:
```
source /opt/ros/jazzy/setup.bash && source ~/unicorn_ws/install/setup.bash
```
Offline scripts run under the conda interpreter (`~/miniforge3/envs/unicorn/bin/python3`); the
system python3 has no trajectory_planning_helpers.

- Margins/launch agreement: `python3 stack_master/scripts/check_avoidance_margins.py` (exit 0)
- Speed continuity: `python3 stack_master/scripts/test_speed_continuity.py`
- Track bounds: `python3 stack_master/scripts/check_track_bounds.py --all` — **exit 1 today**: map f ships with d_left/d_right SWAPPED (402 stations vs 0). ifac and map_test are correct. Do NOT run `--fix` casually, and treat any sweep result taken on map f as measured through an inverted corridor.
- Unit tests (151): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest state_machine/test planner/spliner/test planner/lane_change_planner/test controller/test planner/gb_optimizer/scripts perception/scripts -q`
  (the env var is REQUIRED: launch_testing_ros's pytest entrypoint aborts collection in this env)
  The gb_optimizer and perception suites are ALSO runnable standalone, which is how they report their own measurements:
  `~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/test_static_reopt_apex.py` (53 checks),
  `.../test_static_obstacle_layer.py`, `perception/scripts/test_static_classification.py`, `.../test_save_yaml_roundtrip.py`
- Reopt geometry: `~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/sweep_static_reopt.py --check`
- Static-avoidance feasibility on the real map: `~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_feasibility.py --check` (~4 min)
- NOT collected by pytest and not gates: `planner/gb_optimizer/scripts/test_static_reopt.py` and `planner/lane_change_planner/scripts/probe_grid_corridor.py` are diagnostics that print tables. The pep257/flake8/copyright suites under state_estimation, TODO/ and race_utils/ error out on collection and are not part of any gate.
- Sim: stack_master/STATIC_AVOIDANCE_TEST_RUNBOOK.md (S1–S5). Never claim "verified" without running the gate; paste actual output.
