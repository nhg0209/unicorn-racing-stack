# UNICORN Stack — Working Rules (read first)

## Process
- NEVER run `colcon build` or any build command. The user builds manually. After edits, state which packages need rebuilding.
- NEVER push to `origin` (HMCL-UNIST official). Push only to `personal` remote, and only when asked.
- Commit messages in English, scoped like existing history (`fix(static-avoidance): ...`). User-facing explanations in Korean.
- Do not modify mapping-related code (cartographer configs, mapping_2d.lua, finish_map.py). localization_2d.lua num_subdivisions_per_laser_scan stays 10.
- Plan-first: before editing, list files you will touch and why; keep diffs minimal. Line numbers given in prompts are evidence pointers — re-verify against the actual code before editing.

## Codebase boundaries
- DORMANT — do not modify/launch/"fix": spliner_node, start_spline_node(v1), spliner_planner/*, sqp_planner/*, predictor_opponent_trajectory, dynamic_prediction_server, state_estimation/carstate_node.
- LIVE planners (race.launch.xml): change_avoidance_node (dynamic, OPEN/HOLD/CLOSE lane-hold) + update_waypoints from lane_change_planner, static_avoidance_node from spliner, static_reopt_node + static_obstacle_layer from gb_optimizer.
- Single sources of truth: speed scaling = sector_tuner only; /global_waypoints ownership = global_republisher XOR static_reopt_node. Never add another publisher.
- Params: there is no single convention in this subsystem, so check the node before assuming one.
  - static_avoidance_node — stack_master/config/static_avoidance_params.yaml + declare_parameter + a dyn_param_cb branch + the startup sync list. NO save_params. A declare without a branch is silently never applied (planner/spliner/test/test_param_wiring.py guards this).
  - static_reopt_node — parameters come from stack_master/launch/base_system.launch.xml (reopt_* args, forwarded by race.launch.xml), NOT from a yaml, and there is no reconfigure callback: they are read once at construction. check_avoidance_margins.py verifies the two launch files agree.
  - closed_reopt (reopt_method:=closed_qp) — a FIFTH convention: its six parameters are dataclass defaults in planner/gb_optimizer/gb_optimizer/closed_reopt.py (ReoptParams), reachable from no yaml and no launch arg. Only the solver CHOICE is a launch arg. check_avoidance_margins.py parses the dataclass and reports that chain separately; today it FAILS on headroom (0.320 delivered vs 0.280 needed + 0.05 required) and is non-fatal only because local_window is still the default.
  - static_obstacle_layer — no yaml and no launch params at all; every value is the declare_parameter default in the node.
  - multi_tracking — opponent_tracker_params.yaml + save_yaml (rqt save button). That save must update the block PER KEY; assigning a fresh dict deletes every key it does not list.
- Check race.launch.xml remaps before renaming topics (/planner/avoidance/otwpnts is remapped per-planner; /planner/avoidance/static_feasible deliberately NOT remapped). Planner gates are fail-closed — preserve that direction.

## Static re-optimization: the two solvers
- `reopt_method:=local_window` (DEFAULT, "hump") — one C2 quintic hump per RECORDED REACTIVE APEX, so a box is only covered after the reactive layer has driven past it once. static_reopt_core.py.
- `reopt_method:=closed_qp` (opt-in, unproven in sim) — closed_reopt.py: pick a side per box -> turn each keep-out into corridor bounds analytically -> multiply the corridor by a hard envelope (locality x remaining curvature budget) -> solve `min ||D2 d||^2 + w||d||^2 s.t. lo <= d <= hi` ONCE with quadprog, no iteration -> restore every station with a periodic cubic. Any box it cannot clear by obs_margin + w_veh/2 is dropped and named on /static_reopt/coverage for the reactive layer. closed_reopt_bridge.py makes the result indistinguishable from local_window's, so every veto in the node applies unchanged.

## Coupled invariants
- Margin chain (line-center → obstacle-EDGE everywhere): SM static GB requirement (gb_ego_width_m/2 + lateral_width_static_gb_m) ≤ planner clear-gate stay (width_car/2 + clear_margin_m) ≤ enforced reopt clearance floor − slack. After ANY margin change run python3 stack_master/scripts/check_avoidance_margins.py (exit 0).
- Curvature/lateral-accel judgments use real geometry (Menger), never the kappa_clean + d''-additive model (proven ~45% low on ifac).
- ifac's own raceline peaks at |kappa| 1.448 against a curvlim of 1.5 (96.5%). Two things are forbidden and one is not. FORBIDDEN: comparing your published curvature to ANOTHER solver's published curvature (the hump's uniform resample reads 1.4378 where the raceline it was built from measures 1.4478 — you would be measuring the resampler); and anchoring a gate to the CLEAN line's peak, which penalises a flatter raceline for being flatter (a candidate whose published worst was 1.246 failed a gate a shipped line passed at 1.451). FINE: checking that a line stays inside the budget its own formulation declared (closed_reopt's kappa_budget) — that anchor does not move when the raceline is regenerated. Report clean-line increments; never gate on them.
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
- Reopt geometry (hump): `~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/sweep_static_reopt.py --check`
- Reopt geometry (closed_qp): `~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/gate_closed_reopt.py --check` (C1-C8 + the node contract). Head-to-head against the hump: `compare_reopt.py`; margin sweep: `sweep_obs_margin.py`.
- Static-avoidance feasibility on the real map: `~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_feasibility.py --check` (~4 min)
- NOT collected by pytest and not gates: `planner/gb_optimizer/scripts/test_static_reopt.py` and `planner/lane_change_planner/scripts/probe_grid_corridor.py` are diagnostics that print tables. The pep257/flake8/copyright suites under state_estimation, TODO/ and race_utils/ error out on collection and are not part of any gate.
- Sim: stack_master/STATIC_AVOIDANCE_TEST_RUNBOOK.md (S1–S5). Never claim "verified" without running the gate; paste actual output.
