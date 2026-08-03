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
- Params live in stack_master/config/*.yaml. New param = declare in node + YAML entry + keep save_params round-trip.
- Check race.launch.xml remaps before renaming topics (/planner/avoidance/otwpnts is remapped per-planner; /planner/avoidance/static_feasible deliberately NOT remapped). Planner gates are fail-closed — preserve that direction.

## Coupled invariants
- Margin chain (line-center → obstacle-EDGE everywhere): SM static GB requirement (gb_ego_width_m/2 + lateral_width_static_gb_m) ≤ planner clear-gate stay (width_car/2 + clear_margin_m) ≤ enforced reopt clearance floor − slack. After ANY margin change run python3 stack_master/scripts/check_avoidance_margins.py (exit 0).
- Curvature/lateral-accel judgments use real geometry (Menger), never the kappa_clean + d''-additive model (proven ~45% low on ifac).
- hold_horizon_m (lane_change) > interest_horizon_m (SM); static planner lookahead ≥ global_tracking max_horizon.

## Verification (offline, no build)
- Planner/SM: python3 stack_master/scripts/check_avoidance_margins.py; python3 stack_master/scripts/test_speed_continuity.py
- Unit tests: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest state_machine/test planner/spliner/test controller/test -q
  (the env var is REQUIRED: launch_testing_ros's pytest entrypoint aborts collection in this env)
- Reopt: python3 planner/gb_optimizer/scripts/sweep_static_reopt.py --check
- Sim: stack_master/STATIC_AVOIDANCE_TEST_RUNBOOK.md (S1–S5). Never claim "verified" without running the gate; paste actual output.
