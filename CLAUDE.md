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
  - static_reopt_node — stack_master/config/static_reopt_params.yaml (top-level key `static_reopt_node`), attached with a single `<param from=>`. `reopt` (enable) is the only launch arg left, because that is a per-run choice. There is STILL no reconfigure callback: every value is read once at construction, so a change needs a restart — a yaml here means one source, not live tuning. check_avoidance_margins.py reads that yaml and asserts declare/YAML symmetry, because a key the node never declares is silently ignored and a name that does not match falls back to a default in silence.
  - closed_reopt (the solver) — exposed as `closed_qp_*` in the same yaml. The ReoptParams dataclass KEEPS its defaults so the module still runs with no ROS at all (every offline gate depends on that); the yaml overrides them. w_veh is deliberately absent — it is the same quantity as qp_veh_width and is passed from there. check_avoidance_margins.py reads the yaml (dataclass as fallback) and enforces that chain; it is short of the required headroom by 0.010 m (0.320 delivered vs 0.280 needed + 0.05 required) and that shortfall is carried by a NAMED exception (CLOSED_QP_SHORTFALL_ALLOWED_M) which prints a banner on every run. FLOOR_CONSUMER_SLACK_M was NOT lowered; the exception names its own evidence and how to end it.
  - static_obstacle_layer — no yaml and no launch params at all; every value is the declare_parameter default in the node.
  - multi_tracking — opponent_tracker_params.yaml + save_yaml (rqt save button). That save must update the block PER KEY; assigning a fresh dict deletes every key it does not list.
- Check race.launch.xml remaps before renaming topics (/planner/avoidance/otwpnts is remapped per-planner; /planner/avoidance/static_feasible deliberately NOT remapped). Planner gates are fail-closed — preserve that direction.

## Static re-optimization: one solver
- `closed_reopt.py` — pick a side per box -> turn each keep-out into corridor bounds analytically -> multiply the corridor by a hard envelope (locality x remaining curvature budget) -> solve `min ||D2 d||^2 + w||d||^2 s.t. lo <= d <= hi` ONCE with quadprog, no iteration, at every published station -> anything it cannot clear by obs_margin + w_veh/2 is dropped and NAMED on /static_reopt/coverage for the reactive layer. `closed_reopt_bridge.py` shapes the result into what the node's vetoes read.
- The C2-quintic-hump pipeline (`reoptimize_local_window`, the apex/knot machinery, the reach search, the relax ladder) and the whole-track `reoptimize_with_obstacles` are DELETED — `git log -- planner/gb_optimizer/gb_optimizer/static_reopt_core.py` has them. There is no `reopt_method`: with a fallthrough branch, any typo in it selected a solver that takes minutes per solve.
- `static_reopt_core.py` is now only what the bridge and the node still need: the publishing tail (uniform resample, curvature republish, velocity/accel), `_wrap_normals`, `build_wpnts`, `load_reftrack`.

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
- Track bounds: `python3 stack_master/scripts/check_track_bounds.py --all` — **exit 1 today**: map f (402 stations vs 0) ships with d_left/d_right SWAPPED. ifac, ifac_0807 and map_test are correct. **ifac_0807 was swapped and has been REGENERATED** (127 stations vs 0 at HEAD, 124 vs 0 in the working tree); measurements taken on it before that regeneration went through an inverted corridor and are not comparable to ones taken after. The GENERATOR is fixed (sides now come from the final centerline's geometry — planner/gb_optimizer/gb_optimizer/track_bounds.py, shared with this checker), so a map regenerated from now on is labelled correctly; f was written by the old code and is still on disk. `--fix` relabels it and makes avoidance work, but it is NOT a full repair: the widths went through write_centerline into the trajectory optimizer, so the raceline itself was optimized inside a mirrored corridor and the map needs REGENERATING — which is what ifac_0807 got. Treat any sweep taken on a SWAPPED map as measured through an inverted corridor — the sweeps now detect it per map and say so, instead of consulting a hardcoded list that a new map was never added to.
- Unit tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest state_machine/test planner/spliner/test planner/lane_change_planner/test controller/test planner/gb_optimizer/scripts perception/scripts race_utils/unicorn_gym/f1tenth_gym_ros/test/test_ego_footprint.py -q`
  (the env var is REQUIRED: launch_testing_ros's pytest entrypoint aborts collection in this env)
  **202 as of round 6** (192 before it). QUOTE THE TOTAL, and if it drops, account for the drop
  before anything else. FROM A WORKTREE the last path is EMPTY: git does not populate submodules in
  a worktree and `git submodule update --init race_utils/unicorn_gym` cannot fix it (the pinned
  commit b41910d is local-only -- the remote answers `upload-pack: not our ref`). That silently
  removes exactly 16 tests, which is a plausible-looking number and not a regression. Run it from a
  worktree with the MAIN checkout's absolute path for that one file:
  `.../unicorn-racing-stack/race_utils/unicorn_gym/f1tenth_gym_ros/test/test_ego_footprint.py`.
  The gb_optimizer and perception suites are ALSO runnable standalone, which is how they report their own measurements:
  `~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/test_static_reopt_node.py` (25 checks: the swap/publish/concurrency safety machinery, named per check in its docstring),
  `.../test_static_obstacle_layer.py`, `perception/scripts/test_static_classification.py`, `.../test_save_yaml_roundtrip.py`
- Reopt geometry: `~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/gate_closed_reopt.py --check` (C1-C9 + the node contract; exit code). This replaced `sweep_static_reopt.py --check` as the reopt regression gate when the hump went — that script's five checks were all hump geometry. Margin sweep: `sweep_obs_margin.py`; raceline curvature sweep: `sweep_raceline_curvlim.py`.
- Static-avoidance feasibility on the real map: `~/miniforge3/envs/unicorn/bin/python3 planner/spliner/scripts/sweep_static_feasibility.py --check` (~4 min).
  Its CORNER-CELL count is WALL-CLOCK GATED (the ladder runs under ramp_search_max_ms), so it is not
  reproducible to the cell: measured five times on unchanged code it reads 55-58 of 99 on ifac. Read
  a difference of two as noise; the unbudgeted count (67 on ifac, 18 on ifac_0807) is the geometry's
  own answer and does not move.
- Which d(s) generator: `static_plan_method` in static_avoidance_params.yaml, `sample` (ships) or
  `corridor_qp` (planner/spliner/spliner/corridor_path.py, pure numpy/scipy/quadprog, no ROS).
  Compare them on the same cells: `sweep_static_race.py --method all` (R1-R4; take timings at
  `--jobs 1`, they are wall clock). The QP module's own checks: `planner/spliner/test/test_corridor_path.py`.
  See planner/spliner/CORRIDOR_QP_NOTE.md for what each number means and which of them is a trade.
- NOT collected by pytest and not gates: `planner/lane_change_planner/scripts/probe_grid_corridor.py` is a diagnostic that prints tables. The pep257/flake8/copyright suites under state_estimation, TODO/ and race_utils/ error out on collection and are not part of any gate.
- Sim: stack_master/STATIC_AVOIDANCE_TEST_RUNBOOK.md (S1–S5). Never claim "verified" without running the gate; paste actual output.
