# Static-avoidance validation runbook (S1–S5)

Validates the Frenet grid-sampling static-avoidance planner + the state-machine
persistence/hysteresis/feasibility changes. **Requires `colcon build` first** (the packages below
changed); the two helper scripts run with plain `python3` (no rebuild needed for them).

```bash
cd ~/unicorn_ws
colcon build --packages-select spliner state_machine perception f110_msgs
source install/setup.bash
```

## 0. Margin consistency check (after ANY margin tuning)

The re-opt line must clear obstacles by more than the reactive keep-out or it gets re-avoided
every lap (double avoidance), AND its box-edge clearance (keep-out + apex_bulge) must exceed the
state machine's static GB free requirement (gb_ego_width/2 + lateral_width_static_gb_m) or the
swapped line reads blocked -> phantom TRAILING. Run after touching `static_avoidance_params.yaml`
(`width_car`/`safety_margin`/`apex_bulge`), `state_machine_params.yaml` (`gb_ego_width_m`/
`lateral_width_static_gb_m`), or the `reopt_obs_margin` launch arg:

```bash
python3 stack_master/scripts/check_avoidance_margins.py   # exit 0 = consistent
```

## 1. Bring up the sim (one terminal)

```bash
# reactive static avoidance only:
ros2 launch stack_master race.launch.xml map:=ifac sim:=true ot_planner:=predictive_spliner
# (add reopt:=true reopt_safety_width:=0.25 to also run the Layer-2 obstacle-aware global line)
```

Optional: regulation-ish square box size (default manager size is 0.30 m):
```bash
ros2 param set /static_obstacle_manager size 0.32
```

Lap timing is NOT started by race.launch — run it for lap-time metrics:
```bash
ros2 launch lap_analyser lap_analyser.launch.py
```

## 2. Spawn the scenario obstacles (second terminal)

`--inject overlay` (default) drives the box through the REAL detect→multi_tracking pipeline, so it
exercises the position-persistence classifier (`is_static`, `s_var/d_var`, demotion guard). Use
`--inject merge` for a deterministic ground-truth obstacle that tests the planner/SM in isolation.

```bash
# pick --s0 so the obstacle(s) land on the intended track feature; S3 wants --s0 just after a corner
python3 stack_master/scripts/spawn_static_obstacle.py --scenario S1 --s0 8.0     # straight single
python3 stack_master/scripts/spawn_static_obstacle.py --scenario S2 --s0 8.0     # slalom (3 m alt.)
python3 stack_master/scripts/spawn_static_obstacle.py --scenario S3 --s0 <corner-exit s>
python3 stack_master/scripts/spawn_static_obstacle.py --scenario S4 --s0 8.0     # ~40 cm gap
# S5: spawn nothing (raceline regression). Clear between runs:
python3 stack_master/scripts/spawn_static_obstacle.py --clear
```
(You can also place obstacles by hand with the RViz **Publish Point** tool, or `--obs "8,0.0; 11,-0.4"`.)

## 3. Record metrics (third terminal)

```bash
python3 stack_master/scripts/avoidance_metrics.py --label S1 --collision-thresh 0.05
# ...let the car complete a lap or two past the obstacle, then Ctrl-C for the summary row.
```

## 4. What to check per scenario

Fill this table from each run's summary:

| Scenario | Collision | Lap time [s] | Planner latency mean/max [ms] | State transitions (chatter) | Min clearance [m] |
|----------|-----------|--------------|-------------------------------|-----------------------------|-------------------|
| S1 straight single       | | | | | |
| S2 slalom (old failure)  | | | | | |
| S3 corner-exit           | | | | | |
| S4 ~40 cm gap            | | | | | |
| S5 no obstacle (regress) | no | ≥ baseline | | low | n/a |

Pass criteria:
- **No collision** in S1–S4; car passes each obstacle without stopping (no-stop rule).
- **Planner latency max < 10 ms** (`/planner/avoidance/latency`, `measure:true` is set in the yaml).
- **`/planner/avoidance/static_feasible` toggles** (`ros2 topic echo /planner/avoidance/static_feasible`);
  when it is `False` with a static obstacle ahead, the SM must stay TRAILING (no OVERTAKE).
- **RViz**: `/planner/avoidance/markers` shows grey candidates, red rejected, green selected.
- **Low state-transition count** (min_dwell hysteresis working — no OVERTAKE⇄GB_TRACK chatter).
- **S5 lap time not worse** than the pre-change baseline (record a clean S5 lap before/after).

Re-run S1–S5 after any tuning of `stack_master/config/static_avoidance_params.yaml`,
`state_machine_params.yaml`, or `opponent_tracker_params.yaml`.

## 5. Master acceptance: full 8-lap IFAC scenario (reopt + removal)

Simulates the race format end-to-end: 4 laps WITH static obstacles (reactive avoidance →
re-opt swap), then obstacles removed mid-race → fast clean-line revert for the last 4 laps.

```bash
# bring-up WITH the obstacle-aware global line:
ros2 launch stack_master race.launch.xml map:=ifac sim:=true reopt:=true reopt_safety_width:=0.25
ros2 launch lap_analyser lap_analyser.launch.py

# lap 1: spawn (S1 or S2 placement), let the car avoid reactively; watch for
#   "[static_reopt] batch re-opt -> OBSTACLE-AWARE ... ready" then "swapped to OBSTACLE-AWARE"
python3 stack_master/scripts/spawn_static_obstacle.py --obs "12.0,0.0" --inject merge

# laps 2-4: on the re-opt line the SM must stay GB_TRACK past the obstacle
# (no OVERTAKE flip-flop; /planner/avoidance/static_otwpnts silent) — margin coupling check.
#   expect in the planner log: "raceline clears all N obstacle(s) ahead ... -> planner idle"
#   (the raceline-clear gate; tracking re-projects obstacle (s,d) into the swapped frame).
#   The swap log "swapped to OBSTACLE-AWARE at s=..." must lie OUTSIDE the avoidance hump —
#   the commit gate requires old/new agreement over the whole look-ahead horizon (>=3 m / 1 s).

# TWO-OBSTACLE acceptance (the IFAC format): spawn a pair, e.g.
#   python3 stack_master/scripts/spawn_static_obstacle.py --obs "12.0,0.0; 20.0,-0.3" --inject merge
# expect BOTH apexes captured (max_weave=2 lets the reactive path weave a close pair) and
#   "... 2/2 obstacle apex(es) reshaped ..." in the re-opt log. If the track is too tight at one
#   of them, expect the honest "N apex(es) CORRIDOR-REJECTED ... want X corridor max Y" warning
#   instead — that obstacle stays reactive-only BY DESIGN (no shrunken half-hump is laid).
# The published line must stay within the friction budget: spot-check implied lateral accel
#   (vx^2 * kappa) of /global_waypoints stays <= ggv ay_max (~4.5 for SIM) through the humps.

# SWAP-MOMENT lookahead check: on any "swap deadlock breaker ... (car X.XX m off the new line)"
#   commit, the L1 marker must stay bounded (l1_lat_err_cap, ~1 m) while the car converges onto
#   the new line — no corner-cut into the wall (regression: uncapped sqrt(2)*lat_err lower bound).

# NEAR-START obstacle (regression for the late-swap case): spawn just past the start line —
#   python3 stack_master/scripts/spawn_static_obstacle.py --obs "1.0,0.0" --inject merge
# even when the layer confirms it only AFTER the pass, expect "retro-associated apex(es)
# from N buffered reactive path(s)" and the OBSTACLE-AWARE swap within the same/next lap
# (previously: 0-apex build burned the rebuild trigger -> reactive re-avoidance every lap).

# "obstacle removal" between lap 4 and 5:
python3 stack_master/scripts/spawn_static_obstacle.py --clear
#   expect within ~1 s: "[static_obs_layer] UNLATCHED ..." (sighting streak) OR the miss-lap
#   removal on the next lap; then "[static_reopt] batch re-opt -> CLEAN ready (obstacles cleared)"
#   and "swapped to CLEAN" at the next agreement point.
# backup/manual path (also the bench reset):
ros2 topic pub --once /static_reopt/clear_obstacles std_msgs/msg/Empty
```

Measure across the 8 laps: collision count (0), per-lap lap time, laps-to-revert after the
clear (target: swap committed before the NEXT lap completes), SM state timeline
(`/state_machine`), AEB warn count in the controller log (0 during clean avoidance).

Unit tests for the layer + apex/commit logic (no sim needed):
```bash
python3 planner/gb_optimizer/scripts/test_static_obstacle_layer.py
python3 planner/gb_optimizer/scripts/test_static_reopt_apex.py
```
