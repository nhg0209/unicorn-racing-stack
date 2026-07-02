# Static-avoidance validation runbook (S1–S5)

Validates the Frenet grid-sampling static-avoidance planner + the state-machine
persistence/hysteresis/feasibility changes. **Requires `colcon build` first** (the packages below
changed); the two helper scripts run with plain `python3` (no rebuild needed for them).

```bash
cd ~/unicorn_ws
colcon build --packages-select spliner state_machine perception f110_msgs
source install/setup.bash
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
