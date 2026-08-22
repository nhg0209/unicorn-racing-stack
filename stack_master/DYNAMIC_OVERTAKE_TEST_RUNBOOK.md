# Dynamic Overtaking Test Runbook (D0–D8)

Sim validation for the **dynamic** overtaking path: `planner_change`
(`lane_change_planner/change_avoidance_node.py`) → `/planner/avoidance/otwpnts` →
state machine `OVERTAKE`. The static-path analogue is `STATIC_AVOIDANCE_TEST_RUNBOOK.md`;
this file follows the same structure.

Everything below runs against a built + sourced workspace. No `colcon build` is needed for
yaml or script changes (`--symlink-install`); Python **node** changes do need a rebuild.

---

## §0 — Margin consistency (run first, always)

The dynamic chain couples four files. The planner mirrors the SM's clearance inputs and
derives its own numbers from them, so a mirror that drifts silently produces a planner that
publishes lanes the SM rejects — the car then flaps `OVERTAKE ↔ TRAILING` with neither side
giving up.

```bash
python3 stack_master/scripts/check_avoidance_margins.py   # exit 0 required
```

Checks the static chain **and** (`--- dynamic chain ---`) that:
`sm_gb_ego_width_m` / `sm_lateral_width_m` in `lane_change_params.yaml` still equal the real
values; the derived monitor line sits **above** the SM accept line and **below** the solver
target; `hold_horizon_m > interest_horizon_m`; `engage_gap_m < 10` (the SM's
`getting_closer` window).

Unit test of the phase machine (no ROS running needed):

```bash
python3 planner/lane_change_planner/test/test_phase_machine.py   # 60/60, exit 0
python3 planner/lane_change_planner/test/test_grid_corridor.py   # f-map corridor vs waypoints
```

`test_grid_corridor.py` needs no ROS running: it drives the real solver against the real f map
and its real occupancy grid. Its first line must report ~100 % of raceline points inside the
eroded free space — anything less means the grid pixel convention is off and the rest is noise.

---

## §1 — Bring-up

```bash
source ~/unicorn_ws/src/unicorn-racing-stack/unicorn.sh     # or: unicorn

# T1 — stack + gym + RViz + virtual_perception, opponent already driving
ros2 launch stack_master race.launch.xml map:=f sim:=true \
     ot_planner:=predictive_spliner opp_spawn:=true opp_mode:=path

# T2 — lap timing (race.launch does NOT start it)
ros2 launch lap_analyser lap_analyser.launch.py
```

**Map choice.** Use **`f`** (`ot_flag: true` for the whole lap, L = 76.5 m, width 2.23–3.74 m).
`ifac` also has `ot_flag: true` but is 0.87–2.29 m wide — keep it for the narrow-track refusal
case (D3). **`map_test` has all six sectors `ot_flag: false`, so OVERTAKE is structurally
impossible there** — a run on it proves nothing.

`opp_spawn` / `opp_mode` are new launch args. They exist because `low_level.launch.xml` used
to pass `virtual_opponent`, a name `virtual_perception.launch.xml` does not declare, so it had
no effect and the opponent always came up unspawned and in `manual` mode — sitting still and
invisible.

### Placing the opponent at a known gap

`opp_spawn:=true` puts the opponent at the map's default spawn. For a **reproducible initial
gap** — which every D-scenario needs, because the gates are gap-dependent — use:

```bash
python3 stack_master/scripts/spawn_opponent.py --gap 20 --speed 1.5           # 20 m ahead
python3 stack_master/scripts/spawn_opponent.py --s0 34.0 --d 0.5 --mode ftg   # absolute s
python3 stack_master/scripts/spawn_opponent.py --remove
```

Two gotchas the script works around:

- The opponent's `max_speed` is read **once** at node startup and has no parameter callback, so
  `ros2 param set /opponent_controller max_speed` does nothing. Only the delta topic
  `/sim/opp_speed_delta` works; `--speed` converts to a delta for you (pass `--current` if you
  already nudged it this session).
- In `path` mode the opponent follows `centerline.csv`, which carries no speeds, so it cruises
  at a **constant** speed and does not slow for corners. That is what you want for a regression
  run — it is also why `path` (not `ftg`) is the default.

---

## §2 — Reading the decision chain

Both sides of the gate now log. One run is enough to attribute a failure.

**Planner** (`[LaneChange] …`):

| Line | Means |
|---|---|
| `IDLE: <reason>` (1 Hz) | not engaging; the reason is the per-gate verdict for the nearest dynamic obstacle |
| `IDLE: no lane fits: … L@+X.Xm[dl=… dr=… oppd=…] L:corridor short X.XXm` | geometry refusal, per side, with the exact shortfall and the opponent-d actually used |
| `IDLE: ot_section gate closed/stale` | `/ot_section_check` false or older than `ot_gate_stale_s` |
| `IDLE -> ENTRY (engage): target id=… gap=… side=… meet_in=… v_pass=… v_opp=…` | the engage moment and all its numbers; the lane is COMMITTED here |
| `ENTRY -> HOLD (on lane, dev=…)` | the car reached the committed lane; from here it is followed, not re-solved |
| `HOLD re-plan (<reason>)` / `ENTRY re-plan (…)` | the committed lane was replaced — reason is opponent lateral drift, meeting-point drift, or car off the lane. **Frequent re-plans mean the commit thresholds are too tight and the churn is back** |
| `waypoint bounds disagree with the measured corridor by … SWAPPED` | this map's `d_left`/`d_right` are mirrored; the grid corridor is being used instead |
| `HOLD -> CLOSE (passed …)` / `(target gone)` | pass complete |
| `<PHASE> -> IDLE (<reason>)` | abort; reason is one of *target lost / target pulled away / lane no longer clears target / path infeasible / blocked, dropping back / merge complete* |
| `lane short of target clearance … too close alongside to abort` | clearance failed while level with the opponent — the lane is deliberately held |

**State machine**:

| Line | Means |
|---|---|
| `dyn_OT blocked: <term>` (0.5 Hz) | **the** dynamic-gate diagnostic — names the first failing AND-term with measured-vs-threshold: `ot_sector` / `getting_closer` / `latest+on_spline` / `free_frenet` |
| `OT path NOT free: … path age=…s lookup_err=…m` | free-check rejected via the prediction branch |
| `OT path NOT free (no pred): … predict_dyn=…` | free-check rejected via the fallback branch (normal early in a run — predictions need a clean opponent half-lap first) |
| `OVERTAKE dropped: avoidance path not free for X s` | sustain lost on freeness |
| `OVERTAKE dropped: avoidance path unavailable` | sustain lost on staleness |
| `static_OT check: …` | the STATIC gate. Useful here as a **lag probe** — see below |

```bash
ros2 topic echo /rosout --field msg | grep -E 'LaneChange|dyn_OT|OVERTAKE dropped'
```

### Topic-level checks (no code changes)

```bash
ros2 topic hz   /planner/avoidance/otwpnts   # ~20 Hz — but IDLE is SILENT by design
ros2 topic hz   /state_machine               # ~80 Hz; well below = executor saturated
ros2 topic echo /ot_section_check            # true on f/ifac, ~80 Hz
ros2 topic echo --once /tracking/obstacles   # is_static:false, sane vs
ros2 topic echo /state_machine               # GB_TRACK|TRAILING|OVERTAKE|RECOVERY|…
ros2 topic hz   /opponent_prediction/obstacles_pred   # silent until an opponent half-lap exists
```

**Measuring the executor lag** (the thing `latest_threshold` has to tolerate): the static gate
already prints the age of a message on the *same* 20 Hz `OTWpntArray` path every cycle.

```bash
ros2 topic echo /rosout --field msg | grep -o 'static_OT check.*'
#  -> latest+on_spline=…[n=221,age=0.43(<1.0),…]      age is the number that matters
```

`ros2 param set /state_machine dynamic_avoidance_planner.latest_threshold` does **not** work —
`WaypointData` caches planner yaml values at construction. Edit the yaml and relaunch.

---

## §3 — Metrics recorder

```bash
# widen the pitwall recorder to the dynamic-overtaking topics
ros2 launch pitwall record.launch.py output_dir:=~/runs/dynOT_D1 \
  topic_regex:='/pitwall/.*|/state_machine|/behavior_strategy|/local_waypoints|/car_state/odom|/car_state/odom_frenet|/opp_racecar/odom|/sim/dynamic_obstacles|/tracking/obstacles|/planner/avoidance/.*|/ot_section_check|/opponent_prediction/.*|/lap_data|/rosout'

# live console summary
python3 stack_master/scripts/avoidance_metrics.py --label D1 \
        --latency-topic /planner/lane_change/latency
```

`--latency-topic` is required for dynamic runs: the default is the **static** planner's topic,
so a dynamic run would otherwise record the wrong planner (or nothing). The dynamic planner
only publishes latency when `measure: true` is set in `lane_change_params.yaml`.

Including `/rosout` in the bag puts every gate log on the same timeline as the poses, so a run
can be attributed after the fact from the file alone. `/opp_racecar/odom` is **ground truth**
for the opponent — use it for clearance, not the tracked estimate.

---

## §4 — Scenarios

Each: N ≥ 5 laps, same opponent speed, same spawn `s`. Record the metrics in §5.

| ID | Setup | Isolates |
|---|---|---|
| **D0** | no opponent, 3 laps | lap-time baseline; state stays `GB_TRACK` |
| **D1** | `--gap 20 --speed 1.5`, straight | **"does it engage at all"** — the primary regression |
| **D2** | `--gap 3 --speed 1.5` (inside `engage_gap_m`) | slope escalation `lane_max_slope`→`lane_max_slope_close`, entry-blend cap |
| **D3** | spawn so the meeting point lands on the narrowest section (`f` 2.23 m; repeat on `ifac` 0.87 m) | a clean pass **or** a logged `no lane fits: … corridor short X m` with *sustained* TRAILING. **Both are PASS; flapping between them is FAIL** |
| **D4** | mid-approach `/sim/opp_speed_delta` +1.5 then −1.5 | `engage_min_closing_mps`, `target pulled away`, `CLOSE → HOLD` re-entry |
| **D5** | `ros2 topic pub /sim/ego_lidar_enable std_msgs/msg/Bool "{data: false}"` for ~0.5 s mid-HOLD | `target_lost_s` coasting + `ot_free_lost_sec`. One blank frame must not drop OVERTAKE |
| **D6** | `--mode ftg` (opponent wanders off line) or `--d 0.8` | side selection, `obs_traj_tresh`, and the `tracking_merger` `max_obstacle_d = 1.0` cliff |
| **D7** | D1 twice: `--inject overlay`, then `--inject merge` | separates perception (size clamp to 0.30, id churn, `is_static` lag) from planner+SM logic |
| **D8** | 8 laps, `--speed` ≈ 60 % of raceline pace | **master acceptance** |

---

## §5 — Pass criteria

| Metric | Source | Pass |
|---|---|---|
| State sequence `GB_TRACK → TRAILING → OVERTAKE → GB_TRACK` completes | `/state_machine` | ≥1 per scripted pass; OVERTAKE held ≥ 1.0 s |
| Planner phases `IDLE → ENTRY → HOLD → CLOSE → IDLE (merge complete)` | `/rosout` | all present, in order. Any other `-> IDLE (…)` is an abort — record the reason |
| **Committed-lane churn** | `/rosout` grep `re-plan` | **≤ 1 re-plan per pass.** More than that and the car is chasing a moving path again — loosen `commit_meet_ds_m` / `commit_obs_dd_m` |
| **Min lateral clearance** = min over the pass of \|d_ego − d_opp\| | `/car_state/odom_frenet` + `/opp_racecar/odom` (ground truth) | **≥ 0.40 m**, and **must not drop** relative to the D0/baseline run |
| Collisions | `/pitwall/events` (`gym_bridge` emits `ego: OPPONENT collision` / `ego: WALL collision`) | **0** |
| OVERTAKE↔TRAILING oscillation | `/state_machine` timestamps | ≤1 round trip per pass; **no transition pair closer than 0.8 s** |
| RECOVERY entry | `/state_machine` | **0**. A dropped OVERTAKE mid-pass goes to RECOVERY, not TRAILING (`recovery_entry_d_m = 0.4`), and RECOVERY is self-sustaining |
| Lap time | `/lap_data` | pass lap ≤ baseline + 1.5 s; the lap **after** back to baseline ± 0.3 s (proves CLOSE merged and RECOVERY did not latch) |
| SM health | `ros2 topic hz` | `/state_machine` ≥ 70 Hz throughout |
| Attribution | `/rosout` | for every second spent TRAILING with an opponent < 10 m, exactly one `dyn_OT blocked: <term>` line exists |

**Headline number: commit-to-pass conversion rate** (completed passes ÷ TRAILING→OVERTAKE
commits). That is what the Phase-1 changes are optimizing.

**Advance rule:** conversion rate strictly up, min clearance not down, transitions/lap not up.
If a step is neutral, **revert it** — a carried neutral change is future attribution debt.

---

## §6 — A/B order (do NOT combine)

The fixes landed as separate commits so each can be tested alone. Ordering hazards:

1. **`latest_threshold` alone.** It gates the freshness check *and* cache adoption. Paired with
   `on_spline_min_dist_thres_m` you cannot tell whether the car committed because it saw fresh
   data or because it accepted a *farther* path — and the second carries the real safety cost.
2. **`lateral_width_m` (SM) and `sep_monitor_slack_m` (planner) are opposite ends of one
   inequality.** Move them in the same run and the observed slack is the difference of two
   edits, with a sign that is easy to get backwards.
3. **Never flip `use_prediction` with any margin change.** It shifts the effective clearance by
   up to ~0.5 m — an order of magnitude more than any margin edit. Test it **last**, alone.
4. **Freeze the solver geometry during Phase 1:** `pass_overlap_m`, `pass_hold_max_m`,
   `lane_max_slope*`, `meet_s_slew_m`, `opp_pred_slack_m`, `engage_gap_m`.

Toggles that restore prior behaviour: `hold_clear_check: false` (HOLD clearance re-check),
`free_check_dynamic_ot_slow: false` (slow-opponent routing), `sep_monitor_slack_m: -0.014`
(the historical 0.336 monitor line), `use_prediction`, `engage_min_closing_mps: -3.0`,
`lane_commit: false` (per-cycle re-solve — the wobble), `trust_grid_bounds: false`
(waypoint corridor bounds instead of the measured one).

---

## §7 — Sim pass ≠ real-car pass

1. **Sim forces `det_src=scan`** (`race.launch.xml`), so the real car's default `kiss` (Livox
   BEV) detection path is **never exercised**. `--inject merge` bypasses detection entirely.
2. **Executor lag reproduces in kind but not in magnitude.** Same single-threaded executor and
   same inline velocity replan, but sim adds gym raycasting and RViz on the same host, and it
   is RMW-dependent (`unicorn.sh` records FastDDS busy-spinning a core → ~22 Hz vs CycloneDDS
   at the full 80 Hz). Since `latest_threshold` is lag-dependent, **treat it as a per-platform
   tuned value** and record `hz /state_machine` on both.
3. **`tracking_merger` is in the loop in sim and absent on the real car**: 25 Hz republish, a
   `max_obstacle_d = 1.0` filter that drops *real* obstacles too, and a restamped header.
4. **`merge`-mode object data is unrealistically clean**: fixed `id`, `vd` always 0 — which
   defeats the `opp_pred_slack_m` clamp and the whole tracker classification chain.
5. **Obstacle size differs by seam**: `detect` clamps up to `min_size_m = 0.30`, `merge` passes
   the raw 0.20 m, the real kiss detector reports its own box. Every margin in the chain moves
   with it.

Require an **overlay-mode** pass (D7-a) before touching the car, and re-measure the message age
against the car's own `static_OT check … age=` there.
