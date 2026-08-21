# Two planners, one obstacle, opposite directions

## What happened (run_0810_2102, sim, ifac_0807, reopt:=true, 4 static boxes, NO opponent)

Reproduce the numbers below with:

```
~/miniforge3/envs/unicorn/bin/python3 stack_master/scripts/analyze_run_log.py ~/run_0810_2102.log
```

```
--- lane_change engages: 2 ---
        t   id    gap   side  offset  v_opp  v_pass   nearest static selection before it
  382.981    2    3.2   left    0.68   0.00     4.3   (none logged)
  391.819   18    4.4  right    0.58   0.20     4.1   390.464 left  d_end=+1.00 -> OPPOSITE, step 1.58 m

--- SM path adoptions: 3 ---
        t  planner adopted                  |steering| clipped within 2 s after
  385.431  static_avoidance_planner                                           0
  389.863  dynamic_avoidance_planner                                          0
  393.681  dynamic_avoidance_planner                                         16   <-- saturated
```

The chain, all of it measured, none of it inferred: the lane-change planner engaged on a
**stationary box** (`v_opp` 0.00 and 0.20 m/s, and the opponent vehicle was never spawned), chose
the **opposite side** to the static planner on the same box, the SM adopted the dynamic path
1.86 s later, the reference stepped **1.58 m**, and 0.91 s after that the steering saturated for
16 consecutive cycles (394.591 → 394.993) into the wall (396.092 iTTC, 398.201 respawn, 398.628
collision).

Two independent code facts let it engage, and the second is the load-bearing one:

1. `obs_cb` (`change_avoidance_node.py:503-505`) splits obstacles on `not o.is_static`. That flag is
   the tracker's position-persistence verdict, and it is `False` on every freshly created track
   (`staticFlag` stays `None` until `min_nb_meas`). So "not static" also means "just appeared", and a
   parked box arrives as a dynamic obstacle.
2. The only speed condition in the gate was `engage_min_closing_mps`, and it is a **closing** speed:
   `ego - o.vs` is at its **largest** when the obstacle is standing still. A parked box does not slip
   past that gate, it passes it maximally.

## The three candidates

`v_opp` column = does it refuse the two engagements above. `real OT` = does a genuine opponent still
get overtaken. Cost = what has to change.

| | A. min sustained target speed | B. SM hysteresis / commit on planner switch | C. same obstacle id → static wins |
|---|---|---|---|
| **blocks 382.981 (v_opp 0.00)** | yes — 0.00 < 0.35 | no — it still engages, the swap is only slower | yes, if static also claimed that box |
| **blocks 391.819 (v_opp 0.20)** | yes — 0.20 < 0.35 | no | yes |
| **removes the 1.58 m step** | yes, at the source: the dynamic path is never produced | only damps it. A dwell delays the adoption; the step itself is unbounded either way, so B has to add a **lateral-step bound**, not just a timer | yes for this pair |
| **real OT still works** | yes. 0.5 / 1.0 / 2.0 / 4.0 m/s all still engage (`test_engage_gate.py`). Cost is `engage_moving_s` = 0.3 s of dwell at engage, once | **risk**: any dwell on planner choice delays a genuinely needed dynamic overtake, and dynamic OT is the time-critical one | yes |
| **cost** | 2 params, one predicate, no message change | touches `_check_overtaking_mode` ordering, `_IMMEDIATE_STATES`, and the static-only re-entry cooldown — three coupled mechanisms, all untested today | `OTWpntArray` carries no target obstacle id → new `f110_msgs` field (rebuild + every consumer) or a new topic. Highest |
| **what it does NOT block** | a track whose *tracked* `vs` is wrong for ≥ 0.3 s (see P1); and the SM still has no lateral-step bound, so two REAL opponents, or any future disagreement, still steps the reference | lane_change still commits a lane around a parked box. This log has **7** `NO feasible candidate` events, so there are windows where the dynamic path is the only one on offer — B alone leaves the wrong geometry in play | the case where the two planners pick *different* obstacles and still disagree laterally; and it does nothing when static is infeasible |

**A is implemented.** It is the only one of the three that removes the wrong actor's authority at the
source rather than arbitrating after the fact, and it is the only one with no message change and no
new coupling. B is the right *second* step and is written up below. C is worth it only if an
`f110_msgs` change is already on the table for another reason.

## What A actually is

`_track_moving` / `_is_overtakable` mirror `static_avoidance_node._track_near_zero` from the other
side. That planner will not treat a dynamic-flagged obstacle as static until it has read below
`static_near_zero_mps` (0.15) for `static_promote_sec`; this one will not engage one until it has
read at or above `engage_min_vs_mps` (0.35) for `engage_moving_s` (0.3 s), continuously.

The mirror is the point, and it is asserted offline:

```
stack_master/scripts/check_avoidance_margins.py
  OK   engage_min_vs_mps 0.35 >= static_near_zero_mps 0.15
       -- no obstacle can be claimed by both planners (gap 0.15-0.35 m/s trails)
```

**Overlap is the crash; a gap is not.** If the engage floor dropped below the static band, an
obstacle inside the overlap would get a static keep-out one way and a lane change the other way at
the same time — which is this run. The band *between* them (0.15–0.35 m/s, sustained) is owned by
neither planner and resolves to TRAILING, which is the designed fallback and is what the stack
already does whenever the static planner refuses. Closing the gap instead of leaving it would
re-admit the 0.20 m/s reading that engaged on a stationary box, so the gap is deliberate.

Time, not speed, is what separates "parked" from "just seen" — both read ~0. That argument is
already written out in `_near_zero_static`'s docstring for the other direction; this is the same
argument, and the dwell is why a noise spike on a parked box cannot arm the gate.

## B, for whoever picks it up

Findings from mapping the SM, each one a precondition for doing B properly:

- The static-vs-dynamic choice is a single latched bool, `static_overtaking_mode`
  (`state_machine_node.py:225, 1532, 1579`), consumed at `get_splini_wpts:2003-2010`. There is **no**
  dwell, latch, streak or committed-side memory on it — and no test anywhere covers it.
- `ObstacleTransition` (`state_transitions.py:168`) evaluates the gates as
  `_check_overtaking_mode() or _check_static_overtaking_mode()` — dynamic **first**, short-circuit.
  A passing dynamic gate sets `static_overtaking_mode = False` in the same call, so the static
  planner's opinion is never evaluated.
- A static drop arms `static_ot_reentry_cooldown_sec` (0.3 s) against **static** re-entry only
  (`:1546-1555`). For that window the dynamic gate is the only one that can pass, which biases the
  SM toward exactly this swap.
- `min_dwell_sec` (0.2) cannot help: TRAILING and OVERTAKE are both in `_IMMEDIATE_STATES`
  (`:388`), so the OVERTAKE→TRAILING→OVERTAKE edge has zero damping. The yaml comment on
  `ot_free_lost_sec` already concedes this.
- The SM has **no lateral blend** on a producer swap. The static planner blends onto its *own* last
  published path (`_reanchor_commit`, `:2465-2518`), which does nothing when the SM changes
  producer. `_warn_splice_step` (`:1989-2001`) only checks the join *inside* one window, so a
  window-to-window jump of 1.58 m is logged by nothing.
- Separate latent bug found while mapping, not touched here: `get_farthest_target`
  (`:2367-2384`) can set `local_wpnts_src = OVERTAKE` by comparing `closest_gap` between the two
  avoidance caches while leaving `static_overtaking_mode` untouched — so it can select OVERTAKE
  *because the static cache had the farther blocker* and then publish the **dynamic** cache. The gap
  comparison and the geometry selection read different variables.
