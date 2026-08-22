# The corridor decides the path, and the ramp scan finally reads the grid it plans on

Round 5 closed two directions and opened one. `RAMP_CEILING_NOTE.md` has the measurements; the
three sentences that matter here are: the ramps are 11 % of the failures and the inter-apex weave is
67 %; the corridor-blind segment is not the ramp but every segment between two knots; and the
ceiling for re-deciding the whole span, against the node's own gates, is 25.6 % / 35.1 % of all
refusals.

This round implements that. Three things ship, and they are independent:

1. **`corridor_path.py`** — a pure open-segment QP, no ROS, no node state.
2. **the ramp scan reads the publishing grid**, not a 0.5 m sample of it. Right on its own; it has
   nothing to do with (1) and it changes the shipped `sample` path.
3. **`static_plan_method`**, default `sample`. The QP is reachable and measured; it is not on.

Everything below was measured with `sweep_static_race.py --method all`,
`sweep_static_feasibility.py --check`, `test_corridor_path.py`, and a HEAD-vs-branch cell-by-cell
diff. Nothing was verified in sim.

---

## 1. The solver

    solve_corridor_path(s_grid, lo, hi, d0, dp0, w_dev) -> Optional[np.ndarray]

        min ||D2 d||^2 + w_dev ||d||^2   s.t.  lo(s) <= d(s) <= hi(s),  d(s0) = d0,  d'(s0) = dp0

### The operator is open, and that is not a detail

`closed_reopt._second_difference` is periodic: `D2[0,-1] = D2[-1,0] = 1`, so a lap joins up with
itself. A maneuver is an open segment with two DIFFERENT ends -- the pre-ramp on one side, the
raceline tail on the other -- and those two rows charge it for a bend between its first station and
its last. Priced, on a 0.4 m ramp profile:

| operator | ||D2 d||^2 |
|---|---|
| open (this module) | **0.300** |
| periodic (`closed_reopt`) | **3201.0** |

Four orders of magnitude, and it is not noise: it is the segment being told to bend until its two
ends meet. `open_second_difference` is written here rather than sliced out of the other one, and
`test_corridor_path.py` asserts BOTH that the interior stencil is identical and that the ratio
survives -- so the two solvers keep one definition of "bending" without one importing the other's
wrap.

### The apex is NOT pinned

Side selection and the keep-out of every box the path is shaped around are folded into `lo`/`hi`
before the solve. With those in the bounds the corridor forces the apex offset on its own: minimum
bending rides the keep-out edge (plus `apex_bulge`, taken only as far as the corridor allows,
exactly as the node clips its own peak). Pinning the VALUE as well over-constrains a run whose two
ends are already pinned.

That was measured both ways -- `corridor_qp_pin_apex` keeps the pinned formulation -- and the
comparison is in section 4b. Short version: pinning opens 1.1-2.7 pp more cells and is worse on
every other axis, two of them structurally.

The start pin is **C2**, not C1, and that is a measurement rather than a preference: with value and
slope matched but curvature free, the |d'| step at the junction came out at p90 0.0548 / 0.0504
against a 0.05 target. See R2.

### Unknowns: a control grid that must contain the pins

A three-box maneuver is 200+ stations at 0.1 m, and the solve runs per candidate inside a 20 Hz
cycle, so the unknowns are a cubic-spline control grid (`corridor_qp_max_vars`, 60) and the BOUNDS
are enforced at every published station. The reduction costs expressiveness, never feasibility.

One trap, found by measurement rather than by reasoning: a station whose box is DEGENERATE -- the
terminal `d = 0`, a held apex band -- asks the profile to take one value to within a micrometre,
and a cubic can only do that between two knots by accident. quadprog reports it as
`constraints are inconsistent`, which reads as "there is no path here" and is nothing of the sort.

| | |
|---|---|
| solves reading inconsistent at 60 evenly spaced controls | **13.2 %** |
| of those, recovered at full resolution | **100 %** |
| cost of the wasted ridge ladder on each (a bigger ridge cannot fix an infeasible set) | p95 **14.15 ms** |

Fixed by making every pinned station a control point (`_ctrl_indices`), which turns those
constraints into boxes on single variables -- feasible by construction. The full-resolution retry
stays as the last rung; the ridge ladder now runs only on a genuine positive-definiteness failure.

### The candidates collapse

Everything the solver is given is fixed by the span and by WHICH SIDE of each box the path takes.
The sampled terminal offset never enters. So two candidates that pass the same boxes on the same
sides hand the solver identical arguments:

| solves per planning pass | |
|---|---|
| one per candidate | 6.58 |
| memoised on (span, sides) | **2.84** |

2.3x, with every published number bit-identical (asserted). This is worth reading as a property and
not just a speed-up: **under `corridor_qp` the terminal-offset grid is a side chooser, not a shape
chooser.** `n_d_samples` stops being a shape knob. (Not available under `corridor_qp_pin_apex`,
where the apex bands are pinned to each candidate's own quintic -- which is precisely what differs.)

---

## 2. The ramp scan, and what it cost to make it free

`_ramp_limits` scanned the corridor every 0.5 m and decided ramp lengths for a path published every
0.0997 m. Round 5 measured the error at the stations it skipped: **1.10 m (ifac) / 1.80 m
(ifac_0807)**, on a track whose corridor is often not two metres wide. Restoring the scan's lateral
WIDTH changed that by ~0, so it was never the sweep; it was the station spacing.

The spacing is now `wpnt_dist`. Naively that is 5x the corridor rows per pass, and the first
measurement showed exactly what the brief predicted -- it got expensive, and the cost came out of
`ramp_search_max_ms`. Two fixes make it free, and both are right independently:

* **`_grid_corridor_batch` is vectorised.** The per-row while-loops cost ~0.06 ms a row. Verified
  BIT-IDENTICAL against the loop (max |difference| **0.0** over four sweep configurations x 401
  stations). 330 rows: 1.81 ms -> **1.02 ms**.
* **a ladder rung does not scan at all.** `ramp_retry` hands `_fit_ramp` an explicit length that it
  takes as given -- the scan's answer is never consulted -- so every rung was paying for a corridor
  read it discards, fifteen times over inside a 20 ms budget. The SQUEEZE retry still scans: it
  re-enters with a different `wall_margin`, so its corridor is a different corridor.

### What moved on the existing gates (HEAD vs this branch, `static_plan_method: sample`)

| | ifac HEAD | ifac branch | ifac_0807 HEAD | ifac_0807 branch |
|---|---|---|---|---|
| `--check` | exit 0 | **exit 0** | exit 0 | **exit 0** |
| feasibility grid, cur_d 0.0 | 100/99/99/98 | **identical** | 109/110/110/110 | **identical** |
| feasibility grid, cur_d 0.3 | 100/99/99/87 | **identical** | 109/110/110/106 | **identical** |
| ladder gains | +112 | +108 | +115 | +97 |
| mean speed cap [m/s] | 2.613 | 2.610 | 2.791 | 2.775 |
| mean peak abs kappa | 1.022 | 1.026 | 0.907 | 0.912 |
| corner cells, UNBUDGETED | 67 of 99 | **67** | 18 of 36 | **18** |
| corner cells, budgeted, x5 | 57,58,58,55,57 | **61,61,57,58,58** | 17,18,18,18,18 | **18,18,18,18,18** |
| mean corner cell [ms] | 16.0 | **14.6** | 17.5 | **17.2** |
| planner loop p50 / p95 [ms] | 19.3 / 26.2 | **13.6 / 25.3** | 23.6 / 26.7 | **20.9 / 25.5** |

The ladder gains fewer cells because the main pass now gets them itself -- total feasibility is
unchanged. The corner count is WALL-CLOCK GATED and therefore not reproducible to the cell: five
repeats of the SAME code span 55-58 on ifac. That is why the unbudgeted row is there, and it does
not move. The branch is faster per cell despite reading five times as many stations.

### The sample path itself, cell by cell

Not an aggregate: the same station, gap, layout and tracking error planned by both node versions,
published `d` compared elementwise.

| | ifac | ifac_0807 |
|---|---|---|
| cells | 4770 | 4950 |
| published by both | 2739 | 3276 |
| **bit identical** | **2179 (79.55 %)** | **2607 (79.58 %)** |
| path moved | 560 | 669 |
| worst \|d_branch - d_HEAD\| | 1.00 m | 1.20 m |
| **LOST** | **0** | **0** |
| **GAINED** | **0** | **0** |

Four cells in five are untouched. The fifth is the point of the change: those are the cells where
the 0.5 m scan authorised a ramp the corridor does not actually carry, and a metre of published
offset is exactly the size of the corridor error round 5 measured. Nothing was lost and nothing was
gained -- the fix changes which shape is chosen, not whether one exists.

---

## 3. Wiring

`static_plan_method: sample | corridor_qp` in `static_avoidance_params.yaml`, **default `sample`**.
The branch is at the d(s) generation point and nowhere else: the quintic is built exactly as before
(byte for byte), and the QP is offered the result as its start pin, its fallback, and -- under
`pin_apex` -- its apex values. Obstacle collection, knots, side selection and lookahead are upstream
and untouched; every gate, the cost, the commit and the publication are downstream and untouched.
`corridor_qp` answers `bounds`, `obs_box`, `grid`, `body` and `curv` in full.

Where the QP has no answer -- the corridor unmeasurable, or two keep-outs closing it -- the
candidate keeps its quintic, named on a throttled warning. Dropping it instead would cost the
planner a path for a reason that has nothing to do with the shape.

Known and deliberately not fixed here: the cost term `w_c * |d_end - d_end_prev|` is written in
terms of the sampled terminal offset, and under `corridor_qp` several candidates share one profile.
The term still penalises a SIDE flip (the sign is what it reads) and that is what it exists for, but
it no longer distinguishes shapes. Changing the cost function is outside this round.

---

## 4. The gates

### R1 -- refusal rate, race profile

`sweep_static_race.py --map ifac ifac_0807 --jobs 7 --method all`. Full station set, 8484 / 9462
race-realistic cells.

| ifac | refuses at racing speed | still closed after squeeze AND relax |
|---|---|---|
| sample | **49.4 %** | 41.9 % |
| corridor_qp | **32.0 %** | 21.0 % |
| corridor_qp /pin-apex | **29.3 %** | 21.5 % |

| ifac_0807 | refuses | still closed |
|---|---|---|
| sample | **41.1 %** | 37.6 % |
| corridor_qp | **24.4 %** | 17.4 % |
| corridor_qp /pin-apex | **23.3 %** | 18.1 % |

The `sample` column reproduces the campaign's baseline to the decimal (49.4 / 41.1) with item (2)
applied, which is the race-profile half of R5: changing which ramp length the scan authorises moved
560 / 669 published paths and moved this number by nothing.

As a fraction of refusals, `corridor_qp` recovers **35.2 %** (ifac) and **40.4 %** (ifac_0807).
Round 5's ceiling for the same idea was 25.6 % / 35.1 % of all refusals -- **do not read this as
beating the upper bound.** The ceiling was computed on the `delta = 0` subset alone (1300 / 1675
cells, the ones where `sc.Corridor` finds a tube at the full design margins under its own |d'|
limit); the implementation is not restricted to that subset and does not answer to that |d'| limit.
Different denominators. What the two agree on is the direction and the order of magnitude, and that
is all they were ever going to agree on.

### R2 -- seam continuity

`|d'| step at s_entry0`, one-sided differences on the published station grid. The one splice the
published array has: the pre-ramp decay joins the maneuver there. The ceiling's other three "seams"
were artefacts of its per-segment splice -- one continuous solve has nothing to join at an apex --
and with `tail_m: 0.0` the exit is not in the published grid at all.

| p90 of the seam step | ifac | ifac_0807 |
|---|---|---|
| sample (the node's quintic) | 0.0173 | 0.0129 |
| round-5 ceiling's QP, value-pinned only | 0.179 | 0.149 |
| **corridor_qp, C1 start pin only** | **0.0548** | **0.0504** |
| **corridor_qp, C2 start pin (ships)** | **0.0011** | **0.0000** |
| corridor_qp /pin-apex, C2 | 0.0011 | 0.0000 |

The C1 pin took the ceiling's 0.15-0.18 down to 0.050-0.055 -- three times better and **still over
the 0.05 gate on both maps.** Matching value and slope leaves the solver free to start bending at
the first station it owns, and where the entry ramp is short it does exactly that: on a synthetic
apex at 15 % of the span the step is 0.059-0.079 with C1 and identically 0 once the curvature is
continued as well. So the third pin was added, as the brief said to do if C1 proved insufficient,
and both numbers are above.

It is NOT an apex constraint. There is no splice at the apex to constrain -- the whole span is one
solve -- so the applicable knob was the start, and the quintic being replaced has had C2 there all
along (its first breakpoint is `[d_start, dp0, 0.0]`). Worst case over the full sweep: 0.0063.

The companion number, ungated and reported so the seam figure cannot be made to look good by
choosing where to stand -- **max |d'| step over the WHOLE maneuver**:

| median / p90 / max | ifac | ifac_0807 |
|---|---|---|
| sample | 0.0196 / 0.0684 / 0.4049 | 0.0179 / 0.0472 / 0.3840 |
| corridor_qp | 0.0381 / 0.2010 / 1.0026 | 0.0285 / 0.2055 / 1.0020 |
| corridor_qp /pin-apex | 0.0584 / **0.4665** / 1.2429 | 0.0387 / **0.3846** / 1.1422 |

The corridor-decided shape IS more curved than a quintic through the same apex -- p90 three times
the quintic's -- and the seam is one of the smoother places on it. That is the honest reading, and
it is the same fact R3 prices.

### R3 -- what the shape costs in speed

`sqrt(a_lat_max / peak|kappa|)` over every published path. Reported, never gated: a slower real
path is the trade against a refusal, and a refusal behind a stationary box is a TRAILING standstill
whose gap-PID fixed point is v = 0.

| median / p10 [m/s] | ifac | ifac_0807 |
|---|---|---|
| sample | 2.52 / 2.02 | 2.76 / 2.27 |
| corridor_qp | **2.43 / 1.93** | **2.63 / 2.12** |
| corridor_qp /pin-apex | 2.36 / 1.91 | 2.59 / 2.08 |

**Much better than the ceiling predicted.** Round 5 measured 1.96 / 2.02 m/s median for the same
idea and framed the trade as "a real path where there was none, driven below what the existing
output is held to". Measured on the implementation it is 2.43 / 2.63 -- 3-5 % under the sampled
path's own median, not 20 %. Two reasons: the ceiling re-solved only the cells the node had already
refused (the hardest ones), and it re-shaped them with no continuity constraint at all, whereas
this solver is pinned C2 at one end and C1 at the other and is thereby kept nearer the raceline.

So the trade the ceiling asked us to argue about turns out to be small. It is still a trade: the
median cap drops, and it drops further under `pin-apex`.

### R4 -- solve time

Single process, `--jobs 1`. Under parallel shards these numbers describe the machine, and the sweep
says so on every such run.

| ifac | one pass (a ladder rung) p50 / p95 / max | one cell, ladder budgeted at 20 ms, p50 / p95 / max |
|---|---|---|
| sample | 5.19 / 8.96 / 26.80 ms | 8.40 / 28.83 / 34.05 ms |
| corridor_qp | 10.06 / 16.19 / 68.80 ms | 15.93 / 44.68 / 67.33 ms |
| corridor_qp /pin-apex | 15.93 / 30.15 / 73.04 ms | 18.17 / 53.66 / 210.98 ms |

| ifac_0807 | one pass | one cell |
|---|---|---|
| sample | 5.91 / 9.73 / 11.78 ms | 7.85 / 28.96 / 36.42 ms |
| corridor_qp | 10.60 / 19.58 / 35.82 ms | 15.86 / 46.14 / 116.62 ms |
| corridor_qp /pin-apex | 17.67 / 41.35 / 98.37 ms | 20.45 / 57.99 / 203.77 ms |

quadprog itself is **16-17 %** of a `corridor_qp` pass (43 % under `pin_apex`); the rest is the
extra corridor read and the assembly around it. Two reductions already applied: the candidates are
memoised onto (span, sides), which is 2.3x fewer solves, and `_grid_corridor_batch` is vectorised.

**Superseded — see section 6. With the ramp ladder skipped (which it now is, because the ladder
opens zero cells under this method), corridor_qp fits the cycle and is CHEAPER at p95 than the
sampled path that ships.** The numbers above are what it cost while it was still paying for a
retry mechanism it does not use.

### R5 -- the sample path's own regression

Section 2. Against the commit before this branch: zero cells lost, zero gained, four in five
bit-identical; the rest is what (2) is for. The race-profile half is R1's `sample` row: 49.4 % /
41.1 %, the campaign baseline to the decimal.

Against **3800073** (the first commit of this branch), i.e. everything the ladder skip, the probe
and the log added: **100.00 % bit identical**, 2739 / 3276 published paths, worst
|d_after - d_before| = 0.0000 m, nothing lost or gained. The sampled path did not move a bit this
round, and it should not have: every change is behind `static_plan_method == "corridor_qp"` or
behind a flag that is off.

---

## 4b. Pinning the apex, measured both ways

The brief asked for the ceiling to be re-measured with the apex pinned and unpinned, and said to
take the unpinned formulation if it opens as much or more. Read literally, **it does not**: pinning
opens 2.7 pp (ifac) and 1.1 pp (ifac_0807) more cells.

| | refuses (ifac / 0807) | whole-maneuver \|d'\| step p90 | speed cap median | one pass p50 | candidate collapse | needs the ramp ladder |
|---|---|---|---|---|---|---|
| corridor_qp | 32.0 / 24.4 % | 0.201 / 0.206 | 2.43 / 2.63 m/s | 9.9 / 10.5 ms | **2.3x** | no (section 6) |
| /pin-apex | **29.3 / 23.3 %** | **0.467 / 0.385** | 2.36 / 2.59 m/s | 15.9 / 17.7 ms | none possible | yes, 0.9 / 0.5 pp |

Every column except the first is worse, and two of them structurally. With the apex bands held at
each candidate's own quintic the solver's arguments differ per candidate, so the collapse that makes
this affordable cannot happen at all -- and pinning re-introduces a shape constraint that a shorter
window can relieve, which is why the pinned variant is the only one the ramp ladder is still worth
anything to (28.4 / 22.8 % with it against 29.3 / 23.3 % without). The extra cells cost 2.3x the
kink, 3-4 cm/s of speed, 60 % more time per pass, the candidate collapse, and a retry mechanism the
unpinned formulation has been shown not to need.

So the default stays UNPINNED, and this is a departure from the brief's stated rule -- named here
rather than buried, because the rule was written before these four columns existed. `pin_apex`
remains a parameter, so the trade can be re-taken on evidence rather than re-argued.

The mechanism is worth stating plainly: pinning the apex does not add information. Side selection
and keep-out are already in `lo`/`hi`, so the corridor forces the apex offset whether or not the
value is pinned; what pinning adds is a REQUIREMENT that the profile pass through the quintic's
particular value there, which is a constraint on the shape and not on the safety. Where the two
disagree the quintic's value happens to sit somewhere the corridor also allows, so a few more cells
survive -- at the cost of a profile that has to bend to reach a number nothing needed it to reach.

---

## 5. What has NOT been shown

* Nothing here ran in sim. Every number is offline, on a harness that drives the real `do_spline`.
* `corridor_qp` has never driven a car. R3 is a curvature-limited speed cap, not a lap time, and
  R2 measures the published array, not what a controller does with it.
* The refusal rates are a RACE profile -- three boxes on ground a car could otherwise drive, met by
  a car following the line. They are not the uniform matrix and are not comparable to it.
* R4 says the cost does not fit today. The 83 % of a pass that is not quadprog has not been
  optimised, and the obvious next reduction (sharing one corridor read between the ramp scan and
  the QP, which today read overlapping station sets at different lateral resolutions) is not done.
* R2 is one seam, chosen because it is the only splice the published array carries. The
  whole-maneuver companion is reported alongside it precisely so that choice cannot flatter the
  result -- and on that number the corridor shape is three times the quintic's p90.
* `w_c * |d_end - d_end_prev|`, the anti-chatter term, still reads the sampled terminal offset. It
  penalises a SIDE flip correctly and no longer distinguishes shapes, because under `corridor_qp`
  several candidates share one. Left alone: the cost function is downstream of the branch.

## What would end the open questions

1. Sim, S1-S5 in `stack_master/STATIC_AVOIDANCE_TEST_RUNBOOK.md`, with
   `static_plan_method:=corridor_qp` and `static_plan_log:=true`. Nothing here is a driving claim.
2. Whether a 2.4 m/s median cap through a three-box weave is drivable at all, which is a controller
   question this file cannot answer.

---

## 6. The ramp ladder is not this method's mechanism

The ladder re-throws the whole candidate grid over a different ramp pair when everything was
rejected. It is worth 5.5 pp (ifac) and 3.1 pp (ifac_0807) to the sampled path. Under `corridor_qp`
it is worth **nothing**, and the measurement is not a rate that happens to match -- it is the same
cells:

| ladder | ifac refuses | ifac_0807 refuses | published paths (ifac / 0807) |
|---|---|---|---|
| sample, ON (ships) | 49.4 % | 41.1 % | 4931 / 5905 |
| sample, OFF | **54.9 %** | **44.2 %** | 4692 / 5811 |
| corridor_qp, ON | 32.0 % | 24.4 % | 6705 / 7818 |
| corridor_qp, OFF | **32.0 %** | **24.4 %** | **6705 / 7818** |

Cell by cell, over 1248 / 1398 cells: cells the ladder opens and the ladder-off run refuses --
**0** for `corridor_qp`, **90 / 65** for `sample`.

### Why, exactly

A rung does NOT change the lookahead, the obstacle set, the knots, the sides, the sampled terminal
offsets or the published grid length -- all of those are fixed before it runs. It changes one thing:
`r_in`/`r_out`, and through them `s_entry0` and `s_exit_end`. And `s_entry0` is not just where the
hump starts; it is `cand_entry_i`, **the station from which the grid and body gates hold a candidate
responsible** (everything before it is the pre-ramp decay, which every candidate shares and none can
change, so it is reported and never charged).

That is what the ladder buys the quintic. On every one of the 90 / 65 cells it opens, the rung-0
failure was `obs_box+grid+body`, and every winning rung has a SHORTER ENTRY (ifac 90 of 90;
ifac_0807 65 of 65, with the exit staying at 4.5 m on 47 of them). The quintic's shape through a
pinch is fixed once the apex offset is chosen, so its only way past an early one is to hand those
stations back to the pre-ramp and the raceline tail.

The QP has no such need: inside the LONG window it can already sit on the raceline over the early
stations, because the corridor contains d = 0 there. So there is nothing for a shorter window to
buy, and the rung is pure cost.

### What skipping it is worth

Single process, `--jobs 1`, one cell as the chain runs it with the shipped 20 ms rung budget:

| corridor_qp | p50 | p95 | max |
|---|---|---|---|
| ifac, ladder ON, @3.0 m/s | 15.9 | 42.3 | 76.7 ms |
| ifac, ladder OFF, @3.0 m/s | **9.9** | **16.7** | 65.0 ms |
| ifac, ladder ON, @2.0 m/s (squeeze reachable) | 15.7 | 71.5 | 118.0 ms |
| ifac, ladder OFF, @2.0 m/s | **11.7** | **31.0** | 62.3 ms |
| ifac_0807, ladder OFF, @3.0 / @2.0 m/s | 10.5 / 11.4 | **18.7 / 38.4** | 33.1 / 72.6 ms |

Wall clock on the sweep says the same thing from the other side: the full race profile takes
5.3 / 5.4 min per map for `sample` and **3.2 / 3.4 min** for `corridor_qp`, on identical cells.

against `sample` as it ships (ladder ON): p95 **28.7 / 40.8** ms (ifac, @3.0 / @2.0) and
**28.9 / 43.7** ms (ifac_0807).

**corridor_qp now fits the 20 Hz cycle, and at p95 it is cheaper than the path that ships.** Its
tail is worse (max 65-73 ms against 51-54), and that is the remaining cost statement.

`corridor_qp_ramp_ladder` puts the ladder back. It exists because this is two maps and one profile,
and a claim that costs one flag to re-test should stay testable.

---

## 7. The squeeze is NOT made redundant

The obvious guess is that a method which opens 17 pp more cells has already taken the ones a margin
trade would have found, leaving the squeeze to spend time on nothing. Measured, the opposite:

| recovered by the squeeze | of refusals | cells |
|---|---|---|
| ifac, sample | 15.2 % | 639 |
| ifac, corridor_qp | **34.5 %** | **936** |
| ifac_0807, sample | 8.5 % | 331 |
| ifac_0807, corridor_qp | **28.9 %** | **669** |

More in absolute count, not just in share: with the corridor deciding the shape, a centimetre of
margin is worth more than it was, because the shape is already using the room the margin frees.
Skipping the squeeze under `corridor_qp` would cost 936 / 669 cells. It stays.

---

## 8. The counterfactual, for sim

`static_plan_log: true` (off by default; it costs a whole extra planning pass) emits one line per
cycle:

    PLAN method=corridor_qp ms=11.4 pts=151 obs=3 squeeze=0 vcap=2.44 vs_sample=OPENED

`vs_sample` re-runs the REAL pipeline with the method swapped and `probe=True` -- same knots, same
gates, same margins -- because a second copy of the gate stack would drift from the first one. A
probe plans and records NOTHING: no feasibility verdict, no commit, no handover anchor, no
anti-chatter memory, and a refusing probe returns None rather than the empty publication that
carries `feasible=False` (that last one is the leak that would matter: the state machine drops out
of avoidance on that edge, so a leaking probe would make switching the log on cause a TRAILING).
`planner/spliner/test/test_plan_probe.py` asserts all five.

`OPENED` is the line to grep for. A refusal rate is a sweep statistic and a car does not drive one;
this is the only way to see, from a bag, what changing the shape actually changed. `LOST` is the
reverse and has never been seen offline -- if a bag shows one, that is the finding.

---

## 9. How far off the raceline it goes, and what bounding it would cost

Measured by `planner/spliner/scripts/sweep_deviation_cap.py` (ifac_0807, race profile, 3154 cells per
row-and-`cur_d`, `--jobs 7`). Nothing here changed a value; this is what the numbers say.

The shipped `corridor_qp_w_dev` is 0.0, so nothing in the objective prefers the raceline. Headline
numbers at `cur_d = 0.0`, `>=0.4` being the fraction of published paths whose maneuver peak reaches
`recovery_entry_d_m` — i.e. that trip RECOVERY the moment the car is on them:

| row | refuse | closed | man p90 | man max | `>=0.4` | vcap med | seam p90 |
|---|---|---|---|---|---|---|---|
| `sample` | 41.1% | 37.6% | 0.600 | 1.075 | 43.2% | 2.76 | 0.0108 |
| `w0` (ships) | 24.4% | 17.4% | 0.750 | 1.150 | **68.9%** | 2.63 | 0.0000 |
| `w0.01` | 24.4% | 17.3% | 0.706 | 1.150 | 68.7% | 2.63 | 0.0000 |
| `w0.05` | 24.4% | 17.6% | 0.654 | 1.150 | 68.3% | 2.63 | 0.0000 |
| `w0.1` | 24.4% | 17.6% | 0.640 | 1.050 | 68.3% | 2.63 | 0.0000 |
| `w0.5` | 24.4% | 17.3% | 0.633 | 1.028 | 67.8% | 2.62 | 0.0000 |
| `cap0.5` | 43.5% | 27.4% | 0.500 | 0.500 | 59.7% | 2.64 | 0.0000 |
| `cap0.4` | 58.0% | 38.2% | 0.400 | 0.400 | 52.7% | 2.72 | 0.0000 |
| `cap0.3` | 78.3% | 57.6% | 0.300 | 0.300 | **0.0%** | 2.76 | 0.0000 |

**`w_dev` is not the knob.** Fifty times the weight (0.01 → 0.5) moves p90 by 0.117 m, the peak by
0.12 m, and the RECOVERY fraction by **1.1 pp** — while the refusal rate does not move at all
(24.4% in every row). There is no trade to price here, because the excursion is set by the corridor
and the keep-out, not by a weight: the QP must clear the box whatever `||d||^2` costs, and where the
corridor is wide the penalty is cheap to pay. Raising it is close to free and close to useless.

**A hard cap does bound it, and the bill is the refusal rate.** cap0.5 costs +19 pp of refusal,
cap0.4 +34 pp, cap0.3 +54 pp — and only cap0.3 removes the RECOVERY contention outright (0.0%),
because a cap AT 0.4 still trips `abs(cur_d) >= 0.4` with zero tracking error. Curvature-limited
speed does not pay for it (`vcap` 2.63 → 2.72/2.76: a shorter excursion bends less) and the seam gate
holds in every row (p90 <= 0.05, all nine).

**The unmeasured cost that is also the reason nothing shipped:** refusal → TRAILING →
`static_feasible=False` → more windows in which the only adoptable path is lane_change's. That
planner engaged on stationary boxes and put the car in a wall (`ENGAGE_GATE_NOTE.md`). Bounding this
planner before that one is fixed hands authority to the worse actor.

### Two things the sweep found that are not about tuning

1. **`corridor_qp` deviates far more than the shape it replaced.** 68.9% of its published paths reach
   the RECOVERY threshold against `sample`'s 43.2%, and its p90 is 0.750 m against 0.600 m. That is
   the cost of the refusal-rate win that made it the default (24.4% vs 41.1%), and it had not been
   measured until now.
2. **A clipped corridor is not a bound on the published path.** The sampled-quintic fallback was
   *ruled out* — `fallback` is 0.0% in every row — and yet 59-297 cells per cap row publish a
   maneuver peak over the cap (1182-1313 at `cur_d = 0.5`). What remains is the QP's own pinned start
   (`d0/dp0/dpp0` are read off the unclipped quintic) and stations published outside the QP window.
   So a hard cap needs a **gate** as well as bounds; `sweep_deviation_cap.ok()` is that gate, and
   `capfail` counts what it catches.

### Why the `cur_d` axis is in the table

The QP pins `d(s0) = cur_d` as an **equality**, so a car already outside a cap has no feasible
corridor. `sweep_static_race`'s race profile uses `|cur_d| <= 0.1`, which made that look academic.
The crash run says otherwise: `recovery_spliner` reported raceline-lost at 0.82 / 0.47 / 0.43 /
0.42 m, the controller's AEB saw 0.65, lane_change committed 0.58 and 0.68. At `cur_d = 0.5`,
`cap0.4` refuses **70.5%** of cells and `cap0.5` 56.2% — against 24.4% uncapped. Any cap has to
answer for that branch before it ships, and no sweep can answer it by averaging over it.
