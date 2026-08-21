# The ramps are not where it fails, and the ceiling is in the segment nothing scans

Round 5 asked three questions about the `delta = 0` cells -- the ones where a corridor exists at the
FULL design margins and the planner still publishes nothing. All three are answered here, and the
first answer is that the target the last three rounds converged on from three directions is **the
wrong target**.

Everything below is measured by `planner/spliner/scripts/sweep_static_ramp.py`. Nothing in
`static_avoidance_node.py`, its yaml, or any margin was modified. The QP is offline arithmetic on
the candidates the node itself built.

## Before any number is believed: the corridor, and the operator

| check | result |
|---|---|
| **A vs B** -- this file's per-station corridor against the node's own `_grid_corridor`, the call `do_spline` makes for its sampling limits | **0.00e+00 m** on both maps, over 15 sampled stations each |
| **the operator** | `closed_reopt._second_difference` wraps the lap (`D2[0,-1] = D2[-1,0] = 1`). A ramp is an OPEN run with pinned ends, so `_open_second_difference` is literally `[1:-1]` of it. On a 0.4 m ramp profile the closed operator scores 3201 against the open one's 0.3 -- that difference IS the two phantom wrap rows, and charging an open segment for them would bend it to join its own two ends. |
| **A vs C** -- the node's own RAMP SCAN (`_ramp_limits`: stations every 0.5 m, lateral step 0.10 m, sweep narrowed to `max\|d_ends\| + apex_bulge + 0.10`) | worst **1.10 m** (ifac), **1.80 m** (ifac_0807). Restoring the sweep width changes it by ~0, so it is the **0.5 m station spacing**: at path stations the scan never visits, the corridor is up to a metre from what its neighbours imply. Not an error -- it is the resolution the ramp length was decided at. |

## Fidelity of the restatement

The analysis rebuilds `do_spline`'s main pass (knot loop, terminal-offset grid, ramp scan, BPoly
assembly, and all five gates) so the candidates can be inspected -- the node keeps none of them.
It is checked against the node's own `(N sampled) reject bounds=/obs_box=/grid=/body=/curv=` line on
**every** `delta = 0` cell, not on a sample:

| | ifac | ifac_0807 |
|---|---|---|
| match | **1300** | **1675** |
| mismatch | **0** | **0** |
| plan not rebuildable | 0 | 0 |
| candidate wrongly feasible | 0 | 0 |

The gating also reproduces round 4 exactly (ifac 3605 still-closed / 36.1 % at `delta = 0`;
ifac_0807 3611 / 46.4 %), which is two independently written scripts agreeing on the population.

## (1) Where the violation is

Over the rejected candidates of the `delta = 0` cells, the station that made the NODE reject -- its
first failing gate's first offending station:

| segment | ifac (8499 cand.) | ifac_0807 (10959 cand.) |
|---|---|---|
| prefix | 0.0 % | 0.0 % |
| **entry ramp** | **7.2 %** | **7.9 %** |
| apex band | 22.0 % | 20.9 % |
| **weave** (between two apexes) | **66.9 %** | **68.4 %** |
| **return ramp** | **3.9 %** | **2.8 %** |

The two ramps together are **11.1 %** and **10.7 %**. Two thirds of the failures are in the
inter-apex weave -- precisely the segment `_ramp_limits` never looks at: it scans
`[knot0 - ramp_len, knot0]` and `[knot_last, knot_last + return_len]` and nothing between them.

By knot count, which is the cleanest form of the statement (a one-knot path has no weave at all):

| path carries | entry | apex | weave | return |
|---|---|---|---|---|
| ifac, 1 knot (162 cand.) | **48 %** | 30 % | -- | **22 %** |
| ifac, 2 knots (5511) | 7 % | 22 % | 66 % | 5 % |
| ifac, 3 knots (2826) | 6 % | 21 % | 73 % | 1 % |
| ifac_0807, 1 knot (393) | **37 %** | 37 % | -- | **25 %** |
| ifac_0807, 2 knots (7038) | 6 % | 21 % | 70 % | 3 % |
| ifac_0807, 3 knots (3528) | 8 % | 18 % | 74 % | 0 % |

On a ONE-BOX path the failure genuinely is the ramps -- 70 % (ifac) / 62 % (ifac_0807) of it. But
one-knot paths are **22 of 1300** and **55 of 1675** `delta = 0` cells, i.e. 1.7 % and 3.3 %. The
population is two- and three-knot paths, and there the ramps are under 12 %.

### The width hypothesis: not supported, and it does not matter either way

"The ramp passes through ground narrower than the apex" splits the two maps:

| corridor width at the worst corridor violation vs at its apex station | ifac | ifac_0807 |
|---|---|---|
| median width at the violation | 195 cm | 130 cm |
| median width at the apex | 140 cm | 110 cm |
| narrower at the violation | **24.3 %** | **63.0 %** |
| median (w_here - w_apex) | **+40 cm** | **-25 cm** |

On ifac the violating station is *wider* than the apex; on ifac_0807 it is narrower. So the
hypothesis is false on one map and true on the other, which makes it not the mechanism. What IS
map-invariant is the LOCATION: 89 % / 90 % of the worst corridor violations sit in the weave on
both maps.

### How far outside, and what the corridor asked for instead

The violation amount is exactly the gap between the `d` the corridor allows and the `d` the shape
produced (the reported figure is `d_here` clipped to `[lo, hi]` subtracted from `d_here`):

| | ifac | ifac_0807 |
|---|---|---|
| median | **30.0 cm** | **32.2 cm** |
| p90 | 56.0 cm | 71.2 cm |
| max | 130.7 cm | 127.9 cm |

Third of a metre at the median. No margin trade reaches that, which is consistent with `delta`
being zero in every one of these cells: the room is there, the shape does not use it.

Distance from the apex the violation belongs to: ifac median +1.36 m, ifac_0807 median -1.23 m,
p10/p90 inside ±2.8 m on both. 2.8 % / 3.5 % of rejected candidates have NO corridor violation
anywhere -- every one of those is an `obs_box` rejection, which is the keep-out rectangle and not a
corridor question.

## (2) The ceiling

The node's apex offsets and knot placement are kept EXACTLY as it chose them. Only `d(s)` is
re-solved, as `min ||D2 d||^2 s.t. lo(s) <= d(s) <= hi(s)` with the segment ends pinned to the
node's own values, and the result is put through every gate the node runs -- `bounds`, `obs_box`,
the sampling `grid`, the body floor at kernel 7, and the corner-fair curvature pair. Same principle
as the oracle: an answer to a different question cannot be compared to the node's.

### A. the ramps only, exactly as item (2) specifies

| | ifac | ifac_0807 |
|---|---|---|
| cells opened | **10 of 1300** | **15 of 1675** |
| % of `delta = 0` | **0.8 %** | **0.9 %** |
| % of ALL refused cells | **0.2 %** | **0.4 %** |
| by layout | 1box = 10 | 1box = 15 |

Every cell it opens is a one-box cell -- exactly what (1) predicts. A corridor-aware entry and
return ramp fixes the paths whose failure is in the entry and return ramps, and those are 2-3 % of
the population.

**A corridor-decided ramp, on its own, is worth two to four tenths of one percent.** That closes the
direction.

### B. the same question asked of the whole span

Identical formulation, identical checks, one difference: the QP also owns the inter-apex weave,
with the apex bands still pinned station-by-station to the node's own values. It exists because (1)
says that is where the violations are, and it still keeps what item (2) said to keep.

| | ifac | ifac_0807 |
|---|---|---|
| cells opened | **1074 of 1300** | **1364 of 1675** |
| % of `delta = 0` | **82.6 %** | **81.4 %** |
| % of ALL refused cells | **25.6 %** | **35.1 %** |
| by layout | 1box 15, 2box_6m 561, 3box_6m 498 | 1box 21, 2box_6m 671, 3box_6m 672 |

What still kills a QP candidate in B: `body` (1697 / 2871), `obs_box` (1609 / 1904), `grid`
(768 / 680). The body floor at kernel 7 becomes the leading refusal once the corridor decides the
shape -- worth knowing, because it is the one gate the squeeze may never touch.

## (3) What the curvature costs

**Nothing, on either scope, on either map.** `kappa_add_max` 5.0 and `kappa_abs_max` 5.5 take back
**0** cells. The curvature gate is not what stands in the way, and (2)'s pass rate is identical
before and after it.

What IS a cost is the speed the shape implies:

| | peak abs kappa median / p90 / max | speed cap sqrt(a_lat/kappa) median / p10 |
|---|---|---|
| ifac, A ramps only (n=10) | 4.585 / 4.609 / 4.636 | **1.14 / 1.14 m/s** |
| ifac, B whole span (n=1074) | 1.562 / 3.032 / 4.861 | **1.96 / 1.41 m/s** |
| ifac_0807, A (n=15) | 4.682 / 4.906 / 4.981 | **1.13 / 1.11 m/s** |
| ifac_0807, B (n=1364) | 1.477 / 2.876 / 4.093 | **2.02 / 1.44 m/s** |

`sweep_static_feasibility` gates the cells the node ALREADY publishes at a mean cap of 2.50 m/s
(it measures 2.61 on ifac, 2.79 on ifac_0807). So B opens its cells at a shape whose
curvature-limited speed sits **below** what the planner's existing output is held to. A real path
where there was none, driven slower than the rest. That is the trade to argue about; the curvature
gate is not.

### Continuity

The QP is pinned in VALUE only, so `d'` is unconstrained at the seams. Worst `d'` jump over the four
seams, on the cells that opened:

| | QP median / p90 / max | the node's own quintic, same candidate |
|---|---|---|
| ifac B | 0.007 / 0.179 / 0.509 | 0.002 / 0.031 |
| ifac_0807 B | 0.002 / 0.149 / 0.508 | 0.002 / 0.014 |
| ifac A | 0.336 / 0.361 / 0.385 | 0.012 / 0.063 |

The median is the node's own; a minority of opened paths carry a real kink at the junction (p90 is
5-6x the quintic's, and the ramps-only scope is 30x). Every one of them still passes the node's
curvature gate, because that gate is loose. An implementation would want slope equality at the
seams; the ceiling deliberately does not impose it, because a ceiling must not be understated.

## What this says about the next round

1. `_fit_ramp` / `_ramp_limits` is **not** the bounded change to make. Measured against the node's
   own gates, it is worth 0.2 % (ifac) / 0.4 % (ifac_0807) of refusals. The 23 % ramp-length share,
   the `obs_box+grid` signature and the 36 % `delta = 0` share all pointed at it, and all three
   pointed wrong -- they identified a symptom that the ramps happen to be a visible part of.
2. The corridor-blind segment is not the ramp. It is **every segment between two knots**. The scan
   covers the two outer ramps and nothing between them, and two thirds of the failures are in what
   it skips. Even inside the ramps it scans, it samples every 0.5 m against a corridor that moves by
   up to a metre between samples.
3. The upper bound on fixing that -- keeping the node's own apex offsets and knot placement, and
   answering every one of the node's own gates -- is **82.6 % / 81.4 % of the `delta = 0` cells**,
   i.e. **25.6 % / 35.1 % of all refusals**.
4. The price is **speed**, not curvature: median cap ~2.0 m/s against the 2.5 m/s the existing
   output is gated at. And the leading refusal that remains is the **body floor**, which no margin
   trade may touch.
5. Nothing here is implemented, and nothing here justifies implementing B as written: it is a
   ceiling, measured with a QP that ignores slope continuity and re-shapes the whole maneuver.
