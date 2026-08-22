# A minimum-violation fallback: measured, designed, and not recommended

Round-4 item (4) was conditional on the minimum-violation distance being small. It is small — and
that is the reason not to build it. This note records the design so the option is on file, and
the measurement that says the mechanism it would add already exists.

**Nothing in this note is implemented. No planner behaviour or parameter changed.**

## What was measured

For every race-realistic cell the planner refuses and neither retry recovers, `min_violation`
(planner/spliner/scripts/sweep_static_corridor.py) reports the smallest `delta`, subtracted from
`safety_margin_d` and `wall_margin` together, at which a continuous corridor exists. `half_car`
is never in the search: that is the car, not a margin.

| | ifac | ifac_0807 |
|---|---|---|
| still-closed cells | 3605 | 3611 |
| `delta = 0` — a corridor exists at the FULL design margins | 36.1 % | 46.4 % |
| `delta` in (0, 7 cm] — the squeeze already has authority here | 37.5 % | 37.4 % |
| `delta` in (7, 10 cm] — past the squeeze's wall floor | 22.1 % | 15.3 % |
| `delta` in (10, 15 cm] | 4.2 % | 0.9 % |
| needs more margin than exists | 0.0 % | 0.0 % |

Median over the cells that need any margin at all: **5.6 cm (ifac), 5.2 cm (ifac_0807)**.

## Why that kills the proposal rather than motivating it

The squeeze pass IS a minimum-violation path generator. `_squeeze_schedule` interpolates from
(safety 0.15, wall 0.15) to (0.05, 0.08) in `squeeze_steps` attempts, and the state machine can
force it at racing speed via `/planner/avoidance/relax` after `static_deadlock_timeout_s`. It can
therefore already spend `delta` up to 7 cm before its wall floor binds.

So the two numbers to put side by side are:

* **73.6 %** of still-closed cells (ifac) need `delta <= 7 cm`, i.e. lie inside the authority the
  squeeze already has — 36.1 % of them need *nothing at all*.
* The squeeze recovers **14.0 %** of refusals (ifac), 7.1 % (ifac_0807).

A mechanism that already has permission to spend the margin, and does not convert the cells the
margin would open, is not short of permission. What it is short of is a path shape that fits the
corridor the margin reveals — which is the same conclusion the corridor sweep reached from the
other direction: in the still-closed cells `obs_box+grid` is the dominant reject signature
(1979 of 3605 on ifac), meaning candidates are being rejected for leaving the drivable area, not
for clearance arithmetic.

Adding a second margin-trading path would therefore buy approximately what the first one buys,
at the cost of a second code path that publishes reduced-clearance geometry. That is the L6
regression's shape: spending safety to paper over a geometry problem.

## The design, for the record

Had the numbers gone the other way (most cells needing 15–20 cm, i.e. beyond every floor), the
shape would have been:

1. After the ladder and the squeeze both return `None`, run one further pass with
   `safety_margin_d` and `wall_margin` set to the measured `min_violation` for the current
   obstacle set, rounded up to the nearest centimetre, capped at the existing squeeze floors.
2. Publish with `ot_line = "min_violation"`, distinct from `"squeeze"`, so the state machine can
   cap speed separately and the bag shows which mechanism produced the line.
3. Gate the whole thing behind a parameter defaulting to **off**, so the shipped behaviour is
   byte-identical until someone turns it on deliberately.
4. Gate it additionally on `cur_vs` below `squeeze_max_speed_mps` and on a live relax request, so
   it can only ever fire in the standstill it exists to break.

Step 1 is the only novel part, and it is what the measurement says is redundant.

## What to do instead

The target the last three rounds have converged on from three independent directions — the missed
cells' 23 % ramp-length share, the `obs_box+grid` signature in the ceiling cells, and the 36 % of
still-closed cells with `delta = 0` — is the **entry and return ramp geometry**: the hump reaches
its apex offset but its ramps cross the keep-out or the wall on the way in and out. That is a
bounded change to `_fit_ramp` / `_ramp_limits`, not a rewrite, and it spends no margin at all.
It was explicitly out of scope for round 4.
