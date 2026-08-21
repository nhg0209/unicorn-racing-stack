#!/usr/bin/env python3
"""One open-segment QP: the corridor decides the whole lateral profile, not just the apex.

    min ||D2 d||^2 + w_dev ||d||^2      s.t.   lo(s) <= d(s) <= hi(s)
                                               d(s0) = d0,  d'(s0) = dp0

`lo`/`hi` are the caller's business and carry everything that is not smoothness: the measured
track corridor, the keep-out of every box the path is shaped around, and the terminal condition.
Because side selection and keep-out are already IN the bounds, the corridor forces the apex offset
-- there is nothing to pin at the apex, and pinning it would over-constrain a segment whose ends
are pinned already. That is measured, not assumed: see RAMP_CEILING_NOTE / CORRIDOR_QP_NOTE.

NO ROS, NO node state. Every offline gate depends on that: the same arithmetic has to be runnable
from a sweep script with nothing but numpy, scipy and quadprog on the path.

Two things this module deliberately does NOT do.

  * It does not reuse `closed_reopt._second_difference`. That operator is PERIODIC -- it carries
    D2[0, -1] = D2[-1, 0] = 1 so a lap joins up with itself -- and this run is a segment with two
    different ends (the pre-ramp on one side, the raceline tail on the other). Those two wrap rows
    would charge the segment for a bend between its first station and its last, several metres
    apart with the whole maneuver in between, and bend it to join its own two ends. On a 0.4 m ramp
    profile the closed operator scores 3201 against this one's 0.3; that difference IS the two
    phantom rows. `open_second_difference` is the same three-point stencil with them gone.

  * It does not solve one variable per published station. The published grid is 0.1 m, so a
    three-box maneuver is 200+ stations and the solve runs once per terminal-offset candidate
    inside a 20 Hz cycle. The unknowns are a cubic-spline control grid of `max_vars` points; the
    BOUNDS are still enforced at every published station (`lo <= B c <= hi`), so the reduction
    costs expressiveness and never costs feasibility -- which is the direction that matters, since
    a station the constraint skips is exactly the failure this whole round is about.

A TRAP WORTH CARRYING OUT OF THIS FILE, because it will recur anywhere a reduced basis meets a
tight constraint. If a station whose box is DEGENERATE -- pinned to one value within a micrometre --
does not coincide with a control point, a cubic can only satisfy it between its knots by accident,
and quadprog reports `constraints are inconsistent, no solution`. That message names the CORRIDOR,
and the corridor is fine; what is too coarse is the basis. Measured here: 13.2 % of solves, every
one of them recovered at full resolution. The diagnosis is cheap once you know to make it -- re-solve
with one unknown per station, where boxes on the unknowns themselves are feasible by construction
whenever lo <= hi -- and the fix is `_ctrl_indices`, which forces every pinned station to be a
control point. Read an infeasibility report from a reduced-basis QP as a question, not a verdict.
"""
from typing import Optional, Tuple

import numpy as np

try:                                   # pragma: no cover - import shape, not logic
    import quadprog
except ImportError:                    # pragma: no cover
    quadprog = None

from scipy.interpolate import CubicSpline

__all__ = ["open_second_difference", "solve_corridor_path", "cut_keepout"]

_RIDGE_LADDER = (1e-9, 1e-7, 1e-5)     # relative to the Hessian's own scale; see _solve_qp
_PIN_TOL = 1e-4                        # a box this tight is a PIN, not a corridor; see _ctrl_indices


def open_second_difference(s_grid: np.ndarray) -> np.ndarray:
    """(n-2, n) second-derivative stencil over an OPEN station run: one row per INTERIOR station.

    The general non-uniform three-point formula, which collapses to the familiar
    [1, -2, 1] / ds^2 when the stations are evenly spaced (they are -- the published grid is
    uniform at wpnt_dist -- but the closed form costs nothing and removes an assumption).

    There is no row for station 0 or station n-1: an open segment has no second difference at its
    ends. That absence is the whole difference from the periodic operator, and it is why this
    function exists instead of a slice of one.
    """
    s = np.asarray(s_grid, dtype=float)
    n = s.size
    if n < 3:
        raise ValueError("open_second_difference needs at least 3 stations")
    h1 = s[1:-1] - s[:-2]                                    # (n-2,)
    h2 = s[2:] - s[1:-1]
    if np.any(h1 <= 0.0) or np.any(h2 <= 0.0):
        raise ValueError("station grid must be strictly increasing")
    D2 = np.zeros((n - 2, n))
    rows = np.arange(n - 2)
    D2[rows, rows] = 2.0 / (h1 * (h1 + h2))
    D2[rows, rows + 1] = -2.0 / (h1 * h2)
    D2[rows, rows + 2] = 2.0 / (h2 * (h1 + h2))
    return D2


def cut_keepout(lo: np.ndarray, hi: np.ndarray, mask: np.ndarray,
                box_lo: float, box_hi: float, pass_left: bool, bulge: float = 0.0) -> None:
    """Subtract one box's keep-out from the corridor, IN PLACE, over the stations `mask` selects.

    `pass_left` is the side the caller already chose; this function does not choose. The bulge is a
    PREFERENCE, not a constraint: it is taken only as far as the corridor allows, which is the same
    thing the node's own peak-clipping does (`np.clip(da + _bulge_away_from(...), lo_k, hi_k)`) and
    is what stops a section narrower than the design margins from reading as impassable.

    Applied over the box's WHOLE s-inflated interval rather than at the apex station alone. That is
    stricter than the quintic, which only carries the bulge at its peak -- the conservative
    direction, and the one that keeps a corridor-solved path from passing closer to a box than a
    sampled one would.
    """
    if not np.any(mask):
        return
    if pass_left:
        want = box_hi + bulge
        lo[mask] = np.maximum(lo[mask], np.minimum(want, hi[mask]))
    else:
        want = box_lo - bulge
        hi[mask] = np.minimum(hi[mask], np.maximum(want, lo[mask]))


def _uniform_ds(s: np.ndarray) -> Optional[float]:
    """The spacing if the grid is uniform to a part in 1e6, else None."""
    d = np.diff(s)
    if d.size == 0 or np.any(d <= 0.0):
        return None
    ds = float(d.mean())
    return ds if float(np.max(np.abs(d - ds))) <= 1e-6 * ds else None


_BASIS_CACHE = {}


def _spline_basis(s_ctrl: np.ndarray, s_eval: np.ndarray) -> np.ndarray:
    """(len(s_eval), len(s_ctrl)) natural-cubic-spline evaluation matrix: d(s_eval) = B @ c.

    ONE CubicSpline over the identity matrix, not one per control point: the interpolant is linear
    in the control values, so the whole basis comes out of a single vector-valued solve. Doing it
    the obvious way costs ~13 ms for 60 controls, which is the entire cycle budget.
    """
    return np.asarray(CubicSpline(s_ctrl, np.eye(len(s_ctrl)), axis=0,
                                  bc_type="natural")(s_eval), dtype=float)


def _unit_basis(n: int, ctrl: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(B, AtA, BtB) for a UNIT-spaced run of n stations with control points at station `ctrl`.

    Cached, and cached on (n, ctrl) alone, because all three are invariant under an affine
    reparameterisation of s: a natural cubic spline through control points at `a + ds*t` evaluated
    at `a + ds*k` gives the same weights as one at `t` evaluated at `k`, and the second-difference
    operator just picks up 1/ds^2. The caller scales AtA by 1/ds^4 and is done.

    This is what makes the per-candidate cost the quadprog solve and nothing else: r_in/r_out take
    a handful of distinct values, so a cycle's candidates share a handful of layouts.
    """
    hit = _BASIS_CACHE.get((n, ctrl))
    if hit is not None:
        return hit
    t = np.arange(n, dtype=float)
    B = np.eye(n) if len(ctrl) >= n else _spline_basis(np.asarray(ctrl, dtype=float), t)
    A = open_second_difference(t) @ B
    out = (B, A.T @ A, B.T @ B)
    if len(_BASIS_CACHE) < 512:
        _BASIS_CACHE[(n, ctrl)] = out
    return out


def _ctrl_indices(n: int, n_var: int, forced: np.ndarray) -> Tuple[int, ...]:
    """Which stations carry an unknown: `n_var` spread evenly, plus every station in `forced`.

    THE PINS HAVE TO BE CONTROL POINTS. A station whose box is degenerate (the terminal d = 0, a
    held apex band) asks the profile to take one value to within a micrometre; if that station sits
    BETWEEN two control points, a cubic can only satisfy it by accident, and quadprog reports the
    whole constraint set inconsistent rather than "your basis is too coarse". Measured before this:
    13 % of solves at 60 controls, and nearly all of them under corridor_qp_pin_apex -- the variant
    with the most degenerate boxes. Making the pinned stations unknowns turns those constraints
    into plain boxes on single variables, which are feasible by construction.
    """
    base = np.round(np.linspace(0.0, float(n - 1), max(int(n_var), 4))).astype(int)
    return tuple(int(v) for v in np.unique(np.concatenate([base, np.asarray(forced, dtype=int)])))


_INCONSISTENT = "inconsistent"                          # sentinel: the CONSTRAINTS have no solution


def _solve_qp(G0: np.ndarray, C: np.ndarray, b: np.ndarray, meq: int):
    """quadprog with a ridge ladder. G0 is the Hessian WITHOUT the ridge.

    Returns the solution, `_INCONSISTENT`, or None.

    D2'D2 is singular -- an affine profile has no second difference -- so a ridge is needed for the
    Cholesky, and it has to be RELATIVE: with ds = 0.1 the bending block carries a 1/ds^4 = 1e4
    factor, and an absolute 1e-9 is 1e-13 of that. Escalating rather than starting large keeps the
    answer the one the objective actually asked for whenever the small ridge suffices.

    The two failures are told APART, because only one of them is worth a retry. A bigger ridge
    cannot make an infeasible constraint set feasible, and quadprog does not decide infeasibility
    cheaply -- it iterates first. Re-running the whole ladder on it turned a 0.5 ms solve into
    14 ms at p95, which was the entire corridor_qp cost overrun. `_INCONSISTENT` goes back to the
    caller, which has a retry that CAN help: a finer control grid.
    """
    n = G0.shape[0]
    scale = max(1.0, float(np.trace(G0)) / max(n, 1))
    zero = np.zeros(n)
    for eps in _RIDGE_LADDER:
        try:
            return quadprog.solve_qp(G0 + (eps * scale) * np.eye(n), zero, C, b, meq)[0]
        except ValueError as e:                        # noqa: PERF203 -- three rungs at most
            if "positive definite" not in str(e):
                return _INCONSISTENT
        except Exception:                              # noqa: BLE001
            return None
    return None


def solve_corridor_path(s_grid, lo, hi, d0: float, dp0: float,
                        w_dev: float = 0.0, max_vars: int = 60,
                        dpp0: Optional[float] = None) -> Optional[np.ndarray]:
    """Minimum-bending d(s) inside [lo, hi], leaving the start at (d0, dp0). None if unsolvable.

    Returns d at EVERY station of `s_grid` -- the caller's own grid, so the result splices back
    without an interpolation step of its own.

    The start pin is equalities on the station grid, one-sided, at the publishing resolution --
    because that is the discretisation the seam is measured in, and a controller reading the
    published points sees exactly these differences:

        d[0] = d0                                        value
        d[1] = d0 + dp0*(s1-s0)                          slope       (C1)
        d[2] - 2 d[1] + d[0] = dpp0*(s1-s0)^2            curvature   (C2, only if dpp0 is given)

    C1 ALONE IS NOT ENOUGH, and the number says so. With the value and slope matched but the
    curvature free, the solver is entitled to start bending at the first station it owns: measured
    over the full race profile, the |d'| step at that station came out at p90 0.0548 (ifac) /
    0.0504 (ifac_0807) against a 0.05 target -- three times better than the round-5 ceiling's
    unpinned QP (0.149-0.179) and still over. `dpp0` closes it by continuing the prefix's own
    curvature instead of leaving it to the objective. The quintic has had this all along: its first
    breakpoint is `[d_start, dp0, 0.0]`, i.e. C2 by construction.

    The pinned stations' boxes are widened to admit the pinned values if the corridor does not
    already contain them -- the pin is given, not negotiable, and what judges the result is the full
    gate stack on the path afterwards, not this function.

    The END is the caller's job, through `lo`/`hi`: collapsing the last two stations onto 0 is what
    makes the profile meet the raceline tail with d = 0 and d' = 0.
    """
    if quadprog is None:
        return None
    s = np.asarray(s_grid, dtype=float)
    lo = np.array(lo, dtype=float, copy=True)
    hi = np.array(hi, dtype=float, copy=True)
    n = s.size
    if n < 5 or lo.size != n or hi.size != n:
        return None
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(np.isfinite(s))):
        return None
    if np.any(np.diff(s) <= 0.0) or np.any(hi < lo - 1e-9):
        return None

    h0 = float(s[1] - s[0])
    d1 = float(d0) + float(dp0) * h0
    pins = [(0, float(d0)), (1, d1)]
    if dpp0 is not None:
        pins.append((2, 2.0 * d1 - float(d0) + float(dpp0) * h0 * h0))
    for j, v in pins:                                   # the pin is given: admit it
        lo[j] = min(lo[j], v)
        hi[j] = max(hi[j], v)
    lo -= 1e-6                                          # the same slack closed_reopt gives its box
    hi += 1e-6

    ds = _uniform_ds(s)
    b = np.concatenate([[v for _j, v in pins], lo, -hi])
    n_eq = len(pins)
    # Stations the caller has pinned (a degenerate box), plus the ones the start pin fixes.
    forced = np.concatenate([np.flatnonzero(hi - lo <= _PIN_TOL), [j for j, _v in pins]])

    def _attempt(ctrl):
        if ds is not None:
            B, AtA, BtB = _unit_basis(n, ctrl)
            AtA = AtA / (ds ** 4)
        else:                                           # non-uniform stations: build it straight
            B = (np.eye(n) if len(ctrl) >= n
                 else _spline_basis(s[np.asarray(ctrl, dtype=int)], s))
            A = open_second_difference(s) @ B
            AtA, BtB = A.T @ A, B.T @ B
        Bt = B.T                                        # (n_var, n)
        C = np.hstack([Bt[:, [j for j, _v in pins]], Bt, -Bt])
        c = _solve_qp(2.0 * (AtA + float(w_dev) * BtB), C, b, n_eq)
        if c is _INCONSISTENT or c is None:
            return c
        d = B @ c
        return d if np.all(np.isfinite(d)) else None

    # Full resolution is the second rung and the last: with one unknown per station the bounds are
    # boxes on the unknowns themselves, so the set is feasible by construction whenever lo <= hi
    # -- which the caller has already checked. Anything that still comes back inconsistent there is
    # a corridor that genuinely has no path in it, and the caller is told so with None.
    first = _ctrl_indices(n, int(np.clip(max_vars, 4, n)), forced)
    for ctrl in dict.fromkeys((first, tuple(range(n)))):
        out = _attempt(ctrl)
        if out is not _INCONSISTENT:
            return out
    return None
