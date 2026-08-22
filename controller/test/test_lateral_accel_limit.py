#!/usr/bin/env python3
"""Harness test of the steering tyre limit (Controller._clip_to_tyre_limit).

The controller's five steering modifiers are all tracking DEMAND; none of them knows what the
tyres can deliver. Measured on bag ggv_0812_1645 (5 clean laps, replayed through the real class
by analysis/replay_steering.py, full coverage): the controller commands p90 14.36 / max 22.37
m/s^2 of lateral acceleration against a car that never achieved more than 11.71.

So the command is clipped to delta_max = atan(L * a_lat_ctrl(v) / max(v, floor)^2), with
a_lat_ctrl = a_lat_margin * ggv ay_max INTERPOLATED at the current speed.

THE MARGIN IS THE POINT, and it has been wrong twice.
  * 1.00 (ceiling 5.70) shipped. That is the PLANNER's limit and the raceline spends all of it,
    so there was zero budget left for returning to the line at an apex; the car came back "far
    too sluggish". It also removed capability the record shows for 41.4% of the run.
  * 1.10 (ceiling 6.27) was the first correction, justified by a "tyre knee at 6.0-6.5". That
    knee was an artefact of a replay silently covering 16.3% of the bag. At full coverage there
    is no cliff and 6.27 still cuts 29.9% of demonstrated capability.
The rule now is to anchor the ceiling at the p90 of what the tyre DEMONSTRABLY delivers (7.62),
which margin 1.35 realises at 7.70. test_the_default_margin_leaves_recovery_budget pins both
failure directions: too low cuts real cornering, too high clips nothing.

NOT tested here, and not testable offline: whether the car returns to the line under the new
ceiling. The replay is open loop. That is a sim question.

Run:
  python3 controller/test/test_lateral_accel_limit.py
"""
import types
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
MOD = REPO / "controller" / "controller" / "combined" / "src" / "Controller.py"
ctl = types.ModuleType("ctl")
ctl.__dict__["__file__"] = str(MOD)
exec(compile(MOD.read_text(), str(MOD), "exec"), ctl.__dict__)

WHEELBASE = 0.321
GGV_AY = 5.7                                          # what CAR/SIM ship today
FLAT_GGV = np.array([[0.0, GGV_AY], [72.0, GGV_AY]])
RACELINE_WORST_A_LAT = 5.70                           # measured max vx^2*|kappa| on the line
# What the car DEMONSTRABLY delivered, v*|yaw_rate| over the whole of bag ggv_0812_1645
# (replay_steering.py --sweep). An earlier round put the roof at a "6.0-6.5 tyre knee"; that
# came from a replay covering 16.3% of the bag, and at full coverage there is no cliff at all.
ACHIEVED_P90, ACHIEVED_MAX = 7.62, 11.71


def controller(**over):
    """A real Controller, constructed the way the manager constructs it."""
    kw = dict(t_clip_min=0.7, t_clip_max=8.0, m_l1=0.47, q_l1=-0.2, curvature_factor=0.145,
              KP=0.3, KI=0.0, KD=0.03, heading_error_thres=10.0, steer_gain_for_speed=1.0,
              future_constant=0.05, speed_lookahead=0.25, lat_err_coeff=1.0,
              acc_scaler_for_steer=1.0, dec_scaler_for_steer=1.0, start_scale_speed=6.5,
              end_scale_speed=10.0, downscale_factor=0.5, speed_lookahead_for_steer=0.0,
              trailing_gap=1.0, trailing_vel_gain=0.25, trailing_p_gain=1.35,
              trailing_i_gain=0.0, trailing_d_gain=1.0, blind_trailing_speed=1.5,
              loop_rate=50, wheelbase=WHEELBASE, speed_factor_for_lat_err=1.0,
              speed_factor_for_curvature=1.0, speed_diff_thres=0.1, start_speed=10.0,
              start_curvature_factor=1.5, AEB_thres=0.5, converter=None,
              logger_info=lambda *a, **k: None, logger_warn=lambda *a, **k: None)
    c = ctl.Controller(**kw)
    c.ggv_table = FLAT_GGV
    for k, v in over.items():
        setattr(c, k, v)
    return c


def steer_cycle(c, speed, demand_lat_m=3.0, cycles=1):
    """Drive the REAL calc_steering_angle_for_future; returns the commands it produced.

    The L1 point is placed hard to one side so the raw pure-pursuit demand is far past
    anything the tyres could hold at `speed` -- the case the clip exists for.
    """
    c.state = "GB_TRACK"
    c.opponent = None
    c.speed_now = float(speed)
    c.acc_now = np.zeros(10)
    c.future_lat_err = 0.0
    c.future_position = np.array([[0.0, 0.0, 0.0]])
    wpnts = np.zeros((20, 9))
    wpnts[:, 0] = np.linspace(0.0, 10.0, 20)          # x ahead of the car
    wpnts[:, 2] = speed                               # vx column, read for the lookahead speed
    c.waypoint_array_in_map = wpnts
    L1_point = np.array([4.0, demand_lat_m])
    out = []
    for _ in range(cycles):
        out.append(c.calc_steering_angle_for_future(L1_point, 4.0, 0.0, 0.0, [speed, 0.0]))
    return out


def implied_a_lat(delta, v, wheelbase=WHEELBASE):
    return v * v * abs(np.tan(delta)) / wheelbase


def yaml_params():
    cfg = REPO / "stack_master" / "config" / "controller.yaml"
    return next(iter(yaml.safe_load(cfg.read_text()).values()))["ros__parameters"]


def test_the_default_margin_leaves_recovery_budget():
    # THE regression, both halves of it. margin 1.0 sets the controller's ceiling equal to the
    # PLANNER's limit, and the raceline already spends all of it -- so at a corner apex there is
    # nothing left to steer with toward the line. It shipped, and the car could not follow the
    # path. The ceiling must also not cut into what the tyre demonstrably delivers, which is the
    # error the FIRST correction still made: 6.27 removes capability the record shows for 29.9%
    # of the run.
    for margin in (controller().a_lat_margin, yaml_params()["a_lat_margin"]):
        ceiling = margin * GGV_AY
        assert margin > 1.0, "margin 1.0 is the ceiling that regressed on the real car"
        assert ceiling > RACELINE_WORST_A_LAT, \
            f"ceiling {ceiling:.2f} <= the raceline's own worst demand {RACELINE_WORST_A_LAT}"
        # anchored at the p90 of demonstrated capability: below it, the clip is taking away
        # cornering the car has been recorded doing
        assert ceiling >= 0.95 * ACHIEVED_P90, \
            f"ceiling {ceiling:.2f} cuts into demonstrated capability (p90 {ACHIEVED_P90})"
        # ...and there is still a roof: never permit more than the car has EVER achieved, or
        # the clip is decoration
        assert ceiling <= ACHIEVED_MAX, \
            f"ceiling {ceiling:.2f} exceeds the achieved max {ACHIEVED_MAX} -- clips nothing real"
    assert controller().a_lat_margin == yaml_params()["a_lat_margin"], \
        "the code default and controller.yaml disagree about the ceiling"
    budget = controller().a_lat_margin * GGV_AY - RACELINE_WORST_A_LAT
    print(f"PASS the ceiling sits {budget:.2f} m/s^2 above the raceline, at the p90 of what the "
          f"tyre delivers, under its max")


def test_delta_max_is_the_bicycle_model_at_each_speed():
    c = controller(a_lat_margin=1.0)                  # 1.0 keeps the arithmetic readable
    for v in (2.0, 3.0, 6.0, 9.0):
        c.speed_now = v
        expected = np.arctan(WHEELBASE * GGV_AY / (v * v))
        got = c._clip_to_tyre_limit(0.53)             # a demand past any of these limits
        assert abs(got - expected) < 1e-12, f"v={v}: {got} != {expected}"
        # ...and that angle is exactly the ceiling, which is the point of the formula
        assert abs(implied_a_lat(got, v) - GGV_AY) < 1e-9
    # the limit must TIGHTEN with speed, or it is not a tyre limit
    lims = [c._clip_to_tyre_limit(0.53) for c.speed_now in (2.0, 4.0, 8.0)]
    assert lims[0] > lims[1] > lims[2], lims
    print("PASS delta_max = atan(L*a_lat_ctrl/v^2) at every speed, and tightens with speed")


def test_the_budget_is_interpolated_not_folded_to_a_scalar():
    # ggv.csv is flat at 5.7 today, so min / mean / interp agree and a scalar would look right.
    # It stops looking right the moment the table becomes speed-dependent: pin the interpolation.
    sloped = np.array([[0.0, 8.0], [10.0, 4.0]])
    c = controller(ggv_table=sloped, a_lat_margin=1.0)
    assert abs(c.a_lat_max_now(5.0) - 6.0) < 1e-12, c.a_lat_max_now(5.0)
    assert abs(c.a_lat_max_now(2.5) - 7.0) < 1e-12
    assert c.a_lat_max_now(5.0) != sloped[:, 1].min(), "folded to the table minimum"
    # outside the table numpy holds the end values -- no extrapolation past what was measured
    assert abs(c.a_lat_max_now(99.0) - 4.0) < 1e-12
    assert abs(c.a_lat_max_now(-1.0) - 8.0) < 1e-12
    print("PASS ay_max is interpolated at the current speed, not reduced to one number")


def test_the_v_floor_bounds_the_low_speed_limit():
    c = controller(a_lat_v_floor=0.5, a_lat_margin=1.0)
    at_floor = np.arctan(WHEELBASE * GGV_AY / 0.25)
    for v in (0.0, 0.1, 0.5):
        c.speed_now = v
        assert abs(c._clip_to_tyre_limit(1.5) - at_floor) < 1e-12, v
    assert at_floor < np.pi / 2                       # finite, unlike the v -> 0 limit
    # above the floor the floor stops acting
    c.speed_now = 1.0
    assert c._clip_to_tyre_limit(1.5) < at_floor
    # and without it the expression runs away to pi/2 -- what the floor exists to prevent
    assert np.arctan(WHEELBASE * GGV_AY / 1e-6) > np.pi / 2 - 1e-6
    print("PASS v_floor holds delta_max finite below 0.5 m/s and stops acting above it")


def test_the_margin_scales_the_ceiling():
    for margin in (0.8, 1.0, 1.35, 1.6):
        c = controller(a_lat_margin=margin, speed_now=6.0)
        assert abs(c.a_lat_max_now(6.0) - margin * GGV_AY) < 1e-12
        assert abs(implied_a_lat(c._clip_to_tyre_limit(0.53), 6.0) - margin * GGV_AY) < 1e-9
    # and a bigger margin is a looser limit, monotonically
    lims = [controller(a_lat_margin=m, speed_now=6.0)._clip_to_tyre_limit(0.53)
            for m in (1.0, 1.35, 1.6)]
    assert lims[0] < lims[1] < lims[2], lims
    print("PASS a_lat_margin scales the ceiling, and a bigger margin is a looser limit")


def test_the_clipped_value_is_what_the_next_cycle_starts_from():
    # The rate limiter and the driver-visible command both read curr_steering_angle. If the clip
    # did not reach it, the reference would drift away from what was published, a cycle at a
    # time, and the rate limit would be measured against a command that never existed.
    c = controller()
    cmds = steer_cycle(c, speed=6.0, cycles=3)
    ceiling = c.a_lat_margin * GGV_AY
    lim = np.arctan(WHEELBASE * ceiling / 36.0)
    for cmd in cmds:
        assert abs(cmd) <= lim + 1e-12, f"{cmd} > {lim}"
        assert implied_a_lat(cmd, 6.0) <= ceiling + 1e-9
    assert c.curr_steering_angle == cmds[-1], (c.curr_steering_angle, cmds[-1])
    print(f"PASS the command is clipped to {lim:.4f} rad and carried into curr_steering_angle")


def test_off_leaves_the_old_path_untouched():
    # The live escape hatch: `ros2 param set /controller_manager a_lat_limit_enable false`.
    # It must not merely widen the limit -- it must not run the clip at all.
    off = controller(a_lat_limit_enable=False)

    def explode(*a, **k):
        raise AssertionError("_clip_to_tyre_limit ran with a_lat_limit_enable false")

    off._clip_to_tyre_limit = explode
    off_cmds = steer_cycle(off, speed=6.0, cycles=5)

    # ...and the numbers it produces are the ones a ceiling too high to bite produces, i.e.
    # nothing else in the function changed
    wide = controller(a_lat_margin=1e6)
    wide_cmds = steer_cycle(wide, speed=6.0, cycles=5)
    assert off_cmds == wide_cmds, list(zip(off_cmds, wide_cmds))
    assert max(abs(x) for x in off_cmds) > np.arctan(
        WHEELBASE * controller().a_lat_margin * GGV_AY / 36.0), \
        "the test case must be one the limit would actually have bitten"
    print("PASS with the parameter off the clip never runs and the command is unchanged")


def test_a_missing_table_is_reported_not_guessed():
    warned = []
    c = controller(ggv_table=None, logger_warn=lambda m, *a, **k: warned.append(m))
    assert c.a_lat_max_now(6.0) is None
    cmds = steer_cycle(c, speed=6.0, cycles=4)
    assert max(abs(x) for x in cmds) > np.arctan(
        WHEELBASE * controller().a_lat_margin * GGV_AY / 36.0), \
        "silently clipped anyway"
    assert len(warned) == 1, f"warned {len(warned)} times, want exactly once: {warned}"
    assert "INERT" in warned[0]
    print("PASS no ggv -> the limit is inert and says so once, rather than inventing a number")


def test_the_loader_reads_the_shipped_csvs():
    for version in ("CAR", "SIM"):
        path = REPO / "stack_master" / "config" / version / "veh_dyn_info" / "ggv.csv"
        tbl = ctl.load_ggv_ay(str(path))
        assert tbl.ndim == 2 and tbl.shape[1] == 2, tbl.shape
        assert tbl.shape[0] >= 2, f"{version}: {tbl.shape[0]} row(s), cannot interpolate"
        assert np.all(np.diff(tbl[:, 0]) > 0), f"{version}: speed column is not increasing"
        assert np.all(np.isfinite(tbl[:, 1])) and np.all(tbl[:, 1] > 0), tbl[:, 1]
        # column 1 must be ay_max (the THIRD csv column), not ax_max -- today 5.7 against 7.0
        raw = np.atleast_2d(np.loadtxt(str(path), delimiter=",", comments="#"))
        assert np.allclose(tbl[:, 1], raw[:, 2]), "loader picked the wrong column"
        print(f"  {version}: {tbl.shape[0]} rows, ay_max "
              f"{tbl[:, 1].min():.2f}-{tbl[:, 1].max():.2f} m/s^2")
    print("PASS the loader reads both shipped ggv tables, header comments and all")


if __name__ == "__main__":
    test_the_default_margin_leaves_recovery_budget()
    test_delta_max_is_the_bicycle_model_at_each_speed()
    test_the_budget_is_interpolated_not_folded_to_a_scalar()
    test_the_v_floor_bounds_the_low_speed_limit()
    test_the_margin_scales_the_ceiling()
    test_the_clipped_value_is_what_the_next_cycle_starts_from()
    test_off_leaves_the_old_path_untouched()
    test_a_missing_table_is_reported_not_guessed()
    test_the_loader_reads_the_shipped_csvs()
    print("ALL PASS")
