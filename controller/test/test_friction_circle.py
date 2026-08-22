#!/usr/bin/env python3
"""Harness test of the friction circle on the speed command (Controller._ax_avail).

WHY IT EXISTS. Measured on ifac_0807's right-hand slalom (real bag ggv_0812_1645, laps 2 and 3
alike, reproduced by analysis/replay_steering.py --section): at the apex s=5.2 the raceline asks
kappa -0.27 at a planned 4.51 m/s, the car is doing 5.27, so 5.27^2*0.27 = 7.4 m/s^2 of lateral
demand against a ggv ay_max of 5.7. The car is already outside the friction circle -- and the
controller commands 5.77 m/s there, i.e. it asks for MORE speed, because speed_lookahead 0.25 s
reads ~1.2 m ahead, which at the apex is the corner exit. Every m/s^2 spent going faster is
lateral force the tyre no longer has, and d grows monotonically to +0.42.

So the UP slew allowance on the published speed command becomes

    ax_avail = min( ax_machines(v), ax_max(v) * (1 - (a_lat/(ay_max(v)*margin))^p)^(1/p) )
    a_lat    = v^2 * |kappa| of the raceline at the nearest waypoint

with every one of ax_max, ay_max, ax_machines and p READ from the same three files the offline
velocity profile is solved against. Past the circle ax_avail is 0 and the command may not rise.
Braking is deliberately NOT forced -- the decel allowance is untouched.

NOT tested here, and not testable offline: whether d actually stops growing. The replay is open
loop; it shows the command flattening and nothing about the trajectory that follows.

Run:
  python3 controller/test/test_friction_circle.py
"""
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
MOD = REPO / "controller" / "controller" / "combined" / "src" / "Controller.py"
ctl = types.ModuleType("ctl")
ctl.__dict__["__file__"] = str(MOD)
exec(compile(MOD.read_text(), str(MOD), "exec"), ctl.__dict__)

CAR = REPO / "stack_master" / "config" / "CAR"
AY, AX, AXM, BAX = 5.7, 7.0, 9.5, 10.0            # what CAR ships today, at low speed
FLAT = lambda x: np.array([[0.0, x], [72.0, x]])  # noqa: E731
# The slew limit is derived from those tables too since 2026-08-12 (Controller._slew_limits), so
# the step these fixtures expect follows from them rather than from controller.yaml's 5.0.
SLEW_ACCEL, SLEW_DECEL = min(AX, AXM), min(AX, BAX)


def controller(**over):
    """A Controller with the friction circle wired the way the manager wires it."""
    kw = dict(t_clip_min=0.7, t_clip_max=8.0, m_l1=0.47, q_l1=-0.2, curvature_factor=0.145,
              KP=0.3, KI=0.0, KD=0.03, heading_error_thres=10.0, steer_gain_for_speed=1.0,
              future_constant=0.05, speed_lookahead=0.25, lat_err_coeff=1.0,
              acc_scaler_for_steer=1.0, dec_scaler_for_steer=1.0, start_scale_speed=6.5,
              end_scale_speed=10.0, downscale_factor=0.5, speed_lookahead_for_steer=0.0,
              trailing_gap=1.0, trailing_vel_gain=0.25, trailing_p_gain=1.35,
              trailing_i_gain=0.0, trailing_d_gain=1.0, blind_trailing_speed=1.5,
              loop_rate=50, wheelbase=0.321, speed_factor_for_lat_err=1.0,
              speed_factor_for_curvature=1.0, speed_diff_thres=0.1, start_speed=10.0,
              start_curvature_factor=1.5, AEB_thres=0.5, converter=None,
              logger_info=lambda *a, **k: None, logger_warn=lambda *a, **k: None)
    c = ctl.Controller(**kw)
    c.ggv_table, c.ggv_ax_table = FLAT(AY), FLAT(AX)
    c.ax_machines_table, c.dyn_model_exp = FLAT(AXM), 2.0
    c.b_ax_table = FLAT(BAX)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def at(c, speed, kappa):
    """Put the car at `speed` on a waypoint of curvature `kappa`, then ask the circle."""
    c.speed_now = float(speed)
    c.idx_nearest_waypoint = 0
    w = np.zeros((5, 9))
    w[:, 5] = kappa
    c.waypoint_array_in_map = w
    return c._ax_avail()


def test_outside_the_circle_there_is_no_acceleration_left():
    # THE case. Apex of the slalom: kappa -0.27 at 5.27 m/s is 7.4 m/s^2 against ay_max 5.7.
    c = controller()
    assert at(c, 5.27, -0.27) == 0.0
    # exactly ON the circle is also zero, and the clamp means past it never goes negative or NaN
    # abs, not rel: v -> kappa -> v^2*kappa round-trips with ~1e-7 of float error,
    # which is 1e-7 m/s^2 of 'available' acceleration and physically zero
    v_on = np.sqrt(AY / 0.27)
    assert at(c, v_on, -0.27) == pytest.approx(0.0, abs=1e-6)
    for v in (6.0, 8.0, 20.0):
        got = at(c, v, -0.27)
        assert got == 0.0 and np.isfinite(got), f"v={v}: {got}"
    print("PASS past ay_max the available longitudinal acceleration is exactly 0, never NaN")


def test_p2_is_the_ellipse():
    c = controller(dyn_model_exp=2.0)
    # (ax/ax_max)^2 + (a_lat/ay_max)^2 = 1
    for frac in (0.0, 0.3, 0.6, 0.8, 1.0):
        a_lat = frac * AY
        v = 3.0
        kappa = a_lat / (v * v)
        got = at(c, v, kappa)
        want = min(AXM, AX * np.sqrt(1.0 - frac ** 2))
        assert got == pytest.approx(want, rel=1e-9, abs=1e-6), f"frac={frac}: {got} != {want}"
        assert (got / AX) ** 2 + frac ** 2 == pytest.approx(1.0, rel=1e-9) or got == AXM
    # p=1 is the diamond, and it is strictly tighter than the ellipse in between
    c1 = controller(dyn_model_exp=1.0)
    v, kappa = 3.0, 0.5 * AY / 9.0
    assert at(c1, v, kappa) == pytest.approx(AX * 0.5, rel=1e-9)
    assert at(c1, v, kappa) < at(c, v, kappa)
    print("PASS p=2 traces the ellipse and p=1 the diamond, from racecar_f110.ini's exponent")


def test_the_machine_cap_and_the_interpolation_both_apply():
    # ax_max_machines is genuinely speed-dependent (9.5 at rest, 4.62 at 15 m/s), so it must be
    # interpolated and it must be able to bind before the friction term does.
    c = controller(ax_machines_table=np.array([[0.0, 9.5], [15.0, 4.5]]))
    assert at(c, 0.0, 0.0) == pytest.approx(AX)          # 9.5 > 7.0, friction term wins
    assert at(c, 15.0, 0.0) == pytest.approx(4.5)        # machine cap wins
    assert at(c, 7.5, 0.0) == pytest.approx(7.0)         # interpolated 7.0 == ggv ax_max
    # ...and the ggv columns are interpolated too, not folded to a scalar
    c2 = controller(ggv_table=np.array([[0.0, 8.0], [10.0, 4.0]]),
                    ggv_ax_table=np.array([[0.0, 8.0], [10.0, 4.0]]))
    assert c2._interp(c2.ggv_table, 5.0) == pytest.approx(6.0)
    assert at(c2, 5.0, 0.0) == pytest.approx(6.0)        # ax_max(5) = 6.0, under the 9.5 machine
    print("PASS the machine cap binds where it should and every table is interpolated")


def test_the_margin_scales_the_lateral_budget():
    tight, loose = controller(a_comb_margin=1.0), controller(a_comb_margin=1.5)
    # 1.2 x ay_max of lateral: clearly past the default circle, comfortably inside a 1.5x one.
    # (Sitting exactly ON the boundary makes this a float-noise test, not a margin test -- the
    # boundary itself is pinned in test_outside_the_circle_there_is_no_acceleration_left.)
    v, kappa = 3.0, 1.2 * AY / 9.0
    assert at(tight, v, kappa) == 0.0
    assert at(loose, v, kappa) == pytest.approx(AX * np.sqrt(1 - (1.2 / 1.5) ** 2), rel=1e-6)
    print("PASS a_comb_margin scales the lateral budget the circle divides by")


def test_it_refuses_acceleration_without_commanding_braking():
    # The published command may not RISE past the circle; it must still be free to FALL, at the
    # unchanged decel limit. Forcing a decrease is a different intervention.
    # max_accel/max_decel_mps2 are passed but IGNORED here: _slew_limits derives from the tables
    # whenever they are readable, and this fixture supplies all four. They are the fallback the
    # missing-input test below exercises.
    c = controller(max_accel_mps2=5.0, max_decel_mps2=5.0)
    c.speed_now, c.idx_nearest_waypoint = 5.27, 0
    w = np.zeros((5, 9))
    w[:, 5] = -0.27
    c.waypoint_array_in_map = w
    c._speed_cmd_prev = 5.0
    assert c._slew_limit_speed(7.0) == pytest.approx(5.0), "the command rose outside the circle"
    c._speed_cmd_prev = 5.0
    assert c._slew_limit_speed(1.0) == pytest.approx(5.0 - SLEW_DECEL / 50.0), \
        "braking was blocked too"
    # with grip to spare the normal accel limit is back
    w[:, 5] = 0.0
    c._speed_cmd_prev = 5.0
    assert c._slew_limit_speed(7.0) == pytest.approx(5.0 + SLEW_ACCEL / 50.0)
    print("PASS outside the circle the command cannot rise, and braking is left alone")


def test_off_is_the_old_slew_limiter_exactly():
    off = controller(a_comb_limit_enable=False)

    def explode(*a, **k):
        raise AssertionError("_ax_avail ran with a_comb_limit_enable false")

    off._ax_avail = explode
    off.speed_now, off.idx_nearest_waypoint = 5.27, 0
    w = np.zeros((5, 9))
    w[:, 5] = -0.27
    off.waypoint_array_in_map = w
    off._speed_cmd_prev = 5.0
    assert off._slew_limit_speed(7.0) == pytest.approx(5.0 + SLEW_ACCEL / 50.0)
    print("PASS with the parameter off the circle never runs and the slew limit is the derived "
          "one")


def test_a_missing_input_disables_it_and_says_so_once():
    """A missing table makes the circle inert AND, where it is one of the slew inputs, drops the
    slew limit to its controller.yaml fallback. Each consumer warns once, on its own.

    THE TWO ARE NOT THE SAME SET, which is the point of the table below. The circle needs all
    four inputs; _slew_limits needs ggv_ax, ax_machines and b_ax and does NOT need ay or p. So
    losing ggv.csv's ay column or the .ini's dyn_model_exp leaves the slew limit fully derived,
    while losing ggv_ax or ax_machines drops it to the yaml. Asserting one warning for every case
    (which this test did until the slew limit started reading these files) would have hidden a
    silent fallback on two of the four.
    """
    # missing input -> (does the slew limit still derive?, the accel it then uses)
    # The slew fallback is all-or-nothing on purpose: losing any ONE of its three tables sends
    # BOTH directions to the yaml, rather than deriving one side and guessing the other.
    cases = {"ggv_table": (True, SLEW_ACCEL),        # ay: circle only
             "dyn_model_exp": (True, SLEW_ACCEL),    # p: circle only
             "ggv_ax_table": (False, 5.0),           # shared -> circle inert AND slew falls back
             "ax_machines_table": (False, 5.0),      # shared -> circle inert AND slew falls back
             "b_ax_table": (None, 5.0)}              # slew only: circle fine, slew falls back
    for missing, (slew_ok, accel) in cases.items():
        warned = []
        c = controller(**{missing: None}, logger_warn=lambda m, *a, **k: warned.append(m))
        c.speed_now, c.idx_nearest_waypoint = 5.27, 0
        w = np.zeros((5, 9))
        w[:, 5] = 0.0            # grip to spare, so the circle is not what caps the rise
        c.waypoint_array_in_map = w
        if slew_ok is None:      # b_ax is not a circle input, so the circle still answers
            assert c._ax_avail() is not None, missing
        else:
            assert c._ax_avail() is None, missing
        c._speed_cmd_prev = 5.0
        for _ in range(5):
            out = c._slew_limit_speed(7.0)
            c._speed_cmd_prev = 5.0
        assert out == pytest.approx(5.0 + accel / 50.0), f"{missing}: wrong slew allowance"

        inert = [m for m in warned if "INERT" in m]
        fell_back = [m for m in warned if "slew limit falls back" in m]
        want_inert = 0 if slew_ok is None else 1
        assert len(inert) == want_inert, f"{missing}: {len(inert)} circle warnings, want {want_inert}"
        # b_ax missing means the slew limit falls back even though the circle is fine
        want_fb = 0 if slew_ok is True else 1
        assert len(fell_back) == want_fb, f"{missing}: {len(fell_back)} slew warnings, want {want_fb}"
    print("PASS a missing veh_dyn input disables exactly the consumers that read it, each "
          "warning once and naming what is in force")


def test_the_loader_reads_all_three_shipped_files():
    for version in ("CAR", "SIM"):
        ay, ax, axm, bax, p = ctl.load_veh_dyn(str(REPO / "stack_master" / "config" / version))
        for name, tbl in (("ay", ay), ("ax", ax), ("axm", axm), ("b_ax", bax)):
            assert tbl.ndim == 2 and tbl.shape[1] == 2 and tbl.shape[0] >= 2, (name, tbl.shape)
            assert np.all(np.diff(tbl[:, 0]) > 0), f"{version}/{name}: speed not increasing"
            assert np.all(np.isfinite(tbl[:, 1])) and np.all(tbl[:, 1] > 0)
        assert 1.0 <= p <= 2.0, p
        print(f"  {version}: ay {ay[:, 1].min():.2f}, ax {ax[:, 1].min():.2f}, axm "
              f"{axm[:, 1].min():.2f}-{axm[:, 1].max():.2f}, b_ax {bax[:, 1].min():.2f}-"
              f"{bax[:, 1].max():.2f}, p {p}")
    # and it refuses a directory that is missing any one of them, rather than defaulting.
    # b_ax_max_machines.csv must be in BOTH lists: left out of the copy list it would be absent
    # from every case, and each one would then raise for the wrong file while still passing.
    files = ("veh_dyn_info/ggv.csv", "veh_dyn_info/ax_max_machines.csv",
             "veh_dyn_info/b_ax_max_machines.csv", "racecar_f110.ini")
    for drop in files:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp)
            (dst / "veh_dyn_info").mkdir()
            for rel in files:
                if rel != drop:
                    (dst / rel).write_bytes((CAR / rel).read_bytes())
            with pytest.raises(Exception):
                ctl.load_veh_dyn(str(dst))
    print("PASS the loader reads all four files and refuses a directory missing any of them")


if __name__ == "__main__":
    test_outside_the_circle_there_is_no_acceleration_left()
    test_p2_is_the_ellipse()
    test_the_machine_cap_and_the_interpolation_both_apply()
    test_the_margin_scales_the_lateral_budget()
    test_it_refuses_acceleration_without_commanding_braking()
    test_off_is_the_old_slew_limiter_exactly()
    test_a_missing_input_disables_it_and_says_so_once()
    test_the_loader_reads_all_three_shipped_files()
    print("ALL PASS")
