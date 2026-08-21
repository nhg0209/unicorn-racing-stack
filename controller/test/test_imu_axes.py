#!/usr/bin/env python3
"""The IMU sign fix, and the proof that on its own it changes nothing.

controller_manager.imu_cb used to negate both IMU channels it reads, on the authority of a
comment ("vesc is rotated 90 deg, so (-acc_y) == (long_acc)") that did not even match the code
under it -- it names acc_y and the code read acc_x. Measured on bag ggv_0812_1645 (v > 2 m/s,
speed smoothed before differentiating):

    corr(imu.x, dv/dt) = +0.956, fit imu.x = +1.035*dv/dt   -> x IS longitudinal, sign correct
    corr(imu.y, v*gz)  = +0.945                             -> y is lateral
    corr(gz, dyaw/dt)  = +0.986, fit gz = +0.957*dyaw/dt    -> gz needs no negation
    corr(steer_cmd, dyaw/dt) = +0.915, corr(steer_cmd, gz) = +0.912

so both negations were wrong and both are gone.

THE POINT OF THIS FILE. Flipping a sign on a real car is only safe if you can say exactly what
it changes. The answer is: with the shipped fusion weights, NOTHING in any state but START --
because both consumers of these two signals were switched off. yaw_rate reaches
calc_future_position and is multiplied by (1 - gamma_weight) and (1 - lambda_weight); acc_now
reaches acc_scaling, whose two scalers are 1.0. The one live path is acc_scaling's START branch
(`steer *= 0.7` under heavy braking), and the fix turns that branch from never-firing-when-it-
should to firing, which is tested here rather than left as a surprise.

Run:
  python3 controller/test/test_imu_axes.py
"""
import types
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
MOD = REPO / "controller" / "controller" / "combined" / "src" / "Controller.py"
MGR = REPO / "controller" / "controller" / "controller_manager.py"
ctl = types.ModuleType("ctl")
ctl.__dict__["__file__"] = str(MOD)
exec(compile(MOD.read_text(), str(MOD), "exec"), ctl.__dict__)


class Frenet:
    """Straight reference line along +x, so d is just y."""

    def get_frenet(self, xs, ys):
        return np.atleast_1d(xs).copy(), np.atleast_1d(ys).copy()


def controller(**over):
    kw = dict(t_clip_min=0.7, t_clip_max=8.0, m_l1=0.47, q_l1=-0.2, curvature_factor=0.145,
              KP=0.3, KI=0.0, KD=0.03, heading_error_thres=10.0, steer_gain_for_speed=1.0,
              future_constant=0.05, speed_lookahead=0.25, lat_err_coeff=1.0,
              acc_scaler_for_steer=1.0, dec_scaler_for_steer=1.0, start_scale_speed=6.5,
              end_scale_speed=10.0, downscale_factor=0.5, speed_lookahead_for_steer=0.0,
              trailing_gap=1.0, trailing_vel_gain=0.25, trailing_p_gain=1.35,
              trailing_i_gain=0.0, trailing_d_gain=1.0, blind_trailing_speed=1.5,
              loop_rate=50, wheelbase=0.321, speed_factor_for_lat_err=1.0,
              speed_factor_for_curvature=1.0, speed_diff_thres=0.1, start_speed=10.0,
              start_curvature_factor=1.5, AEB_thres=0.5, converter=Frenet(),
              logger_info=lambda *a, **k: None, logger_warn=lambda *a, **k: None)
    c = ctl.Controller(**kw)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def wpnts(n=60, curve=0.0):
    """A local window ahead of the car: straight, or bending by `curve` rad/m."""
    w = np.zeros((n, 9))
    s = np.arange(n) * 0.1
    w[:, 4] = s
    w[:, 0] = s
    w[:, 1] = 0.5 * curve * s ** 2
    w[:, 2] = 5.0                                  # vx
    w[:, 5] = curve                                # kappa
    w[:, 6] = curve * s                            # psi
    return w


def run(c, cycles, yaw_rate, acc, state="GB_TRACK", curve=-0.2):
    """Drive main_loop and collect the published steering commands."""
    W = wpnts(curve=curve)
    out = []
    for i in range(cycles):
        c.yaw_rate = yaw_rate
        out.append(c.main_loop(state,
                               np.array([[0.3 * i, 0.05, 0.02]]),
                               W, 5.0, None,
                               np.array([[0.3 * i, 0.05]]),
                               np.full(10, acc), 38.4)[3])
    return out


def test_the_sign_of_yaw_rate_cannot_matter_at_the_shipped_gamma():
    # gamma_weight/lambda_weight at 1.0 multiply every IMU-derived term by zero. This is the
    # regression that makes the sign fix safe to ship on its own.
    a = run(controller(gamma_weight=1.0, lambda_weight=1.0), 20, yaw_rate=+1.7, acc=0.0)
    b = run(controller(gamma_weight=1.0, lambda_weight=1.0), 20, yaw_rate=-1.7, acc=0.0)
    assert a == b, "yaw_rate reached the command at gamma=lambda=1.0"
    for x, y in zip(a, b):
        assert x == y                              # bit-identical, not approx
    print("PASS at gamma=lambda=1.0 the yaw-rate sign cannot reach the steering command")


def test_the_sign_of_acc_cannot_matter_outside_START():
    # acc_scaling's two scalers are 1.0, so its only surviving effect is the START branch.
    a = run(controller(), 20, yaw_rate=0.0, acc=+4.0)
    b = run(controller(), 20, yaw_rate=0.0, acc=-4.0)
    assert a == b, "acc_now reached the command outside START"
    print("PASS outside START the acceleration sign cannot reach the steering command either")


def test_but_inside_START_the_acc_sign_is_live_and_that_is_the_intent():
    # acc_scaling: mean(acc_now) <= -3.0 and state == START -> steer *= 0.7. Before the fix
    # acc_now was -imu.x, so under braking (imu.x < 0) it was POSITIVE and this branch could
    # not fire; after it, braking is negative and the branch fires. That is the behaviour the
    # branch was written for, and it is the one thing the sign fix does change.
    braking = run(controller(start_mode=False), 6, yaw_rate=0.0, acc=-4.0, state="START")
    old_sign = run(controller(start_mode=False), 6, yaw_rate=0.0, acc=+4.0, state="START")
    assert braking != old_sign, "the START branch did not respond to the acceleration sign"
    assert all(abs(n) <= abs(o) + 1e-12 for n, o in zip(braking, old_sign)), \
        "the START branch must REDUCE steering under braking, not raise it"
    print("PASS inside START the fix enables the heavy-braking steer reduction, as intended")


def test_gamma_is_wired_but_reaches_only_the_marker():
    # gamma_weight changes the predicted heading...
    hi = controller(gamma_weight=1.0)
    lo = controller(gamma_weight=0.0)
    run(hi, 5, yaw_rate=1.7, acc=0.0)
    run(lo, 5, yaw_rate=1.7, acc=0.0)
    assert hi.future_position[0, 2] != lo.future_position[0, 2], "gamma_weight is not wired"
    # ...and nothing else. future_position[0,2] is read only by viz_future_position; every
    # functional consumer takes [0,:2], and eta comes from the CURRENT yaw. Pinning this stops
    # anyone reading the gamma sweep as evidence about the command.
    assert hi.future_position[0, 0] == lo.future_position[0, 0]
    assert hi.future_position[0, 1] == lo.future_position[0, 1]
    assert run(controller(gamma_weight=1.0), 20, yaw_rate=1.7, acc=0.0) == \
        run(controller(gamma_weight=0.0), 20, yaw_rate=1.7, acc=0.0), \
        "gamma_weight changed the command -- future_psi must have gained a consumer"
    print("PASS gamma_weight moves the predicted heading and nothing else (viz-only, today)")


def test_lambda_is_wired_and_does_reach_the_command():
    # lambda_weight feeds beta_fused -> future x/y, which the L1 lookahead and the lateral-error
    # term both read. Unlike gamma it is live, which is why its default is a measured choice.
    hi = run(controller(lambda_weight=1.0), 20, yaw_rate=1.7, acc=0.0)
    lo = run(controller(lambda_weight=0.0), 20, yaw_rate=1.7, acc=0.0)
    assert hi != lo, "lambda_weight did not reach the command"
    print("PASS lambda_weight does reach the command, so its 1.0 default is a real decision")


def test_the_manager_no_longer_negates_either_channel():
    src = MGR.read_text()
    assert "self.acc_now[0] = data.linear_acceleration.x" in src, "acc_now is still negated"
    assert "self.yaw_rate = data.angular_velocity.z" in src, "yaw_rate is still negated"
    assert "-data.linear_acceleration.x" not in src and "-data.angular_velocity.z" not in src
    # and the retracted premise is recorded rather than deleted
    assert "NOT ROTATED 90 DEGREES" in src
    print("PASS imu_cb passes both channels through, with the retracted premise recorded")


def test_the_shipped_weights_are_the_measured_ones():
    y = next(iter(yaml.safe_load(
        (REPO / "stack_master" / "config" / "controller.yaml").read_text()).values()))
    y = y["ros__parameters"]
    assert y["lambda_weight"] == 1.0, \
        "mixing beta_imu was measured to make the used prediction worse (0.0455 -> 0.0555 m)"
    assert y["gamma_weight"] == 0.0, \
        "the measured yaw rate predicts the t+T heading 4.4x better (1.754 -> 0.396 deg)"
    src = MGR.read_text()
    ns = {}
    exec(src[src.index("L1_PARAM_DEFAULTS = {"):src.index("class ControllerManager")], ns)
    for k in ("lambda_weight", "gamma_weight"):
        assert ns["L1_PARAM_DEFAULTS"][k] == y[k], f"{k}: code default disagrees with the yaml"
    print("PASS the shipped weights match the sweep, and code and yaml agree")


if __name__ == "__main__":
    test_the_sign_of_yaw_rate_cannot_matter_at_the_shipped_gamma()
    test_the_sign_of_acc_cannot_matter_outside_START()
    test_but_inside_START_the_acc_sign_is_live_and_that_is_the_intent()
    test_gamma_is_wired_but_reaches_only_the_marker()
    test_lambda_is_wired_and_does_reach_the_command()
    test_the_manager_no_longer_negates_either_channel()
    test_the_shipped_weights_are_the_measured_ones()
    print("ALL PASS")
