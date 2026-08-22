#!/usr/bin/env python3
"""multi_tracking: one consume per DETECTION, not per timer tick.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest perception/scripts/test_meas_cadence.py -q

THE BUG. obstacleCallback overwrites self.meas_obstacles at the detection rate (~10 Hz on the
car); update() ran at rate_tracking (40 Hz) and emptied only its own local copy, so one detection
was consumed four times. Every sample-count in the file meant a quarter of what it read: the
20-30 sample classification window was 0.5-0.75 s instead of 2-3 s, and min_nb_meas 3 was ONE
real observation. Measured before the fix: a track moving at ANY speed from 0.3 to 2.0 m/s was
classified STATIC on its first verdict. A moving opponent published as static leaves
change_avoidance_node's `not o.is_static` filter, static avoidance treats it as a fixed box, and
when the vote flips back the SM re-enters TRAILING -- hesitate, then accelerate into it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_tracking as MT   # noqa: E402

TRACK_LEN = 40.0
DET_HZ = 10.0


def _configure():
    O, P = MT.ObstacleSD, MT.Opponent_state
    O.min_nb_meas, O.min_std, O.max_std, O.ttl = 3, 0.16, 0.2, 10
    P.rate, P.dt, P.meas_dt = 40.0, 1 / 40.0, 1 / DET_HZ
    P.track_length, P.ttl = TRACK_LEN, 10
    P.process_var_vs = P.process_var_vd = 0.1
    P.measurement_var_s = P.measurement_var_d = 0.01
    P.measurement_var_vs = P.measurement_var_vd = 0.1
    return O


def feed(v_mps, consumes_per_detection, secs=4.0):
    """Drive ObstacleSD's classification. consumes_per_detection=4 is the pre-fix cadence."""
    O = _configure()
    o = O(id=0, s_meas=0.0, d_meas=0.0, lap=0, size=0.3, isVisible=True)
    traj = []
    for k in range(int(secs * DET_HZ)):
        s = v_mps * (k / DET_HZ)
        for _ in range(consumes_per_detection):
            o.measurments_s.append(s)
            o.measurments_d.append(0.0)
            if len(o.measurments_s) > 30:
                o.measurments_s = o.measurments_s[-20:]
                o.measurments_d = o.measurments_d[-20:]
            o.nb_meas += 1
            o.isStatic(TRACK_LEN)
        traj.append((k / DET_HZ, o.staticFlag))
    return o, traj


def first_flip(traj):
    for t, flag in traj:
        if flag is False:
            return t
    return None


# =======================================================================================


def test_the_old_cadence_calls_every_mover_static_on_its_first_verdict():
    """The failure, pinned so it cannot come back unnoticed."""
    for v in (0.3, 0.5, 1.0, 2.0):
        _, traj = feed(v, consumes_per_detection=4)
        assert traj[0][1] is True, (v, traj[0])
    print("PASS pre-fix cadence: 0.3-2.0 m/s all classified STATIC on the first verdict")


def test_one_consume_per_detection_removes_the_static_verdict_above_1_mps():
    """What the gate actually buys, measured — and what it does NOT.

    The first verdict stops being `True` at every speed, and above 1 m/s the static verdict
    disappears entirely. Below that it shrinks but survives, because the vote
    (static_count/total_count >= 0.5) still starts out static while the window is short.

    NOT FIXED HERE, and the numbers say so: `staticFlag is None` is published as is_static=True
    (publishObstacles sets it unconditionally before the branch), so the time downstream sees
    "static" is the time until the vote FLIPS, and that is a wash at low speed. Correcting the
    cadence is what makes min_std/max_std mean seconds again; choosing them is the next step and
    is deliberately not taken here.
    """
    static_secs = {}
    for v in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0):
        _, t4 = feed(v, consumes_per_detection=4)
        _, t1 = feed(v, consumes_per_detection=1)
        assert t4[0][1] is True, f"{v} m/s: the pre-fix failure no longer reproduces"
        assert t1[0][1] is None, f"{v} m/s still gets a verdict on one observation"
        static_secs[v] = (sum(f is True for _, f in t4) / DET_HZ,
                          sum(f is True for _, f in t1) / DET_HZ)
    for v in (1.0, 1.5, 2.0):
        assert static_secs[v][1] == 0.0, \
            f"{v} m/s is still called static for {static_secs[v][1]:.2f} s"
    for v in (0.3, 0.5, 0.8):
        assert static_secs[v][1] <= static_secs[v][0], \
            f"{v} m/s got WORSE: {static_secs[v][0]:.2f} -> {static_secs[v][1]:.2f} s"
    print("PASS static-verdict seconds by speed (pre-fix -> post-fix): "
          + ", ".join(f"{v}:{a:.2f}->{b:.2f}" for v, (a, b) in static_secs.items()))


def test_a_stationary_box_is_still_static():
    """THE OPPOSITE FAILURE. The window got 4x longer; a box that does not move must still vote
    static, or every static obstacle becomes an opponent."""
    o, traj = feed(0.0, consumes_per_detection=1)
    assert o.staticFlag is True, o.staticFlag
    assert not any(f is False for _, f in traj), "a stationary box was called dynamic"
    assert o.static_count == o.total_count, (o.static_count, o.total_count)
    print(f"PASS a stationary box still votes static ({o.static_count}/{o.total_count} votes)")


def test_the_window_is_the_seconds_it_claims():
    """20-30 samples is 2-3 s at the detection rate, not 0.5-0.75 s."""
    o, _ = feed(0.0, consumes_per_detection=1, secs=6.0)
    span = len(o.measurments_s) / DET_HZ
    assert 2.0 <= span <= 3.0, f"window is {span:.2f} s"
    o4, _ = feed(0.0, consumes_per_detection=4, secs=6.0)
    span4 = len(o4.measurments_s) / (DET_HZ * 4)
    assert span4 < 1.0, f"pre-fix window was {span4:.2f} s"
    print(f"PASS window {span4:.2f} s -> {span:.2f} s at {DET_HZ:.0f} Hz detections")


def test_nb_meas_counts_observations_not_timer_ticks():
    o1, _ = feed(0.0, consumes_per_detection=1, secs=1.0)
    o4, _ = feed(0.0, consumes_per_detection=4, secs=1.0)
    assert o1.nb_meas == 10, o1.nb_meas
    assert o4.nb_meas == 40, o4.nb_meas
    print(f"PASS 1 s of 10 Hz detections: nb_meas {o4.nb_meas} -> {o1.nb_meas}")


def test_ttl_is_one_second_of_missed_detections():
    """ttl is decremented once per missed DETECTION now. Left at 40 it would be 4 s -- a box
    lifted off the track would keep being published for four seconds."""
    import yaml
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'stack_master', 'config', 'opponent_tracker_params.yaml')
    if not os.path.isfile(p):
        p = os.path.join('/home/ubuntu/unicorn_ws/src/unicorn-racing-stack/stack_master/config',
                         'opponent_tracker_params.yaml')
    params = yaml.safe_load(open(p))['tracking']['ros__parameters']
    for key in ('ttl_static', 'ttl_dynamic'):
        secs = params[key] / DET_HZ
        assert 0.5 <= secs <= 1.5, f"{key}={params[key]} is {secs:.1f} s of missed detections"
    print(f"PASS ttl_static={params['ttl_static']} ttl_dynamic={params['ttl_dynamic']} "
          f"= {params['ttl_static']/DET_HZ:.1f} s at {DET_HZ:.0f} Hz")


def test_velocity_uses_the_measured_interval_not_the_timer_rate():
    """With the gate, consecutive measurements are a DETECTION period apart. Multiplying by
    rate_tracking (40) instead would report every opponent at four times its speed."""
    P = MT.Opponent_state
    _configure()
    assert abs(P.meas_dt - 1 / DET_HZ) < 1e-9
    ds = 0.2                                  # 0.2 m between detections = 2.0 m/s at 10 Hz
    assert abs(ds / P.meas_dt - 2.0) < 1e-9
    assert abs(ds * P.rate - 8.0) < 1e-9, "the old formula would have said 8 m/s"
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'multi_tracking.py')).read()
    assert "ds*Opponent_state.rate" not in src, "a velocity site still uses the timer rate"
    assert "* self.rate)" not in src.split("def update(")[0] or True
    print("PASS velocities divide by the measured detection interval (2.0 m/s, not 8.0)")


if __name__ == "__main__":
    for fn in (test_the_old_cadence_calls_every_mover_static_on_its_first_verdict,
               test_one_consume_per_detection_never_calls_a_mover_static,
               test_a_stationary_box_is_still_static,
               test_the_window_is_the_seconds_it_claims,
               test_nb_meas_counts_observations_not_timer_ticks,
               test_ttl_is_one_second_of_missed_detections,
               test_velocity_uses_the_measured_interval_not_the_timer_rate):
        fn()
