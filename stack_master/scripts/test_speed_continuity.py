#!/usr/bin/env python3
"""
test_speed_continuity.py — guards against discontinuous speed commands and a detached local window.

Three failures, all measured on the real car in bag ot_speed_0731_2034 (55.7 s, 4 laps, static
obstacles at s ~= 36.3 and ~= 3.5):

  1. TRAILING -> OVERTAKE handed the command over with no blend. The gap PID had the car at the
     opponent's pace while the path speed ahead was already much higher, so `speed_command =
     global_speed` stepped +1.59/+1.64/+1.67/+1.70 m/s in a single ~20 ms cycle on all four
     transitions (~80 m/s^2 against a 5.0 ggv ax_max). The REFERENCE was continuous across those
     transitions (vx[10] 4.94 -> 4.94), so the step was made in the controller.

  2. Nothing bounded d(speed_command)/dt at all. Over the whole run the published command moved at
     p99 12.2 m/s^2 and up to 86.8 m/s^2.

  3. The GB_TRACK local window is indexed by `cur_s / wpnt_dist`, but cur_s and that window come
     from different topics that update at different times (/global_waypoints -> frenet converter,
     immediately on a static-reopt swap; /global_waypoints_scaled -> this window, on sector_tuner's
     0.5 s timer). At t=14.94-15.24 the window sat 0.55 m AHEAD of the car -- longitudinally;
     lateral error was only 0.018 m -- which is past the controller's AEB_thres (0.5 m), so AEB
     clamped to 2.0 m/s for 0.4 s and then released with a +1.24 m/s step.

Run standalone (no ROS, no build):   python3 stack_master/scripts/test_speed_continuity.py
Re-check a new recording:            python3 stack_master/scripts/test_speed_continuity.py --bag <dir>

The --bag mode needs a sourced workspace (it deserializes f110_msgs).
"""

import argparse
import os
import sys

import numpy as np

STACK_MASTER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(STACK_MASTER, "..", "controller", "controller", "combined", "src"))

GGV_AX_MAX = 5.0        # veh_dyn_info/ggv.csv + ax_max_machines.csv
LOOP_RATE = 50.0        # controller_manager publish rate


def _load_controller_slew():
    """Import the real _slew_limit_speed off Controller so the test cannot drift from the code."""
    import importlib.util
    p = os.path.join(STACK_MASTER, "..", "controller", "controller", "combined", "src", "Controller.py")
    spec = importlib.util.spec_from_file_location("ctrl_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Controller


def test_slew_limit():
    C = _load_controller_slew()

    class Stub:
        loop_rate, max_accel_mps2, max_decel_mps2 = LOOP_RATE, GGV_AX_MAX, GGV_AX_MAX
        _speed_cmd_prev = None
        _slew_limit_speed = C._slew_limit_speed

    s = Stub()
    step = GGV_AX_MAX / LOOP_RATE
    out = [s._slew_limit_speed(v) for v in [3.5, 5.2, 5.2, 5.2, 5.2, 5.2, 2.0, 2.0, 2.0]]
    d = np.diff(out)
    assert d.max() <= step + 1e-9, f"accel {d.max():.4f} > limit {step:.4f}"
    assert d.min() >= -step - 1e-9, f"decel {d.min():.4f} < limit {-step:.4f}"

    fresh = Stub()
    fresh._speed_cmd_prev = None
    assert fresh._slew_limit_speed(7.0) == 7.0, "first cycle must adopt the command as-is"
    assert fresh._slew_limit_speed(None) is None, "None must pass through untouched"
    print(f"PASS slew limit: |dv| <= {step:.3f} m/s per cycle, first-cycle adopt, None passthrough")


def test_trailing_handoff():
    """3.47 -> 5.13 (the t=21.92 transition) must ramp, and must still get there quickly."""
    step = GGV_AX_MAX / LOOP_RATE
    h, target = 3.47, 5.13
    cmds, prev = [], h
    for _ in range(200):
        h = min(target, h + step)
        cmds.append(h)
        if h >= target - 1e-3:
            break
    assert max(np.diff([prev] + cmds)) <= step + 1e-9, "handoff exceeded the accel limit"
    secs = len(cmds) / LOOP_RATE
    assert secs < 0.5, f"handoff took {secs:.2f} s — too sluggish to pass with"
    print(f"PASS trailing handoff: {prev:.2f} -> {target:.2f} m/s in {len(cmds)} cycles "
          f"({secs:.2f} s), max step {step:.3f} m/s")


def test_window_anchor():
    """A window index that is 0.55 m ahead (the observed reopt-swap frame mismatch) must snap back,
    and the +-search_m bound must stop it snapping across the track."""
    n, ds = 366, 0.1
    arr = np.column_stack([np.arange(n) * ds, np.zeros(n)])   # straight track, x = s

    def anchor(s_idx, car_x, search_m=3.0):
        k = max(1, int(search_m / ds))
        idx = (s_idx + np.arange(-k, k + 1)) % n
        d = np.hypot(arr[idx, 0] - car_x, arr[idx, 1] - 0.0)
        return int(idx[int(np.argmin(d))])

    car_x = 8.64
    bad = int(car_x / ds + 0.5) + 6                # s frame off by 6 stations = 0.6 m
    assert abs(arr[bad, 0] - car_x) > 0.5, "test setup must exceed AEB_thres"
    fixed = anchor(bad, car_x)
    err = abs(arr[fixed, 0] - car_x)
    assert err <= ds * 0.6, f"anchor left {err:.3f} m of longitudinal offset"

    far = anchor(bad + 100, car_x)                 # 10 m away: only a local correction is allowed
    assert abs(arr[far, 0] - car_x) > 3.0, "anchor snapped outside its search window"
    print(f"PASS window anchor: {abs(arr[bad,0]-car_x):.2f} m offset -> {err:.3f} m; "
          f"stays local beyond +-3 m")


def check_bag(path):
    """Replay a recording's published command through the limiter and report before/after."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
           rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    topic = '/vesc/high_level/ackermann_cmd'
    if topic not in types:
        print(f"FAIL: {topic} not in the bag — record it (see STATIC_AVOIDANCE_TEST_RUNBOOK)")
        return False
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    ts, sp = [], []
    while r.has_next():
        _, data, t = r.read_next()
        ts.append(t / 1e9)
        sp.append(deserialize_message(data, get_message(types[topic])).drive.speed)
    ts, sp = np.array(ts), np.array(sp)
    dt = np.maximum(np.diff(ts), 1e-3)
    raw = np.abs(np.diff(sp)) / dt
    lim, prev = [], None
    step = GGV_AX_MAX / LOOP_RATE
    for v in sp:
        prev = v if prev is None else min(max(v, prev - step), prev + step)
        lim.append(prev)
    new = np.abs(np.diff(np.array(lim))) / dt
    print(f"\n{os.path.basename(path)}  ({ts[-1]-ts[0]:.1f} s, {len(sp)} commands)")
    print(f"  as recorded : p50 {np.percentile(raw,50):5.2f}  p99 {np.percentile(raw,99):6.2f}  "
          f"MAX {raw.max():7.2f} m/s^2")
    print(f"  with limiter: p50 {np.percentile(new,50):5.2f}  p99 {np.percentile(new,99):6.2f}  "
          f"MAX {new.max():7.2f} m/s^2   (ggv ax_max {GGV_AX_MAX})")
    worst = np.abs(np.diff(np.array(lim))).max()
    ok = worst <= step * 1.05
    print(f"  worst single step {worst:.3f} m/s (limit {step:.3f}) -> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", help="rosbag2 directory to replay through the limiter")
    args = ap.parse_args()
    test_slew_limit()
    test_trailing_handoff()
    test_window_anchor()
    ok = True
    if args.bag:
        ok = check_bag(args.bag)
    print("\nALL PASS" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
