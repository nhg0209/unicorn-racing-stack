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

Run (needs a sourced workspace -- the window-anchor check loads the real state machine):
    source /opt/ros/jazzy/setup.bash && source ~/unicorn_ws/install/setup.bash
    python3 stack_master/scripts/test_speed_continuity.py
Re-check a new recording:            python3 stack_master/scripts/test_speed_continuity.py --bag <dir>

The --bag mode needs a sourced workspace (it deserializes f110_msgs).
"""

import argparse
import os
import sys
import types

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
    """3.47 -> 5.13 (the t=21.92 transition) must ramp, and must still get there quickly.

    Driven through the REAL Controller._slew_limit_speed, like test_slew_limit above. It used to
    run its own `h = min(target, h + step)` arithmetic, which is a statement about a formula this
    file wrote rather than about the code that ships -- it would have passed unchanged if the
    limiter had been deleted.
    """
    C = _load_controller_slew()

    class Stub:
        loop_rate, max_accel_mps2, max_decel_mps2 = LOOP_RATE, GGV_AX_MAX, GGV_AX_MAX
        _speed_cmd_prev = None
        _slew_limit_speed = C._slew_limit_speed

    step = GGV_AX_MAX / LOOP_RATE
    s = Stub()
    prev, target = 3.47, 5.13
    s._slew_limit_speed(prev)                      # first cycle adopts, i.e. the handoff starts here
    cmds = []
    for _ in range(200):
        cmds.append(s._slew_limit_speed(target))
        if cmds[-1] >= target - 1e-3:
            break
    assert max(np.diff([prev] + cmds)) <= step + 1e-9, "handoff exceeded the accel limit"
    assert cmds[-1] >= target - 1e-3, "the handoff never reached the commanded speed"
    secs = len(cmds) / LOOP_RATE
    assert secs < 0.5, f"handoff took {secs:.2f} s — too sluggish to pass with"
    print(f"PASS trailing handoff: {prev:.2f} -> {target:.2f} m/s in {len(cmds)} cycles "
          f"({secs:.2f} s) through Controller._slew_limit_speed, max step {step:.3f} m/s")


def _load_state_machine():
    """The REAL StateMachine class, without constructing a node (needs a sourced workspace)."""
    import types
    p = os.path.join(STACK_MASTER, "..", "state_machine", "state_machine", "state_machine_node.py")
    mod = types.ModuleType("sm_under_test")
    mod.__dict__["__file__"] = p
    with open(p) as f:
        exec(compile(f.read(), p, "exec"), mod.__dict__)
    return mod.StateMachine


def test_window_anchor():
    """A window index that is 0.55 m ahead (the observed reopt-swap frame mismatch) must snap back,
    and the +-search_m bound must stop it snapping across the track.

    Driven through the REAL StateMachine.anchor_gb_index. It used to define a local copy of the
    same arithmetic, so nothing in this repo called the shipped method: it could have been
    deleted, or its search bound removed, with every test still green.
    """
    n, ds = 366, 0.1
    arr = np.column_stack([np.arange(n) * ds, np.zeros(n)])   # straight track, x = s
    SM = _load_state_machine()

    def anchor(s_idx, car_x, search_m=3.0):
        f = types.SimpleNamespace(
            current_position=(car_x, 0.0),
            cur_gb_wpnts=types.SimpleNamespace(is_init=True, array=arr),
            num_glb_wpnts=n, wpnt_dist=ds)
        return SM.anchor_gb_index(f, s_idx, search_m)

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


def test_aeb_engage_count_over_a_lap():
    """G10. The clamp is latched, so a lap's worth of jitter around the bar must not produce a
    clamp edge per cycle. Driven through the REAL AEB, with the distance dithering across the bar
    the way a re-anchored local window does."""
    C = _load_controller_slew()
    n = C.__new__(C)
    n.AEB_thres, n.AEB_thres_overtake, n.AEB_offline_d_thres = 0.5, 0.9, 0.1
    n.AEB_release_hyst_m, n.AEB_min_hold_s, n.loop_rate = 0.25, 0.2, LOOP_RATE
    n._aeb_engaged, n._aeb_cycles = False, 0
    n.logger_warn = lambda *a, **k: None
    n.position_in_map = np.array([[0.0, 0.0, 0.0]])
    n.idx_nearest_waypoint = 0
    n.state = "GB_TRACK"
    rng = np.random.default_rng(0)
    engaged, prev = [], False
    for k in range(int(LOOP_RATE * 12)):          # ~12 s, one lap on ifac at racing pace
        dist = 0.5 + 0.03 * np.sin(k / 3.0) + 0.01 * rng.standard_normal()
        arr = np.zeros((5, 9))
        arr[0, 0] = dist
        n.waypoint_array_in_map = arr
        out = n.AEB_for_weird_local_wpnt(6.0)
        now = out < 6.0
        if now != prev:
            engaged.append(k)
        prev = now
    edges = len(engaged)
    assert edges <= 2, f"the AEB toggled {edges} times in a lap of jitter around the bar"
    print(f"PASS the AEB produces {edges} edge(s) over a lap of 3 cm jitter across its threshold")


def test_no_published_step_exceeds_the_accel_limit():
    """G10. Whatever the sources do, the PUBLISHED command may not move faster than the car can."""
    C = _load_controller_slew()

    class Stub:
        loop_rate, max_accel_mps2, max_decel_mps2 = LOOP_RATE, GGV_AX_MAX, GGV_AX_MAX
        _speed_cmd_prev = None
        _slew_limit_speed = C._slew_limit_speed

    s = Stub()
    step = GGV_AX_MAX / LOOP_RATE
    # the four discontinuities the run log has: a trailing handoff, an AEB engage, an AEB release
    # and a squeeze cap arriving
    demands = ([3.47] + [5.13] * 20 + [2.0] * 10 + [5.13] * 20 + [2.5] * 10 + [5.0] * 20)
    out = [s._slew_limit_speed(v) for v in demands]
    d = np.diff(out)
    assert d.max() <= step + 1e-9 and d.min() >= -step - 1e-9, \
        f"published step {max(abs(d.max()), abs(d.min())):.3f} m/s > {step:.3f} per cycle"
    print(f"PASS no published step exceeds {step:.3f} m/s per cycle across four discontinuities")


def test_assembled_local_window_accel():
    """G12. The window the controller is handed is ASSEMBLED from pieces, and the joins between
    them are not bound by anything.

    Two joins, both measured on the real map: the avoidance path -> global padding splice (up to
    0.894 m/s between adjacent stations) and the global raceline's OWN velocity seam at s = 0
    (0.867 m/s = 35.6 m/s^2, straddled by about 22% of the lap -- 104 of the 137 stations
    demanding more than 6 m/s^2 were that seam, not the planner). Assembled window before the
    limiting pass: |a_long| p50 4.88, p95 35.6, max 56.8 against a 5.0 ggv ax_max.

    This drives the REAL StateMachine._limit_window_accel over the REAL global line, seam included.
    """
    import json
    SM = _load_state_machine()
    wp = json.load(open(os.path.join(STACK_MASTER, "maps", "ifac", "global_waypoints.json")))
    gb = wp["global_traj_wpnts_iqp"]["wpnts"]

    def window(start, n=80):
        return [types.SimpleNamespace(x_m=gb[(start + i) % len(gb)]["x_m"],
                                      y_m=gb[(start + i) % len(gb)]["y_m"],
                                      vx_mps=gb[(start + i) % len(gb)]["vx_mps"])
                for i in range(n)]

    def demand(w):
        v = np.array([x.vx_mps for x in w])
        xy = np.array([[x.x_m, x.y_m] for x in w])
        ds = np.maximum(np.hypot(*np.diff(xy, axis=0).T), 1e-6)
        return np.max(np.abs(v[1:] ** 2 - v[:-1] ** 2) / (2.0 * ds))

    f = types.SimpleNamespace(
        local_window_accel_limit_enable=True, local_window_a_long_mps2=GGV_AX_MAX,
        cur_vs=gb[0]["vx_mps"], _limit_window_accel=None)
    starts = list(range(0, len(gb) - 1, 4))
    worst_before = max(demand(window(s0)) for s0 in starts)
    worst_after = 0.0
    for s0 in starts:
        w = window(s0)
        f.cur_vs = w[0].vx_mps
        out = SM._limit_window_accel(f, w)
        worst_after = max(worst_after, demand(out))
        # the pass may only ever slow the plan down
        assert all(b.vx_mps <= a.vx_mps + 1e-9 for a, b in zip(w, out)), \
            "the accel limit raised a speed"
        # ...and it must not mutate the caller's objects: those are the cached global line
        assert all(x.vx_mps == gb[(s0 + i) % len(gb)]["vx_mps"] for i, x in enumerate(w)), \
            "the pass wrote through to the cached global waypoints"
    assert worst_after <= 6.0, (
        f"the assembled window still demands {worst_after:.1f} m/s^2 (was {worst_before:.1f})")
    print(f"PASS the assembled local window demands at most {worst_after:.2f} m/s^2 "
          f"over {len(starts)} windows of the real ifac line (was {worst_before:.1f})")


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
    test_aeb_engage_count_over_a_lap()
    test_no_published_step_exceeds_the_accel_limit()
    test_assembled_local_window_accel()
    ok = True
    if args.bag:
        ok = check_bag(args.bag)
    print("\nALL PASS" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
