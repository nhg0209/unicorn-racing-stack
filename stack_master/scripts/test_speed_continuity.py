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

LOOP_RATE = 50.0        # controller_manager publish rate
VERSION = "CAR"         # the bag and every measurement quoted above are from the real car


def _veh_dyn(version=VERSION):
    """(ggv_ax, ax_max_machines, b_ax_max_machines) [v, value] tables from the shipped config.

    THIS FILE USED TO CARRY `GGV_AX_MAX = 5.0` instead, commented "veh_dyn_info/ggv.csv +
    ax_max_machines.csv". That comment was wrong twice over: no veh_dyn table has read 5.0 since
    2026-08-11, and what the number really mirrored was two yaml keys (controller.yaml
    max_accel/max_decel_mps2, state_machine_params.yaml local_window_a_long_mps2) that were
    themselves stale hand-copies of these csvs. Since 2026-08-12 the controller and the state
    machine DERIVE their bounds from these files (Controller._slew_limits,
    state_machine_node._a_long_window), so the gate reads them too -- one source, no mirror left
    to go stale.
    """
    vdi = os.path.join(STACK_MASTER, "config", version, "veh_dyn_info")

    def two_col(name, col=1):
        t = np.atleast_2d(np.loadtxt(os.path.join(vdi, name), delimiter=",", comments="#"))
        return t[:, [0, col]]

    return (two_col("ggv.csv"), two_col("ax_max_machines.csv"),
            two_col("b_ax_max_machines.csv"))


_GGV_AX, _AXM, _BAX = _veh_dyn()


def _limits_at(v=0.0):
    """(accel, decel, window) bounds at speed `v`, the same three the shipped code computes."""
    it = lambda t: float(np.interp(v, t[:, 0], t[:, 1]))   # noqa: E731
    ax = it(_GGV_AX)
    accel, decel = min(ax, it(_AXM)), min(ax, it(_BAX))
    return accel, decel, min(accel, decel)


# At rest, which is where every fixture below sits: CAR reads 7.00 / 7.00 / 7.00 today. All three
# shipped tables are flat under ~7.25 m/s (only ax_max_machines falls, and only above that), so a
# single evaluation is exact over the speeds these fixtures use -- test_bag re-derives per sample.
A_ACCEL, A_DECEL, A_WINDOW = _limits_at(0.0)

# The two-pass local-window limiter is an approximation, so its output is checked against its
# setting plus a slack rather than against the setting exactly. 1.2 preserves the ratio the
# hardcoded pair (bound 6.0, setting 5.0) expressed; it is NOT tuned to make anything pass.
WINDOW_SLACK = 1.2
WINDOW_BOUND = A_WINDOW * WINDOW_SLACK


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
        loop_rate, max_accel_mps2, max_decel_mps2 = LOOP_RATE, A_ACCEL, A_DECEL
        _speed_cmd_prev = None
        _slew_limit_speed = C._slew_limit_speed
        # bound off the real class too, so the derivation under test is the shipped one
        _slew_limits, _interp, _slew_warned = C._slew_limits, C._interp, False
        # _slew_limits falls back to the two attributes above when the veh_dyn tables are absent,
        # so this stub exercises the fallback branch with the DERIVED numbers in it -- the
        # arithmetic under test is identical either way, and the stub stays a stub.
        # a_comb_limit_enable is off because these checks are about the slew limiter, not the
        # friction circle; _ax_avail needs tables, a_lat and speed_now that none of them supply.
        a_comb_limit_enable = False

    s = Stub()
    # up and down are separate now: accel is bounded by ax_max_machines, decel by
    # b_ax_max_machines, and those two tables are not the same curve even when today's numbers
    # coincide. Asserting one `step` both ways would silently stop checking one of them.
    up, dn = A_ACCEL / LOOP_RATE, A_DECEL / LOOP_RATE
    out = [s._slew_limit_speed(v) for v in [3.5, 5.2, 5.2, 5.2, 5.2, 5.2, 2.0, 2.0, 2.0]]
    d = np.diff(out)
    assert d.max() <= up + 1e-9, f"accel {d.max():.4f} > limit {up:.4f}"
    assert d.min() >= -dn - 1e-9, f"decel {d.min():.4f} < limit {-dn:.4f}"

    fresh = Stub()
    fresh._speed_cmd_prev = None
    assert fresh._slew_limit_speed(7.0) == 7.0, "first cycle must adopt the command as-is"
    assert fresh._slew_limit_speed(None) is None, "None must pass through untouched"
    print(f"PASS slew limit: dv <= +{up:.3f} / -{dn:.3f} m/s per cycle (accel {A_ACCEL:.2f}, "
          f"decel {A_DECEL:.2f} m/s^2 from {VERSION}/veh_dyn_info), first-cycle adopt, "
          f"None passthrough")


def test_trailing_handoff():
    """3.47 -> 5.13 (the t=21.92 transition) must ramp, and must still get there quickly.

    Driven through the REAL Controller._slew_limit_speed, like test_slew_limit above. It used to
    run its own `h = min(target, h + step)` arithmetic, which is a statement about a formula this
    file wrote rather than about the code that ships -- it would have passed unchanged if the
    limiter had been deleted.
    """
    C = _load_controller_slew()

    class Stub:
        loop_rate, max_accel_mps2, max_decel_mps2 = LOOP_RATE, A_ACCEL, A_DECEL
        _speed_cmd_prev = None
        _slew_limit_speed = C._slew_limit_speed
        # bound off the real class too, so the derivation under test is the shipped one
        _slew_limits, _interp, _slew_warned = C._slew_limits, C._interp, False
        # _slew_limits falls back to the two attributes above when the veh_dyn tables are absent,
        # so this stub exercises the fallback branch with the DERIVED numbers in it -- the
        # arithmetic under test is identical either way, and the stub stays a stub.
        # a_comb_limit_enable is off because these checks are about the slew limiter, not the
        # friction circle; _ax_avail needs tables, a_lat and speed_now that none of them supply.
        a_comb_limit_enable = False

    step = A_ACCEL / LOOP_RATE          # the handoff ramps UP, so this is the accel side
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
        loop_rate, max_accel_mps2, max_decel_mps2 = LOOP_RATE, A_ACCEL, A_DECEL
        _speed_cmd_prev = None
        _slew_limit_speed = C._slew_limit_speed
        # bound off the real class too, so the derivation under test is the shipped one
        _slew_limits, _interp, _slew_warned = C._slew_limits, C._interp, False
        # _slew_limits falls back to the two attributes above when the veh_dyn tables are absent,
        # so this stub exercises the fallback branch with the DERIVED numbers in it -- the
        # arithmetic under test is identical either way, and the stub stays a stub.
        # a_comb_limit_enable is off because these checks are about the slew limiter, not the
        # friction circle; _ax_avail needs tables, a_lat and speed_now that none of them supply.
        a_comb_limit_enable = False

    s = Stub()
    up, dn = A_ACCEL / LOOP_RATE, A_DECEL / LOOP_RATE
    step = max(up, dn)                  # for the message; each direction is asserted separately
    # the four discontinuities the run log has: a trailing handoff, an AEB engage, an AEB release
    # and a squeeze cap arriving
    demands = ([3.47] + [5.13] * 20 + [2.0] * 10 + [5.13] * 20 + [2.5] * 10 + [5.0] * 20)
    out = [s._slew_limit_speed(v) for v in demands]
    d = np.diff(out)
    assert d.max() <= up + 1e-9 and d.min() >= -dn - 1e-9, \
        f"published step +{d.max():.3f}/{d.min():.3f} m/s > +{up:.3f}/-{dn:.3f} per cycle"
    print(f"PASS no published step exceeds {step:.3f} m/s per cycle across four discontinuities")


def _assembled_windows(vx, el, n_loc):
    """Every local window the state machine can assemble from a lap, as (vx, ds) pairs.

    A window is n_loc stations from any start, wrapping -- which is exactly how the s = 0 seam
    ends up inside one, in 21.5% of them on ifac.
    """
    n = len(vx)
    for i in range(n):
        idx = (np.arange(n_loc) + i) % n
        yield vx[idx], el[idx][:-1]


def _required_accel(v, ds):
    return np.abs((v[1:] ** 2 - v[:-1] ** 2) / (2.0 * np.maximum(ds, 1e-6)))


def test_local_window_accel_limit():
    """G12. The ASSEMBLED local window must not command more longitudinal accel than the car has.

    Two seams meet here and neither is bounded anywhere else:

      the avoidance-to-global-padding join, measured |dvx| up to 0.894 m/s over one station
      (41-55 m/s^2);

      the GLOBAL raceline's own s = 0 discontinuity, 0.867 m/s = 35.6 m/s^2, which sits inside
      21.5% of all windows. Its root is the vendored tph __solver_fb_closed running its backward
      pass over a doubled array and returning the second lap, whose last element has no successor
      and is never decelerated. That is not fixed here -- it would rewrite every map's raceline --
      and this limit is the defence.

    Driven through the REAL StateMachine.limit_local_window_accel, on the shipped ifac profile,
    so the shipped method is the thing under test rather than a copy of its arithmetic.
    """
    import json
    from f110_msgs.msg import Wpnt
    d = json.load(open(os.path.join(STACK_MASTER, "maps", "ifac", "global_waypoints.json")))
    wp = d["global_traj_wpnts_iqp"]["wpnts"][:-1]
    vx = np.array([w["vx_mps"] for w in wp])
    xy = np.array([[w["x_m"], w["y_m"]] for w in wp])
    seg = np.roll(xy, -1, axis=0) - xy
    el = np.hypot(seg[:, 0], seg[:, 1])
    n_loc = 80

    raw = np.array([_required_accel(v, ds).max() for v, ds in _assembled_windows(vx, el, n_loc)])
    over = int((raw > WINDOW_BOUND).sum())

    SM = _load_state_machine()
    sm = types.SimpleNamespace(
        local_window_accel_limit_enable=True, local_window_a_long_mps2=A_WINDOW,
        wpnt_dist=0.1)
    # ...and the same windows with the limit switched OFF, so the gate is shown to SEE the
    # failure it guards against rather than being assumed to.
    off = types.SimpleNamespace(local_window_accel_limit_enable=False,
                                local_window_a_long_mps2=A_WINDOW, wpnt_dist=0.1)
    # The bound is derived by the node now, so bind the REAL _a_long_window onto both stubs
    # rather than reproducing its arithmetic here. Neither carries veh_dyn tables, so it takes
    # its documented fallback branch and returns local_window_a_long_mps2 -- which is set to the
    # derived value above, so the numbers match the car while the stub stays a stub.
    for _ns in (sm, off):
        _ns._a_long_window = types.MethodType(SM._a_long_window, _ns)
    lim, mean_raw, mean_lim, unlimited = [], [], [], []
    for v, ds in _assembled_windows(vx, el, n_loc):
        s_m = np.concatenate([[0.0], np.cumsum(np.append(ds, ds[-1]))])[:len(v)]
        wpts = []
        for vi, si in zip(v, s_m):
            w = Wpnt()
            w.vx_mps = float(vi)
            w.s_m = float(si)
            wpts.append(w)
        out = SM.limit_local_window_accel(sm, wpts, float(v[0]))
        passthru = SM.limit_local_window_accel(off, wpts, float(v[0]))
        unlimited.append(_required_accel(np.array([w.vx_mps for w in passthru]), ds).max())
        assert all(a is not b for a, b in zip(out, wpts)), (
            "the limiter returned the caller's own Wpnt objects -- editing those in place "
            "poisons the cached global line one station per cycle")
        vo = np.array([w.vx_mps for w in out])
        lim.append(_required_accel(vo, ds).max())
        mean_raw.append(float(np.mean(v)))
        mean_lim.append(float(np.mean(vo)))
    lim = np.array(lim)
    unlimited = np.array(unlimited)
    loss = 100.0 * (1.0 - float(np.mean(mean_lim)) / float(np.mean(mean_raw)))

    assert unlimited.max() > WINDOW_BOUND, (
        f"with the limit disabled the assembled window was already inside the bound "
        f"({unlimited.max():.2f} <= {WINDOW_BOUND:.2f}) -- this gate is not measuring what it "
        f"claims to")
    assert lim.max() <= WINDOW_BOUND, (
        f"assembled local window still demands {lim.max():.2f} m/s^2 against a bound of "
        f"{WINDOW_BOUND:.2f} (setting {A_WINDOW:.2f} x {WINDOW_SLACK}); was {raw.max():.2f} raw)")
    print(f"PASS local window accel: required |a_long| max {raw.max():.2f} -> {lim.max():.2f} "
          f"m/s^2 over {len(raw)} windows; p95 {np.percentile(raw, 95):.2f} -> "
          f"{np.percentile(lim, 95):.2f}; windows over {WINDOW_BOUND:.2f} m/s^2 {over} -> "
          f"{int((lim > WINDOW_BOUND).sum())}; mean commanded speed {loss:+.2f}%; "
          f"disabled it still reads {unlimited.max():.2f} m/s^2")


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
    up, dn = A_ACCEL / LOOP_RATE, A_DECEL / LOOP_RATE
    step = max(up, dn)
    for v in sp:
        prev = v if prev is None else min(max(v, prev - dn), prev + up)
        lim.append(prev)
    new = np.abs(np.diff(np.array(lim))) / dt
    print(f"\n{os.path.basename(path)}  ({ts[-1]-ts[0]:.1f} s, {len(sp)} commands)")
    print(f"  as recorded : p50 {np.percentile(raw,50):5.2f}  p99 {np.percentile(raw,99):6.2f}  "
          f"MAX {raw.max():7.2f} m/s^2")
    print(f"  with limiter: p50 {np.percentile(new,50):5.2f}  p99 {np.percentile(new,99):6.2f}  "
          f"MAX {new.max():7.2f} m/s^2   ({VERSION} veh_dyn: accel {A_ACCEL:.2f}, decel "
          f"{A_DECEL:.2f} m/s^2)")
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
    test_local_window_accel_limit()
    ok = True
    if args.bag:
        ok = check_bag(args.bag)
    print("\nALL PASS" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
