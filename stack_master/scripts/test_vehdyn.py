#!/usr/bin/env python3
"""Tests for the vehicle-dynamics measurement tooling.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest stack_master/scripts/test_vehdyn.py -q
    python3 stack_master/scripts/test_vehdyn.py          # standalone, prints each result

Two halves.

SAFETY. The three abort paths, the dry run, the space -> condition derivation and the
pose gate. These are the parts that decide whether a car at 9 m/s stops, so they live in
Guard/Plan free of ROS and are driven directly here. A safety path reachable only by
launching a car is a safety path nobody tests.

ANALYSIS REGRESSION. vehdyn_analyze.py is re-run against ~/vehdyn_0812_1948 and checked
against what that session measured. If the numbers move, the new code is wrong -- the bag
did not change. Skipped (not failed) where the bag is absent, since it is not in the repo.
"""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vehdyn_analyze as VA          # noqa: E402
import vehdyn_test_node as VT        # noqa: E402

BAG = os.path.expanduser('~/vehdyn_0812_1948')
PARAMS = os.path.join(os.path.dirname(HERE), 'config', 'vehdyn_test_params.yaml')


def base_params(**over):
    """The shipped yaml, as the node would see it."""
    import yaml
    with open(PARAMS) as f:
        p = yaml.safe_load(f)['/**']['ros__parameters']
    p.update(over)
    return p


# =======================================================================================
# SAFETY
# =======================================================================================


def test_dry_run_is_the_shipped_default():
    """The file that ships must not move a car when launched as-is."""
    assert base_params()['dry_run'] is True, \
        "vehdyn_test_params.yaml shipped with dry_run false -- launching it would drive"
    print("PASS dry_run defaults to true in the shipped config")


def test_abort_on_joy_axis_and_button():
    """simple_mux only yields on buttons[4], so ANY joy input has to abort here instead."""
    g = VT.Guard(base_params())
    dz = base_params()['joy_abort_axis_deadzone']
    assert g.check_joy([0.0, 0.0], [0, 0]) is None, "idle joy must not abort"
    assert g.check_joy([0.0, dz * 0.5], [0]) is None, "inside the deadzone must not abort"
    why = g.check_joy([0.0, dz + 0.01, 0.0], [0])
    assert why and 'axis 1' in why, why
    why = g.check_joy([0.0], [0, 0, 1])
    assert why and 'button 2' in why, why
    assert g.check_joy([], []) is None, "an empty Joy message must not abort"
    print(f"PASS joy abort: axis beyond {dz}, or any button, and neither on idle input")


def test_abort_on_leaving_the_safety_box():
    g = VT.Guard(base_params(box_center_x=0.0, box_center_y=0.0,
                             box_w=8.0, box_h=6.0, safety_margin_m=1.0))
    # half-extents are 8/2-1 = 3.0 and 6/2-1 = 2.0
    assert g.check_box(0.0, 0.0) is None
    assert g.check_box(2.99, 1.99) is None
    assert g.check_box(3.01, 0.0) is not None, "past the x edge and it did not abort"
    assert g.check_box(0.0, -2.01) is not None, "past the y edge and it did not abort"
    # a margin that eats the box is caught rather than silently inverting the test
    tiny = VT.Guard(base_params(box_w=1.0, box_h=1.0, safety_margin_m=1.0))
    assert 'degenerate' in tiny.check_box(0.0, 0.0)
    print("PASS box abort on both axes, and a degenerate box is refused rather than inverted")


def test_abort_on_distance_budget_and_stale_odom():
    g = VT.Guard(base_params(odom_timeout_s=1.0))
    assert g.check_budget(9.9, 10.0) is None
    assert g.check_budget(10.1, 10.0) is not None
    assert g.check_budget(1e9, None) is None, "no budget = no ceiling"
    # fail closed
    assert g.check_odom(None) is not None, "never having seen odometry must not be permissive"
    assert g.check_odom(0.5) is None
    assert g.check_odom(1.5) is not None
    print("PASS distance budget aborts, and missing/stale odometry fails CLOSED")


def test_speed_command_is_slew_limited():
    """A step command is never published; a step is what the measurement is trying to see."""
    dt, rate = 0.02, 3.0
    v = 0.0
    for _ in range(5):
        nxt = VT.slew(v, 9.0, rate, dt)
        assert nxt - v <= rate * dt + 1e-9, "slew limit exceeded going up"
        v = nxt
    assert v == pytest.approx(5 * rate * dt)
    v = 5.0
    for _ in range(5):
        nxt = VT.slew(v, 0.0, rate, dt)
        assert v - nxt <= rate * dt + 1e-9, "slew limit exceeded coming down"
        v = nxt
    assert VT.slew(1.0, 1.0, rate, dt) == 1.0
    print(f"PASS speed command slews at <= {rate} m/s^2 in both directions")


def test_pose_gate_refuses_a_misplaced_or_misaimed_car():
    ok, d, dy = VT.pose_within((0, 0), 0.0, (0, 0), 0.0, 0.5, 20.0)
    assert ok
    ok, d, dy = VT.pose_within((0.6, 0), 0.0, (0, 0), 0.0, 0.5, 20.0)
    assert not ok and d == pytest.approx(0.6), "0.6 m out of a 0.5 m tolerance was accepted"
    ok, d, dy = VT.pose_within((0, 0), math.radians(30), (0, 0), 0.0, 0.5, 20.0)
    assert not ok and dy == pytest.approx(30.0), "30 deg out of a 20 deg tolerance was accepted"
    # wrap: +179 vs -179 is 2 deg apart, not 358
    ok, d, dy = VT.pose_within((0, 0), math.radians(179), (0, 0), math.radians(-179), 0.5, 20.0)
    assert ok and dy == pytest.approx(2.0, abs=1e-6), (ok, dy)
    print("PASS pose gate refuses distance and heading errors, and wraps heading correctly")


def test_space_changes_the_derived_conditions():
    """Change the room and every derived target must follow. Nothing is hardcoded."""
    # a box big enough not to bind: this test is about area_radius_m / straight_len_m, and the
    # box-vs-area cross-check has its own test below
    big = dict(box_w=50.0, box_h=50.0)
    short = VT.derive_conditions(base_params(straight_len_m=6.0, area_radius_m=3.0, **big))
    long_ = VT.derive_conditions(base_params(straight_len_m=12.0, area_radius_m=6.0, **big))
    assert long_['v_straight'] > short['v_straight'], "a longer straight gave no more speed"
    assert long_['circle_R_m'] > short['circle_R_m'], "a bigger area gave no bigger circle"
    # the documented worked examples: L is straight_len - 2*margin
    for straight_len, want_v in ((6.0, 6.5), (8.0, 8.0), (10.0, 9.2)):
        c = VT.derive_conditions(base_params(straight_len_m=straight_len, **big,
                                             safety_margin_m=1.0, v_max_allowed=99.0))
        assert c['v_straight'] == pytest.approx(want_v, abs=0.15), \
            f"L={straight_len - 2}: got {c['v_straight']:.2f}, expected ~{want_v}"
    print("PASS derived targets track the configured space (L=4/6/8 m -> 6.5/8.0/9.2 m/s)")


def test_a_space_too_small_is_refused_not_driven():
    c = VT.derive_conditions(base_params(straight_len_m=2.2, safety_margin_m=1.0))
    assert not c['straight_feasible'], "a 0.2 m straight was declared usable"
    assert any('SKIPPED' in w for w in c['warnings']), c['warnings']
    _, plan = VT.build_plan(base_params(mode='full', long_mode='shuttle',
                                        straight_len_m=2.2, safety_margin_m=1.0), c)
    assert not any(m['kind'].startswith('T1') for m in plan), \
        "an infeasible longitudinal maneuver still made it into the plan"
    print("PASS a straight too short to measure in is skipped with a warning, not attempted")


def test_v_max_allowed_caps_the_circle_rather_than_the_measurement():
    """If the cap binds before the tyre does, shrink R so washout stays reachable."""
    c = VT.derive_conditions(base_params(area_radius_m=20.0, v_max_allowed=5.0,
                                         box_w=50.0, box_h=50.0, ggv_ay_max_ref=7.0))
    assert c['v_washout'] == pytest.approx(5.0)
    assert c['circle_R_m'] == pytest.approx(25.0 / 7.0, rel=1e-6), c['circle_R_m']
    assert any('reduced' in w for w in c['warnings'])
    print("PASS v_max_allowed shrinks the circle instead of making washout unreachable")


def test_the_circle_must_fit_the_abort_box_not_just_area_radius():
    """area_radius_m and the abort box are two declarations of the same space, and they disagree.

    THE CASE THIS IS FOR, measured on the car 2026-08-13: box 3.5 x 8.0 m with margin 1.0 gives
    +-0.75 m of usable width, but area_radius_m 4.0 derives R = 2.85 m. The plan table said the
    circle fit; the box would have aborted the run a quarter turn in, at speed, after the
    operator had read a table saying it was fine. The box is what stops the car, so it wins.
    """
    p = base_params(box_w=3.5, box_h=8.0, safety_margin_m=1.0,
                    area_radius_m=4.0, car_width_m=0.30)
    c = VT.derive_conditions(p)
    assert c['circle_R_area_m'] == pytest.approx(2.85), c['circle_R_area_m']
    assert c['circle_R_box_m'] == pytest.approx(0.60), c['circle_R_box_m']
    assert c['circle_R_m'] == pytest.approx(0.60), \
        "the plan used the area radius, which does not fit the box that aborts the run"
    assert any('ABORT BOX IS SMALLER' in w for w in c['warnings']), c['warnings']

    # every planned circle must sit inside the box that will abort it
    half = min(p['box_w'] / 2 - p['safety_margin_m'], p['box_h'] / 2 - p['safety_margin_m'])
    assert c['circle_R_m'] + p['car_width_m'] / 2 <= half + 1e-9

    # ...and when the box is the generous one, area_radius_m still binds
    c2 = VT.derive_conditions(base_params(box_w=20.0, box_h=20.0, area_radius_m=4.0,
                                          safety_margin_m=1.0, car_width_m=0.30))
    assert c2['circle_R_m'] == pytest.approx(2.85), "the box overrode a smaller declared area"
    assert not any('ABORT BOX IS SMALLER' in w for w in c2['warnings'])
    print("PASS the circle is clamped to whichever of area_radius_m / abort box is smaller")


def test_the_straight_cannot_exceed_the_abort_box():
    p = base_params(straight_len_m=20.0, box_w=4.0, box_h=8.0, safety_margin_m=1.0)
    c = VT.derive_conditions(p)
    assert c['straight_usable_m'] == pytest.approx(6.0), c['straight_usable_m']
    assert any('abort box only allows' in w for w in c['warnings']), c['warnings']
    print("PASS a straight longer than the abort box is clamped to the box")


def test_venue_mode_forces_circle_only():
    p = base_params(mode='full', venue_mode=True)
    cond = VT.derive_conditions(p)
    mode, plan = VT.build_plan(p, cond)
    assert mode == 'circle'
    kinds = {m['kind'] for m in plan}
    assert kinds == {'T2_washout'}, kinds
    assert {m['side'] for m in plan} == {'left', 'right'}
    print(f"PASS venue_mode forces circle only: {len(plan)} washout runs, both sides")


def test_full_mode_orders_the_steady_state_before_every_step():
    p = base_params(mode='full', long_mode='oval')
    mode, plan = VT.build_plan(p, VT.derive_conditions(p))
    assert mode == 'full'
    kinds = [m['kind'] for m in plan]
    assert any(k == 'T3_step_accel' for k in kinds) and any(k == 'T4_step_brake' for k in kinds)
    # every step maneuver must budget settle time before the step
    assert float(p['step_settle_s']) > 0.0, \
        "step_settle_s is 0 -- T3/T4 would ramp a_x and a_lat together, which is the " \
        "mistake that invalidated the 2026-08-12 runs"
    print(f"PASS full mode plans {len(plan)} maneuvers with a "
          f"{p['step_settle_s']} s constant-speed hold before each step")


def test_every_maneuver_carries_a_distance_budget():
    p = base_params(mode='full')
    _, plan = VT.build_plan(p, VT.derive_conditions(p))
    assert plan
    for m in plan:
        assert m['budget_m'] > 0.0, f"{m['kind']} has no distance ceiling"
    print(f"PASS all {len(plan)} maneuvers carry a precomputed distance ceiling")


def test_the_plan_table_names_the_dry_run_and_the_box():
    p = base_params()
    cond = VT.derive_conditions(p)
    mode, plan = VT.build_plan(p, cond)
    txt = VT.format_plan(p, cond, mode, plan)
    assert 'NOTHING WILL BE PUBLISHED' in txt
    assert 'safety box' in txt and 'budget[m]' in txt
    print("PASS the pre-run table states the dry run, the box and the per-maneuver budget")


def test_dry_run_publishes_nothing_when_the_node_actually_runs():
    """Not 'the flag is set' -- spin the real node and prove the drive topic stays silent.

    No hardware: the node only subscribes and publishes, so odometry and joy are faked here
    and the drive topic is watched from the same process.
    """
    rclpy = pytest.importorskip('rclpy')
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'vtn_run', os.path.join(HERE, 'vehdyn_test_node.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rclpy.init()
    try:
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        p = base_params(dry_run=True, mode='circle')
        overrides = [Parameter(k, value=v) for k, v in p.items()
                     if not isinstance(v, (dict, list))]

        # build the node class the same way main() does, without spinning main()
        seen = []
        watcher = Node('vehdyn_watcher')
        watcher.create_subscription(AckermannDriveStamped, p['drive_topic'],
                                    lambda m: seen.append(m), 10)
        odom_pub = watcher.create_publisher(Odometry, p['odom_topic'], 10)

        node = mod._build_node(overrides)
        exe = rclpy.executors.SingleThreadedExecutor()
        exe.add_node(node)
        exe.add_node(watcher)
        for _ in range(60):
            o = Odometry()
            o.header.stamp = watcher.get_clock().now().to_msg()
            o.pose.pose.orientation.w = 1.0
            odom_pub.publish(o)
            exe.spin_once(timeout_sec=0.01)
        assert seen == [], f"dry_run published {len(seen)} drive command(s)"
        node.destroy_node()
        watcher.destroy_node()
        print("PASS dry_run: node spun with live odometry and published no drive command")
    finally:
        rclpy.shutdown()


# =======================================================================================
# ANALYSIS -- pure helpers
# =======================================================================================


def test_circle_fit_recovers_a_known_circle_and_reports_a_short_arc():
    th = np.linspace(0, 2 * np.pi, 400)
    cx, cy, R, resid, arc = VA.fit_circle(3.0 + 2.5 * np.cos(th), -1.0 + 2.5 * np.sin(th))
    assert (cx, cy) == pytest.approx((3.0, -1.0), abs=1e-6)
    assert R == pytest.approx(2.5, abs=1e-6) and resid < 1e-6
    assert arc == pytest.approx(2 * np.pi, abs=1e-6)
    # a SHORT arc fits with a tiny residual and an unconstrained radius -- the trap the
    # arc gate exists for
    th2 = np.linspace(0, 0.4, 60)
    _, _, R2, resid2, arc2 = VA.fit_circle(2.5 * np.cos(th2), 2.5 * np.sin(th2))
    assert resid2 < 1e-6 and arc2 < 0.5, (resid2, arc2)
    print("PASS circle fit is exact on a full circle; a short arc is flagged by arc, not residual")


def test_find_runs_bridges_short_gaps_and_drops_short_runs():
    t = np.arange(0, 10, 0.1)
    m = np.zeros_like(t, bool)
    m[10:40] = True          # 3.0 s
    m[42:44] = True          # rejoined across a 0.2 s gap
    m[80:82] = True          # 0.2 s on its own -> dropped
    runs = VA.find_runs(m, t, max_gap_s=0.5, min_dur_s=1.0)
    assert len(runs) == 1, runs
    assert t[runs[0][1]] - t[runs[0][0]] == pytest.approx(3.3, abs=0.05)
    print("PASS run finder bridges sub-gap breaks and drops runs below the minimum duration")


# =======================================================================================
# ANALYSIS -- regression against the 2026-08-12 measurement
# =======================================================================================

needs_bag = pytest.mark.skipif(not os.path.isdir(BAG),
                               reason=f"measurement bag {BAG} not present")


@needs_bag
def test_analyzer_reproduces_the_0812_measurement():
    """THE regression test for the analysis half.

    Every number below was measured in the 2026-08-12 session. The bag cannot change, so if
    these move, the analysis code moved and it is wrong.
    """
    import yaml
    with open(PARAMS) as f:
        P = yaml.safe_load(f)['/**']['ros__parameters']
    cfg = P['analyze']
    data = VA.read_bag(BAG, cfg['bag_topics'])
    A = VA.Analysis(data, cfg, P)

    # --- bias ---
    assert A.bias[0] == pytest.approx(-0.407, abs=0.02), A.bias
    assert A.bias[1] == pytest.approx(+0.474, abs=0.02), A.bias
    print(f"  bias ax {A.bias[0]:+.3f} ay {A.bias[1]:+.3f}   (0812: -0.407 / +0.474)")

    # --- axis validation ---
    ax = A.validate_axes()
    assert ax['corr_long'] == pytest.approx(0.926, abs=0.03), ax
    assert ax['corr_lat'] > 0.90, ax
    assert ax['pass'], ax
    assert ax['raw_dvdt_max'] > 1000, "the raw-difference blow-up is gone -- did the " \
                                      "resampling stop being necessary, or the data change?"
    print(f"  corr(ax,dv/dt) {ax['corr_long']:+.3f}  corr(ay,v*gz) {ax['corr_lat']:+.3f}"
          f"   (0812: +0.926 / +0.957)")

    # --- longitudinal ---
    lo = A.longitudinal()
    acc_hi = lo['accel_range'][1]
    brk_hi = abs(lo['brake_range'][0])
    assert 8.5 <= acc_hi <= 10.5, f"accel p95 peak {acc_hi:.2f}, 0812 reported 8.8..10.1"
    assert 12.0 <= brk_hi <= 14.5, f"brake p05 peak {brk_hi:.2f}, 0812 reported 12.6..14.1"
    print(f"  accel p95 up to {acc_hi:.1f}, brake p05 down to {-brk_hi:.1f}"
          f"   (0812: 8.8..10.1 / -12.6..-14.1)")

    # --- lateral ---
    lat = A.lateral()
    assert len(lat['runs']) == 6, \
        f"found {len(lat['runs'])} circle runs, the session drove 3 left + 3 right"
    assert sum(1 for r in lat['runs'] if r['side'] == 'left') == 3
    assert sum(1 for r in lat['runs'] if r['side'] == 'right') == 3
    g_lo, g_hi = lat['geom_range']
    assert 8.5 <= g_lo and g_hi <= 12.5, f"omega^2*R {g_lo:.1f}..{g_hi:.1f}, 0812 said 9.8..11.5"
    i_lo, i_hi = lat['imu_range']
    assert 11.0 <= i_lo and i_hi <= 14.0, f"imu p95 {i_lo:.1f}..{i_hi:.1f}, 0812 said 12.5..13.6"
    # the accelerometer must read HIGH -- roll leaks gravity into the lateral axis
    assert lat['method_gap'] > 0, \
        "imu.ay no longer reads above the geometric method; one of them changed"
    # and the conservative one must be the adopted one
    assert lat['adopted'] == pytest.approx(g_lo), "the higher method was adopted"
    print(f"  omega^2*R {g_lo:.1f}..{g_hi:.1f}, imu p95 {i_lo:.1f}..{i_hi:.1f}, "
          f"adopted {lat['adopted']:.2f}   (0812: 9.8..11.5 / 12.5..13.6)")

    # --- left/right, reported apart ---
    L, R = sorted(lat['left_imu']), sorted(lat['right_imu'])
    assert len(L) == 3 and len(R) == 3
    for got, want in zip(L, [11.8, 12.2, 12.4]):
        assert got == pytest.approx(want, abs=0.5), (L, "0812 left 11.8/12.2/12.4")
    for got, want in zip(R, [12.5, 12.9, 13.4]):
        assert got == pytest.approx(want, abs=0.5), (R, "0812 right 12.5/12.9/13.4")
    print(f"  left  {'/'.join(f'{x:.1f}' for x in L)}   (0812 11.8/12.2/12.4)")
    print(f"  right {'/'.join(f'{x:.1f}' for x in R)}   (0812 12.5/12.9/13.4)")

    # --- current range ---
    cur = A.current()
    assert cur['range'][0] == pytest.approx(-74.9, abs=0.5), cur['range']
    assert cur['range'][1] == pytest.approx(84.0, abs=0.5), cur['range']
    print(f"  motor current {cur['range'][0]:.1f}..{cur['range'][1]:.1f} A   (0812: -74.9..+84.0)")
    print("PASS the analyzer reproduces the 2026-08-12 measurement")


@needs_bag
def test_the_analyzer_writes_candidates_and_never_touches_veh_dyn_info(tmp_path):
    """The output contract: shipped csv format, under vehdyn_measured, live files untouched."""
    live = os.path.join(os.path.dirname(HERE), 'config', 'CAR', 'veh_dyn_info')
    before = {f: open(os.path.join(live, f), 'rb').read()
              for f in os.listdir(live) if f.endswith('.csv')}

    out = str(tmp_path / 'measured')
    rc = subprocess.call([sys.executable, os.path.join(HERE, 'vehdyn_analyze.py'),
                          BAG, '--mode', 'circle', '--out', out],
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    assert rc == 0, "analyzer exited non-zero on the reference bag"

    after = {f: open(os.path.join(live, f), 'rb').read()
             for f in os.listdir(live) if f.endswith('.csv')}
    assert before == after, "THE ANALYZER MODIFIED A LIVE veh_dyn_info FILE"

    for name in ('ggv.csv', 'ax_max_machines.csv', 'b_ax_max_machines.csv', 'report.md'):
        assert os.path.isfile(os.path.join(out, name)), f"{name} was not written"

    # shipped format: same row count and speed grid as the real files, header present
    for name, ncol in (('ggv.csv', 3), ('ax_max_machines.csv', 2), ('b_ax_max_machines.csv', 2)):
        arr = np.atleast_2d(np.loadtxt(os.path.join(out, name), delimiter=',', comments='#'))
        assert arr.shape == (61, ncol), (name, arr.shape)
        assert arr[0, 0] == 0.0 and arr[-1, 0] == 15.0, (name, arr[0, 0], arr[-1, 0])
        head = open(os.path.join(out, name)).readline()
        assert head.startswith('#') and 'MEASURED' in head, (name, head)

    # circle mode must copy the machines tables through untouched
    live_bax = np.atleast_2d(np.loadtxt(os.path.join(live, 'b_ax_max_machines.csv'),
                                        delimiter=',', comments='#'))
    new_bax = np.atleast_2d(np.loadtxt(os.path.join(out, 'b_ax_max_machines.csv'),
                                       delimiter=',', comments='#'))
    assert new_bax[:, 1] == pytest.approx(live_bax[:, 1]), \
        "circle mode changed b_ax_max_machines -- it is a brake property, not a surface one"

    rep = open(os.path.join(out, 'report.md')).read()
    for must in ('REGENERATE THE RACELINE', 'Venue re-calibration reference', 'k =',
                 'COPIED UNCHANGED'):
        assert must in rep, f"report.md does not mention {must!r}"
    print("PASS candidates are written in the shipped format; live veh_dyn_info is untouched")


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        if not os.path.isdir(BAG) and 'reproduces' in fn.__name__ or \
           not os.path.isdir(BAG) and 'candidates' in fn.__name__:
            print(f"SKIP {fn.__name__} (no bag)")
            continue
        try:
            if fn.__code__.co_argcount:
                print(f"SKIP {fn.__name__} (needs a pytest fixture)")
                continue
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
