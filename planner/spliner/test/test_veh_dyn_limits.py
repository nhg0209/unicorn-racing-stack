#!/usr/bin/env python3
"""The avoidance planner's acceleration limits come from veh_dyn_info, or say why not.

WHY. Race day resurfaces the grip. The intended procedure is: measure a washout, multiply a
factor k into ggv.csv, regenerate the raceline (seconds), go. That only works if k lands in ONE
file. static_avoidance_node used to carry a_lat_max 6.0 / a_long_max 4.0 / a_long_accel 3.0 of
its own, outside that path, so a re-grip would have left the avoidance planner solving against
the old surface -- silently, until the car pushed wide on an avoidance it had planned too fast.

What this pins:
  * which file feeds which limit, and that the mapping is the one the offline velocity profile
    uses (friction bound AND the matching machine bound, drive side vs brake side),
  * that every table is INTERPOLATED at the speed of interest, not folded to a scalar,
  * and that an unreadable config FALLS BACK to the yaml values with exactly one warning,
    because this is a planner that is already driving when it finds out.

Run:
  python3 planner/spliner/test/test_veh_dyn_limits.py
"""
import shutil
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MOD = REPO / "planner" / "spliner" / "spliner" / "static_avoidance_node.py"
CAR = REPO / "stack_master" / "config" / "CAR"

san = types.ModuleType("san")
san.__dict__["__file__"] = str(MOD)
exec(compile(MOD.read_text(), str(MOD), "exec"), san.__dict__)

V_Q = 3.0   # the ego speed these tests interrogate the node at


def shipped(v, cfg=CAR):
    """(ay_max, ax_max, ax_max_machines, b_ax_max_machines) at speed `v`, READ FROM cfg's csvs.

    NOT a hardcoded mirror. It was one, and it went stale twice in two days -- ggv 5.0->7.0 with
    ay 4.5->5.7 (2026-08-11), then b_ax 5.0->10.0 (2026-08-12, measured braking -12.6..-14.1
    m/s^2 over six straight-line stops). Both times the mirror failed, not the code it guarded.

    Interpolated rather than read off row 0 because ax_max_machines is the one shipped table that
    is genuinely speed-dependent (9.5 at rest, 4.62 at 15 m/s).

    WHAT THIS COSTS, AND WHERE IT IS PAID BACK. Reading the expectation from the same files as
    the code under test cannot catch the two longitudinal limits being wired to each other's csv
    -- and today it could not anyway, because min(7.0, 10.0) and min(7.0, 9.5) are both 7.0, so
    the shipped numbers no longer separate them. test_the_brake_and_drive_sides_read_different
    _files pins that mapping on synthetic tables where all four values differ, which is the only
    place it can be pinned independently of what happens to be shipped.
    """
    vdi = cfg / "veh_dyn_info"

    def col(name, c):
        t = np.atleast_2d(np.loadtxt(vdi / name, delimiter=",", comments="#"))
        return float(np.interp(v, t[:, 0], t[:, c]))

    return (col("ggv.csv", 2), col("ggv.csv", 1),
            col("ax_max_machines.csv", 1), col("b_ax_max_machines.csv", 1))


AY, AX, AXM_0, BAX = shipped(V_Q)


class _Log:
    def __init__(self):
        self.warns, self.infos = [], []

    def warn(self, m, *a, **k):
        self.warns.append(m)

    def info(self, m, *a, **k):
        self.infos.append(m)

    def error(self, m, *a, **k):
        self.warns.append(m)


def node(**over):
    """The node without ROS, the way the offline sweeps build it (__new__, no __init__)."""
    n = san.ObstacleSpliner.__new__(san.ObstacleSpliner)
    n.name = "static_avoidance_planner"
    n._log = _Log()
    n.get_logger = lambda: n._log
    n.a_lat_max, n.a_long_max, n.a_long_accel = 6.0, 4.0, 3.0     # the yaml fallbacks
    n._veh_dyn, n._veh_dyn_warned = None, False
    n.cur_vs = 3.0
    for k, v in over.items():
        setattr(n, k, v)
    return n


def test_each_limit_comes_from_the_file_its_use_implies():
    t = san.load_veh_dyn_tables(str(CAR))
    lim = san.resolve_veh_dyn_limits(t, 3.0)
    # the curvature cap is the raceline's own lateral budget
    assert lim["a_lat_max"] == pytest.approx(AY)
    # braking: friction bound AND the brake machine bound
    assert lim["a_long_max"] == pytest.approx(min(AX, BAX))
    # driving: friction bound AND the drive machine bound
    assert lim["a_long_accel"] == pytest.approx(min(AX, AXM_0))
    # and that is a real change from what shipped, in three different directions
    assert lim["a_lat_max"] < 6.0 and lim["a_long_max"] > 4.0 and lim["a_long_accel"] > 3.0
    print(f"PASS a_lat_max 6.0->{lim['a_lat_max']:.2f}, a_long_max 4.0->{lim['a_long_max']:.2f}, "
          f"a_long_accel 3.0->{lim['a_long_accel']:.2f}")


def test_the_tables_are_interpolated_not_folded():
    # Every shipped column but ax_max_machines is flat, so min / mean / interp agree today and a
    # scalar would look right. Pin the interpolation on synthetic tables that are not flat.
    t = (np.array([[0.0, 8.0], [10.0, 4.0]]),     # ay
         np.array([[0.0, 9.0], [10.0, 3.0]]),     # ax
         np.array([[0.0, 20.0], [10.0, 20.0]]),   # ax_machines, never binding
         np.array([[0.0, 20.0], [10.0, 20.0]]))   # b_ax_machines, never binding
    mid = san.resolve_veh_dyn_limits(t, 5.0)
    assert mid["a_lat_max"] == pytest.approx(6.0), mid
    assert mid["a_long_max"] == pytest.approx(6.0) and mid["a_long_accel"] == pytest.approx(6.0)
    assert mid["a_lat_max"] != min(t[0][:, 1]), "folded to the table minimum"
    # numpy holds the end values outside the table -- no extrapolation past what was measured
    assert san.resolve_veh_dyn_limits(t, 99.0)["a_lat_max"] == pytest.approx(4.0)
    # ...and on the SHIPPED tables the speed dependence is real, it just does not bite yet:
    # ax_max_machines falls under the 7.0 friction bound only above ~11 m/s.
    real = san.load_veh_dyn_tables(str(CAR))
    assert san.resolve_veh_dyn_limits(real, 3.0)["a_long_accel"] == pytest.approx(7.0)
    assert san.resolve_veh_dyn_limits(real, 15.0)["a_long_accel"] < 5.0
    print("PASS every column is interpolated at the speed asked for")


def test_the_brake_and_drive_sides_read_different_files():
    """a_long_max must come from b_ax_max_machines, a_long_accel from ax_max_machines.

    The SHIPPED tables cannot prove this and have not been able to since 2026-08-12: min(ggv 7.0,
    b_ax 10.0) and min(ggv 7.0, ax_machines 9.5) are both 7.0, so the two limits are numerically
    identical and swapping the two csvs in resolve_veh_dyn_limits would change nothing anyone
    could observe. That check used to ride on hardcoded constants in this file; now that `shipped`
    reads the real files, this is where it lives. Four distinct values, both machine tables
    binding, so every wire is separable.
    """
    def flat(x):
        return np.array([[0.0, x], [20.0, x]])

    t = (flat(9.0),    # ay              -- largest, so a lateral/longitudinal swap also shows
         flat(8.0),    # ax              -- friction bound, deliberately NOT binding on either side
         flat(3.0),    # ax_max_machines -> drive side only
         flat(2.0))    # b_ax_max_machines -> brake side only
    lim = san.resolve_veh_dyn_limits(t, 5.0)
    assert lim["a_lat_max"] == pytest.approx(9.0), "a_lat_max is not ggv's ay column"
    assert lim["a_long_max"] == pytest.approx(2.0), "the brake side is not reading b_ax_max_machines"
    assert lim["a_long_accel"] == pytest.approx(3.0), "the drive side is not reading ax_max_machines"
    print("PASS the brake side reads b_ax_max_machines, the drive side ax_max_machines")


def test_both_shipped_config_directories_load():
    for version in ("CAR", "SIM"):
        t = san.load_veh_dyn_tables(str(REPO / "stack_master" / "config" / version))
        assert len(t) == 4
        for name, tbl in zip(("ay", "ax", "ax_machines", "b_ax_machines"), t):
            assert tbl.ndim == 2 and tbl.shape[1] == 2 and tbl.shape[0] >= 2, (version, name)
            assert np.all(np.diff(tbl[:, 0]) > 0), f"{version}/{name}: speed not increasing"
            assert np.all(np.isfinite(tbl[:, 1])) and np.all(tbl[:, 1] > 0), (version, name)
    print("PASS both shipped config directories load, all four tables")


def test_a_node_built_without_init_still_answers():
    # THE bug this test was added for. The offline sweeps build ObstacleSpliner with __new__ and
    # never run __init__, so `self._veh_dyn` does not exist at all -- not None, absent. Reading
    # it directly raised AttributeError inside the solve, Harness.cell() swallowed it, and both
    # race sweeps died with KeyError: 'reject' on every cell. A sweep that reports malformed
    # results instead of refusals is worse than one that crashes.
    bare = san.ObstacleSpliner.__new__(san.ObstacleSpliner)
    bare.a_lat_max, bare.a_long_max, bare.a_long_accel = 5.7, 5.0, 7.0
    assert not hasattr(bare, "_veh_dyn")
    assert bare._a_limits() == (5.7, 5.0, 7.0)
    print("PASS a node built with __new__ (the sweep harness) falls back instead of raising")


def test_a_missing_file_raises_rather_than_defaulting_silently():
    files = ("veh_dyn_info/ggv.csv", "veh_dyn_info/ax_max_machines.csv",
             "veh_dyn_info/b_ax_max_machines.csv")
    for drop in files:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp)
            (dst / "veh_dyn_info").mkdir()
            for rel in files:
                if rel != drop:
                    shutil.copy(CAR / rel, dst / rel)
            with pytest.raises(Exception):
                san.load_veh_dyn_tables(str(dst))
    print("PASS a missing table raises; the caller decides what to do about it")


def test_an_unreadable_config_falls_back_and_warns_exactly_once():
    # THE requirement: this planner is already driving. It must not gain a startup failure.
    n = node()
    n.declare_parameter = lambda name, default, desc=None: types.SimpleNamespace(value="NOPE")
    n._load_veh_dyn()
    assert n._veh_dyn is None
    assert n._a_limits() == (6.0, 4.0, 3.0), "the yaml fallbacks were not kept"
    assert len(n._log.warns) == 1, f"warned {len(n._log.warns)} times, want exactly one"
    w = n._log.warns[0]
    assert "6.0" in w and "4.0" in w and "3.0" in w, "the warning must name the values in force"
    assert "will NOT reach this planner" in w, "the warning must name the consequence"
    print("PASS an unreadable config keeps the fallbacks and warns once, naming the cost")


def test_a_readable_config_reaches_the_solver_and_tracks_the_ego_speed():
    n = node()
    n.declare_parameter = lambda name, default, desc=None: types.SimpleNamespace(value="CAR")
    n._load_veh_dyn()
    assert n._veh_dyn is not None and not n._log.warns, n._log.warns
    assert n._a_limits() == pytest.approx((AY, min(AX, BAX), min(AX, AXM_0)))
    # _a_limits reads cur_vs, so the same node answers differently at a different ego speed
    n.cur_vs = 15.0
    assert n._a_limits()[2] < 5.0, "the limit did not follow the ego speed"
    # a node that has not seen odometry yet must still answer
    del n.cur_vs
    assert n._a_limits()[0] == pytest.approx(AY)
    print("PASS a readable config feeds the solver and the limits follow the ego speed")


def test_the_yaml_still_carries_the_fallbacks_it_documents():
    import yaml
    p = yaml.safe_load(
        (REPO / "stack_master/config/static_avoidance_params.yaml").read_text())
    p = p["static_avoidance_planner"]["ros__parameters"]
    # the keys stay declared (dropping them would break the declare/YAML symmetry the node's
    # own param wiring test guards) and they stay equal to the in-code fallbacks
    assert (p["a_lat_max"], p["a_long_max"], p["a_long_accel"]) == (6.0, 4.0, 3.0)
    print("PASS the yaml keys remain, as the documented fallbacks")


if __name__ == "__main__":
    test_each_limit_comes_from_the_file_its_use_implies()
    test_the_tables_are_interpolated_not_folded()
    test_both_shipped_config_directories_load()
    test_a_missing_file_raises_rather_than_defaulting_silently()
    test_an_unreadable_config_falls_back_and_warns_exactly_once()
    test_a_readable_config_reaches_the_solver_and_tracks_the_ego_speed()
    test_the_yaml_still_carries_the_fallbacks_it_documents()
    print("ALL PASS")
