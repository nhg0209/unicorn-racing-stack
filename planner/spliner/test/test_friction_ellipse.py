#!/usr/bin/env python3
"""static_avoidance_node: the avoidance speed profile respects the friction circle.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest planner/spliner/test/test_friction_ellipse.py -q

THE BUG. The curvature cap spent the full lateral limit and the two longitudinal passes spent the
full longitudinal limit, independently, so at the shoulders of an apex the profile demanded
sqrt(7.0^2 + 7.0^2) = 9.90 m/s^2 from a tyre that has 7.0. The car cannot steer to a plan like
that, and a fast straight is where the forward pass is at full acceleration -- so straight-line
avoidance failed while corner-exit avoidance, where speed is already pinned by curvature and the
longitudinal term is small, worked.
"""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'planner', 'spliner'))

import types                                                          # noqa: E402
MOD = os.path.join(REPO, 'planner', 'spliner', 'spliner', 'static_avoidance_node.py')
san = types.ModuleType("san")
san.__dict__["__file__"] = MOD
exec(compile(open(MOD).read(), MOD, "exec"), san.__dict__)

CAR = os.path.join(REPO, 'stack_master', 'config', 'CAR')
DS = 0.1


def weave(n=220, amp=0.55, w=18):
    """Two curvature lobes: the shape a published avoidance actually has."""
    x = np.arange(n)
    return (amp * np.exp(-0.5 * ((x - n * 0.35) / w) ** 2)
            - amp * np.exp(-0.5 * ((x - n * 0.62) / w) ** 2))


def profile(kappa, v_gb, a_lat, a_brk, a_acc, p=None):
    """The shipped passes. p=None reproduces the pre-fix behaviour (no ellipse)."""
    v = np.minimum(v_gb, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-3)))

    def avail(v_i, k_i, a_long):
        if p is None:
            return a_long
        used = min((v_i * v_i * abs(float(k_i))) / max(a_lat, 1e-6), 1.0)
        return a_long * max(0.0, 1.0 - used ** p) ** (1.0 / p)

    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * avail(v[i + 1], kappa[i + 1], a_brk) * DS))
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * avail(v[i - 1], kappa[i - 1], a_acc) * DS))
    return v


def combined(v, kappa):
    a_lat = v * v * np.abs(kappa)
    a_lon = np.zeros_like(v)
    a_lon[1:] = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * DS)
    return np.hypot(a_lat, a_lon)


# =======================================================================================


def test_p_is_read_from_the_ini_not_hardcoded():
    p = san.load_dyn_model_exp(CAR)
    assert 1.0 <= p <= 2.0, p
    src = open(MOD).read()
    assert "dyn_model_exp" in src and "load_dyn_model_exp" in src
    # the value must come from the file, so changing the file must change the answer
    import re
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, 'racecar_f110.ini')
        shutil.copy(os.path.join(CAR, 'racecar_f110.ini'), dst)
        txt = re.sub(r'("dyn_model_exp"\s*:\s*)[-+0-9.eE]+', r'\g<1>1.4', open(dst).read())
        open(dst, 'w').write(txt)
        assert san.load_dyn_model_exp(tmp) == pytest.approx(1.4), \
            "p did not follow the file -- it is coming from somewhere else"
        # ...and an out-of-range value is refused rather than planned against
        open(dst, 'w').write(re.sub(r'("dyn_model_exp"\s*:\s*)[-+0-9.eE]+', r'\g<1>3.0', txt))
        with pytest.raises(ValueError):
            san.load_dyn_model_exp(tmp)
    print(f"PASS dyn_model_exp read from racecar_f110.ini ({p}), not hardcoded")


def test_the_pre_fix_profile_exceeds_the_tyre_on_a_fast_straight():
    """The failure, pinned."""
    k = weave()
    a = 7.0
    v = profile(k, np.full(len(k), 8.0), a, a, a, p=None)
    dm = combined(v, k)
    assert dm.max() > 9.0, dm.max()
    assert dm.max() == pytest.approx(math.hypot(a, a), rel=0.02), dm.max()
    print(f"PASS pre-fix: combined demand peaks at {dm.max():.2f} m/s^2 against a_lat_max {a}")


def test_the_ellipse_brings_the_demand_inside_the_tyre():
    k = weave()
    a = 7.0
    for v_gb in (8.0, 6.0, 5.0):
        v = profile(k, np.full(len(k), v_gb), a, a, a, p=2.0)
        dm = combined(v, k)
        assert dm.max() <= a * 1.01, f"v_gb={v_gb}: {dm.max():.2f} > {a}"
    print("PASS with the ellipse the combined demand stays inside a_lat_max at 5-8 m/s")


def test_a_slow_corner_case_is_not_made_worse():
    """Where the curvature cap already binds, the longitudinal term is small and the ellipse must
    barely bite -- otherwise every corner-exit avoidance gets slower for nothing."""
    k = weave()
    a = 7.0
    v_gb = np.full(len(k), 3.5)
    v_old = profile(k, v_gb.copy(), a, a, a, p=None)
    v_new = profile(k, v_gb.copy(), a, a, a, p=2.0)
    assert np.allclose(v_old, v_new, atol=1e-9), \
        f"slow case changed: max |dv| = {np.max(np.abs(v_old - v_new)):.4f} m/s"
    print(f"PASS the 3.5 m/s corner case is bit-identical "
          f"(demand {combined(v_old, k).max():.2f} m/s^2, unchanged)")


def test_the_ellipse_costs_speed_only_where_it_must():
    """The trade this buys: slower through the apex shoulders, same everywhere else."""
    k = weave()
    a = 7.0
    v_old = profile(k, np.full(len(k), 8.0), a, a, a, p=None)
    v_new = profile(k, np.full(len(k), 8.0), a, a, a, p=2.0)
    assert np.all(v_new <= v_old + 1e-9), "the ellipse made the profile FASTER somewhere"
    lost = v_old - v_new
    assert lost.max() > 0.05, "the ellipse changed nothing on a fast straight"
    apex = int(len(k) * 0.35)
    assert lost[apex - 30:apex + 30].max() >= lost.max() * 0.5, \
        "the speed loss is not concentrated where the curvature is"
    print(f"PASS speed cost: max {lost.max():.2f} m/s, mean {lost.mean():.3f} m/s, "
          f"concentrated at the apex")


def test_a_node_built_without_init_still_gets_an_exponent():
    """The offline sweeps build this class with __new__. p must not be missing, and must not
    fall back to the DIAMOND (1.0), which would slow every avoidance."""
    bare = san.ObstacleSpliner.__new__(san.ObstacleSpliner)
    assert bare._dyn_exp() == pytest.approx(2.0)
    bare._dyn_model_exp = 1.5
    assert bare._dyn_exp() == pytest.approx(1.5)
    print("PASS a __new__-built node falls back to the ellipse (2.0), never the diamond")


if __name__ == "__main__":
    for fn in (test_p_is_read_from_the_ini_not_hardcoded,
               test_the_pre_fix_profile_exceeds_the_tyre_on_a_fast_straight,
               test_the_ellipse_brings_the_demand_inside_the_tyre,
               test_a_slow_corner_case_is_not_made_worse,
               test_the_ellipse_costs_speed_only_where_it_must,
               test_a_node_built_without_init_still_gets_an_exponent):
        fn()
