#!/usr/bin/env python3
"""Harness test of the static planner's anchoring: commit re-anchor, pre-ramp, grid start.

Three separate ways the published path could carry the car's instantaneous tracking error, or be
anchored somewhere the car is not:

  1. the committed path was DROPPED whenever the car deviated more than commit_dev_max (0.35 m)
     from it -- below the controller's ~0.5 m steady-state error -- and the fresh plan was anchored
     at the displaced car, so the same error dropped it again the next cycle;
  2. everything before the hump's entry ramp held the car's current d flat, up to 10.5 m of
     published path carrying that error as a DC offset, with the hump starting from it;
  3. the grid start index came from cur_s alone, which names the wrong station for up to 0.5 s
     after a static_reopt line swap.

Run (after sourcing the workspace):
  python3 planner/spliner/test/test_path_anchoring.py
"""
import types
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parents[1] / "spliner" / "static_avoidance_node.py"
src = MOD.read_text()
san = types.ModuleType("san")
san.__dict__["__file__"] = str(MOD)
exec(compile(src, str(MOD), "exec"), san.__dict__)
ObstacleSpliner = san.ObstacleSpliner


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def warning(self, *a, **k): pass


class _Converter:
    """Straight raceline along +x; +d is +y. get_cartesian(s, d) -> (2, N)."""
    @staticmethod
    def get_cartesian(s, d):
        return np.vstack([np.asarray(s, float), np.asarray(d, float)])


def node(cur_d=0.0):
    n = ObstacleSpliner.__new__(ObstacleSpliner)
    n.name = "static_avoidance_planner"
    n.get_logger = lambda: _Log()
    n.cur_d = cur_d
    n.cur_x, n.cur_y = 0.0, 0.0
    n.converter = _Converter()
    n.commit_dev_max = 0.6
    n.commit_reanchor_len_m = 2.0
    n.commit_reanchor_max_m = 1.0
    n.preramp_len_m = 3.0
    n.gb_max_idx = 400
    return n


def committed_path(apex_d=0.5, apex_s=6.0, span=12.0, step=0.1):
    s = np.arange(0.0, span, step)
    d = apex_d * np.exp(-((s - apex_s) ** 2) / 2.0)
    return {'s_mod': s.copy(), 'd': d.copy(),
            'xy': np.column_stack([s, d]), 'v': np.ones_like(s),
            'psi': np.zeros_like(s), 'kappa': np.zeros_like(s),
            'obs': [], 'obs_margin': 0.30, 'squeeze': False}


def test_reanchor_moves_the_entry_and_keeps_the_apex():
    n = node(cur_d=0.45)
    c = committed_path()
    s_local = c['s_mod'].copy()
    apex_before = float(np.max(c['d']))
    tail_before = c['d'][s_local > 4.0].copy()

    assert n._reanchor_commit(c, car_prog=0.0, s_local=s_local) is True
    assert abs(float(np.interp(0.0, s_local, c['d'])) - 0.45) < 1e-6, \
        "the path must now start at the car's actual d"
    assert np.allclose(c['d'][s_local > 4.0], tail_before), \
        "everything past the blend -- apex, clearance, exit ramp -- must be untouched"
    assert abs(float(np.max(c['d'])) - apex_before) < 1e-9
    # ...and the correction is faded, not stepped
    seg = c['d'][s_local <= 2.0]
    assert np.all(np.abs(np.diff(seg)) < 0.05), "the blend must not introduce a step"
    print(f"PASS commit re-anchor bends the entry (+0.45 m) and keeps the apex ({apex_before:.2f})")


def test_reanchor_refuses_a_correction_that_is_not_an_entry_fix():
    n = node(cur_d=1.5)                       # way past commit_reanchor_max_m
    c = committed_path()
    assert n._reanchor_commit(c, 0.0, c['s_mod'].copy()) is False, \
        "a correction this large is a re-plan, not an entry adjustment"
    print("PASS an over-large deviation still falls through to a full re-plan")


def test_preramp_decays_the_current_offset_to_the_raceline():
    # Reproduces the published-prefix construction from do_spline for s_entry0 far ahead.
    cur_d, s_entry0, preramp = 0.45, 10.5, 3.0
    s_local = np.arange(0.0, 20.0, 0.1)
    dv = np.zeros(len(s_local))
    pre = min(preramp, s_entry0)
    t = np.clip(s_local / pre, 0.0, 1.0)
    w = 1.0 - t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
    m_pre = s_local <= s_entry0
    dv[m_pre] = cur_d * w[m_pre]

    assert abs(dv[0] - cur_d) < 1e-9, "must start at the car's actual offset"
    assert abs(dv[np.argmin(np.abs(s_local - 3.0))]) < 1e-6, "must reach the raceline by preramp_len"
    held = dv[(s_local > 3.0) & (s_local <= s_entry0)]
    assert np.all(np.abs(held) < 1e-6), "and hold the raceline until the hump opens"
    # the old behaviour, for contrast
    old = np.full(len(s_local), cur_d)
    carried_old = float(np.sum(np.abs(old[m_pre]) > 0.02) * 0.1)
    carried_new = float(np.sum(np.abs(dv[m_pre]) > 0.02) * 0.1)
    assert carried_new < 0.35 * carried_old
    print(f"PASS pre-ramp carries the offset {carried_new:.1f} m instead of {carried_old:.1f} m")


def test_grid_start_is_anchored_by_position():
    # After a line swap cur_s names the right NUMBER on the wrong parameterisation: here the scaled
    # line is 0.8 m "behind" what cur_s implies, so the s-derived index is 8 stations off.
    n = node()
    n.cur_x, n.cur_y = 12.0, 0.0
    gb = [types.SimpleNamespace(x_m=i * 0.1 - 0.8, y_m=0.0, s_m=i * 0.1) for i in range(400)]
    s_idx = 120                                     # = int(cur_s / 0.1) with cur_s = 12.0
    assert abs(gb[s_idx].x_m - n.cur_x) > 0.7, "the harness must reproduce the mis-anchor"
    fixed = n._anchor_car_idx(s_idx, gb, 0.1)
    assert abs(gb[fixed].x_m - n.cur_x) < 0.06, f"expected the nearest station, got x={gb[fixed].x_m}"
    # bounded search: a station a whole lap away must never win
    n.cur_x = 12.0
    far = n._anchor_car_idx(s_idx, gb, 0.1, search_m=3.0)
    assert abs(far - s_idx) <= 30, "the search must stay inside its window"
    print("PASS the grid start is re-anchored by position, within a bounded window")


if __name__ == "__main__":
    test_reanchor_moves_the_entry_and_keeps_the_apex()
    test_reanchor_refuses_a_correction_that_is_not_an_entry_fix()
    test_preramp_decays_the_current_offset_to_the_raceline()
    test_grid_start_is_anchored_by_position()
    print("ALL PASS")
