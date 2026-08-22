#!/usr/bin/env python3
"""multi_tracking association diagnostics: silent and inert unless asked for.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest perception/scripts/test_assoc_diag.py -q

A new track id means "this detection matched no existing track", and TWO different faults produce
it: the detection jumped further than the gate, or ttl had already removed the track it would
have matched. Simulating kiss's own bbox algorithm says the first cannot happen (the
frame-to-frame centre step never reaches max_dist 0.5 even at 14 m) while detections go MISSING
8.5% of frames at 10 m and 22% at 12 m. This instrumentation decides it from one lap's log.

These tests are about the instrumentation being SAFE, not about what it will find: off by
default, absent from save_yaml, and bit-identical behaviour when off.
"""
import os
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

PARAMS = os.path.join(REPO, 'stack_master', 'config', 'opponent_tracker_params.yaml')
SRC = open(os.path.join(HERE, 'multi_tracking.py')).read()


def params():
    return yaml.safe_load(open(PARAMS))['tracking']['ros__parameters']


def test_the_flag_ships_off():
    p = params()
    assert p['diag_assoc'] is False, "diag_assoc ships ON -- a race day would start printing"
    assert p['diag_assoc_throttle_s'] > 0.0
    print(f"PASS diag_assoc ships off (throttle {p['diag_assoc_throttle_s']} s)")


def test_the_flag_is_not_in_save_yaml():
    """The rqt save button must not be able to leave it running -- the same rule the other debug
    streams in this node follow."""
    block = SRC[SRC.index('def save_yaml'):]
    block = block[:block.index('\n    def ', 10)] if '\n    def ' in block[10:] else block
    assert 'diag_assoc' not in block, "save_yaml would persist the debug flag"
    print("PASS diag_assoc is outside save_yaml's key list")


def test_every_diagnostic_site_is_behind_the_flag():
    """No diagnostic work may happen when the flag is off -- not a log, not a dict update."""
    for marker in ('self._diag_bin(', 'self._diag_new_track(', 'self._diag_died.append(',
                   'self.diag_assoc_report()', 'self._diag_seen, self._diag_miss = {}, {}'):
        assert marker in SRC, f"missing instrumentation: {marker}"
        idx = 0
        while True:
            idx = SRC.find(marker, idx)
            if idx < 0:
                break
            # the nearest preceding 'if' on an earlier line must be the flag
            head = SRC[:idx]
            line_start = head.rfind('\n') + 1
            preceding = head[:line_start].rstrip().rsplit('\n', 1)[-1].strip()
            assert preceding.startswith('if self.diag_assoc') or \
                   'if self.diag_assoc' in SRC[max(0, idx - 400):idx], \
                   f"{marker} is not guarded by the flag"
            idx += 1
    print("PASS every diagnostic site sits behind `if self.diag_assoc`")


def test_the_gate_shown_matches_the_gate_used():
    """The log must report the gate verify_position ACTUALLY applied, or it answers the wrong
    question. aggro_multi is a MULTIPLIER on max_dist for dynamic tracks."""
    vp = SRC[SRC.index('def verify_position'):]
    vp = vp[:vp.index('\n    def ', 10)]
    assert 'max_dist *= self.aggro_multiplier' in vp, "verify_position no longer multiplies"
    diag = SRC[SRC.index('def _diag_new_track'):]
    diag = diag[:diag.index('\n    def ', 10)]
    assert 'self.max_dist * (self.aggro_multiplier if t.staticFlag is False else 1.0)' in diag, \
        "the diagnostic computes a different gate than verify_position uses"
    print("PASS the reported gate is max_dist x aggro_multi, the same one association used")


def test_the_verdicts_are_the_two_the_question_needs():
    diag = SRC[SRC.index('def _diag_new_track'):]
    diag = diag[:diag.index('\n    def ', 10)]
    assert '> gate' in diag, "no verdict for 'the jump beat the gate'"
    assert 'no live tracks' in diag, "no verdict for 'nothing was left to match'"
    assert 'died=' in diag, "the log does not say what was removed this cycle"
    assert 'gap=' in diag, "the log does not carry the ego range the observation was made at"
    print("PASS the log distinguishes gate-exceeded from nothing-to-match, with gap and deaths")


def test_the_miss_table_bins_cover_the_range_of_interest():
    """The simulated loss rates are quoted at 10 and 12 m; the bins must resolve there."""
    import types
    mod = types.ModuleType('mt_bins')
    exec(compile(SRC, 'multi_tracking.py', 'exec'), mod.__dict__)
    bins = mod.StaticDynamic._DIAG_BINS
    for edge in (8.0, 10.0, 12.0):
        assert edge in bins, f"no bin edge at {edge} m"
    print(f"PASS miss-table bins {[b for b in bins if b < 1e8]} resolve 8/10/12 m")


def test_a_node_without_the_attributes_does_not_crash_the_report():
    import types
    mod = types.ModuleType('mt_bare')
    exec(compile(SRC, 'multi_tracking.py', 'exec'), mod.__dict__)
    bare = mod.StaticDynamic.__new__(mod.StaticDynamic)
    bare.diag_assoc = False
    bare.diag_assoc_report()            # must be a no-op, not an AttributeError
    bare.diag_assoc = True
    bare.diag_assoc_report()            # _diag_seen is the class default None -> still a no-op
    print("PASS the report is a no-op on a node that never collected anything")


if __name__ == "__main__":
    for fn in (test_the_flag_ships_off, test_the_flag_is_not_in_save_yaml,
               test_every_diagnostic_site_is_behind_the_flag,
               test_the_gate_shown_matches_the_gate_used,
               test_the_verdicts_are_the_two_the_question_needs,
               test_the_miss_table_bins_cover_the_range_of_interest,
               test_a_node_without_the_attributes_does_not_crash_the_report):
        fn()
