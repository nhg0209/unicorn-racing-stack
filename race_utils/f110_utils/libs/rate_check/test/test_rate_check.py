#!/usr/bin/env python3
"""The invariant that would have caught seven bugs, and the cost of carrying it.

Run:
  ~/miniforge3/envs/unicorn/bin/python3 race_utils/f110_utils/libs/rate_check/test/test_rate_check.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rate_check.rate_check import RateCheck        # noqa: E402


class _Log:
    def __init__(self): self.msgs = []
    def warn(self, m, **k): self.msgs.append(str(m))
    info = error = warn


class _Node:
    def __init__(self): self._log = _Log()
    def get_logger(self): return self._log


def _drive(rc, hz, n, clock):
    """Advance a fake monotonic clock at `hz` and tick `n` times."""
    for _ in range(n):
        clock[0] += 1.0 / hz
        rc.tick()


def _patched(clock):
    time.monotonic = lambda: clock[0]            # noqa: S3D -- test double, restored by caller


def test_a_matching_rate_says_nothing():
    real = time.monotonic
    clock = [0.0]
    _patched(clock)
    try:
        n = _Node()
        rc = RateCheck(n, nominal_hz=40.0, name="t", n_samples=50)
        _drive(rc, 40.0, 60, clock)
        assert not n._log.msgs, n._log.msgs
    finally:
        time.monotonic = real
    print("PASS a node running at its nominal rate is silent")


def test_a_mismatch_warns_once_and_carries_the_consequence():
    real = time.monotonic
    clock = [0.0]
    _patched(clock)
    try:
        n = _Node()
        rc = RateCheck(n, nominal_hz=40.0, name="multi_tracking", n_samples=50,
                       consequence="the speed estimate and the KF dt")
        _drive(rc, 25.0, 400, clock)             # long past n_samples: still exactly one warning
        assert len(n._log.msgs) == 1, f"warned {len(n._log.msgs)} times"
        m = n._log.msgs[0]
        assert "40.0" in m and "25.0" in m, m    # BOTH numbers, not just the complaint
        assert "1.60" in m, m                    # and the factor everything downstream is out by
        assert "the speed estimate and the KF dt" in m, m
    finally:
        time.monotonic = real
    print("PASS a mismatch warns once, with nominal, measured, factor and consequence")


def test_it_stops_working_after_it_is_done():
    """`n_samples` then free: the loop must not keep paying for a finished diagnostic."""
    real = time.monotonic
    clock = [0.0]
    _patched(clock)
    try:
        rc = RateCheck(_Node(), nominal_hz=40.0, n_samples=10)
        _drive(rc, 25.0, 20, clock)
        assert rc._done
        before = (rc._n, rc._sum)
        _drive(rc, 25.0, 1000, clock)
        assert (rc._n, rc._sum) == before, "a finished RateCheck is still accumulating"
    finally:
        time.monotonic = real
    print("PASS after n_samples it accumulates nothing and returns on one branch")


def test_a_stall_is_not_a_period():
    """One debugger pause must not be able to fake a rate problem."""
    real = time.monotonic
    clock = [0.0]
    _patched(clock)
    try:
        n = _Node()
        rc = RateCheck(n, nominal_hz=40.0, n_samples=50)
        _drive(rc, 40.0, 25, clock)
        clock[0] += 30.0                         # a 30 s stall
        rc.tick()
        _drive(rc, 40.0, 40, clock)
        assert not n._log.msgs, n._log.msgs
    finally:
        time.monotonic = real
    print("PASS an outlier period is discarded rather than averaged in")


def test_no_node_and_no_nominal_are_both_safe():
    rc = RateCheck(None, nominal_hz=0.0)
    assert rc.tick() is None and rc._done, "a zero nominal must disable the check, not divide by it"
    RateCheck(None, nominal_hz=40.0, n_samples=2)     # node=None must not raise on construction
    print("PASS a missing node or nominal disables the check instead of raising")


if __name__ == "__main__":
    test_a_matching_rate_says_nothing()
    test_a_mismatch_warns_once_and_carries_the_consequence()
    test_it_stops_working_after_it_is_done()
    test_a_stall_is_not_a_period()
    test_no_node_and_no_nominal_are_both_safe()
    print("ALL PASS")
