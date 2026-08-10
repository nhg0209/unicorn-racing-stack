#!/usr/bin/env python3
"""Say it out loud when a node's NOMINAL rate is not the rate it actually runs at.

This exists because the same defect has now been found seven times in one campaign, in seven
unrelated places, and every one of them was a constant that disagreed with reality while nothing
in the system compared the two:

    the offline harness's lap length      wpnts[-1].s_m + spacing, against the node's wpnts[-1].s_m
    the harness's wall_margin             0.10, against the shipped 0.15
    the harness's kappa_add_max           2.0, against the shipped 5.0
    the harness's SWAPPED-bounds warning  built on `os`, which the module never imported
    the harness's cur_vs                  3.0, exactly ON squeeze_max_speed_mps, so the squeeze
                                          pass never once ran in any measurement
    state_machine's rate                  80, against a measured 38 Hz
    multi_tracking's rate_tracking        40, against a measured 25 Hz

Seven bugs is the wrong way to count them. It is one missing invariant, and the invariant is
cheap: a loop knows when it was last entered, so it knows its own period, so it can be asked
whether that period is the one it was configured with.

WHAT A WRONG NOMINAL RATE COSTS is never just the rate. It is used as `dt` wherever a derivative
or an integral is taken, so it silently rescales everything downstream of it -- which is why the
warning takes a `consequence` string and prints it. In multi_tracking the same constant divides
the speed estimate (`vs = ds * rate`), sets the Kalman filter's process noise (`dt = 1/rate`),
sizes the classification window (`win_t = (n-1)/rate`) and counts down a demotion; at 40 against a
true 25 every one of those is out by 1.6x, so a stationary box reads as moving at 1.6x and fails
the static vote it needs to pass. Nobody would have found that from a rate number alone.

CHEAP, AND THEN FREE. It stores one float and one int, does one subtraction per call, and after
`n_samples` calls it takes a single already-False branch and returns. It never allocates after
construction, never formats a string unless it is about to warn, and warns AT MOST ONCE.

It reports; it does not correct. Whether the fix is to raise the real rate or to lower the
nominal is a judgement about the node -- for tracking the two are not equivalent, because
lowering the nominal to match also relaxes every speed threshold by the same 1.6x, including the
one that demotes a moving opponent from `static`. The helper's job is to make the choice visible,
not to make it.

    self._rate_check = RateCheck(self, nominal_hz=self.rate_hz, name="state_machine",
                                 consequence="every dt-derived quantity in the loop")
    ...
    def loop(self):
        self._rate_check.tick()

`node` may be None (or any object with a `get_logger()`), which is what keeps this testable with
no ROS on the path -- the same property every offline gate in this repo depends on.
"""
import time

__all__ = ["RateCheck"]

_DEFAULT_SAMPLES = 200          # ~5 s at 40 Hz: long enough that startup jitter cannot dominate
_DEFAULT_TOL = 0.10             # report a disagreement of more than 10 %
# Periods outside this band are not scheduling, they are a stall, a debugger, or a clock jump, and
# averaging them in would let one pause fake a rate problem that is not there.
_SANE_LO, _SANE_HI = 0.05, 20.0


class RateCheck:
    """Measure the period between successive `tick()`s; warn once if it disagrees with nominal."""

    def __init__(self, node=None, nominal_hz=None, name="", consequence="",
                 n_samples=_DEFAULT_SAMPLES, tol=_DEFAULT_TOL):
        self._node = node
        self._nominal = float(nominal_hz) if nominal_hz else 0.0
        self._name = name or (getattr(node, "name", "") or "node")
        self._consequence = consequence
        self._need = int(n_samples)
        self._tol = float(tol)
        self._prev = None
        self._sum = 0.0
        self._n = 0
        # The one flag every later call tests and nothing else. Set it now when there is nothing
        # to check, so a misconfigured construction costs one branch per loop and no more.
        self._done = self._nominal <= 0.0 or self._need < 2

    def tick(self):
        """Call once per loop entry. Returns the measured Hz when it reports, else None."""
        if self._done:
            return None
        now = time.monotonic()
        prev, self._prev = self._prev, now
        if prev is None:
            return None
        dt = now - prev
        lo, hi = _SANE_LO / self._nominal, _SANE_HI / self._nominal
        if not (lo <= dt <= hi):
            return None                       # a stall or a clock jump, not a period
        self._sum += dt
        self._n += 1
        if self._n < self._need:
            return None
        self._done = True                     # whatever happens next, this object is finished
        measured = self._n / self._sum
        if abs(measured - self._nominal) <= self._tol * self._nominal:
            return measured
        self._warn(measured)
        return measured

    def _warn(self, measured):
        pct = 100.0 * (measured - self._nominal) / self._nominal
        msg = (f"[{self._name}] RATE MISMATCH: configured {self._nominal:.1f} Hz, measured "
               f"{measured:.1f} Hz over {self._n} periods ({pct:+.0f} %). "
               f"The configured value is what the code divides by, so it is not just a rate: "
               f"{self._consequence or 'every dt-derived quantity in this loop'} is out by a "
               f"factor of {self._nominal / measured:.2f}. "
               f"Fix the loop or fix the constant -- but do not leave them disagreeing.")
        log = getattr(self._node, "get_logger", None)
        if log is None:
            print(msg)
            return
        try:
            log().warn(msg)
        except Exception:                     # noqa: BLE001 -- a diagnostic never breaks a loop
            print(msg)
