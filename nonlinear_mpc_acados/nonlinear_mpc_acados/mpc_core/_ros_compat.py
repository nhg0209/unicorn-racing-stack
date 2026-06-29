"""ROS-free compat shims for mpc_core.

The MPC classes in this package are pure Python — they don't import rospy,
tf, or any ROS message types. This module provides minimal replacements for
the ROS facilities the original code used:

- `NullLogger` mimics `rospy.loginfo / logwarn / logwarn_throttle`. The ROS
  wrapper injects its own logger (a `rclpy` node logger) via the MPC
  constructor's `logger=` kwarg; `NullLogger` is the standalone fallback.
- `monotonic_now()` replaces `rospy.Time.now().to_sec()`.
- `yaw_to_quat(yaw)` replaces `tf.transformations.quaternion_from_euler(0, 0, yaw)`.
  Returns `(x, y, z, w)` — same convention as tf.
"""
from __future__ import annotations

import math
import sys
import time


class NullLogger:
    """Stdlib-only replacement for the rospy logger interface used by the MPC.

    Implements the subset of rospy logging that the MPC code calls:
      info, warn, warning, error, debug, info_throttle, warn_throttle.

    Throttled variants suppress repeats of the same `msg` template within
    `period` seconds — matches `rospy.logwarn_throttle(period, msg, *args)`.
    """

    def __init__(self, prefix: str = ""):
        self._prefix = prefix
        self._last_throttle: dict[str, float] = {}

    def _emit(self, level: str, msg: str, args: tuple) -> None:
        try:
            text = msg % args if args else msg
        except Exception:
            text = msg + " " + " ".join(str(a) for a in args)
        print(f"[{level}]{self._prefix} {text}", file=sys.stderr)

    def info(self, msg, *args):    self._emit("INFO", str(msg), args)
    def warn(self, msg, *args):    self._emit("WARN", str(msg), args)
    def warning(self, msg, *args): self._emit("WARN", str(msg), args)
    def error(self, msg, *args):   self._emit("ERROR", str(msg), args)
    def debug(self, msg, *args):   self._emit("DEBUG", str(msg), args)

    def _throttled(self, level_fn, period: float, msg: str, args: tuple):
        now = time.monotonic()
        last = self._last_throttle.get(msg, 0.0)
        if now - last >= period:
            self._last_throttle[msg] = now
            level_fn(msg, *args)

    def info_throttle(self, period, msg, *args):
        self._throttled(self.info, period, str(msg), args)

    def warn_throttle(self, period, msg, *args):
        self._throttled(self.warn, period, str(msg), args)


def monotonic_now() -> float:
    """Replacement for `rospy.Time.now().to_sec()`. Monotonic clock seconds."""
    return time.monotonic()


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """yaw → (x, y, z, w). Equivalent to tf.transformations.quaternion_from_euler(0, 0, yaw)."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))
