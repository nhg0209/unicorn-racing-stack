"""
static_obstacle_layer.py — persistent static-obstacle map for the IFAC re-optimization flow.

Sits between perception and static_reopt_node. It watches the tracking stream
(`/tracking/obstacles`, f110_msgs/ObstacleArray), keeps ONLY the static obstacles (filters out
the moving opponent and transient detections), maintains a persistent, confidence-tracked map
of them, decides when one has been removed, and republishes the confirmed set as a MarkerArray
on `/static_reopt/obstacles` for static_reopt_node to re-optimize around.

Two consumers of the raw tracking stream must be kept apart (see memory: project-ifac-static-reopt):
  * the reactive planner (predictive_spliner) keeps avoiding whatever perception reports — it is
    the safety net for not-yet-mapped obstacles AND for obstacles whose re-optimization failed;
  * this layer feeds ONLY the confirmed static set to the global re-optimizer.
Double-avoidance is handled geometrically (design A: obs_margin in the re-opt), not by
suppression here, so this layer does not need to touch the reactive path.

Add/confirm:  a detection is "static-like" if |vs|,|vd| are ~0 (and, optionally, is_static);
matched to an existing track by proximity; promoted to CONFIRMED after `confirm_hits` sightings.
Remove:       positive absence — only when the ego actually had a chance to see the spot. Each
lap, a confirmed obstacle that the ego passed within `obs_horizon_m` of but did NOT re-detect
counts as a miss; `removal_miss_laps` consecutive missed laps (and not before `removal_min_lap`,
the fixed-schedule prior) drop it. "Not detected" is only trusted when the ego was in range —
an occlusion by the opponent is never enough on its own.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import ObstacleArray, WpntArray


@dataclass
class _Track:
    x: float
    y: float
    r: float
    s: float
    hits: int = 0
    confirmed: bool = False
    seen_this_lap: bool = False
    opportunity_this_lap: bool = False
    miss_laps: int = 0
    marker_id: int = 0
    clear_streak: int = 0    # consecutive tracking msgs with a clear in-window view and no sighting


class StaticObstacleLayer(Node):
    def __init__(self):
        super().__init__("static_obstacle_layer")

        self.declare_parameter("static_vel_thresh", 0.3)     # [m/s] |vs|,|vd| below -> static-like
        # also require perception's is_static: ON by default now that the tracker flag is
        # position-persistence + window-speed hardened. Velocity alone (|vs|<thresh) also
        # latched a MOVING opponent whenever its EKF speed dipped (slow corner / spin-up),
        # letting a head-to-head opponent corrupt the obstacle-aware re-optimized line.
        self.declare_parameter("require_is_static", True)
        self.declare_parameter("match_radius", 0.5)          # [m] associate detection to a track
        self.declare_parameter("confirm_hits", 5)            # sightings to confirm a static obstacle
        self.declare_parameter("min_radius", 0.10)           # [m] floor on obstacle radius
        self.declare_parameter("obs_horizon_m", 6.0)         # [m] range within which the ego can see
        self.declare_parameter("removal_enable", True)
        # One genuinely-missed full lap is strong evidence now that lap counting is guarded by
        # the forward-progress odometer (a spurious seam crossing can no longer charge a miss).
        self.declare_parameter("removal_miss_laps", 1)       # consecutive missed laps -> removed
        self.declare_parameter("removal_min_lap", 0)         # fixed-schedule prior: no removal before
        self.declare_parameter("publish_period", 0.5)        # [s] republish confirmed set
        # Sighting-based fast unlatch: while a confirmed track sits inside a HIGH-CONFIDENCE view
        # window ahead of the ego (tighter than obs_horizon_m), every tracking message that shows
        # NO visible detection there grows a streak; at unlatch_clear_msgs the track is dropped on
        # the spot — no waiting for the lap accounting. A premature unlatch is survivable (the
        # reactive layer still avoids whatever perception reports, and the layer re-confirms), so
        # this can be aggressive. The streak must complete within one approach (reset outside the
        # window) and is SUSPENDED while a dynamic obstacle (the opponent) sits between the ego
        # and the spot — the main occlusion-driven false-unlatch vector in head-to-head.
        self.declare_parameter("unlatch_enable", True)
        self.declare_parameter("unlatch_gap_min", 1.0)       # [m] window near edge
        self.declare_parameter("unlatch_gap_max", 5.0)       # [m] window far edge (< obs_horizon_m,
                                                             # sized so a fast pass still fits the streak)
        self.declare_parameter("unlatch_clear_msgs", 20)     # consecutive clear msgs -> unlatch (~0.5 s @40 Hz)
        # Streak is also SUSPENDED while the ego is OFF the raceline (|d| above this): mid-avoidance
        # the sensor geometry on the very obstacle being avoided is skewed (wall-adjacent boxes
        # flicker under detect's boundary inflation at those angles) — observed: a live obstacle
        # was unlatched DURING its own avoidance, then re-confirmed 0.2 s later (set flap 1->0->1).
        self.declare_parameter("unlatch_max_ego_d", 0.20)    # [m] suspend streak when |ego d| exceeds

        self.static_vel_thresh = float(self.get_parameter("static_vel_thresh").value)
        self.require_is_static = bool(self.get_parameter("require_is_static").value)
        self.match_radius = float(self.get_parameter("match_radius").value)
        self.confirm_hits = int(self.get_parameter("confirm_hits").value)
        self.min_radius = float(self.get_parameter("min_radius").value)
        self.obs_horizon_m = float(self.get_parameter("obs_horizon_m").value)
        self.removal_enable = bool(self.get_parameter("removal_enable").value)
        self.removal_miss_laps = int(self.get_parameter("removal_miss_laps").value)
        self.removal_min_lap = int(self.get_parameter("removal_min_lap").value)
        self.publish_period = float(self.get_parameter("publish_period").value)
        self.unlatch_enable = bool(self.get_parameter("unlatch_enable").value)
        self.unlatch_gap_min = float(self.get_parameter("unlatch_gap_min").value)
        self.unlatch_gap_max = float(self.get_parameter("unlatch_gap_max").value)
        self.unlatch_clear_msgs = int(self.get_parameter("unlatch_clear_msgs").value)
        self.unlatch_max_ego_d = float(self.get_parameter("unlatch_max_ego_d").value)

        self._ego_d: Optional[float] = None
        self._tracks: List[_Track] = []
        self._next_marker_id = 0
        self._ego_s: Optional[float] = None
        self._last_ego_s: Optional[float] = None
        self._lap = 0
        self._s_progressed = 0.0
        self._track_length: Optional[float] = None

        self.pub = self.create_publisher(MarkerArray, "/static_reopt/obstacles", 10)
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacles_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.frenet_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.glb_cb, 10)
        # Pit/bench reset: `ros2 topic pub --once /static_reopt/clear_obstacles std_msgs/msg/Empty`
        # drops every track at once (e.g. the obstacles were physically removed mid-race).
        self.create_subscription(Empty, "/static_reopt/clear_obstacles", self.clear_cb, 1)
        self.create_timer(self.publish_period, self.publish_cb)

        self.get_logger().info("[static_obs_layer] up — mapping static obstacles from /tracking/obstacles")

    # ----------------------------------------------------------------------------------
    def glb_cb(self, msg: WpntArray):
        if msg.wpnts:
            self._track_length = msg.wpnts[-1].s_m

    def frenet_cb(self, msg: Odometry):
        s = msg.pose.pose.position.x
        self._ego_s = s
        self._ego_d = msg.pose.pose.position.y   # unlatch streak suspends while off-line
        # Lap boundary needs the previous sample near the end of the lap AND a full lap of
        # accumulated forward travel (mirrors static_reopt_node's gate): a parked car whose s
        # flickers across the seam, a reversing car, or a localization jump must never count a
        # lap — each spurious lap can charge a miss against every unseen obstacle. Without the
        # track length no laps are counted, which just keeps removal disabled (safe).
        L = self._track_length
        if L:
            if self._last_ego_s is not None:
                ds = s - self._last_ego_s
                if ds < -1.0:
                    ds += L                   # a genuine wrap is one lap of forward travel
                if 0.0 <= ds < 1.0:           # ignore backward/teleport frames in the odometer
                    self._s_progressed += ds
            if (self._last_ego_s is not None
                    and s < self._last_ego_s - 1.0
                    and self._last_ego_s > 0.85 * L
                    and self._s_progressed > 0.9 * L):
                self._on_lap_complete()
                self._lap += 1
                self._s_progressed = 0.0
        self._last_ego_s = s

        # mark observation opportunities: obstacle is within sensor range ahead of the ego
        if self._track_length:
            for t in self._tracks:
                gap = (t.s - s) % self._track_length
                if gap < self.obs_horizon_m:
                    t.opportunity_this_lap = True

    def obstacles_cb(self, msg: ObstacleArray):
        seen_now = set()      # tracks matched to a VISIBLE detection in this message
        for obs in msg.obstacles:
            if getattr(obs, "is_actually_a_gap", False):
                continue
            if not self._is_static_like(obs):
                continue
            r = max(obs.size / 2.0, self.min_radius)
            t = self._associate(obs.x_m, obs.y_m, r, obs.s_center)
            # a remembered-but-unseen obstacle (is_visible=False) is tracker memory, not an
            # observation — it must not defeat the clear-view streak below
            if t is not None and getattr(obs, "is_visible", True):
                seen_now.add(id(t))
        self._update_unlatch_streaks(msg, seen_now)

    def _is_static_like(self, obs) -> bool:
        slow = abs(obs.vs) < self.static_vel_thresh and abs(obs.vd) < self.static_vel_thresh
        if self.require_is_static:
            return slow and bool(obs.is_static)
        return slow

    def _associate(self, x: float, y: float, r: float, s: float) -> _Track:
        best = None
        best_d = self.match_radius
        for t in self._tracks:
            d = math.hypot(t.x - x, t.y - y)
            if d < best_d:
                best_d = d
                best = t
        if best is None:
            t = _Track(x=x, y=y, r=r, s=s, hits=1, marker_id=self._next_marker_id)
            self._next_marker_id += 1
            self._tracks.append(t)
            return t
        # EMA update of position/size; count the sighting
        a = 0.3
        best.x = (1 - a) * best.x + a * x
        best.y = (1 - a) * best.y + a * y
        best.r = max((1 - a) * best.r + a * r, self.min_radius)
        best.s = s
        best.hits += 1
        best.seen_this_lap = True
        best.miss_laps = 0
        if not best.confirmed and best.hits >= self.confirm_hits:
            best.confirmed = True
            self.get_logger().info(
                f"[static_obs_layer] CONFIRMED static obstacle @({best.x:.2f},{best.y:.2f}) r={best.r:.2f}")
        return best

    def _update_unlatch_streaks(self, msg: ObstacleArray, seen_now):
        """Sighting-based fast unlatch (see the unlatch_* parameter block)."""
        if not self.unlatch_enable or self._ego_s is None or not self._track_length:
            return
        L = self._track_length
        # forward gaps of dynamic obstacles (the opponent) — occlusion guard input
        dyn_gaps = [(o.s_center - self._ego_s) % L
                    for o in msg.obstacles
                    if not getattr(o, "is_actually_a_gap", False) and not self._is_static_like(o)]
        survivors: List[_Track] = []
        for t in self._tracks:
            if not t.confirmed:
                survivors.append(t)
                continue
            gap = (t.s - self._ego_s) % L
            if not (self.unlatch_gap_min <= gap <= self.unlatch_gap_max):
                t.clear_streak = 0            # the streak must complete within ONE approach
                survivors.append(t)
                continue
            if id(t) in seen_now:
                t.clear_streak = 0
                survivors.append(t)
                continue
            if self._ego_d is not None and abs(self._ego_d) > self.unlatch_max_ego_d:
                survivors.append(t)           # ego mid-avoidance (off-line): geometry unreliable,
                continue                      # suspend the streak, don't count a miss
            if any(g < gap for g in dyn_gaps):
                survivors.append(t)           # opponent between ego and the spot: suspend, don't count
                continue
            t.clear_streak += 1
            if t.clear_streak >= self.unlatch_clear_msgs:
                self.get_logger().info(
                    f"[static_obs_layer] UNLATCHED static obstacle @({t.x:.2f},{t.y:.2f}) — "
                    f"{t.clear_streak} consecutive clear views of its spot, no detection")
                continue                      # drop it now; no waiting for the lap accounting
            survivors.append(t)
        self._tracks = survivors

    def clear_cb(self, _msg: Empty):
        n = len(self._tracks)
        self._tracks = []
        self.get_logger().info(f"[static_obs_layer] CLEARED {n} track(s) by external request")
        self.publish_cb()  # push the empty set immediately, don't wait for the timer

    def _on_lap_complete(self):
        """At each lap boundary: update misses/removal for confirmed obstacles, reset flags."""
        survivors: List[_Track] = []
        for t in self._tracks:
            if t.confirmed and self.removal_enable and self._lap >= self.removal_min_lap:
                if t.opportunity_this_lap and not t.seen_this_lap:
                    t.miss_laps += 1
                elif t.seen_this_lap:
                    t.miss_laps = 0
                if t.miss_laps >= self.removal_miss_laps:
                    self.get_logger().info(
                        f"[static_obs_layer] REMOVED static obstacle @({t.x:.2f},{t.y:.2f}) "
                        f"— {t.miss_laps} missed laps (was cleared)")
                    continue  # drop it
            # unconfirmed candidates that never re-appear also decay away
            if not t.confirmed and t.opportunity_this_lap and not t.seen_this_lap:
                t.hits -= 1
                if t.hits <= 0:
                    continue
            t.seen_this_lap = False
            t.opportunity_this_lap = False
            survivors.append(t)
        self._tracks = survivors

    # ----------------------------------------------------------------------------------
    def publish_cb(self):
        arr = MarkerArray()
        # clear stale markers on the consumer, then publish the current confirmed set
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for t in self._tracks:
            if not t.confirmed:
                continue
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "static_reopt_obstacles"
            m.id = t.marker_id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = t.x
            m.pose.position.y = t.y
            m.pose.orientation.w = 1.0
            m.scale.x = 2.0 * t.r
            m.scale.y = 2.0 * t.r
            m.scale.z = 0.3
            m.color.a = 0.6
            m.color.r = 1.0
            m.color.g = 0.5
            arr.markers.append(m)
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = StaticObstacleLayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
