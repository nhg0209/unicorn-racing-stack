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


class StaticObstacleLayer(Node):
    def __init__(self):
        super().__init__("static_obstacle_layer")

        self.declare_parameter("static_vel_thresh", 0.3)     # [m/s] |vs|,|vd| below -> static-like
        self.declare_parameter("require_is_static", False)   # also require perception's is_static
        self.declare_parameter("match_radius", 0.5)          # [m] associate detection to a track
        self.declare_parameter("confirm_hits", 5)            # sightings to confirm a static obstacle
        self.declare_parameter("min_radius", 0.10)           # [m] floor on obstacle radius
        self.declare_parameter("obs_horizon_m", 6.0)         # [m] range within which the ego can see
        self.declare_parameter("removal_enable", True)
        self.declare_parameter("removal_miss_laps", 2)       # consecutive missed laps -> removed
        self.declare_parameter("removal_min_lap", 0)         # fixed-schedule prior: no removal before
        self.declare_parameter("publish_period", 0.5)        # [s] republish confirmed set

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

        self._tracks: List[_Track] = []
        self._next_marker_id = 0
        self._ego_s: Optional[float] = None
        self._last_ego_s: Optional[float] = None
        self._lap = 0
        self._track_length: Optional[float] = None

        self.pub = self.create_publisher(MarkerArray, "/static_reopt/obstacles", 10)
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacles_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.frenet_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.glb_cb, 10)
        self.create_timer(self.publish_period, self.publish_cb)

        self.get_logger().info("[static_obs_layer] up — mapping static obstacles from /tracking/obstacles")

    # ----------------------------------------------------------------------------------
    def glb_cb(self, msg: WpntArray):
        if msg.wpnts:
            self._track_length = msg.wpnts[-1].s_m

    def frenet_cb(self, msg: Odometry):
        s = msg.pose.pose.position.x
        self._ego_s = s
        # lap wrap detection (s resets to ~0)
        if self._last_ego_s is not None and s < self._last_ego_s - 1.0:
            self._on_lap_complete()
            self._lap += 1
        self._last_ego_s = s

        # mark observation opportunities: obstacle is within sensor range ahead of the ego
        if self._track_length:
            for t in self._tracks:
                gap = (t.s - s) % self._track_length
                if gap < self.obs_horizon_m:
                    t.opportunity_this_lap = True

    def obstacles_cb(self, msg: ObstacleArray):
        for obs in msg.obstacles:
            if getattr(obs, "is_actually_a_gap", False):
                continue
            if not self._is_static_like(obs):
                continue
            r = max(obs.size / 2.0, self.min_radius)
            self._associate(obs.x_m, obs.y_m, r, obs.s_center)

    def _is_static_like(self, obs) -> bool:
        slow = abs(obs.vs) < self.static_vel_thresh and abs(obs.vd) < self.static_vel_thresh
        if self.require_is_static:
            return slow and bool(obs.is_static)
        return slow

    def _associate(self, x: float, y: float, r: float, s: float):
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
            return
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
