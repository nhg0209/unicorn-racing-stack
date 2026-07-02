"""
static_reopt_node.py — obstacle-aware global-waypoint republisher (IFAC static obstacles).

Runtime replacement for gb_optimizer's global_trajectory_publisher during obstacle races.
It holds the CLEAN raceline bundle (read once from global_waypoints.json) and, when static
obstacles are present, a re-optimized OBSTACLE-AWARE bundle (main mincurv_iqp + shortest_path
SP, both from the width-modulated reftrack), and republishes whichever is active on the exact
same topics the rest of the stack already consumes. gb_optimizer itself is untouched — only
its library functions are imported (see static_reopt_core, memory: project-ifac-static-reopt).

HARD GUARANTEE — it must NEVER stop publishing a valid raceline:
  * re-optimization runs in a BACKGROUND thread, so the republish timer never blocks;
  * ANY re-opt failure (infeasible QP, exception) is swallowed — the node keeps the last
    valid bundle (obstacle-aware if one was ever built for the current obstacles, else clean);
  * the timer only ever publishes a fully-built bundle held under a lock.
So even a track-blocking obstacle degrades to "publish the previous good line" (the reactive
planner then handles what the global line cannot) — it never emits nothing or a broken line.

Obstacle input (phase 1, before the perception static-layer exists): a MarkerArray on
`/static_reopt/obstacles`; each marker -> Obstacle(x, y, r=max(scale.x,scale.y)/2 or the
`default_obs_radius` param). An empty array clears obstacles. This is trivial to drive from a
test script or RViz and will later be fed by the static-obstacle layer.

Line swap happens at start/finish (frenet s wrap) so the two closed lines coincide -> C2
continuity; a timeout forces the swap if no frenet odom is seen (bench testing).
"""

import copy
import os
import threading
from contextlib import redirect_stdout
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from f110_msgs.msg import WpntArray

from .readwrite_global_waypoints import read_global_waypoints
from . import static_reopt_core as core


class _Bundle:
    """The full set of messages the republisher emits (mirrors global_trajectory_publisher)."""

    __slots__ = (
        "map_info", "est_lap_time",
        "cent_wpnts", "cent_markers",
        "glb_wpnts", "glb_markers",
        "sp_wpnts", "sp_markers",
        "trackbounds",
    )

    def __init__(self, map_info, est_lap_time, cent_wpnts, cent_markers,
                 glb_wpnts, glb_markers, sp_wpnts, sp_markers, trackbounds):
        self.map_info = map_info
        self.est_lap_time = est_lap_time
        self.cent_wpnts = cent_wpnts
        self.cent_markers = cent_markers
        self.glb_wpnts = glb_wpnts
        self.glb_markers = glb_markers
        self.sp_wpnts = sp_wpnts
        self.sp_markers = sp_markers
        self.trackbounds = trackbounds


class StaticReoptNode(Node):
    def __init__(self):
        super().__init__("static_reopt_node")

        self.declare_parameter("map", "")
        self.declare_parameter("racecar_version", "SIM")
        self.declare_parameter("safety_width", 0.5)
        self.declare_parameter("safety_width_sp", 0.5)
        # obs_margin: extra clearance for the re-optimized line. Also prevents DOUBLE
        # AVOIDANCE (design A) — must exceed gb_ego_width_m/2 - safety_width/2 so the
        # reactive planner does not re-avoid obstacles already handled by the global line.
        # Co-tune with the (currently untuned) reactive static avoidance.
        self.declare_parameter("obs_margin", 0.15)
        self.declare_parameter("default_obs_radius", 0.15)
        self.declare_parameter("republish_period", 1.0)     # [s] keep-alive republish
        self.declare_parameter("swap_at_startfinish", True)
        self.declare_parameter("swap_timeout_s", 4.0)        # force swap if no s-wrap seen
        self.declare_parameter("obs_change_tol", 0.10)       # [m] min move to count as a change
        self.declare_parameter("compute_sp", True)

        self.map_name = self.get_parameter("map").value
        self.racecar_version = self.get_parameter("racecar_version").value
        self.safety_width = float(self.get_parameter("safety_width").value)
        self.safety_width_sp = float(self.get_parameter("safety_width_sp").value)
        self.obs_margin = float(self.get_parameter("obs_margin").value)
        self.default_obs_radius = float(self.get_parameter("default_obs_radius").value)
        self.republish_period = float(self.get_parameter("republish_period").value)
        self.swap_at_sf = bool(self.get_parameter("swap_at_startfinish").value)
        self.swap_timeout_s = float(self.get_parameter("swap_timeout_s").value)
        self.obs_change_tol = float(self.get_parameter("obs_change_tol").value)
        self.compute_sp = bool(self.get_parameter("compute_sp").value)

        self.input_path = os.path.join(
            get_package_share_directory("stack_master"), "config", self.racecar_version)

        # --- load the clean bundle (guaranteed-valid fallback + no-obstacle output) --------
        if not self.map_name:
            raise RuntimeError("static_reopt_node requires the 'map' parameter")
        self.clean_bundle = self._load_clean_bundle(self.map_name)
        self.reftrack = core.load_reftrack(
            os.path.join(get_package_share_directory("stack_master"), "maps",
                         self.map_name, "centerline.csv"))
        self.get_logger().info(
            f"[static_reopt] loaded clean bundle + reftrack ({self.reftrack.shape[0]} pts) for '{self.map_name}'")

        # --- state ------------------------------------------------------------------------
        self._lock = threading.Lock()          # guards active/target/obstacle_bundle
        self.active = self.clean_bundle         # what the timer publishes (always valid)
        self.target = self.clean_bundle         # what we want to swap TO at start/finish
        self.obstacle_bundle: Optional[_Bundle] = None  # last valid obstacle-aware bundle
        self.swap_pending = False
        self.swap_request_time = 0.0

        self._obstacles: List[core.Obstacle] = []      # current requested obstacle set
        self._reopt_running = False
        self._reopt_dirty = False               # obstacle set changed while a re-opt ran
        self._reopt_lock = threading.Lock()

        self._last_s = None

        # --- pub/sub ----------------------------------------------------------------------
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.pub_glb = self.create_publisher(WpntArray, "/global_waypoints", latched)
        self.pub_glb_m = self.create_publisher(MarkerArray, "/global_waypoints/markers", latched)
        self.pub_sp = self.create_publisher(WpntArray, "/global_waypoints/shortest_path", latched)
        self.pub_sp_m = self.create_publisher(MarkerArray, "/global_waypoints/shortest_path/markers", latched)
        self.pub_cent = self.create_publisher(WpntArray, "/centerline_waypoints", latched)
        self.pub_cent_m = self.create_publisher(MarkerArray, "/centerline_waypoints/markers", latched)
        self.pub_bounds = self.create_publisher(MarkerArray, "/trackbounds/markers", latched)
        self.pub_info = self.create_publisher(String, "/map_infos", latched)
        self.pub_lap = self.create_publisher(Float32, "/estimated_lap_time", latched)

        self.create_subscription(MarkerArray, "/static_reopt/obstacles", self.obstacles_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.frenet_cb, 10)

        self.create_timer(self.republish_period, self.republish_cb)
        self.get_logger().info("[static_reopt] up — publishing CLEAN line; awaiting obstacles")

    # ----------------------------------------------------------------------------------
    # bundle construction
    # ----------------------------------------------------------------------------------
    def _load_clean_bundle(self, map_name: str) -> _Bundle:
        (map_info, est, cent_m, cent_w, glb_m, glb_w, sp_m, sp_w, bounds) = read_global_waypoints(map_name)
        return _Bundle(map_info, est, cent_w, cent_m, glb_w, glb_m, sp_w, sp_m, bounds)

    def _build_obstacle_bundle(self, obstacles: List[core.Obstacle]) -> _Bundle:
        """Run the width-modulated re-optimization and assemble a full bundle. May raise;
        the caller runs this inside a try/except so a failure never reaches the timer."""
        res = core.reoptimize_with_obstacles(
            self.reftrack, obstacles, self.input_path,
            params=core.ModulationParams(obs_margin=self.obs_margin),
            safety_width=self.safety_width, safety_width_sp=self.safety_width_sp,
            compute_sp=self.compute_sp)

        traj, br, bl, est = res["main"]
        d_r, d_l = core.dist_to_bounds(traj, br, bl)
        glb_w, glb_m = core.build_wpnts(traj, d_r, d_l, second_traj=False)

        if self.compute_sp and "sp" in res:
            sp_traj, sbr, sbl, sp_est = res["sp"]
            sd_r, sd_l = core.dist_to_bounds(sp_traj, sbr, sbl)
            sp_w, sp_m = core.build_wpnts(sp_traj, sd_r, sd_l, second_traj=True)
        else:
            sp_w, sp_m = copy.deepcopy(self.clean_bundle.sp_wpnts), copy.deepcopy(self.clean_bundle.sp_markers)

        rep = res["report"]
        info = String()
        info.data = (f"[static_reopt] obstacle-aware (mincurv_iqp) est {est:.3f}s; "
                     f"affected {rep.n_affected}, infeasible {rep.n_infeasible}, "
                     f"min_halfwidth {rep.min_halfwidth_seen:.3f}m")
        lap = Float32(); lap.data = float(est)

        # centerline + trackbounds are map-fixed -> reuse the clean ones
        return _Bundle(info, lap,
                       self.clean_bundle.cent_wpnts, self.clean_bundle.cent_markers,
                       glb_w, glb_m, sp_w, sp_m, self.clean_bundle.trackbounds)

    # ----------------------------------------------------------------------------------
    # obstacle input -> background re-optimization
    # ----------------------------------------------------------------------------------
    def obstacles_cb(self, msg: MarkerArray):
        from visualization_msgs.msg import Marker
        obs: List[core.Obstacle] = []
        for m in msg.markers:
            # only ADD markers are obstacles; skip DELETE / DELETEALL housekeeping markers
            if getattr(m, "action", Marker.ADD) != Marker.ADD:
                continue
            r = max(m.scale.x, m.scale.y) / 2.0
            if r <= 1e-3:
                r = self.default_obs_radius
            obs.append(core.Obstacle(m.pose.position.x, m.pose.position.y, r))

        if not self._obstacles_changed(obs):
            return
        self._obstacles = obs
        self.get_logger().info(f"[static_reopt] obstacle set changed -> {len(obs)} obstacle(s); re-optimizing")
        self._kick_reopt()

    def _obstacles_changed(self, new: List[core.Obstacle]) -> bool:
        if len(new) != len(self._obstacles):
            return True
        tol = self.obs_change_tol
        for a, b in zip(new, self._obstacles):
            if abs(a.x - b.x) > tol or abs(a.y - b.y) > tol or abs(a.r - b.r) > tol:
                return True
        return False

    def _kick_reopt(self):
        """Start a background re-opt worker (or flag a rerun if one is already running)."""
        with self._reopt_lock:
            if self._reopt_running:
                self._reopt_dirty = True
                return
            self._reopt_running = True
        threading.Thread(target=self._reopt_worker, daemon=True).start()

    def _reopt_worker(self):
        while True:
            with self._lock:
                obstacles = list(self._obstacles)

            if not obstacles:
                # no obstacles -> target the clean line
                with self._lock:
                    self.obstacle_bundle = None
                    self.target = self.clean_bundle
                    self._request_swap_locked()
            else:
                try:
                    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                        bundle = self._build_obstacle_bundle(obstacles)
                    with self._lock:
                        self.obstacle_bundle = bundle
                        self.target = bundle
                        self._request_swap_locked()
                    self.get_logger().info("[static_reopt] obstacle-aware line ready; swap pending")
                except Exception as e:  # noqa: BLE001 — must never propagate
                    # keep whatever valid line we already have; do NOT change active
                    fallback = "previous obstacle line" if self.obstacle_bundle else "clean line"
                    self.get_logger().warn(
                        f"[static_reopt] re-opt FAILED ({type(e).__name__}: {str(e)[:80]}); "
                        f"keeping {fallback} — reactive planner handles the gap")
                    if self.obstacle_bundle is not None:
                        with self._lock:
                            self.target = self.obstacle_bundle
                            self._request_swap_locked()

            # loop again if the obstacle set changed while we were optimizing
            with self._reopt_lock:
                if self._reopt_dirty:
                    self._reopt_dirty = False
                    continue
                self._reopt_running = False
                return

    def _request_swap_locked(self):
        """Mark a swap pending. Caller must hold self._lock."""
        self.swap_pending = True
        self.swap_request_time = self.get_clock().now().nanoseconds * 1e-9

    # ----------------------------------------------------------------------------------
    # swap at start/finish (or timeout)
    # ----------------------------------------------------------------------------------
    def frenet_cb(self, msg: Odometry):
        s = msg.pose.pose.position.x  # frenet s
        with self._lock:
            crossed_sf = (self._last_s is not None and s < self._last_s - 1.0)  # s wrapped to ~0
            self._last_s = s
            if self.swap_pending and self.swap_at_sf and crossed_sf:
                self._do_swap_locked("start/finish")

    def republish_cb(self):
        with self._lock:
            # timeout / no-frenet fallback: force the swap so we never get stuck on a stale line
            if self.swap_pending:
                now = self.get_clock().now().nanoseconds * 1e-9
                if not self.swap_at_sf or (now - self.swap_request_time) > self.swap_timeout_s:
                    self._do_swap_locked("timeout" if self.swap_at_sf else "immediate")
            active = self.active

        # publish the active (always-valid) bundle
        self.pub_glb.publish(active.glb_wpnts)
        self.pub_glb_m.publish(active.glb_markers)
        self.pub_sp.publish(active.sp_wpnts)
        self.pub_sp_m.publish(active.sp_markers)
        self.pub_cent.publish(active.cent_wpnts)
        self.pub_cent_m.publish(active.cent_markers)
        self.pub_bounds.publish(active.trackbounds)
        self.pub_info.publish(active.map_info)
        self.pub_lap.publish(active.est_lap_time)

    def _do_swap_locked(self, reason: str):
        """Switch the active bundle to the target. Caller must hold self._lock."""
        if self.active is not self.target:
            kind = "CLEAN" if self.target is self.clean_bundle else "OBSTACLE-AWARE"
            self.get_logger().info(f"[static_reopt] swapping active line -> {kind} ({reason})")
        self.active = self.target
        self.swap_pending = False


def main(args=None):
    rclpy.init(args=args)
    node = StaticReoptNode()
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
