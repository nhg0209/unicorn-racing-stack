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

# Pin BLAS/OpenMP to ONE thread BEFORE numpy loads. OpenBLAS otherwise spawns a large pool
# that, on the tiny windowed-QP matrices, pegs every core for ~1s per solve and starves the
# sim + control loop (whole-system stutter whenever an obstacle triggers a re-opt). setdefault
# so an explicit launch <env> still wins; this covers `ros2 run` / missing-env launches too.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from contextlib import redirect_stdout
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String, Float32, Bool
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
        self.declare_parameter("obs_margin", 0.20)
        self.declare_parameter("default_obs_radius", 0.15)
        self.declare_parameter("republish_period", 1.0)     # [s] keep-alive republish
        self.declare_parameter("swap_at_startfinish", True)
        self.declare_parameter("swap_timeout_s", 4.0)        # force swap if no s-wrap seen
        self.declare_parameter("obs_change_tol", 0.10)       # [m] min move to count as a change
        self.declare_parameter("compute_sp", True)
        # re-optimization method:
        #   "local_window" (default) — fast ONLINE windowed min-curvature QP stitched into the
        #                    clean raceline (ms/solve); the intended lap-2+ obstacle-aware line.
        #   "global"       — legacy whole-track mincurv_iqp (minutes/solve; offline only).
        self.declare_parameter("reopt_method", "local_window")
        # The avoidance is a smooth WIDE arc: a smootherstep bump peaking at the required clearance
        # over a half-width R = clip(reach_time * local_speed, reach_min, reach_max). Bigger R =
        # gentler, faster arc that reaches toward the adjacent corners (carries speed, steers less).
        self.declare_parameter("reach_time", 1.2)            # [s] arc half-width ~ this * local speed
        self.declare_parameter("reach_min", 4.0)             # [m] MIN arc half-width (slow sections)
        self.declare_parameter("reach_max", 10.0)            # [m] MAX arc half-width (fast sections)
        self.declare_parameter("qp_veh_width", 0.30)         # [m] vehicle width for the corridor/
                                                             # clearance floor (obstacle clearance
                                                             # is obs_margin, separate)
        self.declare_parameter("wall_margin", 0.12)          # [m] keep the arc this far off the
                                                             # track walls (larger = safer, but tight
                                                             # sections may go infeasible)

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
        self.reopt_method = str(self.get_parameter("reopt_method").value)
        self.reach_time = float(self.get_parameter("reach_time").value)
        self.reach_min = float(self.get_parameter("reach_min").value)
        self.reach_max = float(self.get_parameter("reach_max").value)
        self.qp_veh_width = float(self.get_parameter("qp_veh_width").value)
        self.wall_margin = float(self.get_parameter("wall_margin").value)

        self.input_path = os.path.join(
            get_package_share_directory("stack_master"), "config", self.racecar_version)

        # --- load the clean bundle (guaranteed-valid fallback + no-obstacle output) --------
        if not self.map_name:
            raise RuntimeError("static_reopt_node requires the 'map' parameter")
        self.clean_bundle = self._load_clean_bundle(self.map_name)
        self.reftrack = core.load_reftrack(
            os.path.join(get_package_share_directory("stack_master"), "maps",
                         self.map_name, "centerline.csv"))
        # clean raceline as arrays for the windowed QP (x,y + dist-to-bounds + speed)
        gw = self.clean_bundle.glb_wpnts.wpnts
        self._clean_xy = np.array([[w.x_m, w.y_m] for w in gw], dtype=float)
        self._clean_dr = np.array([w.d_right for w in gw], dtype=float)
        self._clean_dl = np.array([w.d_left for w in gw], dtype=float)
        self._clean_vx = np.array([w.vx_mps for w in gw], dtype=float)
        self._clean_kappa = np.array([w.kappa_radpm for w in gw], dtype=float)
        self.get_logger().info(
            f"[static_reopt] loaded clean bundle ({len(gw)} raceline pts) + reftrack "
            f"({self.reftrack.shape[0]} pts) for '{self.map_name}'; reopt_method={self.reopt_method}")

        # --- state ------------------------------------------------------------------------
        # BATCH workflow: obstacles are just COLLECTED during a lap; the re-opt is solved ONCE and
        # swapped at the start/finish crossing (over ALL confirmed obstacles). No incremental solve.
        # single-threaded rclpy executor: obstacles_cb / frenet_cb / republish_cb never run
        # concurrently, and the batch re-opt is synchronous, so no lock is needed.
        self.active = self.clean_bundle         # what is published (always a valid line)
        self._obstacles: List[core.Obstacle] = []      # currently confirmed obstacle set
        self._obstacles_dirty = False           # set changed since the last build -> rebuild at s/f
        # publish the bundle only when the active line CHANGES (+ a slow keep-alive), NOT on every
        # timer tick. Topics are latched (TRANSIENT_LOCAL) so late subscribers still get the last
        # line; a fast periodic republish makes frenet_conversion / sector_tuner re-process the
        # raceline constantly, glitching the frenet odom (wobble). This keeps churn below 2 s.
        self._last_published: Optional[_Bundle] = None
        self._last_publish_t = 0.0
        # after a swap, re-publish `update_map` for a few ticks so sector_tuner / ot_interpolator
        # (which cache the FIRST /global_waypoints, then only rescale velocity) re-take the NEW
        # geometry -> /global_waypoints_scaled actually follows the swapped line.
        self._notify_scaler_ticks = 0

        self._last_s = None
        self._last_frenet_t = None              # wall time of the last frenet msg (stale fallback)
        self._dirty_since = 0.0                 # wall time the set went dirty (stale-frenet fallback)

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
        # tells sector_tuner (and any consumer that caches the first global line) to re-take
        # the new geometry after a swap. Relative topic "update_map" == sector_tuner's sub.
        self.pub_update_map = self.create_publisher(Bool, "update_map", 10)

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
        if self.reopt_method == "local_window":
            # Fast ONLINE windowed min-curvature QP stitched into the clean raceline.
            res = core.reoptimize_local_window(
                self._clean_xy, self._clean_dr, self._clean_dl, self.reftrack,
                obstacles, self.input_path,
                params=core.ModulationParams(obs_margin=self.obs_margin),
                w_veh=self.qp_veh_width, clean_vx=self._clean_vx, wall_margin=self.wall_margin,
                reach_time=self.reach_time, reach_min=self.reach_min, reach_max=self.reach_max,
                clean_kappa=self._clean_kappa)
        else:
            # Legacy whole-track mincurv_iqp (offline-grade; minutes/solve).
            res = core.reoptimize_with_obstacles(
                self.reftrack, obstacles, self.input_path,
                params=core.ModulationParams(obs_margin=self.obs_margin),
                safety_width=self.safety_width, safety_width_sp=self.safety_width_sp,
                compute_sp=self.compute_sp)

        traj, br, bl, est = res["main"]
        if "d_right" in res:                    # local_window returns exact widths
            d_r, d_l = res["d_right"], res["d_left"]
        else:
            d_r, d_l = core.dist_to_bounds(traj[:, 1:3], br, bl)  # traj is [s,x,y,...]
        glb_w, glb_m = core.build_wpnts(traj, d_r, d_l, second_traj=False)

        if self.compute_sp and "sp" in res:
            sp_traj, sbr, sbl, sp_est = res["sp"]
            sd_r, sd_l = core.dist_to_bounds(sp_traj[:, 1:3], sbr, sbl)
            sp_w, sp_m = core.build_wpnts(sp_traj, sd_r, sd_l, second_traj=True)
        else:
            sp_w, sp_m = copy.deepcopy(self.clean_bundle.sp_wpnts), copy.deepcopy(self.clean_bundle.sp_markers)

        rep = res["report"]
        info = String()
        if self.reopt_method == "local_window":
            nf = res.get("n_failed", 0)
            info.data = (f"[static_reopt] obstacle-aware (local_window) est {est:.3f}s; "
                         f"{res.get('n_windows', 0)} window(s) re-optimized, {nf} too tight "
                         f"(reactive layer covers those); affected {rep.n_affected}")
            if nf:
                self.get_logger().warning(
                    f"[static_reopt] {nf} obstacle window(s) too tight for the global line — "
                    f"the reactive static-avoidance layer must handle them")
        else:
            info.data = (f"[static_reopt] obstacle-aware (mincurv_iqp) est {est:.3f}s; "
                         f"affected {rep.n_affected}, infeasible {rep.n_infeasible}, "
                         f"min_halfwidth {rep.min_halfwidth_seen:.3f}m")
        lap = Float32(); lap.data = float(est)

        # centerline + trackbounds are map-fixed -> reuse the clean ones
        return _Bundle(info, lap,
                       self.clean_bundle.cent_wpnts, self.clean_bundle.cent_markers,
                       glb_w, glb_m, sp_w, sp_m, self.clean_bundle.trackbounds)

    # ----------------------------------------------------------------------------------
    # obstacle input: COLLECT only (batch re-opt happens at the start/finish crossing)
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
        # Just RECORD the confirmed set; do NOT solve here. The batch re-opt runs once at the next
        # start/finish crossing (frenet_cb) with ALL obstacles -> one consistent line, no mid-lap churn.
        self._obstacles = obs
        self._obstacles_dirty = True
        self._dirty_since = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info(
            f"[static_reopt] obstacle set -> {len(obs)} obstacle(s); will batch re-opt at start/finish")

    def _obstacles_changed(self, new: List[core.Obstacle]) -> bool:
        if len(new) != len(self._obstacles):
            return True
        tol = self.obs_change_tol
        for a, b in zip(new, self._obstacles):
            if abs(a.x - b.x) > tol or abs(a.y - b.y) > tol or abs(a.r - b.r) > tol:
                return True
        return False

    def _rebuild_and_swap(self, reason: str):
        """Solve ONE re-opt over the whole confirmed obstacle set and swap immediately. Called at
        the start/finish crossing (car at s≈0 where reopt==clean, so the swap is jump-free). The
        solve is ~10 ms (BLAS-pinned) so running it here does not stall the loop."""
        obstacles = list(self._obstacles)
        self._obstacles_dirty = False
        try:
            if obstacles:
                with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                    bundle = self._build_obstacle_bundle(obstacles)
            else:
                bundle = self.clean_bundle
        except Exception as e:  # noqa: BLE001 — must never propagate to the executor
            self.get_logger().warn(
                f"[static_reopt] batch re-opt FAILED ({type(e).__name__}: {str(e)[:80]}); "
                f"keeping the current line — reactive planner handles the gap")
            return
        changed = bundle is not self.active
        self.active = bundle
        if changed:
            self._notify_scaler_ticks = 3        # tell caching consumers to re-take the new geometry
        kind = "CLEAN" if bundle is self.clean_bundle else f"OBSTACLE-AWARE ({len(obstacles)} obs)"
        self.get_logger().info(f"[static_reopt] batch re-opt -> swap to {kind} ({reason})")
        self._publish_active(bundle)             # publish now, don't wait for the republish tick
        if changed:
            self.pub_update_map.publish(Bool(data=True))   # /global_waypoints already latched above

    def _publish_active(self, active: "_Bundle"):
        now = self.get_clock().now().nanoseconds * 1e-9
        self._last_published = active
        self._last_publish_t = now
        self.pub_glb.publish(active.glb_wpnts)
        self.pub_glb_m.publish(active.glb_markers)
        self.pub_sp.publish(active.sp_wpnts)
        self.pub_sp_m.publish(active.sp_markers)
        self.pub_cent.publish(active.cent_wpnts)
        self.pub_cent_m.publish(active.cent_markers)
        self.pub_bounds.publish(active.trackbounds)
        self.pub_info.publish(active.map_info)
        self.pub_lap.publish(active.est_lap_time)

    # ----------------------------------------------------------------------------------
    # start/finish crossing -> batch re-opt + swap
    # ----------------------------------------------------------------------------------
    def frenet_cb(self, msg: Odometry):
        s = msg.pose.pose.position.x  # frenet s
        self._last_frenet_t = self.get_clock().now().nanoseconds * 1e-9
        crossed_sf = (self._last_s is not None and s < self._last_s - 1.0)  # s wrapped to ~0
        self._last_s = s
        if crossed_sf and self._obstacles_dirty:
            self._rebuild_and_swap("start/finish")

    def republish_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        # STALE-FRENET fallback: no odom -> we never see the s-wrap. Solve once so a headless /
        # bag run still gets the obstacle line. Never fires while the car is actually driving.
        frenet_stale = (self._last_frenet_t is None
                        or (now - self._last_frenet_t) > self.swap_timeout_s)
        if self._obstacles_dirty and frenet_stale and (now - self._dirty_since) > self.swap_timeout_s:
            self._rebuild_and_swap("frenet stale fallback")

        active = self.active
        # publish ONLY when the line changed, or as a slow keep-alive (>= 5 s). No churn.
        if (active is not self._last_published) or (now - self._last_publish_t) >= 5.0:
            self._publish_active(active)
        if self._notify_scaler_ticks > 0:
            self._notify_scaler_ticks -= 1
            self.pub_update_map.publish(Bool(data=True))


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
