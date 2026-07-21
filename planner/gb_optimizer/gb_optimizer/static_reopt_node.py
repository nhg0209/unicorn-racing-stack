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
from collections import deque

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
from std_msgs.msg import String, Float32, Bool, Empty
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from f110_msgs.msg import WpntArray, OTWpntArray

from .readwrite_global_waypoints import read_global_waypoints
from . import static_reopt_core as core


class _Bundle:
    """The full set of messages the republisher emits (mirrors global_trajectory_publisher)."""

    __slots__ = (
        "map_info", "est_lap_time",
        "cent_wpnts", "cent_markers",
        "glb_wpnts", "glb_markers",
        "sp_wpnts", "sp_markers",
        "trackbounds", "n_apex",
    )

    def __init__(self, map_info, est_lap_time, cent_wpnts, cent_markers,
                 glb_wpnts, glb_markers, sp_wpnts, sp_markers, trackbounds, n_apex=0):
        # n_apex = humps ACTUALLY laid into this line. 0 means the geometry is the clean raceline
        # even if obstacles were passed in (no apex recorded, or the corridor could not hold it).
        self.n_apex = n_apex
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
        # AVOIDANCE (design A) — must be >= the reactive keep-out (width_car/2 + safety_margin
        # in static_avoidance_params.yaml, currently 0.15+0.15=0.30) or the re-opt line sits
        # inside the reactive planner's own keep-out and gets re-avoided every lap. Verify with
        # stack_master/scripts/check_avoidance_margins.py after tuning either side.
        self.declare_parameter("obs_margin", 0.35)
        self.declare_parameter("default_obs_radius", 0.15)
        self.declare_parameter("republish_period", 1.0)     # [s] keep-alive republish
        # `update_map` re-notify period. MUST stay below sector_tuner's 0.5 s scale timer so a
        # consumer that consumed the flag before it had the new line re-takes it within one of
        # its own ticks (otherwise it publishes the OLD geometry until the next notify).
        self.declare_parameter("notify_period", 0.2)        # [s] update_map re-notify
        self.declare_parameter("notify_ticks", 10)          # re-notify count after a swap (~2 s)
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
        self.declare_parameter("reach_time", 1.2)            # [s] DEPRECATED (lap-time search owns
                                                             # the reach; kept for param compat)
        self.declare_parameter("reach_min", 1.0)             # [m] MIN arc half-width SEARCHED
                                                             # (must stay below the locality cap
                                                             # ~0.1*lap or the search is empty)
        self.declare_parameter("reach_max", 10.0)            # [m] MAX arc half-width searched
        self.declare_parameter("qp_veh_width", 0.30)         # [m] vehicle width for the corridor/
                                                             # clearance floor (obstacle clearance
                                                             # is obs_margin, separate)
        self.declare_parameter("wall_margin", 0.05)          # [m] keep the arc this far off the
                                                             # track walls. Aligned with the reactive
                                                             # spliner's wall reserve: the recorded
                                                             # apex was DRIVEN at that reserve, and a
                                                             # bigger one here shrank the hump below
                                                             # the proven clearance (all-or-nothing
                                                             # fit then rejects it).

        self.map_name = self.get_parameter("map").value
        self.racecar_version = self.get_parameter("racecar_version").value
        self.safety_width = float(self.get_parameter("safety_width").value)
        self.safety_width_sp = float(self.get_parameter("safety_width_sp").value)
        self.obs_margin = float(self.get_parameter("obs_margin").value)
        self.default_obs_radius = float(self.get_parameter("default_obs_radius").value)
        self.republish_period = float(self.get_parameter("republish_period").value)
        self.notify_period = float(self.get_parameter("notify_period").value)
        self.notify_ticks = int(self.get_parameter("notify_ticks").value)
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
        # RECORDED reactive-spline apexes (design B): on the exploration lap the reactive spliner
        # publishes the avoidance path it actually drives; per confirmed obstacle we keep the
        # map-frame apex point (the spline point beside it, max |d| over the lap). The re-opt then
        # RESHAPES that apex into the global line (keep apex, grow gentle ramps). key = quantized
        # obstacle (x,y); value = (x_apex, y_apex, |d_apex|).
        self._apex_by_obs = {}
        self._apex_assoc_tol = 2.0              # [m] max obstacle->spline-point dist to associate
        self._apex_min_d = 0.05                 # [m] ignore near-raceline paths (not avoiding)
        # Apex records are keyed by the UPSTREAM track id (static_obstacle_layer publishes a stable
        # marker_id per track, assigned once at creation and kept across its EMA position updates).
        # The old key quantized (x,y) on a 0.2 m grid, which is SMALLER than obs_change_tol (0.25 m):
        # every position update that passed the change gate necessarily landed in a different bin
        # and so DELETED that obstacle's apex. Combined with detection flicker (one empty frame
        # wiped every record) that is why the line reverted to clean after a lap.
        self._obs_ids: List[int] = []           # marker ids, index-aligned with _obstacles
        self._apex_miss = {}                    # key -> consecutive frames the obstacle was absent
        self.declare_parameter("apex_miss_frames", 20)   # tolerate this many missing frames
        self.apex_miss_frames = int(self.get_parameter("apex_miss_frames").value)
        # publish the bundle only when the active line CHANGES (+ a slow keep-alive), NOT on every
        # timer tick. Topics are latched (TRANSIENT_LOCAL) so late subscribers still get the last
        # line; a fast periodic republish makes frenet_conversion / sector_tuner re-process the
        # raceline constantly, glitching the frenet odom (wobble). This keeps churn below 2 s.
        self._last_published: Optional[_Bundle] = None
        self._last_publish_t = 0.0
        # after a swap, re-publish `update_map` for a few ticks so sector_tuner / ot_interpolator
        # (which cache the FIRST /global_waypoints, then only rescale velocity) re-take the NEW
        # geometry -> /global_waypoints_scaled actually follows the swapped line.
        # The notification runs on its OWN fast timer (notify_period), NOT the 1 s keep-alive:
        # sector_tuner re-takes the geometry only when its 0.5 s scale timer happens to see the
        # flag set, so a 1 s re-notify is SLOWER than the consumer it is trying to reach and can
        # leave it publishing the OLD geometry for a whole second after the swap.
        self._notify_scaler_ticks = 0

        self._last_s = None
        self._last_frenet_t = None              # wall time of the last frenet msg (stale fallback)
        self._dirty_since = 0.0                 # wall time the set went dirty (stale-frenet fallback)
        # LAP GATING (S2). `s < last_s - 1.0` alone fires on ANY backward jump of s, not only a real
        # wrap: a localization correction, or the frenet projection flipping to another nearest
        # segment near the seam, both trigger it — and because the seam is the MAP's s=0, not where
        # the car started, a car launched at s=20 m "completed" the exploration lap after 15 m.
        # Require BOTH a genuine seam crossing AND a full lap of forward progress.
        self._track_len = float(self.clean_bundle.glb_wpnts.wpnts[-1].s_m)
        self._s_progressed = 0.0
        # A swap is only jump-free where the two lines COINCIDE. The offset seam sits in the largest
        # apex-free gap, not at s=0, so swapping blindly at s=0 can step the reference laterally.
        # Solve at the lap boundary, then commit at the first station where they agree.
        self._pending: Optional[_Bundle] = None
        self._pending_dev = None                # per-station |pending - active| [m]
        # REACTIVE-SPLINE state. The swap must NOT land while the car is inside (or about to enter)
        # a reactive avoidance: waiting for s=0 meant the line could switch right next to the NEXT
        # obstacle, and the car then failed to avoid it. The reactive planner publishes a non-empty
        # path exactly while an obstacle is inside its planning horizon, so "reactive idle" is a
        # direct read of "no obstacle near, safe to change the line under the planner".
        self._reactive_active = False
        self._reactive_idle_t = 0.0             # wall time the reactive path went idle
        self.declare_parameter("swap_idle_s", 0.3)     # [s] reactive must stay idle this long
        self.swap_idle_s = float(self.get_parameter("swap_idle_s").value)
        # SWAP HORIZON: the lines must agree not only where the car IS but over the stretch the
        # controller is about to look at — committing right at a hump entrance changed the line
        # inside the lookahead and jerked the car. Horizon = max(min_m, time_s * current speed).
        self.declare_parameter("swap_horizon_min_m", 3.0)
        self.declare_parameter("swap_horizon_time_s", 1.0)
        self.swap_horizon_min_m = float(self.get_parameter("swap_horizon_min_m").value)
        self.swap_horizon_time_s = float(self.get_parameter("swap_horizon_time_s").value)
        self._last_vs = 0.0                     # frenet forward speed (horizon scaling)
        # SWAP DEADLOCK BREAKER: a car stuck TRAILING behind the obstacle sits INSIDE the pending
        # hump — the horizon gate can then never pass and the reactive flicker blocks the idle
        # gate, so the very swap that would free the car (the new line drives around the
        # obstacle) never lands. Once the pending has waited this long with the car slow, commit
        # anyway: at low speed the controller preview is short, so the mid-hump line change is
        # benign — and it un-sticks the car.
        self.declare_parameter("swap_deadlock_s", 5.0)       # [s] pending age before forcing
        self.declare_parameter("swap_deadlock_max_vs", 2.0)  # [m/s] only force while this slow
        # never force-commit a line the car is further off than this: a sane hump keeps the
        # car within ~apex distance of the new line; anything bigger means the pending geometry
        # is poisoned (observed: a ratcheted 1.41 m hump) and committing it strands the car.
        self.declare_parameter("swap_deadlock_max_dev", 0.6)
        self.swap_deadlock_s = float(self.get_parameter("swap_deadlock_s").value)
        self.swap_deadlock_max_vs = float(self.get_parameter("swap_deadlock_max_vs").value)
        self.swap_deadlock_max_dev = float(self.get_parameter("swap_deadlock_max_dev").value)
        self._pending_since = 0.0               # wall time the current pending was installed
        # RETRO-APEX buffer: recent reactive paths. A near-start obstacle is often CONFIRMED only
        # after the car already passed it (layer confirm latency) — the avoidance that was just
        # driven is gone from /planner/avoidance/static_otwpnts by then, so its apex was lost and
        # the obstacle-aware line slipped a whole lap. Replaying the buffer when the confirmed set
        # changes recovers that apex within the same lap.
        self.declare_parameter("apex_buffer_sec", 3.0)
        self.apex_buffer_sec = float(self.get_parameter("apex_buffer_sec").value)
        self._path_buffer = deque()             # (t, wx, wy, wd) of recent non-idle reactive paths
        # SOLVE RETRY BACKOFF: a failed/0-hump build keeps the dirty flag armed so the rebuild is
        # retried — but the retry tick is frenet_cb (40 Hz); without a backoff a persistently
        # infeasible corridor would re-solve every frame (~10 ms each) and starve the executor.
        self.declare_parameter("solve_retry_backoff_s", 1.0)
        self.solve_retry_backoff_s = float(self.get_parameter("solve_retry_backoff_s").value)
        self._solve_backoff_until = 0.0

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
        # reactive STATIC avoidance path (exploration lap) — records each obstacle's apex for the
        # reshape. The static_avoidance_planner is remapped to /planner/avoidance/static_otwpnts
        # (race.launch.xml); /planner/avoidance/otwpnts is the DYNAMIC (opponent) planner, wrong here.
        self.create_subscription(OTWpntArray, "/planner/avoidance/static_otwpnts", self.otwpnts_cb, 10)
        # Same pit/bench reset the layer listens to: drop the apex records right away instead of
        # letting them age out over apex_miss_frames publishes (the layer's empty set follows).
        self.create_subscription(Empty, "/static_reopt/clear_obstacles", self.clear_cb, 1)

        self.create_timer(self.republish_period, self.republish_cb)
        self.create_timer(self.notify_period, self.notify_cb)
        self.get_logger().info("[static_reopt] up — publishing CLEAN line; awaiting obstacles")
        # Bag-record the coupled margins: double-avoidance prevention depends on obs_margin
        # covering the reactive keep-out (see check_avoidance_margins.py).
        self.get_logger().info(
            f"[static_reopt] margins: obs_margin={self.obs_margin:.2f} wall_margin={self.wall_margin:.2f} "
            f"qp_veh_width={self.qp_veh_width:.2f} safety_width={self.safety_width:.2f}")

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
            # Reshape the recorded reactive apexes into the global line (keep apex, gentle ramps).
            apexes = self._apex_list(obstacles)
            res = core.reoptimize_local_window(
                self._clean_xy, self._clean_dr, self._clean_dl, self.reftrack,
                apexes, self.input_path,
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
            n_ap = res.get("n_windows", 0)
            dropped = res.get("apex_dropped", [])
            n_obs = len(obstacles)
            n_no_apex = max(0, n_obs - n_ap - len(dropped))
            info.data = (f"[static_reopt] obstacle-aware (apex reshape) est {est:.3f}s; "
                         f"{n_ap}/{n_obs} obstacle apex(es) reshaped, {len(dropped)} corridor-"
                         f"rejected, {n_no_apex} without a recorded reactive apex")
            if dropped:
                # All-or-nothing: a hump the corridor cannot hold at ~full amplitude would NOT
                # clear its obstacle — the shrunken line got re-avoided by the reactive layer
                # every lap. Those obstacles stay reactive-only, and this says so per apex.
                det = "; ".join(f"@({d['xy'][0]:.2f},{d['xy'][1]:.2f}) want {d['want']:+.2f} "
                                f"corridor max {d['fit']:+.2f}" for d in dropped)
                self.get_logger().warning(
                    f"[static_reopt] {len(dropped)} apex(es) CORRIDOR-REJECTED (track too tight "
                    f"there; reactive layer keeps handling them): {det}")
            if n_no_apex:
                self.get_logger().warning(
                    f"[static_reopt] {n_no_apex} obstacle(s) had no recorded reactive apex yet — "
                    f"the reactive static-avoidance layer handles them until an apex is captured")
        else:
            info.data = (f"[static_reopt] obstacle-aware (mincurv_iqp) est {est:.3f}s; "
                         f"affected {rep.n_affected}, infeasible {rep.n_infeasible}, "
                         f"min_halfwidth {rep.min_halfwidth_seen:.3f}m")
        lap = Float32(); lap.data = float(est)

        # centerline + trackbounds are map-fixed -> reuse the clean ones
        return _Bundle(info, lap,
                       self.clean_bundle.cent_wpnts, self.clean_bundle.cent_markers,
                       glb_w, glb_m, sp_w, sp_m, self.clean_bundle.trackbounds,
                       n_apex=int(res.get("n_windows", 0)))

    # ----------------------------------------------------------------------------------
    # obstacle input: COLLECT only (batch re-opt happens at the start/finish crossing)
    # ----------------------------------------------------------------------------------
    def obstacles_cb(self, msg: MarkerArray):
        from visualization_msgs.msg import Marker
        obs: List[core.Obstacle] = []
        ids: List[int] = []
        for m in msg.markers:
            # only ADD markers are obstacles; skip DELETE / DELETEALL housekeeping markers
            if getattr(m, "action", Marker.ADD) != Marker.ADD:
                continue
            r = max(m.scale.x, m.scale.y) / 2.0
            if r <= 1e-3:
                r = self.default_obs_radius
            obs.append(core.Obstacle(m.pose.position.x, m.pose.position.y, r))
            ids.append(int(getattr(m, "id", 0)))

        # Age out apex records on EVERY frame (not only on a "changed" frame): a track that is
        # missing for a single frame must not lose its apex, but one gone for good must.
        live = set(self._keys_for(obs, ids))
        for k in list(self._apex_by_obs):
            if k in live:
                self._apex_miss[k] = 0
            else:
                self._apex_miss[k] = self._apex_miss.get(k, 0) + 1
                if self._apex_miss[k] >= self.apex_miss_frames:
                    del self._apex_by_obs[k]
                    self._apex_miss.pop(k, None)

        if not self._obstacles_changed(obs):
            self._obs_ids = ids           # ids can be re-issued without the position changing
            self._adopt_orphan_apexes()   # re-issued ids must not orphan the apex records
            return
        # Just RECORD the confirmed set; do NOT solve here. The batch re-opt runs once at the next
        # start/finish crossing (frenet_cb) with ALL obstacles -> one consistent line, no mid-lap churn.
        self._obstacles = obs
        self._obs_ids = ids
        self._adopt_orphan_apexes()
        self._mark_dirty()
        self.get_logger().info(
            f"[static_reopt] obstacle set -> {len(obs)} obstacle(s); will batch re-opt at start/finish")
        # RETRO association: an obstacle confirmed only AFTER the car passed it (layer confirm
        # latency, typical for obstacles right past the start line) missed its live apex — the
        # avoidance path that produced it is already empty. Replay the recent-path buffer so the
        # just-driven maneuver still contributes its apex, and the solve can run THIS lap.
        if obs and self._path_buffer:
            replayed = False
            for _t, wx, wy, wd in self._path_buffer:
                replayed = self._record_apexes(wx, wy, wd) or replayed
            if replayed:
                self.get_logger().info(
                    f"[static_reopt] retro-associated apex(es) from {len(self._path_buffer)} "
                    f"buffered reactive path(s)")

    def clear_cb(self, _msg: Empty):
        self._apex_by_obs.clear()
        self._apex_miss.clear()
        self._path_buffer.clear()   # no retro-resurrection of just-cleared apexes
        if self._obstacles:
            self._obstacles = []
            self._obs_ids = []
            self._mark_dirty()
        self.get_logger().info("[static_reopt] obstacle set + apex records CLEARED by external request")

    def _mark_dirty(self):
        """Arm the rebuild trigger AND discard any pending bundle: a pending built from the
        PREVIOUS obstacle/apex state is stale — it both blocked the fresh rebuild (the solve gate
        requires _pending is None) and, once committed, installed an outdated line. Observed: a
        spurious unlatch->re-confirm flap (set 1->0->1) left a pending CLEAN revert queued while
        the obstacle was confirmed again — the re-opt then never ran."""
        self._obstacles_dirty = True
        self._dirty_since = self.get_clock().now().nanoseconds * 1e-9
        self._pending = None
        self._pending_dev = None

    def _obstacles_changed(self, new: List[core.Obstacle]) -> bool:
        if len(new) != len(self._obstacles):
            return True
        tol = self.obs_change_tol
        for a, b in zip(new, self._obstacles):
            if abs(a.x - b.x) > tol or abs(a.y - b.y) > tol or abs(a.r - b.r) > tol:
                return True
        return False

    # ----------------------------------------------------------------------------------
    # reactive-spline apex recording (design B) — exploration lap learns each obstacle's apex
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _quant_key(o: core.Obstacle):
        """Fallback key when the publisher supplies no usable marker ids."""
        return ("q", round(o.x / 0.2), round(o.y / 0.2))

    def _keys_for(self, obs: List[core.Obstacle], ids: List[int]):
        """One stable key per obstacle. Prefers the upstream track id (static_obstacle_layer assigns
        marker_id once per track and keeps it across EMA position updates) so an apex survives the
        obstacle drifting; falls back to position quantization only if the ids are absent or
        ambiguous (all zero / duplicated), which would otherwise alias two obstacles onto one key."""
        if ids and len(ids) == len(obs) and any(i != 0 for i in ids) and len(set(ids)) == len(ids):
            return [("id", i) for i in ids]
        return [self._quant_key(o) for o in obs]

    def otwpnts_cb(self, msg: OTWpntArray):
        """Record each confirmed obstacle's apex from the reactive avoidance path. For each
        obstacle we take the spline point NEAREST it (the point beside it = its apex) and keep the
        one with the largest |d| seen over the lap (the widest clearance the reactive layer
        committed to). The map-frame (x,y) is stored so the core re-projects it convention-free."""
        wps = msg.wpnts
        # Track whether the reactive layer is currently commanding an avoidance. An empty path, or
        # one that never leaves the raceline, means idle.
        now = self.get_clock().now().nanoseconds * 1e-9
        active = bool(wps) and max((abs(w.d_m) for w in wps), default=0.0) >= self._apex_min_d
        if self._reactive_active and not active:
            self._reactive_idle_t = now
        self._reactive_active = active
        if not wps:
            return
        wx = np.fromiter((w.x_m for w in wps), float, len(wps))
        wy = np.fromiter((w.y_m for w in wps), float, len(wps))
        wd = np.fromiter((w.d_m for w in wps), float, len(wps))
        if active:
            # keep recent avoidance paths for RETRO association (obstacle confirmed after the pass)
            self._path_buffer.append((now, wx, wy, wd))
            while self._path_buffer and (now - self._path_buffer[0][0]) > self.apex_buffer_sec:
                self._path_buffer.popleft()
        if not self._obstacles:
            return
        if self._record_apexes(wx, wy, wd):
            # A new/better apex is a REASON to rebuild. Without this, an obstacle set collected
            # BEFORE its apex existed (near-start obstacle: confirmed only after the pass; the
            # seam-crossing retry then built a 0-apex clean-equivalent and burned the flag) left
            # the solve gate in frenet_cb permanently closed -> reactive re-avoidance every lap.
            self._mark_dirty()

    def _adopt_orphan_apexes(self):
        """Re-key apex records whose track id vanished (the layer re-issues marker ids on an
        unlatch->re-confirm flap or track replacement): an orphaned record lying beside a CURRENT
        obstacle is the same physical avoidance. Losing it silently dropped that obstacle's hump
        from the next rebuild — the line 'forgot' obstacle 1 the moment obstacle 2 triggered a
        re-solve, and the swap yanked the car off the vanished hump."""
        keys = self._keys_for(self._obstacles, self._obs_ids)
        live = set(keys)
        orphans = [k for k in self._apex_by_obs if k not in live]
        for o, key in zip(self._obstacles, keys):
            if key in self._apex_by_obs or not orphans:
                continue
            best, best_d = None, 1.5              # apex sits within ~clearance of its obstacle
            for k in orphans:
                rec = self._apex_by_obs[k]
                d = float(np.hypot(rec[0] - o.x, rec[1] - o.y))
                if d < best_d:
                    best, best_d = k, d
            if best is not None:
                self._apex_by_obs[key] = self._apex_by_obs.pop(best)
                self._apex_miss.pop(best, None)
                orphans.remove(best)
                self.get_logger().info(
                    f"[static_reopt] adopted orphaned apex record for obstacle "
                    f"@({o.x:.2f},{o.y:.2f}) (track id re-issued)")

    def _clean_offset(self, x: float, y: float):
        """Signed lateral offset of a map point from the CLEAN raceline (+d = d_left side).
        Returns (d, station_index, left_normal)."""
        i = int(np.argmin(np.hypot(self._clean_xy[:, 0] - x, self._clean_xy[:, 1] - y)))
        j = (i + 1) % len(self._clean_xy)
        t = self._clean_xy[j] - self._clean_xy[i]
        norm = float(np.hypot(t[0], t[1]))
        if norm < 1e-9:                                   # duplicated closing point
            j = (i + 2) % len(self._clean_xy)
            t = self._clean_xy[j] - self._clean_xy[i]
            norm = float(np.hypot(t[0], t[1])) or 1.0
        left = np.array([-t[1], t[0]]) / norm             # +d side (d_left convention)
        d = float((np.array([x, y]) - self._clean_xy[i]) @ left)
        return d, i, left

    def _apex_plausible(self, x: float, y: float) -> bool:
        """Corridor sanity for a CANDIDATE apex: its signed offset from the clean raceline must
        fit inside the drivable band at that station (minus roughly half a car). Reactive paths
        produced while the car/frames were displaced (stuck phases, mid-swap transients) can
        carry outlier d values; recording one poisons the hump the re-opt lays."""
        d, i, _ = self._clean_offset(x, y)
        return -(self._clean_dr[i] - 0.12) <= d <= (self._clean_dl[i] - 0.12)

    def _record_apexes(self, wx, wy, wd) -> bool:
        """Associate one reactive path with the confirmed obstacles and update the apex records.
        Returns True when any apex was newly recorded or MOVED by more than 5 cm in amplitude
        (or 10 cm in position) — a rebuild-worthy change.

        The record is NEWEST-WINS, not keep-the-max: the historical max RATCHETED — one outlier
        path from a displaced frame (stuck/flap phases) permanently inflated the hump (measured:
        apex growing 0.61 -> 0.96 -> 1.41 m on a 1.39 m-wide track; the breaker then committed a
        line the car could never reach). The newest qualifying path self-corrects outliers, and
        _apex_plausible rejects candidates outside the physical corridor outright.

        OWNERSHIP first: give every spline point to its NEAREST obstacle, then let each obstacle
        pick its apex only from the points it owns — a second obstacle sitting inside the FIRST
        one's avoidance hump must not record that hump as its own apex."""
        keys = self._keys_for(self._obstacles, self._obs_ids)
        ox = np.fromiter((o.x for o in self._obstacles), float, len(self._obstacles))
        oy = np.fromiter((o.y for o in self._obstacles), float, len(self._obstacles))
        owner = np.argmin(np.hypot(wx[:, None] - ox[None, :], wy[:, None] - oy[None, :]), axis=1)
        changed = False
        for idx, (o, key) in enumerate(zip(self._obstacles, keys)):
            mine = np.where(owner == idx)[0]
            if mine.size == 0:
                continue
            dist = np.hypot(wx[mine] - o.x, wy[mine] - o.y)
            # apex = the largest deviation among MY points, and it must be beside me
            j = int(mine[int(np.argmax(np.abs(wd[mine])))])
            if float(dist.min()) > self._apex_assoc_tol or abs(wd[j]) < self._apex_min_d:
                continue
            if float(np.hypot(wx[j] - o.x, wy[j] - o.y)) > self._apex_assoc_tol:
                continue
            # ABEAM guard: the apex must sit BESIDE its obstacle ALONG the track. Ramp points of
            # a NEIGHBOURING avoidance (or the decaying return right after passing this obstacle)
            # sweep within association range when obstacles sit a few metres apart — under
            # newest-wins they OVERWROTE a good record with the ramp's small d, and the first
            # obstacle's hump then vanished from the next rebuild ("the line forgot obstacle 1").
            d_rec, i_rec, left = self._clean_offset(float(wx[j]), float(wy[j]))
            d_obs, i_obs, _ = self._clean_offset(float(o.x), float(o.y))
            n_cl = len(self._clean_xy)
            gap_st = abs(i_rec - i_obs)
            gap_st = min(gap_st, n_cl - gap_st) * (self._track_len / max(n_cl - 1, 1))
            if gap_st > 1.0:
                continue
            if not self._apex_plausible(float(wx[j]), float(wy[j])):
                self.get_logger().warning(
                    f"[static_reopt] REJECTED implausible apex xy=({wx[j]:.2f},{wy[j]:.2f}) "
                    f"d={wd[j]:+.2f}m for obstacle @({o.x:.2f},{o.y:.2f}) — outside the track "
                    f"corridor (displaced-frame outlier)", throttle_duration_sec=2.0)
                continue
            # OVERSHOOT clamp: while the car rides the hump with steering slip, the replanned
            # path is anchored at the DISPLACED car, so its widest point can exceed what the
            # avoidance needs (measured: apex creeping 0.6 -> 0.85+ while "steering clipped").
            # The record must hold what the obstacle REQUIRES: obstacle offset + radius +
            # keep-out(0.31) + bulge(0.10) + slack — clamp anything wider back onto that.
            ax_x, ax_y = float(wx[j]), float(wy[j])
            side = 1.0 if d_rec >= d_obs else -1.0
            need = d_obs + side * (float(o.r) + 0.45)
            if (d_rec - need) * side > 0.0:
                ax_x = float(self._clean_xy[i_rec][0] + need * left[0])
                ax_y = float(self._clean_xy[i_rec][1] + need * left[1])
                d_rec = need
            prev = self._apex_by_obs.get(key)
            self._apex_by_obs[key] = (ax_x, ax_y, abs(float(d_rec)))
            if prev is None:
                changed = True
                self.get_logger().info(
                    f"[static_reopt] recorded reactive apex for obstacle @({o.x:.2f},{o.y:.2f}) "
                    f"-> apex xy=({ax_x:.2f},{ax_y:.2f}) d={d_rec:+.2f}m")
            elif (abs(abs(float(d_rec)) - prev[2]) > 0.05
                  or float(np.hypot(prev[0] - ax_x, prev[1] - ax_y)) > 0.10):
                changed = True
        return changed

    def _apex_list(self, obstacles: List[core.Obstacle]) -> List[tuple]:
        """Map-frame (x,y) apex points for the confirmed obstacles that have a recorded reactive
        apex. Obstacles never reactively avoided contribute nothing (clean line kept there)."""
        out = []
        for key in self._keys_for(obstacles, self._obs_ids):
            rec = self._apex_by_obs.get(key)
            if rec is not None:
                out.append((rec[0], rec[1]))
        return out

    def _rebuild_and_swap(self, reason: str):
        """Solve ONE re-opt over the whole confirmed obstacle set. Called once a full exploration
        lap is done; the swap itself is DEFERRED to a station where the old and new lines coincide
        (see _commit_pending) — the offset seam sits in the largest apex-free gap, not at s=0, so
        swapping at s=0 can step the reference laterally. The solve is ~10 ms (BLAS-pinned)."""
        obstacles = list(self._obstacles)
        # S3b (extended): with obstacles confirmed but 0 recorded apexes there is NOTHING to build —
        # the core would lay no hump and return a bit-for-bit CLEAN line. Building it anyway (a)
        # silently reverted a working obstacle-aware line, and (b) even from the clean line it
        # BURNED the dirty flag: the apex captured on the NEXT avoidance then had no armed trigger
        # left, so the obstacle-aware line was never built (near-start obstacles re-avoided
        # reactively every lap). Keep the current line and stay armed; the apex-capture path in
        # otwpnts_cb re-triggers the solve the moment an apex exists.
        if obstacles and not self._apex_list(obstacles):
            self.get_logger().warning(
                f"[static_reopt] {len(obstacles)} obstacle(s) confirmed but 0 recorded apexes — "
                f"keeping the current line, waiting for a reactive apex",
                throttle_duration_sec=5.0)
            self._obstacles_dirty = True         # re-arm; do NOT burn the flag
            return
        self._obstacles_dirty = False
        now = self.get_clock().now().nanoseconds * 1e-9
        try:
            if obstacles:
                with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                    bundle = self._build_obstacle_bundle(obstacles)
            else:
                bundle = self.clean_bundle
        except Exception as e:  # noqa: BLE001 — must never propagate to the executor
            self.get_logger().warn(
                f"[static_reopt] batch re-opt FAILED ({type(e).__name__}: {str(e)[:80]}); "
                f"keeping the current line — will retry (reactive planner handles the gap)")
            self._obstacles_dirty = True         # re-arm; a transient failure must not eat the trigger
            self._solve_backoff_until = now + self.solve_retry_backoff_s
            return
        if bundle is self.active:
            return
        # Same guard, now on the BUILT result: every apex was corridor-rejected (all-or-nothing
        # fit), so the bundle is geometrically clean. Never install it: from an obstacle-aware
        # active it would silently revert the avoidance; from the CLEAN active it is a no-op swap.
        # Corridor rejection is DETERMINISTIC (same apexes + same track -> same answer), so the
        # trigger stays burned — no retry loop; a new/better apex re-arms it via otwpnts_cb, and
        # transient solve failures take the exception path above instead.
        n_ap = getattr(bundle, "n_apex", 0)
        if obstacles and n_ap == 0:
            self.get_logger().warning(
                f"[static_reopt] re-opt laid 0 humps for {len(obstacles)} obstacle(s) — every "
                f"apex corridor-rejected (track too tight; see the CORRIDOR-REJECTED log). "
                f"Keeping the current line; those obstacles stay reactive-only")
            return
        self._pending = bundle
        self._pending_dev = self._line_dev(bundle, self.active)
        self._pending_since = now
        kind = "CLEAN" if (bundle is self.clean_bundle or n_ap == 0) else f"OBSTACLE-AWARE ({n_ap} apex)"
        self.get_logger().info(
            f"[static_reopt] batch re-opt -> {kind} ready ({reason}); committing where the lines "
            f"meet (max dev {float(np.max(self._pending_dev)):.3f} m)")

    @staticmethod
    def _bundle_xy(b: "_Bundle"):
        wp = b.glb_wpnts.wpnts
        return (np.array([w.s_m for w in wp]),
                np.array([w.x_m for w in wp]), np.array([w.y_m for w in wp]))

    def _line_dev(self, a: "_Bundle", b: "_Bundle") -> np.ndarray:
        """Per-station GEOMETRIC distance from each of `a`'s points to the polyline `b`.

        NOT s-matched: the hump changes the total arc length (a 0.6 m apex measured +0.95 m of
        lap), so comparing points at equal s reads ~0.95 m of PHANTOM deviation at stations 10 m
        away from the hump whose geometry is identical (measured: 4% of the lap "agreeing" at
        5 cm vs the true 82%). That blocked the commit gates everywhere but the seam and made
        the deadlock breaker refuse geometrically-perfect lines."""
        _, xa, ya = self._bundle_xy(a)
        _, xb, yb = self._bundle_xy(b)
        ax, ay = xb, yb
        bx, by = np.roll(xb, -1), np.roll(yb, -1)
        dx, dy = bx - ax, by - ay
        seg2 = np.maximum(dx * dx + dy * dy, 1e-12)
        t = ((xa[:, None] - ax[None, :]) * dx[None, :]
             + (ya[:, None] - ay[None, :]) * dy[None, :]) / seg2[None, :]
        t = np.clip(t, 0.0, 1.0)
        cx = ax[None, :] + t * dx[None, :]
        cy = ay[None, :] + t * dy[None, :]
        return np.sqrt(((xa[:, None] - cx) ** 2 + (ya[:, None] - cy) ** 2).min(axis=1))

    def _commit_pending(self, s: float, force: bool = False):
        """Swap once BOTH hold: the reactive layer is idle (no obstacle in its horizon, so changing
        the global line cannot pull the rug from under an avoidance in progress), and the lines
        agree (< 5 cm) over the WHOLE stretch the controller is about to consume — from the car to
        swap_horizon ahead. Agreement at the current station alone let the swap land right at a
        hump entrance: the line then changed inside the controller's lookahead and jerked the car.

        This replaces waiting for the start/finish crossing: s=0 is an arbitrary point that can fall
        right beside the NEXT obstacle, and swapping there made the car miss it.

        force=True (stale-frenet bench fallback only): no odometry means no meaningful station or
        reactive state — commit unconditionally so headless/bag runs still get the line."""
        if self._pending is None:
            return
        if not force:
            now = self.get_clock().now().nanoseconds * 1e-9
            # Map the car onto the PENDING line GEOMETRICALLY: the car's s lives on the ACTIVE
            # line and the hump changes total arc length (~1 m for a 0.6 m apex), so matching by
            # s picked a pending station up to ΔL (~10 stations) away from where the car really is.
            sb, xb, yb = self._bundle_xy(self.active)
            Lb = float(sb[-1]) if len(sb) and sb[-1] > 0 else self._track_len
            car_x = float(np.interp(s % Lb, sb, xb))
            car_y = float(np.interp(s % Lb, sb, yb))
            sa, xa, ya = self._bundle_xy(self._pending)
            j0 = int(np.argmin(np.hypot(xa - car_x, ya - car_y)))
            # DEADLOCK BREAKER (see the parameter block): stuck trailing = the car sits inside
            # the pending hump at low speed; the normal gates can then never pass. Commit anyway.
            stuck = ((now - self._pending_since) > self.swap_deadlock_s
                     and abs(self._last_vs) < self.swap_deadlock_max_vs)
            if stuck:
                dev_car = float(self._pending_dev[j0])
                if dev_car > self.swap_deadlock_max_dev:
                    # A sane hump never puts the new line this far from a car that was following
                    # the active line — the pending geometry is poisoned. Committing it strands
                    # the car >1 m off its own raceline (AEB clamps, lookahead lands on the far
                    # line). Discard; the next (sanity-checked) apex update rebuilds a sane one.
                    self.get_logger().error(
                        f"[static_reopt] deadlock breaker REFUSED: pending line is {dev_car:.2f} m "
                        f"from the car (> {self.swap_deadlock_max_dev:.2f}) — discarding the "
                        f"pending; waiting for a sane apex/rebuild")
                    self._pending = None
                    self._pending_dev = None
                    return
                self.get_logger().warn(
                    f"[static_reopt] swap deadlock breaker: pending waited "
                    f"{now - self._pending_since:.1f}s with the car at {self._last_vs:.1f} m/s "
                    f"— committing mid-hump to un-stick (car {dev_car:.2f} m off the new line)")
            else:
                if self._reactive_active or (now - self._reactive_idle_t) < self.swap_idle_s:
                    return
                horizon = max(self.swap_horizon_min_m, self.swap_horizon_time_s * abs(self._last_vs))
                La = float(sa[-1]) if len(sa) and sa[-1] > 0 else self._track_len
                ahead = ((sa - sa[j0]) % La) <= horizon
                if not ahead.any() or float(np.max(self._pending_dev[ahead])) > 0.05:
                    return
        bundle = self._pending
        self._pending = None
        self._pending_dev = None
        self.active = bundle
        # tell caching consumers to re-take the new geometry, repeatedly and FAST (notify_cb)
        self._notify_scaler_ticks = self.notify_ticks
        kind = "CLEAN" if bundle is self.clean_bundle else "OBSTACLE-AWARE"
        self.get_logger().info(f"[static_reopt] swapped to {kind} at s={s:.2f} m")
        self._publish_active(bundle)             # publish now, don't wait for the republish tick
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
        self._last_vs = float(msg.twist.twist.linear.x)   # swap-horizon scaling
        self._last_frenet_t = self.get_clock().now().nanoseconds * 1e-9
        L = self._track_len
        if self._last_s is not None:
            ds = s - self._last_s
            if ds < -1.0:
                ds += L                       # a genuine wrap is one lap of forward travel
            if 0.0 <= ds < 1.0:               # ignore backward/teleport frames in the odometer
                self._s_progressed += ds
        # A real seam crossing needs the PREVIOUS sample to be near the end of the lap; requiring a
        # full lap of travel on top makes the trigger independent of WHERE the car started (the
        # map's s=0 is not the start of the exploration lap) and immune to a parked car's s
        # flickering across the seam (its odometer never advances).
        crossed_sf = (self._last_s is not None
                      and s < self._last_s - 1.0
                      and self._last_s > 0.85 * L
                      and self._s_progressed > 0.9 * L)
        self._last_s = s
        # SOLVE as soon as the set is dirty AND at least one apex has been captured — no need to
        # wait for the lap boundary, the solve is the cheap part and the swap is gated separately.
        # The lap crossing stays as a retry tick for the case where no apex existed yet.
        if (self._obstacles_dirty and self._pending is None
                and self._last_frenet_t >= self._solve_backoff_until):
            # An EMPTY set must rebuild immediately: _apex_list([]) is falsy, so without the extra
            # clause the clean revert would wait for the next seam crossing — up to a full lap on
            # an obsolete detour after the obstacles were physically removed. The clean bundle is
            # precomputed, the "solve" is free.
            if self._apex_list(self._obstacles) or crossed_sf or not self._obstacles:
                if crossed_sf:
                    self._s_progressed = 0.0
                self._rebuild_and_swap(
                    "obstacles cleared" if not self._obstacles
                    else ("apex captured" if not crossed_sf else "start/finish"))
        self._commit_pending(s)

    def republish_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        # STALE-FRENET fallback: no odom -> we never see the s-wrap. Solve once so a headless /
        # bag run still gets the obstacle line. Never fires while the car is actually driving.
        frenet_stale = (self._last_frenet_t is None
                        or (now - self._last_frenet_t) > self.swap_timeout_s)
        # Only when there is NO odometry at all (bag / headless). With odometry the lap gate in
        # frenet_cb owns the trigger — this path would otherwise swap at an arbitrary s.
        if self._obstacles_dirty and frenet_stale and (now - self._dirty_since) > self.swap_timeout_s:
            self._rebuild_and_swap("frenet stale fallback")
            if self._pending is not None:        # no odometry -> no station/horizon gate available
                self._commit_pending(self._last_s if self._last_s is not None else 0.0, force=True)

        active = self.active
        # publish ONLY when the line changed, or as a slow keep-alive (>= 5 s). No churn.
        if (active is not self._last_published) or (now - self._last_publish_t) >= 5.0:
            self._publish_active(active)

    def notify_cb(self):
        """Fast `update_map` re-notify after a swap (own timer, notify_period << sector_tuner's
        0.5 s scale timer). sector_tuner re-takes the geometry only when its scale timer sees the
        flag set; a single notify can be consumed BEFORE the new /global_waypoints arrives (the
        1-byte Bool can beat the ~36 kB WpntArray), leaving it re-scaling the OLD line. Repeating
        the notify closes that window within one consumer tick."""
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
