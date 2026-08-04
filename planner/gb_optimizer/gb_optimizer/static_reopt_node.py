"""
static_reopt_node.py — obstacle-aware global-waypoint republisher (IFAC static obstacles).

Runtime replacement for gb_optimizer's global_trajectory_publisher during obstacle races.
It holds the CLEAN raceline bundle (read once from global_waypoints.json) and, when static
obstacles are present, a re-optimized OBSTACLE-AWARE bundle (main mincurv_iqp + shortest_path
SP, both from the width-modulated reftrack), and republishes whichever is active on the exact
same topics the rest of the stack already consumes. gb_optimizer itself is untouched — only
its library functions are imported (see static_reopt_core, memory: project-ifac-static-reopt).

HARD GUARANTEE — it must NEVER stop publishing a valid raceline:
  * ANY re-opt failure (infeasible QP, exception) is swallowed — the node keeps the last
    valid bundle (obstacle-aware if one was ever built for the current obstacles, else clean);
  * `active` is only ever rebound to a fully-built bundle.
So even a track-blocking obstacle degrades to "publish the previous good line" (the reactive
planner then handles what the global line cannot) — it never emits nothing or a broken line.

THREADING — the solve is SYNCHRONOUS on the single-threaded rclpy executor, inside frenet_cb.
This header used to claim a background thread and a lock, and neither has ever existed (see the
state block in __init__): the `local_window` solve is ~10 ms with BLAS pinned to one thread, which
is why running it inline is acceptable, and the absence of concurrency is precisely why no lock is
needed. The claim mattered because it read as a latency guarantee the code does not provide — a
slow solve DOES block the executor, and the legacy `reopt_method: "global"` path (minutes per
solve) must therefore never be used online.

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
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import WpntArray, OTWpntArray

from grid_filter.grid_filter import GridFilter

from .readwrite_global_waypoints import read_global_waypoints
from . import static_reopt_core as core


class _Bundle:
    """The full set of messages the republisher emits (mirrors global_trajectory_publisher)."""

    __slots__ = (
        "map_info", "est_lap_time",
        "cent_wpnts", "cent_markers",
        "glb_wpnts", "glb_markers",
        "sp_wpnts", "sp_markers",
        "trackbounds", "n_apex", "clearance_ok", "clearance_by_key", "floor_by_key",
        "coverage",
    )

    def __init__(self, map_info, est_lap_time, cent_wpnts, cent_markers,
                 glb_wpnts, glb_markers, sp_wpnts, sp_markers, trackbounds, n_apex=0,
                 clearance_ok=True, clearance_by_key=None, floor_by_key=None,
                 coverage=None):
        # n_apex = humps ACTUALLY laid into this line. 0 means the geometry is the clean raceline
        # even if obstacles were passed in (no apex recorded, or the corridor could not hold it).
        self.n_apex = n_apex
        # clearance_ok = every obstacle this line claims to have reshaped is really cleared by
        # obs_margin, measured on the built geometry. False = do NOT publish it (see
        # _check_line_clearance); the previous good line is kept instead.
        self.clearance_ok = clearance_ok
        # clearance_by_key = per-obstacle edge clearance of THIS line at the positions it was
        # built from. The baseline the drift trigger compares live positions against, so it must
        # travel with the line rather than with the node (see _clearance_drifted).
        self.clearance_by_key = clearance_by_key or {}
        # floor_by_key = the clearance each LAID hump was accepted at (the coverage
        # ladder may settle below obs_margin). What the line promised for that box.
        self.floor_by_key = floor_by_key or {}
        # coverage = one record per confirmed obstacle: what this line does about it and
        # why. Published on /static_reopt/coverage and used to compare candidate lines.
        self.coverage = coverage or []
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
        # [m] lowest clearance a hump may be ACCEPTED at when obs_margin is unreachable on either
        # side (see the coverage ladder in static_reopt_core.build_offset_profile). Set at the
        # REACTIVE keep-out (width_car/2 + safety_margin): below obs_margin the global line stops
        # relieving the reactive layer of that box, but down to here it still clears it by what the
        # reactive planner would itself have driven -- strictly better than not covering it. Set
        # equal to obs_margin to disable the ladder.
        self.declare_parameter("relax_floor", 0.30)
        # FINAL wall gate: every published point is checked against the eroded occupancy map before
        # the line is allowed to swap in. The corridor the fit works in is derived from the
        # waypoints' d_left/d_right, which are an estimate; the map is the only wall source that
        # cannot be mislabelled or smoothed. Same erosion kernel as the reactive planner so the two
        # agree on where a wall is. 0 disables the gate.
        self.declare_parameter("wall_gate_kernel", 3)
        # [m] apex spacing below which two obstacles are treated as ONE cluster: same side ->
        # merged into a single hump, opposite sides -> dropped as a pair when no reach can swing
        # between them inside the curvature budget. 0 disables the pre-pass.
        self.declare_parameter("apex_merge_gap_m", 2.0)
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
        # [m] CONSEQUENCE trigger, complementing obs_change_tol's cause trigger. obs_change_tol
        # fires on how far a box moved, which says nothing about whether the move mattered; this
        # fires when the ACTIVE line has stopped clearing a box it used to clear. Set at the SM's
        # static GB requirement (gb_ego_width_m/2 + lateral_width_static_gb_m = 0.25) plus slack,
        # i.e. rebuild BEFORE the free-check would call the line blocked and drop to TRAILING.
        # Must stay between that requirement and obs_margin — check_avoidance_margins.py verifies.
        self.declare_parameter("clearance_dirty_m", 0.30)
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
        # [m] corridor-fit TOLERANCE. The corridor is a smoothed estimate that ripples ~2 mm between
        # adjacent stations while the amplitude cap parks the hump peak exactly on it, so a
        # zero-tolerance fit rejects every wide reach over a sub-millimetre violation and collapses
        # the hump. Measured on ifac: 0 -> 5 mm took the fitted reach from 1.24 m to 5.00 m, the
        # laid curvature from 1.98 to 1.46 (curvlim 1.5) and +1.62 s of lap time back to +0.05 s.
        # Raise only if the walls are being shaved; it is spent out of wall_margin.
        self.declare_parameter("fit_tol", 0.005)

        self.map_name = self.get_parameter("map").value
        self.racecar_version = self.get_parameter("racecar_version").value
        self.safety_width = float(self.get_parameter("safety_width").value)
        self.safety_width_sp = float(self.get_parameter("safety_width_sp").value)
        self.obs_margin = float(self.get_parameter("obs_margin").value)
        self.relax_floor = float(self.get_parameter("relax_floor").value)
        self.wall_gate_kernel = int(self.get_parameter("wall_gate_kernel").value)
        self.apex_merge_gap_m = float(self.get_parameter("apex_merge_gap_m").value)
        self.default_obs_radius = float(self.get_parameter("default_obs_radius").value)
        self.republish_period = float(self.get_parameter("republish_period").value)
        self.notify_period = float(self.get_parameter("notify_period").value)
        self.notify_ticks = int(self.get_parameter("notify_ticks").value)
        self.swap_at_sf = bool(self.get_parameter("swap_at_startfinish").value)
        self.swap_timeout_s = float(self.get_parameter("swap_timeout_s").value)
        self.obs_change_tol = float(self.get_parameter("obs_change_tol").value)
        self.clearance_dirty_m = float(self.get_parameter("clearance_dirty_m").value)
        self.compute_sp = bool(self.get_parameter("compute_sp").value)
        self.reopt_method = str(self.get_parameter("reopt_method").value)
        self.reach_time = float(self.get_parameter("reach_time").value)
        self.reach_min = float(self.get_parameter("reach_min").value)
        self.reach_max = float(self.get_parameter("reach_max").value)
        self.qp_veh_width = float(self.get_parameter("qp_veh_width").value)
        self.wall_margin = float(self.get_parameter("wall_margin").value)
        self.fit_tol = float(self.get_parameter("fit_tol").value)

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
        # Apex movement at or above this counts as a MAJOR change: worth discarding a queued
        # (pending) line over. Below it the queued line and the rebuilt one differ by centimetres,
        # and the queued one is allowed to commit first — see _mark_dirty.
        self.declare_parameter("apex_major_change_m", 0.10)
        # [m] how far past the obstacle's abeam station a reactive path must extend on BOTH sides
        # before its apex is believed. Rejects the exit ramp of an already-passed committed slice.
        self.declare_parameter("apex_span_margin_m", 0.5)
        # [m] max |d| the recorded apex may fall SHORT of what the obstacle geometrically requires.
        # Symmetric with the long-standing overshoot clamp; without it a decaying tail reads as a
        # legitimately tighter apex.
        self.declare_parameter("apex_undershoot_m", 0.12)
        # [m] max along-track distance from the obstacle to the recorded point. 1.0 admitted the
        # polluted records (0.4-1.0 m) as readily as the healthy ones (all <= 0.1 m).
        self.declare_parameter("apex_abeam_gap_m", 0.5)
        self.apex_major_change_m = float(self.get_parameter("apex_major_change_m").value)
        self.apex_span_margin_m = float(self.get_parameter("apex_span_margin_m").value)
        self.apex_undershoot_m = float(self.get_parameter("apex_undershoot_m").value)
        self.apex_abeam_gap_m = float(self.get_parameter("apex_abeam_gap_m").value)
        self._apex_change_major = False
        self._apex_dirty_pending = False   # an apex changed; arm the rebuild at idle
        # Obstacle keys whose clearance drift has already re-armed the solve against the CURRENTLY
        # active line. Without this latch the trigger re-fires every frame until the new line is
        # committed — and _mark_dirty discards the pending, so the very swap that would restore
        # the clearance could never land. Cleared when the active line changes.
        self._clearance_dirty_keys = set()
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
        # Per-obstacle coverage of the line that is ACTUALLY driving. Latched: what this answers --
        # "is that box covered by the global line, and if not why" -- is exactly the question asked
        # after the fact, from a bag or a late subscriber.
        self.pub_coverage = self.create_publisher(MarkerArray, "/static_reopt/coverage", latched)

        # Eroded occupancy map for the final wall gate (see _wall_gate_ok).
        self.map_filter = None
        if self.wall_gate_kernel > 0:
            self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
            self.map_filter.set_erosion_kernel_size(self.wall_gate_kernel)

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

    # ORDER OF PREFERENCE for what a line does about one obstacle. Used to compare two candidate
    # lines obstacle-by-obstacle so a rebuild can never quietly cover LESS than what is already
    # driving -- see _coverage_regresses. "already_clear" ranks with "laid": the line needs no hump
    # there and is not failing to provide one.
    _COVER_RANK = {"laid": 2, "already_clear": 2, "no_apex": 1, "dropped": 0}

    def _coverage(self, obstacles, pairs, res, gaps_by_id=None, traj=None) -> List[dict]:
        """One record per confirmed obstacle: what this line does about it, and why.

        Everything here already existed inside the build, scattered across `apex_laid`,
        `apex_dropped` and a subtraction; what was missing was a single attributed answer per
        obstacle. Without it "2/3 reshaped" is the only thing anyone downstream (or reading a bag)
        can see, and it does not say WHICH obstacle is uncovered or why -- so a line that silently
        stopped covering one is indistinguishable from one that never could.
        """
        apex_of = {id(o): i for i, (_xy, o) in enumerate(pairs)}
        laid_by_i = {a["obs_i"]: a for a in res.get("apex_laid", []) if a.get("obs_i") is not None}
        drop_by_i = {d["obs_i"]: d for d in res.get("apex_dropped", []) if d.get("obs_i") is not None}
        gaps = (gaps_by_id if gaps_by_id is not None
                else dict(zip((id(o) for o in obstacles), self._line_clearances(traj, obstacles)))
                if traj is not None else {})
        out = []
        for o in obstacles:
            i_a = apex_of.get(id(o))
            clr = float(gaps.get(id(o), float("nan")))
            if i_a is None:
                # No recorded reactive apex. Distinguish "needs nothing" from "needs something we
                # cannot place": the core skips an obstacle the line already clears BEFORE any apex
                # is involved, and calling that a missing apex is what the old arithmetic did.
                status = "already_clear" if clr >= self.obs_margin else "no_apex"
            elif i_a in laid_by_i:
                status = "laid"
            elif i_a in drop_by_i:
                status = "dropped:" + str(drop_by_i[i_a].get("reason", "?"))
            else:
                status = "already_clear" if clr >= self.obs_margin else "no_apex"
            out.append({"id": int(getattr(o, "id", -1)) if hasattr(o, "id") else -1,
                        "x": float(o.x), "y": float(o.y), "r": float(o.r),
                        "status": status, "clearance_m": clr})
        return out

    def _coverage_regresses(self, new_cov, old_cov) -> Optional[str]:
        """Does `new_cov` do WORSE than `old_cov` for any obstacle? Returns a reason, or None.

        The policy this enforces is not "all or nothing". A line that covers two of three
        obstacles is worth publishing -- the third is the reactive layer's, and it says so. What
        must never happen is a REGRESSION, because the fallback is not the clean line, it is
        whatever older line is still active: refusing to publish a line that covers less would
        leave the car on a better one, and publishing it would take coverage away silently.
        """
        if not old_cov:
            return None
        old_rank = {}
        for c in old_cov:
            old_rank[(round(c["x"], 2), round(c["y"], 2))] = self._COVER_RANK.get(
                c["status"].split(":")[0], 0)
        for c in new_cov:
            k = (round(c["x"], 2), round(c["y"], 2))
            if k not in old_rank:
                continue                     # a newly confirmed obstacle cannot be a regression
            new_r = self._COVER_RANK.get(c["status"].split(":")[0], 0)
            if new_r < old_rank[k]:
                return (f"obstacle @({c['x']:.2f},{c['y']:.2f}) would go from covered to "
                        f"'{c['status']}'")
        return None

    def _publish_coverage(self, bundle) -> None:
        """/static_reopt/coverage — one marker per confirmed obstacle, coloured by status, with the
        status and measured clearance in its text. Latched, so a late subscriber (or a bag opened
        afterwards) sees what the line that is actually driving does about every obstacle."""
        arr = MarkerArray()
        clear_m = Marker()
        clear_m.action = Marker.DELETEALL
        arr.markers.append(clear_m)
        for i, c in enumerate(getattr(bundle, "coverage", []) or []):
            head = c["status"].split(":")[0]
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "static_reopt_coverage"
            m.id = i
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y = c["x"], c["y"]
            m.pose.position.z = 0.6
            m.pose.orientation.w = 1.0
            m.scale.z = 0.25
            m.color.a = 1.0
            # green = the line handles it, amber = waiting on an apex, red = the reactive layer's
            m.color.r = 0.0 if head in ("laid", "already_clear") else 1.0
            m.color.g = 1.0 if head in ("laid", "already_clear", "no_apex") else 0.0
            m.text = f"id={c['id']} {c['status']} {c['clearance_m']:+.2f}m"
            arr.markers.append(m)
        self.pub_coverage.publish(arr)

    def _wall_gate_ok(self, traj) -> bool:
        """Is every PUBLISHED point inside the eroded free space of the occupancy map?

        The last gate before a swap, and the only one that consults the map rather than a model of
        it. Everything upstream reasons about the corridor derived from the waypoints' d_left /
        d_right, which is an estimate: it is smoothed, it is offset by a wall_margin someone can
        lower, and on some maps the two sides ship exchanged. The map cannot be any of those.

        Vectorised `GridFilter.is_point_inside` -- same pixel convention (row = y, no vertical
        flip) and the same erosion kernel as the reactive planner, so the two agree on where a wall
        is. Returns True when no map has arrived (bench/bag runs): a missing map must not silently
        block every swap, and the corridor checks upstream still apply.
        """
        f = self.map_filter
        img = getattr(f, "eroded_image", None) if f is not None else None
        if img is None or f.resolution is None or f.origin is None:
            return True
        try:
            xy = np.asarray(traj)[:, 1:3]
            px = ((xy[:, 0] - f.origin[0]) / f.resolution).astype(int)
            py = ((xy[:, 1] - f.origin[1]) / f.resolution).astype(int)
            h, w = img.shape
            inside = (px >= 0) & (py >= 0) & (px < w) & (py < h)
            free = np.zeros(len(xy), dtype=bool)
            free[inside] = img[py[inside], px[inside]] == 255
            if free.all():
                return True
            bad = np.flatnonzero(~free)
            self.get_logger().error(
                f"[static_reopt] REFUSING to publish: {bad.size} of {len(xy)} published points are "
                f"outside the eroded free space (first @({xy[bad[0], 0]:.2f},{xy[bad[0], 1]:.2f})). "
                f"The line would put the car into a wall — check reopt_wall_margin / "
                f"reopt_qp_veh_width and the map's d_left/d_right")
            return False
        except Exception as e:                       # a gate must never crash the build
            self.get_logger().debug(f"[static_reopt] wall gate skipped: {e}")
            return True

    def _line_clearances(self, traj, obstacles: List[core.Obstacle]) -> List[float]:
        """Distance from the line to each obstacle's EDGE [m], index-aligned with `obstacles`.

        One hypot scan per obstacle over every published station — the same measurement the core's
        acceptance floor enforces per hump, so the two cannot disagree about what "cleared" means.
        """
        xy = np.asarray(traj)[:, 1:3]
        return [float(np.min(np.hypot(xy[:, 0] - o.x, xy[:, 1] - o.y))) - float(o.r)
                for o in obstacles]

    def _check_line_clearance(self, traj, obstacles: List[core.Obstacle], laid: List[dict],
                              apex_obs: List[core.Obstacle] = None, floors: dict = None) -> bool:
        """Measure what the line ACTUALLY clears, in the map frame, and VETO it if it does not.

        Everything upstream reports intent -- "2/2 obstacle apex(es) reshaped" is printed whether
        the resulting line passes 0.6 m from the box or through it. It was true for six laps
        straight on the real car (bag verify_0731_2114) while the published line stayed ~0.2 m from
        an obstacle that needs radius + obs_margin, so the reactive layer re-avoided it every single
        lap and the log never said why.

        It used to be a pure report, on the argument that refusing the swap leaves the clean line,
        which clears even less. That argument only holds for the FIRST build: the fallback is not
        the clean line, it is whatever is currently active — and once an obstacle-aware line is
        active, publishing a worse one is a strict regression the stack cannot detect. The core now
        drops a hump it cannot lay to obs_margin (reason "clearance") and reports it, so a line
        arriving here short is a genuine escape: the woven/resampled geometry is not what the
        per-hump gate accepted. Keep the previous good line and say exactly which obstacle failed.

        Obstacles with NO hump at all (never reactively avoided, or honestly dropped) are expected
        to read short — they are the reactive layer's job and must not veto the line that was built
        for the others. Only apexes the core claims it LAID are held to the floor, and which those
        are is decided by IDENTITY (`apex_obs`, index-aligned with the apexes the core was given),
        not by proximity.

        Returns True when the line may be published.
        """
        try:
            if traj is None or len(traj) == 0 or not obstacles:
                return True
            gaps = self._line_clearances(traj, obstacles)
            floors = floors or {}
            bad = [(o, d) for o, d in zip(obstacles, gaps)
                   if d < floors.get(id(o), self.obs_margin)]
            apex_obs = apex_obs or []
            laid_obs = [apex_obs[a["obs_i"]] for a in (laid or [])
                        if a.get("obs_i") is not None and a["obs_i"] < len(apex_obs)]
            if not bad:
                self.get_logger().info(
                    f"[static_reopt] line clearance OK: all {len(obstacles)} obstacle(s) cleared by "
                    f">= {self.obs_margin:.2f} m")
                return True
            # Which of the short ones did the core promise to have reshaped? By IDENTITY: the core
            # reports the index of the obstacle each hump belongs to, and `laid_obs` resolves those
            # back to the obstacle objects.
            #
            # This used to match by position -- "some laid apex within 1.5 m of this box" -- which
            # is not the same question. Two obstacles a metre apart are inside each other's radius,
            # so a hump laid for one made its DROPPED neighbour count as broken, and the neighbour's
            # (expected, by-design) short clearance then vetoed a line that was correct for
            # everything it actually claimed. On a cluster that is every publish.
            broken = [(o, d) for o, d in bad if any(o is lo for lo in laid_obs)]
            det = "; ".join(f"@({o.x:.2f},{o.y:.2f}) r={o.r:.2f} -> {d:+.2f} m" for o, d in bad)
            if not broken:
                self.get_logger().info(
                    f"[static_reopt] {len(bad)}/{len(obstacles)} obstacle(s) are NOT cleared by the "
                    f"line (need >= {self.obs_margin:.2f} m): {det}. No hump was laid for them — "
                    f"the reactive layer keeps handling those, as designed")
                return True
            det_b = "; ".join(f"@({o.x:.2f},{o.y:.2f}) r={o.r:.2f} -> {d:+.2f} m "
                              f"(promised {floors.get(id(o), self.obs_margin):.2f})"
                              for o, d in broken)
            self.get_logger().error(
                f"[static_reopt] REFUSING to publish: {len(broken)} obstacle(s) the re-opt claims "
                f"to have reshaped are cleared by less than the floor they were accepted at: "
                f"{det_b}. Keeping the previous line — check the recorded apex and "
                f"reopt_wall_margin / reopt_fit_tol")
            return False
        except Exception as e:                      # a diagnostic must never break the build
            self.get_logger().debug(f"[static_reopt] clearance check failed: {e}")
            return True

    def _build_obstacle_bundle(self, obstacles: List[core.Obstacle]) -> _Bundle:
        """Run the width-modulated re-optimization and assemble a full bundle. May raise;
        the caller runs this inside a try/except so a failure never reaches the timer."""
        if self.reopt_method == "local_window":
            # Reshape the recorded reactive apexes into the global line (keep apex, gentle ramps).
            # The apex's OWN obstacle goes with it: obs_margin is a hard acceptance floor in the
            # core, so a hump that would not clear its box is grown once and then dropped for the
            # reactive layer rather than laid short (which used to pass the amplitude-ratio test
            # at 0.90 x apex = ~0.345 m of edge clearance, under the 0.35 m everything downstream
            # assumes the line is built to).
            pairs = self._apex_pairs(obstacles)
            apexes = [xy for xy, _o in pairs]
            apex_obs = [(float(o.x), float(o.y), float(o.r)) for _xy, o in pairs]
            res = core.reoptimize_local_window(
                self._clean_xy, self._clean_dr, self._clean_dl, self.reftrack,
                apexes, self.input_path,
                params=core.ModulationParams(obs_margin=self.obs_margin),
                w_veh=self.qp_veh_width, clean_vx=self._clean_vx, wall_margin=self.wall_margin,
                reach_time=self.reach_time, reach_min=self.reach_min, reach_max=self.reach_max,
                clean_kappa=self._clean_kappa, fit_tol=self.fit_tol,
                apex_obstacles=apex_obs, relax_floor=self.relax_floor,
                apex_merge_gap_m=self.apex_merge_gap_m)
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
        clearance_ok = True                 # `global` method: no per-apex geometry to check
        clearance_by_key = {}
        floor_by_key = {}
        coverage = []
        if self.reopt_method == "local_window":
            n_ap = res.get("n_windows", 0)
            dropped = res.get("apex_dropped", [])
            n_obs = len(obstacles)
            # PER-OBSTACLE STATUS, by identity. n_no_apex used to be inferred arithmetically
            # (n_obs - laid - dropped), which counted every obstacle the core skipped as
            # already-clear as if its apex were missing -- exactly backwards, since those are the
            # ones needing nothing. Attribute each obstacle instead.
            coverage = self._coverage(obstacles, pairs, res, gaps_by_id=None, traj=traj)
            n_no_apex = sum(1 for c in coverage if c["status"] == "no_apex")
            n_clear = sum(1 for c in coverage if c["status"] == "already_clear")
            info.data = (f"[static_reopt] obstacle-aware (apex reshape) est {est:.3f}s; "
                         f"{n_ap}/{n_obs} obstacle apex(es) reshaped, {len(dropped)} rejected, "
                         f"{n_clear} already clear, {n_no_apex} without a recorded reactive apex")
            # PER-APEX shape log. Without this a collapsed reach is invisible: the summary above
            # reports the hump as "reshaped" whether it was laid at the 5 m reach the search asked
            # for or bisected down to 1.2 m by the corridor fit — and the sharp one costs >1 s/lap
            # and exceeds curvlim. Log what was actually laid, and WARN when the reach came back
            # less than half of what was requested or the geometry is near the curvature limit.
            curvlim = float(res.get("curvlim", 0.0))
            for a in res.get("apex_laid", []):
                r_req = max(a.get("r_req", 0.0), 1e-6)
                shrink = min(a["r_in"], a["r_out"]) / r_req
                kp, kmsg = a.get("kappa_peak", 0.0), ""
                if curvlim > 0.0:
                    kmsg = f" ({kp / curvlim:.0%} of curvlim {curvlim:.2f})"
                # Clearance of the WOVEN profile — the number the acceptance floor is about, and
                # the one the reactive planner and the SM will independently re-derive downstream.
                gap = float(a.get("clear", float("nan")))
                cmsg = "" if gap != gap else f"; clears {gap:+.3f} m (floor {self.obs_margin:.2f})"
                line = (f"[static_reopt] apex @({a['xy'][0]:.2f},{a['xy'][1]:.2f}) "
                        f"laid d={a['laid']:+.3f} (want {a['want']:+.3f}) "
                        f"reach {a['r_in']:.2f}/{a['r_out']:.2f} m of {r_req:.2f} requested "
                        f"({shrink:.0%}); max|kappa| {kp:.2f}{kmsg}{cmsg}")
                if shrink < 0.5 or (curvlim > 0.0 and kp > 0.9 * curvlim):
                    self.get_logger().warning(
                        line + " <- SHARP: the corridor fit shrank this hump. Check "
                        "reopt_wall_margin / reopt_fit_tol and the reactive apex_bulge.")
                else:
                    self.get_logger().info(line)
            if dropped:
                # All-or-nothing: a hump that cannot be laid clear of its obstacle would NOT
                # relieve the reactive layer — the shrunken line got re-avoided every lap. Those
                # obstacles stay reactive-only, and this says so per apex, with the measured
                # clearance where the acceptance floor is what rejected it.
                def _det(d):
                    s = (f"@({d['xy'][0]:.2f},{d['xy'][1]:.2f}) want {d['want']:+.2f} "
                         f"max {d['fit']:+.2f} [{d.get('reason', 'corridor')}]")
                    if "clear" in d:
                        s += f" clears {d['clear']:+.2f} of {d['need']:.2f} needed"
                    return s
                self.get_logger().warning(
                    f"[static_reopt] {len(dropped)} apex(es) REJECTED (reason 'corridor' = track "
                    f"too tight, 'curvature' = the hump would exceed curvlim, 'clearance' = even "
                    f"grown wider it does not clear the box by obs_margin; reactive layer keeps "
                    f"handling them): {'; '.join(_det(d) for d in dropped)}")
            if n_no_apex:
                self.get_logger().warning(
                    f"[static_reopt] {n_no_apex} obstacle(s) had no recorded reactive apex yet — "
                    f"the reactive static-avoidance layer handles them until an apex is captured")
            # The floor each hump was ACCEPTED at travels with the bundle: the coverage ladder can
            # settle below obs_margin, and judging such a hump against obs_margin afterwards would
            # veto exactly the coverage the ladder just bought. Built BEFORE the veto so the veto
            # holds every hump to what it actually promised.
            obs_key = {id(o): k for o, k in zip(obstacles,
                                                self._keys_for(obstacles, self._obs_ids))}
            floors_by_id = {}
            for a in res.get("apex_laid", []):
                i_a = a.get("obs_i")
                if i_a is None or i_a >= len(pairs):
                    continue
                o_a = pairs[i_a][1]
                floors_by_id[id(o_a)] = float(a.get("floor", self.obs_margin))
                k = obs_key.get(id(o_a))
                if k is not None:
                    floor_by_key[k] = floors_by_id[id(o_a)]
            clearance_ok = self._check_line_clearance(traj, obstacles, res.get("apex_laid", []),
                                                      apex_obs=[o for _xy, o in pairs],
                                                      floors=floors_by_id)
            # ...and the wall, which no upstream check consults directly. Same refusal path.
            clearance_ok = self._wall_gate_ok(traj) and clearance_ok
            # Baseline for the drift trigger: what THIS line clears each box by, at the positions
            # it was built from. Travels with the bundle so the comparison is always against the
            # line that is actually active (a pending bundle may never commit).
            clearance_by_key = dict(zip(self._keys_for(obstacles, self._obs_ids),
                                        self._line_clearances(traj, obstacles)))
        else:
            info.data = (f"[static_reopt] obstacle-aware (mincurv_iqp) est {est:.3f}s; "
                         f"affected {rep.n_affected}, infeasible {rep.n_infeasible}, "
                         f"min_halfwidth {rep.min_halfwidth_seen:.3f}m")
        lap = Float32(); lap.data = float(est)

        # centerline + trackbounds are map-fixed -> reuse the clean ones
        return _Bundle(info, lap,
                       self.clean_bundle.cent_wpnts, self.clean_bundle.cent_markers,
                       glb_w, glb_m, sp_w, sp_m, self.clean_bundle.trackbounds,
                       n_apex=int(res.get("n_windows", 0)), clearance_ok=clearance_ok,
                       clearance_by_key=clearance_by_key, floor_by_key=floor_by_key,
                       coverage=coverage)

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

        if not self._clearance_drifted(obs, ids) and not self._obstacles_changed(obs):
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
            replayed = self._record_apexes_best_of(list(self._path_buffer))
            if replayed:
                self.get_logger().info(
                    f"[static_reopt] retro-associated apex(es) from {len(self._path_buffer)} "
                    f"buffered reactive path(s)")

    def clear_cb(self, _msg: Empty):
        self._apex_by_obs.clear()
        self._apex_miss.clear()
        self._path_buffer.clear()   # no retro-resurrection of just-cleared apexes
        self._clearance_dirty_keys.clear()
        if self._obstacles:
            self._obstacles = []
            self._obs_ids = []
            self._mark_dirty()
        self.get_logger().info("[static_reopt] obstacle set + apex records CLEARED by external request")

    def _mark_dirty(self, keep_pending: bool = False):
        """Arm the rebuild trigger, and by default discard any pending bundle.

        A pending built from the PREVIOUS obstacle/apex state is normally stale — it both blocked
        the fresh rebuild (the solve gate requires _pending is None) and, once committed, installed
        an outdated line. Observed: a spurious unlatch->re-confirm flap (set 1->0->1) left a pending
        CLEAN revert queued while the obstacle was confirmed again — the re-opt then never ran.

        `keep_pending` is for the case where that reasoning does not apply: a MINOR apex refinement,
        where the queued line and the line the rebuild would produce differ by centimetres. Throwing
        the pending away there costs a whole swap opportunity, because the commit gates (reactive
        idle + the lines agreeing over the controller's look-ahead) are hardest to satisfy exactly
        where the obstacles are — so the lap-2 swap slips to lap 3 for a 5 cm improvement nobody
        asked for. Keeping it lets the good-enough line commit now; the flag stays armed and the
        refinement lands on the next solve, which runs as soon as the pending clears.
        """
        self._obstacles_dirty = True
        self._dirty_since = self.get_clock().now().nanoseconds * 1e-9
        if not keep_pending:
            self._pending = None
            self._pending_dev = None

    def _clearance_drifted(self, obs: List[core.Obstacle], ids: List[int]) -> bool:
        """Has an obstacle drifted into the clearance of the line the car is FOLLOWING?

        `obs_change_tol` is a cause trigger: it fires on how far a box moved, and says nothing
        about whether the move mattered. Between the tolerance and the margin chain there is a
        band where a sub-tolerance drift is still enough to take the active line under the SM's
        static requirement — the line then reads BLOCKED, the car drops to TRAILING, and nothing
        ever re-solves because the obstacle "did not move". This is the consequence trigger for
        exactly that band: it fires on the quantity the downstream gates actually test.

        Only obstacles this line CLEARED when it was built can fire it. One that was already
        short (its hump was honestly dropped; the reactive layer owns it) would otherwise re-arm a
        deterministic rebuild that changes nothing, every frame, forever.

        Latched per key against the active line, because _mark_dirty discards the pending bundle:
        an unlatched trigger would re-fire until the new line commits and so prevent it from ever
        committing.
        """
        base = getattr(self.active, "clearance_by_key", None)
        if not base or not obs:
            return False
        try:
            _s, xa, ya = self._bundle_xy(self.active)
        except Exception:                       # a trigger must never break the callback
            return False
        thr = self.clearance_dirty_m
        drifted = False
        for o, key in zip(obs, self._keys_for(obs, ids)):
            was = base.get(key)
            if was is None or was < thr or key in self._clearance_dirty_keys:
                continue                        # never cleared by this line, or already reported
            now = float(np.min(np.hypot(xa - o.x, ya - o.y))) - float(o.r)
            if now >= thr:
                continue
            self._clearance_dirty_keys.add(key)
            drifted = True
            self.get_logger().warning(
                f"[static_reopt] obstacle @({o.x:.2f},{o.y:.2f}) drifted into the active line: "
                f"clearance {was:+.2f} -> {now:+.2f} m, under {thr:.2f} — re-optimizing before "
                f"the state machine reads the line as blocked")
        return drifted

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
        went_idle = self._reactive_active and not active
        if went_idle:
            self._reactive_idle_t = now
        self._reactive_active = active
        # ARM THE REBUILD ONLY WHEN THE MANEUVER IS OVER. A new/better apex is a reason to
        # rebuild -- without it an obstacle set collected BEFORE its apex existed leaves the solve
        # gate permanently closed -- but arming it MID-MANEUVER is what made the trigger harmful.
        # The reactive planner republishes at 20 Hz while the car is driving the avoidance, so a
        # record that keeps refining armed _mark_dirty on nearly every message, and each call with
        # a major change discarded the pending bundle: the very swap the records were being
        # collected for could not land. Wait for the reactive layer to go idle, which is exactly
        # "the maneuver this apex describes has finished", and rebuild once with the settled
        # record. A MINOR refinement still keeps any pending bundle alive (see _mark_dirty).
        # BEFORE the empty-message return below: an empty publish is the most common way the
        # reactive layer goes idle, and returning first would swallow the trigger entirely.
        if went_idle and self._apex_dirty_pending:
            self._apex_dirty_pending = False
            self.get_logger().info(
                "[static_reopt] reactive layer went idle with a new/updated apex -> rebuilding",
                throttle_duration_sec=2.0)
            self._mark_dirty(keep_pending=not self._apex_change_major)
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
            self._apex_dirty_pending = True

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
    def _apex_candidate(self, wx, wy, wd, owner, idx, o):
        """The apex ONE reactive path offers for ONE obstacle, or None.

        Every rejection here exists because a bad record is unrecoverable: the apex sets where the
        hump is centred, and a hump centred in the wrong place misses its box by a few centimetres
        with no downstream check able to put it back. Forensics on the failing run: records drifted
        up to 1.0 m DOWNSTREAM of the obstacle, and the resulting humps were 3-12 cm short.

        The predicates, in the order they fire:

          SPAN. The path must extend past the obstacle's abeam station in BOTH directions. The
          committed-path reuse in the reactive planner republishes the slice still AHEAD of the
          car, so once the car is beside a box the republished slice is that box's exit RAMP --
          a decaying tail whose widest point sits downstream of the obstacle. Under newest-wins
          those tails overwrote good records every cycle. A path that genuinely passed the box
          starts before it and ends after it.

          UNDERSHOOT. The overshoot clamp below has always bounded how much WIDER than necessary a
          record may be. Nothing bounded how much narrower, so a decaying tail that survived the
          other gates was recorded as a legitimately tighter apex. Symmetric bound.

          ABEAM. The recorded point must sit beside the obstacle along the track: 0.5 m, down from
          1.0. Measured on the run, healthy records were all within 0.1 m and the polluted ones sat
          at 0.4-1.0 m, so the old gate admitted exactly the bad half.
        """
        mine = np.where(owner == idx)[0]
        if mine.size == 0:
            return None
        dist = np.hypot(wx[mine] - o.x, wy[mine] - o.y)
        j = int(mine[int(np.argmax(np.abs(wd[mine])))])
        if float(dist.min()) > self._apex_assoc_tol or abs(wd[j]) < self._apex_min_d:
            return None
        if float(np.hypot(wx[j] - o.x, wy[j] - o.y)) > self._apex_assoc_tol:
            return None
        d_rec, i_rec, _left = self._clean_offset(float(wx[j]), float(wy[j]))
        d_obs, i_obs, left_obs = self._clean_offset(float(o.x), float(o.y))
        n_cl = len(self._clean_xy)
        st_m = self._track_len / max(n_cl - 1, 1)          # metres per station
        # --- SPAN -------------------------------------------------------------------------
        _d0, i_first, _l0 = self._clean_offset(float(wx[0]), float(wy[0]))
        _d1, i_last, _l1 = self._clean_offset(float(wx[-1]), float(wy[-1]))
        margin = max(1, int(self.apex_span_margin_m / max(st_m, 1e-6)))
        fwd_len = (i_last - i_first) % n_cl
        fwd_obs = (i_obs - i_first) % n_cl
        if not (margin <= fwd_obs <= fwd_len - margin):
            return None
        # --- ABEAM ------------------------------------------------------------------------
        gap_st = abs(i_rec - i_obs)
        gap_st = min(gap_st, n_cl - gap_st) * st_m
        if gap_st > self.apex_abeam_gap_m:
            return None
        if not self._apex_plausible(float(wx[j]), float(wy[j])):
            self.get_logger().warning(
                f"[static_reopt] REJECTED implausible apex xy=({wx[j]:.2f},{wy[j]:.2f}) "
                f"d={wd[j]:+.2f}m for obstacle @({o.x:.2f},{o.y:.2f}) — outside the track "
                f"corridor (displaced-frame outlier)", throttle_duration_sec=2.0)
            return None
        # --- OVERSHOOT clamp / UNDERSHOOT reject ------------------------------------------
        # `need` is what the obstacle REQUIRES: its own offset + radius + keep-out + bulge + slack.
        side = 1.0 if d_rec >= d_obs else -1.0
        need = d_obs + side * (float(o.r) + 0.45)
        if (d_rec - need) * side > 0.0:                    # wider than required -> clamp onto it
            d_rec = need
        elif (need - d_rec) * side > self.apex_undershoot_m:
            return None                                     # narrower than required -> a tail
        # --- ANCHOR AT THE OBSTACLE -------------------------------------------------------
        # Store the point beside the OBSTACLE, not the point the path happened to be widest at.
        # d_rec survives as amplitude/side evidence; i_rec does not, because it is the thing that
        # drifts. This is what makes a stale record cost nothing: it can only be wrong about how
        # far out to go, never about where.
        ax_x = float(self._clean_xy[i_obs][0] + d_rec * left_obs[0])
        ax_y = float(self._clean_xy[i_obs][1] + d_rec * left_obs[1])
        return ax_x, ax_y, abs(float(d_rec)), float(need)

    def _commit_apex(self, key, o, cand) -> bool:
        """Store one accepted apex. Returns True when it is a rebuild-worthy change."""
        ax_x, ax_y, amp, _need = cand
        prev = self._apex_by_obs.get(key)
        self._apex_by_obs[key] = (ax_x, ax_y, amp)
        if prev is None:
            self._apex_change_major = True                 # a hump that did not exist before
            self.get_logger().info(
                f"[static_reopt] recorded reactive apex for obstacle @({o.x:.2f},{o.y:.2f}) "
                f"-> apex xy=({ax_x:.2f},{ax_y:.2f}) |d|={amp:.2f}m (anchored abeam)")
            return True
        d_amp = abs(amp - prev[2])
        d_pos = float(np.hypot(prev[0] - ax_x, prev[1] - ax_y))
        if d_amp > 0.05 or d_pos > 0.10:
            if d_amp >= self.apex_major_change_m or d_pos >= self.apex_major_change_m:
                self._apex_change_major = True
            return True
        return False

    def _apex_frozen(self, key, o) -> bool:
        """Is this obstacle's apex settled, so further reactive paths must not touch it?

        Once the ACTIVE line carries a hump for an obstacle and that hump still clears it by the
        floor it was accepted at, the record has done its job. Re-recording from every subsequent
        reactive publish can only move it -- and the reactive planner is still publishing while the
        car drives the very hump the line already provides, so those publishes are the least
        trustworthy ones there are. Released by the two triggers that mean the world changed: the
        obstacle moving more than obs_change_tol, or the clearance-drift check.
        """
        floor = getattr(self.active, "floor_by_key", {}).get(key)
        if floor is None:
            return False
        try:
            _s, xa, ya = self._bundle_xy(self.active)
            now = float(np.min(np.hypot(xa - o.x, ya - o.y))) - float(o.r)
        except Exception:
            return False
        return now >= floor - 1e-9

    def _record_apexes(self, wx, wy, wd) -> bool:
        """Associate one reactive path with the confirmed obstacles and update the apex records.
        Returns True when any apex was newly recorded or MOVED by more than 5 cm in amplitude
        (or 10 cm in position) — a rebuild-worthy change. Also sets `_apex_change_major`, which
        separates "a new obstacle to lay a hump for" from "the same hump, a few centimetres
        different": only the former is worth discarding a pending bundle over (see _mark_dirty).

        NEWEST-WINS among paths that pass _apex_candidate: the historical max RATCHETED (one
        outlier permanently inflated the hump), and the acceptance predicates are what keep
        "newest" from meaning "whatever the exit ramp said last".

        OWNERSHIP first: give every spline point to its NEAREST obstacle, then let each obstacle
        pick its apex only from the points it owns — a second obstacle sitting inside the FIRST
        one's avoidance hump must not record that hump as its own apex."""
        self._apex_change_major = False
        keys = self._keys_for(self._obstacles, self._obs_ids)
        if not self._obstacles:
            return False
        ox = np.fromiter((o.x for o in self._obstacles), float, len(self._obstacles))
        oy = np.fromiter((o.y for o in self._obstacles), float, len(self._obstacles))
        owner = np.argmin(np.hypot(wx[:, None] - ox[None, :], wy[:, None] - oy[None, :]), axis=1)
        changed = False
        for idx, (o, key) in enumerate(zip(self._obstacles, keys)):
            if self._apex_frozen(key, o):
                continue
            cand = self._apex_candidate(wx, wy, wd, owner, idx, o)
            if cand is None:
                continue
            changed = self._commit_apex(key, o, cand) or changed
        return changed

    def _record_apexes_best_of(self, paths) -> bool:
        """Retro association over the buffered paths: take the BEST candidate per obstacle, not
        the last one.

        Replaying the buffer newest-last meant the most recent path won even when an earlier one
        had passed the obstacle properly and the latest was its decaying tail. With the acceptance
        predicates in place the tails are rejected outright, but among survivors "newest" is still
        arbitrary -- so pick the one whose amplitude is closest to what the obstacle geometrically
        requires."""
        keys = self._keys_for(self._obstacles, self._obs_ids)
        if not self._obstacles:
            return False
        ox = np.fromiter((o.x for o in self._obstacles), float, len(self._obstacles))
        oy = np.fromiter((o.y for o in self._obstacles), float, len(self._obstacles))
        best = {}
        for (_t, wx, wy, wd) in paths:
            owner = np.argmin(np.hypot(wx[:, None] - ox[None, :], wy[:, None] - oy[None, :]), axis=1)
            for idx, (o, key) in enumerate(zip(self._obstacles, keys)):
                if self._apex_frozen(key, o):
                    continue
                cand = self._apex_candidate(wx, wy, wd, owner, idx, o)
                if cand is None:
                    continue
                err = abs(cand[2] - abs(cand[3]))
                if key not in best or err < best[key][0]:
                    best[key] = (err, o, cand)
        changed = False
        for key, (_err, o, cand) in best.items():
            changed = self._commit_apex(key, o, cand) or changed
        return changed

    def _apex_pairs(self, obstacles: List[core.Obstacle]) -> List[tuple]:
        """((x, y) apex, obstacle) for every confirmed obstacle that has a recorded reactive apex.

        The pairing is what lets the core enforce a clearance floor: an apex on its own is just a
        point to interpolate through, while the obstacle it belongs to is the thing the laid hump
        has to measurably miss. Obstacles never reactively avoided contribute nothing (clean line
        kept there, reactive layer handles them)."""
        out = []
        for o, key in zip(obstacles, self._keys_for(obstacles, self._obs_ids)):
            rec = self._apex_by_obs.get(key)
            if rec is not None:
                out.append(((rec[0], rec[1]), o))
        return out

    def _apex_list(self, obstacles: List[core.Obstacle]) -> List[tuple]:
        """Map-frame (x,y) apex points only — the trigger gates just need to know one exists."""
        return [xy for xy, _o in self._apex_pairs(obstacles)]

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
        # The built line measurably fails to clear an obstacle it claims to have reshaped. Never
        # install it: from an obstacle-aware active it is a strict regression nothing downstream
        # can detect, and from the clean active it buys a detour that still needs the reactive
        # layer. Deterministic (same apexes + track -> same answer), so the trigger stays burned;
        # a new/better apex re-arms it via otwpnts_cb.
        if obstacles and not getattr(bundle, "clearance_ok", True):
            return
        n_ap = getattr(bundle, "n_apex", 0)
        if obstacles and n_ap == 0:
            self.get_logger().warning(
                f"[static_reopt] re-opt laid 0 humps for {len(obstacles)} obstacle(s) — every "
                f"apex rejected (track too tight, or the hump would exceed curvlim; see the "
                f"REJECTED log for the per-apex reason). "
                f"Keeping the current line; those obstacles stay reactive-only")
            return
        # COVERAGE POLICY. Publish the best coverage available, never a regression. A line that
        # covers two of three obstacles is worth swapping to -- the third stays the reactive
        # layer's and the coverage topic says so -- but a line that covers LESS than what is
        # already driving must not be installed, because the fallback here is not the clean line,
        # it is the older, better-covered line the car is on.
        regress = self._coverage_regresses(getattr(bundle, "coverage", []),
                                           getattr(self.active, "coverage", []))
        if regress:
            self.get_logger().warning(
                f"[static_reopt] rebuild REFUSED: it would reduce coverage — {regress}. Keeping "
                f"the active line; see /static_reopt/coverage", throttle_duration_sec=5.0)
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
        self._clearance_dirty_keys.clear()       # new line -> new clearance baseline to drift from
        # tell caching consumers to re-take the new geometry, repeatedly and FAST (notify_cb)
        self._notify_scaler_ticks = self.notify_ticks
        kind = "CLEAN" if bundle is self.clean_bundle else "OBSTACLE-AWARE"
        self.get_logger().info(f"[static_reopt] swapped to {kind} at s={s:.2f} m")
        self._publish_active(bundle)             # publish now, don't wait for the republish tick
        self._publish_coverage(bundle)
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
