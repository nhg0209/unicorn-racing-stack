#!/usr/bin/env python3
"""Lane-hold dynamic overtaking planner (node ``planner_change``).

Replaces the snapshot-anchored cosine evasion with a phased lane-change maneuver
so the published path no longer chases a moving opponent:

  IDLE  -> planner stays SILENT until the gap to the opponent has closed to
           engage_gap_m: the approach is owned by TRAILING (controller gap PID)
           or plain raceline driving, at full speed.
  OPEN  -> side latched once + lane offset sized from the opponent's actual
           lateral position (never below the SM free-check requirement); publish
           a quintic blend from the car onto the centerline-parallel lane.
  HOLD  -> keep following the lane regardless of where the opponent currently
           is. The offset auto-grows (slew-limited, wall-capped) if the opponent
           drifts toward our lane, so the SM free-check keeps passing during the
           whole approach instead of aborting OVERTAKE at mid gap.
  CLOSE -> once the opponent is passed (wrapped signed s-gap > pass_gap_m, held
           for pass_hyst_s), ramp back to the raceline at a latched s.

The lane geometry (centerline d(s) + signed offset) is stable by construction,
so the state machine's cached-path splicing sees a consistent trajectory instead
of a per-cycle re-anchored spline (the source of the old wobble). Velocities are
published as 0: the state machine velocity-replans every received path from its
curvature.

Publishes (unchanged interface):
    /planner/avoidance/otwpnts  OTWpntArray   evasion path (empty topic silence when IDLE)
    /planner/avoidance/merger   Float32MultiArray  [target s_end, path end s]
    /planner/avoidance/markers_sqp  MarkerArray    path viz
"""
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import (
    FloatingPointRange,
    ParameterDescriptor,
    ParameterType,
    SetParametersResult,
)
from rclpy.parameter import Parameter

from nav_msgs.msg import Odometry
from f110_msgs.msg import (
    Wpnt,
    WpntArray,
    Obstacle,
    ObstacleArray,
    OTWpntArray,
    OpponentTrajectory,
    BehaviorStrategy,
)
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, Float32, Bool, Header

from scipy.interpolate import BPoly
from frenet_conversion.frenet_converter import FrenetConverter

from ccma import CCMA
import trajectory_planning_helpers as tph
from transforms3d.euler import quat2euler
from grid_filter.grid_filter import GridFilter

PHASE_IDLE = "IDLE"
PHASE_OPEN = "OPEN"
PHASE_HOLD = "HOLD"
PHASE_CLOSE = "CLOSE"


def euler_from_quaternion(quaternion):
    """quaternion is [x, y, z, w]; returns (roll, pitch, yaw)."""
    x, y, z, w = quaternion
    return quat2euler([w, x, y, z])


class ChangeAvoidanceNode(Node):
    def __init__(self):
        super().__init__('change_avoidance_node')

        # --- lane / raceline data ---
        self.scaled_wpnts_msg = WpntArray()
        self.scaled_vmax = None
        self.scaled_max_idx = None
        self.scaled_max_s = None
        self.scaled_delta_s = None
        self.center_wpnts_msg = WpntArray()
        self.center_wpnts_received = False
        self.center_s = None            # centerline sampled in the raceline frenet frame
        self.center_d = None            # (any lane = center_d(s) + lane_sign * offset)
        self.global_waypoints = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.converter = None

        # --- perception state ---
        self.obs_all: List[Obstacle] = []
        self.obs_dynamic: List[Obstacle] = []
        self.opponent_waypoints = []
        self.opp_on_trajectory = False
        self.opponent_wpnts_sm = None

        # --- ego state ---
        self.current_s = None
        self.current_d = None
        self.current_vs = None
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.behavior_state = ""
        self.local_wpnts = None

        # --- maneuver state ---
        self.phase = PHASE_IDLE
        self.side = None                # latched at OPEN: 'left' | 'right'
        self.lane_sign = 0              # +1 = left of the centerline, -1 = right
        self.lane_offset_cur = 0.0      # current lane offset from the centerline [m]
        self.engaged_offset = 0.0       # offset chosen at engagement (floor for the ratchet)
        self.target = None              # dict: id/s/d/vs/size/last_seen
        self.pass_cnt = 0
        self.close_s = None             # wrapped s where the return ramp starts
        self.close_frozen = False
        self.blocked_since = None
        self.idle_until = 0.0           # re-engage block after finishing/aborting
        self.last_loop_t = None
        self._dt = 0.05
        self.mrk_decim = 0

        # --- constants ---
        self.width_car = 0.30
        self.ot_section_check = False
        self.ot_section_check_t = None

        # --- static params ---
        self.declare_parameter('measure', False,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.measure = self.get_parameter('measure').get_parameter_value().bool_value
        # Lane-change is only validated in overtaking sectors; outside them trailing is the
        # safe fallback in head-to-head, so the ENGAGE gate fails CLOSED when the SM stops
        # publishing. A maneuver already in progress is never aborted by sector state.
        self.declare_parameter('require_ot_section', True,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.require_ot_section = self.get_parameter('require_ot_section').get_parameter_value().bool_value
        self.declare_parameter('debug_plot', False,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.debug_plot = self.get_parameter('debug_plot').get_parameter_value().bool_value

        # --- tunable params (defaults; overridable via yaml / live reconfigure) ---
        self.lane_offset_m = 0.35       # MINIMUM lane offset from the centerline; the actual
                                        # offset is sized (and live-grown) from the opponent's
                                        # lateral position so the SM clearance always holds
        self.obs_traj_tresh = 1.0       # engage: max |obs d - ego d|
        self.spline_bound_mindist = 0.25
        self.engage_gap_m = 5.0         # stay silent (TRAILING closes in at speed) until the
                                        # s-gap to the target is below this
        self.offset_slew_mps = 0.6      # how fast the lane offset may grow/shrink [m/s]
        self.open_ramp_min_m = 3.0
        self.open_ramp_time_s = 0.8     # blend length = max(min, t * v)
        self.blend_min_m = 1.5          # HOLD re-anchor blend
        self.close_ramp_min_m = 3.0
        self.close_ramp_time_s = 0.8
        self.close_arm_m = 1.0          # ramp starts this far ahead of the car
        self.tail_m = 2.0               # raceline tail after the return ramp
        self.hold_horizon_m = 22.0      # published path length (> SM interest_horizon 20 m,
                                        # else an escaping opponent past the path end makes
                                        # the SM free-check read NOT-free and aborts)
        self.pass_gap_m = 1.2           # ego must lead by this to trigger CLOSE
        self.pass_hyst_s = 0.3
        self.sep_margin_m = 0.50        # required lateral clearance beyond obs half-size
                                        # (must exceed SM gb_ego_width/2 + lateral_width_m = 0.4)
        self.target_lost_s = 1.0
        self.reengage_block_s = 1.0
        self.kernel_size = 3            # GridFilter erosion (7 ate ~0.35 m and killed the lanes)

        self._declare_tunables()

        # CCMA init
        self.ccma = CCMA(w_ma=10, w_cc=5)

        # Publishers
        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers_sqp", QoSProfile(depth=10))
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", QoSProfile(depth=10))
        self.merger_pub = self.create_publisher(Float32MultiArray, "/planner/avoidance/merger", QoSProfile(depth=10))
        self.lanes_pub = self.create_publisher(MarkerArray, "/planner/avoidance/lanes", QoSProfile(depth=10))
        if self.measure:
            self.measure_pub = self.create_publisher(Float32, "/planner/pspliner_sqp/latency", QoSProfile(depth=10))

        # Subscribers
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obs_cb, QoSProfile(depth=10))
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, QoSProfile(depth=10))
        self.create_subscription(Odometry, "/car_state/odom", self.state_cartesian_cb, QoSProfile(depth=10))
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, QoSProfile(depth=10))
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, QoSProfile(depth=10))
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.scaled_wpnts_cb, QoSProfile(depth=10))
        self.create_subscription(WpntArray, "/centerline_waypoints", self.center_wpnts_cb, QoSProfile(depth=10))
        self.create_subscription(OpponentTrajectory, "/opponent_trajectory", self.opponent_trajectory_cb, QoSProfile(depth=10))
        self.create_subscription(Bool, "/ot_section_check", self.ot_sections_check_cb, QoSProfile(depth=10))

        self.add_on_set_parameters_callback(self.dyn_param_cb)

        self.converter = self.initialize_converter()

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        # Wait for the centerline waypoints, then project them into the raceline frenet frame
        self.wait_for_message_attr('center_wpnts_received')
        self.build_center_table(center_wpnts=self.center_wpnts_msg)

        self.get_logger().info("[LaneChange] Waiting for messages and services...")
        self.wait_for_loop_messages()
        self.get_logger().info("[LaneChange] Ready!")

        self.create_timer(1.0 / 20.0, self.loop)

    #################### PARAMS ####################
    def _declare_tunables(self):
        def dbl(name, default, lo, hi, desc=""):
            self.declare_parameter(name, default, ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE, description=desc,
                floating_point_range=[FloatingPointRange(from_value=float(lo), to_value=float(hi), step=0.001)]))

        dbl('lane_offset_m', self.lane_offset_m, 0.15, 0.8, "minimum lane offset from centerline")
        dbl('obs_traj_tresh', self.obs_traj_tresh, 0.1, 2.0, "engage: max |obs d - ego d|")
        dbl('spline_bound_mindist', self.spline_bound_mindist, 0.05, 1.0, "min distance to track bounds")
        dbl('engage_gap_m', self.engage_gap_m, 2.0, 15.0, "trail until the target gap is below this")
        dbl('offset_slew_mps', self.offset_slew_mps, 0.1, 2.0, "lane offset grow/shrink rate")
        dbl('open_ramp_min_m', self.open_ramp_min_m, 1.0, 10.0, "min open blend length")
        dbl('open_ramp_time_s', self.open_ramp_time_s, 0.0, 3.0, "open blend = max(min, t*v)")
        dbl('blend_min_m', self.blend_min_m, 0.5, 5.0, "HOLD re-anchor blend length")
        dbl('close_ramp_min_m', self.close_ramp_min_m, 1.0, 10.0, "min close ramp length")
        dbl('close_ramp_time_s', self.close_ramp_time_s, 0.0, 3.0, "close ramp = max(min, t*v)")
        dbl('close_arm_m', self.close_arm_m, 0.0, 5.0, "close ramp starts this far ahead of car")
        dbl('tail_m', self.tail_m, 0.0, 10.0, "raceline tail after the return ramp")
        dbl('hold_horizon_m', self.hold_horizon_m, 10.0, 40.0, "published path length")
        dbl('pass_gap_m', self.pass_gap_m, 0.3, 5.0, "lead needed to trigger the return")
        dbl('pass_hyst_s', self.pass_hyst_s, 0.0, 2.0, "pass condition dwell time")
        dbl('sep_margin_m', self.sep_margin_m, 0.2, 1.5, "clearance beyond obstacle half-size")
        dbl('target_lost_s', self.target_lost_s, 0.2, 5.0, "coast time before target is dropped")
        dbl('reengage_block_s', self.reengage_block_s, 0.0, 5.0, "idle time after finishing")

    def dyn_param_cb(self, params: List[Parameter]):
        for param in params:
            if param.name in (
                    'lane_offset_m', 'obs_traj_tresh', 'spline_bound_mindist',
                    'engage_gap_m', 'offset_slew_mps',
                    'open_ramp_min_m', 'open_ramp_time_s', 'blend_min_m',
                    'close_ramp_min_m', 'close_ramp_time_s', 'close_arm_m', 'tail_m',
                    'hold_horizon_m', 'pass_gap_m', 'pass_hyst_s', 'sep_margin_m',
                    'target_lost_s', 'reengage_block_s'):
                setattr(self, param.name, param.value)
        self.get_logger().info(
            f"[LaneChange] params: lane_offset_min={self.lane_offset_m:.2f} "
            f"engage_gap={self.engage_gap_m:.1f} pass_gap={self.pass_gap_m:.2f} "
            f"sep_margin={self.sep_margin_m:.2f} hold_horizon={self.hold_horizon_m:.1f}")
        return SetParametersResult(successful=True)

    #################### CALLBACKS ####################
    def obs_cb(self, data: ObstacleArray):
        self.obs_all = list(data.obstacles)
        self.obs_dynamic = [o for o in data.obstacles if not o.is_static]

    def state_frenet_cb(self, data: Odometry):
        self.current_s = data.pose.pose.position.x
        self.current_d = data.pose.pose.position.y
        self.current_vs = data.twist.twist.linear.x

    def state_cartesian_cb(self, data: Odometry):
        self.current_x = data.pose.pose.position.x
        self.current_y = data.pose.pose.position.y
        quaternion = [data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        _, _, self.current_yaw = euler_from_quaternion(quaternion)

    def gb_cb(self, data: WpntArray):
        new_wp = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        changed = (self.global_waypoints is None or new_wp.shape != self.global_waypoints.shape
                   or not np.allclose(new_wp, self.global_waypoints))
        self.global_waypoints = new_wp
        self.gb_max_idx = data.wpnts[-1].id
        self.gb_max_s = data.wpnts[-1].s_m
        # The global line can CHANGE at runtime (static re-opt swap): rebuild the converter AND
        # re-project the centerline table into the CURRENT raceline frame.
        if changed and self.converter is not None:
            self.converter = self.initialize_converter()
            if self.center_wpnts_received:
                self.build_center_table(center_wpnts=self.center_wpnts_msg)

    def scaled_wpnts_cb(self, data: WpntArray):
        self.scaled_wpnts_msg = data
        v_max = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
        if self.scaled_vmax != v_max:
            self.scaled_vmax = v_max
            self.scaled_max_idx = data.wpnts[-1].id
            self.scaled_max_s = data.wpnts[-1].s_m
            self.scaled_delta_s = data.wpnts[1].s_m - data.wpnts[0].s_m

    def center_wpnts_cb(self, data: WpntArray):
        self.center_wpnts_msg = data
        self.center_wpnts_received = True

    def behavior_cb(self, data: BehaviorStrategy):
        self.behavior_state = data.state
        self.local_wpnts = data.local_wpnts

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.opponent_waypoints = data.oppwpnts
        self.opp_on_trajectory = bool(data.opp_is_on_trajectory)
        self.opponent_wpnts_sm = np.array([wpnt.s_m for wpnt in data.oppwpnts])

    def ot_sections_check_cb(self, data: Bool):
        self.ot_section_check = data.data
        self.ot_section_check_t = self.now_sec()

    #################### SMALL HELPERS ####################
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _sdiff(self, a: float, b: float) -> float:
        """Signed wrapped s difference a-b in (-L/2, L/2]."""
        L = self.scaled_max_s
        return ((a - b + L / 2.0) % L) - L / 2.0

    def wait_for_message_attr(self, attr_name: str):
        while not getattr(self, attr_name, False):
            rclpy.spin_once(self)

    def wait_for_loop_messages(self):
        while (self.scaled_max_s is None or self.current_x is None
               or self.current_s is None or self.local_wpnts is None):
            rclpy.spin_once(self)

    def initialize_converter(self) -> "FrenetConverter":
        while self.global_waypoints is None:
            rclpy.spin_once(self)
        converter = FrenetConverter(self.global_waypoints[:, 0], self.global_waypoints[:, 1])
        self.get_logger().info("[LaneChange] initialized FrenetConverter object")
        return converter

    #################### LANE GENERATION (kept from the original node) ####################
    def resample_lane(self, xy: np.ndarray, resolution: float = 0.1) -> np.ndarray:
        deltas = np.diff(xy, axis=0)
        dists = np.hypot(deltas[:, 0], deltas[:, 1])
        s = np.concatenate([[0], np.cumsum(dists)])
        s_new = np.arange(0, s[-1], resolution)
        x_new = np.interp(s_new, s, xy[:, 0])
        y_new = np.interp(s_new, s, xy[:, 1])
        return np.stack([x_new, y_new], axis=1)

    def _center_d(self, s_query):
        """Centerline lateral offset in the raceline frenet frame, interpolated over UNWRAPPED s.

        Any overtaking lane is center_d(s) + lane_sign * lane_offset_cur, so the offset can be
        re-sized per engagement (and grown live) without regenerating any geometry.
        """
        L = self.center_s[-1] + (self.center_s[1] - self.center_s[0]) if self.scaled_max_s is None \
            else self.scaled_max_s
        cs_ext = np.concatenate([self.center_s, self.center_s + L, self.center_s + 2 * L])
        cd_ext = np.concatenate([self.center_d, self.center_d, self.center_d])
        return np.interp(s_query, cs_ext, cd_ext)

    def build_center_table(self, center_wpnts: WpntArray):
        xy = np.array([[w.x_m, w.y_m] for w in center_wpnts.wpnts])
        xy = self.resample_lane(xy, resolution=0.1)
        cs, cd = self.converter.get_frenet(xy[:, 0], xy[:, 1])
        order = np.argsort(np.asarray(cs))
        self.center_s = np.asarray(cs)[order]
        self.center_d = np.asarray(cd)[order]
        self.get_logger().info(
            f"[LaneChange] centerline table built: {len(self.center_s)} pts, "
            f"d range [{float(np.min(self.center_d)):.2f}, {float(np.max(self.center_d)):.2f}]")
        self._publish_lane_markers()

    def _publish_lane_markers(self):
        """Draw the two MINIMUM-offset lanes (the live offset may sit further out)."""
        if self.center_s is None:
            return
        mrks = MarkerArray()
        for i, sign in enumerate((+1, -1)):
            resp = self.converter.get_cartesian(self.center_s, self.center_d + sign * self.lane_offset_m)
            resp = resp.T if resp.ndim == 2 else resp
            mrk = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
            mrk.ns = "ot_lanes"
            mrk.id = i
            mrk.type = Marker.LINE_STRIP
            mrk.action = Marker.ADD
            mrk.pose.orientation.w = 1.0
            mrk.scale.x = 0.03
            mrk.color.a = 0.6
            mrk.color.r = 0.1 if sign > 0 else 0.9
            mrk.color.b = 0.9 if sign > 0 else 0.1
            mrk.color.g = 0.5
            mrk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.02) for p in resp]
            mrks.markers.append(mrk)
        self.lanes_pub.publish(mrks)

    #################### TARGET TRACKING ####################
    def _update_target(self, now: float, dt: float):
        if self.target is None:
            return
        match = None
        for o in self.obs_dynamic:
            if o.id == self.target['id']:
                match = o
                break
        if match is None:
            # id churn in the tracker: re-associate by nearest wrapped s
            cands = [(abs(self._sdiff(o.s_center, self.target['s'])), o) for o in self.obs_dynamic]
            cands = [c for c in cands if c[0] < 3.0]
            if cands:
                match = min(cands, key=lambda c: c[0])[1]
        if match is not None:
            self.target.update(id=match.id, s=match.s_center, d=match.d_center,
                               vs=match.vs, size=max(match.size, 0.25), last_seen=now)
        else:
            # coast on the EKF velocity so the pass check keeps evolving while occluded
            self.target['s'] = (self.target['s'] + self.target['vs'] * dt) % self.scaled_max_s

    def _target_lost_for(self, now: float) -> float:
        return now - self.target['last_seen'] if self.target is not None else float('inf')

    def _pick_engage_target(self) -> Optional[dict]:
        """Nearest dynamic obstacle ahead once TRAILING has closed the gap to engage_gap_m.
        Beyond that gap the planner stays silent: the approach happens at trailing/raceline
        speed, not crawling behind a premature avoidance path."""
        best = None
        for o in self.obs_dynamic:
            if not o.is_visible:
                continue
            gap = (o.s_center - self.current_s) % self.scaled_max_s
            if gap > self.engage_gap_m:
                continue
            if abs(o.d_center - self.current_d) > self.obs_traj_tresh:
                continue
            if best is None or gap < best[0]:
                best = (gap, o)
        if best is None:
            return None
        o = best[1]
        return dict(id=o.id, s=o.s_center, d=o.d_center, vs=o.vs,
                    size=max(o.size, 0.25), last_seen=self.now_sec())

    #################### SIDE SELECTION ####################
    def _opp_d_band(self, s_query: np.ndarray) -> np.ndarray:
        """Expected opponent lateral position over s (their line if known, else current d)."""
        if self.opp_on_trajectory and len(self.opponent_waypoints) > 10:
            os_ = np.array([w.s_m for w in self.opponent_waypoints])
            od_ = np.array([w.d_m for w in self.opponent_waypoints])
            order = np.argsort(os_)
            os_, od_ = os_[order], od_[order]
            L = self.scaled_max_s
            os_ext = np.concatenate([os_, os_ + L, os_ + 2 * L])
            od_ext = np.concatenate([od_, od_, od_])
            return np.interp(s_query, os_ext, od_ext)
        return np.full_like(s_query, self.target['d'])

    def _choose_lane(self) -> bool:
        """Latch the overtaking side AND size the lane offset from the opponent's expected
        occupancy: offset >= what keeps the SM free-check clearance (sep_margin beyond the
        obstacle half-size) over the pass window, floored at lane_offset_m, capped by the
        walls. Returns False when neither side fits."""
        if self.scaled_max_s is None or self.target is None:
            return False
        v = max(self.current_vs or 0.0, 1.0)
        window = max(8.0, 2.0 * v)     # s-range the pass will roughly occupy
        s_lin = self.current_s + np.arange(0.0, window, 0.5)
        opp_d = self._opp_d_band(s_lin)
        c = self._center_d(s_lin)
        need = self.target['size'] / 2.0 + self.sep_margin_m

        wp = self.scaled_wpnts_msg.wpnts
        idxs = ((s_lin % self.scaled_max_s) / self.scaled_delta_s).astype(int) % self.scaled_max_idx
        d_left = np.array([wp[j].d_left for j in idxs])
        d_right = np.array([wp[j].d_right for j in idxs])
        wall_need = self.width_car / 2.0 + self.spline_bound_mindist

        cands = {}
        off_req = max(float(np.max(need + opp_d - c)), self.lane_offset_m)
        off_max = float(np.min(d_left - wall_need - c))
        if off_req <= off_max:
            cands['left'] = (off_max - off_req, +1, off_req)
        off_req = max(float(np.max(need - opp_d + c)), self.lane_offset_m)
        off_max = float(np.min(c + d_right - wall_need))
        if off_req <= off_max:
            cands['right'] = (off_max - off_req, -1, off_req)
        if not cands:
            return False
        # slight stickiness to the previously used side to damp flip-flop between engagements
        if self.side in cands:
            head, sign, off = cands[self.side]
            cands[self.side] = (head + 0.05, sign, off)
        side = max(cands, key=lambda k: cands[k][0])
        _, self.lane_sign, off = cands[side]
        self.side = side
        self.engaged_offset = off
        self.lane_offset_cur = off
        return True

    def _update_offset(self, dt: float):
        """Live-adjust the lane offset while approaching/beside the target: grow (slew-limited)
        if the opponent drifts toward our lane so the SM free-check keeps passing, shrink back
        toward the engaged offset when room returns; always wall-capped. Frozen once the
        target has been passed (CLOSE keeps a stable lane to ramp off from)."""
        if self.target is None or self.lane_sign == 0:
            return
        rel = self._sdiff(self.target['s'], self.current_s)
        if rel < -self.pass_gap_m:
            return
        need = self.target['size'] / 2.0 + self.sep_margin_m
        c_t = float(self._center_d(np.array([self.current_s + max(rel, 0.0)]))[0])
        if self.lane_sign > 0:
            off_need = need + self.target['d'] - c_t
        else:
            off_need = need - self.target['d'] + c_t

        s_lin = self.current_s + np.arange(0.0, 8.0, 0.5)
        c = self._center_d(s_lin)
        wp = self.scaled_wpnts_msg.wpnts
        idxs = ((s_lin % self.scaled_max_s) / self.scaled_delta_s).astype(int) % self.scaled_max_idx
        wall_need = self.width_car / 2.0 + self.spline_bound_mindist
        if self.lane_sign > 0:
            off_max = float(np.min(np.array([wp[j].d_left for j in idxs]) - wall_need - c))
        else:
            off_max = float(np.min(c + np.array([wp[j].d_right for j in idxs]) - wall_need))

        tgt = float(np.clip(max(self.engaged_offset, off_need, self.lane_offset_m),
                            0.0, max(off_max, 0.0)))
        step = self.offset_slew_mps * dt
        self.lane_offset_cur += float(np.clip(tgt - self.lane_offset_cur, -step, step))

    #################### PATH BUILD + PUBLISH ####################
    def _lane_at(self, s: float) -> float:
        return float(self._center_d(np.array([s]))[0]) + self.lane_sign * self.lane_offset_cur

    def _lane_slope(self, s: float) -> float:
        return (self._lane_at(s + 0.3) - self._lane_at(s - 0.3)) / 0.6

    def _build_path(self, closing: bool) -> Optional[dict]:
        """Sample the maneuver path from the car forward. Returns dict with arrays or None."""
        L = self.scaled_max_s
        s0 = self.current_s
        v = max(self.current_vs or 0.0, 1.0)

        horizon = self.hold_horizon_m
        close_len = max(self.close_ramp_min_m, self.close_ramp_time_s * v)
        cs_u = None
        if closing and self.close_s is not None:
            cs_u = s0 + ((self.close_s - s0) % L)
            horizon = max(horizon, (cs_u - s0) + close_len + self.tail_m + 1.0)
        horizon = min(horizon, L * 0.9)

        n = max(int(horizon / 0.1), 20)
        s_lin = s0 + np.arange(n) * 0.1
        d_arr = self._center_d(s_lin) + self.lane_sign * self.lane_offset_cur

        # --- entry blend from the actual car state ---
        e_psi = float(self.converter.get_e_psi(self.current_x, self.current_y, self.current_yaw))
        dp0 = float(np.tan(np.clip(e_psi, -0.5, 0.5)))
        d_gap = abs((self.current_d or 0.0) - float(d_arr[0]))
        open_len = max(self.open_ramp_min_m, self.open_ramp_time_s * v)
        blend_len = float(np.clip(d_gap * 8.0, self.blend_min_m, open_len))
        # engaging close behind the target: finish the lane change BEFORE reaching it, else the
        # SM free-check sees the still-blending (near-raceline) part at the target's s -> not
        # free -> the commit flaps and the car crawls behind the opponent
        if self.phase == PHASE_OPEN and self.target is not None:
            gap_t = -self._sdiff(self.current_s, self.target['s'])
            if gap_t > 0.0:
                blend_len = min(blend_len, max(1.0, 0.85 * gap_t))

        ce = None
        direct_close = closing and cs_u is not None and (cs_u - s0) < blend_len + 0.5
        if direct_close:
            # too little room to rejoin the lane first: ramp straight home from the car
            ce = max(cs_u, s0 + 0.5) + close_len
            bp = BPoly.from_derivatives([s0, ce], [[self.current_d, dp0, 0.0], [0.0, 0.0, 0.0]])
            m = s_lin <= ce
            d_arr[m] = bp(s_lin[m])
            d_arr[s_lin > ce] = 0.0
        else:
            sb = s0 + blend_len
            bp = BPoly.from_derivatives(
                [s0, sb],
                [[self.current_d, dp0, 0.0], [self._lane_at(sb), self._lane_slope(sb), 0.0]])
            m = s_lin <= sb
            d_arr[m] = bp(s_lin[m])
            if closing and cs_u is not None:
                cs_u = max(cs_u, sb + 0.5)
                ce = cs_u + close_len
                bp2 = BPoly.from_derivatives(
                    [cs_u, ce],
                    [[self._lane_at(cs_u), self._lane_slope(cs_u), 0.0], [0.0, 0.0, 0.0]])
                m2 = (s_lin >= cs_u) & (s_lin <= ce)
                d_arr[m2] = bp2(s_lin[m2])
                d_arr[s_lin > ce] = 0.0

        # --- clip to the track corridor (never demand more room than the walls give) ---
        wp = self.scaled_wpnts_msg.wpnts
        idxs = ((s_lin % L) / self.scaled_delta_s).astype(int) % self.scaled_max_idx
        d_left = np.array([wp[j].d_left for j in idxs])
        d_right = np.array([wp[j].d_right for j in idxs])
        wall_need = self.width_car / 2.0 + self.spline_bound_mindist
        d_arr = np.clip(d_arr, -(d_right - wall_need), d_left - wall_need)

        return dict(s_lin=s_lin, d_arr=d_arr, cs_u=cs_u, ce=ce)

    def _publish_path(self, path: dict) -> bool:
        """Convert, safety-check, smooth and publish. Returns False if the path is infeasible."""
        L = self.scaled_max_s
        s_wrapped = path['s_lin'] % L
        resp = self.converter.get_cartesian(s_wrapped, path['d_arr'])
        resp = resp.T if resp.ndim == 2 else resp
        samples = np.asarray(resp, dtype=float).reshape(-1, 2)
        if samples.shape[0] < 3 or not np.all(np.isfinite(samples)):
            return False

        for i in range(samples.shape[0]):
            if not self.map_filter.is_point_inside(samples[i, 0], samples[i, 1]):
                return False

        smoothed_xy = self.ccma.filter(samples)
        smoothed_sd = self.converter.get_frenet(smoothed_xy[:, 0], smoothed_xy[:, 1])
        evasion_x = np.asarray(smoothed_xy[:, 0])
        evasion_y = np.asarray(smoothed_xy[:, 1])
        evasion_s = np.asarray(smoothed_sd[0]) % L
        evasion_d = np.asarray(smoothed_sd[1])
        evasion_coords = np.column_stack((evasion_x, evasion_y))

        # Guard: drop coincident points (zero-length segments -> NaN psi/kappa/velocity).
        if len(evasion_coords) >= 2:
            seg = np.linalg.norm(np.diff(evasion_coords, axis=0), axis=1)
            keep = np.concatenate([[True], seg > 1e-6])
            if not np.all(keep):
                evasion_x = evasion_x[keep]
                evasion_y = evasion_y[keep]
                evasion_s = evasion_s[keep]
                evasion_d = evasion_d[keep]
                evasion_coords = evasion_coords[keep]
        if len(evasion_coords) < 3 or not np.all(np.isfinite(evasion_coords)):
            return False

        evasion_psi, evasion_kappa = tph.calc_head_curv_num.calc_head_curv_num(
            path=evasion_coords,
            el_lengths=0.1 * np.ones(len(evasion_coords) - 1),
            is_closed=False,
        )
        evasion_psi += np.pi / 2

        msg = OTWpntArray(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
        msg.ot_side = self.side or ""
        # Velocity intentionally zero: the state machine velocity-replans every received
        # avoidance path from its curvature (update_velocity), planner owns geometry only.
        msg.wpnts = [
            Wpnt(id=i, s_m=float(s), d_m=float(d), x_m=float(x), y_m=float(y),
                 psi_rad=float(p), kappa_radpm=float(k), vx_mps=0.0)
            for i, (x, y, s, d, p, k) in enumerate(
                zip(evasion_x, evasion_y, evasion_s, evasion_d, evasion_psi, evasion_kappa))
        ]
        self.evasion_pub.publish(msg)

        if self.target is not None:
            self.merger_pub.publish(Float32MultiArray(
                data=[(self.target['s'] + self.target['size'] / 2.0) % L, float(evasion_s[-1])]))
        self._visualize_path(evasion_x, evasion_y)
        return True

    #################### SAFETY MONITORS ####################
    def _path_blocked_ahead(self, path: dict, now: float) -> bool:
        """True when an obstacle sits ON the maneuver path ahead (persistent).

        The latched target while BESIDE the car is exempt: bailing to the raceline
        next to an opponent is worse than holding the lane.
        """
        blocked = False
        for o in self.obs_all:
            rel = self._sdiff(o.s_center, self.current_s)
            if not (-0.5 < rel < 12.0):
                continue
            if (self.target is not None and o.id == self.target['id']
                    and rel < self.width_car * 2 + 0.4):
                continue   # beside / almost beside the latched target: no bail-out
            i = int(np.clip(rel / 0.1, 0, len(path['d_arr']) - 1))
            sep = abs(float(path['d_arr'][i]) - o.d_center) - max(o.size, 0.25) / 2.0
            if sep < self.sep_margin_m * 0.8:   # small hysteresis vs the side-selection margin
                blocked = True
                break
        if blocked:
            if self.blocked_since is None:
                self.blocked_since = now
        else:
            self.blocked_since = None
        return blocked and (now - self.blocked_since) > 0.3

    def _ot_gate_open(self) -> bool:
        if not self.require_ot_section:
            return True
        now = self.now_sec()
        if self.ot_section_check_t is None or (now - self.ot_section_check_t) > 1.0:
            # The SM only publishes /ot_section_check on some transition paths, so staleness
            # here can also mean "SM idle in GB_TRACK" -- still fail closed, trailing is safe.
            self.get_logger().warn(
                "[LaneChange] /ot_section_check stale or never received -- lane change suppressed",
                throttle_duration_sec=5.0)
            return False
        return self.ot_section_check

    #################### PHASE MACHINE ####################
    def _to_idle(self, now: float, why: str):
        self.get_logger().info(f"[LaneChange] {self.phase} -> IDLE ({why})")
        self.phase = PHASE_IDLE
        self.target = None
        self.close_s = None
        self.close_frozen = False
        self.pass_cnt = 0
        self.blocked_since = None
        self.idle_until = now + self.reengage_block_s
        self._clear_markers()

    def _passed(self) -> bool:
        if self.target is None:
            return False
        rel = self._sdiff(self.current_s, self.target['s'])   # >0: ego is ahead
        rel_v = (self.current_vs or 0.0) - self.target['vs']
        return rel > self.pass_gap_m and rel_v > -0.3

    def loop(self):
        start_time = time.perf_counter()
        now = self.now_sec()
        dt = 0.05 if self.last_loop_t is None else float(np.clip(now - self.last_loop_t, 0.01, 0.2))
        self.last_loop_t = now
        self._dt = dt

        self._update_target(now, dt)

        if self.phase == PHASE_IDLE:
            self._step_idle(now)
        elif self.phase == PHASE_OPEN:
            self._step_open(now)
        elif self.phase == PHASE_HOLD:
            self._step_hold(now)
        elif self.phase == PHASE_CLOSE:
            self._step_close(now)

        if self.measure:
            self.measure_pub.publish(Float32(data=time.perf_counter() - start_time))

    def _step_idle(self, now: float):
        if now < self.idle_until or not self._ot_gate_open():
            return
        target = self._pick_engage_target()
        if target is None:
            return
        self.target = target
        if not self._choose_lane():
            self.target = None
            return
        self.phase = PHASE_OPEN
        self.pass_cnt = 0
        self.get_logger().info(
            f"[LaneChange] IDLE -> OPEN: target id={target['id']} "
            f"gap={self._sdiff(target['s'], self.current_s):.1f} m "
            f"side={self.side} offset={self.lane_offset_cur:.2f} m")

    def _step_open(self, now: float):
        if self._target_lost_for(now) > self.target_lost_s:
            self._to_idle(now, "target lost")
            return
        rel = self._sdiff(self.current_s, self.target['s'])
        if rel < -(self.engage_gap_m + 4.0):
            self._to_idle(now, "target pulled away")
            return

        self._update_offset(self._dt)
        path = self._build_path(closing=False)
        if path is None or not self._check_lane_clear_vs_target(path):
            # while still clearly behind the target a safe re-selection is possible
            if rel < -(self.width_car * 2 + 0.6):
                self._to_idle(now, "lane no longer viable")
                return
        if path is None or not self._publish_path(path):
            self._to_idle(now, "path infeasible")
            return
        if self._path_blocked_ahead(path, now) and rel < -(self.width_car * 2 + 0.6):
            self._to_idle(now, "lane blocked ahead")
            return

        if abs((self.current_d or 0.0) - self._lane_at(self.current_s)) < 0.15:
            self.phase = PHASE_HOLD
            self.get_logger().info("[LaneChange] OPEN -> HOLD (on lane)")
        self._visualize_phase()

    def _step_hold(self, now: float):
        lost = self._target_lost_for(now)
        rel = self._sdiff(self.current_s, self.target['s']) if self.target else 0.0
        # lost while already ahead (typical: opponent left the lidar FOV behind us) counts as
        # passing; lost for a long time while still nominally behind means nothing left to pass
        passed_now = self._passed() or (lost > self.target_lost_s and rel > 0.0)
        if not passed_now and lost > 3.0 * self.target_lost_s:
            self._arm_close(now, "target gone")
            return

        self.pass_cnt = self.pass_cnt + 1 if passed_now else 0
        if self.pass_cnt >= max(int(self.pass_hyst_s * 20), 1):
            self._arm_close(now, f"passed (lead {rel:.1f} m)")
            return

        self._update_offset(self._dt)
        path = self._build_path(closing=False)
        if path is None or not self._publish_path(path):
            self._to_idle(now, "path infeasible")
            return
        if self._path_blocked_ahead(path, now) and rel < -(self.pass_gap_m + 0.5):
            # blocked and clearly behind the target again -> give up, SM falls back to TRAILING
            self._to_idle(now, "blocked, dropping back")
            return
        self._visualize_phase()

    def _arm_close(self, now: float, why: str):
        self.phase = PHASE_CLOSE
        self.close_s = (self.current_s + self.close_arm_m) % self.scaled_max_s
        self.close_frozen = False
        self.get_logger().info(f"[LaneChange] HOLD -> CLOSE ({why})")
        # publish the close path in the SAME cycle so the SM freshness gate never sees a gap
        self._step_close(now)

    def _step_close(self, now: float):
        L = self.scaled_max_s
        # opponent re-passed us while we were closing -> reopen the hold
        if self.target is not None and self._target_lost_for(now) < self.target_lost_s:
            rel = self._sdiff(self.current_s, self.target['s'])
            if rel < 0.3:
                self.phase = PHASE_HOLD
                self.close_s = None
                self.close_frozen = False
                self.pass_cnt = 0
                self.get_logger().info("[LaneChange] CLOSE -> HOLD (target re-passed)")
                return

        on_lane = abs((self.current_d or 0.0) - self._lane_at(self.current_s)) < 0.10
        if not self.close_frozen:
            if on_lane:
                # slide the ramp start so it always begins just ahead of the car: whenever the
                # SM adopts the newest path there is no lateral step
                cand = (self.current_s + self.close_arm_m) % L
                if self._sdiff(cand, self.close_s) > 0.0:
                    self.close_s = cand
            else:
                self.close_frozen = True   # car started ramping: freeze the geometry

        # postpone the merge while an obstacle occupies the return corridor (only while the
        # ramp is still sliding: once the car is ramping the geometry stays frozen)
        if not self.close_frozen:
            close_len = max(self.close_ramp_min_m,
                            self.close_ramp_time_s * max(self.current_vs or 0.0, 1.0))
            cs_rel = (self.close_s - self.current_s) % L
            for o in self.obs_all:
                rel_o = self._sdiff(o.s_center, self.current_s)
                if not (-0.5 < rel_o < cs_rel + close_len + 2.0):
                    continue
                lane_d_at_o = self._lane_at(self.current_s + max(rel_o, 0.0))
                corridor_lo = min(0.0, lane_d_at_o) - 0.3
                corridor_hi = max(0.0, lane_d_at_o) + 0.3
                if corridor_lo - o.size / 2.0 < o.d_center < corridor_hi + o.size / 2.0:
                    push = rel_o + o.size / 2.0 + 0.5
                    cand = (self.current_s + max(self.close_arm_m, push)) % L
                    if self._sdiff(cand, self.close_s) > 0.0:
                        self.close_s = cand

        path = self._build_path(closing=True)
        if path is None or not self._publish_path(path):
            self._to_idle(now, "close path infeasible")
            return
        self._visualize_phase()

        # merge complete: on the raceline and past the latched ramp region
        past_ramp = self._sdiff(self.current_s, self.close_s) > 0.5
        if path['ce'] is not None:
            past_ramp = past_ramp or self._sdiff(self.current_s, path['ce'] % L) > 0.0
        if abs(self.current_d or 0.0) < 0.12 and past_ramp:
            self._to_idle(now, "merge complete")

    def _check_lane_clear_vs_target(self, path: dict) -> bool:
        """The latched target must stay clear of the lane at ITS current s (it may have moved
        laterally since the side vote)."""
        if self.target is None:
            return True
        rel = self._sdiff(self.target['s'], self.current_s)
        if not (0.0 < rel < len(path['d_arr']) * 0.1):
            return True
        i = int(np.clip(rel / 0.1, 0, len(path['d_arr']) - 1))
        sep = abs(float(path['d_arr'][i]) - self.target['d']) - self.target['size'] / 2.0
        return sep >= self.sep_margin_m * 0.8

    #################### VIZ ####################
    def _clear_markers(self):
        mrks = MarkerArray()
        del_mrk = Marker(header=Header(stamp=self.get_clock().now().to_msg()))
        del_mrk.action = Marker.DELETEALL
        mrks.markers.append(del_mrk)
        self.mrks_pub.publish(mrks)

    def _visualize_path(self, xs, ys):
        self.mrk_decim = (self.mrk_decim + 1) % 4
        if self.mrk_decim != 0:
            return
        mrks = MarkerArray()
        mrk = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
        mrk.ns = "lane_change_path"
        mrk.id = 0
        mrk.type = Marker.LINE_STRIP
        mrk.action = Marker.ADD
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 0.06
        mrk.color.a = 1.0
        mrk.color.r = 0.63
        mrk.color.g = 0.13
        mrk.color.b = 0.94
        mrk.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in zip(xs, ys)]
        mrks.markers.append(mrk)
        self.mrks_pub.publish(mrks)

    def _visualize_phase(self):
        if self.mrk_decim != 0:
            return
        mrks = MarkerArray()
        mrk = Marker(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
        mrk.ns = "lane_change_phase"
        mrk.id = 1
        mrk.type = Marker.TEXT_VIEW_FACING
        mrk.action = Marker.ADD
        rel = self._sdiff(self.current_s, self.target['s']) if self.target else 0.0
        mrk.text = f"{self.phase} {self.side or ''} rel={rel:+.1f}m"
        mrk.pose.position.x = float(self.current_x)
        mrk.pose.position.y = float(self.current_y)
        mrk.pose.position.z = 0.8
        mrk.pose.orientation.w = 1.0
        mrk.scale.z = 0.35
        mrk.color.a = 1.0
        mrk.color.r = 1.0
        mrk.color.g = 1.0
        mrk.color.b = 1.0
        mrks.markers.append(mrk)
        self.mrks_pub.publish(mrks)


def main(args=None):
    rclpy.init(args=args)
    node = ChangeAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
