#!/usr/bin/env python3
"""tracking_quali — qualification tracking: every detection IS a static obstacle.

Quali has no opponent car, so multi_tracking's static/dynamic EKF classification
is unnecessary. This node forwards every /detect/raw_obstacles measurement as a
STATIC obstacle (is_static=True, is_visible=True, vs=vd=0) with frame-to-frame
stable ids (nearest-neighbour association in Frenet). Static-only means the
state machine always takes the static-avoidance OVERTAKING path and the
dynamic trailing behavior never triggers on quali obstacles.

Obstacles are avoided ONLY during the slow laps: once the slow phase is over
the crew starts clearing the track, so quali_lap_manager latches
/quali/obstacles_off=True and this node publishes an EMPTY obstacle array
from then on — remaining detections are people/props being carried away, and
the car runs the raceline at full pace ignoring everything.

Topics
  in : /detect/raw_obstacles   (ObstacleArray, Frenet filled at the source)
       /global_waypoints       (track length + cartesian for viz)
       /quali/obstacles_off    (Bool, TRANSIENT_LOCAL, from quali_lap_manager)
  out: /tracking/obstacles     (remapped to /tracking/obstacles_raw when the
                                virtual_perception merger owns the topic)
       /tracking/raw_obstacles (pass-through, same as multi_tracking)
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import Bool
from f110_msgs.msg import ObstacleArray, WpntArray
from visualization_msgs.msg import Marker, MarkerArray
from frenet_conversion.frenet_converter import FrenetConverter


class TrackingQuali(Node):
    def __init__(self):
        super().__init__('tracking')
        # frame-to-frame association gate (m, Frenet euclid) for stable ids
        self.declare_parameter('match_gate_m', 1.0)
        # keep a lost track alive this long so the state machine doesn't flap
        self.declare_parameter('hold_sec', 0.3)

        self.match_gate = float(self.get_parameter('match_gate_m').value)
        self.hold_sec = float(self.get_parameter('hold_sec').value)

        self.track_length = None
        self.converter = None
        self.obstacles_off = False
        self.tracks = {}          # id -> {'obs': Obstacle, 'last': Time}
        self.next_id = 0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(ObstacleArray, '/detect/raw_obstacles', self.obstacle_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.wpnts_cb, 10)
        self.create_subscription(Bool, '/quali/obstacles_off', self.obstacles_off_cb, latched)

        self.obstacles_pub = self.create_publisher(ObstacleArray, '/tracking/obstacles', 5)
        self.raw_pub = self.create_publisher(ObstacleArray, '/tracking/raw_obstacles', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/tracking/static_dynamic_marker_pub', 5)

        self.get_logger().info(
            f"[tracking_quali] all detections -> STATIC obstacles (overtake only, no trailing); "
            f"gate={self.match_gate}m hold={self.hold_sec}s "
            f"(seen during slow laps only; empty after /quali/obstacles_off)")

    def wpnts_cb(self, msg):
        if self.converter is not None or not msg.wpnts:
            return
        self.track_length = msg.wpnts[-1].s_m
        xs = np.array([w.x_m for w in msg.wpnts])
        ys = np.array([w.y_m for w in msg.wpnts])
        self.converter = FrenetConverter(xs, ys)
        self.get_logger().info("[tracking_quali] FrenetConverter ready")

    def obstacles_off_cb(self, msg):
        if msg.data != self.obstacles_off:
            self.get_logger().warn(
                f"[tracking_quali] obstacles_off -> {msg.data} "
                f"({'ALL obstacles ignored from now on' if msg.data else 'obstacles active'})")
        self.obstacles_off = msg.data

    def obstacle_cb(self, msg):
        now = self.get_clock().now()
        self.raw_pub.publish(msg)
        if self.track_length is None:
            return

        # after the slow phase the track is being cleared: ignore everything
        if self.obstacles_off:
            self.tracks.clear()
            out = ObstacleArray()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = 'map'
            self.obstacles_pub.publish(out)
            self.publish_markers(out)
            return

        detections = list(msg.obstacles)

        # greedy nearest-neighbour association against live tracks for stable ids
        unmatched = dict(self.tracks)
        pairs = []
        for det in detections:
            best_id, best_dist = None, self.match_gate
            for tid, tr in unmatched.items():
                ds = (det.s_center - tr['obs'].s_center + self.track_length / 2) \
                    % self.track_length - self.track_length / 2
                dist = math.hypot(ds, det.d_center - tr['obs'].d_center)
                if dist < best_dist:
                    best_id, best_dist = tid, dist
            if best_id is not None:
                pairs.append((best_id, det))
                del unmatched[best_id]
            else:
                pairs.append((self.next_id, det))
                self.next_id += 1

        for tid, det in pairs:
            det.id = tid
            det.is_static = True
            det.is_visible = True
            self.tracks[tid] = {'obs': det, 'last': now}

        # expire tracks not seen for hold_sec; held ones stay published meanwhile
        for tid in [t for t, tr in self.tracks.items()
                    if (now - tr['last']).nanoseconds * 1e-9 > self.hold_sec]:
            del self.tracks[tid]

        out = ObstacleArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'map'
        out.obstacles = [tr['obs'] for tr in self.tracks.values()]
        self.obstacles_pub.publish(out)
        self.publish_markers(out)

    def publish_markers(self, out):
        marks = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        marks.markers.append(wipe)
        if self.converter is None:
            self.marker_pub.publish(marks)
            return
        for o in out.obstacles:
            xy = self.converter.get_cartesian(np.array([o.s_center]), np.array([o.d_center]))
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'quali_static'
            m.id = o.id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(xy[0][0])
            m.pose.position.y = float(xy[1][0])
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = max(o.size, 0.1)
            m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.8
            marks.markers.append(m)
        self.marker_pub.publish(marks)


def main():
    rclpy.init()
    node = TrackingQuali()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
