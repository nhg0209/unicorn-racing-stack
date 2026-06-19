#!/usr/bin/env python3
"""
opponent_predictor.py - constant-velocity opponent prediction.

A lightweight stand-in for the ROS1 GP trajectory predictor: it rolls each
dynamic obstacle forward along the raceline at its measured Frenet speed (vs) and
publishes the predicted obstacle + a short prediction horizon. This unblocks the
predictive planners (lane_change / sqp / dynamic_avoidance) and the state
machine, which subscribe these topics.

Subscribes:
  /tracking/obstacles  (f110_msgs/ObstacleArray) - obstacles with Frenet s/d/vs
  /global_waypoints    (f110_msgs/WpntArray)      - raceline for s<->xy

Publishes:
  /opponent_prediction/obstacles      (f110_msgs/ObstacleArray)  - current dynamic obstacles
  /opponent_prediction/obstacles_pred (f110_msgs/PredictionArray) - CV rollout
"""
import numpy as np
import rclpy
from rclpy.node import Node

from f110_msgs.msg import (
    ObstacleArray, Prediction, PredictionArray, WpntArray,
)

try:
    from frenet_conversion.frenet_converter import FrenetConverter
except Exception:
    FrenetConverter = None


class OpponentPredictor(Node):

    def __init__(self):
        super().__init__('opponent_predictor')
        self.declare_parameter('horizon_s', 2.0)   # how far ahead to predict [s]
        self.declare_parameter('dt', 0.2)          # prediction step [s]
        self.declare_parameter('rate_hz', 20.0)
        self.horizon = float(self.get_parameter('horizon_s').value)
        self.dt = float(self.get_parameter('dt').value)

        self.converter = None
        self.track_len = None
        self.obs = None

        self.create_subscription(WpntArray, '/global_waypoints', self._gb_cb, 10)
        self.create_subscription(ObstacleArray, '/tracking/obstacles', self._obs_cb, 10)
        self.obs_pub = self.create_publisher(
            ObstacleArray, '/opponent_prediction/obstacles', 10)
        self.pred_pub = self.create_publisher(
            PredictionArray, '/opponent_prediction/obstacles_pred', 10)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self._tick)
        self.get_logger().info('OpponentPredictor ready (constant-velocity)')

    def _gb_cb(self, msg):
        if FrenetConverter is None or len(msg.wpnts) < 3:
            return
        x = np.array([w.x_m for w in msg.wpnts])
        y = np.array([w.y_m for w in msg.wpnts])
        psi = np.array([w.psi_rad for w in msg.wpnts])
        try:
            self.converter = FrenetConverter(x, y, psi)
            self.track_len = float(self.converter.raceline_length)
        except Exception as e:
            self.get_logger().warn(f'FrenetConverter init failed: {e}')

    def _obs_cb(self, msg):
        self.obs = msg

    def _tick(self):
        if self.obs is None:
            return
        stamp = self.get_clock().now().to_msg()

        out = ObstacleArray()
        out.header.stamp = stamp
        out.header.frame_id = 'map'
        pred = PredictionArray()
        pred.header.stamp = stamp
        pred.header.frame_id = 'map'

        n_steps = max(1, int(self.horizon / self.dt))
        for o in self.obs.obstacles:
            out.obstacles.append(o)              # current estimate (pass-through)
            if o.is_static or self.converter is None or self.track_len is None:
                continue
            pred.id = o.id
            for k in range(1, n_steps + 1):
                t = k * self.dt
                s_pred = (o.s_center + o.vs * t) % self.track_len
                try:
                    xy = self.converter.get_cartesian(s_pred, o.d_center)
                    px, py = float(xy[0]), float(xy[1])
                except Exception:
                    continue
                pr = Prediction()
                pr.id = o.id
                pr.pred_x = px
                pr.pred_y = py
                pr.pred_vx = float(o.vs)
                pr.pred_s = float(s_pred)
                pr.pred_d = float(o.d_center)
                pr.pred_vs = float(o.vs)
                pred.predictions.append(pr)

        self.obs_pub.publish(out)
        self.pred_pub.publish(pred)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentPredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
