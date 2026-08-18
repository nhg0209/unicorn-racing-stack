"""wall_alarm_node.py — 가상 벽 침범 경보 (관측 전용, 제어 0줄).

벽 없는 실험장(3D 라이다 가상 트랙)에서 rviz상의 벽을 넘으면 알려준다.
노트북 observer 런치에 포함시켜 사용:
    ros2 run nonlinear_mpc_acados wall_alarm

- /global_waypoints (f110_msgs/WpntArray) 에서 트랙 경계 로드 (1회)
- /car_state/odom 위치로 wall_margin 계산
- margin < warn_margin  → WARN 로그 (스로틀)
- margin < 0            → ERROR 로그 + 터미널 벨(\\a) + /wall_alarm/hit(Bool) True
- 상시: /wall_alarm/margin (Float32), rviz 마커(/wall_alarm/marker —
  여유에 따라 초록→노랑→빨강 구), 침범 순간 위치에 빨간 큐브 잔상(latch)
"""
from __future__ import annotations

import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker
from f110_msgs.msg import WpntArray

from nonlinear_mpc_acados.wall_alarm_core import build_track, wall_margin


class WallAlarm(Node):
    def __init__(self):
        super().__init__('wall_alarm')
        self.declare_parameter('odom_topic', '/car_state/odom')
        self.declare_parameter('wpnt_topic', '/global_waypoints')
        self.declare_parameter('r_car', 0.15)        # 차 반폭 [m]
        self.declare_parameter('warn_margin', 0.15)  # 이하면 WARN [m]
        self.declare_parameter('min_speed', 0.3)     # 정지 중 잡음 억제 [m/s]
        gp = self.get_parameter
        self._r_car = float(gp('r_car').value)
        self._warn = float(gp('warn_margin').value)
        self._min_v = float(gp('min_speed').value)
        self._track = None
        self._hits = 0
        self.create_subscription(WpntArray, str(gp('wpnt_topic').value), self._wpnt_cb, 10)
        self.create_subscription(Odometry, str(gp('odom_topic').value), self._odom_cb, 10)
        self._pub_margin = self.create_publisher(Float32, '~/margin', 10)
        self._pub_hit = self.create_publisher(Bool, '~/hit', 10)
        self._pub_marker = self.create_publisher(Marker, '~/marker', 10)
        self.get_logger().info('[wall_alarm] 대기 — /global_waypoints 수신 후 활성화')

    def _wpnt_cb(self, msg: WpntArray):
        if self._track is not None or not msg.wpnts:
            return
        self._track = build_track(
            [(w.x_m, w.y_m, w.d_left, w.d_right) for w in msg.wpnts])
        self.get_logger().info(
            f'[wall_alarm] 트랙 로드 ({len(msg.wpnts)} wpnts) — 감시 시작 '
            f'(r_car={self._r_car}, warn<{self._warn}m)')

    def _odom_cb(self, msg: Odometry):
        if self._track is None:
            return
        p = msg.pose.pose.position
        v = abs(msg.twist.twist.linear.x)
        margin, _, _ = wall_margin(self._track, p.x, p.y, self._r_car)
        self._pub_margin.publish(Float32(data=float(margin)))
        hit = margin < 0.0 and v >= self._min_v
        self._pub_hit.publish(Bool(data=bool(hit)))
        self._publish_marker(p, margin, hit)
        if hit:
            self._hits += 1
            # (2026-07-03 사용자 요청: 소리 제거 — 로그로만 경보)
            self.get_logger().error(
                f'💥 WALL HIT #{self._hits}! margin={margin:.2f}m @ ({p.x:.2f},{p.y:.2f})')
        elif margin < self._warn and v >= self._min_v:
            self.get_logger().warn(
                f'벽 근접: margin={margin:.2f}m', throttle_duration_sec=1.0)

    def _publish_marker(self, p, margin, hit):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'wall_alarm'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(p.x)
        m.pose.position.y = float(p.y)
        m.pose.position.z = 0.3
        m.scale.x = m.scale.y = m.scale.z = 0.25
        if hit:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0
        elif margin < self._warn:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.8, 0.0, 0.9
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 0.2, 0.6
        self._pub_marker.publish(m)
        if hit:
            hm = Marker()
            hm.header = m.header
            hm.ns = 'wall_alarm_hits'
            hm.id = self._hits + 1
            hm.type = Marker.CUBE
            hm.action = Marker.ADD
            hm.pose.position.x = float(p.x)
            hm.pose.position.y = float(p.y)
            hm.pose.position.z = 0.05
            hm.scale.x = hm.scale.y = 0.3
            hm.scale.z = 0.1
            hm.color.r, hm.color.a = 1.0, 1.0
            self._pub_marker.publish(hm)


def main(args=None):
    rclpy.init(args=args)
    node = WallAlarm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
