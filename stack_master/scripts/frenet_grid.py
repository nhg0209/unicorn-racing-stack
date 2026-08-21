#!/usr/bin/env python3
"""frenet_grid — RViz 에 Frenet s-그리드 시각화 (튜닝 보조, 0819 신설)

/global_waypoints 를 받아서:
  /frenet_grid_markers (MarkerArray, latched+2s 재발행)
    - 1 m 마다 트랙 가로 틱(좌우 경계까지) + s 정수 라벨
    - 5 m 마다 굵은 틱 + 큰 라벨 (노랑)
  /frenet_car_marker (Marker, 10 Hz)
    - 차 위에 "s=23.4 d=-0.12" 라이브 텍스트 (존/섹터 s 값 바로 읽기용)

실행 (빌드 불필요):
  python3 stack_master/scripts/frenet_grid.py [--ros-args -p s_step:=1.0 -p major_every:=5]
RViz: MarkerArray 디스플레이 2개 추가 (/frenet_grid_markers, /frenet_car_marker)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
from f110_msgs.msg import WpntArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class FrenetGrid(Node):
    def __init__(self):
        super().__init__('frenet_grid')
        self.declare_parameter('s_step', 1.0)        # 틱 간격 [m]
        self.declare_parameter('major_every', 5)     # N틱마다 메이저
        self.declare_parameter('text_z', 0.05)
        self.grid = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid_pub = self.create_publisher(MarkerArray, '/frenet_grid_markers', latched)
        self.car_pub = self.create_publisher(Marker, '/frenet_car_marker', 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.on_wpnts, latched)
        self.create_subscription(WpntArray, '/global_waypoints', self.on_wpnts, 10)
        self.frenet_odom = None
        self.map_odom = None
        self.create_subscription(Odometry, '/car_state/frenet/odom', self.on_frenet, 10)
        self.create_subscription(Odometry, '/car_state/odom', self.on_odom, 10)
        self.create_timer(2.0, self.republish)   # RViz 늦접속 대비
        self.create_timer(0.1, self.pub_car)
        self.get_logger().info('frenet_grid up — waiting for /global_waypoints')

    def on_wpnts(self, msg):
        if self.grid is not None or len(msg.wpnts) < 10:
            return
        w = msg.wpnts
        s = np.array([p.s_m for p in w]); x = np.array([p.x_m for p in w])
        y = np.array([p.y_m for p in w])
        dl = np.array([p.d_left for p in w]); dr = np.array([p.d_right for p in w])
        L = s[-1]
        step = float(self.get_parameter('s_step').value)
        major = int(self.get_parameter('major_every').value)
        tz = float(self.get_parameter('text_z').value)
        ma = MarkerArray()

        def mk(mid, mtype, ns, scale, rgba):
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.pose.orientation.w = 1.0
            return m

        minor = mk(0, Marker.LINE_LIST, 'ticks_minor', (0.02, 0., 0.), (0.6, 0.6, 0.6, 0.5))
        majr = mk(1, Marker.LINE_LIST, 'ticks_major', (0.05, 0., 0.), (1.0, 0.9, 0.2, 0.8))
        tid = 10
        sq = np.arange(0.0, L, step)
        for i, sv in enumerate(sq):
            xi = np.interp(sv, s, x); yi = np.interp(sv, s, y)
            # 접선은 psi 컨벤션 안 믿고 수치미분 (랩어라운드 포함)
            s2 = (sv + 0.15) % L
            tx = np.interp(s2, s, x) - xi; ty = np.interp(s2, s, y) - yi
            n = np.hypot(tx, ty)
            if n < 1e-6:
                continue
            nx, ny = -ty / n, tx / n   # 좌측 normal
            dli = np.interp(sv, s, dl); dri = np.interp(sv, s, dr)
            p1 = Point(x=xi + nx * dli, y=yi + ny * dli, z=0.0)
            p2 = Point(x=xi - nx * dri, y=yi - ny * dri, z=0.0)
            is_major = (i % major == 0)
            (majr if is_major else minor).points += [p1, p2]
            t = mk(tid, Marker.TEXT_VIEW_FACING, 'labels',
                   (0., 0., 0.35 if is_major else 0.18),
                   (1., 0.9, 0.2, 1.) if is_major else (0.9, 0.9, 0.9, 0.9))
            t.pose.position.x = xi + nx * (dli + 0.25)   # 좌측 경계 밖에 라벨
            t.pose.position.y = yi + ny * (dli + 0.25)
            t.pose.position.z = tz
            t.text = f'{sv:g}'
            ma.markers.append(t)
            tid += 1
        ma.markers = [minor, majr] + ma.markers
        self.grid = ma
        self.grid_pub.publish(ma)
        self.get_logger().info(f'grid published: L={L:.1f} m, {len(sq)} ticks (step {step})')

    def republish(self):
        if self.grid is not None:
            self.grid_pub.publish(self.grid)

    def on_frenet(self, msg):
        self.frenet_odom = msg

    def on_odom(self, msg):
        self.map_odom = msg

    def pub_car(self):
        if self.frenet_odom is None or self.map_odom is None:
            return
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id, m.type, m.action = 'car_frenet', 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        p = self.map_odom.pose.pose.position
        m.pose.position.x, m.pose.position.y, m.pose.position.z = p.x, p.y, 0.6
        m.pose.orientation.w = 1.0
        m.scale.z = 0.3
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 1.0, 1.0, 1.0
        f = self.frenet_odom.pose.pose.position
        m.text = f's={f.x:.1f} d={f.y:+.2f}'
        self.car_pub.publish(m)


def main():
    rclpy.init()
    n = FrenetGrid()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
