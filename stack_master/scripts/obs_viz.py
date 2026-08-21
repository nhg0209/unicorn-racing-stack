#!/usr/bin/env python3
"""obs_viz — RViz 장애물/회피 상태 시각화 (0821 신설, frenet_grid 패턴 — 빌드 불필요).

/external_obstacles : 수신 장애물 전부 회색 구
/mpc_debug          : 선택 장애물 = 빨강 구 + dmin 텍스트,
                      side_pref = 초록(좌)/파랑(우) 화살표,
                      obs_blocked_d>0 = 차 위 "BLOCKED" 빨강 배너
RViz: MarkerArray 디스플레이 /mpc_obs_viz 추가.
실행: python3 stack_master/scripts/obs_viz.py  (mpcc launch 에 포함됨)
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray

# mpc_debug Float32MultiArray 레이아웃 (mpc_debug_logger DBG_FIELDS 와 동일)
I_CAR_X, I_CAR_Y = 3, 4
I_N_OBS, I_SEL_DMIN, I_SEL_X, I_SEL_Y, I_SIDE = 9, 10, 11, 12, 13
I_BLOCKED = 30   # DBG_FIELDS 실측 순서: …27 brake_anticip_a, 28 slew_down, 29 accel_preview, 30 obs_blocked_d


class ObsViz(Node):
    def __init__(self):
        super().__init__('obs_viz')
        self.pub = self.create_publisher(MarkerArray, '/mpc_obs_viz', 1)
        self.create_subscription(PoseArray, '/external_obstacles',
                                 self._obs_cb, 1)
        self.create_subscription(Float32MultiArray, '/mpc_debug',
                                 self._dbg_cb, 1)
        self._raw = []
        self.get_logger().info('obs_viz up — RViz 에 /mpc_obs_viz 추가')

    def _obs_cb(self, msg):
        self._raw = [(p.position.x, p.position.y) for p in msg.poses]

    def _mk(self, mid, mtype, x, y, z, sx, sy, sz, r, g, b, a=0.9, text=''):
        m = Marker()
        m.header.frame_id = 'map'
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        m.lifetime.nanosec = 300_000_000   # 0.3s — 상태 사라지면 자동 소멸
        if text:
            m.text = text
        return m

    def _dbg_cb(self, msg):
        d = msg.data
        if len(d) <= I_BLOCKED:
            return
        arr = MarkerArray()
        mid = 0
        # 수신 장애물 전부 (회색)
        for x, y in self._raw[:30]:
            arr.markers.append(self._mk(mid, Marker.SPHERE, x, y, 0.15,
                                        0.25, 0.25, 0.25, 0.6, 0.6, 0.6, 0.7))
            mid += 1
        n_obs, dmin = d[I_N_OBS], d[I_SEL_DMIN]
        sx, sy, side = d[I_SEL_X], d[I_SEL_Y], int(d[I_SIDE])
        cx, cy = d[I_CAR_X], d[I_CAR_Y]
        if n_obs > 0 and (abs(sx) > 1e-3 or abs(sy) > 1e-3):
            # 선택 장애물 (빨강) + dmin 라벨
            arr.markers.append(self._mk(100, Marker.SPHERE, sx, sy, 0.2,
                                        0.35, 0.35, 0.35, 1.0, 0.1, 0.1))
            arr.markers.append(self._mk(101, Marker.TEXT_VIEW_FACING,
                                        sx, sy, 0.6, 0, 0, 0.3,
                                        1.0, 0.3, 0.3,
                                        text=f'd={dmin:.2f}'))
            # side 화살표: 좌(+1) 초록 / 우(-1) 파랑
            if side != 0:
                a = self._mk(102, Marker.ARROW, 0, 0, 0, 0.08, 0.15, 0.1,
                             0.1 if side > 0 else 0.2,
                             0.9 if side > 0 else 0.3,
                             0.2 if side > 0 else 0.9)
                p0 = Point(x=sx, y=sy, z=0.4)
                p1 = Point(x=sx, y=sy + (0.8 if side > 0 else -0.8), z=0.4)
                a.points = [p0, p1]
                arr.markers.append(a)
        if d[I_BLOCKED] > 0:
            arr.markers.append(self._mk(103, Marker.TEXT_VIEW_FACING,
                                        cx, cy, 1.0, 0, 0, 0.45,
                                        1.0, 0.1, 0.1,
                                        text='BLOCKED'))
        self.pub.publish(arr)


def main():
    rclpy.init()
    rclpy.spin(ObsViz())


if __name__ == '__main__':
    main()
