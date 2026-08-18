"""wall_alarm_core.py — 가상 벽 마진 계산 (순수 로직, ROS 무관).

3D 라이다 기반 가상 트랙(벽 없는 실험장)에서 차가 rviz상의 벽을 넘었는지
관측하기 위한 코어. /global_waypoints 의 (x, y, d_left, d_right) 와 차 위치로
"벽까지 남은 여유 [m]" 를 계산한다. 음수 = 침범.

margin = min(d_left - lat, d_right + lat) - r_car
  lat   : 웨이포인트 접선의 좌법선 방향 부호거리 (+ = 라인 왼쪽)
  r_car : 차 반폭 (기본 0.15) — 차 중심이 아니라 차체 모서리 기준 여유
"""
from __future__ import annotations

import math


def build_track(wpnts):
    """[(x, y, d_left, d_right), ...] -> 내부 표현 (좌표/법선/폭).

    법선은 이웃 점 차분 접선의 좌회전 90°. 폐루프 가정(끝점은 랩어라운드).
    """
    n = len(wpnts)
    if n < 3:
        raise ValueError("need >= 3 waypoints")
    xs = [w[0] for w in wpnts]
    ys = [w[1] for w in wpnts]
    track = []
    for i in range(n):
        tx = xs[(i + 1) % n] - xs[i - 1]
        ty = ys[(i + 1) % n] - ys[i - 1]
        norm = math.hypot(tx, ty) or 1e-9
        # 좌법선 = (-ty, tx)/|t|
        track.append((xs[i], ys[i], -ty / norm, tx / norm,
                      float(wpnts[i][2]), float(wpnts[i][3])))
    return track


def wall_margin(track, x, y, r_car=0.15):
    """차 위치 (x, y) 의 벽 여유 [m]. 음수 = 차체가 벽을 넘음.

    반환: (margin, lat, idx) — lat 는 라인 기준 부호 횡오프셋(+좌).
    """
    best = None
    bi = 0
    for i, (wx, wy, nx, ny, dl, dr) in enumerate(track):
        d2 = (wx - x) ** 2 + (wy - y) ** 2
        if best is None or d2 < best:
            best = d2
            bi = i
    wx, wy, nx, ny, dl, dr = track[bi]
    lat = (x - wx) * nx + (y - wy) * ny
    margin = min(dl - lat, lat + dr) - r_car
    return margin, lat, bi
