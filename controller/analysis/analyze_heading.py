#!/usr/bin/env python3
"""
pp_heading 헤딩 추종 진단 스크립트.

ros2 bag(/pp_debug + /car_state/odom)을 읽어 헤딩오차 e_h를 코너/직선/코너탈출
구간으로 나눠 통계를 내고, 원인(droop vs lag vs oscillation)을 판정한다.

녹화:
  ros2 bag record -o pp_run /pp_debug /car_state/odom /car_state/odom_frenet /local_waypoints
  # pp_heading으로 2~3바퀴 주행 후 Ctrl-C

분석:
  python3 analyze_heading.py /path/to/pp_run        # (디렉토리 또는 .db3/.mcap)

/pp_debug Float64MultiArray 인덱스 (pp_heading_controller.py와 일치):
  0 a_lat_pp  1 a_lat_geo  2 a_lat_meas  3 delta_geo[deg]  4 delta_pid[deg]
  5 delta_cmd[deg]  6 e_h[rad]  7 ld  8 v_ref  9 v_cmd
"""
import sys
import os
import math
import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except Exception as e:
    sys.exit(f"[err] ROS2 환경에서 실행하세요 (unicorn). import 실패: {e}")

# debug field indices (match pp_heading_controller.py)
A_LAT_PP, A_LAT_GEO, A_LAT_MEAS, D_GEO, D_PID, D_CMD, E_H, LD, V_REF, V_CMD = range(10)
CORNER_A_LAT = 1.0   # |a_lat| > 이 값이면 코너로 간주 [m/s²]


def _open(path):
    if os.path.isdir(path):
        # storage id 추론
        sid = 'mcap' if any(f.endswith('.mcap') for f in os.listdir(path)) else 'sqlite3'
    else:
        sid = 'mcap' if path.endswith('.mcap') else 'sqlite3'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=sid),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, types


def load(path):
    reader, types = _open(path)
    pp_t, pp_d, od_t, od_yaw, od_yr, od_v = [], [], [], [], [], []
    msgcls = {}
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in msgcls:
            if topic not in types:
                continue
            msgcls[topic] = get_message(types[topic])
        if topic == '/pp_debug':
            m = deserialize_message(data, msgcls[topic])
            pp_t.append(t * 1e-9); pp_d.append(list(m.data))
        elif topic == '/car_state/odom':
            m = deserialize_message(data, msgcls[topic])
            q = m.pose.pose.orientation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            od_t.append(t * 1e-9); od_yaw.append(yaw)
            od_yr.append(m.twist.twist.angular.z)
            od_v.append(abs(m.twist.twist.linear.x))
    return (np.array(pp_t), np.array(pp_d, dtype=float),
            np.array(od_t), np.array(od_yaw), np.array(od_yr), np.array(od_v))


def stats(name, e):
    if len(e) == 0:
        print(f"  {name:16s}: (no samples)"); return
    print(f"  {name:16s}: mean={np.mean(e):+.3f}  |mean|={np.mean(np.abs(e)):.3f}  "
          f"rms={np.sqrt(np.mean(e**2)):.3f}  max|e|={np.max(np.abs(e)):.3f}  [rad]  (n={len(e)})")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 analyze_heading.py <bag_dir_or_file>")
    path = sys.argv[1]
    pp_t, pp_d, od_t, od_yaw, od_yr, od_v = load(path)
    if len(pp_d) == 0:
        sys.exit("[err] /pp_debug 메시지가 없습니다. 녹화 토픽을 확인하세요.")

    e_h   = pp_d[:, E_H]
    a_lat = pp_d[:, A_LAT_GEO]
    d_geo = pp_d[:, D_GEO]
    d_pid = pp_d[:, D_PID]
    v_cmd = pp_d[:, V_CMD]
    ld    = pp_d[:, LD]
    a_meas = pp_d[:, A_LAT_MEAS] if pp_d.shape[1] > A_LAT_MEAS else np.zeros_like(e_h)

    corner = np.abs(a_lat) > CORNER_A_LAT
    straight = ~corner
    # 코너 탈출: 코너→직선 전환 후 0.5초 구간
    dt = np.median(np.diff(pp_t)) if len(pp_t) > 1 else 0.02
    exit_win = max(1, int(0.5 / dt))
    exit_mask = np.zeros_like(corner)
    edges = np.where((~corner[1:]) & (corner[:-1]))[0] + 1   # corner→straight 시작 인덱스
    for i in edges:
        exit_mask[i:i+exit_win] = True

    print("\n=== 헤딩오차 e_h = ψ_path − yaw 통계 (구간별) ===")
    stats("전체", e_h)
    stats("코너", e_h[corner])
    stats("직선", e_h[straight])
    stats("코너탈출0.5s", e_h[exit_mask])

    print("\n=== 조향 항 기여 (코너 평균, deg) ===")
    if np.any(corner):
        print(f"  delta_geo={np.mean(d_geo[corner]):+.2f}  delta_pid={np.mean(d_pid[corner]):+.2f}")
    print(f"  속도 평균 v_cmd={np.mean(v_cmd):.2f}  ld 평균={np.mean(ld):.2f}  "
          f"코너비율={100*np.mean(corner):.0f}%")
    if np.any(corner):
        print(f"  코너 a_lat: 명령={np.mean(np.abs(a_lat[corner])):.2f}  측정={np.mean(np.abs(a_meas[corner])):.2f} [m/s²]")

    # ── 판정 휴리스틱 ───────────────────────────────────────────────────────
    print("\n=== 진단 ===")
    cm = e_h[corner]
    if len(cm) > 10:
        bias = np.mean(cm)                 # 부호 있는 평균 → droop
        rms = np.sqrt(np.mean(cm**2))
        # 부호변화율(진동성)
        zc = np.mean(np.abs(np.diff(np.sign(cm)))) / 2.0
        if abs(bias) > 0.05 and abs(bias) > 0.6 * rms:
            print(f"  ▶ 코너 e_h 평균이 한쪽으로 치우침(bias={bias:+.3f}) → 정상상태 DROOP.")
            print(f"    원인: P-only(Ki=0) + 곡률 피드포워드 없음. 처방: heading_ki 소량(0.05~0.1)")
            print(f"    또는 곡률 FF 추가. (지금 delta_pid가 코너에서 큰지 위 값 확인)")
        elif zc > 0.25:
            print(f"  ▶ 코너 e_h 부호변화 잦음(zc={zc:.2f}) → 진동/under-damped.")
            print(f"    처방: heading_kd↑(0.08→0.12), ld_max/t_headway↑, future_constant↑")
        else:
            print(f"  ▶ 코너 e_h가 큰데 bias/진동 뚜렷치 않음(rms={rms:.3f}) → LAG.")
            print(f"    처방: lookahead↑(t_headway 0.3→0.5), future_constant↑, 곡률 FF")
    ex = e_h[exit_mask]
    if len(ex) > 5 and np.mean(np.abs(ex)) > 1.2 * np.mean(np.abs(e_h[straight]) + 1e-9):
        print(f"  ▶ 코너탈출 |e_h|={np.mean(np.abs(ex)):.3f} > 직선 → 탈출 전이 LAG.")
        print(f"    처방: future_constant↑, 곡률 FF(전이에서 즉시 펴짐), acc 기반 steer scaling")

    # ── 플롯 ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t0 = pp_t - pp_t[0]
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        ax[0].plot(t0, np.degrees(e_h), label='e_h [deg]')
        ax[0].fill_between(t0, -50, 50, where=corner, alpha=0.12, color='r', label='corner')
        ax[0].axhline(0, color='k', lw=0.5); ax[0].legend(); ax[0].set_ylabel('heading err')
        ax[1].plot(t0, d_geo, label='δ_geo'); ax[1].plot(t0, d_pid, label='δ_pid')
        ax[1].plot(t0, pp_d[:, D_CMD], 'k', lw=0.7, label='δ_cmd'); ax[1].legend(); ax[1].set_ylabel('steer [deg]')
        ax[2].plot(t0, v_cmd, label='v_cmd'); ax[2].plot(t0, np.abs(a_lat), label='|a_lat| cmd')
        ax[2].plot(t0, np.abs(a_meas), label='|a_lat| meas'); ax[2].legend(); ax[2].set_ylabel('v / a_lat'); ax[2].set_xlabel('t [s]')
        out = os.path.join(os.path.dirname(os.path.abspath(path.rstrip('/'))) or '.', 'pp_heading_analysis.png')
        fig.tight_layout(); fig.savefig(out, dpi=110)
        print(f"\n  플롯 저장: {out}")
    except Exception as e:
        print(f"\n  (플롯 생략: {e})")


if __name__ == '__main__':
    main()
