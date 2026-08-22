#!/usr/bin/env python3
"""bag2mpccsv.py — rosbag(/mpc_debug) → mpc_logs 형식 CSV (GP 학습 전처리).

0810 GP 파이프라인 1-2단계의 스크립트가 세션 스크래치패드에 있다 소실됐다
(메모리 nuc14-gp-0810). 이번엔 repo 에 둔다.

/mpc_debug (Float32MultiArray, DBG_FIELDS 32필드)를 mpc_debug_logger.csv_header()
형식 CSV 로 변환한다. extract_residuals.py --source ekf 가 요구하는 컬럼
(t, v_actual, steer_cmd, car_x/y/yaw, current_s, feasible, vy_ekf, r_ekf, a_x_cmd)
은 전부 /mpc_debug 안에 있으므로 다른 토픽은 불필요. 나머지 컬럼은 NaN.

ROS 환경에서 실행 (rosbag2_py) — 차 위에서 돌려서 CSV 만 노트북으로 가져오면
bag 원본(수백 MB~GB)을 옮길 필요가 없다.

사용:  python3 bag2mpccsv.py <bag_dir> [out.csv]
"""
import csv
import math
import os
import sys

DBG_FIELDS = [
    "a_x_cmd", "steer_cmd", "v_actual", "car_x", "car_y", "car_yaw",
    "current_s", "near_idx", "ref_v", "n_obs_in", "sel_dmin", "sel_x", "sel_y",
    "side_pref", "opti_value", "solve_ms", "kappa_abs", "kappa_signed",
    "q_cte_scale", "q_lag_scale", "q_v_scale", "q_drate_scale", "v_max_cost",
    "t_ctrl", "feasible_msg", "vy_ekf", "r_ekf",
    "brake_anticip_a", "slew_down_rate", "accel_preview_s",
    "obs_blocked_d", "speed_cmd",
]
EXTRA = ["feasible", "min_lateral_margin", "pred_dx_n0", "pred_dy_n0",
         "pred_x_end", "pred_y_end", "mpcc_active", "switch_count",
         "vx_odom", "vy_odom", "r_odom"]
LAP_JUMP = 5.0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    bag = sys.argv[1].rstrip('/')
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
        f'~/mpc_logs/mpc_frombag_{os.path.basename(bag)}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import Float32MultiArray

    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions('', ''))

    n = 0
    lap = 0
    last_s = None
    warned = False
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'lap'] + DBG_FIELDS + EXTRA)
        while r.has_next():
            topic, data, t_bag = r.read_next()
            if topic != '/mpc_debug':
                continue
            m = deserialize_message(data, Float32MultiArray)
            d = list(m.data)
            if len(d) < len(DBG_FIELDS):
                if not warned:
                    print(f'[warn] /mpc_debug {len(d)}필드 < {len(DBG_FIELDS)} — '
                          f'모자란 필드는 NaN (mpc_node 버전 차이)')
                    warned = True
                d += [math.nan] * (len(DBG_FIELDS) - len(d))
            row = dict(zip(DBG_FIELDS, d))
            # t: t_ctrl(제어루프 시각)이 정석 — 0/NaN 이면 bag 시각 폴백
            t = row['t_ctrl']
            if not (t and math.isfinite(t) and t > 1.0):
                t = t_bag / 1e9
            s = row['current_s']
            if last_s is not None and (last_s - s) > LAP_JUMP:
                lap += 1
            last_s = s
            w.writerow([t, lap] + d
                       + [row['feasible_msg'], math.nan, math.nan, math.nan,
                          math.nan, math.nan, 1, 0,
                          row['v_actual'], row['vy_ekf'], row['r_ekf']])
            n += 1
    print(f'{n} rows, {lap + 1} laps → {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
