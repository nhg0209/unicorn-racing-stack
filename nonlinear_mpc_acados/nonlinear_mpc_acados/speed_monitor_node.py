"""실시간 속도 비교 플롯 노드 — /mpc_debug 의 v_cmd/v_actual/ref_v 를 롤링 시간창으로.

관측 전용(제어 무관). MPCC 코어 미import. 노트북에서 차 토픽 구독해 실행:
    ros2 run nonlinear_mpc_acados speed_monitor
"""
from __future__ import annotations
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped

from nonlinear_mpc_acados.speed_monitor_core import extract_speeds, trim_window


class SpeedMonitor(Node):
    def __init__(self):
        super().__init__('speed_monitor')
        self.declare_parameter('topic', '/mpc_debug')
        self.declare_parameter('window_sec', 15.0)
        self.declare_parameter('refresh_hz', 20.0)
        self.declare_parameter('show_ref', True)
        self.declare_parameter('y_max', 0.0)
        self.declare_parameter('i_cmd', 0)
        self.declare_parameter('i_actual', 2)
        self.declare_parameter('i_ref', 8)
        # 2026-07-02: mpc_debug[0] 은 unified layout 이후 a_x(가속도)라 속도가
        # 아님. 기본은 실제 발행 속도명령(/vesc/ackermann_cmd drive.speed)을
        # v_cmd 로 그린다. 빈 문자열이면 예전처럼 i_cmd 인덱스 사용.
        self.declare_parameter('cmd_topic', '/vesc/ackermann_cmd')
        gp = self.get_parameter
        self._window = float(gp('window_sec').value)
        self._show_ref = bool(gp('show_ref').value)
        self._y_max = float(gp('y_max').value)
        self._i_cmd = int(gp('i_cmd').value)
        self._i_actual = int(gp('i_actual').value)
        self._i_ref = int(gp('i_ref').value)
        self._t0 = self.get_clock().now().nanoseconds * 1e-9
        self._lock = threading.Lock()
        self._t, self._cmd, self._act, self._ref = [], [], [], []
        self._last_ack_speed = None
        self._cmd_topic = str(gp('cmd_topic').value).strip()
        if self._cmd_topic:
            self.create_subscription(AckermannDriveStamped, self._cmd_topic,
                                     self._ack_cb, 10)
        self.create_subscription(Float32MultiArray, str(gp('topic').value), self._cb, 10)
        self.get_logger().info('[speed_monitor] /mpc_debug 구독 시작 — 창 닫으면 종료')

    def _ack_cb(self, msg: AckermannDriveStamped):
        self._last_ack_speed = float(msg.drive.speed)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9 - self._t0

    def _cb(self, msg: Float32MultiArray):
        s = extract_speeds(msg.data, self._i_cmd, self._i_actual, self._i_ref)
        if s is None:
            return
        now = self._now()
        # cmd_topic 사용 시: 최신 ackermann drive.speed 를 v_cmd 로 (없으면 NaN 대신 0 유지 회피 위해 skip)
        cmd_val = s[0]
        if self._cmd_topic:
            if self._last_ack_speed is None:
                return
            cmd_val = self._last_ack_speed
        with self._lock:
            self._t.append(now)
            self._cmd.append(cmd_val)
            self._act.append(s[1])
            self._ref.append(s[2])
            k = trim_window(self._t, now, self._window)
            if k:
                del self._t[:k]; del self._cmd[:k]; del self._act[:k]; del self._ref[:k]

    def snapshot(self):
        with self._lock:
            return list(self._t), list(self._cmd), list(self._act), list(self._ref)


def main(args=None):
    rclpy.init(args=args)
    node = SpeedMonitor()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as e:  # 디스플레이/백엔드 없음
        node.get_logger().error(
            f"matplotlib 로드 실패({e}) — 디스플레이 필요. 노트북에서 실행하거나 X-forward.")
        rclpy.shutdown()
        return

    fig, ax = plt.subplots()
    ln_ref, = ax.plot([], [], 'g--', label='ref_v (target)')
    ln_cmd, = ax.plot([], [], 'b-',  label='v_cmd (command)')
    ln_act, = ax.plot([], [], 'r-',  label='v_actual (actual)')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('speed [m/s]')
    ax.legend(loc='upper left')
    ax.grid(True)

    def update(_frame):
        t, cmd, act, ref = node.snapshot()
        ln_cmd.set_data(t, cmd)
        ln_act.set_data(t, act)
        ln_ref.set_data(t, ref) if node._show_ref else ln_ref.set_data([], [])
        if t:
            ax.set_xlim(max(0.0, t[-1] - node._window), max(node._window, t[-1]))
        if node._y_max > 0.0:
            ax.set_ylim(0.0, node._y_max)
        else:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        return ln_ref, ln_cmd, ln_act

    interval_ms = max(10.0, 1000.0 / max(1.0, float(node.get_parameter('refresh_hz').value)))
    _anim = FuncAnimation(fig, update, interval=interval_ms, blit=False,
                          cache_frame_data=False)  # noqa: F841 (GC 방지 위해 참조 유지)
    try:
        plt.show()
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
