#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELRS Joy Node (ROS 2 Jazzy)
- Reads CRSF packets from ELRS receiver via USB-TTL serial
- Publishes sensor_msgs/Joy topic (Xbox-compatible layout)
- No CRC check - uses channel value range validation instead
  (for USB-TTL chips that can't hit exact 420000 baud)

Ported from ROS1 rospy keeping the full safety feature set:
- per-axis asymmetric calibration (cal_min/mid/max) and deadzone
- per-button polarity invert, configurable threshold
- post-reconnect settling window (publish suppressed for settling_sec)
- LB (joy_idx==4) 3-position band model with N-frame debounce
- A  (joy_idx==0) asymmetric N-frame debounce
- failsafe-on-signal-loss neutral Joy publish + filter state reset
- USB EIO reconnect loop (OSError treated like SerialException)
- raw CRSF channel snapshot on ~debug_channels (Int32MultiArray)
"""

import time

import rclpy
import serial
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32MultiArray


class ELRSJoyNode(Node):
    CRSF_SYNC = 0xC8
    CRSF_FRAMETYPE_RC_CHANNELS = 0x16
    CRSF_NUM_CHANNELS = 16

    # Valid channel range (with margin)
    CH_MIN = 100
    CH_MAX = 1900
    CH_MID = 992

    # For normalization
    NORM_MIN = 172
    NORM_MAX = 1811

    def __init__(self):
        super().__init__(
            'elrs_joy_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.port = self._p('port', '/dev/ttyUSB0')
        self.baud_rate = int(self._p('baud_rate', 416666))
        self.frame_id = self._p('frame_id', 'elrs_joy')
        self.publish_rate = float(self._p('publish_rate', 100))

        # Xbox-compatible Joy message layout
        self.num_axes = int(self._p('num_axes', 8))
        self.num_buttons = int(self._p('num_buttons', 11))
        self.axes_joy_indices = list(self._p('axes_joy_indices', [1, 3]))
        self.axes_crsf_channels = list(self._p('axes_crsf_channels', [0, 2]))
        self.button_joy_indices = list(self._p('button_joy_indices', [4, 5]))
        self.button_crsf_channels = list(self._p('button_crsf_channels', [5, 6]))
        self.button_invert = list(self._p('button_invert', [0, 0]))
        self.button_threshold = int(self._p('button_threshold', 992))
        self.axes_invert = list(self._p('axes_invert', [1.0, 1.0]))
        self.axes_cal_min = list(self._p('axes_cal_min', [172, 172]))
        self.axes_cal_mid = list(self._p('axes_cal_mid', [992, 992]))
        self.axes_cal_max = list(self._p('axes_cal_max', [1811, 1811]))
        self.deadzone = float(self._p('deadzone', 0.05))
        self.failsafe_timeout = float(self._p('failsafe_timeout', 2.0))

        self.joy_pub = self.create_publisher(Joy, 'joy', 10)
        self.debug_ch_pub = self.create_publisher(Int32MultiArray, '~/debug_channels', 10)

        self.channels = [self.CH_MID] * self.CRSF_NUM_CHANNELS
        self.last_valid_time = time.time()
        self.connected = False
        self.serial_port = None
        self.buffer = bytearray()

        self.accept_count = 0
        self.reject_count = 0
        self.last_stats_time = time.time()
        self.stats_printed = False

        self.settling_sec = float(self._p('settling_sec', 0.2))
        self.settling_until = 0.0

        self.lb_pressed_max = int(self._p('lb_pressed_max', 350))
        self.lb_idle_min = int(self._p('lb_idle_min', 700))
        self.lb_idle_max = int(self._p('lb_idle_max', 1300))
        self.lb_released_min = int(self._p('lb_released_min', 1600))
        self.lb_debounce_frames = int(self._p('lb_debounce_frames', 5))
        self.lb_state = 0
        self.lb_pending = 0
        self.lb_pending_count = 0

        self.a_debounce_frames = int(self._p('a_debounce_frames', 5))
        self.a_state = 0
        self.a_pending_count = 0

    def _p(self, name, default):
        try:
            v = self.get_parameter(name).value
            if v is not None:
                return v
        except Exception:
            pass
        try:
            self.declare_parameter(name, default)
            v = self.get_parameter(name).value
            if v is not None:
                return v
        except Exception:
            pass
        return default

    def normalize_axis(self, value, cal_min=None, cal_mid=None, cal_max=None):
        if cal_min is None:
            cal_min = self.NORM_MIN
        if cal_mid is None:
            cal_mid = self.CH_MID
        if cal_max is None:
            cal_max = self.NORM_MAX
        if value >= cal_mid:
            span = cal_max - cal_mid
            normalized = (value - cal_mid) / span if span > 0 else 0.0
        else:
            span = cal_mid - cal_min
            normalized = (value - cal_mid) / span if span > 0 else 0.0
        normalized = max(-1.0, min(1.0, normalized))
        if abs(normalized) < self.deadzone:
            normalized = 0.0
        return normalized

    def channel_to_button(self, value, invert=0):
        if invert:
            return 1 if value > self.button_threshold else 0
        return 1 if value < self.button_threshold else 0

    def validate_channels(self, channels):
        for i in range(min(4, len(channels))):
            if channels[i] < self.CH_MIN or channels[i] > self.CH_MAX:
                return False
        if len(set(channels[:4])) < 2:
            if not all(abs(ch - self.CH_MID) < 50 for ch in channels[:4]):
                return False
        return True

    def parse_rc_channels(self, payload):
        if len(payload) < 22:
            return False

        channels = []
        bit_offset = 0
        for _ in range(self.CRSF_NUM_CHANNELS):
            byte_offset = bit_offset // 8
            bit_shift = bit_offset % 8

            if byte_offset + 1 < len(payload):
                value = payload[byte_offset] >> bit_shift
                bits_from_first = 8 - bit_shift
                if bits_from_first < 11 and byte_offset + 1 < len(payload):
                    value |= payload[byte_offset + 1] << bits_from_first
                    bits_from_second = 11 - bits_from_first
                    if bits_from_second > 8 and byte_offset + 2 < len(payload):
                        value |= payload[byte_offset + 2] << (bits_from_first + 8)
                value &= 0x7FF
                channels.append(value)
            else:
                channels.append(self.CH_MID)

            bit_offset += 11

        if self.validate_channels(channels):
            self.channels = channels
            self.last_valid_time = time.time()
            self.accept_count += 1
            return True
        self.reject_count += 1
        return False

    def parse_crsf_frame(self):
        while len(self.buffer) > 2:
            sync_idx = -1
            for i in range(len(self.buffer)):
                if self.buffer[i] == self.CRSF_SYNC:
                    sync_idx = i
                    break

            if sync_idx == -1:
                self.buffer.clear()
                return

            if sync_idx > 0:
                self.buffer = self.buffer[sync_idx:]

            if len(self.buffer) < 3:
                return

            frame_length = self.buffer[1]

            if frame_length < 2 or frame_length > 64:
                self.buffer = self.buffer[1:]
                continue

            total_size = 2 + frame_length
            if len(self.buffer) < total_size:
                return

            frame_type = self.buffer[2]

            if frame_type == self.CRSF_FRAMETYPE_RC_CHANNELS:
                payload = self.buffer[3:total_size - 1]
                if self.parse_rc_channels(payload):
                    if not self.connected:
                        self.connected = True
                        self.settling_until = time.time() + self.settling_sec
                        self.get_logger().info(
                            "CRSF receiver connected! (settling %.0fms)" %
                            (self.settling_sec * 1000.0))

            self.buffer = self.buffer[total_size:]

        if len(self.buffer) > 512:
            self.buffer = self.buffer[-256:]

    def publish_joy(self):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.axes = [0.0] * self.num_axes
        msg.buttons = [0] * self.num_buttons
        for i, (joy_idx, crsf_ch) in enumerate(zip(self.axes_joy_indices, self.axes_crsf_channels)):
            sign = self.axes_invert[i] if i < len(self.axes_invert) else 1.0
            cal_min = self.axes_cal_min[i] if i < len(self.axes_cal_min) else self.NORM_MIN
            cal_mid = self.axes_cal_mid[i] if i < len(self.axes_cal_mid) else self.CH_MID
            cal_max = self.axes_cal_max[i] if i < len(self.axes_cal_max) else self.NORM_MAX
            msg.axes[joy_idx] = sign * self.normalize_axis(
                self.channels[crsf_ch], cal_min, cal_mid, cal_max)
        for i, (joy_idx, crsf_ch) in enumerate(zip(self.button_joy_indices, self.button_crsf_channels)):
            inv = self.button_invert[i] if i < len(self.button_invert) else 0
            if joy_idx == 4:
                msg.buttons[joy_idx] = self._lb_filtered_button(self.channels[crsf_ch], inv)
            elif joy_idx == 0:
                msg.buttons[joy_idx] = self._a_filtered_button(self.channels[crsf_ch], inv)
            else:
                msg.buttons[joy_idx] = self.channel_to_button(self.channels[crsf_ch], invert=inv)
        self.joy_pub.publish(msg)
        dbg = Int32MultiArray()
        dbg.data = [int(v) for v in self.channels]
        self.debug_ch_pub.publish(dbg)

    def _lb_filtered_button(self, raw_value, invert):
        if invert:
            if raw_value >= self.lb_released_min:
                candidate = 1
            elif raw_value <= self.lb_pressed_max:
                candidate = 0
            else:
                candidate = None
        else:
            if raw_value <= self.lb_pressed_max:
                candidate = 1
            elif self.lb_idle_min <= raw_value <= self.lb_idle_max:
                candidate = 0
            elif raw_value >= self.lb_released_min:
                candidate = 0
            else:
                candidate = None

        if candidate is None:
            self.lb_pending_count = 0
            return self.lb_state

        if candidate == self.lb_state:
            self.lb_pending_count = 0
            return self.lb_state

        if self.lb_state == 1 and candidate == 0:
            self.lb_state = 0
            self.lb_pending_count = 0
            self.get_logger().info("[elrs_joy] LB 1 -> 0 (released, immediate)")
            return self.lb_state

        if candidate == self.lb_pending:
            self.lb_pending_count += 1
        else:
            self.lb_pending = candidate
            self.lb_pending_count = 1

        if self.lb_pending_count >= self.lb_debounce_frames:
            self.lb_state = 1
            self.lb_pending_count = 0
            self.get_logger().info(
                "[elrs_joy] LB 0 -> 1 (debounced over %d frames)" % self.lb_debounce_frames)

        return self.lb_state

    def _a_filtered_button(self, raw_value, invert):
        candidate = self.channel_to_button(raw_value, invert=invert)

        if candidate == self.a_state:
            self.a_pending_count = 0
            return self.a_state

        if self.a_state == 1 and candidate == 0:
            self.a_state = 0
            self.a_pending_count = 0
            self.get_logger().info("[elrs_joy] A 1 -> 0 (released, immediate)")
            return self.a_state

        self.a_pending_count += 1
        if self.a_pending_count >= self.a_debounce_frames:
            self.a_state = 1
            self.a_pending_count = 0
            self.get_logger().info(
                "[elrs_joy] A 0 -> 1 (debounced over %d frames)" % self.a_debounce_frames)

        return self.a_state

    def check_failsafe(self):
        elapsed = time.time() - self.last_valid_time
        if elapsed > self.failsafe_timeout:
            if self.connected:
                self.get_logger().warning(
                    "CRSF signal lost! (no valid packet for %.1fs)" % elapsed)
                self.connected = False
                self.lb_state = 0
                self.lb_pending = 0
                self.lb_pending_count = 0
                self.a_state = 0
                self.a_pending_count = 0
                msg = Joy()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.frame_id
                msg.axes = [0.0] * self.num_axes
                msg.buttons = [0] * self.num_buttons
                self.joy_pub.publish(msg)

    def print_stats(self):
        now = time.time()
        if now - self.last_stats_time > 10.0:
            total = self.accept_count + self.reject_count
            if total > 0:
                rate = 100.0 * self.accept_count / total
                self.get_logger().info(
                    "CRSF frames: %d accepted, %d rejected (%.1f%% accept rate)" %
                    (self.accept_count, self.reject_count, rate))
            self.accept_count = 0
            self.reject_count = 0
            self.last_stats_time = now
            self.stats_printed = True

    def connect_serial(self):
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.01,
            )
            self.get_logger().info(
                "Serial port %s opened at %d baud" % (self.port, self.baud_rate))
            return True
        except serial.SerialException as e:
            self.get_logger().error("Failed to open serial port: %s" % str(e))
            return False

    def run(self):
        if not self.connect_serial():
            return

        period = 1.0 / max(self.publish_rate, 1.0)
        self.get_logger().info("ELRS Joy Node started (no-CRC mode)")
        self.get_logger().info("  Port: %s @ %d baud" % (self.port, self.baud_rate))
        self.get_logger().info("  Failsafe timeout: %.1fs" % self.failsafe_timeout)

        while rclpy.ok():
            try:
                rclpy.spin_once(self, timeout_sec=0.0)

                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    self.buffer.extend(data)
                    self.parse_crsf_frame()

                if self.connected and time.time() >= self.settling_until:
                    self.publish_joy()

                self.check_failsafe()
                if not self.stats_printed:
                    self.print_stats()
                time.sleep(period)

            except (serial.SerialException, OSError) as e:
                self.get_logger().error("Serial error: %s. Reconnecting..." % str(e))
                self.connected = False
                try:
                    if self.serial_port and self.serial_port.is_open:
                        self.serial_port.close()
                except Exception:
                    pass
                self.serial_port = None
                self.buffer.clear()
                while rclpy.ok():
                    time.sleep(1.0)
                    if self.connect_serial():
                        break
            except KeyboardInterrupt:
                break

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()


def main(args=None):
    rclpy.init(args=args)
    node = ELRSJoyNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
