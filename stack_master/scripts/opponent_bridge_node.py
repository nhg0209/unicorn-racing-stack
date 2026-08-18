#!/usr/bin/env python3
"""Opponent-position bridge over plain UDP unicast (2-car).

Both cars localize on the SAME map and publish their pose on /car_state/odom,
each in its OWN ROS_DOMAIN_ID. Those domains are isolated, and on this network
DDS discovery does NOT cross between machines (mixed WiFi/Ethernet, multicast
not forwarded by the AP). So instead of bridging DDS across domains, this node
forwards the pose over a plain UDP unicast socket -- which works anywhere ping
works -- and keeps ROS/DDS strictly local to each car's own domain.

Each car runs one instance:
  - subscribes our own /car_state/odom (our domain) and, at `rate` Hz, sends the
    latest Odometry (ROS-serialized) as a UDP datagram to the peer car's IP:port;
  - listens on the same UDP port, deserializes datagrams from the peer, and
    publishes them on our domain as /opponent_state/odom.

Because both cars share the map (same origin), the peer's map-frame pose is
directly usable here -- no transform. The full Odometry is preserved (pose +
twist + covariance + frame ids), so downstream MPC/planners get velocity too.

Parameters:
  peer_ip   (str)   : the other car's IP on the shared LAN. REQUIRED.
  port      (int)   = 47600   UDP port (same on both cars; each binds + sends).
  in_topic  (str)   = /car_state/odom       our own pose (subscribed)
  out_topic (str)   = /opponent_state/odom  the peer's pose (published)
  rate      (double)= 50.0   send-rate cap [Hz] (throttle)
  timeout   (double)= 0.5    stop publishing peer pose if no datagram within [s]
"""
import socket
import struct
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import serialize_message, deserialize_message
from nav_msgs.msg import Odometry

_MAGIC = b"OPP1"  # 4-byte tag so stray datagrams are ignored


class OpponentBridge(Node):
    def __init__(self):
        super().__init__("opponent_bridge")
        self.peer_ip = str(self.declare_parameter("peer_ip", "").value)
        self.port = int(self.declare_parameter("port", 47600).value)
        self.in_topic = str(self.declare_parameter("in_topic", "/car_state/odom").value)
        self.out_topic = str(self.declare_parameter("out_topic", "/opponent_state/odom").value)
        self.rate = float(self.declare_parameter("rate", 50.0).value)
        self.timeout = float(self.declare_parameter("timeout", 0.5).value)

        if not self.peer_ip:
            self.get_logger().error("peer_ip is required; bridge disabled.")
            self._ok = False
            return
        self._ok = True

        # Publish RELIABLE so the default RViz Odometry display and typical odom
        # consumers (MPC/planner, default QoS = RELIABLE) receive it without any
        # per-display QoS override. Subscribe BEST_EFFORT so we accept /car_state/odom
        # regardless of how the localization stack advertises it (best-effort sub is
        # compatible with both reliable and best-effort publishers).
        pub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Odometry, self.out_topic, pub_qos)
        self.create_subscription(Odometry, self.in_topic, self._on_odom, sub_qos)

        # UDP: one socket bound to the shared port, used for both rx and tx.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(0.5)

        self._latest = None
        self._fresh = False
        self._lock = threading.Lock()

        # rx thread: peer datagrams -> /opponent_state/odom
        self._run = True
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx.start()
        # tx timer: throttle our pose out to the peer at `rate` Hz
        self.create_timer(1.0 / max(1.0, self.rate), self._on_tx)

        self.get_logger().info(
            f"opponent bridge (UDP): '{self.in_topic}' -> {self.peer_ip}:{self.port} "
            f"-> '{self.out_topic}' @ {self.rate:.0f} Hz")

    def _on_odom(self, msg):
        with self._lock:
            self._latest = msg
            self._fresh = True

    def _on_tx(self):
        with self._lock:
            msg = self._latest
            fresh = self._fresh
            self._fresh = False
        if msg is None or not fresh:
            return
        try:
            payload = _MAGIC + serialize_message(msg)
            self.sock.sendto(payload, (self.peer_ip, self.port))
        except OSError as e:
            self.get_logger().warn(f"udp send failed: {e}", throttle_duration_sec=2.0)

    def _rx_loop(self):
        while self._run:
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) < 4 or data[:4] != _MAGIC:
                continue
            try:
                odom = deserialize_message(data[4:], Odometry)
            except Exception as e:  # noqa: BLE001 - never let a bad datagram kill rx
                self.get_logger().warn(f"udp decode failed: {e}", throttle_duration_sec=2.0)
                continue
            self.pub.publish(odom)

    def destroy_node(self):
        self._run = False
        try:
            self.sock.close()
        except OSError:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = OpponentBridge()
    if not getattr(node, "_ok", False):
        node.destroy_node()
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
