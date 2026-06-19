#!/usr/bin/env python3
"""obstacle_merger — object-level VIL seam.

Intercepts the real tracking/detection output and merges the virtual obstacles
into it, so downstream planners see real + virtual as ONE f110_msgs/ObstacleArray
(same man-in-the-middle pattern as scan_augmentor, but at the object level):

  /tracking/obstacles_raw (real)   ─┐
  /sim/dynamic_obstacles (virtual) ─┼─► /tracking/obstacles (merged)
  /sim/static_obstacles  (virtual) ─┘

Remap your tracking node to publish /tracking/obstacles_raw; this node owns
/tracking/obstacles. In pure sim (no real tracking) it just emits the virtual
obstacles, so the object-level path is testable without perception.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from f110_msgs.msg import ObstacleArray


class ObstacleMerger(Node):
    def __init__(self):
        super().__init__('obstacle_merger')
        self.declare_parameter('in_topic', '/tracking/obstacles_raw')
        self.declare_parameter('out_topic', '/tracking/obstacles')
        self.declare_parameter('virtual_topics',
                               ['/sim/dynamic_obstacles', '/sim/static_obstacles'])
        self.declare_parameter('rate_hz', 25.0)
        self.declare_parameter('watchdog_sec', 0.5)

        self.watchdog = float(self.get_parameter('watchdog_sec').value)
        self.enabled = True
        self.real = None             # (obstacles, time)
        self.virt = {}               # topic -> (obstacles, time)

        for t in self.get_parameter('virtual_topics').value:
            self.create_subscription(ObstacleArray, t,
                                     lambda m, tt=t: self._virt_cb(tt, m), 10)
        self.create_subscription(ObstacleArray, self.get_parameter('in_topic').value,
                                 self._real_cb, 10)
        self.create_subscription(Bool, '/vil/enable', self._en_cb, 10)
        self.pub = self.create_publisher(ObstacleArray, self.get_parameter('out_topic').value, 10)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self._tick)
        self.get_logger().info(
            f"[obstacle_merger] {self.get_parameter('in_topic').value} + virtual "
            f"-> {self.get_parameter('out_topic').value}")

    def _real_cb(self, msg):
        self.real = (list(msg.obstacles), self.get_clock().now())

    def _virt_cb(self, topic, msg):
        self.virt[topic] = (list(msg.obstacles), self.get_clock().now())

    def _en_cb(self, msg):
        self.enabled = bool(msg.data)

    def _fresh(self, entry):
        if entry is None:
            return []
        obs, t = entry
        if (self.get_clock().now() - t).nanoseconds * 1e-9 < self.watchdog:
            return obs
        return []

    def _tick(self):
        out = ObstacleArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        out.obstacles = list(self._fresh(self.real))
        if self.enabled:
            for entry in self.virt.values():
                out.obstacles.extend(self._fresh(entry))
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
