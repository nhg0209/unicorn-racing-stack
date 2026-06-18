#!/usr/bin/env python3

import math
import numpy as np
from numpy import random

import rclpy
from rclpy.node import Node

from f110_msgs.msg import ObstacleArray, Obstacle, WpntArray
from nav_msgs.msg import Odometry


class ObstaclePublisher(Node):
    def __init__(self):
        super().__init__('random_obstacle_publisher')

        self.create_subscription(Odometry, '/car_state/odom_frenet', self.odom_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.global_trajectory_cb, 10)
        self.obstacle_pub = self.create_publisher(ObstacleArray, '/obstacles', 10)

        self.declare_parameter('n_obstacles', 8)
        self.declare_parameter('publish_at_lookahead', False)
        self.declare_parameter('lookahead_distance', 5.0)
        self.declare_parameter('obstacle_width', 0.2)
        self.declare_parameter('obstacle_length', 0.3)
        self.declare_parameter('obstacle_max_d_from_traj', 1.0)
        self.declare_parameter('rnd_seed', 84)

        self.n_sectors = self.get_parameter('n_obstacles').value + 1
        self.publish_at_lookahead = self.get_parameter('publish_at_lookahead').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.obstacle_width = self.get_parameter('obstacle_width').value
        self.obstacle_length = self.get_parameter('obstacle_length').value
        self.obstacle_max_d_from_traj = self.get_parameter('obstacle_max_d_from_traj').value
        seed = self.get_parameter('rnd_seed').value

        self.obstacle_array = []
        self.has_traj = False
        self.has_odom = False
        self.gen = random.default_rng(seed)
        self.s = 0.0
        self.final_s = 0.0
        self.initialized = False

        # 25 Hz main loop
        self.timer = self.create_timer(1.0 / 25.0, self.loop)

    def loop(self):
        if not self.initialized:
            if self.has_traj and self.has_odom:
                self.update_obstacles()
                self.initialized = True
            return
        self.publish_obstacles()

    def update_obstacles(self):
        if self.has_traj:
            self.obstacle_array.clear()
            self.final_s = self.gb_wpnts[-1].s_m
            s_spacing = self.final_s / self.n_sectors
            margin = max(0.5, self.obstacle_length)
            for sec in range(self.n_sectors):
                s_start = sec * s_spacing
                s_end = s_start + s_spacing - margin
                ob = self.generate_random_obstacle(sec, s_start, s_end)
                self.obstacle_array.append(ob)

    def generate_random_obstacle(self, id, s_start, s_end):
        ob = Obstacle()
        ob.id = id
        # random s within s_start and s_end
        p1 = self.gen.random()
        ob.s_start = s_start + (s_end - s_start) * p1
        ob.s_end = ob.s_start + self.obstacle_length

        # get track bounds at s_start
        wpt_id = self.get_closest_point_on_traj(ob.s_start)
        track_right = -min(self.gb_wpnts[wpt_id].d_right, self.obstacle_max_d_from_traj)  # negative
        track_left = min(self.gb_wpnts[wpt_id].d_left - self.obstacle_width, self.obstacle_max_d_from_traj)

        p2 = self.gen.random()
        ob.d_right = track_right + (track_left - track_right) * p2
        ob.d_left = ob.d_right + self.obstacle_width
        ob.is_actually_a_gap = False

        return ob

    def get_closest_point_on_traj(self, s):
        min_d = 1000
        id = 0
        for wpt in self.gb_wpnts:
            d = (wpt.s_m - s) ** 2
            if d < min_d:
                min_d = d
                id = wpt.id
        return id

    def publish_obstacles(self):
        obstacle_msg = ObstacleArray()
        obstacle_msg.header.stamp = self.get_clock().now().to_msg()
        obstacle_msg.header.frame_id = "frenet"
        if self.publish_at_lookahead:
            for ob in self.obstacle_array:
                # too lazy to handle wrapping
                dist = math.fmod(self.s + self.lookahead_distance, self.final_s) - ob.s_start
                if dist > 0 and dist < (self.lookahead_distance + 1):
                    obstacle_msg.obstacles.append(ob)
        else:
            obstacle_msg.obstacles = self.obstacle_array
        self.obstacle_pub.publish(obstacle_msg)

    def global_trajectory_cb(self, msg):
        self.has_traj = True
        self.gb_wpnts = msg.wpnts

    def odom_cb(self, msg):
        self.has_odom = True
        self.s = msg.pose.pose.position.x


def main():
    rclpy.init()
    node = ObstaclePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
