import math
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class EStop:
    def __init__(self, node):
        p = lambda name: node.get_parameter(name).value
        self._logger     = node.get_logger()

    def should_stop(self, scan: LaserScan, odom: Odometry, cmd=None):
        if cmd is None:
            cmd = AckermannDriveStamped()

        # TODO: Implement an emergency stop (E-Stop) using:
        #   - 2D LiDAR scan data
        #   - Wheel odometry data from the VESC
        #   - TTC (Time-to-Collision) based logic
        #
        # You may modify `cmd` (the original control command) in this function.
        #
        # Useful information:
        #   - scan.ranges                  : distance array [m] for each LiDAR beam
        #   - scan.angle_min               : angle of the first beam [rad]
        #   - scan.angle_max               : angle of the last beam [rad]
        #   - scan.angle_increment         : angular step between beams [rad]
        #   - odom.twist.twist.linear.x    : vehicle forward speed [m/s]
        #   - odom.twist.twist.angular.z   : vehicle yaw rate [rad/s]

        if True:
            self._logger.warn(f'EStop triggered')
            cmd.drive.speed = 0.0
        return cmd
    
    
    
    
    