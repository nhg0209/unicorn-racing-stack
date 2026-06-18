#!/usr/bin/env python3
"""
Standalone velocity scaler node (unicorn ROS1 `vel_scaler_node.py` port).

In the ROS1 stack this was a separate node that subscribed to
`/dyn_sector_speed/parameter_updates` (a dynamic_reconfigure topic) and
republished the velocity-scaled global waypoints. The merged `sector_tuner`
node already performs this scaling, so this node is kept for parity / optional
standalone use. In ROS2 there is no dynamic_reconfigure parameter_updates topic,
so the sector scalings are read from this node's own parameters and updated via
parameter events on the /sector_tuner node.
"""
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib
from f110_msgs.msg import WpntArray
from std_msgs.msg import Bool

from sector_tuner.parameter_event_handler import ParameterEventHandler


class VelocityScaler(Node):
    """Sector scaler for the velocity of the global waypoints."""

    def __init__(self) -> None:
        super().__init__('velocity_scaler',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.debug_plot = self._get_param_default('debug_plot', False)

        # sectors params
        self.glb_wpnts_og = None
        self.glb_wpnts_scaled = None
        self.glb_wpnts_sp_og = None
        self.glb_wpnts_sp_scaled = None
        self.update_map = False

        # get initial scaling from parameters (loaded from speed_scaling.yaml)
        self.sectors_params = self.parameters_to_dict()
        self.n_sectors = self.sectors_params['n_sectors']

        # dyn params via parameter events on the sector_tuner node
        self.glb_wpnts_name = "/global_waypoints"
        self.handler = ParameterEventHandler(self)
        self.callback_handle = self.handler.add_parameter_event_callback(
            callback=self.dyn_param_cb,
        )
        self.create_subscription(WpntArray, self.glb_wpnts_name, self.glb_wpnts_cb, 10)
        self.create_subscription(WpntArray, self.glb_wpnts_name + "/shortest_path", self.glb_wpnts_sp_cb, 10)
        self.create_subscription(Bool, "update_map", self.update_map_cb, 10)

        # new glb_waypoints pub
        self.scaled_points_pub = self.create_publisher(WpntArray, "/global_waypoints_scaled", 10)
        self.scaled_points_sp_pub = self.create_publisher(WpntArray, "/global_waypoints_scaled/shortest_path", 10)

        self.get_logger().info("Waiting for global waypoints...")
        self.timer = self.create_timer(0.5, self.loop)

    def _get_param_default(self, name, default):
        try:
            val = self.get_parameter(name).value
            return val if val is not None else default
        except Exception:
            return default

    def parameters_to_dict(self):
        params = {}
        for key in self._parameters:
            keylist = key.split('.')
            paramit = params
            for subkey in keylist[:-1]:
                paramit = paramit.setdefault(subkey, {})
            paramit[keylist[-1]] = self.get_parameter(key).value
        return params

    def update_map_cb(self, data: Bool):
        self.update_map = True

    def glb_wpnts_cb(self, data: WpntArray):
        self.glb_wpnts_og = data

    def glb_wpnts_sp_cb(self, data: WpntArray):
        self.glb_wpnts_sp_og = data

    def dyn_param_cb(self, parameter_event):
        """Notices the change in the parameters and scales the global waypoints."""
        if parameter_event.node not in ('/sector_tuner', '/velocity_scaler'):
            return
        updated = self.parameters_to_dict()
        if 'global_limit' in updated:
            self.sectors_params['global_limit'] = updated['global_limit']
        for i in range(self.n_sectors):
            if f"Sector{i}" in updated and 'scaling' in updated[f"Sector{i}"]:
                self.sectors_params[f"Sector{i}"]['scaling'] = np.clip(
                    updated[f"Sector{i}"]['scaling'], 0, self.sectors_params['global_limit'])
        self.get_logger().info(str(self.sectors_params))

    def get_vel_scaling(self, s):
        """Gets the velocity scaling for the points, interpolating between sectors."""
        hl_change = 10

        if self.n_sectors > 1:
            for i in range(self.n_sectors):
                if i == 0:
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start'] - hl_change, self.sectors_params[f'Sector{i}']['start'] + hl_change],
                            fp=[self.sectors_params[f'Sector{self.n_sectors-1}']['scaling'], self.sectors_params[f'Sector{i}']['scaling']]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start'] - hl_change):
                        scaler = self.sectors_params[f"Sector{i}"]['scaling']
                    elif (s >= self.sectors_params[f'Sector{i+1}']['start'] - hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start']):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i+1}']['start'] - hl_change, self.sectors_params[f'Sector{i+1}']['start'] + hl_change],
                            fp=[self.sectors_params[f'Sector{i}']['scaling'], self.sectors_params[f'Sector{i+1}']['scaling']]
                        )
                elif i != self.n_sectors - 1:
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start'] - hl_change, self.sectors_params[f'Sector{i}']['start'] + hl_change],
                            fp=[self.sectors_params[f'Sector{i-1}']['scaling'], self.sectors_params[f'Sector{i}']['scaling']]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start'] - hl_change):
                        scaler = self.sectors_params[f"Sector{i}"]['scaling']
                    elif (s >= self.sectors_params[f'Sector{i+1}']['start'] - hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start']):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i+1}']['start'] - hl_change, self.sectors_params[f'Sector{i+1}']['start'] + hl_change],
                            fp=[self.sectors_params[f'Sector{i}']['scaling'], self.sectors_params[f'Sector{i+1}']['scaling']]
                        )
                else:
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start'] - hl_change, self.sectors_params[f'Sector{i}']['start'] + hl_change],
                            fp=[self.sectors_params[f'Sector{i-1}']['scaling'], self.sectors_params[f'Sector{i}']['scaling']]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i}']['end'] - hl_change):
                        scaler = self.sectors_params[f"Sector{i}"]['scaling']
                    elif (s >= self.sectors_params[f'Sector{i}']['end'] - hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['end'] - hl_change, self.sectors_params[f'Sector{i}']['end'] + hl_change],
                            fp=[self.sectors_params[f'Sector{i}']['scaling'], self.sectors_params[f'Sector{0}']['scaling']]
                        )
        elif self.n_sectors == 1:
            scaler = self.sectors_params["Sector0"]['scaling']

        return scaler

    def scale_points(self):
        """Scales the global waypoints' velocities."""
        scaling = []

        if self.glb_wpnts_scaled is None or self.update_map:
            self.glb_wpnts_scaled = self.glb_wpnts_og
            self.glb_wpnts_sp_scaled = self.glb_wpnts_sp_og
            self.update_map = False

        for i, wpnt in enumerate(self.glb_wpnts_og.wpnts):
            vel_scaling = self.get_vel_scaling(i)
            new_vel = wpnt.vx_mps * vel_scaling
            self.glb_wpnts_scaled.wpnts[i].vx_mps = new_vel
            scaling.append(vel_scaling)

        if self.debug_plot:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(scaling)
            plt.legend(['og', 'scaled'])
            plt.ylim(0, 1)
            plt.pause(0.001)

    def loop(self):
        if self.glb_wpnts_og is None:
            return
        self.scale_points()
        self.scaled_points_pub.publish(self.glb_wpnts_scaled)


def main():
    rclpy.init()
    node = VelocityScaler()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
