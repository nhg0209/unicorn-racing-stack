"""
create_path.launch.py — build the global raceline from an EXISTING map using the
IFAC2026-ported 2-stage optimizer (IQP -> SP) with [map]_modi.png path guidance.

This is the IFAC-optimizer counterpart of raceline_generator.launch.xml (which
uses gb_optimizer's native global_planner_node). It follows the same gb flow:
serve the saved map, generate global_waypoints.json, republish it, and slice the
speed / overtaking sectors.

Pipeline (sequential via OnProcessExit):
  1. centerline_extractor  -> maps/<map>/centerline.csv + boundary_{right,left}.csv
        (uses maps/<map>/<map>_modi.png for widths if present; real walls from <map>.png)
  2. trajectory_optimizer  -> maps/<map>/global_waypoints.json  (IQP + SP, unicorn schema)
  3. global_trajectory_publisher (reads json -> /global_waypoints, ...)
     + sector_slicer + ot_sector_slicer  (write speed_scaling.yaml / ot_sectors.yaml)

Vehicle + algo params come from stack_master/config/<racecar_version>/racecar_f110.ini
and .../veh_dyn_info/ (CAR by default).

    ros2 launch stack_master create_path.launch.xml map:=map_test
    ros2 launch stack_master create_path.launch.xml map:=f safety_width_iqp:=0.8 safety_width_sp:=0.4
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    maps_dir = os.path.join(get_package_share_directory('stack_master'), 'maps')

    map_arg      = DeclareLaunchArgument('map', description='Map folder under stack_master/maps/')
    sim_arg      = DeclareLaunchArgument('sim', default_value='false',
                                         description='Pick SIM vs CAR config folder')
    rcv_arg      = DeclareLaunchArgument('racecar_version', default_value='CAR',
                                         description='Config folder under stack_master/config/ (racecar_f110.ini + veh_dyn_info)')
    reverse_arg  = DeclareLaunchArgument('reverse', default_value='false',
                                         description='Reverse raceline direction (CW)')
    siqp_arg     = DeclareLaunchArgument('safety_width_iqp', default_value='-1.0',
                                         description='IQP safety width [m]. <0 (default) = use racecar_f110.ini optim_opts_mincurv.width_opt; >0 to override')
    ssp_arg      = DeclareLaunchArgument('safety_width_sp', default_value='-1.0',
                                         description='SP safety width [m]. <0 (default) = use racecar_f110.ini optim_opts_shortest_path.width_opt; >0 to override')
    mintime_arg  = DeclareLaunchArgument('enable_mintime', default_value='false',
                                         description='Also run opt_mintime (CasADi required)')
    show_arg     = DeclareLaunchArgument('show_plots', default_value='false',
                                         description='Show extractor matplotlib plots (needs display)')

    map_name        = LaunchConfiguration('map')
    racecar_version = LaunchConfiguration('racecar_version')
    reverse         = LaunchConfiguration('reverse')
    safety_iqp      = LaunchConfiguration('safety_width_iqp')
    safety_sp       = LaunchConfiguration('safety_width_sp')
    enable_mintime  = LaunchConfiguration('enable_mintime')
    show_plots      = LaunchConfiguration('show_plots')

    map_folder = PathJoinSubstitution([maps_dir, map_name])
    map_yaml   = PathJoinSubstitution([maps_dir, map_name, [map_name, '.yaml']])

    # ── map source: serve the saved map (RViz / parity with raceline_generator) ──
    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server', output='screen',
        parameters=[{'yaml_filename': map_yaml}],
    )
    lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager', output='screen',
        parameters=[{'autostart': True, 'node_names': ['map_server']}],
    )

    # ── 1. centerline extraction (writes centerline.csv + boundary CSVs) ──
    extractor = Node(
        package='gb_optimizer', executable='centerline_extractor',
        name='centerline_extractor', output='screen',
        parameters=[{
            'map_name':   map_name,
            'reverse':    reverse,
            'output_csv': 'centerline.csv',
            'show_plots': show_plots,
        }],
    )

    # ── 2. IQP -> SP optimization (writes global_waypoints.json) ──
    optimizer = Node(
        package='gb_optimizer', executable='trajectory_optimizer',
        name='trajectory_optimizer', output='screen', emulate_tty=True,
        parameters=[{
            'map_name':         map_name,
            'racecar_version':  racecar_version,
            'safety_width_iqp': safety_iqp,
            'safety_width_sp':  safety_sp,
            'enable_mintime':   enable_mintime,
        }],
    )

    # ── 3. republish json -> topics + slice sectors ──
    republisher = Node(
        package='gb_optimizer', executable='global_trajectory_publisher',
        name='global_planner', output='screen',
        parameters=[{'map': map_name}],
    )
    sector_slicer = Node(
        package='sector_tuner', executable='sector_slicer', name='sector_node', output='screen',
        parameters=[{'save_dir': map_folder}],
    )
    ot_sector_slicer = Node(
        package='overtaking_sector_tuner', executable='ot_sector_slicer',
        name='ot_sector_node', output='screen',
        parameters=[{'save_dir': map_folder}],
    )

    # optimizer runs after the extractor exits; republisher + slicers after the optimizer exits.
    after_extractor = RegisterEventHandler(
        OnProcessExit(target_action=extractor, on_exit=[optimizer]))
    after_optimizer = RegisterEventHandler(
        OnProcessExit(target_action=optimizer,
                      on_exit=[republisher, sector_slicer, ot_sector_slicer]))

    return LaunchDescription([
        map_arg, sim_arg, rcv_arg, reverse_arg, siqp_arg, ssp_arg, mintime_arg, show_arg,
        map_server, lifecycle,
        extractor,
        after_extractor,
        after_optimizer,
    ])
