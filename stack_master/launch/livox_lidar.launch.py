"""Livox MID-360 driver bringup for the real car — publishes PointCloud2.

Launches livox_ros_driver2_node directly with xfer_format=0
(sensor_msgs/PointCloud2), because kiss_icp_localization subscribes to
/livox/lidar as PointCloud2. The stock msg_MID360_launch.py uses xfer_format=1
(livox_ros_driver2/CustomMsg), which is a topic-type mismatch — DDS silently
drops it and kiss never receives a cloud. We do NOT use rviz_MID360_launch.py
(also PointCloud2) because it additionally spawns an rviz node (bad on the
headless car). Params otherwise mirror the stock MID360 launch and read the
same install-share MID360_config.json (host 192.168.1.102, lidar .197).

The macOS dylib shim (SetEnvironmentVariable DYLD_LIBRARY_PATH) is kept — it is
a no-op on Linux but lets the class_loader plugin dlopen its deps under macOS SIP.

Publishes /livox/lidar (PointCloud2) + /livox/imu in frame_id 'livox_frame'.
Included by low_level.launch.xml under the `livox` arg (default on).

  ros2 launch stack_master livox_lidar.launch.py
  ros2 launch stack_master livox_lidar.launch.py use_system_timestamp:=false
"""
import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    livox_share = get_package_share_directory('livox_ros_driver2')
    # <install>/livox_ros_driver2/share/livox_ros_driver2  ->  <install>
    ws_install = os.path.dirname(os.path.dirname(os.path.dirname(livox_share)))
    user_config_path = os.path.join(livox_share, 'config', 'MID360_config.json')

    # macOS dylib resolution: prepend every install/<pkg>/lib so the livox
    # class_loader plugin can dlopen its dependencies (SIP strips DYLD from
    # setup.bash's child env). No-op on Linux.
    dyld_extra = ':'.join(sorted(glob.glob(os.path.join(ws_install, '*', 'lib'))))
    set_dyld = SetEnvironmentVariable(
        name='DYLD_LIBRARY_PATH',
        value=PythonExpression([
            "'", dyld_extra, "' + (':' + __import__('os').environ['DYLD_LIBRARY_PATH'] "
            "if 'DYLD_LIBRARY_PATH' in __import__('os').environ else '')"
        ]),
    )

    livox_driver = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[{
            'xfer_format': 0,          # 0 = sensor_msgs/PointCloud2 (kiss subscribes this)
            'multi_topic': 0,          # all LiDARs share /livox/lidar
            'data_src': 0,             # 0 = live lidar
            'publish_freq': 10.0,
            'output_data_type': 0,
            'frame_id': 'livox_frame',
            'lvx_file_path': '/home/livox/livox_test.lvx',
            'user_config_path': user_config_path,
            'cmdline_input_bd_code': 'livox0000000001',
        }],
    )

    return LaunchDescription([
        # Declared for launch-interface compat (low_level passes it); timestamp
        # handling lives in MID360_config.json + the kiss side (stamp_at_scan_end).
        DeclareLaunchArgument(
            'use_system_timestamp', default_value='true',
            description='livox stamp source (currently informational; see kiss.yaml).'),
        set_dyld,
        livox_driver,
    ])
