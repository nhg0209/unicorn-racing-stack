"""Livox MID-360 driver bringup for the real car (macOS).

Wrapper around livox_ros_driver2/launch_ROS2/msg_MID360_launch.py that fixes
the macOS dylib resolution problem: install/setup.bash does NOT populate
DYLD_LIBRARY_PATH (macOS SIP strips it from the child env), so the livox
component plugin's dlopen() fails to resolve its sibling dylibs. We enumerate
every install/<pkg>/lib in this workspace and prepend them via
SetEnvironmentVariable, which launch's process spawn honours even under SIP.

Publishes /livox/lidar (PointCloud2) + /livox/imu in frame_id 'livox_frame'.
Included by low_level.launch.xml under the `livox` arg (default on).

  ros2 launch stack_master livox_lidar.launch.py
  ros2 launch stack_master livox_lidar.launch.py use_system_timestamp:=false
"""
import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    use_system_timestamp = LaunchConfiguration('use_system_timestamp')

    livox_share = get_package_share_directory('livox_ros_driver2')
    # <install>/livox_ros_driver2/share/livox_ros_driver2  ->  <install>
    ws_install = os.path.dirname(os.path.dirname(os.path.dirname(livox_share)))

    # macOS dylib resolution: prepend every install/<pkg>/lib so the livox
    # class_loader plugin can dlopen its dependencies (SIP strips DYLD from
    # setup.bash's child env).
    dyld_extra = ':'.join(sorted(glob.glob(os.path.join(ws_install, '*', 'lib'))))
    set_dyld = SetEnvironmentVariable(
        name='DYLD_LIBRARY_PATH',
        value=PythonExpression([
            "'", dyld_extra, "' + (':' + __import__('os').environ['DYLD_LIBRARY_PATH'] "
            "if 'DYLD_LIBRARY_PATH' in __import__('os').environ else '')"
        ]),
    )

    livox_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(livox_share, 'launch_ROS2', 'msg_MID360_launch.py')
        ),
        launch_arguments={'use_system_timestamp': use_system_timestamp}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_system_timestamp', default_value='true',
            description='livox 스탬프 소스. true=ROS now()(센서 동기/mapping), '
                        'false=장치 하드웨어 시각(LIO-only localization).'),
        set_dyld,
        livox_driver,
    ])
