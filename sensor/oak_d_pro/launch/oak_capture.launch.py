import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_dir = get_package_share_directory('oak_d_pro')
    depthai_dir = get_package_share_directory('depthai_ros_driver')

    return LaunchDescription([
        DeclareLaunchArgument('record', default_value='false'),
        DeclareLaunchArgument('bag_path', default_value=''),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(depthai_dir, 'launch', 'camera.launch.py')
            ),
            launch_arguments={
                'params_file': os.path.join(pkg_dir, 'config', 'rgbd_capture.yaml'),
            }.items(),
        ),

        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-a',
                '--exclude', '.*/(compressed|compressedDepth|theora|zstd)',
                '-o', LaunchConfiguration('bag_path'),
            ],
            condition=IfCondition(LaunchConfiguration('record')),
        ),
    ])
