from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_dir_arg = DeclareLaunchArgument(
        'config_dir', default_value='',
        description='Cartographer configuration directory')
    config_base_arg = DeclareLaunchArgument(
        'config_base', default_value='localization.lua',
        description='Cartographer configuration basename')

    return LaunchDescription([
        config_dir_arg,
        config_base_arg,
        Node(
            package='set_pose',
            executable='set_pose_node',
            name='setpose',
            output='screen',
            parameters=[{
                'config_dir': LaunchConfiguration('config_dir'),
                'config_base': LaunchConfiguration('config_base'),
            }],
        ),
    ])
