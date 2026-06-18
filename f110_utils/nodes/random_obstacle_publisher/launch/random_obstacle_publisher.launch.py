from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('n_obstacles', default_value='8'),
        DeclareLaunchArgument('publish_at_lookahead', default_value='false'),
        DeclareLaunchArgument('lookahead_distance', default_value='5.0'),
        DeclareLaunchArgument('rnd_seed', default_value='84'),
        DeclareLaunchArgument('obstacle_width', default_value='0.2'),
        DeclareLaunchArgument('obstacle_length', default_value='0.3'),
        DeclareLaunchArgument('obstacle_max_d_from_traj', default_value='1.0'),
    ]

    node = Node(
        package='random_obstacle_publisher',
        executable='random_obstacle_publisher',
        name='random_obstacle_publisher',
        output='screen',
        parameters=[{
            'n_obstacles': LaunchConfiguration('n_obstacles'),
            'publish_at_lookahead': LaunchConfiguration('publish_at_lookahead'),
            'lookahead_distance': LaunchConfiguration('lookahead_distance'),
            'obstacle_width': LaunchConfiguration('obstacle_width'),
            'obstacle_length': LaunchConfiguration('obstacle_length'),
            'obstacle_max_d_from_traj': LaunchConfiguration('obstacle_max_d_from_traj'),
            'rnd_seed': LaunchConfiguration('rnd_seed'),
        }],
    )

    return LaunchDescription(args + [node])
