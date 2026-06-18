import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    urg_node_dir = get_package_share_directory('urg_node')

    launch_description = LaunchDescription([
        DeclareLaunchArgument(
            'sensor_interface',
            default_value='ethernet',
            description='sensor_interface: supported: serial, ethernet'),
    ])

    def expand_param_file_name(context):
        sensor_interface = context.launch_configurations['sensor_interface']
        param_file = os.path.join(
            urg_node_dir, 'launch', f'urg_node_{sensor_interface}.yaml'
        )
        if os.path.exists(param_file):
            return [SetLaunchConfiguration('param', param_file)]
        # 파일이 없으면 에러를 내고 싶다면 여기서 raise 해도 됨
        return []

    launch_description.add_action(OpaqueFunction(function=expand_param_file_name))

    hokuyo_node = Node(
        package='urg_node',
        executable='urg_node_driver',   # ✅ Jazzy에서는 executable 키가 필수
        name='urg_node',                # (선택) 노드 이름
        output='screen',
        parameters=[LaunchConfiguration('param')],
    )

    launch_description.add_action(hokuyo_node)
    return launch_description
