"""pp_heading 단독 실행 — low_level + localization이 이미 떠 있을 때
컨트롤러 + 웨이포인트 발행만 띄운다 (글로벌 레이스라인 직접 추종, time-trial).

IFAC2026 stack_master/pp_heading.launch.py의 unicorn 포팅. state_machine / 로컬
플래너 없이 글로벌 레이스라인(global_waypoints.json)을 그대로 추종한다.

전제 — 각각 따로 실행되어 있어야 함:
  1. low_level   (low_level.launch.xml)   → simple_mux + drive 토픽 수신단
  2. localization(localization.launch.xml)→ /car_state/odom

이 launch가 띄우는 것 (둘만):
  global_trajectory_publisher  maps/<map>/global_waypoints.json
        → /global_waypoints (+ /global_waypoints/shortest_path, /centerline_waypoints, /trackbounds/markers)
  pp_heading_controller
        sub  /car_state/odom + /local_waypoints (→ waypoint_topic 로 remap, 기본 /global_waypoints)
        pub  drive_topic (→ simple_mux → 차량)

  ※ 컨트롤러 노드는 /local_waypoints 를 구독한다. 로컬 플래너/state_machine 이 없으므로
    /local_waypoints 를 글로벌 레이스라인(/global_waypoints)으로 remap 해 직접 추종한다.

사용:
  ros2 launch stack_master pp_heading.launch.py map:=<map>
  # drive 토픽이 다르면(실차 mux 입력 등):
  ros2 launch stack_master pp_heading.launch.py map:=<map> drive_topic:=...
  # 속도 스케일된 라인을 따르고 싶고 sector_tuner를 별도로 띄웠다면:
  ros2 launch stack_master pp_heading.launch.py map:=<map> waypoint_topic:=/global_waypoints_scaled
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_arg = DeclareLaunchArgument(
        'map', description='Map folder under stack_master/maps/ (must already have global_waypoints.json)')
    drive_arg = DeclareLaunchArgument(
        'drive_topic', default_value='/vesc/high_level/ackermann_cmd',
        description='Controller output topic (-> simple_mux -> vehicle). unicorn sim/real default.')
    wpnt_arg = DeclareLaunchArgument(
        'waypoint_topic', default_value='/global_waypoints',
        description="WpntArray the controller follows (remapped onto its /local_waypoints sub): "
                    "/global_waypoints (raw raceline, default) or /global_waypoints_scaled "
                    "(needs sector_tuner running separately)")

    map_name = LaunchConfiguration('map')
    drive_topic = LaunchConfiguration('drive_topic')
    waypoint_topic = LaunchConfiguration('waypoint_topic')

    # ── 1. 웨이포인트 발행: global_waypoints.json → /global_waypoints (+ sp/centerline/trackbounds) ──
    global_repub = Node(
        package='gb_optimizer', executable='global_trajectory_publisher',
        name='global_trajectory_publisher', output='screen',
        parameters=[{'map': map_name}],
    )

    # ── 2. pp_heading 컨트롤러 (글로벌 레이스라인 직접 추종) ──
    #    노드는 /local_waypoints 를 구독 → waypoint_topic(기본 /global_waypoints)으로 remap.
    #    발행이 먼저 올라오도록 잠깐 지연 후 기동.
    pp_yaml = os.path.join(
        get_package_share_directory('controller'), 'config', 'pp_heading_params.yaml')
    pp_heading = TimerAction(period=2.0, actions=[Node(
        package='controller', executable='pp_heading_controller', name='pp_heading_controller',
        parameters=[pp_yaml, {'drive_topic': drive_topic}],
        remappings=[('/local_waypoints', waypoint_topic)],
        output='screen',
    )])

    return LaunchDescription([map_arg, drive_arg, wpnt_arg, global_repub, pp_heading])
