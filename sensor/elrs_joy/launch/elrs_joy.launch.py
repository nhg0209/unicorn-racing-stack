"""elrs_joy_node — CP2102 / legacy serial @ /dev/ELRS (no CRC, range-only validation)."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = {
        "port": "/dev/ELRS",
        "baud_rate": 115200,  # CP2102 cannot hit 420000 exactly; legacy node uses range-only validation
        "publish_rate": 100,

        # Xbox-compatible Joy: 8 axes, 11 buttons
        "num_axes": 8,
        "num_buttons": 11,

        # axes[1]=Throttle(CH2), axes[3]=Steering(CH0)
        "axes_joy_indices": [1, 3],
        "axes_crsf_channels": [2, 0],
        # [throttle, steering]. steering(CH0) 좌우 반전 수정: -1.0 → 1.0
        "axes_invert": [1.0, 1.0],

        # Per-axis calibration measured on this transmitter
        # throttle: 174 / 1007 / 1811   (full range, mid offset from 992)
        # steering: 174 /  992 / 1773   (positive end short of 1811)
        "axes_cal_min": [174, 174],
        "axes_cal_mid": [1007, 992],
        "axes_cal_max": [1811, 1773],

        # buttons[0]=A(CH7, latched, invert=1), buttons[4]=LB(CH5), buttons[5]=RB(CH6)
        "button_joy_indices": [0, 4, 5],
        "button_crsf_channels": [7, 5, 6],
        "button_invert": [1, 0, 0],
        "button_threshold": 992,

        "deadzone": 0.05,
        "failsafe_timeout": 0.5,

        # LB 3-position guard
        "lb_pressed_max": 350,
        "lb_idle_min": 700,
        "lb_idle_max": 1300,
        "lb_released_min": 1600,
        "lb_debounce_frames": 5,

        # A asymmetric debounce
        "a_debounce_frames": 5,

        "frame_id": "elrs_joy",
    }

    return LaunchDescription([
        Node(
            package="elrs_joy",
            executable="elrs_joy_node",
            name="elrs_joy_node",
            output="screen",
            parameters=[params],
        ),
    ])
