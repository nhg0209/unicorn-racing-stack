"""elrs_joy_ft232_node — FT232RL @ /dev/ELRS_FT232, CRSF native 420000 baud, CRC-8 enabled.

ls /dev/tty.*      to find the correct port for your FT232-based receiver.
The default port is "auto": the launch file probes PORT_CANDIDATES in order and
uses the first one that exists. Override with port:=... if yours differs.

CLI overrides:
    ros2 launch elrs_joy elrs_joy_ft232.launch.py enable_crc:=false
    ros2 launch elrs_joy elrs_joy_ft232.launch.py port:=/dev/tty.usbserial-FT12345

Watch CRC effectiveness:
    ros2 topic echo /elrs_joy_node/debug_stats
"""

import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# Probed in order when port:=auto (first existing match wins); entries may be globs.
# The FT232 device name differs per machine/OS, so cover both: stable Linux by-id
# paths first (survive replug), then generic Linux /dev/ttyUSB*, then macOS names.
# To pin a port per-machine, set ELRS_PORT in unicorn.local.sh (or pass port:=...).
PORT_CANDIDATES = [
    "/dev/serial/by-id/*FT232*",   # Linux, stable across replug
    "/dev/serial/by-id/*FTDI*",    # Linux, stable across replug
    "/dev/ttyUSB0",                # Linux, generic
    "/dev/ttyUSB1",                # Linux, generic
    "/dev/tty.usbserial-10",       # macOS
    "/dev/tty.usbserial-310",      # macOS
]


def resolve_port(context, *args, **kwargs):
    port = LaunchConfiguration("port").perform(context)
    if port == "auto":
        port = os.environ.get("ELRS_PORT") or _probe_port()
        if port is None:
            raise RuntimeError(
                "elrs_joy: no FT232 serial port found. Tried: "
                + ", ".join(PORT_CANDIDATES)
                + ". Plug in the receiver, set ELRS_PORT in unicorn.local.sh, or pass "
                "port:=/dev/... (Linux: `ls /dev/serial/by-id/ /dev/ttyUSB*`; "
                "macOS: `ls /dev/tty.usbserial-*`)."
            )
        print(f"[elrs_joy] auto-selected serial port: {port}")
    return [_make_node(port)]


def _probe_port():
    """First existing device matching PORT_CANDIDATES (each may be a glob)."""
    for pattern in PORT_CANDIDATES:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _make_node(port):
    static_params = {
        "publish_rate": 100,

        "num_axes": 8,
        "num_buttons": 11,

        "axes_joy_indices": [1, 3],
        "axes_crsf_channels": [2, 0],
        # [throttle, steering]. steering(CH0) 좌우 반전 수정: -1.0 → 1.0
        "axes_invert": [1.0, 1.0],

        # Per-axis CRSF calibration (carried over from legacy elrs_joy.launch)
        "axes_cal_min": [174, 174],
        "axes_cal_mid": [1007, 992],
        "axes_cal_max": [1811, 1773],

        "button_joy_indices": [0, 4, 5],
        "button_crsf_channels": [7, 5, 6],
        "button_invert": [1, 0, 0],
        "button_threshold": 992,

        "deadzone": 0.05,
        "failsafe_timeout": 0.5,

        "lb_pressed_max": 350,
        "lb_idle_min": 700,
        "lb_idle_max": 1300,
        "lb_released_min": 1600,
        "lb_debounce_frames": 5,

        "a_debounce_frames": 5,

        "frame_id": "elrs_joy",
    }

    cli_params = {
        "port": port,
        "baud_rate": PythonExpression(["int(\"", LaunchConfiguration("baud_rate"), "\")"]),
        "enable_crc": PythonExpression(["\"", LaunchConfiguration("enable_crc"), "\".lower() == \"true\""]),
        "debug_stats_hz": PythonExpression(["float(\"", LaunchConfiguration("debug_stats_hz"), "\")"]),
        "rb_stability_frames": PythonExpression(["int(\"", LaunchConfiguration("rb_stability_frames"), "\")"]),
        "rb_released_min": PythonExpression(["int(\"", LaunchConfiguration("rb_released_min"), "\")"]),
        "rb_jerk_max": PythonExpression(["int(\"", LaunchConfiguration("rb_jerk_max"), "\")"]),
    }

    return Node(
        package="elrs_joy",
        executable="elrs_joy_ft232_node",
        name="elrs_joy_node",
        output="screen",
        parameters=[static_params, cli_params],
    )


def generate_launch_description():
    args = [
        DeclareLaunchArgument("enable_crc", default_value="true"),
        DeclareLaunchArgument("port", default_value="auto"),
        DeclareLaunchArgument("baud_rate", default_value="420000"),
        DeclareLaunchArgument("debug_stats_hz", default_value="1.0"),
        DeclareLaunchArgument("rb_stability_frames", default_value="10"),
        DeclareLaunchArgument("rb_released_min", default_value="992"),
        DeclareLaunchArgument("rb_jerk_max", default_value="200"),
    ]

    return LaunchDescription(args + [OpaqueFunction(function=resolve_port)])
