#!/usr/bin/env bash
# Smoke test: bring up the two sim entry launches headless and check the core
# topics/nodes are alive. Prints a PASS/FAIL line per launch and an overall
# verdict. Meant to run inside the build image (see docker/Dockerfile).
# NOTE: no `set -u` — RoboStack's activate.d hooks reference unbound vars
# (e.g. CONDA_BUILD) and would abort the script.
export CONDA_BUILD="${CONDA_BUILD:-}"
. /opt/conda/etc/profile.d/conda.sh && conda activate unicorn
source /ws/install/setup.bash
export RAYCASTER_DIR=/ws/src/unicorn-racing-stack/race_utils/raycaster
export ROS_DOMAIN_ID=0

# virtual display so RViz / pygame don't abort (they are not under test)
if command -v Xvfb >/dev/null 2>&1; then
  Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb.log 2>&1 &
  export DISPLAY=:99
  sleep 1
fi

cleanup() {
  for p in gym_bridge opponent_vehicle opponent_controller scan_augmentor \
           obstacle_merger static_obstacle_manager simple_mux rviz2 \
           waypoint_publisher pp_node frenet_odom state_machine spliner \
           detect_node tracking_node opponent_predictor robot_state_publisher \
           "ros2 launch" "_ros2_launch"; do
    pkill -f "$p" 2>/dev/null
  done
  sleep 3
}

run_one() {  # name  <launch args...>
  local name="$1"; shift
  echo "----- $name -----"
  ros2 launch "$@" > "/tmp/${name}.log" 2>&1 &
  local up=0
  for i in $(seq 1 50); do
    ros2 topic list 2>/dev/null | grep -q '^/scan$' && { up=1; break; }
    sleep 1
  done
  sleep 6
  local nodes scan odom died
  nodes=$(ros2 node list 2>/dev/null | grep -vE 'transform_listener|_ros2cli' | wc -l)
  scan=$(timeout 6 ros2 topic hz /scan --window 5 2>/dev/null | grep -c 'average rate')
  odom=$(timeout 6 ros2 topic hz /car_state/odom --window 5 2>/dev/null | grep -c 'average rate')
  # node deaths, excluding rviz (display not under test)
  died=$(grep 'process has died' "/tmp/${name}.log" 2>/dev/null | grep -vc 'rviz')
  cleanup
  echo "$name: scan_up=$up nodes=$nodes scan_hz=$scan odom_hz=$odom non_rviz_deaths=$died"
  if [ "$scan" -ge 1 ] && [ "$odom" -ge 1 ] && [ "$nodes" -ge 5 ] && [ "$died" -eq 0 ]; then
    echo "$name: PASS"; return 0
  fi
  echo "$name: FAIL"
  echo "--- non-rviz deaths in ${name}.log ---"
  grep 'process has died' "/tmp/${name}.log" | grep -v rviz
  echo "--- first errors in ${name}.log ---"
  grep -iE 'Error|Traceback|No module|cannot|ImportError' "/tmp/${name}.log" \
    | grep -ivE 'shutdown|ExternalShutdown|signal_handler' | head -n 10
  return 1
}

rc=0
run_one lowlevel  stack_master low_level.launch.xml  map:=f sim:=true || rc=1
run_one race stack_master race.launch.xml map:=f sim:=true || rc=1

echo "================================"
[ "$rc" -eq 0 ] && echo "SMOKE: PASS" || echo "SMOKE: FAIL"
exit $rc
