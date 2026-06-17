#!/bin/bash
###############################################################################
# build_packages_on_car.sh
#
# Build the UNICORN racing-stack ROS 2 Jazzy workspace on the F1TENTH car
# (Ubuntu 24.04 + ROS 2 Jazzy) — FULL profile (sensor drivers, SLAM, PF).
#
# This meta-repo lives at <workspace>/src/unicorn-racing-stack. Component repos
# are imported as SIBLINGS into <workspace>/src. In addition to the sim build
# it installs heavy apt deps, removes CAC's car-only COLCON_IGNORE markers,
# imports CAC's nested Hokuyo driver repos, and builds range_libc.
#
# Usage: bash build_packages_on_car.sh
###############################################################################
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this meta-repo
WS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"                  # colcon workspace root
CAC="${WS_DIR}/src/creating_autonomous_car"

echo "============================================"
echo " Building UNICORN racing-stack (CAR / full)"
echo " Workspace : ${WS_DIR}"
echo " Meta-repo : ${SCRIPT_DIR}"
echo "============================================"

# 1. apt dependencies (incl. cartographer / pcl / ceres) ----------------------
echo "[1/8] Installing apt dependencies..."
sudo apt update
sudo apt install -y \
    python3-rosdep python3-pip python3-skimage python3-numpy cython3 \
    python3-vcstool python3-colcon-common-extensions \
    ros-jazzy-xacro ros-jazzy-rmw-cyclonedds-cpp \
    libboost-dev libboost-iostreams-dev libcairo2-dev libceres-dev \
    libgflags-dev libgoogle-glog-dev liblua5.2-dev libprotobuf-dev \
    protobuf-compiler libabsl-dev libpcl-dev google-mock gedit

# 2. Import component repos (siblings) + CAC's nested urg_node repos -----------
echo "[2/8] Importing component repos via vcstool..."
mkdir -p "${WS_DIR}/src"
vcs import "${WS_DIR}/src" < "${SCRIPT_DIR}/unicorn.repos"
if [ -f "${CAC}/sensor/urg_node/additional_repos.repos" ]; then
    vcs import "${CAC}/sensor/urg_node" < "${CAC}/sensor/urg_node/additional_repos.repos"
fi

# 3. Enable car-only packages (remove sim COLCON_IGNORE markers) --------------
echo "[3/8] Enabling car-only packages..."
for ig in \
    "${CAC}/sensor/vesc/COLCON_IGNORE" \
    "${CAC}/sensor/urg_node/COLCON_IGNORE" \
    "${CAC}/slam/cartographer/COLCON_IGNORE" \
    "${CAC}/slam/cartographer_ros/COLCON_IGNORE" \
    "${CAC}/slam/particle_filter/COLCON_IGNORE" ; do
    if [ -f "${ig}" ]; then rm -v "${ig}"; else echo "  (absent) ${ig}"; fi
done

# 4. Python dependencies ------------------------------------------------------
echo "[4/8] Installing Python dependencies..."
pip install -e "${CAC}/simulator/f1tenth_gym" --break-system-packages
pip install transforms3d pynput --break-system-packages
pip install --upgrade coverage --break-system-packages

# 5. range_libc (particle-filter localization) --------------------------------
echo "[5/8] Installing range_libc..."
( cd "${CAC}/slam/range_libc/pywrapper" && pip3 install . --user --break-system-packages )
python3 -c "import range_libc; print('  range_libc import OK')" || \
    echo "  WARNING: range_libc import failed."

# 6. rosdep -------------------------------------------------------------------
echo "[6/8] Running rosdep..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update
rosdep install --from-paths "${WS_DIR}/src" --ignore-src -r -y

# 7. Build --------------------------------------------------------------------
echo "[7/8] Building workspace with colcon..."
cd "${WS_DIR}"
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# 8. ~/.bashrc (idempotent) ---------------------------------------------------
echo "[8/8] Updating ~/.bashrc..."
BASHRC="${HOME}/.bashrc"
SOURCE_LINE="source ${WS_DIR}/install/setup.bash"
if ! grep -qF "${SOURCE_LINE}" "${BASHRC}" 2>/dev/null; then
    printf '\n# UNICORN racing-stack (Jazzy) workspace\n%s\n' "${SOURCE_LINE}" >> "${BASHRC}"
fi
if ! grep -qF "RMW_IMPLEMENTATION" "${BASHRC}" 2>/dev/null; then
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> "${BASHRC}"
fi

echo ""
echo "============================================"
echo " Build complete! (CAR / full)"
echo " Run 'source ~/.bashrc' or open a new terminal."
echo "============================================"
