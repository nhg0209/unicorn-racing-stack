#!/usr/bin/env bash
# unicorn.sh — enter the unicorn-racing-stack dev environment in one step.
#
# SOURCE it (do not execute). Add an alias to your ~/.bashrc:
#     alias unicorn='source /path/to/unicorn-racing-stack/unicorn.sh'
# then just run:  unicorn
#
# It (1) activates the RoboStack conda env, (2) selects CycloneDDS + ROS domain,
# (3) sources the colcon workspace, and (4) defines ros2kill / cbuild helpers.

# --- locate this repo and the colcon workspace root (<ws>/src/<repo>) ---
_URS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_URS_WS="$(cd "$_URS_REPO/../.." && pwd)"

# --- 1) conda env: RoboStack ROS 2 Jazzy ('unicorn') ---
source "$(conda info --base)/etc/profile.d/conda.sh"

# Start from a CLEAN ROS environment. These vars are 100% ROS-owned, so reset
# them BEFORE activating: whatever the host shell leaked — a system ROS or another
# workspace `source`d in ~/.bashrc, ANY distro, ANY path — is discarded, and
# conda's activation + this workspace rebuild them. You can't pattern-match every
# user's ~/.bashrc, so don't try; reset to a known-good baseline instead.
unset AMENT_PREFIX_PATH AMENT_CURRENT_PREFIX CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_PACKAGE_PATH 2>/dev/null

conda activate unicorn

# Never let ~/.local user-site packages shadow the conda env (stale numba, etc.).
export PYTHONNOUSERSITE=1

# The mixed path vars (PYTHONPATH/LD_LIBRARY_PATH/PATH also carry non-ROS entries
# like CUDA) can't just be unset, so drop only the ROS leakage: /opt/ros/* and
# apt-style ROS python dirs (.../lib/pythonX/dist-packages — conda/colcon use
# site-packages, so dist-packages is exclusively system ROS). This is what
# shadowed rosidl_generator_c on the Orin ("generate_c() takes 1 arg but 2 given").
_urs_clean_paths() {
    local var p new parts
    for var in PYTHONPATH LD_LIBRARY_PATH PATH; do
        IFS=: read -ra parts <<< "${!var-}"
        new=""
        for p in "${parts[@]}"; do
            [[ "$p" == /opt/ros/* ]] && continue
            [[ "$p" == */lib/python*/dist-packages ]] && continue
            new="${new:+$new:}$p"
        done
        export "$var=$new"
    done
}
_urs_clean_paths

# --- 2) middleware + ROS domain ---
# CycloneDDS is far lighter than the default FastDDS on this many-node single-host
# graph: FastDDS busy-spins a whole core (~22 Hz sim), CycloneDDS idles at ~21%
# CPU and hits the full 80 Hz. IMPORTANT: `conda activate` clears
# RMW_IMPLEMENTATION, so it must be (re)set AFTER activation — that is the whole
# reason this lives in a sourced script instead of ~/.bashrc.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

# CycloneDDS config file: cyclonedds.xml targets the CAR network (192.168.60.x,
# explicit peers, no multicast). A dev laptop wants CycloneDDS DEFAULTS (auto
# interface + multicast), so only point CYCLONEDDS_URI at the file when that
# subnet is actually present.
if command -v ip >/dev/null 2>&1 && ip -o addr show 2>/dev/null | grep -q '192\.168\.60\.'; then
    export CYCLONEDDS_URI="file://$_URS_REPO/cyclonedds.xml"
fi

# --- 3) colcon workspace overlay + gym raycaster dir ---
[ -f "$_URS_WS/install/setup.bash" ] && source "$_URS_WS/install/setup.bash"
export RAYCASTER_DIR="$_URS_REPO/race_utils/raycaster"

# --- 4) helpers ---
# Kill every ROS 2 process (nodes, launchers, daemon), any package/language.
ros2kill() {
    ros2 daemon stop 2>/dev/null                     # graceful CLI daemon shutdown
    pkill -9 -f '_ros2_daemon'      2>/dev/null       # the daemon process
    pkill -9 -f -- '--ros-args'     2>/dev/null       # any node started via ros2 run/launch
    pkill -9 -f 'ros2 (run|launch)' 2>/dev/null       # the launcher itself
    pkill -9 -f '/opt/ros/'         2>/dev/null       # rviz2 etc. from a ROS install path
    echo "[ros2kill] killed all ROS 2 nodes"
}

# colcon build (Release) + re-source. No args = whole workspace; args = packages.
cbuild() {
    local sel=()
    [ $# -gt 0 ] && sel=(--packages-select "$@")
    ( cd "$_URS_WS" && colcon build "${sel[@]}" --symlink-install \
          --cmake-args -DCMAKE_BUILD_TYPE=Release ) \
        && source "$_URS_WS/install/setup.bash"
}

echo "[unicorn] env ready  |  RMW=$RMW_IMPLEMENTATION  ROS_DOMAIN_ID=$ROS_DOMAIN_ID  |  helpers: cbuild, ros2kill"
