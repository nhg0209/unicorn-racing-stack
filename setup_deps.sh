#!/usr/bin/env bash
#
# setup_deps.sh — install every dependency for unicorn-racing-stack (+ pitwall)
#                 into the ACTIVE RoboStack (conda) environment, in one shot.
#
#   conda activate ros_env
#   ./setup_deps.sh
#
# Then:
#   colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
#
# How it works: it reads the dependency keys straight from every package.xml
# (the same data `rosdep` reads), maps each ROS key to its RoboStack conda
# package `ros-<distro>-<name>`, keeps the ones that actually exist in the
# RoboStack channel, and `conda install`s them. System/Python deps that have no
# RoboStack package are installed explicitly via conda-forge / pip.
#
# Note: `rosdep install` itself is avoided on RoboStack — it resolves keys to
# system apt/brew packages, which would pollute the host and clash with the
# conda ROS. Parsing package.xml + conda is the reliable conda-native path.

set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"

# --- 0. sanity ---------------------------------------------------------------
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: activate your RoboStack env first:  conda activate ros_env" >&2
  exit 1
fi
command -v conda >/dev/null || { echo "ERROR: conda not found on PATH" >&2; exit 1; }

# --- paths -------------------------------------------------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/unicorn-racing-stack
SRC="$(cd "$REPO/.." && pwd)"                          # .../src  (colcon src dir)
TARGETS=("$REPO" "$SRC/pitwall")                       # trees we actually build

echo "==> src dir       : $SRC"
echo "==> build targets : unicorn-racing-stack, pitwall"
echo "==> conda env     : $CONDA_PREFIX"

# --- 1. mark the non-target stacks so colcon skips them ----------------------
for d in unicorn-racing-stack-ros1 race_stack creating_autonomous_car raycast_test; do
  [ -d "$SRC/$d" ] && touch "$SRC/$d/COLCON_IGNORE"
done

# --- 2. collect external dependency keys from package.xml --------------------
internal=$(grep -rhoP '(?<=<name>)[^<]+' --include=package.xml "${TARGETS[@]}" \
             | tr -d ' ' | sort -u)
alldeps=$(grep -rhoP '(?<=<(depend|build_depend|exec_depend|build_export_depend|buildtool_depend|test_depend)>)[^<]+' \
             --include=package.xml "${TARGETS[@]}" | sed 's/^ *//;s/ *$//' | sort -u)
external=$(comm -23 <(printf '%s\n' "$alldeps") <(printf '%s\n' "$internal"))

# --- 3. cache the list of available RoboStack packages (once) ----------------
AVAIL="${TMPDIR:-/tmp}/robostack_${ROS_DISTRO}_pkgs.txt"
if [ ! -s "$AVAIL" ]; then
  echo "==> indexing RoboStack channel (one-time, ~1 min)…"
  conda search -c "robostack-${ROS_DISTRO}" "ros-${ROS_DISTRO}-*" 2>/dev/null \
    | awk -v p="ros-${ROS_DISTRO}-" 'NR>2 && index($1,p)==1 {print $1}' \
    | sort -u > "$AVAIL"
fi

# --- 4. classify keys: RoboStack ROS package vs. everything else -------------
ros_pkgs=(); leftover=()
while IFS= read -r k; do
  [ -z "$k" ] && continue
  cand="ros-${ROS_DISTRO}-${k//_/-}"
  if grep -qxF "$cand" "$AVAIL"; then ros_pkgs+=("$cand"); else leftover+=("$k"); fi
done <<< "$external"

# --- 5. install the RoboStack ROS packages -----------------------------------
if [ "${#ros_pkgs[@]}" -gt 0 ]; then
  echo "==> conda install ${#ros_pkgs[@]} ROS packages from RoboStack…"
  conda install -c conda-forge -c "robostack-${ROS_DISTRO}" -y "${ros_pkgs[@]}"
fi

# --- 6. system libraries + compatibility pins (stable, conda-forge) ----------
echo "==> conda install system libs + compatibility pins…"
conda install -c conda-forge -y \
  opencv transforms3d eigen yaml-cpp libboost-devel \
  "setuptools<80" \
  "asio=1.29.0"
#   setuptools<80 : colcon --symlink-install needs setup.py develop --editable (removed in 80)
#   asio=1.29.0   : transport_drivers use asio::io_service (removed in asio 1.30)

# --- 7. pure-Python deps (pip) ----------------------------------------------
echo "==> pip install Python deps…"
python -m pip install --upgrade \
  numba gymnasium casadi scikit-learn scikit-image shapely tqdm trajectory_planning_helpers

# the simulator gym core is a pip package (COLCON_IGNORE'd), install editable
python -m pip install -e "$REPO/race_utils/unicorn_gym/f1tenth_gym"

# --- 8. summary --------------------------------------------------------------
echo
echo "Installed ${#ros_pkgs[@]} RoboStack ROS packages + system/Python deps."
if [ "${#leftover[@]}" -gt 0 ]; then
  echo "Keys with no RoboStack package (covered by the conda/pip block above,"
  echo "or hardware-only / not-in-target — review only if a build fails):"
  printf '   %s\n' "${leftover[@]}"
fi
echo
echo "Now build:"
echo "   colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"
