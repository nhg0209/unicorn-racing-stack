#!/usr/bin/env bash
# First-start setup for the dev container: build the RoboStack `unicorn` env and
# the colcon workspace against the MOUNTED source (INSTALL.md Path A). Runs once;
# the env + install/ persist in the container for subsequent opens.
set -e
REPO=/ws/src/unicorn-racing-stack
source /opt/conda/etc/profile.d/conda.sh

if ! conda env list | grep -q '/envs/unicorn'; then
  echo "[postCreate] creating conda env 'unicorn' (ROS 2 Jazzy)…"
  conda env create -f "$REPO/environment.yml"
fi
conda activate unicorn

# range_libc (header-only pybind11, no PyPI fetch) — INSTALL.md A1b
pip install --no-build-isolation -e "$REPO/race_utils/raycaster/range_libc/pywrapper" || true

# quadprog: replace the broken PyPI wheel with conda-forge — INSTALL.md A1c
python -c 'import quadprog' 2>/dev/null || \
  { pip uninstall -y quadprog; conda install -y -c conda-forge quadprog=0.1.13; }

# build (ROS_VERSION/ROS_DISTRO defensively set in case activate.d didn't run)
cd /ws
export ROS_VERSION="${ROS_VERSION:-2}" ROS_DISTRO="${ROS_DISTRO:-jazzy}"
colcon build --symlink-install --base-paths src/unicorn-racing-stack \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "[postCreate] done. New shells auto-activate 'unicorn' + source install/. Try:  cb  (build) / sauce (re-source)"
