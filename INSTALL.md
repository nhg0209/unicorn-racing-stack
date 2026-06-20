# Installation — unicorn-racing-stack (ROS 2 Jazzy)

**RoboStack (conda) is the default and verified path** — Linux + macOS, no system
ROS. Copy-paste the blocks top to bottom. A system-ROS `apt` route is in
[Path B](#path-b--system-ros-2-jazzy-ubuntu-2404) but is **not yet tested**.

## 0. Clone

```bash
mkdir -p ~/unicorn_ws/src && cd ~/unicorn_ws/src
git clone --recursive https://github.com/hmcl-unist/unicorn-racing-stack.git
```

<details><summary>already cloned without <code>--recursive</code>? / workspace layout</summary>

`race_utils/raycaster` is a submodule — populate it with:
```bash
cd unicorn-racing-stack && git submodule update --init --recursive
```
The colcon **workspace root** is `~/unicorn_ws` (the dir that holds `src/`); this
repo lives at `~/unicorn_ws/src/unicorn-racing-stack`. Build from the workspace root.
</details>

---

# Path A — RoboStack (conda)  ★

```bash
# A0  Miniforge — skip if you already have conda/mamba. Any Linux/macOS, x86_64/arm64.
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh" && exec $SHELL

# A1  env: ROS 2 Jazzy + pins + pip + gym, all from environment.yml
cd ~/unicorn_ws/src/unicorn-racing-stack
conda env create -f environment.yml && conda activate unicorn

# A2  range_libc (raycaster / localization backends)
pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper

# A3  quadprog (required by the state machine — swap the broken PyPI wheel)
pip uninstall -y quadprog && conda install -y -c conda-forge quadprog=0.1.13

# A4  build
cd ~/unicorn_ws
colcon build --symlink-install --base-paths src/unicorn-racing-stack --cmake-args -DCMAKE_BUILD_TYPE=Release
```

<details><summary>Why each step / which pins (read only if something breaks)</summary>

- **A0 — Miniforge, not Anaconda:** minimal conda installer that defaults to the
  `conda-forge` channel (exactly what RoboStack needs), no bundle bloat or
  commercial-license terms. `$(uname)-$(uname -m)` picks the Linux/macOS + arch build.
- **A1 — `conda activate unicorn` is the only "source ROS" you need.** RoboStack's
  activation hooks set up the ROS environment. Do **not** also `source /opt/ros/...`
  — mixing system ROS with RoboStack breaks the build.
- **A2 — range_libc** is a header-only **pybind11** binding; `--no-build-isolation`
  compiles it entirely from the conda toolchain (no PyPI fetch). Only needed for
  `particle_filter` localization and the `rm`/`cddt`/`glt` raycaster backends
  (default `lut` runs without it).
- **A3 — quadprog:** `trajectory_planning_helpers` pins `quadprog==0.1.7`, whose
  PyPI wheel links the wrong `libgfortran` in a fresh conda env and crashes the
  state machine at import (`undefined symbol: ...qpgen2_...`). The conda-forge
  build has correct linkage and is API-compatible.
- **A4 — `--base-paths`** scopes the build to THIS repo, so anything else in `src/`
  is ignored automatically (no `COLCON_IGNORE` needed on siblings).
- **`environment.yml` pins — do not loosen:** `setuptools<80` (colcon
  `--symlink-install` needs `setup.py develop`), `asio=1.29.0` (transport_drivers
  use `asio::io_service`, removed in asio ≥1.30), `numba>=0.65` (accepts numpy 2.x
  so the gym dynamics actually JIT).
</details>

## Enter the environment (alias)

```bash
cd ~/unicorn_ws/src/unicorn-racing-stack
echo "alias unicorn='source $(pwd)/unicorn.sh'" >> ~/.bashrc && exec bash
unicorn            # conda env + CycloneDDS + workspace, all sourced
cbuild [pkgs...]   # colcon build (Release) + re-source; no args = whole workspace
ros2kill           # kill every ROS 2 node / launcher / daemon
```

<details><summary>What <code>unicorn.sh</code> sets (and why CycloneDDS matters)</summary>

It must be **sourced**: `RMW_IMPLEMENTATION` has to be set *after* `conda activate`
(which clears it). The default FastDDS busy-spins a core on this many-node graph
(~22 Hz sim); **CycloneDDS** idles at ~21% CPU and hits the full ~80 Hz. On the car
(network `192.168.60.x`) it also loads the repo's `cyclonedds.xml`; on a laptop it
stays on CycloneDDS defaults. Adjust `ROS_DOMAIN_ID` (default `1`).
</details>

## Run

```bash
unicorn
ros2 launch stack_master headtohead.launch.xml sim:=true map:=f   # full autonomy + virtual opponent
#   low_level.launch.xml = vehicle + sensors only
```

Per-platform build/smoke status: [BUILD.md](BUILD.md).

---

# Path B — System ROS 2 Jazzy (Ubuntu 24.04)

> ⚠️ **Not yet tested** — Path A is the verified one. Documented for completeness;
> expect to fix gaps.

<details><summary>apt / rosdep steps (unverified)</summary>

```bash
# B1  install ROS 2 Jazzy (https://docs.ros.org/en/jazzy/Installation.html), then:
source /opt/ros/jazzy/setup.bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip

# B2  rosdep (ROS + apt deps from every package.xml)
sudo rosdep init    # first time only; ignore "already exists"
rosdep update
rosdep install --from-paths src/unicorn-racing-stack --ignore-src -r -y

# B3  python layer (same requirements.txt as Path A)
pip install --user -r src/unicorn-racing-stack/requirements.txt
pip install --user -e src/unicorn-racing-stack/simulator/f1tenth_gym

# B4  build
colcon build --symlink-install --base-paths src/unicorn-racing-stack --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Ubuntu 24.04 usually does **not** need Path A's `setuptools`/`asio` pins (the distro
ships compatible versions). Source with `source /opt/ros/jazzy/setup.bash && source install/setup.bash`.
</details>

---

<details><summary>Manual conda bootstrap (without <code>environment.yml</code>)</summary>

```bash
conda create -n unicorn -c conda-forge -c robostack-jazzy ros-jazzy-desktop -y
conda activate unicorn
conda config --env --add channels conda-forge
conda config --env --add channels robostack-jazzy
conda config --env --set channel_priority strict

conda install -c conda-forge -c robostack-jazzy -y \
  compilers cmake pkg-config make ninja colcon-common-extensions rosdep
conda install -c conda-forge -c robostack-jazzy -y \
  ros-jazzy-ackermann-msgs ros-jazzy-asio-cmake-module ros-jazzy-diagnostic-updater \
  ros-jazzy-foxglove-bridge ros-jazzy-io-context ros-jazzy-nav2-lifecycle-manager \
  ros-jazzy-nav2-map-server ros-jazzy-nav2-msgs ros-jazzy-robot-localization \
  ros-jazzy-rosbag2-storage-mcap ros-jazzy-serial-driver ros-jazzy-tf-transformations \
  ros-jazzy-xacro ros-jazzy-teleop-twist-keyboard ros-jazzy-cartographer-ros \
  ros-jazzy-joint-state-publisher transforms3d opencv matplotlib-base

pip install -r requirements.txt
pip install -e simulator/f1tenth_gym
conda install -c conda-forge -y "setuptools<80" "asio=1.29.0"   # see pin notes above
```
</details>

<details><summary>Hardware-driver build notes (real car only — not needed for sim)</summary>

The **simulation stack builds cleanly** with Path A. Only the real-car drivers need care:

- **`vesc_driver`** → `transport_drivers` use `asio::io_service` (removed in asio ≥1.30).
  Fixed by the `asio=1.29.0` pin in `environment.yml`.
- **`urg_node`** (Hokuyo LiDAR) → its siblings (`urg_c`, `laser_proc`, `urg_node_msgs`)
  sit nested inside `urg_node/`; colcon won't descend. Move them up:
  ```bash
  cd src/unicorn-racing-stack/sensor/urg_node && mv urg_c laser_proc urg_node_msgs ../
  ```

For **simulation only**, just ignore the hardware trees:
```bash
touch src/unicorn-racing-stack/sensor/urg_node/COLCON_IGNORE
touch src/unicorn-racing-stack/sensor/vesc/COLCON_IGNORE
touch src/unicorn-racing-stack/stack_master/COLCON_IGNORE
```
</details>
