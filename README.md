# UNICORN Racing Stack — ROS 2 Jazzy

A full **F1TENTH autonomous racing stack** on ROS 2 Jazzy — perception → tracking
→ prediction → planning → state machine → control — with an in-repo
**f1tenth_gym** simulator for software-in-the-loop testing. Self-contained: clone,
build, run.

> The ROS 1 (catkin) stack lives on the **`ros1`** branch (frozen, for reference).

RoboStack (conda) makes the build OS- and arch-agnostic. Verified platforms:

|  | Ubuntu x86_64 | Ubuntu arm64 | macOS arm64 | Windows |
|---|:---:|:---:|:---:|:---:|
| **Status** | ✅ verified | ✅ verified | 🔺 partial | ⬜ untested |
| **Hardware** | NUC, desktop | Jetson (Orin) | Mac mini, MacBook | conda |

**Install support:** the **conda (RoboStack)** path below is the only tested and
supported one. **System ROS 2 Jazzy (apt/rosdep)** and **Docker** are planned —
not yet officially supported.

## Get started

**RoboStack (conda) is the default and verified path** — ROS 2 Jazzy + every
dependency into one conda env (`unicorn`), on **Linux and macOS**, without
touching system ROS. Copy-paste the three blocks below top to bottom.

### 1. conda — skip if you already have it

If `conda` (or `mamba`) is already on your PATH, **skip this step**. Otherwise
install **Miniforge** (recommended — minimal, `conda-forge` by default, no
license terms):

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh" && exec $SHELL
conda config --set auto_activate_base false
```

### 2. clone

```bash
mkdir -p ~/unicorn_ws/src && cd ~/unicorn_ws/src
git clone --recursive https://github.com/hmcl-unist/unicorn-racing-stack.git
cd unicorn-racing-stack
```

<details><summary>cloned without <code>--recursive</code>? / workspace layout</summary>

`race_utils/raycaster` and `race_utils/unicorn_gym` are submodules — populate them:
```bash
git submodule update --init --recursive
```
The colcon **workspace root** is `~/unicorn_ws` (the dir that holds `src/`); this
repo lives at `~/unicorn_ws/src/unicorn-racing-stack`.
</details>

### 3. install

One script does everything — creates the `unicorn` env, registers the `unicorn`
alias in `~/.bashrc` and `~/.zshrc`, installs the pip layer, fixes `quadprog`,
raises OS socket buffers for CycloneDDS, and builds (Release). Runs from bash or zsh:

```bash
./setup_conda_onLaptop.sh   # sim / laptop: skips hardware-only nodes
# ./setup_conda_onCar.sh    # the car: full build (everything)
```

The laptop build skips only the **hardware-only** packages (`urg_node`,
`vesc_driver`/`vesc_ackermann`, `particle_filter`) and then runs the car script. It
**keeps** `vesc_msgs` (so you can `ros2 topic echo` VESC messages) and the full
cartographer stack incl. `cartographer_rviz` (to view submaps in RViz). Running
`setup_conda_onCar.sh` directly clears those ignores and builds everything.

<details><summary>prefer to run the steps yourself? (same thing, in order)</summary>

```bash
conda env create -f environment.yml                                              # conda layer: ROS 2 Jazzy + deps
echo "alias unicorn='source $(pwd)/unicorn.sh'" >> ~/.bashrc                      # alias (add to ~/.zshrc too for zsh)
source unicorn.sh                                                                 # enter the env now
pip install -r requirements.txt                                                   # pip layer
pip install -e ./race_utils/unicorn_gym/f1tenth_gym                               # gym core -> f110_gym
pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper   # range_libc
pip uninstall -y quadprog && conda install -y -c conda-forge quadprog=0.1.13      # quadprog swap (must be LAST)
cbuild                                                                            # colcon build (Release)
```
</details>

## Quick simulation start

After the install script, open a **new shell** (or `source ~/.bashrc` / `~/.zshrc`), then:

```bash
unicorn   # enter the env: conda + PYTHONNOUSERSITE=1 + CycloneDDS + workspace, all sourced
ros2 launch stack_master race.launch.xml sim:=true map:=f   # full autonomy + virtual opponent
#   low_level.launch.xml = vehicle + sensors only
```

`unicorn` also defines helpers: `cbuild [pkgs...]` (colcon build Release + re-source;
no args = whole workspace) and `ros2kill` (kill every ROS 2 node / launcher / daemon).

<details><summary>What <code>unicorn.sh</code> sets — and why you always enter with it</summary>

Always enter with `unicorn` (which **sources** `unicorn.sh`), never a bare
`conda activate`. It works in bash and zsh, and:
- sets **`PYTHONNOUSERSITE=1`** so a stale `~/.local/lib/python*` can't shadow
  the env;
- selects **CycloneDDS** — `RMW_IMPLEMENTATION` must be set *after* `conda
  activate` (which clears it). The default FastDDS busy-spins a core on this
  many-node graph (~22 Hz sim); CycloneDDS idles at ~21% CPU and hits the full
  ~80 Hz. It points `CYCLONEDDS_URI` at the repo's `cyclonedds.xml` (loopback +
  a Wi-Fi interface, multicast on). Adjust `ROS_DOMAIN_ID` (default `1`);
- **resets the ROS env to a clean baseline** so a system ROS / other workspace
  `source`d in your rc file (any distro/path) can't shadow the conda env. Do
  **not** globally `source /opt/ros/<distro>/setup.*` in your rc file;
- sources the colcon workspace and defines `cbuild` / `ros2kill`.

For a fully isolated env immune to any host rc file, use the container
(`.devcontainer` / `.docker`).
</details>

## System ROS 2 (apt / rosdep) — not yet tested

<details><summary>Path B — system ROS 2 Jazzy on Ubuntu 24.04 (unverified)</summary>

> ⚠️ The conda path above is the verified one. This is documented for
> completeness; expect to fix gaps.

```bash
# B1  install ROS 2 Jazzy (https://docs.ros.org/en/jazzy/Installation.html), then:
source /opt/ros/jazzy/setup.bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip

# B2  rosdep (ROS + apt deps from every package.xml)
sudo rosdep init    # first time only; ignore "already exists"
rosdep update
rosdep install --from-paths src/unicorn-racing-stack --ignore-src -r -y

# B3  python layer (same requirements.txt as the conda path)
pip install --user -r src/unicorn-racing-stack/requirements.txt
pip install --user -e src/unicorn-racing-stack/race_utils/unicorn_gym/f1tenth_gym

# B4  build
colcon build --symlink-install --base-paths src/unicorn-racing-stack --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Ubuntu 24.04 usually does **not** need the conda path's `setuptools`/`asio` pins
(the distro ships compatible versions). Source with
`source /opt/ros/jazzy/setup.bash && source install/setup.bash`.
</details>

<details><summary>Manual conda bootstrap (without <code>environment.yml</code> / <code>setup_conda_onCar.sh</code>)</summary>

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
pip install -e race_utils/unicorn_gym/f1tenth_gym
conda install -c conda-forge -y "setuptools<80" "asio=1.29.0"   # see pin notes above
```
</details>

<details><summary>Hardware-driver build notes (real car only — not needed for sim)</summary>

The **simulation stack builds cleanly** via `setup_conda_onLaptop.sh` (it marks
these trees `COLCON_IGNORE` for you). Only the real-car drivers need care:

- **`vesc_driver`** → `transport_drivers` use `asio::io_service` (removed in asio
  ≥1.30). Fixed by the `asio=1.29.0` pin in `environment.yml`.
- **`urg_node`** (Hokuyo LiDAR) → its siblings (`urg_c`, `laser_proc`,
  `urg_node_msgs`) sit nested inside `urg_node/`; colcon won't descend. Move them up:
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
