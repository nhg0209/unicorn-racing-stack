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

`race_utils/raycaster` and `race_utils/unicorn_gym` are submodules — populate them with:
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

# A1  conda env: ROS 2 Jazzy + build tooling + pinned libs (the conda layer)
cd ~/unicorn_ws/src/unicorn-racing-stack
conda env create -f environment.yml

# A2  ENTER the env via unicorn.sh — THIS is how you enter it, now and in every new
#     shell. It activates the env AND sets PYTHONNOUSERSITE=1 (so ~/.local can't
#     shadow it — a real install footgun) + CycloneDDS + workspace + cbuild/ros2kill.
echo "alias unicorn='source $(pwd)/unicorn.sh'" >> ~/.bashrc   # 'unicorn' in every new shell
source unicorn.sh                                              # enter it NOW (this shell)

# A3  python (pip) layer — run UNDER unicorn (so ~/.local is ignored), from the repo
#     root (the -e paths are relative to it)
pip install -r requirements.txt
pip install -e ./race_utils/unicorn_gym/f1tenth_gym                              # gym core -> f110_gym
pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper  # range_libc

# A4  quadprog — swap the broken PyPI wheel. MUST be AFTER A3 (trajectory_planning_
#     helpers re-pulls quadprog==0.1.7, whose wheel crashes the state machine at import).
pip uninstall -y quadprog && conda install -y -c conda-forge quadprog=0.1.13

# A5  build  (cbuild = colcon build Release + re-source, provided by unicorn.sh)
cbuild
```

<details><summary>Why each step / which pins (read only if something breaks)</summary>

- **A0 — Miniforge, not Anaconda:** minimal conda installer that defaults to the
  `conda-forge` channel (exactly what RoboStack needs), no bundle bloat or
  commercial-license terms. `$(uname)-$(uname -m)` picks the Linux/macOS + arch build.
- **A1 — conda layer only.** `environment.yml` installs ONLY conda packages (ROS 2
  Jazzy + toolchain + pinned libs). The pip layer is deliberately split into A3 so it
  runs *after* `unicorn.sh` sets `PYTHONNOUSERSITE=1`. (RoboStack's activation is the
  only "source ROS" you need — do **not** also `source /opt/ros/...`.)
- **A2 — always enter with `unicorn`, not bare `conda activate`.** `unicorn.sh` sets
  `PYTHONNOUSERSITE=1`, which is what stops a stale `~/.local/lib/python*` from
  shadowing the env. If A3 ran without it, pip would see `~/.local` copies as
  "already satisfied", skip installing them into the env, and the nodes would then
  fail to import them at runtime (`No module named gymnasium` / `…helpers`). It also
  selects CycloneDDS and sources the workspace + helpers — see the toggle below.
- **A3 — pip layer under unicorn.** `requirements.txt` (numba/gymnasium/casadi/scipy/
  tph), the editable gym core (`f110_gym`), and `range_libc` (header-only **pybind11**
  binding; `--no-build-isolation` builds it from the conda toolchain, no PyPI fetch —
  only needed for `particle_filter` + the `rm`/`cddt`/`glt` raycaster backends; default
  `lut` runs without it). Run from the repo root so the `-e` paths resolve.
- **A4 — quadprog, LAST.** `trajectory_planning_helpers` pins `quadprog==0.1.7`, whose
  PyPI wheel links the wrong `libgfortran` and crashes the state machine at import
  (`undefined symbol: ...qpgen2_...`). The conda-forge build is correct and
  API-compatible. It must come after A3 because A3 re-pulls the broken wheel.
- **A5 — `cbuild`** runs `colcon build --symlink-install --base-paths
  src/unicorn-racing-stack` (Release) from the workspace root and re-sources.
  `--base-paths` scopes the build to THIS repo (no `COLCON_IGNORE` on siblings).
- **`environment.yml` pins — do not loosen:** `setuptools<80` (colcon
  `--symlink-install` needs `setup.py develop`), `asio=1.29.0` (transport_drivers
  use `asio::io_service`, removed in asio ≥1.30), `numba>=0.65` (in requirements.txt;
  accepts numpy 2.x so the gym dynamics actually JIT).
</details>

## Entering the env (every new shell)

The `unicorn` alias was set in **A2**. From any new terminal, one word enters everything:

```bash
unicorn            # conda env + PYTHONNOUSERSITE=1 + CycloneDDS + workspace, all sourced
cbuild [pkgs...]   # colcon build (Release) + re-source; no args = whole workspace
ros2kill           # kill every ROS 2 node / launcher / daemon
```

<details><summary>What <code>unicorn.sh</code> sets — and why you always enter with it</summary>

Always enter with `unicorn` (which **sources** `unicorn.sh`), never a bare
`conda activate`. It:
- sets **`PYTHONNOUSERSITE=1`** so a stale `~/.local/lib/python*` can't shadow the
  env — at install time (A3) *and* at run time;
- selects **CycloneDDS** — `RMW_IMPLEMENTATION` must be set *after* `conda activate`
  (which clears it). The default FastDDS busy-spins a core on this many-node graph
  (~22 Hz sim); CycloneDDS idles at ~21% CPU and hits the full ~80 Hz. On the car
  (`192.168.60.x`) it loads the repo's `cyclonedds.xml`; on a laptop it stays on
  CycloneDDS defaults. Adjust `ROS_DOMAIN_ID` (default `1`);
- **resets the ROS env to a clean baseline** so a system ROS / other workspace
  `source`d in `~/.bashrc` (any distro/path) can't shadow the conda env (e.g.
  `rosidl_generator_c: generate_c() takes 1 positional argument but 2 were given`).
  Do **not** globally `source /opt/ros/<distro>/setup.bash` in `~/.bashrc`;
- sources the colcon workspace and defines `cbuild` / `ros2kill`.

For a fully isolated env immune to any host `~/.bashrc`, use the container
(`.devcontainer` / `.docker`).
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
pip install --user -e src/unicorn-racing-stack/race_utils/unicorn_gym/f1tenth_gym

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
pip install -e race_utils/unicorn_gym/f1tenth_gym
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
