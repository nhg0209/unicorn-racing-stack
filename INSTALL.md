# Installation Guide — unicorn-racing-stack (ROS 2 Jazzy)

This guide takes you from a clean machine to a working `colcon` build, step by step.
Everything is **copy-paste ready**. Pick **one** of the two environments:

- **Path A — RoboStack (conda):** works on **Linux and macOS**, does not touch system ROS.
  This is the verified path used to build this repo.
- **Path B — System ROS 2 Jazzy (Ubuntu 24.04):** the classic `apt` + `rosdep` path.

The **build command is identical** in both environments
(`colcon build --symlink-install`). Only the one-time dependency bootstrap differs
(conda packages vs. apt/rosdep), because the two ecosystems ship dependencies differently.

---

## 0. Clone the repository (REQUIRED: submodules)

`race_utils/raycaster` is a git **submodule**, so a plain clone will leave it empty.
Clone it **into the `src/` of a colcon workspace**:

```bash
mkdir -p ~/unicorn_ws/src && cd ~/unicorn_ws/src
git clone --recursive https://github.com/hmcl-unist/unicorn-racing-stack.git
# if you already cloned without --recursive:
#   cd unicorn-racing-stack && git submodule update --init --recursive
```

> Throughout this guide the **colcon workspace root** is `~/unicorn_ws` (the
> directory that contains `src/`), and this repo lives at
> `~/unicorn_ws/src/unicorn-racing-stack`. Build commands run from the workspace
> root; `unicorn.sh` resolves these paths automatically.

---

# Path A — RoboStack (conda)  ★ recommended

**Prerequisite — conda/mamba installed.** Everything else (ROS, build tools, the
`asio`/`setuptools` pins, the pip layer, and the editable gym core) is captured in
`environment.yml`, so the whole setup is **one env command + one build**.

### A0. Install Miniforge (skip if you already have conda/mamba)

Use **Miniforge**, not Anaconda — the minimal conda installer (~100 MB; conda +
mamba) that **defaults to the `conda-forge` channel** (exactly what RoboStack
needs), with no bundle bloat or commercial-license terms. RoboStack recommends it.

The same command works on **any Linux (Ubuntu) or macOS**, x86_64 or arm64 —
`$(uname)-$(uname -m)` selects the right installer:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh"     # follow prompts; let it run `conda init`
exec $SHELL                                    # reopen the shell so `conda` is on PATH
```

### A1. Create the environment

```bash
# from the repo root (src/unicorn-racing-stack)
conda env create -f environment.yml   # env `unicorn`: ROS + pins + pip + gym, all at once
conda activate unicorn
```

> `conda activate unicorn` is all you need to "source" ROS — RoboStack's activation
> hooks set up the ROS environment automatically. Do **not** also `source /opt/ros/...`;
> mixing system ROS with RoboStack breaks the build.

#### A1b. range_libc (optional — localization / non-`lut` raycaster)

`range_libc` is a tiny **pybind11** binding (header-only, no Cython, no numpy at
build). It's kept as a one-line step and built with `--no-build-isolation` so it
compiles entirely from the conda toolchain (`pybind11` is in `environment.yml`) —
no PyPI fetch, robust on Linux/macOS/aarch64. Only needed for `particle_filter`
localization and the raycaster `rm`/`cddt`/`glt` backends (default `lut` runs
without it):

```bash
# after `conda activate unicorn`, from the repo root
pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper
```

#### A1c. quadprog (REQUIRED for the state machine)

`trajectory_planning_helpers` (a pip dep) pins `quadprog==0.1.7`, whose PyPI wheel
links the wrong `libgfortran` in a fresh conda env and fails at import with
`undefined symbol: ...qpgen2_...`, crashing the state machine. Replace it with the
conda-forge build (correct linkage, API-compatible at runtime):

```bash
# after `conda activate unicorn`
pip uninstall -y quadprog && conda install -y -c conda-forge quadprog=0.1.13
```

### A2. Build

```bash
# from the colcon workspace root (the dir containing src/).
# --base-paths scopes the build to THIS repo only, so anything else that happens
# to sit in src/ (other projects, scratch clones) is ignored automatically — no
# COLCON_IGNORE needed on those siblings.
colcon build --symlink-install --base-paths src/unicorn-racing-stack \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### A3. One-command environment (`unicorn.sh` + alias)

`unicorn.sh` (repo root) enters the whole dev environment in one step: it
activates the `unicorn` conda env, **selects CycloneDDS**, sets the ROS domain,
sources the colcon workspace, and adds `cbuild` / `ros2kill` helpers. It must be
**sourced** — `RMW_IMPLEMENTATION` has to be set *after* `conda activate` (which
clears it), and the default FastDDS busy-spins a core on this many-node graph
(~22 Hz sim) while CycloneDDS idles at ~21% CPU and hits the full ~80 Hz.

Add an alias once (run from the repo root so the path resolves), then a single
word enters the environment:

```bash
# from the repo root (src/unicorn-racing-stack)
echo "alias unicorn='source $(pwd)/unicorn.sh'" >> ~/.bashrc
exec bash                # reload the shell

unicorn                  # env active + workspace sourced + CycloneDDS selected
cbuild [pkgs...]         # colcon build (Release) + re-source; no args = whole ws
ros2kill                 # kill every ROS 2 node / launcher / daemon
```

> On the **car** (network `192.168.60.x`), `unicorn.sh` also points
> `CYCLONEDDS_URI` at the repo's `cyclonedds.xml`; on a laptop it stays on
> CycloneDDS defaults. Adjust `ROS_DOMAIN_ID` (default `1`) as needed.

<details>
<summary><b>Manual bootstrap</b> — only if you can't / don't want to use <code>environment.yml</code> (installs the latest conda packages, step by step)</summary>

**Install conda (Miniforge)** — see **A0** above (skip if you already have conda/mamba).

**Create the env + pin channels:**

```bash
conda create -n unicorn -c conda-forge -c robostack-jazzy ros-jazzy-desktop -y
conda activate unicorn
conda config --env --add channels conda-forge
conda config --env --add channels robostack-jazzy
conda config --env --set channel_priority strict
```

**Build tools + extra ROS deps** (`ros-jazzy-desktop` alone is not enough):

```bash
conda install -c conda-forge -c robostack-jazzy -y \
  compilers cmake pkg-config make ninja colcon-common-extensions rosdep
conda install -c conda-forge -c robostack-jazzy -y \
  ros-jazzy-ackermann-msgs ros-jazzy-asio-cmake-module ros-jazzy-diagnostic-updater \
  ros-jazzy-foxglove-bridge ros-jazzy-io-context ros-jazzy-nav2-lifecycle-manager \
  ros-jazzy-nav2-map-server ros-jazzy-nav2-msgs ros-jazzy-robot-localization \
  ros-jazzy-rosbag2-storage-mcap ros-jazzy-serial-driver ros-jazzy-tf-transformations \
  ros-jazzy-xacro ros-jazzy-teleop-twist-keyboard ros-jazzy-cartographer-ros \
  ros-jazzy-joint-state-publisher \
  transforms3d opencv matplotlib-base
```

**Python (pip) layer + editable gym core:**

```bash
# from the repo root (src/unicorn-racing-stack)
pip install -r requirements.txt
pip install -e simulator/f1tenth_gym
```

**RoboStack compatibility pins** (see *Why these pins* at the end — DO NOT loosen):

```bash
conda install -c conda-forge -y "setuptools<80"   # colcon --symlink-install needs setup.py develop
conda install -c conda-forge -y "asio=1.29.0"     # transport_drivers use asio::io_service (gone in 1.30+)
```

</details>

Jump to **[Sourcing & running](#sourcing--running)**.

---

# Path B — System ROS 2 Jazzy (Ubuntu 24.04)

> ⚠️ **Not yet tested.** Path A (RoboStack) is the verified path. This `apt` /
> `rosdep` route is documented for completeness but has not been validated end to
> end — expect to fix gaps.

### B1. Install ROS 2 Jazzy

Follow the official docs: <https://docs.ros.org/en/jazzy/Installation.html>
(`ros-jazzy-desktop`). Then:

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip
```

### B2. Resolve dependencies with rosdep

```bash
sudo rosdep init   # first time only; ignore "already exists"
rosdep update

# from the workspace root (dir containing src/); scope to THIS repo only so
# anything else in src/ is ignored
rosdep install --from-paths src/unicorn-racing-stack --ignore-src -r -y
```

`rosdep` installs the ROS + system `apt` dependencies declared in every `package.xml`.

### B3. Python dependencies

```bash
# from the workspace root; same pip layer as Path A (requirements.txt is the
# single source of truth — sklearn/skimage/shapely are intentionally commented out there)
pip install --user -r src/unicorn-racing-stack/requirements.txt
pip install --user -e src/unicorn-racing-stack/simulator/f1tenth_gym
```

> On Ubuntu 24.04 you typically do **not** need the `setuptools`/`asio` pins from Path A —
> the distro ships `setuptools < 80` and `libasio < 1.30`, which are already compatible.

### B4. Build

```bash
colcon build --symlink-install --base-paths src/unicorn-racing-stack \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

---

## Sourcing & running

**RoboStack (Path A):**
```bash
conda activate unicorn                 # sets up ROS automatically
source install/setup.bash              # overlay this workspace (after a successful build)
```

**System ROS (Path B):**
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

Optional middleware settings (either path):
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=1
```

---

## Known issues / hardware drivers

The **simulation stack builds cleanly** with the steps above. The only packages that need
extra care are the **real-car hardware drivers** — not needed for simulation:

- **`vesc_driver` (VESC motor controller).** Its dependency `io_context`/`serial_driver`
  (the `transport_drivers` set) uses `asio::io_service`, which was **removed in asio ≥ 1.30**.
  RoboStack ships asio 1.36, so it fails to compile. Fixed by the **`asio=1.29.0` pin**
  (already in `environment.yml`). (On system ROS the distro asio is older, so it just works.)

- **`urg_node` (Hokuyo LiDAR driver).** Its sibling packages (`urg_c`, `laser_proc`,
  `urg_node_msgs`) are laid out **nested inside** `urg_node/`. colcon stops at the first
  `package.xml` it finds and does **not** descend further, so those three are never
  discovered and `urg_node` can't find them. If you need the LiDAR driver, move them up so
  they are siblings:
  ```bash
  cd src/unicorn-racing-stack/sensor/urg_node
  mv urg_c laser_proc urg_node_msgs ../
  ```

`stack_master` (the real-car bringup) depends on both drivers, so those two fixes are only
required if you build for the physical car. For **simulation only**, you may instead ignore
the hardware trees:
```bash
touch src/unicorn-racing-stack/sensor/urg_node/COLCON_IGNORE
touch src/unicorn-racing-stack/sensor/vesc/COLCON_IGNORE
touch src/unicorn-racing-stack/stack_master/COLCON_IGNORE
```

---

## Why these pins (background)

- **`setuptools < 80`** — colcon's `--symlink-install` installs Python/ament_python packages
  via `setup.py develop --editable`. setuptools 80 removed the `develop`/`install` commands,
  so every Python package fails with `error: option --editable not recognized`. Any
  setuptools 79.x or older works.

- **`asio = 1.29.0`** — see the `vesc_driver` note above. `asio::io_service` was renamed to
  `asio::io_context` in asio 1.30. The `transport_drivers` headers still use the old name,
  so they need asio ≤ 1.29 **or** a source patch
  (`asio::io_service` → `asio::io_context`,
  `asio::io_service::work` → `asio::executor_work_guard<asio::io_context::executor_type>`).
  A source patch is the only fully portable fix — it builds on any asio version, RoboStack or
  apt — but requires vendoring `transport_drivers` into the workspace.

These are **environment** mismatches (RoboStack ships newer libraries than the upstream ROS
packages were written for), **not** platform-specific (macOS/Linux) bugs.
