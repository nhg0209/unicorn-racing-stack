# UNICORN Racing Stack — ROS 2 Jazzy

A full **F1TENTH autonomous racing stack** on ROS 2 Jazzy — perception → tracking
→ prediction → planning → state machine → control — with an in-repo
**f1tenth_gym** simulator for software-in-the-loop testing. Self-contained: clone,
build, run.

> The ROS 1 (catkin) stack lives on the **`main`** branch (frozen, for reference).

## Install

**RoboStack (conda) is the default and only verified path** — ROS 2 Jazzy + every
dependency into a conda env (`unicorn`), on **Linux and macOS**, without touching
system ROS. A system-ROS (`apt`/`rosdep`) path exists in
[INSTALL.md](INSTALL.md#path-b) but is **not yet tested**.

Full step-by-step: **[INSTALL.md](INSTALL.md)**. TL;DR:

```bash
mkdir -p ~/unicorn_ws/src && cd ~/unicorn_ws/src
git clone --recursive https://github.com/hmcl-unist/unicorn-racing-stack.git
cd unicorn-racing-stack && conda env create -f environment.yml && conda activate unicorn
pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper   # range_libc
pip uninstall -y quadprog && conda install -y -c conda-forge quadprog=0.1.13       # quadprog
cd ~/unicorn_ws && colcon build --symlink-install --base-paths src/unicorn-racing-stack
```

Add the `unicorn` alias ([INSTALL.md A3](INSTALL.md)) to enter the env in one word.

## Run the simulator

```bash
unicorn   # conda env + CycloneDDS + workspace, all sourced
ros2 launch stack_master headtohead.launch.xml sim:=true map:=f   # full autonomy + virtual opponent
#   low_level.launch.xml = vehicle + sensors only
```

Per-platform build/smoke status: **[BUILD.md](BUILD.md)**.
