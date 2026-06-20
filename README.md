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
cd unicorn-racing-stack
conda env create -f environment.yml                       # conda layer (ROS 2 Jazzy)
echo "alias unicorn='source $(pwd)/unicorn.sh'" >> ~/.bashrc && source unicorn.sh   # enter env
pip install -r requirements.txt && pip install -e ./race_utils/unicorn_gym/f1tenth_gym
pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper    # range_libc
pip uninstall -y quadprog && conda install -y -c conda-forge quadprog=0.1.13        # quadprog (last)
cbuild                                                     # build + re-source
```

**Enter the env with `unicorn` in every new shell** — sourcing `unicorn.sh` sets
`PYTHONNOUSERSITE=1` (so `~/.local` can't shadow it), selects CycloneDDS, and sources
the workspace. Details: **[INSTALL.md](INSTALL.md)**.

## Run the simulator

```bash
unicorn   # conda env + CycloneDDS + workspace, all sourced
ros2 launch stack_master headtohead.launch.xml sim:=true map:=f   # full autonomy + virtual opponent
#   low_level.launch.xml = vehicle + sensors only
```

## Verified platforms

RoboStack (conda) makes the build OS- and arch-agnostic. Verified working:

| Platform | Hardware | Status |
|---|---|---|
| Ubuntu **x86_64** | NUC, desktop | ✅ verified |
| Ubuntu **arm64** | Jetson (Orin) | ✅ verified |
| macOS **arm64** | Mac mini, MacBook (Apple silicon) | ✅ verified |
| **Windows** (conda) | — | ⬜ not yet tested — should work via conda |

Per-platform build/smoke detail: **[BUILD.md](BUILD.md)**.
