# UNICORN Racing Stack — ROS 2 Jazzy (meta-repo)

The `jazzy` branch of `unicorn-racing-stack`: a full **F1TENTH autonomous racing
stack** on ROS 2 Jazzy — perception → tracking → prediction → planning → state
machine → control — with an in-repo **f1tenth_gym** simulator for full
software-in-the-loop testing.

> The ROS 1 (catkin) stack lives on the **`main`** branch (frozen, for migration reference).

## Install

**RoboStack (conda) is the default and only verified path.** It installs ROS 2
Jazzy + every dependency into a conda env (`unicorn`) on **Linux and macOS**,
without touching system ROS. A system-ROS (`apt` / `rosdep`) path is documented
too but **not yet tested**.

Full step-by-step is in **[INSTALL.md](INSTALL.md)**. Short version:

```bash
mkdir -p ~/unicorn_ws/src && cd ~/unicorn_ws/src
git clone --recursive https://github.com/hmcl-unist/unicorn-racing-stack.git
cd unicorn-racing-stack && conda env create -f environment.yml && conda activate unicorn
#   + range_libc and quadprog steps — see INSTALL.md A1b / A1c
cd ~/unicorn_ws && colcon build --symlink-install --base-paths src/unicorn-racing-stack
```

Set up the `unicorn` alias (INSTALL.md A3) to enter the environment in one word.

## Run the simulator

```bash
unicorn   # conda env + CycloneDDS + workspace, all sourced (see INSTALL.md A3)
ros2 launch stack_master headtohead.launch.xml sim:=true map:=f   # full autonomy + virtual opponent
#   low_level.launch.xml = vehicle + sensors only
```

Per-platform build/smoke status: **[BUILD.md](BUILD.md)**.

## How auto-update works

```
component repo push ──► notify-race-stack.yml ──► repository_dispatch ──►
        unicorn-racing-stack / integration.yml ──► vcs import + colcon build/test
```

- Every component repo you control gets a copy of
  [`ci_templates/notify-race-stack.yml`](ci_templates/notify-race-stack.yml) at
  `.github/workflows/notify-race-stack.yml`. On push it fires a
  `repository_dispatch` (`event-type: submodule_updated`).
- [`integration.yml`](.github/workflows/integration.yml) reacts to that dispatch
  (and to pushes/PRs on `jazzy`): it imports the latest component sources into a
  fresh `src/` and runs `colcon build` + `colcon test`, catching integration
  breakage immediately. The exact commits built are exported to
  `unicorn.lock.repos` (uploaded as an artifact; optionally committed on green).

### One-time secret setup (`RACE_STACK_TOKEN`)

1. Create a **fine-grained PAT** scoped to **only** `HMCL-UNIST/unicorn-racing-stack`
   with **Contents: Read and write**, with an expiry.
2. Add it as an **organisation secret** `RACE_STACK_TOKEN` (HMCL-UNIST → Settings
   → Secrets and variables → Actions), restricted to the component repos. One
   org secret = one token to rotate, instead of duplicating it per repo.
3. `ForzaETH/ccma` is third-party (you cannot add the notify workflow there) —
   it is pinned to a commit in `unicorn.repos`; bump it deliberately, or add an
   `on: schedule` run to catch upstream drift.

## Migration status & roadmap

This branch is the **structure + automation scaffold**. Code porting is staged.

| Component | State | Plan |
|---|---|---|
| `creating_autonomous_car` | **active** (ROS 2 Jazzy) | reused as the base layer; tracked at `jazzy` |
| `cartographer_unicorn` (custom SLAM) | ROS 1 | port → jazzy branch, then enable in `unicorn.repos` |
| `particle_filter_python3` | ROS 1 | port → jazzy branch, then enable |
| `ccma` (ForzaETH util) | pure-Python | pinned commit; pip-installed when its consumer is ported |
| planners / prediction / system_id / state_machine | ROS 1 (on `main`) | port; split into component repos (TBD) and add to `unicorn.repos` |

### Package ownership (avoids colcon name collisions)

CAC and the old unicorn stack share 9 package names (`controller`, `f110_msgs`,
`obstacle_publisher`, `perception`, `stack_master`, `vesc`, `vesc_ackermann`,
`vesc_driver`, `vesc_msgs`). colcon aborts on duplicate names, so **each name is
owned by exactly one component**; the other side must carry a `COLCON_IGNORE`.
When porting a unicorn package that overrides a CAC one, enable it in
`unicorn.repos` and `COLCON_IGNORE` CAC's copy in the same change. Keep this
table updated as the source of truth.
