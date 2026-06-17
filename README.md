# UNICORN Racing Stack — ROS 2 Jazzy (meta-repo)

This is the `jazzy` branch of `unicorn-racing-stack`, restructured as a **lean
orchestration meta-repo**. It carries no packages of its own — it lives inside a
colcon workspace and uses [vcstool](https://github.com/dirk-thomas/vcstool) to
pull the component repos listed in [`unicorn.repos`](unicorn.repos) into the
workspace `src/` as **siblings**. The stack auto-revalidates whenever a
component repo is pushed.

> The previous ROS 1 (catkin) stack lives on the **`main`** branch and in a
> frozen working copy at `unicorn-racing-stack-ros1/` (submodules populated)
> for reference during migration.

## Workspace layout

```
unicorn_racing_stack/                ← colcon workspace root (run `colcon build` here)
├── build/  install/  log/           ← generated
└── src/
    ├── unicorn-racing-stack/        ← THIS meta-repo (jazzy): unicorn.repos, CI, build scripts
    │   ├── unicorn.repos            ← which component repos + branches
    │   ├── unicorn.lock.repos       ← exact commits last built green (from CI)
    │   ├── build_packages_on_local_pc.sh / build_packages_on_car.sh
    │   ├── .github/workflows/integration.yml
    │   └── ci_templates/notify-race-stack.yml
    ├── creating_autonomous_car/     ← component: ROS 2 Jazzy base layer (active)
    └── <future components>/         ← imported here by `vcs import`
```

The build scripts resolve the workspace as `<this repo>/../..` and import
components into that workspace's `src/`.

## Quick start

```bash
# In an existing workspace that already contains src/creating_autonomous_car,
# place this meta-repo next to it and build:
cd <workspace>/src
git clone -b jazzy https://github.com/HMCL-UNIST/unicorn-racing-stack.git
cd unicorn-racing-stack

# Local PC (simulation): installs deps, imports missing components, builds the
# whole workspace. Existing component checkouts are left untouched.
bash build_packages_on_local_pc.sh

# …or on the car (full sensors / SLAM / PF)
bash build_packages_on_car.sh

# …or Docker (run from this meta-repo dir; mounts the workspace root at /ws)
INPUT_GID=$(getent group input | cut -d: -f3) docker compose build dev
docker compose run --rm dev
```

Managing sources by hand (run from the workspace root):

```bash
vcs import src < src/unicorn-racing-stack/unicorn.repos   # clone missing components
vcs pull   src                                            # update tracked repos in place
vcs export --exact src > src/unicorn-racing-stack/unicorn.lock.repos   # snapshot exact commits
```

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
