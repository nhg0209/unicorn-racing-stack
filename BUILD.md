# Build / Platform Test Matrix — unicorn-racing-stack

Tracks where the **INSTALL.md Path A (RoboStack / conda)** flow has been
verified, per target platform. The test is intentionally the *documented simple
setup* from a clean OS:

1. install **Miniforge** (conda)
2. `conda env create -f environment.yml`  → env `unicorn`  (ROS 2 Jazzy + deps + in-repo gym)
3. `pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper`
4. `colcon build --symlink-install --base-paths src/unicorn-racing-stack --cmake-args -DCMAKE_BUILD_TYPE=Release`
5. **smoke**: `ros2 launch stack_master low_level.launch.xml map:=f sim:=true`
   and `headtohead.launch.xml map:=f sim:=true` come up headless (core topics/nodes alive)

Steps 1–5 are exactly what `.docker/Dockerfile` + `.docker/smoke_test.sh` run, so a
green Docker row == the INSTALL.md flow works from scratch on that base. The same
flow backs the `.devcontainer/` (dev) — see `.docker/README.md`.

## How to run a platform test

```bash
# x86 (native on an x86 host):
.docker/run_platform_test.sh ubuntu:24.04 24.04-x86
.docker/run_platform_test.sh ubuntu:22.04 22.04-x86

# arm64 (needs qemu binfmt on an x86 host, or run on real arm hardware):
docker run --privileged --rm tonistiigi/binfmt --install arm64   # one-time, x86 host
.docker/run_platform_test.sh ubuntu:24.04 24.04-arm linux/arm64

# or via docker compose (see .docker/README.md):
docker compose -f .docker/docker-compose.yml build buildtest
docker compose -f .docker/docker-compose.yml run --rm smoke
```
Results append to `.docker/results.txt`; per-run logs are `.docker/build_<label>.log`
and `.docker/smoke_<label>.log`.

## Target platforms

| Platform              | Arch | How tested            | Build | Smoke (low_level + h2h) | Notes |
|-----------------------|------|-----------------------|-------|-------------------------|-------|
| Ubuntu 24.04 (host)   | x86  | native (ros_env)      | ✅    | ✅ 27-lap run           | dev machine; SIM_VERIFICATION_REPORT.md |
| Ubuntu 24.04          | x86  | docker `ubuntu:24.04` | ✅ 42 pkgs | ✅ low_level + h2h     | NUC, 24.04 x86 laptop (2026-06-20) |
| Ubuntu 22.04          | x86  | docker `ubuntu:22.04` | ✅ 42 pkgs | ✅ low_level + h2h     | 22.04 x86 laptop (2026-06-20) |
| Ubuntu 24.04 (Orin)   | arm  | hardware / qemu       | ⬜    | ⬜                      | run on the Orin; qemu cross-build also works (slow) |
| Ubuntu 24.04          | arm  | docker `--platform linux/arm64` (qemu) | ✅ 42 pkgs | ⚠️ n/a  | cross-built on x86 host via qemu binfmt (2026-06-20). Smoke inconclusive under emulation — sim too slow (`scan_hz=0`); verify on real arm hw |
| Ubuntu 22.04          | arm  | hardware / qemu       | ⬜    | ⬜                      | arm laptop |
| macOS (MacBook M4)    | arm  | native (INSTALL.md A) | ⬜    | ⬜                      | **owner fills in** |
| macOS (Mac mini M4)   | arm  | native (INSTALL.md A) | ⬜    | ⬜                      | **owner fills in** |

Legend: ✅ pass · ❌ fail · ⚠️ inconclusive · ⏳ in progress · ⬜ not yet run.

> **arm64 cross-build (qemu):** the full RoboStack env + `colcon build` (all 42
> packages, 0 failures) completes under `--platform linux/arm64` on an x86 host —
> so the stack *builds* for aarch64. The sim **smoke** is not meaningful there: the
> emulated f1tenth_gym is ~10–20× too slow, so `scan_hz`/`odom_hz` read 0 within
> the sampling window and dependent nodes time out. Run the smoke on real arm
> hardware (Orin / Apple-silicon Linux) for a true verdict.

### Fixes that the platform test surfaced (now in the repo)
- `ROS_VERSION` empty under `/bin/sh` (RoboStack activate.d are bash) → broke
  `transport_drivers/udp_msgs` (`if(${ROS_VERSION} EQUAL 2)`). Docker uses `SHELL bash`.
- `trajectory_planning_helpers` pins `quadprog==0.1.7`, whose PyPI wheel links the
  wrong `libgfortran` → state machine crash. Replaced with conda-forge quadprog
  (INSTALL.md **A1c**, baked into the Dockerfile).
- gym raycaster path (`numba_free._find_raycaster_dir`) now ascends to find
  `race_utils/raycaster` (worked under build/ layout too).

### Notes on coverage
- **x86 24.04 / 22.04**: fully testable now in Docker on this host.
- **arm Linux (Orin / arm laptop)**: cross-buildable on an x86 host via qemu
  binfmt (`tonistiigi/binfmt --install arm64` then
  `.docker/run_platform_test.sh ubuntu:24.04 24.04-arm linux/arm64`) — it works but
  the emulated colcon build is *slow* (≈10–20× native). For routine use run the
  same command on real arm hardware (Orin / Apple silicon via Linux VM) or follow
  INSTALL.md Path A on the device. `environment.yml` resolves for aarch64 (vendored
  `sensor/transport_drivers`, conda `asio` pin), so it is expected to build there.
- **macOS**: conda/RoboStack path is supported by INSTALL.md; owner verifies on
  the M4 machines and fills the two rows.

## Results log
See `.docker/results.txt` (machine-appended, one line per run).
