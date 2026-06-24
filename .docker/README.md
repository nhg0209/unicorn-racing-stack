# `.docker/` — container build / smoke / sim (RoboStack conda, ROS 2 Jazzy)

> **Status: experimental / not yet officially supported.** The conda
> ("Get started") path in the top-level README is the only supported install;
> Docker support is planned. The harness here works but is not a stability promise.

Everything here runs the **README "Get started"** flow (Miniforge → `conda env create`
→ range_libc → quadprog → `colcon build`) in a clean container, so a green build
== the stack builds from scratch on that base. Cross-platform (x86_64 + aarch64).

| file | what |
|------|------|
| `Dockerfile` | build-test image (the build *is* the test) |
| `docker-compose.yml` | `buildtest` / `smoke` / `sim` services |
| `docker-compose.dev.yml` | interactive `dev` container (host net + X11 + workspace mount) |
| `smoke_test.sh` | bring up `low_level` + `headtohead` headless, check core topics/nodes |
| `run_platform_test.sh` | build a base + run smoke + append `results.txt` (per-platform matrix driver) |
| `results.txt` | recorded verdicts |

## Quick start (run from the repo root, the dir that holds `.docker/`)

```bash
# build + smoke on the host arch
docker compose -f .docker/docker-compose.yml build buildtest
docker compose -f .docker/docker-compose.yml run --rm smoke

# interactive sim (RViz to the host X server)
xhost +local:
docker compose -f .docker/docker-compose.yml run --rm sim

# another base / platform
BASE=ubuntu:22.04 IMAGE=urs:22.04 docker compose -f .docker/docker-compose.yml build buildtest

# full platform-matrix driver (build + smoke + record)
.docker/run_platform_test.sh ubuntu:24.04 24.04-x86
.docker/run_platform_test.sh ubuntu:22.04 22.04-x86
docker run --privileged --rm tonistiigi/binfmt --install arm64   # one-time, x86 host
.docker/run_platform_test.sh ubuntu:24.04 24.04-arm linux/arm64  # qemu (slow)
```

Verdicts are appended to `results.txt`; per-run logs are `build_<label>.log` /
`smoke_<label>.log`.
