#!/usr/bin/env bash
# Build the stack in a clean container for one base image, then run the sim
# smoke test, and append the verdict to docker/results.txt.
#
#   docker/run_platform_test.sh ubuntu:24.04 24.04-x86
#   docker/run_platform_test.sh ubuntu:22.04 22.04-x86
#   docker/run_platform_test.sh ubuntu:24.04 24.04-arm linux/arm64   # needs qemu binfmt
set -u
BASE="${1:?usage: BASE LABEL [PLATFORM]}"
LABEL="${2:?usage: BASE LABEL [PLATFORM]}"
PLATFORM="${3:-}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO/.docker/results.txt"
TAG="urs-test:${LABEL}"
BLOG="$REPO/.docker/build_${LABEL}.log"
SLOG="$REPO/.docker/smoke_${LABEL}.log"

plat=()
[ -n "$PLATFORM" ] && plat=(--platform "$PLATFORM")

echo "=== [$LABEL] docker build (BASE=$BASE ${PLATFORM:+PLATFORM=$PLATFORM}) ==="
if docker build "${plat[@]}" -f "$REPO/.docker/Dockerfile" \
        --build-arg BASE="$BASE" -t "$TAG" "$REPO" > "$BLOG" 2>&1; then
  build="BUILD_OK"
else
  build="BUILD_FAIL"
fi
echo "[$LABEL] $build   (full log: $BLOG)"
# how many colcon packages finished (visible even on failure).
# grep -c prints the count and exits 1 when 0 -> capture via a var, no `|| echo`.
fin=$(grep -c 'Finished <<<' "$BLOG"); fin=${fin:-0}
fail=$(grep -c 'Failed   <<<' "$BLOG"); fail=${fail:-0}

smoke="SKIP"
if [ "$build" = "BUILD_OK" ]; then
  echo "=== [$LABEL] smoke test (low_level + headtohead, headless) ==="
  if docker run --rm "${plat[@]}" "$TAG" > "$SLOG" 2>&1; then
    smoke="SMOKE_OK"
  else
    smoke="SMOKE_FAIL"
  fi
  grep -E 'lowlevel:|headtohead:|SMOKE:' "$SLOG" || tail -n 8 "$SLOG"
fi

line="$(date -u +%Y-%m-%dT%H:%MZ)  ${LABEL}  BASE=${BASE}  ${build}(pkgs_ok=${fin},fail=${fail})  ${smoke}"
echo "$line" | tee -a "$RESULTS"
