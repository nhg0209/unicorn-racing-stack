# Build-test image for unicorn-racing-stack (RoboStack / conda, ROS 2 Jazzy).
# The image build IS the build test: miniforge -> conda env (environment.yml)
# -> range_libc -> colcon build. If it builds, the stack builds on this base.
#
#   docker build -f docker/Dockerfile --build-arg BASE=ubuntu:24.04 -t urs:24.04 .
#   docker run --rm urs:24.04 bash /ws/src/unicorn-racing-stack/docker/smoke_test.sh
#
# Works for x86_64 and aarch64 (miniforge URL is arch-aware); on an x86 host the
# aarch64 base needs `docker run --platform linux/arm64` + qemu binfmt.
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git bzip2 ca-certificates procps \
        libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 xvfb \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (conda/mamba), arch-aware
RUN arch=$(uname -m) && \
    wget -qO /tmp/mf.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh" && \
    bash /tmp/mf.sh -b -p /opt/conda && rm -f /tmp/mf.sh
ENV PATH=/opt/conda/bin:$PATH

# Use bash for the conda steps: RoboStack's activate.d hooks (which export
# ROS_VERSION/ROS_DISTRO/AMENT_PREFIX_PATH) are bash scripts and do not run under
# the default /bin/sh, which left ROS_VERSION empty and broke transport_drivers.
SHELL ["/bin/bash", "-c"]

# Repo source (context = repo root; see .dockerignore)
COPY . /ws/src/unicorn-racing-stack

# 1) conda env `unicorn` (ROS 2 Jazzy + pins + pip deps + editable in-repo gym)
WORKDIR /ws/src/unicorn-racing-stack
RUN conda env create -f environment.yml

# 2) range_libc (header-only pybind11, from the conda toolchain, no PyPI fetch)
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate unicorn && \
    pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper

# 2b) quadprog: trajectory_planning_helpers pins quadprog==0.1.7, whose PyPI wheel
#     links the wrong libgfortran in a fresh env (import fails with
#     `undefined symbol: ...qpgen2_...` and crashes the state machine). Replace it
#     with the conda-forge build (correct linkage; API-compatible at runtime).
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate unicorn && \
    pip uninstall -y quadprog && conda install -y -c conda-forge quadprog=0.1.13

# 3) colcon build (the actual build test). Fails the image build on error so the
#    docker build exit code reflects build success/failure; the log is tee'd.
WORKDIR /ws
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate unicorn && \
    export ROS_VERSION="${ROS_VERSION:-2}" ROS_DISTRO="${ROS_DISTRO:-jazzy}" && \
    set -o pipefail && \
    colcon build --symlink-install --base-paths src/unicorn-racing-stack \
        --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tee /ws/colcon_build.log

CMD ["bash", "/ws/src/unicorn-racing-stack/docker/smoke_test.sh"]
