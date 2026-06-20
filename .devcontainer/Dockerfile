# Dev container — RoboStack ROS 2 Jazzy via conda (INSTALL.md Path A).
# Toolchain only: the `unicorn` env + `colcon build` are created on first start
# by postCreate.sh against the MOUNTED workspace, so the editable gym and the
# build see your live source. Matches .docker/ (the validated build), so dev ==
# CI and it works on x86_64 + aarch64 (Ubuntu 22.04/24.04, Orin, Apple silicon).
#
# (The previous apt/system-ROS `osrf/ros:jazzy-desktop` version is in git history.)
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
ARG INPUT_GID=104
ENV SHELL=/bin/bash DEBIAN_FRONTEND=noninteractive

# Host-matched user so files created in the container are owned by you on the host.
RUN userdel -r ubuntu 2>/dev/null || true; \
    groupadd --gid ${GID} ${USERNAME} 2>/dev/null || true; \
    useradd --uid ${UID} --gid ${GID} -m ${USERNAME} 2>/dev/null || true; \
    apt-get update && apt-get install -y --no-install-recommends \
        sudo wget git bzip2 ca-certificates procps less nano \
        libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 xvfb && \
    rm -rf /var/lib/apt/lists/*; \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && chmod 0440 /etc/sudoers.d/${USERNAME}; \
    groupadd --gid ${INPUT_GID} input 2>/dev/null || true; usermod -aG input ${USERNAME} 2>/dev/null || true

# Miniforge into /opt/conda, owned by the dev user so it can create the env.
RUN arch=$(uname -m) && \
    wget -qO /tmp/mf.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh" && \
    bash /tmp/mf.sh -b -p /opt/conda && rm -f /tmp/mf.sh && \
    chown -R ${UID}:${GID} /opt/conda
ENV PATH=/opt/conda/bin:$PATH

USER ${USERNAME}
# Auto-activate the env (created at postCreate) in interactive shells.
RUN /opt/conda/bin/conda init bash && \
    { echo 'conda activate unicorn 2>/dev/null'; \
      echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash 2>/dev/null'; \
      echo 'export RAYCASTER_DIR=/ws/src/unicorn-racing-stack/race_utils/raycaster'; \
      echo "alias cb='cd /ws && colcon build --symlink-install --base-paths src/unicorn-racing-stack --cmake-args -DCMAKE_BUILD_TYPE=Release'"; \
      echo "alias sauce='source /ws/install/setup.bash'"; \
    } >> /home/${USERNAME}/.bashrc

CMD ["/bin/bash"]
