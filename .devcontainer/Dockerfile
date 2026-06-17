FROM osrf/ros:jazzy-desktop
ARG USERNAME
ARG UID
ARG GID
ARG INPUT_GID

ENV SHELL=/bin/bash

# Create the host user inside the container
RUN userdel ubuntu 2>/dev/null || true
RUN groupadd --gid $GID $USERNAME \
    && useradd --uid $UID --gid $GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && { groupadd --gid $INPUT_GID input 2>/dev/null || true; } \
    && usermod -aG input $USERNAME

# Base build tools + vcstool/colcon + display libs + cyclonedds
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential cmake git python3-pip \
    python3-vcstool python3-colcon-common-extensions \
    libgl1-mesa-dev mesa-utils libglu1-mesa-dev \
    ros-jazzy-rmw-cyclonedds-cpp \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python packages not available via rosdep (f1tenth_gym sim stack).
# NOTE: component sources are NOT copied here — they are fetched by
# `vcs import` at container-create time (see postCreateCommand), so rosdep over
# the sources also runs then, not at image-build time.
ADD .install_utils/pip_req.txt /tmp/.install_utils/pip_req.txt
RUN pip3 install -r /tmp/.install_utils/pip_req.txt --break-system-packages \
    && rm -rf /tmp/.install_utils

USER $USERNAME

# Workspace root is /ws (the meta-repo is bind-mounted there; src/ is filled by vcs import)
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "export ROS_DOMAIN_ID=1" >> /home/$USERNAME/.bashrc \
    && echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/$USERNAME/.bashrc \
    && echo "alias cb='cd /ws && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> /home/$USERNAME/.bashrc \
    && echo "alias sauce='source /ws/install/setup.bash'" >> /home/$USERNAME/.bashrc \
    && echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> /home/$USERNAME/.bashrc

CMD ["/bin/bash"]
