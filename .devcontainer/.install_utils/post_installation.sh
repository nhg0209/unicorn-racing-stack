#! /bin/bash

# extend .bashrc
cat /home/${USER}/catkin_ws/src/race_stack/.devcontainer/.install_utils/bashrc_ext >> ~/.bashrc

# source
source /opt/ros/noetic/setup.bash && source /home/${USER}/catkin_ws/devel/setup.bash

# install dependencies
pip install ~/catkin_ws/src/race_stack/f110_utils/libs/ccma
# pip install ~/catkin_ws/src/race_stack/planner/graph_based_planner/src/GraphBasedPlanner

# build 
cd /home/${USER}/catkin_ws
catkin build

# source for additional build
source /opt/ros/noetic/setup.bash && source /home/${USER}/catkin_ws/devel/setup.bash

# additional build 
cd /home/${USER}/catkin_ws
catkin build

# f1tenth_simulator build
cd /home/${USER}/catkin_ws
catkin build f1tenth_simulator

# particle_filter build
cd /home/${USER}/catkin_ws
catkin build particle_filter
cd /home/${USER}/catkin_ws/src/race_stack/state_estimation/particle_filter_python3/range_libc/pywrapper
chmod +x compile.sh 
./compile.sh

# source
source /opt/ros/noetic/setup.bash && source /home/${USER}/catkin_ws/devel/setup.bash

# python privileges
find /home/${USER}/catkin_ws -type f -name "*.py" -exec chmod +x {} \;


