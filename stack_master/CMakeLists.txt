cmake_minimum_required(VERSION 3.8)
project(stack_master)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install maps directory, minus the point clouds.
#
# *.pcd is the one map file .gitignore never carries (stack_master/maps/**/*.pcd),
# so it is always moved between machines by hand -- and a hand-moved pcd that ends
# up a dangling/self-referencing symlink kills the whole build: ament's
# --symlink-install links the file and then validates with `if(NOT EXISTS ...)`,
# which resolves symlinks, so an unresolvable source is fatal (a *missing* file is
# harmless -- it is simply not globbed). Keeping the clouds out of the install tree
# makes the build immune to that. Runtime still finds them: the launch files read
# $(env STACK_MASTER_MAPS_ROOT ...), which unicorn.sh defaults to the source tree.
#
# The two branches are NOT interchangeable. ament's symlink-install replaces
# install() with a function parsed as
#   cmake_parse_arguments(ARG "OPTIONAL" "DESTINATION" "DIRECTORY;PATTERN;PATTERN_EXCLUDE")
# which has no EXCLUDE keyword, so CMake's own `PATTERN "*.pcd" EXCLUDE` is swallowed
# into the *include* list (ARG_PATTERN = "*.pcd;EXCLUDE") and would install the pcds
# and nothing else -- silently, with no unparsed-argument warning.
if(AMENT_CMAKE_SYMLINK_INSTALL)
  install(DIRECTORY
    maps
    DESTINATION share/${PROJECT_NAME}/
    PATTERN_EXCLUDE "*.pcd"
  )
else()
  install(DIRECTORY
    maps
    DESTINATION share/${PROJECT_NAME}/
    PATTERN "*.pcd" EXCLUDE
  )
endif()

# Install Python scripts
install(PROGRAMS
  scripts/simple_mux_node.py
  scripts/set_initial_pose_node.py
  scripts/finish_map.py
  scripts/pose_to_odom_node.py
  scripts/map_relay_node.py
  scripts/keyboard_joy_node.py
  scripts/global_velocity_planner.py
  scripts/static_tf_publisher.py
  scripts/vesc_calibration_node.py
  scripts/opponent_bridge_node.py
  scripts/vehdyn_test_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
