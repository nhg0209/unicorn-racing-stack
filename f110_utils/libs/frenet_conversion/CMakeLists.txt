cmake_minimum_required(VERSION 3.8)
project(frenet_conversion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(f110_msgs REQUIRED)

# Generate the conversion services (ROS1 parity: frenet_conversion/srv/*)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Frenet2Glob.srv"
  "srv/Frenet2GlobArr.srv"
  "srv/Glob2Frenet.srv"
  "srv/Glob2FrenetArr.srv"
  DEPENDENCIES f110_msgs
)

# Install the Python module (frenet_conversion.frenet_converter)
ament_python_install_package(${PROJECT_NAME})

# Install the demo node executable
install(PROGRAMS
  scripts/frenet_converter_demo
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
