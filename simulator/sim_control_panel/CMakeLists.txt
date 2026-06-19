cmake_minimum_required(VERSION 3.8...3.27)
project(sim_control_panel)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

set(CMAKE_AUTOMOC ON)

add_library(${PROJECT_NAME} SHARED
  src/sim_control_panel.cpp
  include/sim_control_panel/sim_control_panel.hpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# QT_NO_KEYWORDS avoids clashes between Qt's signals/slots/emit and other libs
target_compile_definitions(${PROJECT_NAME} PRIVATE QT_NO_KEYWORDS)

target_link_libraries(${PROJECT_NAME} Qt5::Widgets Qt5::Core)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rviz_common
  pluginlib
  std_msgs
)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/
  DESTINATION include
)
install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

# Register the panel with RViz via pluginlib (base package: rviz_common)
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(rviz_common pluginlib std_msgs rclcpp)

ament_package()
