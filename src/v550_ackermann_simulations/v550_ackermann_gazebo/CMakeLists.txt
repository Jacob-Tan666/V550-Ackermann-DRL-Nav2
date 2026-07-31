################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.5)
project(v550_ackermann_gazebo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(MSVC)
  add_compile_definitions(_USE_MATH_DEFINES)
endif()

################################################################################
# Find ament packages and libraries for ament and system dependencies
################################################################################
find_package(ament_cmake REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_plugins REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_ros_pkgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

################################################################################
# Build
################################################################################
link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

include_directories(
  include
  ${GAZEBO_INCLUDE_DIRS}
)

set(dependencies
  "geometry_msgs"
  "nav_msgs"
  "rclcpp"
  "sensor_msgs"
  "std_msgs"
  "tf2"
  "tf2_geometry_msgs"
  "tf2_ros"
)

add_library(v550_safe_ackermann_drive SHARED src/v550_safe_ackermann_drive.cpp)
ament_target_dependencies(
  v550_safe_ackermann_drive
  gazebo_plugins
  gazebo_ros
  ${dependencies}
)
target_link_libraries(v550_safe_ackermann_drive ${GAZEBO_LIBRARIES})

add_library(v550_goal_orientation_bt_node SHARED
  src/use_path_approach_orientation_bt_node.cpp
)
ament_target_dependencies(
  v550_goal_orientation_bt_node
  behaviortree_cpp_v3
  geometry_msgs
  nav_msgs
)
target_compile_definitions(v550_goal_orientation_bt_node PRIVATE BT_PLUGIN_EXPORT)

################################################################################
# Install
################################################################################
install(TARGETS v550_safe_ackermann_drive
  DESTINATION lib/${PROJECT_NAME}
)

# BehaviorTree.CPP plugins are discovered through the workspace library path.
install(TARGETS v550_goal_orientation_bt_node
  DESTINATION lib
)

install(PROGRAMS
  scripts/ackermann_cmd_vel_adapter.py
  scripts/dynamic_obstacles.py
  scripts/goal_distance_display.py
  scripts/initial_pose_gazebo_sync.py
  scripts/joint_state_sanitizer.py
  scripts/nav2_startup_gate.py
  scripts/odom_tf_broadcaster.py
  scripts/tile_visualization_windows.py
  scripts/trajectory_history.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY behavior_trees config maps
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY launch models rviz worlds
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY include/
  DESTINATION include/
)

################################################################################
# Macro for ament package
################################################################################
ament_export_include_directories(include)
ament_export_dependencies(gazebo_ros_pkgs)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(nav_msgs)
ament_export_dependencies(rclcpp)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(tf2)
ament_package()
