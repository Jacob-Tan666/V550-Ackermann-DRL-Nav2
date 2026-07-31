#!/bin/bash

set -eo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export INDUSTRIAL_GUI="${INDUSTRIAL_GUI:-true}"
export INDUSTRIAL_RVIZ="${INDUSTRIAL_RVIZ:-true}"
export DYNAMIC_OBSTACLES="${DYNAMIC_OBSTACLES:-true}"
export DYNAMIC_SPEED_SCALE="${DYNAMIC_SPEED_SCALE:-1.0}"
export ODOM_TF_BRIDGE="${ODOM_TF_BRIDGE:-true}"
export NAV2_STARTUP_TIMEOUT="${NAV2_STARTUP_TIMEOUT:-90.0}"

source /opt/ros/humble/setup.bash
source install/setup.bash

missing=0
for pkg in nav2_smac_planner nav2_mppi_controller nav2_bringup nav2_map_server nav2_controller; do
    if ! ros2 pkg prefix "$pkg" >/dev/null 2>&1; then
        echo "[ERROR] Missing ROS package: $pkg"
        missing=1
    fi
done

if [ "$missing" -ne 0 ]; then
    echo "[ERROR] Install Navigation2 for Humble, e.g. sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup"
    exit 1
fi

ros2 launch v550_ackermann_gazebo industrial_navigation.launch.py \
    gui:="${INDUSTRIAL_GUI}" \
    rviz:="${INDUSTRIAL_RVIZ}" \
    dynamic_obstacles:="${DYNAMIC_OBSTACLES}" \
    dynamic_speed_scale:="${DYNAMIC_SPEED_SCALE}" \
    odom_tf_bridge:="${ODOM_TF_BRIDGE}" \
    nav2_startup_timeout:="${NAV2_STARTUP_TIMEOUT}"
