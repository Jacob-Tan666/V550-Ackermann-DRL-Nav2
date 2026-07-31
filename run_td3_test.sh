#!/bin/bash

set -eo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://localhost:$((11345 + ROS_DOMAIN_ID))}"
export TEST_GUI="${TEST_GUI:-true}"
export TEST_EPISODES="${TEST_EPISODES:-10}"
export TEST_MAX_STEPS="${TEST_MAX_STEPS:-250}"
export TEST_MODEL_DIR="${TEST_MODEL_DIR:-src/drl_navigation_ros2/models/TD3}"
export TEST_MODEL_NAME="${TEST_MODEL_NAME:-TD3_best}"
export TEST_RESULTS_DIR="${TEST_RESULTS_DIR:-src/drl_navigation_ros2/models/TD3/test_results}"
export TEST_GAZEBO_READY_TIMEOUT="${TEST_GAZEBO_READY_TIMEOUT:-60}"
export GAZEBO_WORLD="${GAZEBO_WORLD:-v550_drl/wheeltec_v550_ackermann.model}"
export DYNAMIC_OBSTACLES="${DYNAMIC_OBSTACLES:-false}"
export DYNAMIC_SPEED_SCALE="${DYNAMIC_SPEED_SCALE:-1.0}"
export WORLD_MIN_X="${WORLD_MIN_X:--4.0}"
export WORLD_MAX_X="${WORLD_MAX_X:-4.0}"
export WORLD_MIN_Y="${WORLD_MIN_Y:--4.0}"
export WORLD_MAX_Y="${WORLD_MAX_Y:-4.0}"
export WAREHOUSE_KEEPOUTS_ENABLE="${WAREHOUSE_KEEPOUTS_ENABLE:-0}"
export DYNAMIC_LANE_KEEPOUTS_ENABLE="${DYNAMIC_LANE_KEEPOUTS_ENABLE:-0}"

LOG_DIR="gazebo_logs"
GAZEBO_LOG_FILE="$LOG_DIR/gazebo_td3_test.log"
GAZEBO_PID=""

cleanup() {
    if [ -n "$GAZEBO_PID" ] && kill -0 "$GAZEBO_PID" 2>/dev/null; then
        echo "[INFO] Stopping Gazebo test environment..."
        kill -2 "$GAZEBO_PID" 2>/dev/null || true
        wait "$GAZEBO_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

mkdir -p "$LOG_DIR"

echo "[INFO] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "[INFO] GAZEBO_MASTER_URI=$GAZEBO_MASTER_URI"
echo "[INFO] world=$GAZEBO_WORLD, dynamic_obstacles=$DYNAMIC_OBSTACLES, dynamic_speed_scale=$DYNAMIC_SPEED_SCALE"
echo "[INFO] sample_bounds=x[$WORLD_MIN_X,$WORLD_MAX_X], y[$WORLD_MIN_Y,$WORLD_MAX_Y], keepouts=$WAREHOUSE_KEEPOUTS_ENABLE, dynamic_lane_keepouts=$DYNAMIC_LANE_KEEPOUTS_ENABLE"
echo "[INFO] Starting Gazebo test environment..."

ros2 launch v550_ackermann_gazebo ros2_drl.launch.py \
    gui:=${TEST_GUI} pause:=true \
    world:="${GAZEBO_WORLD}" \
    dynamic_obstacles:="${DYNAMIC_OBSTACLES}" \
    dynamic_speed_scale:="${DYNAMIC_SPEED_SCALE}" > "$GAZEBO_LOG_FILE" 2>&1 &
GAZEBO_PID=$!

for _ in $(seq 1 "$TEST_GAZEBO_READY_TIMEOUT"); do
    services="$(ros2 service list 2>/dev/null || true)"
    topics="$(ros2 topic list 2>/dev/null || true)"
    if printf '%s\n' "$services" | rg -q "^/gazebo/set_entity_state$" \
        && printf '%s\n' "$topics" | rg -q "^/scan$" \
        && printf '%s\n' "$topics" | rg -q "^/gazebo/model_states$"; then
        break
    fi
    sleep 1
done

services="$(ros2 service list 2>/dev/null || true)"
topics="$(ros2 topic list 2>/dev/null || true)"
if ! printf '%s\n' "$services" | rg -q "^/gazebo/set_entity_state$" \
    || ! printf '%s\n' "$topics" | rg -q "^/scan$" \
    || ! printf '%s\n' "$topics" | rg -q "^/gazebo/model_states$"; then
    echo "[ERROR] Gazebo did not become ready within ${TEST_GAZEBO_READY_TIMEOUT}s."
    echo "[ERROR] Required: /gazebo/set_entity_state service, /scan topic, /gazebo/model_states topic."
    echo "[ERROR] Check $GAZEBO_LOG_FILE"
    exit 1
fi

echo "[INFO] Gazebo services and robot sensors are ready. Running TD3 evaluation..."
python3 src/drl_navigation_ros2/test_td3.py \
    --ros-domain-id "$ROS_DOMAIN_ID" \
    --model-dir "$TEST_MODEL_DIR" \
    --model-name "$TEST_MODEL_NAME" \
    --episodes "$TEST_EPISODES" \
    --max-steps "$TEST_MAX_STEPS" \
    --results-dir "$TEST_RESULTS_DIR"
