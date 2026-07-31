#!/bin/bash

set -e

LOG_DIR="gazebo_logs"
GAZEBO_LOG_FILE="$LOG_DIR/single_gazebo.log"
TRAIN_LOG_FILE="$LOG_DIR/single_train.log"
export SINGLE_GUI="${SINGLE_GUI:-true}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export START_TIMESTEPS="${START_TIMESTEPS:-20000}"
export MAX_STEPS="${MAX_STEPS:-200}"
export EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-250}"
export DISCOUNT="${DISCOUNT:-0.995}"
export TAU="${TAU:-0.003}"
export POLICY_NOISE="${POLICY_NOISE:-0.08}"
export NOISE_CLIP="${NOISE_CLIP:-0.20}"
export POLICY_FREQ="${POLICY_FREQ:-2}"
export TIMEOUT_PENALTY="${TIMEOUT_PENALTY:--50}"
export EXPLORE_RANDOM_PROB_START="${EXPLORE_RANDOM_PROB_START:-0.35}"
export EXPLORE_RANDOM_PROB_END="${EXPLORE_RANDOM_PROB_END:-0.05}"
export EXPLORE_RANDOM_DECAY_STEPS="${EXPLORE_RANDOM_DECAY_STEPS:-200000}"
export SINGLE_STATUS_EVERY_EPISODES="${SINGLE_STATUS_EVERY_EPISODES:-10}"
export PYTHONUNBUFFERED=1
export GAZEBO_WORLD="${GAZEBO_WORLD:-v550_drl/wheeltec_v550_ackermann.model}"
export GAZEBO_READY_TIMEOUT="${GAZEBO_READY_TIMEOUT:-75}"
export WORLD_MIN_X="${WORLD_MIN_X:--4.0}"
export WORLD_MAX_X="${WORLD_MAX_X:-4.0}"
export WORLD_MIN_Y="${WORLD_MIN_Y:--4.0}"
export WORLD_MAX_Y="${WORLD_MAX_Y:-4.0}"
export WAREHOUSE_KEEPOUTS_ENABLE="${WAREHOUSE_KEEPOUTS_ENABLE:-0}"
export DYNAMIC_LANE_KEEPOUTS_ENABLE="${DYNAMIC_LANE_KEEPOUTS_ENABLE:-0}"

cleanup() {
    echo -e "\n[INFO] Stopping single-process training and Gazebo..."
    if [ -n "$GAZEBO_PID" ]; then
        kill -2 "$GAZEBO_PID" 2>/dev/null || true
        wait "$GAZEBO_PID" 2>/dev/null || true
    fi
    echo "[INFO] Single-process simulation stopped."
}
trap cleanup EXIT INT TERM

source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1

mkdir -p "$LOG_DIR"
: > "$TRAIN_LOG_FILE"

echo "=========================================================="
echo "[INFO] Starting one Gazebo environment for single-process training..."
echo "=========================================================="
echo "[INFO] Gazebo log: $GAZEBO_LOG_FILE"
echo "[INFO] Training log: $TRAIN_LOG_FILE"
echo "[INFO] world=$GAZEBO_WORLD"
echo "[INFO] sample_bounds=x[$WORLD_MIN_X,$WORLD_MAX_X], y[$WORLD_MIN_Y,$WORLD_MAX_Y], keepouts=$WAREHOUSE_KEEPOUTS_ENABLE, dynamic_lane_keepouts=$DYNAMIC_LANE_KEEPOUTS_ENABLE"
echo "[INFO] batch_size=$BATCH_SIZE, start_timesteps=$START_TIMESTEPS, timeout_penalty=$TIMEOUT_PENALTY"

ros2 launch v550_ackermann_gazebo ros2_drl.launch.py \
    gui:=${SINGLE_GUI} pause:=true \
    world:="${GAZEBO_WORLD}" > "$GAZEBO_LOG_FILE" 2>&1 &
GAZEBO_PID=$!
echo "[INFO] Gazebo started with PID $GAZEBO_PID"

for _ in $(seq 1 "$GAZEBO_READY_TIMEOUT"); do
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
    echo "[ERROR] Gazebo did not become ready within ${GAZEBO_READY_TIMEOUT}s."
    echo "[ERROR] Check $GAZEBO_LOG_FILE"
    exit 1
fi

echo "[INFO] Gazebo services and robot sensors are ready."

echo "[INFO] Starting the single-process trainer..."
python3 -u src/drl_navigation_ros2/train_single.py 2>&1 | tee -a "$TRAIN_LOG_FILE"
