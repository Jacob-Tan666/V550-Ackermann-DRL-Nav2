#!/bin/bash

set -e

CPU_CORES=$(nproc)
CPU_RESERVE_CORES="${CPU_RESERVE_CORES:-0}"
MAX_AUTO_WORKERS="${MAX_AUTO_WORKERS:-16}"
AUTO_WORKERS=$((CPU_CORES - CPU_RESERVE_CORES))
if [ "$AUTO_WORKERS" -lt 1 ]; then
    AUTO_WORKERS=1
fi
if [ "$AUTO_WORKERS" -gt "$MAX_AUTO_WORKERS" ]; then
    AUTO_WORKERS="$MAX_AUTO_WORKERS"
fi

export NUM_WORKERS="${NUM_WORKERS:-$AUTO_WORKERS}"
export EVAL_DOMAIN_ID="${EVAL_DOMAIN_ID:-99}"
export VIS_WORKER_ID="${VIS_WORKER_ID:-1}"
export SCAN_BINS="${SCAN_BINS:-50}"
export FRAME_STACK="${FRAME_STACK:-3}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export TRAIN_UTD="${TRAIN_UTD:-1.0}"
export START_TIMESTEPS="${START_TIMESTEPS:-20000}"
export MAX_TOTAL_STEPS="${MAX_TOTAL_STEPS:-2000000}"
export MAX_STEPS="${MAX_STEPS:-200}"
export EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-250}"
export DISCOUNT="${DISCOUNT:-0.995}"
export TAU="${TAU:-0.003}"
export POLICY_NOISE="${POLICY_NOISE:-0.08}"
export NOISE_CLIP="${NOISE_CLIP:-0.20}"
export POLICY_FREQ="${POLICY_FREQ:-2}"
export OU_THETA="${OU_THETA:-0.28,0.45}"
export OU_SIGMA_START="${OU_SIGMA_START:-0.10,0.18}"
export OU_SIGMA_END="${OU_SIGMA_END:-0.01,0.03}"
export OU_NOISE_CLIP="${OU_NOISE_CLIP:-0.25,0.35}"
export OU_NOISE_DECAY_STEPS="${OU_NOISE_DECAY_STEPS:-400000}"
export TIMEOUT_PENALTY="${TIMEOUT_PENALTY:--50}"
export EXPLORE_RANDOM_PROB_START="${EXPLORE_RANDOM_PROB_START:-0.35}"
export EXPLORE_RANDOM_PROB_END="${EXPLORE_RANDOM_PROB_END:-0.02}"
export EXPLORE_RANDOM_DECAY_STEPS="${EXPLORE_RANDOM_DECAY_STEPS:-800000}"
export REPLAY_STRATEGY="${REPLAY_STRATEGY:-per}"
export PER_ALPHA="${PER_ALPHA:-0.6}"
export PER_BETA_START="${PER_BETA_START:-0.4}"
export PER_BETA_END="${PER_BETA_END:-1.0}"
export PER_BETA_DECAY_STEPS="${PER_BETA_DECAY_STEPS:-400000}"
export PER_EPS="${PER_EPS:-0.0001}"
export PER_SUCCESS_PRIORITY_BOOST="${PER_SUCCESS_PRIORITY_BOOST:-2.5}"
export PER_MANEUVER_PRIORITY_BOOST="${PER_MANEUVER_PRIORITY_BOOST:-2.0}"
export OBS_NOISE_ENABLE="${OBS_NOISE_ENABLE:-1}"
export LIDAR_NOISE_STD="${LIDAR_NOISE_STD:-0.008}"
export LIDAR_DROPOUT_PROB="${LIDAR_DROPOUT_PROB:-0.004}"
export LIDAR_SPIKE_PROB="${LIDAR_SPIKE_PROB:-0.002}"
export DISTANCE_NOISE_STD="${DISTANCE_NOISE_STD:-0.01}"
export HEADING_NOISE_STD="${HEADING_NOISE_STD:-0.015}"
export ACTION_DELAY_ENABLE="${ACTION_DELAY_ENABLE:-1}"
export ACTION_DELAY_MIN_STEPS="${ACTION_DELAY_MIN_STEPS:-0}"
export ACTION_DELAY_MAX_STEPS="${ACTION_DELAY_MAX_STEPS:-1}"
export DOMAIN_RANDOMIZATION_ENABLE="${DOMAIN_RANDOMIZATION_ENABLE:-1}"
export RANDOMIZE_EVAL="${RANDOMIZE_EVAL:-0}"
export SIM2REAL_START_SCALE="${SIM2REAL_START_SCALE:-0.25}"
export SIM2REAL_RAMP_EPISODES="${SIM2REAL_RAMP_EPISODES:-20000}"
export FORWARD_SPEED_SCALE_RANGE="${FORWARD_SPEED_SCALE_RANGE:-0.95,1.03}"
export REVERSE_SPEED_SCALE_RANGE="${REVERSE_SPEED_SCALE_RANGE:-0.92,1.03}"
export STEER_SCALE_RANGE="${STEER_SCALE_RANGE:-0.95,1.06}"
export STEER_RESPONSE_RANGE="${STEER_RESPONSE_RANGE:-0.90,1.12}"
export ROLLING_DRAG_RANGE="${ROLLING_DRAG_RANGE:-0.00,0.04}"
export STEP_TIME_RANGE="${STEP_TIME_RANGE:-0.095,0.11}"
export LIDAR_BIAS_RANGE="${LIDAR_BIAS_RANGE:--0.01,0.01}"
export DISTANCE_BIAS_RANGE="${DISTANCE_BIAS_RANGE:--0.015,0.015}"
export HEADING_BIAS_RANGE="${HEADING_BIAS_RANGE:--0.02,0.02}"
export DRAIN_MULTIPLIER="${DRAIN_MULTIPLIER:-8}"
export MIN_DRAIN_LIMIT="${MIN_DRAIN_LIMIT:-256}"
export WEIGHT_SYNC_EVERY_STEPS="${WEIGHT_SYNC_EVERY_STEPS:-500}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-20000}"
export STATUS_EVERY_SEC="${STATUS_EVERY_SEC:-30}"
export RESUME_MODEL="${RESUME_MODEL:-0}"
export MODEL_NAME="${MODEL_NAME:-TD3}"
export MODEL_DIR="${MODEL_DIR:-src/drl_navigation_ros2/models/TD3}"
export TRANSITION_QUEUE_SIZE="${TRANSITION_QUEUE_SIZE:-$(( NUM_WORKERS * 1500 > 20000 ? NUM_WORKERS * 1500 : 20000 ))}"
export GAZEBO_WORLD="${GAZEBO_WORLD:-v550_drl/wheeltec_v550_ackermann.model}"
export WORLD_MIN_X="${WORLD_MIN_X:--4.0}"
export WORLD_MAX_X="${WORLD_MAX_X:-4.0}"
export WORLD_MIN_Y="${WORLD_MIN_Y:--4.0}"
export WORLD_MAX_Y="${WORLD_MAX_Y:-4.0}"
export WAREHOUSE_KEEPOUTS_ENABLE="${WAREHOUSE_KEEPOUTS_ENABLE:-0}"
export DYNAMIC_LANE_KEEPOUTS_ENABLE="${DYNAMIC_LANE_KEEPOUTS_ENABLE:-0}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
LOG_DIR="gazebo_logs"
LEARNER_LOG_FILE="$LOG_DIR/multi_learner.log"
PIDS=""

cleanup() {
    echo -e "\n[INFO] Stopping multi-process training and Gazebo environments..."
    for pid in $PIDS; do
        kill -2 "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    echo "[INFO] Multi-process simulation stopped."
}
trap cleanup EXIT INT TERM

source /opt/ros/humble/setup.bash
source install/setup.bash
mkdir -p "$LOG_DIR"
: > "$LEARNER_LOG_FILE"

launch_worker() {
    local worker_id="$1"
    local domain_id="$2"
    local log_file="$LOG_DIR/gazebo_worker_${worker_id}.log"

    # Only the selected worker opens a Gazebo client.
    local gui_flag="false"
    if [ "$VIS_WORKER_ID" != "0" ] && [ "$worker_id" = "$VIS_WORKER_ID" ]; then
        gui_flag="true"
    fi

    export ROS_DOMAIN_ID="$domain_id"
    export GAZEBO_MASTER_URI="http://localhost:$((11345 + domain_id))"
    ros2 launch v550_ackermann_gazebo ros2_drl.launch.py \
        gui:=${gui_flag} pause:=true \
        world:="${GAZEBO_WORLD}" > "$log_file" 2>&1 &
    local pid=$!
    PIDS="$PIDS $pid"
    echo "  -> Worker ${worker_id} started (ROS_DOMAIN_ID=${domain_id}, gui=${gui_flag}), PID: $pid"
    sleep 3
}

echo "=========================================================="
echo "[INFO] CPU cores: $CPU_CORES, reserve: $CPU_RESERVE_CORES, workers: $NUM_WORKERS"
echo "[INFO] scan_bins=$SCAN_BINS, frame_stack=$FRAME_STACK, replay=$REPLAY_STRATEGY"
echo "[INFO] batch_size=$BATCH_SIZE, train_utd=$TRAIN_UTD, start_timesteps=$START_TIMESTEPS"
echo "[INFO] max_total_steps=$MAX_TOTAL_STEPS, resume_model=$RESUME_MODEL"
echo "[INFO] world=$GAZEBO_WORLD"
echo "[INFO] sample_bounds=x[$WORLD_MIN_X,$WORLD_MAX_X], y[$WORLD_MIN_Y,$WORLD_MAX_Y], keepouts=$WAREHOUSE_KEEPOUTS_ENABLE, dynamic_lane_keepouts=$DYNAMIC_LANE_KEEPOUTS_ENABLE"
echo "[INFO] discount=$DISCOUNT, tau=$TAU, timeout_penalty=$TIMEOUT_PENALTY"
echo "[INFO] OU(theta=$OU_THETA sigma=$OU_SIGMA_START->$OU_SIGMA_END decay=$OU_NOISE_DECAY_STEPS)"
echo "[INFO] obs_noise=$OBS_NOISE_ENABLE action_delay=$ACTION_DELAY_ENABLE domain_rand=$DOMAIN_RANDOMIZATION_ENABLE"
echo "[INFO] learner log: $LEARNER_LOG_FILE"
echo "=========================================================="

# Start the sampling environments first.
for i in $(seq 1 "$NUM_WORKERS"); do
    launch_worker "$i" "$i"
done

# Start one additional headless evaluation environment.
export ROS_DOMAIN_ID="$EVAL_DOMAIN_ID"
export GAZEBO_MASTER_URI="http://localhost:$((11345 + EVAL_DOMAIN_ID))"
ros2 launch v550_ackermann_gazebo ros2_drl.launch.py \
    gui:=false pause:=true \
    world:="${GAZEBO_WORLD}" > "$LOG_DIR/gazebo_eval.log" 2>&1 &
pid=$!
PIDS="$PIDS $pid"
echo "  -> Evaluation environment started (ROS_DOMAIN_ID=$EVAL_DOMAIN_ID, no GUI), PID: $pid"
sleep 3

export ROS_DOMAIN_ID=0
python3 -u src/drl_navigation_ros2/train.py 2>&1 | tee -a "$LEARNER_LOG_FILE"
