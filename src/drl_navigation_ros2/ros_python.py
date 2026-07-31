import os
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Twist
from squaternion import Quaternion

from ros_nodes import (
    BumperSubscriber,
    CmdVelPublisher,
    MarkerPublisher,
    OdomSubscriber,
    PhysicsClient,
    ResetWorldClient,
    ScanSubscriber,
    SensorSubscriber,
    SetModelStateClient,
)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_float_pair(name, default):
    value = os.environ.get(name)
    if value is None:
        return tuple(float(v) for v in default)
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        return tuple(float(v) for v in default)
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return tuple(float(v) for v in default)


class ROS_env:
    def __init__(
        self,
        init_target_distance=3.0,
        target_dist_increase=0.03,
        max_target_dist=8.0,
        target_reached_delta=0.55,
        collision_delta=0.20,
        args=None,
    ):
        rclpy.init(args=args)
        self.cmd_vel_publisher = CmdVelPublisher()
        self.scan_subscriber = ScanSubscriber()
        self.odom_subscriber = OdomSubscriber()
        self.bumper_subscriber = BumperSubscriber()
        self.robot_state_publisher = SetModelStateClient()
        self.world_reset = ResetWorldClient()
        self.physics_client = PhysicsClient()
        self.publish_target = MarkerPublisher()
        self.sensor_subscriber = SensorSubscriber()

        self.element_positions = [
            [-3.25, 3.10],
            [2.75, -3.20],
            [-1.80, -2.90],
            [2.70, 2.25],
        ]
        self.init_target_distance = float(init_target_distance)
        self.target_dist = float(init_target_distance)
        self.target_dist_increase = target_dist_increase
        self.max_target_dist = max_target_dist
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta
        self.target_dist_decrease_collision = 0.02
        self.target_dist_decrease_timeout = 0.04
        self.hard_target_prob = 0.20

        self.max_linear_speed = 0.45
        self.max_reverse_speed = 0.45
        self.max_steer_angle = 0.34
        self.min_motion_speed = 0.05
        self.min_steer_speed = 0.08
        self.high_speed_steer_ratio_min = 0.60
        self.safe_scan_dist = 1.2
        self.scan_max_range = 8.0
        self.self_hit_min_scan = 0.35
        self.collision_scan_threshold = max(self.collision_delta, 0.24)
        self.collision_min_close_count = 3
        self.progress_clip = 0.10
        self.stagnation_progress_eps = 0.005
        self.min_target_sample_dist = max(1.0, self.target_reached_delta * 1.8)
        self.bumper_collision_streak = 0
        self.prev_distance = None
        self.prev_steer_cmd = 0.0
        self.reset_settle_time = 0.12
        self.base_step_time = 0.10
        self.last_reset_pose = None
        self.eval_mode = False
        self.verbose_events = _env_bool("ROS_ENV_EVENT_LOG", False)
        self.training_episode_count = 0
        self.world_min_x = _env_float("WORLD_MIN_X", -4.0)
        self.world_max_x = _env_float("WORLD_MAX_X", 4.0)
        self.world_min_y = _env_float("WORLD_MIN_Y", -4.0)
        self.world_max_y = _env_float("WORLD_MAX_Y", 4.0)
        self.use_warehouse_keepouts = _env_bool("WAREHOUSE_KEEPOUTS_ENABLE", False)
        self.use_dynamic_lane_keepouts = _env_bool("DYNAMIC_LANE_KEEPOUTS_ENABLE", False)

        # Sim2Real options
        self.randomize_eval = _env_bool("RANDOMIZE_EVAL", False)
        self.domain_randomization_enable = _env_bool("DOMAIN_RANDOMIZATION_ENABLE", True)
        self.enable_sensor_noise = _env_bool("OBS_NOISE_ENABLE", True)
        self.enable_action_delay = _env_bool("ACTION_DELAY_ENABLE", True)
        self.sim2real_start_scale = float(np.clip(_env_float("SIM2REAL_START_SCALE", 0.25), 0.0, 1.0))
        self.sim2real_ramp_episodes = max(1, _env_int("SIM2REAL_RAMP_EPISODES", 600))

        self.lidar_noise_std = _env_float("LIDAR_NOISE_STD", 0.015)
        self.lidar_dropout_prob = _env_float("LIDAR_DROPOUT_PROB", 0.01)
        self.lidar_spike_prob = _env_float("LIDAR_SPIKE_PROB", 0.005)
        self.distance_noise_std = _env_float("DISTANCE_NOISE_STD", 0.02)
        self.heading_noise_std = _env_float("HEADING_NOISE_STD", 0.03)

        self.action_delay_min_steps = max(0, _env_int("ACTION_DELAY_MIN_STEPS", 0))
        self.action_delay_max_steps = max(
            self.action_delay_min_steps, _env_int("ACTION_DELAY_MAX_STEPS", 2)
        )

        self.forward_speed_scale_range = _env_float_pair("FORWARD_SPEED_SCALE_RANGE", (0.90, 1.05))
        self.reverse_speed_scale_range = _env_float_pair("REVERSE_SPEED_SCALE_RANGE", (0.85, 1.05))
        self.steer_scale_range = _env_float_pair("STEER_SCALE_RANGE", (0.90, 1.10))
        self.steer_response_range = _env_float_pair("STEER_RESPONSE_RANGE", (0.80, 1.20))
        self.rolling_drag_range = _env_float_pair("ROLLING_DRAG_RANGE", (0.00, 0.08))
        self.step_time_range = _env_float_pair("STEP_TIME_RANGE", (0.09, 0.12))
        self.lidar_bias_range = _env_float_pair("LIDAR_BIAS_RANGE", (-0.02, 0.02))
        self.distance_bias_range = _env_float_pair("DISTANCE_BIAS_RANGE", (-0.03, 0.03))
        self.heading_bias_range = _env_float_pair("HEADING_BIAS_RANGE", (-0.04, 0.04))

        self.domain_forward_scale = 1.0
        self.domain_reverse_scale = 1.0
        self.domain_steer_scale = 1.0
        self.domain_steer_response_scale = 1.0
        self.domain_rolling_drag = 0.0
        self.domain_step_time = self.base_step_time
        self.domain_lidar_bias = 0.0
        self.domain_distance_bias = 0.0
        self.domain_heading_bias = 0.0
        self.action_delay_steps = 0
        self.action_delay_buffer = deque()
        self._configure_episode_dynamics(eval_mode=False)

        self.target = self.set_target_position([0.0, 0.0])

    def _sample_range(self, bounds):
        low = float(min(bounds))
        high = float(max(bounds))
        return float(np.random.uniform(low, high))

    def _sim2real_scale(self, eval_mode):
        if eval_mode:
            return 1.0 if self.randomize_eval else 0.0
        progress = min(1.0, self.training_episode_count / float(self.sim2real_ramp_episodes))
        return self.sim2real_start_scale + progress * (1.0 - self.sim2real_start_scale)

    def _blend_identity_range(self, bounds, scale, identity):
        low = float(min(bounds))
        high = float(max(bounds))
        return (
            identity + scale * (low - identity),
            identity + scale * (high - identity),
        )

    def _scaled_symmetric_range(self, bounds, scale):
        low = float(min(bounds))
        high = float(max(bounds))
        return low * scale, high * scale

    def _reset_action_delay_buffer(self):
        history_len = max(self.action_delay_steps, 0) + 1
        self.action_delay_buffer = deque(
            [[0.0, 0.0] for _ in range(history_len)],
            maxlen=history_len,
        )

    def _configure_episode_dynamics(self, eval_mode):
        randomize = self.domain_randomization_enable and (not eval_mode or self.randomize_eval)
        randomization_scale = self._sim2real_scale(eval_mode) if randomize else 0.0

        if randomize:
            # These are runtime-accessible proxies for mass/friction/servo dynamics.
            self.domain_forward_scale = self._sample_range(
                self._blend_identity_range(self.forward_speed_scale_range, randomization_scale, 1.0)
            )
            self.domain_reverse_scale = self._sample_range(
                self._blend_identity_range(self.reverse_speed_scale_range, randomization_scale, 1.0)
            )
            self.domain_steer_scale = self._sample_range(
                self._blend_identity_range(self.steer_scale_range, randomization_scale, 1.0)
            )
            self.domain_steer_response_scale = self._sample_range(
                self._blend_identity_range(self.steer_response_range, randomization_scale, 1.0)
            )
            self.domain_rolling_drag = self._sample_range(
                self._scaled_symmetric_range(self.rolling_drag_range, randomization_scale)
            )
            self.domain_step_time = self._sample_range(
                self._blend_identity_range(self.step_time_range, randomization_scale, self.base_step_time)
            )
            self.domain_lidar_bias = self._sample_range(
                self._scaled_symmetric_range(self.lidar_bias_range, randomization_scale)
            )
            self.domain_distance_bias = self._sample_range(
                self._scaled_symmetric_range(self.distance_bias_range, randomization_scale)
            )
            self.domain_heading_bias = self._sample_range(
                self._scaled_symmetric_range(self.heading_bias_range, randomization_scale)
            )
        else:
            self.domain_forward_scale = 1.0
            self.domain_reverse_scale = 1.0
            self.domain_steer_scale = 1.0
            self.domain_steer_response_scale = 1.0
            self.domain_rolling_drag = 0.0
            self.domain_step_time = self.base_step_time
            self.domain_lidar_bias = 0.0
            self.domain_distance_bias = 0.0
            self.domain_heading_bias = 0.0

        if self.enable_action_delay and (not eval_mode or self.randomize_eval):
            scaled_delay_max = int(round(self.action_delay_max_steps * randomization_scale))
            scaled_delay_max = max(self.action_delay_min_steps, scaled_delay_max)
            self.action_delay_steps = int(
                np.random.randint(self.action_delay_min_steps, scaled_delay_max + 1)
            )
        else:
            self.action_delay_steps = 0
        self._reset_action_delay_buffer()

        if self.verbose_events:
            print(
                "[Domain] "
                f"scale={randomization_scale:.3f} "
                f"fwd={self.domain_forward_scale:.3f} rev={self.domain_reverse_scale:.3f} "
                f"steer={self.domain_steer_scale:.3f} steer_resp={self.domain_steer_response_scale:.3f} "
                f"drag={self.domain_rolling_drag:.3f} step_time={self.domain_step_time:.3f} "
                f"delay_steps={self.action_delay_steps}",
                flush=True,
            )

    def _apply_action_delay(self, raw_linear, raw_steer):
        delayed_action = [float(np.clip(raw_linear, -1.0, 1.0)), float(np.clip(raw_steer, -1.0, 1.0))]
        if self.action_delay_steps <= 0:
            return delayed_action
        self.action_delay_buffer.append(delayed_action)
        return self.action_delay_buffer.popleft()

    def _apply_observation_noise(self, latest_scan, distance, cos, sin):
        scan = np.asarray(latest_scan, dtype=np.float32)
        scan = np.nan_to_num(
            scan,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        scan = np.clip(scan, 0.0, self.scan_max_range)

        if not self.enable_sensor_noise or (self.eval_mode and not self.randomize_eval):
            return scan, float(distance), float(cos), float(sin)

        noise_scale = self._sim2real_scale(self.eval_mode)

        noisy_scan = scan * (1.0 + self.domain_lidar_bias)
        if self.lidar_noise_std > 0.0 and noise_scale > 0.0:
            noisy_scan = noisy_scan + np.random.normal(
                0.0, self.lidar_noise_std * noise_scale, size=noisy_scan.shape
            ).astype(np.float32)
        if self.lidar_dropout_prob > 0.0 and noise_scale > 0.0:
            dropout_mask = np.random.rand(noisy_scan.size) < (self.lidar_dropout_prob * noise_scale)
            noisy_scan[dropout_mask] = self.scan_max_range
        if self.lidar_spike_prob > 0.0 and noise_scale > 0.0:
            spike_mask = np.random.rand(noisy_scan.size) < (self.lidar_spike_prob * noise_scale)
            spike_count = int(np.count_nonzero(spike_mask))
            if spike_count > 0:
                noisy_scan[spike_mask] = np.random.uniform(
                    0.0, self.scan_max_range, size=spike_count
                ).astype(np.float32)
        noisy_scan = np.clip(noisy_scan, 0.0, self.scan_max_range)

        noisy_distance = float(distance) * (1.0 + self.domain_distance_bias)
        if self.distance_noise_std > 0.0 and noise_scale > 0.0:
            noisy_distance += float(np.random.normal(0.0, self.distance_noise_std * noise_scale))
        noisy_distance = float(np.clip(noisy_distance, 0.0, self.max_target_dist * 1.5))

        heading = float(np.arctan2(sin, cos)) + self.domain_heading_bias
        if self.heading_noise_std > 0.0 and noise_scale > 0.0:
            heading += float(np.random.normal(0.0, self.heading_noise_std * noise_scale))
        noisy_cos = float(np.cos(heading))
        noisy_sin = float(np.sin(heading))
        return noisy_scan, noisy_distance, noisy_cos, noisy_sin

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        delayed_linear, delayed_steer = self._apply_action_delay(lin_velocity, ang_velocity)
        cmd_linear, cmd_steer = self.scale_action(delayed_linear, delayed_steer)
        cmd_linear = float(np.clip(cmd_linear, -self._signed_speed_limit(-1.0), self._signed_speed_limit(1.0)))
        cmd_steer = float(np.clip(cmd_steer, -self.max_steer_angle, self.max_steer_angle))

        if abs(cmd_linear) < self.min_motion_speed:
            cmd_steer = 0.0
            self.prev_steer_cmd = 0.0

        self.cmd_vel_publisher.publish_cmd_vel(cmd_linear, cmd_steer)

        self.sensor_subscriber.latest_scan = None
        self.sensor_subscriber.latest_position = None
        self.sensor_subscriber.latest_heading = None
        self.bumper_subscriber.latest_collision = False

        self.physics_client.unpause_physics()
        time.sleep(self.domain_step_time)
        while True:
            rclpy.spin_once(self.sensor_subscriber, timeout_sec=0.01)
            rclpy.spin_once(self.bumper_subscriber, timeout_sec=0.01)
            latest_scan, latest_position, latest_orientation = self.sensor_subscriber.get_latest_sensor()
            if latest_scan is not None and latest_position is not None and latest_orientation is not None:
                break
            time.sleep(0.01)

        self.physics_client.pause_physics()

        distance, cos, sin, _ = self.get_dist_sincos(latest_position, latest_orientation)
        collision = self.check_collision(latest_scan)
        goal = self.check_target(distance, collision)
        observed_scan, obs_distance, obs_cos, obs_sin = self._apply_observation_noise(
            latest_scan, distance, cos, sin
        )
        obs_action = self._normalize_executed_action(cmd_linear, cmd_steer)
        executed_action = [cmd_linear, cmd_steer]
        reward = self.get_reward(goal, collision, executed_action, latest_scan, distance, cos)
        self.prev_distance = distance
        self.prev_steer_cmd = cmd_steer
        return observed_scan, obs_distance, obs_cos, obs_sin, collision, goal, obs_action, reward

    def _send_stop_cmd(self, repeat=3, interval=0.02):
        for _ in range(repeat):
            self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)
            time.sleep(interval)

    def _sample_observation_after_reset(self):
        self.sensor_subscriber.latest_scan = None
        self.sensor_subscriber.latest_position = None
        self.sensor_subscriber.latest_heading = None
        self.bumper_subscriber.latest_collision = False

        self.physics_client.unpause_physics()
        end_time = time.time() + 0.03
        while time.time() < end_time:
            self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)
            rclpy.spin_once(self.sensor_subscriber, timeout_sec=0.005)
            rclpy.spin_once(self.bumper_subscriber, timeout_sec=0.005)

        latest_scan, latest_position, latest_orientation = self.sensor_subscriber.get_latest_sensor()
        if latest_scan is None or latest_position is None or latest_orientation is None:
            timeout = time.time() + 0.2
            while time.time() < timeout:
                self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)
                rclpy.spin_once(self.sensor_subscriber, timeout_sec=0.01)
                rclpy.spin_once(self.bumper_subscriber, timeout_sec=0.01)
                latest_scan, latest_position, latest_orientation = self.sensor_subscriber.get_latest_sensor()
                if latest_scan is not None and latest_position is not None and latest_orientation is not None:
                    break

        if latest_position is None or latest_orientation is None:
            latest_position, latest_orientation = self.odom_subscriber.get_latest_odom()

        if (latest_position is None or latest_orientation is None) and self.last_reset_pose is not None:
            latest_position = self.last_reset_pose.position
            latest_orientation = self.last_reset_pose.orientation

        self.physics_client.pause_physics()
        self._send_stop_cmd(repeat=2, interval=0.01)

        if latest_scan is None:
            latest_scan = self.scan_subscriber.get_latest_scan()
        if latest_scan is None:
            latest_scan = [self.scan_max_range] * 180

        if latest_position is None or latest_orientation is None:
            raise RuntimeError("Reset observation unavailable: odom/model states were not received in time.")

        distance, cos, sin, _ = self.get_dist_sincos(latest_position, latest_orientation)
        collision = self.check_collision(latest_scan)
        goal = self.check_target(distance, collision)
        observed_scan, obs_distance, obs_cos, obs_sin = self._apply_observation_noise(
            latest_scan, distance, cos, sin
        )
        action = [0.0, 0.0]
        reward = self.get_reward(goal, collision, action, latest_scan, distance, cos)

        self.prev_distance = distance
        self.prev_steer_cmd = 0.0
        return observed_scan, obs_distance, obs_cos, obs_sin, collision, goal, action, reward

    def reset(self):
        self.eval_mode = False
        self.training_episode_count += 1
        self._configure_episode_dynamics(eval_mode=False)
        self.target_dist = float(
            np.clip(self.target_dist, self.init_target_distance, self.max_target_dist)
        )
        self.physics_client.pause_physics()
        self._send_stop_cmd(repeat=4, interval=0.02)

        self.world_reset.reset_world()
        time.sleep(0.05)

        self.prev_distance = None
        self.prev_steer_cmd = 0.0
        self._send_stop_cmd(repeat=3, interval=0.02)
        self.bumper_subscriber.reset_collision()
        self.bumper_subscriber.latest_collision = False
        self.bumper_collision_streak = 0

        self.element_positions = [
            [-2.93, 3.17],
            [2.86, -3.0],
            [-2.77, -0.96],
            [2.83, 2.93],
        ]
        self.set_positions()
        self._send_stop_cmd(repeat=3, interval=0.02)

        self.publish_target.publish(self.target[0], self.target[1])

        self.physics_client.unpause_physics()
        self._send_stop_cmd(repeat=4, interval=0.02)
        time.sleep(self.reset_settle_time)
        self.physics_client.pause_physics()

        return self._sample_observation_after_reset()

    def eval(self, scenario):
        self.eval_mode = True
        self._configure_episode_dynamics(eval_mode=True)
        self.prev_distance = None
        self.prev_steer_cmd = 0.0
        self.bumper_collision_streak = 0
        self.bumper_subscriber.reset_collision()
        self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)

        self.target = [scenario[-1].x, scenario[-1].y]
        self.publish_target.publish(self.target[0], self.target[1])

        for element in scenario[:-1]:
            self.set_position(element.name, element.x, element.y, element.angle)

        self.physics_client.unpause_physics()
        time.sleep(1)
        return self.step(lin_velocity=0.0, ang_velocity=0.0)

    def end_eval_episode(self):
        self.eval_mode = False

    def update_curriculum(self, goal=False, collision=False, timeout=False):
        if self.eval_mode:
            return self.target_dist

        if goal:
            self.target_dist = min(self.max_target_dist, self.target_dist + self.target_dist_increase)
        elif collision:
            self.target_dist = max(
                self.init_target_distance,
                self.target_dist - self.target_dist_decrease_collision,
            )
        elif timeout:
            self.target_dist = max(
                self.init_target_distance,
                self.target_dist - self.target_dist_decrease_timeout,
            )
        return self.target_dist

    def _sample_target_distance_cap(self):
        if np.random.rand() < self.hard_target_prob:
            return self.max_target_dist
        return self.target_dist

    def set_target_position(self, robot_position):
        pos = False
        attempts = 0
        while not pos:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("Unable to sample a valid target position. Check WORLD_* bounds and keepouts.")
            dist_cap = max(self._sample_target_distance_cap(), self.min_target_sample_dist + 1e-3)
            radius = np.random.uniform(self.min_target_sample_dist, dist_cap)
            angle = np.random.uniform(-np.pi, np.pi)
            x = np.clip(
                robot_position[0] + radius * np.cos(angle),
                self.world_min_x,
                self.world_max_x,
            )
            y = np.clip(
                robot_position[1] + radius * np.sin(angle),
                self.world_min_y,
                self.world_max_y,
            )
            goal_distance = np.linalg.norm([x - robot_position[0], y - robot_position[1]])
            pos = (
                self.check_position(x, y, 1.2)
                and self.check_static_keepout(x, y, 0.7)
                and goal_distance >= self.min_target_sample_dist
            )
        self.element_positions.append([x, y])
        return [x, y]

    def set_random_position(self, name):
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        attempts = 0
        while not pos:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(f"Unable to sample a valid position for {name}. Check WORLD_* bounds and keepouts.")
            x = np.random.uniform(self.world_min_x, self.world_max_x)
            y = np.random.uniform(self.world_min_y, self.world_max_y)
            pos = (
                self.check_position(x, y, 1.8)
                and self.check_static_keepout(x, y, 0.8)
                and self.check_dynamic_lane_keepout(x, y, 0.2)
            )
        self.element_positions.append([x, y])
        self.set_position(name, x, y, angle)

    def set_robot_position(self):
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        attempts = 0
        while not pos:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("Unable to sample a valid robot position. Check WORLD_* bounds and keepouts.")
            x = np.random.uniform(self.world_min_x, self.world_max_x)
            y = np.random.uniform(self.world_min_y, self.world_max_y)
            pos = self.check_position(x, y, 1.8) and self.check_static_keepout(x, y, 0.9)
        self.set_position("wheeltec_v550_ackermann", x, y, angle)
        return x, y

    def set_position(self, name, x, y, angle):
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.x = quaternion.x
        pose.orientation.y = quaternion.y
        pose.orientation.z = quaternion.z
        pose.orientation.w = quaternion.w

        self.robot_state_publisher.set_state(name, pose)
        if name == "wheeltec_v550_ackermann":
            self.last_reset_pose = pose
        rclpy.spin_once(self.robot_state_publisher)

    def set_positions(self):
        for i in range(4, 10):
            name = "obstacle" + str(i + 1)
            self.set_random_position(name)

        robot_position = self.set_robot_position()
        self.target = self.set_target_position(robot_position)

    def check_position(self, x, y, min_dist):
        pos = True
        for element in self.element_positions:
            distance_vector = [element[0] - x, element[1] - y]
            distance = np.linalg.norm(distance_vector)
            if distance < min_dist:
                pos = False
        return pos

    def check_static_keepout(self, x, y, margin=0.0):
        if not self.use_warehouse_keepouts:
            return True

        keepout_boxes = [
            # Gazebo warehouse wall/outer boundary clearance.
            (-6.30, -5.15, -9.80, 9.80),
            (4.05, 6.30, -9.80, 9.80),
            (-6.30, 6.30, 7.05, 9.80),
            (-6.30, 6.30, -9.80, -7.25),
            # East rack row and parked/showcase objects.
            (3.15, 5.95, -9.25, 1.65),
            (2.60, 6.20, 3.05, 4.75),
            (2.35, 6.20, 7.75, 9.35),
            (-2.75, 0.65, -9.95, -7.15),
            (-2.65, -0.30, 7.05, 8.55),
        ]
        if self.use_dynamic_lane_keepouts:
            keepout_boxes.extend(
                [
                    (-4.40, 3.70, -6.95, -4.60),
                    (-3.55, 3.90, 4.55, 6.95),
                    (-3.25, 3.35, -0.95, 1.70),
                    (-4.75, -2.85, -4.55, -1.05),
                ]
            )
        for min_x, max_x, min_y, max_y in keepout_boxes:
            if min_x - margin <= x <= max_x + margin and min_y - margin <= y <= max_y + margin:
                return False
        return True

    def check_dynamic_lane_keepout(self, x, y, margin=0.0):
        if not self.use_dynamic_lane_keepouts:
            return True

        lane_boxes = [
            (-4.40, 3.70, -6.95, -4.60),
            (-3.55, 3.90, 4.55, 6.95),
            (-3.25, 3.35, -0.95, 1.70),
            (-4.75, -2.85, -4.55, -1.05),
        ]
        for min_x, max_x, min_y, max_y in lane_boxes:
            if min_x - margin <= x <= max_x + margin and min_y - margin <= y <= max_y + margin:
                return False
        return True

    def check_collision(self, laser_scan):
        bumper_collision = self.bumper_subscriber.get_collision()
        self.bumper_subscriber.reset_collision()

        front_min, close_count = self._summarize_collision_scan(laser_scan)

        if bumper_collision:
            self.bumper_collision_streak += 1
        else:
            self.bumper_collision_streak = 0

        bumper_hit = self.bumper_collision_streak >= 2
        scan_hit = (
            front_min <= self.collision_scan_threshold
            and close_count >= self.collision_min_close_count
        )

        if scan_hit or bumper_hit:
            if self.verbose_events:
                print(
                    "Collision detected: "
                    f"front_min={front_min:.3f}, close_count={close_count}, "
                    f"bumper={bumper_collision}, bumper_streak={self.bumper_collision_streak}, done=True",
                    flush=True,
                )
            return True
        return False

    def check_target(self, distance, collision):
        if distance < self.target_reached_delta and not collision:
            if self.verbose_events:
                print(f"Goal Reached! distance={distance:.3f}", flush=True)
            return True
        return False

    def get_dist_sincos(self, odom_position, odom_orientation):
        odom_x = odom_position.x
        odom_y = odom_position.y
        quaternion = Quaternion(
            odom_orientation.w,
            odom_orientation.x,
            odom_orientation.y,
            odom_orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        pose_vector = [np.cos(angle), np.sin(angle)]
        goal_vector = [self.target[0] - odom_x, self.target[1] - odom_y]

        distance = np.linalg.norm(goal_vector)
        cos, sin = self.cossin(pose_vector, goal_vector)

        if not np.isfinite(distance):
            distance = 9.0
        if not np.isfinite(cos):
            cos = 0.0
        if not np.isfinite(sin):
            sin = 0.0
        return distance, cos, sin, angle

    def _normalize_scan(self, latest_scan):
        latest_scan = np.asarray(latest_scan, dtype=np.float32)
        latest_scan = np.nan_to_num(
            latest_scan,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        return np.clip(latest_scan, 0.0, self.scan_max_range) / self.scan_max_range

    def _signed_speed_limit(self, linear):
        if linear >= 0.0:
            return self.max_linear_speed * self.domain_forward_scale
        return self.max_reverse_speed * self.domain_reverse_scale

    def _signed_speed_ratio(self, linear):
        speed_limit = max(self._signed_speed_limit(linear), 1e-6)
        return float(np.clip(linear / speed_limit, -1.0, 1.0))

    def _speed_ratio(self, linear):
        return abs(self._signed_speed_ratio(linear))

    def scale_action(self, raw_linear, raw_steer):
        raw_linear = float(np.clip(raw_linear, -1.0, 1.0))
        raw_steer = float(np.clip(raw_steer, -1.0, 1.0))

        if raw_linear >= 0.0:
            linear = raw_linear * self.max_linear_speed * self.domain_forward_scale
        else:
            linear = raw_linear * self.max_reverse_speed * self.domain_reverse_scale
        linear *= max(0.0, 1.0 - self.domain_rolling_drag * abs(raw_linear))

        speed_ratio = self._speed_ratio(linear)
        low_speed_scale = float(np.clip(abs(linear) / self.min_steer_speed, 0.0, 1.0))
        high_speed_scale = 1.0 - (1.0 - self.high_speed_steer_ratio_min) * speed_ratio
        eff_max_steer = (
            self.max_steer_angle
            * self.domain_steer_scale
            * low_speed_scale
            * high_speed_scale
        )

        target_steer = float(np.clip(raw_steer * eff_max_steer, -eff_max_steer, eff_max_steer))
        base_alpha = 0.25 + 0.15 * (1.0 - speed_ratio)
        alpha = float(np.clip(base_alpha * self.domain_steer_response_scale, 0.05, 0.80))
        steer = self.prev_steer_cmd + alpha * (target_steer - self.prev_steer_cmd)
        steer = float(np.clip(steer, -self.max_steer_angle, self.max_steer_angle))

        if abs(linear) < self.min_motion_speed:
            steer = 0.0
        return float(linear), steer

    def _normalize_executed_action(self, linear, steer):
        norm_linear = self._signed_speed_ratio(linear)
        if self.max_steer_angle > 1e-6:
            norm_steer = float(np.clip(steer / self.max_steer_angle, -1.0, 1.0))
        else:
            norm_steer = 0.0
        return [norm_linear, norm_steer]

    def get_reward(self, goal, collision, action, laser_scan, distance, cos=0.0):
        if goal:
            return 200.0
        if collision:
            return -200.0

        speed, steer = action

        if not np.isfinite(distance):
            distance = self.max_target_dist
        if not np.isfinite(cos):
            cos = 0.0
        if not np.isfinite(speed):
            speed = 0.0
        if not np.isfinite(steer):
            steer = 0.0

        front_min_scan, _ = self._summarize_collision_scan(laser_scan)
        all_min_scan = self._scan_min(laser_scan)

        progress = 0.0 if self.prev_distance is None else float(self.prev_distance - distance)
        if not np.isfinite(progress):
            progress = 0.0
        progress = float(np.clip(progress, -self.progress_clip, self.progress_clip))

        signed_speed_ratio = self._signed_speed_ratio(speed)
        speed_ratio = abs(signed_speed_ratio)
        steer_ratio = float(np.clip(abs(steer) / max(self.max_steer_angle, 1e-6), 0.0, 1.0))
        steer_delta_ratio = float(
            np.clip(
                abs(steer - self.prev_steer_cmd) / max(self.max_steer_angle, 1e-6),
                0.0,
                1.0,
            )
        )

        progress_reward = 140.0 * progress
        motion_alignment_reward = 1.2 * float(np.clip(signed_speed_ratio * cos, -1.0, 1.0))

        danger_dist = 0.75
        front_risk = max(0.0, (danger_dist - front_min_scan) / danger_dist)
        body_risk = max(0.0, (danger_dist - all_min_scan) / danger_dist)
        clearance_penalty = 8.0 * (max(front_risk, 0.6 * body_risk) ** 2)

        curvature_penalty = 0.8 * (speed_ratio ** 1.5) * (steer_ratio ** 2)
        steer_delta_penalty = 0.15 * steer_delta_ratio
        stagnation_penalty = (
            0.25 if speed_ratio < 0.05 and abs(progress) < self.stagnation_progress_eps else 0.0
        )
        time_penalty = 0.10

        reward = (
            progress_reward
            + motion_alignment_reward
            - clearance_penalty
            - curvature_penalty
            - steer_delta_penalty
            - stagnation_penalty
            - time_penalty
        )
        if not np.isfinite(reward):
            print(
                "Warning: non-finite reward detected, fallback to -200.0 "
                f"(distance={distance}, cos={cos}, speed={speed}, steer={steer})",
                flush=True,
            )
            reward = -200.0
        return float(np.clip(reward, -200.0, 200.0))

    def _summarize_collision_scan(self, laser_scan):
        scan = np.asarray(laser_scan, dtype=np.float32)
        scan = np.nan_to_num(
            scan,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        scan = np.clip(scan, 0.0, self.scan_max_range)

        if scan.size == 0:
            return self.scan_max_range, 0

        sector_half_width = max(8, int(scan.size * 0.15))
        center_index = scan.size // 2
        start_index = max(0, center_index - sector_half_width)
        end_index = min(scan.size, center_index + sector_half_width + 1)
        front_scan = scan[start_index:end_index]

        if front_scan.size == 0:
            return self.scan_max_range, 0

        front_min = float(np.min(front_scan))
        close_count = int(np.count_nonzero(front_scan <= self.collision_scan_threshold))
        return front_min, close_count

    def _scan_min(self, laser_scan):
        scan = np.asarray(laser_scan, dtype=np.float32)
        scan = np.nan_to_num(
            scan,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        scan = np.clip(scan, 0.0, self.scan_max_range)
        if scan.size == 0:
            return self.scan_max_range
        return float(np.min(scan))

    @staticmethod
    def cossin(vec1, vec2):
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        if vec1_norm < 1e-8 or vec2_norm < 1e-8:
            return 1.0, 0.0

        vec1 = vec1 / vec1_norm
        vec2 = vec2 / vec2_norm
        cos = np.dot(vec1, vec2)
        sin = np.cross(vec1, vec2).item()
        return cos, sin
