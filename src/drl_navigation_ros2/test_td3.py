import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point
from rclpy.node import Node
from ros_python import ROSEnvironment
from TD3.TD3 import TD3
from utils import record_eval_positions
from visualization_msgs.msg import Marker


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[Config] Invalid int for {name}={value!r}, fallback to {default}", flush=True)
        return default


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    print(f"[Config] Invalid bool for {name}={value!r}, fallback to {default}", flush=True)
    return default


def _format_metrics(prefix, metrics):
    parts = [prefix]
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


class TrajectoryPublisher(Node):
    def __init__(self, topic="td3_test_trajectory", frame_id="odom"):
        super().__init__("td3_test_trajectory_publisher")
        self.frame_id = frame_id
        self.publisher = self.create_publisher(Marker, topic, 10)

    def publish_episode(self, episode_index, points, success):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "td3_eval_trajectory"
        marker.id = int(episode_index)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.color.a = 0.95
        marker.color.r = 0.0 if success else 1.0
        marker.color.g = 0.8 if success else 0.2
        marker.color.b = 0.2
        marker.points = []
        for x, y in points:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.05
            marker.points.append(point)
        self.publisher.publish(marker)
        rclpy.spin_once(self, timeout_sec=0.01)


def _get_robot_xy(env):
    latest_position, _ = env.odom_subscriber.get_latest_odom()
    if latest_position is None:
        latest_position, _ = env.sensor_subscriber.get_latest_sensor()[1:]
    if latest_position is None and env.last_reset_pose is not None:
        latest_position = env.last_reset_pose.position
    if latest_position is None:
        return None
    return float(latest_position.x), float(latest_position.y)


def _write_episode_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "success",
        "collision",
        "timeout",
        "reward",
        "steps",
        "duration_sec",
        "final_distance",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(episode_rows):
    total = max(len(episode_rows), 1)
    success_count = sum(int(row["success"]) for row in episode_rows)
    collision_count = sum(int(row["collision"]) for row in episode_rows)
    timeout_count = sum(int(row["timeout"]) for row in episode_rows)
    avg_reward = float(sum(row["reward"] for row in episode_rows) / total)
    avg_steps = float(sum(row["steps"] for row in episode_rows) / total)
    avg_duration = float(sum(row["duration_sec"] for row in episode_rows) / total)
    avg_final_distance = float(sum(row["final_distance"] for row in episode_rows) / total)
    return {
        "episodes": len(episode_rows),
        "success_rate": success_count / total,
        "collision_rate": collision_count / total,
        "timeout_rate": timeout_count / total,
        "avg_reward": avg_reward,
        "avg_episode_steps": avg_steps,
        "avg_navigation_time_sec": avg_duration,
        "avg_final_distance": avg_final_distance,
    }


def run_evaluation(model, env, scenarios, max_steps, trajectory_publisher=None):
    episode_rows = []

    for episode_index, scenario in enumerate(scenarios, start=1):
        model.reset_observation_history()
        model.reset_action_noise()

        episode_reward = 0.0
        step_count = 0
        collision = False
        goal = False
        trajectory = []

        latest_scan, distance, cos, sin, collision, goal, action, reward = env.eval(scenario=scenario)
        start_time = time.perf_counter()

        initial_xy = _get_robot_xy(env)
        if initial_xy is not None:
            trajectory.append(initial_xy)

        while step_count < max_steps:
            state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, action)
            if terminal:
                break

            next_action = model.get_action(state, add_noise=False)
            next_action = np.clip(next_action, -model.max_action, model.max_action)
            latest_scan, distance, cos, sin, collision, goal, action, reward = env.step(
                lin_velocity=float(next_action[0]),
                ang_velocity=float(next_action[1]),
            )
            if np.isfinite(reward):
                episode_reward += float(reward)
            step_count += 1

            robot_xy = _get_robot_xy(env)
            if robot_xy is not None:
                trajectory.append(robot_xy)

            if collision or goal:
                break

        duration_sec = time.perf_counter() - start_time
        timeout = (not goal) and (not collision) and step_count >= max_steps
        final_distance = float(distance) if np.isfinite(distance) else float("inf")

        if trajectory_publisher is not None and trajectory:
            trajectory_publisher.publish_episode(
                episode_index=episode_index,
                points=trajectory,
                success=bool(goal),
            )

        if hasattr(env, "end_eval_episode"):
            env.end_eval_episode()

        row = {
            "episode": episode_index,
            "success": bool(goal),
            "collision": bool(collision),
            "timeout": bool(timeout),
            "reward": float(episode_reward),
            "steps": int(step_count),
            "duration_sec": float(duration_sec),
            "final_distance": final_distance,
        }
        episode_rows.append(row)
        print(_format_metrics("[Episode]", row), flush=True)

    return episode_rows, _build_summary(episode_rows)


def _shutdown_env(env, trajectory_publisher=None):
    try:
        env.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)
    except Exception:
        pass

    nodes = [
        trajectory_publisher,
        getattr(env, "cmd_vel_publisher", None),
        getattr(env, "scan_subscriber", None),
        getattr(env, "odom_subscriber", None),
        getattr(env, "bumper_subscriber", None),
        getattr(env, "robot_state_publisher", None),
        getattr(env, "world_reset", None),
        getattr(env, "physics_client", None),
        getattr(env, "publish_target", None),
        getattr(env, "sensor_subscriber", None),
    ]
    for node in nodes:
        if node is None:
            continue
        try:
            node.destroy_node()
        except Exception:
            pass

    if rclpy.ok():
        rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="TD3 inference evaluation for navigation.")
    parser.add_argument("--model-dir", default="src/drl_navigation_ros2/models/TD3")
    parser.add_argument("--model-name", default="TD3_best")
    parser.add_argument("--ros-domain-id", type=int, default=_env_int("ROS_DOMAIN_ID", 99))
    parser.add_argument("--episodes", type=int, default=_env_int("NR_EVAL_EPISODES", 10))
    parser.add_argument("--max-steps", type=int, default=_env_int("EVAL_MAX_STEPS", 250))
    parser.add_argument("--scan-bins", type=int, default=_env_int("SCAN_BINS", 50))
    parser.add_argument("--frame-stack", type=int, default=_env_int("FRAME_STACK", 3))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--results-dir", default="src/drl_navigation_ros2/models/TD3/test_results")
    parser.add_argument(
        "--randomize-eval",
        action="store_true",
        default=_env_bool("RANDOMIZE_EVAL", False),
        help="Enable eval-time domain randomization/noise if environment supports it.",
    )
    parser.add_argument(
        "--publish-trajectory",
        action="store_true",
        default=True,
        help="Publish each episode trajectory to RViz as LINE_STRIP markers.",
    )
    parser.add_argument(
        "--no-publish-trajectory",
        action="store_false",
        dest="publish_trajectory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    os.environ["RANDOMIZE_EVAL"] = "1" if args.randomize_eval else "0"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    kinematic_frame_dim = 5
    state_dim = args.frame_stack * (args.scan_bins + kinematic_frame_dim)
    print(
        _format_metrics(
            "[Config]",
            {
                "device": str(device),
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "scan_bins": args.scan_bins,
                "frame_stack": args.frame_stack,
                "ros_domain_id": args.ros_domain_id,
                "model_dir": args.model_dir,
                "model_name": args.model_name,
                "randomize_eval": int(args.randomize_eval),
            },
        ),
        flush=True,
    )

    model = TD3(
        state_dim=state_dim,
        action_dim=2,
        max_action=1,
        device=device,
        load_model=True,
        model_name=args.model_name,
        load_directory=Path(args.model_dir),
        use_writer=False,
        scan_bins=args.scan_bins,
        frame_stack=args.frame_stack,
    )
    model.actor.eval()
    model.actor_target.eval()
    model.critic.eval()
    model.critic_target.eval()

    env = ROSEnvironment()
    trajectory_publisher = TrajectoryPublisher() if args.publish_trajectory else None

    try:
        scenarios = record_eval_positions(n_eval_scenarios=args.episodes)
        episode_rows, summary = run_evaluation(
            model=model,
            env=env,
            scenarios=scenarios,
            max_steps=args.max_steps,
            trajectory_publisher=trajectory_publisher,
        )

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"td3_eval_{timestamp}.csv"
        json_path = results_dir / f"td3_eval_{timestamp}.json"

        _write_episode_csv(csv_path, episode_rows)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": {
                        "model_dir": str(args.model_dir),
                        "model_name": args.model_name,
                        "episodes": args.episodes,
                        "max_steps": args.max_steps,
                        "scan_bins": args.scan_bins,
                        "frame_stack": args.frame_stack,
                        "ros_domain_id": args.ros_domain_id,
                        "seed": args.seed,
                        "device": str(device),
                        "randomize_eval": bool(args.randomize_eval),
                    },
                    "summary": summary,
                    "episodes": episode_rows,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print(_format_metrics("[Summary]", summary), flush=True)
        print(f"[Output] episode_csv={csv_path}", flush=True)
        print(f"[Output] summary_json={json_path}", flush=True)
    finally:
        _shutdown_env(env, trajectory_publisher=trajectory_publisher)


if __name__ == "__main__":
    main()
