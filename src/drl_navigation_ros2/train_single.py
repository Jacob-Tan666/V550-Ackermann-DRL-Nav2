import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from TD3.TD3 import TD3
from replay_buffer import ReplayBuffer
from ros_python import ROS_env
from utils import record_eval_positions


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[Config] Invalid int for {name}={value!r}, fallback to {default}", flush=True)
        return default


def _env_float_pair(name, default):
    value = os.environ.get(name)
    if value is None:
        return tuple(float(v) for v in default)
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        print(f"[Config] Invalid pair for {name}={value!r}, fallback to {default}", flush=True)
        return tuple(float(v) for v in default)
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        print(f"[Config] Invalid pair for {name}={value!r}, fallback to {default}", flush=True)
        return tuple(float(v) for v in default)


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


def _recent_episode_metrics(episodes):
    if not episodes:
        return {
            "episode_count": 0,
            "goal_rate": 0.0,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
        }

    total = len(episodes)
    goals = sum(int(ep["goal"]) for ep in episodes)
    collisions = sum(int(ep["collision"]) for ep in episodes)
    timeouts = sum(int(ep["timeout"]) for ep in episodes)
    avg_reward = float(sum(ep["reward"] for ep in episodes) / total)
    avg_steps = float(sum(ep["steps"] for ep in episodes) / total)
    return {
        "episode_count": total,
        "goal_rate": goals / total,
        "collision_rate": collisions / total,
        "timeout_rate": timeouts / total,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
    }


def _exploration_random_prob(current_step, start_timesteps, prob_start, prob_end, decay_steps):
    if current_step < start_timesteps:
        return 1.0
    if decay_steps <= 0:
        return prob_end
    progress = min(1.0, max(0.0, (current_step - start_timesteps) / decay_steps))
    return prob_start + progress * (prob_end - prob_start)


def eval_fn(model, env, scenarios, epoch, max_steps):
    """Run periodic evaluation in the same single environment process."""
    print("..............................................", flush=True)
    print(f"Epoch {epoch}. Evaluating {len(scenarios)} scenarios", flush=True)
    max_action = float(getattr(model, "max_action", 1.0))

    total_reward = 0.0
    col = 0
    gl = 0
    total_steps = 0

    for scenario in scenarios:
        scenario_reward = 0.0
        count = 0
        model.reset_observation_history()
        model.reset_action_noise()
        latest_scan, distance, cos, sin, collision, goal, action, reward = env.eval(
            scenario=scenario
        )
        scenario_collision = bool(collision)
        scenario_goal = bool(goal)

        while count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, action
            )
            if terminal:
                break

            next_action = model.get_action(state, False)
            next_action[0] = np.clip(next_action[0], -max_action, max_action)
            next_action[1] = np.clip(next_action[1], -max_action, max_action)
            latest_scan, distance, cos, sin, collision, goal, action, reward = env.step(
                lin_velocity=next_action[0], ang_velocity=next_action[1]
            )

            if np.isfinite(reward):
                scenario_reward += reward
            else:
                print("Warning: non-finite reward in eval step, skip this sample")

            count += 1
            scenario_collision = scenario_collision or bool(collision)
            scenario_goal = scenario_goal or bool(goal)

        col += int(scenario_collision)
        gl += int(scenario_goal)
        total_steps += count
        total_reward += scenario_reward
        if hasattr(env, "end_eval_episode"):
            env.end_eval_episode()

    avg_reward = total_reward / max(len(scenarios), 1)
    avg_col = col / max(len(scenarios), 1)
    avg_goal = gl / max(len(scenarios), 1)
    avg_steps = total_steps / max(len(scenarios), 1)

    summary = {
        "epoch": epoch,
        "avg_reward": avg_reward,
        "avg_collision_rate": avg_col,
        "avg_goal_rate": avg_goal,
        "avg_episode_steps": avg_steps,
    }
    print(_format_metrics("[Eval]", summary), flush=True)
    print("..............................................", flush=True)

    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)
    model.writer.add_scalar("eval/avg_steps", avg_steps, epoch)
    return summary


def main():
    """Single-process TD3 training entrypoint."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    scan_bins = _env_int("SCAN_BINS", 50)
    frame_stack = _env_int("FRAME_STACK", 3)
    kinematic_frame_dim = 5
    state_dim = frame_stack * (scan_bins + kinematic_frame_dim)
    ou_theta = _env_float_pair("OU_THETA", (0.28, 0.45))
    ou_sigma_start = _env_float_pair("OU_SIGMA_START", (0.10, 0.18))
    ou_sigma_end = _env_float_pair("OU_SIGMA_END", (0.02, 0.06))
    ou_noise_clip = _env_float_pair("OU_NOISE_CLIP", (0.25, 0.35))
    ou_noise_decay_steps = _env_int("OU_NOISE_DECAY_STEPS", 200000)

    args = {
        "state_dim": state_dim,
        "action_dim": 2,
        "max_action": 1,
        "scan_bins": scan_bins,
        "frame_stack": frame_stack,
        "kinematic_frame_dim": kinematic_frame_dim,
        "nr_eval_episodes": _env_int("NR_EVAL_EPISODES", 10),
        "max_steps": _env_int("MAX_STEPS", 200),
        "eval_max_steps": _env_int("EVAL_MAX_STEPS", 250),
        "episodes_per_epoch": _env_int("EPISODES_PER_EPOCH", 70),
    }

    max_epochs = _env_int("MAX_EPOCHS", 100)
    start_timesteps = _env_int("START_TIMESTEPS", 20000)
    batch_size = _env_int("BATCH_SIZE", 256)
    training_iterations = _env_int("TRAINING_ITERATIONS", 2)
    save_every = _env_int("SAVE_EVERY", 100)
    status_every_episodes = _env_int("SINGLE_STATUS_EVERY_EPISODES", 10)
    discount = float(os.environ.get("DISCOUNT", 0.995))
    tau = float(os.environ.get("TAU", 0.003))
    policy_noise = float(os.environ.get("POLICY_NOISE", 0.08))
    noise_clip = float(os.environ.get("NOISE_CLIP", 0.20))
    policy_freq = _env_int("POLICY_FREQ", 2)
    timeout_penalty = float(os.environ.get("TIMEOUT_PENALTY", -50.0))
    explore_random_prob_start = float(os.environ.get("EXPLORE_RANDOM_PROB_START", 0.35))
    explore_random_prob_end = float(os.environ.get("EXPLORE_RANDOM_PROB_END", 0.05))
    explore_random_decay_steps = _env_int("EXPLORE_RANDOM_DECAY_STEPS", 200000)
    resume_model = _env_bool("RESUME_MODEL", False)
    model_name = os.environ.get("MODEL_NAME", "TD3")
    model_dir = Path(os.environ.get("MODEL_DIR", "src/drl_navigation_ros2/models/TD3"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        _format_metrics(
            "[Config]",
            {
                "device": str(device),
                "batch_size": batch_size,
                "training_iterations": training_iterations,
                "state_dim": args["state_dim"],
                "scan_bins": scan_bins,
                "frame_stack": frame_stack,
                "discount": discount,
                "tau": tau,
                "policy_noise": policy_noise,
                "noise_clip": noise_clip,
                "ou_theta_linear": ou_theta[0],
                "ou_theta_steer": ou_theta[1],
                "ou_sigma_start_linear": ou_sigma_start[0],
                "ou_sigma_start_steer": ou_sigma_start[1],
                "ou_sigma_end_linear": ou_sigma_end[0],
                "ou_sigma_end_steer": ou_sigma_end[1],
                "ou_noise_decay_steps": ou_noise_decay_steps,
                "policy_freq": policy_freq,
                "timeout_penalty": timeout_penalty,
                "start_timesteps": start_timesteps,
                "max_steps": args["max_steps"],
                "eval_max_steps": args["eval_max_steps"],
                "episodes_per_epoch": args["episodes_per_epoch"],
                "status_every_episodes": status_every_episodes,
                "resume_model": int(resume_model),
                "model_name": model_name,
                "model_dir": str(model_dir),
            },
        ),
        flush=True,
    )

    print("[Startup] Initializing TD3 model", flush=True)
    model = TD3(
        state_dim=args["state_dim"],
        action_dim=args["action_dim"],
        max_action=args["max_action"],
        device=device,
        save_every=save_every,
        load_model=resume_model,
        model_name=model_name,
        save_directory=model_dir,
        load_directory=model_dir,
        scan_bins=args["scan_bins"],
        frame_stack=args["frame_stack"],
        kinematic_frame_dim=args["kinematic_frame_dim"],
        exploration_noise_theta=ou_theta,
        exploration_noise_sigma_start=ou_sigma_start,
        exploration_noise_sigma_end=ou_sigma_end,
        exploration_noise_decay_steps=ou_noise_decay_steps,
        exploration_noise_clip=ou_noise_clip,
    )

    if resume_model:
        restored_env_steps = int(getattr(model, "checkpoint_metadata", {}).get("env_steps", 0))
        print(
            _format_metrics(
                "[Resume]",
                {
                    "model_name": model_name,
                    "model_dir": str(model_dir),
                    "restored_env_steps": restored_env_steps,
                    "restored_train_iters": int(getattr(model, "iter_count", 0)),
                },
            ),
            flush=True,
        )

    print("[Startup] TD3 model initialized", flush=True)
    replay_buffer = ReplayBuffer(buffer_size=100_000, random_seed=42)
    print("[Startup] Initializing ROS environment", flush=True)
    ros = ROS_env()
    print("[Startup] ROS environment initialized", flush=True)
    eval_scenarios = record_eval_positions(n_eval_scenarios=args["nr_eval_episodes"])
    print(f"[Startup] Prepared {len(eval_scenarios)} eval scenarios", flush=True)

    global_steps = restored_env_steps if resume_model else 0
    episode = 0
    epochs = 0
    steps = 0
    total_episodes = 0
    episode_reward = 0.0
    recent_episodes = deque(maxlen=100)
    status_window_start = time.time()
    steps_since_last_status = 0

    print("[Startup] Running initial environment reset", flush=True)
    latest_scan, distance, cos, sin, collision, goal, action, reward = ros.reset()
    print("[Startup] Initial environment reset complete", flush=True)
    model.reset_observation_history()
    model.reset_action_noise()

    try:
        while epochs < max_epochs:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, action
            )
            if terminal:
                terminal_goal = bool(goal)
                terminal_collision = bool(collision)
                completed_reward = episode_reward
                ros.update_curriculum(goal=terminal_goal, collision=terminal_collision, timeout=False)
                latest_scan, distance, cos, sin, collision, goal, action, reward = ros.reset()
                model.reset_observation_history()
                model.reset_action_noise()
                steps = 0
                episode += 1
                total_episodes += 1
                recent_episodes.append(
                    {
                        "reward": completed_reward,
                        "steps": steps,
                        "goal": terminal_goal,
                        "collision": terminal_collision,
                        "timeout": False,
                    }
                )
                episode_reward = 0.0
                continue

            explore_random_prob = _exploration_random_prob(
                global_steps,
                start_timesteps,
                explore_random_prob_start,
                explore_random_prob_end,
                explore_random_decay_steps,
            )
            if global_steps < start_timesteps or np.random.rand() < explore_random_prob:
                next_action = np.random.uniform(
                    low=[-args["max_action"], -args["max_action"]],
                    high=[args["max_action"], args["max_action"]],
                    size=2,
                ).astype(np.float32)
            else:
                next_action = model.get_action(state, True, step=global_steps)
                next_action[0] = np.clip(next_action[0], -args["max_action"], args["max_action"])
                next_action[1] = np.clip(next_action[1], -args["max_action"], args["max_action"])

            latest_scan, distance, cos, sin, collision, goal, action, reward = ros.step(
                lin_velocity=next_action[0], ang_velocity=next_action[1]
            )
            next_state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, action
            )
            timeout = bool((steps + 1) >= args["max_steps"] and not terminal)
            transition_terminal = int(bool(terminal) or timeout)
            transition_reward = float(reward) + (timeout_penalty if timeout else 0.0)

            replay_buffer.add(state, next_action, transition_reward, transition_terminal, next_state)
            global_steps += 1
            steps += 1
            episode_reward += transition_reward if np.isfinite(transition_reward) else -200.0

            if global_steps >= start_timesteps and replay_buffer.size() > batch_size:
                model.checkpoint_metadata = {
                    "env_steps": int(global_steps),
                }
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                    discount=discount,
                    tau=tau,
                    policy_noise=policy_noise,
                    noise_clip=noise_clip,
                    policy_freq=policy_freq,
                    priority_step=global_steps,
                )

            if transition_terminal or steps >= args["max_steps"]:
                completed_steps = steps
                completed_reward = episode_reward
                completed_goal = bool(goal)
                completed_collision = bool(collision)
                total_episodes += 1
                steps_since_last_status += completed_steps
                ros.update_curriculum(goal=completed_goal, collision=completed_collision, timeout=timeout)
                latest_scan, distance, cos, sin, collision, goal, action, reward = ros.reset()
                model.reset_observation_history()
                model.reset_action_noise()
                episode += 1
                steps = 0
                episode_reward = 0.0
                recent_episodes.append(
                    {
                        "reward": completed_reward,
                        "steps": completed_steps,
                        "goal": completed_goal,
                        "collision": completed_collision,
                        "timeout": timeout,
                    }
                )
                if total_episodes % max(status_every_episodes, 1) == 0:
                    recent_metrics = _recent_episode_metrics(recent_episodes)
                    elapsed = max(time.time() - status_window_start, 1e-6)
                    status_window_start = time.time()
                    status_metrics = {
                        "total_episodes": total_episodes,
                        "global_steps": global_steps,
                        "replay_size": replay_buffer.size(),
                        "per_invalid_fallbacks": getattr(replay_buffer, "invalid_priority_events", 0),
                        "explore_random_prob": _exploration_random_prob(
                            global_steps,
                            start_timesteps,
                            explore_random_prob_start,
                            explore_random_prob_end,
                            explore_random_decay_steps,
                        ),
                        "recent_goal_rate": recent_metrics["goal_rate"],
                        "recent_collision_rate": recent_metrics["collision_rate"],
                        "recent_timeout_rate": recent_metrics["timeout_rate"],
                        "recent_episode_reward": recent_metrics["avg_reward"],
                        "recent_episode_steps": recent_metrics["avg_steps"],
                        "ou_sigma_linear": model.get_exploration_noise_sigma(global_steps)["linear"],
                        "ou_sigma_steer": model.get_exploration_noise_sigma(global_steps)["steer"],
                        "steps_per_sec": steps_since_last_status / elapsed,
                    }
                    print(_format_metrics("[Status]", status_metrics), flush=True)
                    model.writer.add_scalar("train_status/recent_goal_rate", recent_metrics["goal_rate"], global_steps)
                    model.writer.add_scalar("train_status/recent_collision_rate", recent_metrics["collision_rate"], global_steps)
                    model.writer.add_scalar("train_status/recent_timeout_rate", recent_metrics["timeout_rate"], global_steps)
                    model.writer.add_scalar("train_status/recent_episode_reward", recent_metrics["avg_reward"], global_steps)
                    model.writer.add_scalar("train_status/recent_episode_steps", recent_metrics["avg_steps"], global_steps)
                    steps_since_last_status = 0

            if episode > 0 and episode % args["episodes_per_epoch"] == 0:
                epochs += 1
                episode = 0
                eval_summary = eval_fn(
                    model=model,
                    env=ros,
                    scenarios=eval_scenarios,
                    epoch=epochs,
                    max_steps=args["eval_max_steps"],
                )
                print(_format_metrics("[EvalSummary]", eval_summary), flush=True)
                latest_scan, distance, cos, sin, collision, goal, action, reward = ros.reset()
                model.reset_observation_history()
                model.reset_action_noise()

    except KeyboardInterrupt:
        print("KeyboardInterrupt. Stopping single-process training...", flush=True)


if __name__ == "__main__":
    main()
