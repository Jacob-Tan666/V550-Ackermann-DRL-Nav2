import os
import time
from collections import deque
from pathlib import Path
from queue import Empty, Full

import numpy as np
import torch
import torch.multiprocessing as mp
from replay_buffer import ReplayBuffer
from ros_python import ROSEnvironment
from TD3.TD3 import TD3
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


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[Config] Invalid float for {name}={value!r}, fallback to {default}", flush=True)
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


def _safe_queue_size(queue_obj):
    try:
        return queue_obj.qsize()
    except (NotImplementedError, AttributeError, OSError):
        return -1


def _format_metrics(prefix, metrics):
    parts = [prefix]
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _is_better_eval(current_best, candidate):
    if current_best is None:
        return True
    candidate_goal = float(candidate.get("avg_goal_rate", 0.0))
    best_goal = float(current_best.get("avg_goal_rate", 0.0))
    if candidate_goal > best_goal + 1e-9:
        return True
    if candidate_goal < best_goal - 1e-9:
        return False

    candidate_reward = float(candidate.get("avg_reward", float("-inf")))
    best_reward = float(current_best.get("avg_reward", float("-inf")))
    if candidate_reward > best_reward + 1e-9:
        return True
    if candidate_reward < best_reward - 1e-9:
        return False

    candidate_collision = float(candidate.get("avg_collision_rate", 1.0))
    best_collision = float(current_best.get("avg_collision_rate", 1.0))
    if candidate_collision < best_collision - 1e-9:
        return True
    if candidate_collision > best_collision + 1e-9:
        return False

    candidate_steps = float(candidate.get("avg_episode_steps", float("inf")))
    best_steps = float(current_best.get("avg_episode_steps", float("inf")))
    return candidate_steps < best_steps - 1e-9


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


def _emit_episode_summary(
    episode_queue,
    worker_id,
    episode_index,
    steps,
    collision,
    goal,
    reward,
    timeout,
):
    try:
        episode_queue.put_nowait(
            {
                "worker_id": int(worker_id),
                "episode_index": int(episode_index),
                "steps": int(steps),
                "collision": bool(collision),
                "goal": bool(goal),
                "reward": float(reward),
                "timeout": bool(timeout),
            }
        )
    except Full:
        pass


def _drain_episode_summaries(episode_queue, recent_episodes, totals):
    drained = 0
    while True:
        try:
            episode = episode_queue.get_nowait()
        except Empty:
            break
        recent_episodes.append(episode)
        totals["episodes"] += 1
        totals["goals"] += int(episode["goal"])
        totals["collisions"] += int(episode["collision"])
        totals["timeouts"] += int(episode["timeout"])
        drained += 1
    return drained


# ============================================================
# Queue helpers
# ============================================================


def _drain_latest_weight(weight_queue):
    """Return only the newest weights so workers never apply stale updates."""
    latest_state_dict = None
    while True:
        try:
            latest_state_dict = weight_queue.get_nowait()
        except Empty:
            break
    return latest_state_dict


def _push_latest_weights(weight_queues, actor_state_dict):
    """Broadcast the newest actor weights, keeping one update per worker."""
    for weight_queue in weight_queues:
        while True:
            try:
                weight_queue.get_nowait()
            except Empty:
                break
        try:
            weight_queue.put_nowait(actor_state_dict)
        except Full:
            pass


# ============================================================
# Evaluation runs only in the learner process.
# ============================================================


def eval_fn(model, env, scenarios, epoch, max_steps):
    """Evaluate centrally without changing the sampling workers."""
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
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(scenario=scenario)
        scenario_collision = bool(collision)
        scenario_goal = bool(goal)
        while count < max_steps:
            state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
            if terminal:
                break
            action = model.get_action(state, False)  # Evaluation is deterministic.
            action[0] = np.clip(action[0], -max_action, max_action)
            action[1] = np.clip(action[1], -max_action, max_action)
            a_in = [action[0], action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
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

    if model.writer is not None:
        model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
        model.writer.add_scalar("eval/avg_col", avg_col, epoch)
        model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)
        model.writer.add_scalar("eval/avg_steps", avg_steps, epoch)
    return summary


# ============================================================
# Workers interact with their environments and collect transitions.
# ============================================================


def env_worker(worker_id, args, transition_queue, episode_queue, weight_queue, global_step_counter, stop_event):
    """Collect transitions in an isolated environment with GPU inference."""
    # Avoid CPU contention between PyTorch worker processes.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # Isolate ROS graphs and random generators per worker.
    os.environ["ROS_DOMAIN_ID"] = str(worker_id + 1)
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)

    print(
        f"[Worker {worker_id}] Starting (ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']})",
        flush=True,
    )

    # Use CUDA for worker inference when it is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Disable worker-side logging and checkpoint writes.
    model = TD3(
        state_dim=args["state_dim"],
        action_dim=args["action_dim"],
        max_action=args["max_action"],
        device=device,
        save_every=0,
        load_model=False,
        use_writer=False,
        scan_bins=args["scan_bins"],
        frame_stack=args["frame_stack"],
        kinematic_frame_dim=args["kinematic_frame_dim"],
        exploration_noise_theta=args["ou_theta"],
        exploration_noise_sigma_start=args["ou_sigma_start"],
        exploration_noise_sigma_end=args["ou_sigma_end"],
        exploration_noise_decay_steps=args["ou_noise_decay_steps"],
        exploration_noise_clip=args["ou_noise_clip"],
    )
    model.actor.eval()

    ros = ROSEnvironment()

    max_steps = args["max_steps"]
    start_timesteps = args["start_timesteps"]
    max_total_steps = args["max_total_steps"]
    timeout_penalty = float(args["timeout_penalty"])
    explore_random_prob_start = float(args["explore_random_prob_start"])
    explore_random_prob_end = float(args["explore_random_prob_end"])
    explore_random_decay_steps = int(args["explore_random_decay_steps"])

    steps_in_episode = 0
    episode_in_worker = 0
    episode_reward = 0.0

    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
    model.reset_observation_history()
    model.reset_action_noise()

    while not stop_event.is_set():
        # Pull the newest actor weights without blocking sampling.
        state_dict = _drain_latest_weight(weight_queue)
        if state_dict is not None:
            # Move the learner's CPU tensors to this worker's device.
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
            model.actor.load_state_dict(state_dict)

        # The learner-owned global step selects warm-up or policy sampling.
        current_global_step = global_step_counter.value

        state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
        if terminal:
            ros.update_curriculum(goal=goal, collision=collision, timeout=False)
            _emit_episode_summary(
                episode_queue=episode_queue,
                worker_id=worker_id,
                episode_index=episode_in_worker + 1,
                steps=steps_in_episode,
                collision=collision,
                goal=goal,
                reward=episode_reward,
                timeout=False,
            )
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            model.reset_observation_history()
            model.reset_action_noise()
            steps_in_episode = 0
            episode_in_worker += 1
            episode_reward = 0.0
            continue

        explore_random_prob = _exploration_random_prob(
            current_global_step,
            start_timesteps,
            explore_random_prob_start,
            explore_random_prob_end,
            explore_random_decay_steps,
        )
        if current_global_step < start_timesteps or np.random.rand() < explore_random_prob:
            # TD3 warm-up uses uniformly random actions.
            action = np.random.uniform(
                low=[-args["max_action"], -args["max_action"]],
                high=[args["max_action"], args["max_action"]],
                size=args["action_dim"],
            ).astype(np.float32)
        else:
            with torch.no_grad():
                action = model.get_action(state, True, step=current_global_step)
            action[0] = np.clip(action[0], -args["max_action"], args["max_action"])
            action[1] = np.clip(action[1], -args["max_action"], args["max_action"])

        a_in = [float(action[0]), float(action[1])]
        latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        next_state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
        timeout = bool((steps_in_episode + 1) >= max_steps and not terminal)
        transition_terminal = int(bool(terminal) or timeout)
        transition_reward = float(reward) + (timeout_penalty if timeout else 0.0)

        # Send the transition to the learner.
        try:
            transition_queue.put(
                (state, action, transition_reward, transition_terminal, next_state),
                timeout=5.0,
            )
        except Full:
            # Drop the oldest queued transition if the learner falls behind.
            pass

        steps_in_episode += 1
        episode_reward += transition_reward if np.isfinite(transition_reward) else -200.0

        if transition_terminal or steps_in_episode >= max_steps:
            timed_out = timeout
            ros.update_curriculum(goal=goal, collision=collision, timeout=timed_out)
            _emit_episode_summary(
                episode_queue=episode_queue,
                worker_id=worker_id,
                episode_index=episode_in_worker + 1,
                steps=steps_in_episode,
                collision=collision,
                goal=goal,
                reward=episode_reward,
                timeout=timed_out,
            )
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            model.reset_observation_history()
            model.reset_action_noise()
            episode_in_worker += 1
            steps_in_episode = 0
            episode_reward = 0.0

        # Stop when the learner reaches the global training budget.
        if current_global_step >= max_total_steps:
            break

    print(f"[Worker {worker_id}] Exiting. Local episodes: {episode_in_worker}", flush=True)


# ============================================================
# Learner process: train, evaluate, and broadcast weights.
# ============================================================


def main():
    mp.set_start_method("spawn", force=True)  # Keep CUDA contexts isolated.
    torch.set_float32_matmul_precision("high")

    # Hyperparameters
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
        "start_timesteps": _env_int("START_TIMESTEPS", 20000),
        # Count the training budget in transitions accepted by the learner.
        "max_total_steps": _env_int("MAX_TOTAL_STEPS", 1_000_000),
    }

    cpu_count = os.cpu_count() or 1
    default_workers = max(1, 16)
    num_workers = _env_int("NUM_WORKERS", default_workers)
    batch_size = _env_int("BATCH_SIZE", 1024)
    train_utd = _env_float("TRAIN_UTD", 1.0)  # Update-to-Data ratio
    eval_every_steps = _env_int("EVAL_EVERY_STEPS", 20000)
    save_every = _env_int("SAVE_EVERY", 100)
    discount = _env_float("DISCOUNT", 0.995)
    tau = _env_float("TAU", 0.003)
    policy_noise = _env_float("POLICY_NOISE", 0.08)
    noise_clip = _env_float("NOISE_CLIP", 0.20)
    policy_freq = _env_int("POLICY_FREQ", 2)
    timeout_penalty = _env_float("TIMEOUT_PENALTY", -50.0)
    explore_random_prob_start = _env_float("EXPLORE_RANDOM_PROB_START", 0.35)
    explore_random_prob_end = _env_float("EXPLORE_RANDOM_PROB_END", 0.05)
    explore_random_decay_steps = _env_int("EXPLORE_RANDOM_DECAY_STEPS", 200000)
    drain_multiplier = _env_int("DRAIN_MULTIPLIER", 8)
    min_drain_limit = _env_int("MIN_DRAIN_LIMIT", 256)
    weight_sync_every_steps = _env_int("WEIGHT_SYNC_EVERY_STEPS", 500)
    transition_queue_size = _env_int(
        "TRANSITION_QUEUE_SIZE",
        max(20000, num_workers * 1500),
    )
    episode_queue_size = _env_int(
        "EPISODE_QUEUE_SIZE",
        max(4000, num_workers * 256),
    )
    status_every_sec = _env_float("STATUS_EVERY_SEC", 30.0)
    resume_model = _env_bool("RESUME_MODEL", False)
    model_name = os.environ.get("MODEL_NAME", "TD3")
    model_dir = Path(os.environ.get("MODEL_DIR", "src/drl_navigation_ros2/models/TD3"))

    args["discount"] = discount
    args["tau"] = tau
    args["policy_noise"] = policy_noise
    args["noise_clip"] = noise_clip
    args["policy_freq"] = policy_freq
    args["timeout_penalty"] = timeout_penalty
    args["explore_random_prob_start"] = explore_random_prob_start
    args["explore_random_prob_end"] = explore_random_prob_end
    args["explore_random_decay_steps"] = explore_random_decay_steps
    args["ou_theta"] = ou_theta
    args["ou_sigma_start"] = ou_sigma_start
    args["ou_sigma_end"] = ou_sigma_end
    args["ou_noise_decay_steps"] = ou_noise_decay_steps
    args["ou_noise_clip"] = ou_noise_clip

    min_replay_size = max(args["start_timesteps"], batch_size)
    drain_limit = max(num_workers * drain_multiplier, min_drain_limit)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        _format_metrics(
            "[Config]",
            {
                "device": str(device),
                "cpu_count": cpu_count,
                "num_workers": num_workers,
                "batch_size": batch_size,
                "train_utd": train_utd,
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
                "start_timesteps": args["start_timesteps"],
                "train_max_steps": args["max_steps"],
                "eval_max_steps": args["eval_max_steps"],
                "max_total_steps": args["max_total_steps"],
                "drain_limit": drain_limit,
                "transition_queue_size": transition_queue_size,
                "weight_sync_every_steps": weight_sync_every_steps,
                "eval_every_steps": eval_every_steps,
                "status_every_sec": status_every_sec,
                "resume_model": int(resume_model),
                "model_name": model_name,
                "model_dir": str(model_dir),
            },
        ),
        flush=True,
    )

    # Only the learner owns a TensorBoard writer.
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
        use_writer=True,
        scan_bins=args["scan_bins"],
        frame_stack=args["frame_stack"],
        kinematic_frame_dim=args["kinematic_frame_dim"],
        exploration_noise_theta=ou_theta,
        exploration_noise_sigma_start=ou_sigma_start,
        exploration_noise_sigma_end=ou_sigma_end,
        exploration_noise_decay_steps=ou_noise_decay_steps,
        exploration_noise_clip=ou_noise_clip,
    )

    # Size the replay buffer for parallel sampling.
    replay_buffer = ReplayBuffer(buffer_size=1_000_000, random_seed=42)

    # The learner owns an isolated evaluation environment.
    os.environ["ROS_DOMAIN_ID"] = "99"
    eval_env = ROSEnvironment()
    eval_scenarios = record_eval_positions(n_eval_scenarios=args["nr_eval_episodes"])

    # Inter-process communication
    transition_queue = mp.Queue(maxsize=transition_queue_size)
    episode_queue = mp.Queue(maxsize=episode_queue_size)
    weight_queues = [mp.Queue(maxsize=1) for _ in range(num_workers)]
    # Workers read this counter; only the learner updates it.
    global_step_counter = mp.Value("l", 0)
    stop_event = mp.Event()

    # Broadcast the initial actor before sampling starts.
    initial_actor_state = {k: v.detach().cpu() for k, v in model.actor.state_dict().items()}
    _push_latest_weights(weight_queues, initial_actor_state)

    # Start workers.
    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=env_worker,
            args=(i, args, transition_queue, episode_queue, weight_queues[i], global_step_counter, stop_event),
        )
        p.start()
        workers.append(p)

    total_env_steps = 0
    total_train_iters = 0
    last_eval_step = 0
    last_weight_sync_step = 0
    epoch = 0
    best_eval_summary = None
    recent_episodes = deque(maxlen=100)
    total_episode_stats = {"episodes": 0, "goals": 0, "collisions": 0, "timeouts": 0}
    last_status_time = time.time()
    last_status_env_steps = 0
    last_status_train_iters = 0

    if resume_model:
        total_env_steps = int(getattr(model, "checkpoint_metadata", {}).get("env_steps", 0))
        total_train_iters = int(getattr(model, "iter_count", 0))
        last_eval_step = total_env_steps
        last_weight_sync_step = total_env_steps
        last_status_env_steps = total_env_steps
        last_status_train_iters = total_train_iters
        with global_step_counter.get_lock():
            global_step_counter.value = total_env_steps
        print(
            _format_metrics(
                "[Resume]",
                {
                    "model_name": model_name,
                    "model_dir": str(model_dir),
                    "restored_env_steps": total_env_steps,
                    "restored_train_iters": total_train_iters,
                },
            ),
            flush=True,
        )

    try:
        while total_env_steps < args["max_total_steps"]:
            _drain_episode_summaries(episode_queue, recent_episodes, total_episode_stats)
            # Collect a transition batch.
            collected = 0
            timeout_val = 1.0
            while collected < drain_limit:
                try:
                    transition = transition_queue.get(timeout=timeout_val)
                except Empty:
                    break
                state, action, reward, terminal, next_state = transition
                replay_buffer.add(state, action, reward, terminal, next_state)
                collected += 1
                timeout_val = 0.0

            if collected == 0:
                _drain_episode_summaries(episode_queue, recent_episodes, total_episode_stats)
                if not any(w.is_alive() for w in workers):
                    print("[Learner] All workers have exited.", flush=True)
                    break
                continue

            total_env_steps += collected
            with global_step_counter.get_lock():
                global_step_counter.value = total_env_steps

            # Train.
            if replay_buffer.size() >= min_replay_size:
                # Preserve the configured update-to-data ratio.
                train_iterations = max(1, int(np.ceil(collected * train_utd)))
                model.checkpoint_metadata = {
                    "env_steps": int(total_env_steps),
                }
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=train_iterations,
                    batch_size=batch_size,
                    discount=discount,
                    tau=tau,
                    policy_noise=policy_noise,
                    noise_clip=noise_clip,
                    policy_freq=policy_freq,
                    priority_step=total_env_steps,
                )
                total_train_iters += train_iterations

            # Periodically broadcast actor weights.
            if total_env_steps - last_weight_sync_step >= weight_sync_every_steps:
                actor_state_dict = {k: v.detach().cpu() for k, v in model.actor.state_dict().items()}
                _push_latest_weights(weight_queues, actor_state_dict)
                last_weight_sync_step = total_env_steps

            # Periodically evaluate in the learner.
            if total_env_steps - last_eval_step >= eval_every_steps:
                epoch += 1
                eval_summary = eval_fn(
                    model=model,
                    env=eval_env,
                    scenarios=eval_scenarios,
                    epoch=epoch,
                    max_steps=args["eval_max_steps"],
                )
                if _is_better_eval(best_eval_summary, eval_summary):
                    best_eval_summary = dict(eval_summary)
                    model.checkpoint_metadata = {
                        "env_steps": int(total_env_steps),
                        "best_eval_summary": dict(best_eval_summary),
                        "best_eval_step": int(total_env_steps),
                        "best_eval_epoch": int(epoch),
                    }
                    model.save(filename=f"{model_name}_best", directory=model_dir)
                    print(
                        _format_metrics(
                            "[BestModel]",
                            {
                                "epoch": epoch,
                                "env_steps": total_env_steps,
                                "avg_reward": best_eval_summary["avg_reward"],
                                "avg_goal_rate": best_eval_summary["avg_goal_rate"],
                                "avg_collision_rate": best_eval_summary["avg_collision_rate"],
                                "avg_episode_steps": best_eval_summary["avg_episode_steps"],
                            },
                        ),
                        flush=True,
                    )
                print(_format_metrics("[EvalSummary]", eval_summary), flush=True)
                last_eval_step = total_env_steps

            now = time.time()
            if now - last_status_time >= status_every_sec:
                _drain_episode_summaries(episode_queue, recent_episodes, total_episode_stats)
                recent_metrics = _recent_episode_metrics(recent_episodes)
                delta_t = max(now - last_status_time, 1e-6)
                delta_env_steps = total_env_steps - last_status_env_steps
                delta_train_iters = total_train_iters - last_status_train_iters
                noise_sigma = model.get_exploration_noise_sigma(total_env_steps)
                status_metrics = {
                    "env_steps": total_env_steps,
                    "train_iters": total_train_iters,
                    "replay_size": replay_buffer.size(),
                    "per_invalid_fallbacks": getattr(replay_buffer, "invalid_priority_events", 0),
                    "episodes": total_episode_stats["episodes"],
                    "explore_random_prob": _exploration_random_prob(
                        total_env_steps,
                        args["start_timesteps"],
                        explore_random_prob_start,
                        explore_random_prob_end,
                        explore_random_decay_steps,
                    ),
                    "recent_goal_rate": recent_metrics["goal_rate"],
                    "recent_collision_rate": recent_metrics["collision_rate"],
                    "recent_timeout_rate": recent_metrics["timeout_rate"],
                    "recent_episode_reward": recent_metrics["avg_reward"],
                    "recent_episode_steps": recent_metrics["avg_steps"],
                    "ou_sigma_linear": noise_sigma["linear"],
                    "ou_sigma_steer": noise_sigma["steer"],
                    "collect_sps": delta_env_steps / delta_t,
                    "train_iter_sps": delta_train_iters / delta_t,
                    "transition_qsize": _safe_queue_size(transition_queue),
                    "episode_qsize": _safe_queue_size(episode_queue),
                }
                print(_format_metrics("[Status]", status_metrics), flush=True)
                if model.writer is not None:
                    model.writer.add_scalar(
                        "train_status/recent_goal_rate", recent_metrics["goal_rate"], total_env_steps
                    )
                    model.writer.add_scalar(
                        "train_status/recent_collision_rate", recent_metrics["collision_rate"], total_env_steps
                    )
                    model.writer.add_scalar(
                        "train_status/recent_timeout_rate", recent_metrics["timeout_rate"], total_env_steps
                    )
                    model.writer.add_scalar(
                        "train_status/recent_episode_reward", recent_metrics["avg_reward"], total_env_steps
                    )
                    model.writer.add_scalar(
                        "train_status/recent_episode_steps", recent_metrics["avg_steps"], total_env_steps
                    )
                    model.writer.add_scalar("train_status/collect_sps", delta_env_steps / delta_t, total_env_steps)
                    model.writer.add_scalar("train_status/train_iter_sps", delta_train_iters / delta_t, total_env_steps)
                last_status_time = now
                last_status_env_steps = total_env_steps
                last_status_train_iters = total_train_iters

            # Exit if a worker dies and no transitions remain.
            if not any(w.is_alive() for w in workers) and transition_queue.empty():
                print("[Learner] All workers finished and queue drained.", flush=True)
                break

        final_recent_metrics = _recent_episode_metrics(recent_episodes)
        print(
            _format_metrics(
                "[Done]",
                {
                    "total_env_steps": total_env_steps,
                    "total_train_iters": total_train_iters,
                    "episodes": total_episode_stats["episodes"],
                    "goal_rate_recent": final_recent_metrics["goal_rate"],
                    "collision_rate_recent": final_recent_metrics["collision_rate"],
                },
            ),
            flush=True,
        )
        if best_eval_summary is not None:
            print(
                _format_metrics(
                    "[BestSummary]",
                    {
                        "best_epoch": int(best_eval_summary["epoch"]),
                        "avg_reward": best_eval_summary["avg_reward"],
                        "avg_goal_rate": best_eval_summary["avg_goal_rate"],
                        "avg_collision_rate": best_eval_summary["avg_collision_rate"],
                        "avg_episode_steps": best_eval_summary["avg_episode_steps"],
                    },
                ),
                flush=True,
            )

    except KeyboardInterrupt:
        print("KeyboardInterrupt. Terminating workers...", flush=True)

    finally:
        # Graceful shutdown
        stop_event.set()
        # Give workers time to exit normally.
        deadline = time.time() + 10.0
        for w in workers:
            remaining = max(0.1, deadline - time.time())
            w.join(timeout=remaining)
        for w in workers:
            if w.is_alive():
                w.terminate()
                w.join()

        # Close the TensorBoard writer.
        if getattr(model, "writer", None) is not None:
            try:
                model.writer.flush()
                model.writer.close()
            except Exception:
                pass

        print("[Learner] Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()
