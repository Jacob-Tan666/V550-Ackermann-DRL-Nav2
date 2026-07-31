from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


DEFAULT_KINEMATIC_FRAME_DIM = 5
DEFAULT_SCAN_MAX_RANGE = 8.0
DEFAULT_DISTANCE_MAX = 8.0


def _conv1d_output_len(input_len, kernel_size, stride, padding, dilation=1):
    return ((input_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _linear_schedule(step, start, end, decay_steps):
    if decay_steps <= 0 or step is None:
        return np.asarray(end, dtype=np.float32)
    progress = min(1.0, max(0.0, float(step) / float(decay_steps)))
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    return start + progress * (end - start)


class OUNoiseProcess:
    def __init__(
        self,
        action_dim,
        theta,
        sigma_start,
        sigma_end,
        sigma_decay_steps,
        clip=None,
        mu=None,
        dt=1.0,
    ):
        self.action_dim = int(action_dim)
        self.theta = np.broadcast_to(np.asarray(theta, dtype=np.float32), (self.action_dim,))
        self.sigma_start = np.broadcast_to(
            np.asarray(sigma_start, dtype=np.float32), (self.action_dim,)
        )
        self.sigma_end = np.broadcast_to(np.asarray(sigma_end, dtype=np.float32), (self.action_dim,))
        self.sigma_decay_steps = int(sigma_decay_steps)
        self.clip = None if clip is None else np.broadcast_to(np.asarray(clip, dtype=np.float32), (self.action_dim,))
        self.mu = (
            np.zeros(self.action_dim, dtype=np.float32)
            if mu is None
            else np.broadcast_to(np.asarray(mu, dtype=np.float32), (self.action_dim,))
        )
        self.dt = float(dt)
        self.state = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self):
        self.state.fill(0.0)

    def current_sigma(self, step=None):
        return _linear_schedule(step, self.sigma_start, self.sigma_end, self.sigma_decay_steps)

    def sample(self, step=None):
        sigma = self.current_sigma(step)
        random_term = np.random.normal(0.0, 1.0, size=self.action_dim).astype(np.float32)
        delta = self.theta * (self.mu - self.state) * self.dt + sigma * np.sqrt(self.dt) * random_term
        self.state = self.state + delta
        if self.clip is not None:
            self.state = np.clip(self.state, -self.clip, self.clip)
        return self.state.astype(np.float32)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, kinematic_state_dim):
        super(Actor, self).__init__()

        self.kinematic_state_dim = int(kinematic_state_dim)
        self.lidar_dim = max(1, int(state_dim) - self.kinematic_state_dim)
        conv1_len = _conv1d_output_len(self.lidar_dim, kernel_size=5, stride=2, padding=2)
        conv2_len = _conv1d_output_len(conv1_len, kernel_size=3, stride=2, padding=1)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.ln1 = nn.LayerNorm([32, conv1_len])
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([64, conv2_len])
        self.flatten = nn.Flatten()

        fc_input_dim = 64 * conv2_len + self.kinematic_state_dim
        self.layer_1 = nn.Linear(fc_input_dim, 512)
        self.ln3 = nn.LayerNorm(512)
        self.layer_2 = nn.Linear(512, 512)
        self.ln4 = nn.LayerNorm(512)
        self.layer_3 = nn.Linear(512, action_dim)

        self.act = nn.Mish()
        self.tanh = nn.Tanh()

    def forward(self, s):
        lidar = s[:, : self.lidar_dim].unsqueeze(1)
        kin = s[:, self.lidar_dim :]

        x = self.act(self.ln1(self.conv1(lidar)))
        x = self.act(self.ln2(self.conv2(x)))
        x = self.flatten(x)

        features = torch.cat([x, kin], dim=1)
        h = self.act(self.ln3(self.layer_1(features)))
        h = self.act(self.ln4(self.layer_2(h)))
        raw_action = self.layer_3(h)

        a_linear = self.tanh(raw_action[:, 0:1])
        a_angular = self.tanh(raw_action[:, 1:2])
        return torch.cat([a_linear, a_angular], dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, kinematic_state_dim):
        super(Critic, self).__init__()

        self.kinematic_state_dim = int(kinematic_state_dim)
        self.lidar_dim = max(1, int(state_dim) - self.kinematic_state_dim)
        conv1_len = _conv1d_output_len(self.lidar_dim, kernel_size=5, stride=2, padding=2)
        conv2_len = _conv1d_output_len(conv1_len, kernel_size=3, stride=2, padding=1)

        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.ln1_1 = nn.LayerNorm([32, conv1_len])
        self.conv1_2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.ln1_2 = nn.LayerNorm([64, conv2_len])
        self.flatten1 = nn.Flatten()

        fc_input_dim = 64 * conv2_len + self.kinematic_state_dim
        self.layer_1 = nn.Linear(fc_input_dim, 512)
        self.ln1_3 = nn.LayerNorm(512)
        self.layer_2_s = nn.Linear(512, 512)
        self.layer_2_a = nn.Linear(action_dim, 512)
        self.ln1_4 = nn.LayerNorm(512)
        self.layer_3 = nn.Linear(512, 1)

        self.conv2_1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.ln2_1 = nn.LayerNorm([32, conv1_len])
        self.conv2_2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.ln2_2 = nn.LayerNorm([64, conv2_len])
        self.flatten2 = nn.Flatten()

        self.layer_4 = nn.Linear(fc_input_dim, 512)
        self.ln2_3 = nn.LayerNorm(512)
        self.layer_5_s = nn.Linear(512, 512)
        self.layer_5_a = nn.Linear(action_dim, 512)
        self.ln2_4 = nn.LayerNorm(512)
        self.layer_6 = nn.Linear(512, 1)

        self.act = nn.Mish()

    def forward(self, s, a):
        lidar = s[:, : self.lidar_dim].unsqueeze(1)
        kin = s[:, self.lidar_dim :]

        x1 = self.act(self.ln1_1(self.conv1_1(lidar)))
        x1 = self.act(self.ln1_2(self.conv1_2(x1)))
        x1 = self.flatten1(x1)
        features1 = torch.cat([x1, kin], dim=1)

        s1 = self.act(self.ln1_3(self.layer_1(features1)))
        s1 = self.act(self.ln1_4(self.layer_2_s(s1) + self.layer_2_a(a)))
        q1 = self.layer_3(s1)

        x2 = self.act(self.ln2_1(self.conv2_1(lidar)))
        x2 = self.act(self.ln2_2(self.conv2_2(x2)))
        x2 = self.flatten2(x2)
        features2 = torch.cat([x2, kin], dim=1)

        s2 = self.act(self.ln2_3(self.layer_4(features2)))
        s2 = self.act(self.ln2_4(self.layer_5_s(s2) + self.layer_5_a(a)))
        q2 = self.layer_6(s2)
        return q1, q2


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("src/drl_navigation_ros2/models/TD3"),
        model_name="TD3",
        load_directory=Path("src/drl_navigation_ros2/models/TD3"),
        use_writer=True,
        log_dir=None,
        scan_bins=50,
        frame_stack=1,
        kinematic_frame_dim=DEFAULT_KINEMATIC_FRAME_DIM,
        exploration_noise_theta=(0.28, 0.45),
        exploration_noise_sigma_start=(0.10, 0.18),
        exploration_noise_sigma_end=(0.02, 0.06),
        exploration_noise_decay_steps=200000,
        exploration_noise_clip=(0.25, 0.35),
    ):
        self.device = device
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.state_dim = int(state_dim)
        self.scan_bins = int(scan_bins)
        self.frame_stack = int(frame_stack)
        self.kinematic_frame_dim = int(kinematic_frame_dim)
        self.kinematic_state_dim = self.kinematic_frame_dim * self.frame_stack
        self.expected_state_dim = self.scan_bins * self.frame_stack + self.kinematic_state_dim
        if self.expected_state_dim != self.state_dim:
            raise ValueError(
                f"state_dim mismatch: expected {self.expected_state_dim}, got {self.state_dim}. "
                f"scan_bins={self.scan_bins}, frame_stack={self.frame_stack}, "
                f"kinematic_frame_dim={self.kinematic_frame_dim}"
            )

        self.scan_max_range = DEFAULT_SCAN_MAX_RANGE
        self.distance_max = DEFAULT_DISTANCE_MAX

        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            kinematic_state_dim=self.kinematic_state_dim,
        ).to(self.device)
        self.actor_target = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            kinematic_state_dim=self.kinematic_state_dim,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        self.critic = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            kinematic_state_dim=self.kinematic_state_dim,
        ).to(self.device)
        self.critic_target = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            kinematic_state_dim=self.kinematic_state_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        if use_writer:
            self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else SummaryWriter()
        else:
            self.writer = None

        self.iter_count = 0
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.checkpoint_metadata = {}
        self.scan_history = deque(maxlen=self.frame_stack)
        self.kinematic_history = deque(maxlen=self.frame_stack)
        self.exploration_noise = OUNoiseProcess(
            action_dim=self.action_dim,
            theta=exploration_noise_theta,
            sigma_start=exploration_noise_sigma_start,
            sigma_end=exploration_noise_sigma_end,
            sigma_decay_steps=exploration_noise_decay_steps,
            clip=exploration_noise_clip,
        )

        if load_model:
            self.load(filename=model_name, directory=load_directory)

    def reset_observation_history(self):
        self.scan_history.clear()
        self.kinematic_history.clear()

    def reset_action_noise(self):
        self.exploration_noise.reset()

    def get_exploration_noise_sigma(self, step=None):
        sigma = self.exploration_noise.current_sigma(step)
        return {
            "linear": float(sigma[0]),
            "steer": float(sigma[1] if sigma.size > 1 else sigma[0]),
        }

    def get_action(self, obs, add_noise, step=None):
        action = self.act(obs)
        if add_noise:
            action = action + self.exploration_noise.sample(step)
        return np.clip(action, -self.max_action, self.max_action)

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.inference_mode():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.1,
        noise_clip=0.25,
        policy_freq=2,
        priority_step=None,
    ):
        av_Q = 0.0
        max_Q = -inf
        av_loss = 0.0
        av_weight = 0.0

        for it in range(iterations):
            sample = replay_buffer.sample_batch(
                batch_size=batch_size,
                step=None if priority_step is None else int(priority_step) + it,
                with_info=True,
            )
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
                batch_indices,
                batch_weights,
            ) = sample

            state = torch.as_tensor(batch_states, dtype=torch.float32, device=self.device)
            next_state = torch.as_tensor(batch_next_states, dtype=torch.float32, device=self.device)
            action = torch.as_tensor(batch_actions, dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(batch_rewards, dtype=torch.float32, device=self.device)
            done = torch.as_tensor(batch_dones, dtype=torch.float32, device=self.device)
            weights = torch.as_tensor(batch_weights, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_action = self.actor_target(next_state)
                if self.action_dim == 2:
                    noise_scale = torch.as_tensor(
                        [0.5 * policy_noise, policy_noise],
                        dtype=next_action.dtype,
                        device=next_action.device,
                    )
                    noise = torch.randn_like(next_action) * noise_scale
                else:
                    noise = torch.randn_like(next_action) * policy_noise
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + ((1.0 - done) * discount * target_Q)

            current_Q1, current_Q2 = self.critic(state, action)
            if (
                not torch.isfinite(target_Q).all()
                or not torch.isfinite(current_Q1).all()
                or not torch.isfinite(current_Q2).all()
                or not torch.isfinite(weights).all()
            ):
                fallback_priority = np.full(
                    batch_indices.shape[0],
                    getattr(replay_buffer, "max_priority", 1.0),
                    dtype=np.float32,
                )
                replay_buffer.update_priorities(batch_indices, fallback_priority)
                print("Warning: non-finite critic batch detected, skip this update.", flush=True)
                continue

            td_error_1 = target_Q - current_Q1
            td_error_2 = target_Q - current_Q2
            critic_loss = (
                (weights * td_error_1.pow(2)).mean() + (weights * td_error_2.pow(2)).mean()
            )
            if not torch.isfinite(critic_loss):
                fallback_priority = np.full(
                    batch_indices.shape[0],
                    getattr(replay_buffer, "max_priority", 1.0),
                    dtype=np.float32,
                )
                replay_buffer.update_priorities(batch_indices, fallback_priority)
                print("Warning: non-finite critic loss detected, skip this update.", flush=True)
                continue

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()

            td_priority = 0.5 * (td_error_1.abs() + td_error_2.abs())
            replay_buffer.update_priorities(
                batch_indices,
                td_priority.detach().cpu().numpy().reshape(-1),
            )

            av_loss += float(critic_loss.item())
            av_Q += float(target_Q.mean().item())
            max_Q = max(max_Q, float(target_Q.max().item()))
            av_weight += float(weights.mean().item())

            if it % policy_freq == 0:
                actor_loss = -self.critic(state, self.actor(state))[0].mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.iter_count += 1
        if self.writer is not None:
            self.writer.add_scalar("train/loss", av_loss / max(iterations, 1), self.iter_count)
            self.writer.add_scalar("train/avg_Q", av_Q / max(iterations, 1), self.iter_count)
            self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)
            self.writer.add_scalar(
                "train/importance_weight_mean", av_weight / max(iterations, 1), self.iter_count
            )
            sigma = self.get_exploration_noise_sigma(priority_step)
            self.writer.add_scalar("train/exploration_sigma_linear", sigma["linear"], self.iter_count)
            self.writer.add_scalar("train/exploration_sigma_steer", sigma["steer"], self.iter_count)

        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), str(directory / f"{filename}_actor.pth"))
        torch.save(self.actor_target.state_dict(), str(directory / f"{filename}_actor_target.pth"))
        torch.save(self.critic.state_dict(), str(directory / f"{filename}_critic.pth"))
        torch.save(self.critic_target.state_dict(), str(directory / f"{filename}_critic_target.pth"))
        torch.save(
            {
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "iter_count": int(self.iter_count),
                "metadata": dict(self.checkpoint_metadata),
            },
            str(directory / f"{filename}_trainer.pth"),
        )

    def load(self, filename, directory):
        directory = Path(directory)
        actor_path = str(directory / f"{filename}_actor.pth")
        actor_target_path = str(directory / f"{filename}_actor_target.pth")
        critic_path = str(directory / f"{filename}_critic.pth")
        critic_target_path = str(directory / f"{filename}_critic_target.pth")
        trainer_path = directory / f"{filename}_trainer.pth"
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor_target.load_state_dict(torch.load(actor_target_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=self.device))
        if trainer_path.exists():
            trainer_state = torch.load(str(trainer_path), map_location=self.device)
            actor_optimizer_state = trainer_state.get("actor_optimizer")
            critic_optimizer_state = trainer_state.get("critic_optimizer")
            if actor_optimizer_state is not None:
                self.actor_optimizer.load_state_dict(actor_optimizer_state)
            if critic_optimizer_state is not None:
                self.critic_optimizer.load_state_dict(critic_optimizer_state)
            self.iter_count = int(trainer_state.get("iter_count", self.iter_count))
            self.checkpoint_metadata = dict(trainer_state.get("metadata", {}))
        print(f"Loaded weights from: {directory}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        latest_scan = np.asarray(latest_scan, dtype=np.float32)
        latest_scan = np.nan_to_num(
            latest_scan,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        latest_scan = np.clip(latest_scan, 0.0, self.scan_max_range) / self.scan_max_range

        split_scans = np.array_split(latest_scan, self.scan_bins)
        scan_frame = np.asarray(
            [float(np.min(segment)) if len(segment) > 0 else 1.0 for segment in split_scans],
            dtype=np.float32,
        )

        distance = self.distance_max if not np.isfinite(distance) else float(distance)
        norm_distance = float(np.clip(distance / self.distance_max, 0.0, 1.0))
        cos = float(np.clip(0.0 if not np.isfinite(cos) else cos, -1.0, 1.0))
        sin = float(np.clip(0.0 if not np.isfinite(sin) else sin, -1.0, 1.0))
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            action = np.zeros(2, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        kin_frame = np.asarray(
            [norm_distance, cos, sin, float(action[0]), float(action[1])],
            dtype=np.float32,
        )

        self.scan_history.append(scan_frame)
        self.kinematic_history.append(kin_frame)
        while len(self.scan_history) < self.frame_stack:
            self.scan_history.appendleft(scan_frame.copy())
        while len(self.kinematic_history) < self.frame_stack:
            self.kinematic_history.appendleft(kin_frame.copy())

        stacked_scan = np.concatenate(list(self.scan_history), axis=0)
        stacked_kin = np.concatenate(list(self.kinematic_history), axis=0)
        state = np.concatenate([stacked_scan, stacked_kin], axis=0).astype(np.float32)

        assert len(state) == self.state_dim, f"Expected state_dim={self.state_dim}, got {len(state)}"
        terminal = 1 if collision or goal else 0
        return state, terminal
