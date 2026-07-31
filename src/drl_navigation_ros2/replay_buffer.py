import os

import numpy as np


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        random_seed=123,
        prioritized=None,
        alpha=None,
        beta_start=None,
        beta_end=None,
        beta_decay_steps=None,
        priority_epsilon=None,
        success_priority_boost=None,
        maneuver_priority_boost=None,
    ):
        self.buffer_size = int(buffer_size)
        self.count = 0
        self.ptr = 0
        self.rng = np.random.default_rng(random_seed)

        self.prioritized = (
            prioritized if prioritized is not None else os.environ.get("REPLAY_STRATEGY", "per").lower() == "per"
        )
        self.alpha = float(alpha if alpha is not None else _env_float("PER_ALPHA", 0.6))
        self.beta_start = float(beta_start if beta_start is not None else _env_float("PER_BETA_START", 0.4))
        self.beta_end = float(beta_end if beta_end is not None else _env_float("PER_BETA_END", 1.0))
        self.beta_decay_steps = int(
            beta_decay_steps if beta_decay_steps is not None else _env_int("PER_BETA_DECAY_STEPS", 400000)
        )
        self.priority_epsilon = float(priority_epsilon if priority_epsilon is not None else _env_float("PER_EPS", 1e-4))
        self.success_priority_boost = float(
            success_priority_boost
            if success_priority_boost is not None
            else _env_float("PER_SUCCESS_PRIORITY_BOOST", 2.5)
        )
        self.maneuver_priority_boost = float(
            maneuver_priority_boost
            if maneuver_priority_boost is not None
            else _env_float("PER_MANEUVER_PRIORITY_BOOST", 1.5)
        )

        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.done_buffer = None
        self.next_state_buffer = None
        self.priorities = np.zeros(self.buffer_size, dtype=np.float32)
        self.max_priority = 1.0
        self.invalid_priority_events = 0

    def _fallback_priority(self):
        fallback = float(self.max_priority)
        if not np.isfinite(fallback) or fallback <= 0.0:
            fallback = 1.0
        return fallback

    def _sanitize_priority_array(self, priorities):
        priorities = np.asarray(priorities, dtype=np.float64)
        fallback = self._fallback_priority()
        priorities = np.nan_to_num(
            priorities,
            nan=fallback,
            posinf=fallback,
            neginf=self.priority_epsilon,
        )
        priorities = np.maximum(priorities, self.priority_epsilon)
        if not np.all(np.isfinite(priorities)):
            priorities = np.full_like(priorities, fallback, dtype=np.float64)
        return priorities

    def _init_storage(self, state, action):
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        self.state_buffer = np.zeros((self.buffer_size,) + state.shape, dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_size,) + action.shape, dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_size,) + state.shape, dtype=np.float32)

    def _priority_hint_scale(self, action, reward, terminal):
        scale = 1.0
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if terminal and float(reward) > 0.0:
            scale *= self.success_priority_boost

        if action.size >= 2:
            reverse_motion = action[0] < -0.10
            tight_turn = abs(action[0]) < 0.25 and abs(action[1]) > 0.45
            if reverse_motion or tight_turn:
                scale *= self.maneuver_priority_boost
        return scale

    def _beta_by_step(self, step):
        if step is None or self.beta_decay_steps <= 0:
            return self.beta_end
        progress = min(1.0, max(0.0, float(step) / float(self.beta_decay_steps)))
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def add(self, s, a, r, t, s2):
        if self.state_buffer is None:
            self._init_storage(s, a)

        state = np.asarray(s, dtype=np.float32)
        action = np.asarray(a, dtype=np.float32)
        reward = float(r)
        done = float(t)
        next_state = np.asarray(s2, dtype=np.float32)

        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr, 0] = reward
        self.done_buffer[self.ptr, 0] = done
        self.next_state_buffer[self.ptr] = next_state

        base_priority = self._fallback_priority() if self.prioritized else 1.0
        base_priority *= self._priority_hint_scale(action, reward, done > 0.5)
        if not np.isfinite(base_priority) or base_priority <= 0.0:
            base_priority = self._fallback_priority()
        self.priorities[self.ptr] = max(base_priority, self.priority_epsilon)
        self.max_priority = max(self._fallback_priority(), float(self.priorities[self.ptr]))

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.count = min(self.count + 1, self.buffer_size)

    def size(self):
        return self.count

    def sample_batch(self, batch_size, step=None, with_info=False):
        if self.count == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        sample_size = min(int(batch_size), self.count)
        replace = self.count < int(batch_size)

        if self.prioritized:
            priorities = self._sanitize_priority_array(self.priorities[: self.count])
            scaled_priorities = priorities**self.alpha
            total_priority = float(np.sum(scaled_priorities))
            if (not np.isfinite(total_priority)) or total_priority <= 0.0:
                probs = np.full(self.count, 1.0 / self.count, dtype=np.float64)
                self.invalid_priority_events += 1
            else:
                probs = scaled_priorities / total_priority
                if not np.all(np.isfinite(probs)):
                    probs = np.full(self.count, 1.0 / self.count, dtype=np.float64)
                    self.invalid_priority_events += 1
            indices = self.rng.choice(self.count, size=sample_size, replace=replace, p=probs)
            beta = self._beta_by_step(step)
            weights = (self.count * probs[indices]) ** (-beta)
            weights = weights / max(np.max(weights), 1e-6)
        else:
            indices = self.rng.choice(self.count, size=sample_size, replace=replace)
            weights = np.ones(sample_size, dtype=np.float64)

        s_batch = self.state_buffer[indices]
        a_batch = self.action_buffer[indices]
        r_batch = self.reward_buffer[indices]
        t_batch = self.done_buffer[indices]
        s2_batch = self.next_state_buffer[indices]

        if with_info:
            return (
                s_batch,
                a_batch,
                r_batch,
                t_batch,
                s2_batch,
                indices.astype(np.int64),
                weights.astype(np.float32).reshape(-1, 1),
            )
        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def update_priorities(self, indices, priorities):
        if not self.prioritized or indices is None:
            return

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        priorities = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if indices.size == 0 or priorities.size == 0:
            return

        clipped = self._sanitize_priority_array(np.abs(priorities)).astype(np.float32)
        self.priorities[indices] = clipped
        self.max_priority = max(self._fallback_priority(), float(np.max(clipped)))

    def return_buffer(self):
        if self.count == 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        s = self.state_buffer[: self.count].copy()
        a = self.action_buffer[: self.count].copy()
        r = self.reward_buffer[: self.count].copy()
        t = self.done_buffer[: self.count].copy()
        s2 = self.next_state_buffer[: self.count].copy()
        return s, a, r, t, s2

    def clear(self):
        self.count = 0
        self.ptr = 0
        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.done_buffer = None
        self.next_state_buffer = None
        self.priorities = np.zeros(self.buffer_size, dtype=np.float32)
        self.max_priority = 1.0
