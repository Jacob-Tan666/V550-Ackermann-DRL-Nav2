"""Tests for the TD3 replay buffer."""

import numpy as np
import pytest

from drl_navigation_ros2.replay_buffer import ReplayBuffer


def _transition(index: int) -> tuple[np.ndarray, np.ndarray, float, bool, np.ndarray]:
    state = np.asarray([index, index + 0.5], dtype=np.float32)
    action = np.asarray([0.2, -0.1], dtype=np.float32)
    next_state = state + 1.0
    return state, action, float(index), False, next_state


def test_replay_buffer_rejects_empty_sampling() -> None:
    buffer = ReplayBuffer(buffer_size=4, prioritized=False)

    with pytest.raises(ValueError, match="empty replay buffer"):
        buffer.sample_batch(batch_size=1)


def test_replay_buffer_overwrites_oldest_transition() -> None:
    buffer = ReplayBuffer(buffer_size=2, random_seed=7, prioritized=False)
    for index in range(3):
        buffer.add(*_transition(index))

    states, _, rewards, _, next_states = buffer.return_buffer()

    assert buffer.size() == 2
    np.testing.assert_allclose(states, [[2.0, 2.5], [1.0, 1.5]])
    np.testing.assert_allclose(rewards[:, 0], [2.0, 1.0])
    np.testing.assert_allclose(next_states, [[3.0, 3.5], [2.0, 2.5]])


def test_prioritized_sampling_sanitizes_invalid_priorities() -> None:
    buffer = ReplayBuffer(buffer_size=4, random_seed=11, prioritized=True)
    for index in range(4):
        buffer.add(*_transition(index))

    buffer.update_priorities(
        np.arange(4),
        np.asarray([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32),
    )
    sample = buffer.sample_batch(batch_size=4, step=100, with_info=True)
    indices = sample[-2]
    weights = sample[-1]

    assert indices.shape == (4,)
    assert weights.shape == (4, 1)
    assert np.all(np.isfinite(buffer.priorities[: buffer.size()]))
    assert np.all(buffer.priorities[: buffer.size()] > 0.0)
    assert np.all(np.isfinite(weights))


def test_clear_resets_storage_and_priority_state() -> None:
    buffer = ReplayBuffer(buffer_size=2, prioritized=True)
    buffer.add(*_transition(0))

    buffer.clear()

    assert buffer.size() == 0
    assert buffer.state_buffer is None
    assert buffer.max_priority == 1.0
    np.testing.assert_array_equal(buffer.priorities, np.zeros(2, dtype=np.float32))
