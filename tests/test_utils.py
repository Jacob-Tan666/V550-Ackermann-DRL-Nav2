"""Tests for evaluation-scene sampling utilities."""

import numpy as np

from drl_navigation_ros2.utils import check_position, record_eval_positions


def test_check_position_enforces_minimum_distance() -> None:
    occupied = [[0.0, 0.0], [4.0, 4.0]]

    assert check_position(2.0, 0.0, occupied, min_dist=1.5)
    assert not check_position(0.5, 0.5, occupied, min_dist=1.5)


def test_record_eval_positions_returns_complete_scenarios(monkeypatch) -> None:
    np.random.seed(42)
    monkeypatch.setenv("WAREHOUSE_KEEPOUTS_ENABLE", "0")
    scenarios = record_eval_positions(n_eval_scenarios=1)

    assert len(scenarios) == 1
    assert [element.name for element in scenarios[0]] == [
        "obstacle5",
        "obstacle6",
        "obstacle7",
        "obstacle8",
        "obstacle9",
        "obstacle10",
        "wheeltec_v550_ackermann",
        "target",
    ]
