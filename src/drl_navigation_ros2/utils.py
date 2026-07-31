from dataclasses import dataclass
import os

import numpy as np


@dataclass
class pos_data:
    name = None
    x = None
    y = None
    angle = None


def check_position(x, y, element_positions, min_dist):
    pos = True
    for element in element_positions:
        distance_vector = [element[0] - x, element[1] - y]
        distance = np.linalg.norm(distance_vector)
        if distance < min_dist:
            pos = False
    return pos


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def check_static_keepout(x, y, margin=0.0):
    if not _env_bool("WAREHOUSE_KEEPOUTS_ENABLE", False):
        return True

    keepout_boxes = [
        (-6.30, -5.15, -9.80, 9.80),
        (4.05, 6.30, -9.80, 9.80),
        (-6.30, 6.30, 7.05, 9.80),
        (-6.30, 6.30, -9.80, -7.25),
        (3.15, 5.95, -9.25, 1.65),
        (2.60, 6.20, 3.05, 4.75),
        (2.35, 6.20, 7.75, 9.35),
        (-2.75, 0.65, -9.95, -7.15),
        (-2.65, -0.30, 7.05, 8.55),
    ]
    if _env_bool("DYNAMIC_LANE_KEEPOUTS_ENABLE", False):
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


def check_dynamic_lane_keepout(x, y, margin=0.0):
    if not _env_bool("DYNAMIC_LANE_KEEPOUTS_ENABLE", False):
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


def set_random_position(name, element_positions):
    angle = np.random.uniform(-np.pi, np.pi)
    pos = False
    min_x = _env_float("WORLD_MIN_X", -4.6)
    max_x = _env_float("WORLD_MAX_X", 3.7)
    min_y = _env_float("WORLD_MIN_Y", -6.8)
    max_y = _env_float("WORLD_MAX_Y", 6.8)
    attempts = 0
    while not pos:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError(f"Unable to sample a valid eval position for {name}. Check WORLD_* bounds and keepouts.")
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        pos = (
            check_position(x, y, element_positions, 1.8)
            and check_static_keepout(x, y, 0.8)
            and (not name.startswith("obstacle") or check_dynamic_lane_keepout(x, y, 0.2))
        )
    element_positions.append([x, y])
    eval_element = pos_data()
    eval_element.name = name
    eval_element.x = x
    eval_element.y = y
    eval_element.angle = angle
    return eval_element


def record_eval_positions(n_eval_scenarios=10):
    scenarios = []
    for _ in range(n_eval_scenarios):
        eval_scenario = []
        element_positions = [[-3.25, 3.10], [2.75, -3.20], [-1.80, -2.90], [2.70, 2.25]]
        for i in range(4, 10):
            name = "obstacle" + str(i + 1)
            eval_element = set_random_position(name, element_positions)
            eval_scenario.append(eval_element)

        eval_element = set_random_position("wheeltec_v550_ackermann", element_positions)
        eval_scenario.append(eval_element)

        eval_element = set_random_position("target", element_positions)
        eval_scenario.append(eval_element)

        scenarios.append(eval_scenario)

    return scenarios
