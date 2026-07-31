from math import atan2

import numpy as np
from numpy import clip
from torch.utils.tensorboard import SummaryWriter
import yaml


class HCM(object):
    def __init__(
        self,
        state_dim,
        max_action,
        save_samples,
        max_added_samples=10_000,
        file_location="src/drl_navigation_ros2/assets/data.yml",
    ):
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter()
        self.iterator = 0
        self.save_samples = save_samples
        self.max_added_samples = max_added_samples
        self.file_location = file_location

    def get_action(self, state, add_noise):
        sin = state[-3]
        cos = state[-4]
        angle = atan2(sin, cos)
        laser_nr = self.state_dim - 5
        limit = 1.5

        if min(state[4 : self.state_dim - 9]) < limit:
            idx = state[:laser_nr].index(min(state[:laser_nr]))
            if idx > laser_nr / 2:
                sign = -1
            else:
                sign = 1

            idx = clip(idx + sign * 5 * (limit / min(state[:laser_nr])), 0, laser_nr)

            angle = ((3.14 / (laser_nr)) * idx) - 1.57

        rot_vel = clip(angle, -1.0, 1.0)
        lin_vel = -abs(rot_vel / 2)
        return [lin_vel, rot_vel]

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99999,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        pass

    def save(self, filename, directory):
        pass

    def load(self, filename, directory):
        pass

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        latest_scan = np.asarray(latest_scan, dtype=np.float32)
        latest_scan = np.nan_to_num(latest_scan, nan=8.0, posinf=8.0, neginf=0.0)
        latest_scan = np.clip(latest_scan, 0.0, 8.0) / 8.0

        max_bins = self.state_dim - 5
        split_scans = np.array_split(latest_scan, max_bins)
        min_values = [float(np.min(segment)) if len(segment) > 0 else 1.0 for segment in split_scans]

        distance = 8.0 if not np.isfinite(distance) else float(distance)
        norm_distance = float(np.clip(distance / 8.0, 0.0, 1.0))
        cos = float(np.clip(0.0 if not np.isfinite(cos) else cos, -1.0, 1.0))
        sin = float(np.clip(0.0 if not np.isfinite(sin) else sin, -1.0, 1.0))
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            action = np.zeros(2, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        state = np.asarray(min_values + [norm_distance, cos, sin, action[0], action[1]], dtype=np.float32)

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        self.iterator += 1
        if self.save_samples and self.iterator < self.max_added_samples:
            action = action if type(action) is list else action
            action = [float(a) for a in action]
            sample = {
                self.iterator: {
                    "latest_scan": latest_scan.tolist(),
                        "distance": float(distance),
                        "cos": float(cos),
                        "sin": float(sin),
                    "collision": collision,
                    "goal": goal,
                    "action": action,
                }
            }
            with open(self.file_location, "a") as outfile:
                yaml.dump(sample, outfile, default_flow_style=False)

        return state, terminal
