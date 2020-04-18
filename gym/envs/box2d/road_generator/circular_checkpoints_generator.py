import math
from typing import List

from gym.envs.box2d.road_generator.point import Point
from gym.envs.box2d.road_generator.checkpoint import Checkpoint
from gym.envs.box2d.road_generator.checkpoints_generator import CheckpointsGenerator

from numpy.random.mtrand import RandomState


class CircularCheckpointsGenerator(CheckpointsGenerator):

    def __init__(self, randomize_alpha=True, randomize_radius=True, track_rad_percentage: float = 1 / 2,
                 randomize_first_curve_direction=False):
        self.randomize_alpha = randomize_alpha
        self.randomize_radius = randomize_radius
        self.track_rad_percentage = track_rad_percentage
        self.randomize_first_curve_direction = randomize_first_curve_direction

    def generate_checkpoints(self, num_checkpoints, np_random: RandomState, track_rad: float) -> List[Checkpoint]:
        assert num_checkpoints > 2
        checkpoints = []
        first_curve_direction = 'left'
        if self.randomize_first_curve_direction:
            if np_random.uniform(0, 1) <= 0.5:
                first_curve_direction = 'right'
            else:
                first_curve_direction = 'left'

        for c in range(num_checkpoints):
            if self.randomize_alpha:
                alpha = 2 * math.pi * c / num_checkpoints \
                        + np_random.uniform(0, 2 * math.pi * 1 / num_checkpoints)
            else:
                alpha = 2 * math.pi * c / num_checkpoints

            if self.randomize_radius:
                rad = np_random.uniform(self.track_rad_percentage * track_rad, track_rad)
            else:
                rad = 1.5 * track_rad

            if c == 0:
                alpha = 0
                rad = 1.5 * track_rad
            elif c == num_checkpoints - 1:
                alpha = 2 * math.pi * c / num_checkpoints
                rad = 1.5 * track_rad

            if first_curve_direction == 'left':
                checkpoints.append(Checkpoint(alpha, Point(rad * math.cos(alpha), rad * math.sin(alpha))))
            else:
                checkpoints.append(Checkpoint(alpha, Point(rad * math.cos(-alpha), rad * math.sin(-alpha))))

        return checkpoints
