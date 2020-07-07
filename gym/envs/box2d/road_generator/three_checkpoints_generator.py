import math
from typing import List, Tuple

from gym.envs.box2d.road_generator.point import Point
from gym.envs.box2d.road_generator.checkpoint import Checkpoint
from gym.envs.box2d.road_generator.checkpoints_generator import CheckpointsGenerator

from numpy.random.mtrand import RandomState


def build_checkpoint(alpha, rad, first_curve_direction):
    if first_curve_direction == 'left':
        return Checkpoint(alpha, Point(rad * math.cos(alpha), rad * math.sin(alpha)))
    else:
        return Checkpoint(alpha, Point(rad * math.cos(-alpha), rad * math.sin(-alpha)))


class ThreeCheckpointsGenerator(CheckpointsGenerator):

    def __init__(self, randomize_alpha=True, randomize_radius=True,
                 track_rad_percentage: float = 1 / 2, alpha_percentage: float = 1 / 2, randomize_first_curve_direction=False):
        self.randomize_alpha = randomize_alpha
        self.randomize_radius = randomize_radius
        self.track_rad_percentage = track_rad_percentage
        self.alpha_percentage = alpha_percentage
        self.randomize_first_curve_direction = randomize_first_curve_direction

    def generate_checkpoints(self, num_checkpoints, np_random: RandomState, track_rad: float) \
            -> Tuple[List[Checkpoint], List[float], List[float]]:
        assert num_checkpoints == 3

        alphas = []
        rads = []

        track_rad = track_rad / 2  # making the track shorter

        first_curve_direction = 'left'
        if self.randomize_first_curve_direction:
            if np_random.uniform(0, 1) <= 0.5:
                first_curve_direction = 'right'
            else:
                first_curve_direction = 'left'

        max_alpha_increment = 2 * math.pi / num_checkpoints

        alpha = 0
        alpha_random_part = 0.0
        rad = 1.5 * track_rad
        checkpoint_1 = build_checkpoint(alpha, rad, first_curve_direction)

        alpha = 2 * math.pi / num_checkpoints
        if self.randomize_alpha:
            alpha_random_part = np_random.uniform(self.alpha_percentage * max_alpha_increment, max_alpha_increment)
            if np_random.uniform(0, 1) <= 0.5:
                alpha += alpha_random_part
            else:
                alpha -= alpha_random_part
        if self.randomize_radius:
            rad = np_random.uniform(self.track_rad_percentage * track_rad, track_rad)
        else:
            rad = 1.5 * track_rad
        alphas.append(alpha_random_part)
        rads.append(rad)
        checkpoint_2 = build_checkpoint(alpha, rad, first_curve_direction)

        alpha = 2 * math.pi * (num_checkpoints - 1) / num_checkpoints
        alpha_random_part = 0.0
        rad = 1.5 * track_rad
        alphas.append(alpha_random_part)
        rads.append(rad)
        checkpoint_3 = build_checkpoint(alpha, rad, first_curve_direction)

        checkpoints = [checkpoint_1, checkpoint_2, checkpoint_3]

        return checkpoints, alphas, rads
