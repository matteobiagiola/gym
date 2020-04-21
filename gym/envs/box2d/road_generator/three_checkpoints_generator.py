import math
from typing import List

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
                 track_rad_percentage: float = 1 / 2, randomize_first_curve_direction=False):
        self.randomize_alpha = randomize_alpha
        self.randomize_radius = randomize_radius
        self.track_rad_percentage = track_rad_percentage
        self.randomize_first_curve_direction = randomize_first_curve_direction

    def generate_checkpoints(self, num_checkpoints, np_random: RandomState, track_rad: float) -> List[Checkpoint]:
        assert num_checkpoints == 3

        track_rad = track_rad / 2  # making the track shorter

        first_curve_direction = 'left'
        if self.randomize_first_curve_direction:
            if np_random.uniform(0, 1) <= 0.5:
                first_curve_direction = 'right'
            else:
                first_curve_direction = 'left'

        alpha = 0
        rad = 1.5 * track_rad
        checkpoint_1 = build_checkpoint(alpha, rad, first_curve_direction)
        alpha = 2 * math.pi / num_checkpoints
        if self.randomize_alpha:
            if np_random.uniform(0, 1) <= 0.5:
                alpha += np_random.uniform(0, 2 * math.pi / num_checkpoints)
            else:
                alpha -= np_random.uniform(0, 2 * math.pi / num_checkpoints)
        if self.randomize_radius:
            rad = np_random.uniform(self.track_rad_percentage * track_rad, track_rad)
        else:
            rad = 1.5 * track_rad
        checkpoint_2 = build_checkpoint(alpha, rad, first_curve_direction)

        alpha = 2 * math.pi * (num_checkpoints - 1) / num_checkpoints
        rad = 1.5 * track_rad

        # point_1 = checkpoint_1.point
        # point_2 = checkpoint_2.point
        # diff = Point(point_2.x - point_1.x, point_2.y - point_1.y)
        # norm = 2/3 * np.sqrt(diff.x**2 + diff.y**2)
        # print('Length first straight road:', norm)
        # a = norm**2 - point_2.x**2 - point_2.y**2
        # b = point_2.x * math.cos(alpha) + point_2.y * math.sin(alpha)
        # rad = b - np.sqrt(b ** 2 + a)

        checkpoint_3 = build_checkpoint(alpha, rad, first_curve_direction)

        checkpoints = [checkpoint_1, checkpoint_2, checkpoint_3]

        return checkpoints
