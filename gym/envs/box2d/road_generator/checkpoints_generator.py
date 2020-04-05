from typing import List

from gym.envs.box2d.road_generator.checkpoint import Checkpoint
from numpy.random.mtrand import RandomState


class CheckpointsGenerator:

    def generate_checkpoints(self, num_checkpoints: int, np_random: RandomState, track_rad: float) -> List[Checkpoint]:
        pass
