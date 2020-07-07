from typing import List, Tuple

from gym.envs.box2d.road_generator.checkpoint import Checkpoint
from numpy.random.mtrand import RandomState


class CheckpointsGenerator:

    def generate_checkpoints(self, num_checkpoints: int, np_random: RandomState, track_rad: float) \
            -> Tuple[List[Checkpoint], List[float], List[float]]:
        pass
