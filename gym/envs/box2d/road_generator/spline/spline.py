from typing import List

from gym.envs.box2d.road_generator.point import Point
from gym.envs.box2d.road_generator.checkpoint import Checkpoint


class Spline:

    def compute(self, points: List[Checkpoint], resolution: int, looped=False) -> List[Point]:
        pass
