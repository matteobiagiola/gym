from gym.envs.box2d.road_generator.point import Point


class Checkpoint:

    def __init__(self, alpha: float, point: Point):
        self.alpha = alpha
        self.point = point

    def get_checkpoint(self):
        return self.alpha, self.point
