from gym.envs.box2d.road_generator.point import Point


class TrackItem:

    def __init__(self, alpha: float, beta: float, point: Point):
        self.alpha = alpha
        self.beta = beta
        self.point = point

    def get_track_item(self):
        return self.alpha, self.beta, self.point

    def __repr__(self):
        return '(' + str(self.alpha) + ', ' + str(self.beta) + ', ' + self.point.__repr__() + ')'
