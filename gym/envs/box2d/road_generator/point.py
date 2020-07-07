import math


class Point:

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def get_point(self):
        return self.x, self.y

    def __repr__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

    def __eq__(self, other):
        if isinstance(other, Point):
            return math.isclose(other.x, self.x) and math.isclose(other.y, self.y)
        return False
