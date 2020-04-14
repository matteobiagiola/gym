import math
from typing import List, Tuple
import numpy as np
import copy

from gym.envs.box2d.road_generator.checkpoint import Checkpoint

from gym.envs.box2d.road_generator.spline.spline import Spline
from gym.envs.box2d.road_generator.point import Point


def _make_resolution_proportionate(current_norm: float, max_norm: float, max_resolution: int) -> int:
    return math.floor((current_norm * max_resolution) / max_norm)


def _is_curve_left(_start_point: Point, _end_point: Point, _next_point: Point):
    return ((_end_point.x - _start_point.x) * (_next_point.y - _start_point.y)
            - (_end_point.y - _start_point.y) * (_next_point.x - _start_point.x)) > 0


def _compute_circle_center(_radius: float, _start_point: Point, _end_point: Point, _left_curve: bool) \
        -> Tuple[Point, Point]:
    m = (_end_point.y - _start_point.y) / (_end_point.x - _start_point.x)
    m_c_end = - 1 / m
    a = _end_point.y
    b = _end_point.x

    if (_left_curve and (_end_point.y - _start_point.y) > 0.0) \
            or (not _left_curve and (_end_point.y - _start_point.y) < 0.0):
        c_x = (b * m_c_end ** 2 + b - np.sqrt((m_c_end ** 2 + 1) * _radius ** 2)) / (m_c_end ** 2 + 1)
        c_y = (a * m_c_end ** 2 + a - m_c_end * np.sqrt((m_c_end ** 2 + 1) * _radius ** 2)) / (m_c_end ** 2 + 1)

        alt_c_x = (b * m_c_end ** 2 + b + np.sqrt((m_c_end ** 2 + 1) * _radius ** 2)) / (m_c_end ** 2 + 1)
        alt_c_y = (a * m_c_end ** 2 + a + m_c_end * np.sqrt((m_c_end ** 2 + 1) * _radius ** 2)) / (m_c_end ** 2 + 1)
    else:
        c_x = (b * m_c_end ** 2 + b + np.sqrt((m_c_end ** 2 + 1) * _radius ** 2)) / (m_c_end ** 2 + 1)
        c_y = (a * m_c_end ** 2 + a + m_c_end * np.sqrt((m_c_end ** 2 + 1) * _radius ** 2)) / (m_c_end ** 2 + 1)

        alt_c_x = (b * m_c_end ** 2 + b - np.sqrt((m_c_end ** 2 + 1) * _radius ** 2)) / (m_c_end ** 2 + 1)
        alt_c_y = (a * m_c_end ** 2 + a - m_c_end * np.sqrt((m_c_end ** 2 + 1) * _radius ** 2)) / (m_c_end ** 2 + 1)

    return Point(c_x, c_y), Point(alt_c_x, alt_c_y)


def _find_offset(_x_arc: np.ndarray, _y_arc: np.ndarray, _theta_circle: np.ndarray,
                 _end_point: Point) -> float:
    for index in range(len(_x_arc)):
        # TODO fix for cases in which one of the coordinate is close to zero
        if math.isclose(_end_point.x, _x_arc[index], abs_tol=0.05) and \
                math.isclose(_end_point.y, _y_arc[index], abs_tol=0.05):
            return _theta_circle[index]
    raise ValueError('Offset not found')


def _find_tangent_point(_x_arc: np.ndarray, _y_arc: np.ndarray, _theta_arc: np.ndarray,
                        _next_point: Point, _c: Point) -> Tuple[float, float, float]:
    for index in range(len(_x_arc)):
        _tangent_point = Point(_x_arc[index], _y_arc[index])
        if abs((_tangent_point.x - _c.x)) > 0.0:
            m_pn = (_next_point.y - _tangent_point.y) / (_next_point.x - _tangent_point.x)
            m_cp = (_tangent_point.y - _c.y) / (_tangent_point.x - _c.x)
            if math.isclose(m_pn * m_cp, -1.0, abs_tol=0.1):
                return _x_arc[index], _y_arc[index], _theta_arc[index]
    raise ValueError('Match not found')


def _has_pinnacle(_start_point: Point, _end_point: Point, _x_arc: np.ndarray,
                  _y_arc: np.ndarray, _resolution: int) -> bool:
    next_straight_road = np.linspace(_start_point.get_point(),
                                     _end_point.get_point(),
                                     _resolution)
    next_straight_road_x, next_straight_road_y = zip(*next_straight_road)
    dy_curve = _y_arc[-1] - _y_arc[-2]
    dx_curve = _x_arc[-1] - _x_arc[-2]
    if math.isclose(_y_arc[-1], _y_arc[-2]) and math.isclose(_x_arc[-1], _x_arc[-2]):
        raise ValueError('Cannot compute direction of curve')
    direction_of_curve = math.atan2(dy_curve, dx_curve)
    dy_straight_road = next_straight_road_y[2] - next_straight_road_y[1]
    dx_straight_road = next_straight_road_x[2] - next_straight_road_x[1]

    if math.isclose(next_straight_road_y[1], next_straight_road_y[0]) \
            and math.isclose(next_straight_road_x[1], next_straight_road_x[0]):
        raise ValueError('Cannot compute direction of straight road')
    direction_of_straight_road = math.atan2(dy_straight_road, dx_straight_road)

    if abs(direction_of_curve - direction_of_straight_road) > (np.pi / 2):
        return True
    return False


def _compute_arc(_radius: float, _start_angle: float, _end_angle: float, _resolution: int, _c: Point) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _theta_arc = np.linspace(_start_angle, _end_angle, _resolution)
    _x_arc, _y_arc = _radius * np.cos(_theta_arc) + _c.x, _radius * np.sin(_theta_arc) + _c.y
    return _x_arc, _y_arc, _theta_arc


def _try_to_compute_curve(_start_point: Point, _end_point: Point, _next_point: Point,
                          _left_curve: bool, _resolution: int, _sampling_resolution: int,
                          _radius: float, _max_norm_arc_length: float,
                          _straight_road: np.ndarray, _warn: bool) \
        -> Tuple[Point, np.ndarray, np.ndarray]:
    c, alt_c = _compute_circle_center(_radius, _start_point, _end_point, _left_curve)
    x_circle, y_circle, theta_circle = _compute_arc(_radius, 0.0, 2 * np.pi, _sampling_resolution, c)

    straight_road_x, straight_road_y = zip(*_straight_road)
    point_at_alpha_zero = Point(c.x + _radius, c.y)

    theta_offset = _find_offset(x_circle, y_circle, theta_circle, _end_point)
    if _left_curve:
        x_arc, y_arc, theta_arc = _compute_arc(_radius, theta_offset, theta_offset + 2 * np.pi,
                                               _sampling_resolution, c)
    else:
        x_arc, y_arc, theta_arc = _compute_arc(_radius, theta_offset, theta_offset - 2 * np.pi,
                                               _sampling_resolution, c)
    try:
        tangent_point_x, tangent_point_y, last_angle = \
            _find_tangent_point(x_arc, y_arc, theta_arc, _next_point, c)
        tangent_point = Point(tangent_point_x, tangent_point_y)
        angle = abs(last_angle - theta_offset)
        arc_length = (np.pi * _radius * 2) * (angle / (2 * np.pi))
        arc_resolution = _make_resolution_proportionate(arc_length, _max_norm_arc_length, _resolution)
        if arc_resolution <= 5:
            arc_resolution = 6
        x_arc, y_arc, theta_arc = _compute_arc(_radius, theta_offset, last_angle, arc_resolution, c)
        if math.isclose(theta_offset - 2 * np.pi, last_angle):
            previous_start_point = None
        elif _has_pinnacle(tangent_point, _next_point, x_arc, y_arc, _resolution):
            previous_start_point = None
        else:
            previous_start_point = copy.deepcopy(tangent_point)
            x_arc[0], y_arc[0] = _end_point.x, _end_point.y

    except ValueError:
        previous_start_point = None

    return previous_start_point, x_arc, y_arc


class PtSpline(Spline):

    def __init__(self, radius: float):
        self.radius = radius
        self.max_norm_arc_length = 2 * np.pi * self.radius

    def compute(self, points: List[Checkpoint], resolution: int, looped=False) -> List[Point]:

        sampling_resolution = math.floor(self.radius / 2) * resolution * 100
        previous_start_point = None
        track = []

        max_norm_straight_road = -1.0
        for i in range(len(points) - 1):
            start_point = points[i].point
            end_point = points[i + 1].point
            diff_for_norm = Point(end_point.x - start_point.x, end_point.y - start_point.y)
            norm = np.sqrt(diff_for_norm.x ** 2 + diff_for_norm.y ** 2)
            if norm > max_norm_straight_road:
                max_norm_straight_road = norm

        if looped:
            points.append(copy.deepcopy(points[0]))
            points.append(copy.deepcopy(points[1]))
        else:
            points.append(copy.deepcopy(points[0]))

        for i in range(0, len(points) - 2):
            start_point = points[i].point
            end_point = points[i + 1].point
            next_point = points[i + 2].point
            if previous_start_point is not None:
                start_point = Point(previous_start_point.x, previous_start_point.y)
            diff_for_norm = Point(end_point.x - start_point.x, end_point.y - start_point.y)
            norm_straight_road = np.sqrt(diff_for_norm.x ** 2 + diff_for_norm.y ** 2)
            new_resolution = _make_resolution_proportionate(norm_straight_road, max_norm_straight_road, resolution)
            if new_resolution <= 3:
                new_resolution = 4
            if i == 0:
                straight_road = np.linspace(start_point.get_point(),
                                            end_point.get_point(),
                                            resolution)
            else:
                straight_road = np.linspace(start_point.get_point(),
                                            end_point.get_point(),
                                            new_resolution)
            track.extend(straight_road)

            if i == len(points) - 3:
                break
            left_curve = _is_curve_left(start_point, end_point, next_point)

            warn = True
            previous_start_point, x_arc, y_arc = \
                _try_to_compute_curve(start_point, end_point, next_point, left_curve, resolution, sampling_resolution,
                                      self.radius, self.max_norm_arc_length, straight_road, warn)

            warn = False
            count = 5
            radius = self.radius
            max_norm_straight_road = self.max_norm_arc_length
            while previous_start_point is None and count >= 0:
                radius = radius + 2.0
                max_norm_arc_length = 2 * np.pi * radius
                previous_start_point, x_arc, y_arc = \
                    _try_to_compute_curve(start_point, end_point, next_point, left_curve, resolution,
                                          sampling_resolution,
                                          radius, max_norm_arc_length, straight_road, warn)
                count -= 1

            if count == 0:
                count = 5
                radius = self.radius
                max_norm_straight_road = self.max_norm_arc_length
                while previous_start_point is None and count >= 0 and radius >= 1.0:
                    radius = radius - 2.0
                    max_norm_arc_length = 2 * np.pi * radius
                    previous_start_point, x_arc, y_arc = \
                        _try_to_compute_curve(start_point, end_point, next_point, left_curve, resolution,
                                              sampling_resolution,
                                              radius, max_norm_arc_length, straight_road, warn)
                    count -= 1

            if previous_start_point is not None:
                track.extend(zip(x_arc, y_arc))

        track_first_element = track[0]
        track_last_element = track[-1]
        if math.isclose(track_first_element[0], track_last_element[0]) \
                and math.isclose(track_first_element[1], track_last_element[1]):
            _ = track.pop()  # pop last element

        track_x, track_y = zip(*track)
        spline = [Point(x, y) for x, y in zip(track_x, track_y)]

        return spline
