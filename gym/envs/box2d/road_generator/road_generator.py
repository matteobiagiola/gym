import math
import copy
from typing import List

from gym.envs.box2d.road_generator.checkpoint import Checkpoint

from gym.envs.box2d.road_generator.track_item import TrackItem
from gym.envs.box2d.road_generator.point import Point
from gym.envs.box2d.road_generator.spline.spline import Spline
from gym.envs.box2d.road_generator.checkpoints_generator import CheckpointsGenerator

from numpy.random.mtrand import RandomState


class RoadGenerator:

    def __init__(self, checkpoints_generator: CheckpointsGenerator, spline: Spline):
        self.fake_alpha = 0.0
        self.checkpoints_generator = checkpoints_generator
        self.spline = spline

    # Compute beta from track generated using splines
    def _compute_beta(self, points: List[Point]) -> List[TrackItem]:

        sign_delta_y = None
        sign_delta_x = None
        sign_switch_y = False

        sign_x_y_switched = False

        _track = []

        for index in range(len(points) - 1):
            x1, y1 = points[index].get_point()
            x2, y2 = points[index + 1].get_point()
            delta_x = x2 - x1
            delta_y = y2 - y1
            beta = (math.pi / 2) - math.atan2(delta_y, -delta_x)

            if math.isclose(x1, x2) and math.isclose(y2, y1):
                # print('dx:', -delta_x, 'dy:', delta_y, 'beta adj:', beta, '-', i)
                _track.append(TrackItem(self.fake_alpha, beta, Point(x1, y1)))
                # make sure that points in which atan2 is undefined do not count for sign switch adjustments
                # print('beta (atan2 not defined):', index, beta)
                continue

            if (sign_delta_y is not None) and (sign_delta_y == 'NEG' and delta_y > 0.0 and -delta_x < 0.0):
                # print('sign switch NEG to POS')
                sign_switch_y = True
            elif (sign_delta_y is not None) and (sign_delta_y == 'POS' and delta_y < 0.0 and -delta_x < 0.0):
                # print('sign switch POS to NEG')
                sign_switch_y = False

            if sign_delta_y is not None and sign_delta_x is not None:
                current_sign_delta_y = 'POS' if delta_y > 0.0 else 'NEG'
                current_sign_delta_x = 'POS' if -delta_x > 0.0 else 'NEG'
                if (current_sign_delta_y == 'POS' and sign_delta_y == 'NEG' and current_sign_delta_x == 'NEG'
                    and sign_delta_x == 'POS') or (current_sign_delta_y == 'NEG' and sign_delta_y == 'POS'
                                                   and current_sign_delta_x == 'NEG' and sign_delta_x == 'POS'):
                    # print('beta:', index, 'subtract pi')
                    sign_x_y_switched = True
                    beta -= math.pi
                elif (current_sign_delta_y == 'NEG' and sign_delta_y == 'POS' and current_sign_delta_x == 'POS'
                      and sign_delta_x == 'NEG') or (current_sign_delta_y == 'POS' and sign_delta_y == 'NEG'
                                                     and current_sign_delta_x == 'POS' and sign_delta_x == 'NEG'):
                    # print('beta:', index, 'add pi')
                    sign_x_y_switched = True
                    beta += math.pi
                else:
                    sign_x_y_switched = False
            # if sign_switch_y and sign_delta_y == 'POS' and -delta_x < 0.0:
            #     beta += 2 * math.pi
            sign_delta_y = 'POS' if delta_y > 0.0 else 'NEG'
            if sign_x_y_switched:
                sign_x_y_switched = False
                sign_delta_x = None
            else:
                sign_delta_x = 'POS' if -delta_x > 0.0 else 'NEG'
            if sign_switch_y and sign_delta_y == 'POS':
                beta += 2 * math.pi

            # print('dx:', -delta_x, 'dy:', delta_y, 'beta adj:', beta, '-', i)
            # print('beta:', index, beta, 'dy:', delta_y, 'dx:', -delta_x, 'sign DY:', sign_delta_y, 'sign DX:', sign_delta_x)
            _track.append(TrackItem(0.0, beta, Point(x1, y1)))
        return _track

    # recompute betas for points in which atan2 is not defined (dy = 0 and dx = 0)
    def _adjust_points_atan_not_defined(self, track_beta_atan: List[TrackItem]) -> List[TrackItem]:

        first_track_item = track_beta_atan[0]
        last_track_item = track_beta_atan[len(track_beta_atan) - 1]
        track_beta_adjusted = [copy.deepcopy(first_track_item)]

        for i in range(len(track_beta_atan) - 2):
            alpha1, beta1, point1 = track_beta_atan[i].get_track_item()
            alpha2, beta2, point2 = track_beta_atan[i + 1].get_track_item()
            alpha3, beta3, point3 = track_beta_atan[i + 2].get_track_item()
            # print('dx:', point2.x - point1.x, 'dy:', point2.y - point1.y, '-', i)
            point2_copy = copy.deepcopy(point2)
            if math.isclose(point2.y, point3.y) and math.isclose(point2.x, point3.x):
                track_beta_adjusted.append(TrackItem(alpha2, (beta3 + beta1) / 2, point2_copy))
            else:
                track_beta_adjusted.append(TrackItem(alpha2, beta2, point2_copy))

        track_beta_adjusted.append(copy.deepcopy(last_track_item))

        return track_beta_adjusted

    def generate_road(self, num_checkpoints: int, resolution: int, np_random: RandomState,
                      track_rad: float, looped=False) -> List:
        checkpoints = self.checkpoints_generator.generate_checkpoints(num_checkpoints, np_random, track_rad)
        track_points = self.spline.compute(checkpoints, resolution, looped=looped)
        track = self._compute_beta(track_points)
        track_adjusted = self._adjust_points_atan_not_defined(track)

        track_to_return = [(track_item.alpha, track_item.beta, track_item.point.x, track_item.point.y)
                           for track_item in track_adjusted]

        return track_to_return
