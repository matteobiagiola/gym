from typing import List

import numpy as np
import math

from gym.envs.box2d.road_generator.checkpoint import Checkpoint
from gym.envs.box2d.road_generator.point import Point
from gym.envs.box2d.road_generator.spline.spline import Spline


class CatmullRomSpline(Spline):

    def _catmull_rom_spline(self, p0: Point, p1: Point, p2: Point, p3: Point, resolution: int) -> List[Point]:
        """
        p0, p1, p2, and p3 should be (x,y) point pairs that define the Catmull-Rom spline.
        nPoints is the number of points to include in this curve segment.
        p0 and p3 are control points
        """
        # Convert the points to numpy so that we can do array multiplication
        p0, p1, p2, p3 = map(np.array, [p0, p1, p2, p3])

        # Calculate t0 to t4
        alpha = 0.5

        def tj(ti, pi, pj):
            xi, yi = pi
            xj, yj = pj
            return (((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5) ** alpha + ti

        t0 = 0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)

        # Only calculate points between p1 and p2
        t = np.linspace(t1, t2, resolution)

        # Reshape so that we can multiply by the points p0 to p3
        # and get a point for each value of t.
        t = t.reshape(len(t), 1)

        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3

        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3

        return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2

    def compute(self, points: List[Checkpoint], resolution: int, looped=False) -> List[Point]:
        """
        Calculate Catmull–Rom for a chain of points and return the combined curve.
        """

        # The curve will contain an array of (x, y) points.
        curve = []

        # compute max norm for proportionate resolution between checkpoints
        max_norm = -1
        for i in range(len(points) - 1):
            point_1 = points[i].point
            point_2 = points[i + 1].point
            start_point, end_point = map(np.array, [point_1.get_point(), point_2.get_point()])
            diff = end_point - start_point
            norm = np.linalg.norm(diff)
            if norm > max_norm:
                max_norm = norm

        if looped:
            checkpoints_length = len(points)
            last_control_point = points[checkpoints_length - 1].point
            first_checkpoint = points[0].point
            second_checkpoint = points[1].point
            second_control_point = points[2].point

            start_point, end_point = map(np.array, [first_checkpoint.get_point(),
                                                    second_checkpoint.get_point()])
            norm = np.linalg.norm(end_point - start_point)
            new_resolution = math.floor(norm * resolution / max_norm)

            curve_segment = self._catmull_rom_spline(last_control_point.get_point(),
                                                     first_checkpoint.get_point(),
                                                     second_checkpoint.get_point(),
                                                     second_control_point.get_point(),
                                                     new_resolution)

            curve.extend(curve_segment)

        for i in range(len(points) - 3):
            first_control_point = points[i].point
            start_checkpoint = points[i + 1].point
            dest_checkpoint = points[i + 2].point
            second_control_point = points[i + 3].point

            start_point, end_point = map(np.array, [start_checkpoint.get_point(),
                                                    dest_checkpoint.get_point()])
            norm = np.linalg.norm(end_point - start_point)
            new_resolution = math.floor(norm * resolution / max_norm)

            curve_segment = self._catmull_rom_spline(first_control_point.get_point(),
                                                     start_checkpoint.get_point(),
                                                     dest_checkpoint.get_point(),
                                                     second_control_point.get_point(),
                                                     new_resolution)
            curve.extend(curve_segment)

        if looped:
            checkpoints_length = len(points)
            last_control_point = points[checkpoints_length - 3].point
            last_but_one_checkpoint = points[checkpoints_length - 2].point
            last_checkpoint = points[checkpoints_length - 1].point
            first_control_point = points[0].point

            start_point, end_point = map(np.array, [last_but_one_checkpoint.get_point(),
                                                    last_checkpoint.get_point()])
            norm = np.linalg.norm(end_point - start_point)
            new_resolution = math.floor(norm * resolution / max_norm)

            curve_segment = self._catmull_rom_spline(last_control_point.get_point(),
                                                     last_but_one_checkpoint.get_point(),
                                                     last_checkpoint.get_point(),
                                                     first_control_point.get_point(),
                                                     new_resolution)
            curve.extend(curve_segment)

        x_intpol, y_intpol = zip(*curve)
        spline = [Point(x, y) for x, y in zip(x_intpol, y_intpol)]

        return spline
