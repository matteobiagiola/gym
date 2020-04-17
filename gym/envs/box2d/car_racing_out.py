import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from gym.envs.box2d.road_generator.spline.catmull_rom_spline import CatmullRomSpline

from gym.envs.box2d.road_generator.circular_checkpoints_generator import CircularCheckpointsGenerator
from gym.envs.box2d.road_generator.spline.pt_spline import PtSpline
from gym.envs.box2d.road_generator.road_generator import RoadGenerator
from gym.envs.box2d.road_generator.checkpoints_generator import CheckpointsGenerator
from gym.envs.box2d.road_generator.spline.spline import Spline

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl

import os
import pickle

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discrete control is reasonable in this environment as well, on/off discretization is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles visited in the track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track generated is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position and gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
# ZOOM        = 0.3        # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)
ZOOM_FINAL = 16  # Camera zoom final value (after animation) if ZOOM_FOLLOW = False

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
# TRACK_WIDTH = 40 / SCALE
TRACK_WIDTH = 70/SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
                self.env._tile_visited(tile.id)
        else:
            obj.tiles.remove(tile)
            # Registering last contact with track
            self.env.update_contact_with_track()
            if tile.is_last_tile:
                # Registering last tile visited
                self.env.update_last_tile_visited()
                # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)


class CarRacingOut(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, verbose=1, import_track: bool = False, dir_with_tracks=None, export_tracks_dir=None,
                 generate_track: bool = False, num_checkpoints: int = 0, spline: Spline = None,
                 chk_generator: CheckpointsGenerator = None, track_closed: bool = True):
        EzPickle.__init__(self)
        self.seed()

        self.import_track = import_track
        self.generate_track = generate_track

        if import_track:
            assert dir_with_tracks is not None
            self.dir_with_tracks = dir_with_tracks
            if verbose == 1:
                print('Dir with tracks:', dir_with_tracks)
            self.track_file_exports = [f for f in os.listdir(self.dir_with_tracks)
                                       if os.path.isfile(os.path.join(self.dir_with_tracks, f))]
            self.track_file_exports.sort(key=lambda f: os.path.getmtime(os.path.join(self.dir_with_tracks, f)))

        if generate_track:
            assert num_checkpoints > 0
            assert spline is not None
            assert chk_generator is not None
            self.num_checkpoints = num_checkpoints
            self.road_generator = RoadGenerator(checkpoints_generator=chk_generator, spline=spline)
            self.spline_resolution = 20
            self.track_closed = track_closed
        if import_track and generate_track:
            raise ValueError('Either tracks are imported or generated. Choose one!')

        self.export_tracks_dir = export_tracks_dir
        if not import_track and export_tracks_dir is not None and verbose == 1:
            print('Dir export tracks:', export_tracks_dir)

        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.last_tile_visited = False
        self.tile_vertices = []
        self.id_tile_visited = -1
        self.nsteps = -1
        # Max time out car is allowed to be out of the track or still
        self.max_time_out = 2.0
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0), (1, 0), (1, -1), (0, -1)]))
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]),
                                       dtype=np.float32)  # steer, gas, brake
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # self.created_seed = seed
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def is_outside_or_still(self):
        if self.t - self.last_touch_with_track > self.max_time_out > 0.0:
            # if too many seconds outside the track
            return True
        return False

    def is_last_tile_visited(self):
        return self.last_tile_visited

    def update_contact_with_track(self):
        self.last_touch_with_track = self.t

    def _tile_visited(self, tile_id):
        self.id_tile_visited = tile_id

    def update_last_tile_visited(self):
        self.last_tile_visited = True

    def _import_track(self):
        track = []
        with open(os.path.join(self.dir_with_tracks,
                               self.np_random.choice(self.track_file_exports)), 'rb') as filehandle:
            if self.verbose == 1:
                print('Filehandle:', filehandle)
            track = pickle.load(filehandle)
        return track

    def _export_track(self, track, track_filename):

        iteration = 0
        filename = self.export_tracks_dir + '/' + track_filename + str(iteration) + '.data'
        while os.path.exists(filename):
            filename = self.export_tracks_dir + '/' + track_filename + str(iteration) + '.data'
            iteration += 1

        with open(filename, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(track, filehandle)

    def _create_track(self):

        track = []
        self.road = []

        if self.import_track:
            track = self._import_track()
        elif self.generate_track:
            # print('np_random:', self.np_random.get_state()[2])
            track = self.road_generator \
                .generate_road(self.num_checkpoints, self.spline_resolution, self.np_random,
                               TRACK_RAD, looped=self.track_closed)
        else:
            # original code
            CHECKPOINTS = 12
            # Create checkpoints
            checkpoints = []
            for c in range(CHECKPOINTS):
                alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
                rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
                if c == 0:
                    alpha = 0
                    rad = 1.5 * TRACK_RAD
                if c == CHECKPOINTS - 1:
                    alpha = 2 * math.pi * c / CHECKPOINTS
                    self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                    rad = 1.5 * TRACK_RAD
                checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

            # print('Checkpoints: ', checkpoints)
            # print "\n".join(str(h) for h in checkpoints)
            # self.road_poly = [ (    # uncomment this to see checkpoints
            #    [ (tx,ty) for a,tx,ty in checkpoints ],
            #    (0.7,0.7,0.9) ) ]
            # self.road = []

            # Go from one checkpoint to another to create track
            x, y, beta = 1.5 * TRACK_RAD, 0, 0
            dest_i = 0
            laps = 0
            # track = []
            no_freeze = 2500
            visited_other_side = False
            while True:
                alpha = math.atan2(y, x)
                if visited_other_side and alpha > 0:
                    laps += 1
                    visited_other_side = False
                if alpha < 0:
                    visited_other_side = True
                    alpha += 2 * math.pi
                while True:  # Find destination from checkpoints
                    failed = True
                    while True:
                        dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                        if alpha <= dest_alpha:
                            failed = False
                            break
                        dest_i += 1
                        if dest_i % len(checkpoints) == 0:
                            break
                    if not failed:
                        break
                    alpha -= 2 * math.pi
                    continue
                r1x = math.cos(beta)
                r1y = math.sin(beta)
                p1x = -r1y
                p1y = r1x
                dest_dx = dest_x - x  # vector towards destination
                dest_dy = dest_y - y
                proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
                while beta - alpha > 1.5 * math.pi:
                    beta -= 2 * math.pi
                while beta - alpha < -1.5 * math.pi:
                    beta += 2 * math.pi
                prev_beta = beta
                proj *= SCALE
                if proj > 0.3:
                    beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
                if proj < -0.3:
                    beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
                x += p1x * TRACK_DETAIL_STEP
                y += p1y * TRACK_DETAIL_STEP

                track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
                if laps > 4:
                    break
                no_freeze -= 1
                if no_freeze == 0:
                    break
            # print "\n".join([str(t) for t in enumerate(track)])

            # Find closed loop range i1..i2, first loop should be ignored, second is OK
            i1, i2 = -1, -1
            i = len(track)
            while True:
                i -= 1
                if i == 0:
                    return False  # Failed
                pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
                if pass_through_start and i2 == -1:
                    i2 = i
                elif pass_through_start and i1 == -1:
                    i1 = i
                    break
            if self.verbose == 1:
                print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
            assert i1 != -1
            assert i2 != -1

            track = track[i1:i2 - 1]

            first_beta = track[0][1]
            first_perp_x = math.cos(first_beta)
            first_perp_y = math.sin(first_beta)
            # Length of perpendicular jump to put together head and tail
            well_glued_together = np.sqrt(
                np.square(first_perp_x * (track[0][2] - track[-1][2])) +
                np.square(first_perp_y * (track[0][3] - track[-1][3])))
            if well_glued_together > TRACK_DETAIL_STEP:
                return False

        for i in range(1, len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))

            t = self.world.CreateStaticBody(fixtures=fixtureDef(
                shape=polygonShape(vertices=[road1_l, road1_r, road2_r, road2_l])
            ))
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            t.id = i
            if i == len(track) - 1:
                t.is_last_tile = True
            else:
                t.is_last_tile = False
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)

        self.track = track[1:]

        if self.export_tracks_dir is not None:
            self._export_track(track, 'track')

        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.last_tile_visited = False
        self.last_touch_with_track = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.id_tile_visited = -1
        self.nsteps = -1

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        # self.seed(self.created_seed)
        
        return self.step(None)[0]

    # does not work yet
    def _is_inside_tile(self, p, q, r):
        # p(x,y) = point to be checked; q(x1, y1) = bottom-left corner point; 
        # r(x2, y2) = top-right corner point
        if p[0] > q[0] and p[0] < r[0] and p[1] > q[1] and p[1] < r[1]:
            return True
        else:
            return False

    def step(self, action):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        self.nsteps += 1

        self.state = self.render("state_pixels")

        x, y = self.car.hull.position

        car_position = (x, y)

        # filename = 'tile_position_' + str(self.id_tile_visited) + '_' + str(self.id_tile_visited + 1) + '.png'
        # figure = plt.figure()
        # plt.plot(top_left_corner_point[0], top_left_corner_point[1], 'o', color="blue")
        # plt.plot(top_right_corner_point[0], top_right_corner_point[1], 'o', color="orange")
        # plt.plot(bottom_right_corner_point[0], bottom_right_corner_point[1], 'o', color="black")
        # plt.plot(bottom_left_corner_point[0], bottom_left_corner_point[1], 'o', color="green")
        # plt.plot(x, y, 'o', color="red")
        # if os.path.exists(filename):
        #     i = 0
        #     filename = 'tile_position_' + str(self.id_tile_visited) + '_' \
        #         + str(self.id_tile_visited + 1) + '_' + str(i) + '.png'
        #     while os.path.exists(filename):
        #         i += 1
        #         filename = 'tile_position_' + str(self.id_tile_visited) + '_' \
        #             + str(self.id_tile_visited + 1) + '_' + str(i) + '.png'
        # plt.savefig(filename)
        # plt.close(figure)

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()

            # car_is_out = True
            # tile_for_which_car_is_in = -1
            # for i in range(len(self.tile_vertices)):
            #     tile_vertex = self.tile_vertices[i]
            #     top_left_corner_point = tile_vertex[0]
            #     top_right_corner_point = tile_vertex[1]
            #     bottom_right_corner_point = tile_vertex[2]
            #     bottom_left_corner_point = tile_vertex[3]
            #     if self._is_inside_tile(car_position, bottom_left_corner_point, top_right_corner_point):
            #         tile_for_which_car_is_in = i
            #         car_is_out = False
            #         break
            # if not car_is_out:
            #     print('Car is in:', car_position, 'in tile with id:', tile_for_which_car_is_in, 
            #             'current_tile:', self.id_tile_visited)
            # else:
            #     print('Car is out:', car_position, 'current_tile:', self.id_tile_visited)

            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                if self.verbose == 1:
                    print('All tiles visited')
                done = True
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100
            if self.is_outside_or_still():
                if self.verbose == 1:
                    print('Done: car was outside track or still for more than '
                          + str(self.max_time_out) + ' seconds')
                done = True
            if self.is_last_tile_visited():
                if self.verbose == 1:
                    print('Done: last tile visited')
                done = True

        return self.state, step_reward, done, {}

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x='left', anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet
        if ZOOM_FOLLOW:
            zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)  # Animate zoom first second
        else:
            zoom = ZOOM_FINAL
        zoom_state = ZOOM * SCALE * STATE_W / WINDOW_W
        zoom_video = ZOOM * SCALE * VIDEO_W / WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)))
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W / 40.0
        h = H / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h, 0)
            gl.glVertex3f((place + 0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)

        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])


    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0: a[0] = 0
        if k == key.RIGHT and a[0] == +1.0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0


    # dir_with_tracks = '/Users/matteobiagiola/workspace/carracing/road-generator/tracks_simple_pt_splines'
    # env = CarRacingOut(verbose=0, import_track=True, dir_with_tracks=dir_with_tracks)

    radius = 25.0
    spline = PtSpline(radius)
    # spline = CatmullRomSpline()
    rad_percentage = 0.6
    chk_generator = CircularCheckpointsGenerator(randomize_alpha=False, randomize_radius=True, randomize_first_curve_direction=True,
                                                 track_rad_percentage=rad_percentage)
    env = CarRacingOut(verbose=0, generate_track=True, spline=spline,
                       chk_generator=chk_generator, num_checkpoints=3, track_closed=False)

    # env = CarRacingOut()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, '/tmp/video-test', force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                # import matplotlib.pyplot as plt
                # plt.imshow(s)
                # plt.savefig("test.jpeg")
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
