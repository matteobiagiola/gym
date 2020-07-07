import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from shapely.geometry import Point
from shapely.geometry import Polygon

from gym.envs.box2d.road_generator.three_checkpoints_generator import ThreeCheckpointsGenerator
from gym.envs.box2d.road_generator.spline.catmull_rom_spline import CatmullRomSpline

from gym.envs.box2d.road_generator.circular_checkpoints_generator import CircularCheckpointsGenerator
from gym.envs.box2d.road_generator.spline.pt_spline import PtSpline
from gym.envs.box2d.road_generator.road_generator import RoadGenerator
from gym.envs.box2d.road_generator.checkpoints_generator import CheckpointsGenerator
from gym.envs.box2d.road_generator.spline.spline import Spline
from gym.envs.box2d.road_generator.checkpoint import Checkpoint

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl

import os
import pickle
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
import random
import cv2
from gym.envs.box2d.vae import ConvVAE

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

# STATE_W = 96  # less than Atari 160x192
# STATE_H = 96
STATE_W = 64
STATE_H = 64
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
# TRACK_RAD = 450 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
# FPS = 60  # Frames per second
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
# ZOOM        = 0.3        # Camera zoom
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)
ZOOM_FINAL = 16  # Camera zoom final value (after animation) if ZOOM_FOLLOW = False
# ZOOM_FINAL = 4  # Camera zoom final value (after animation) if ZOOM_FOLLOW = False

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
# TRACK_WIDTH = 70 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

# very smooth control: 10% -> 0.2 diff in steering allowed (requires more training)
# smooth control: 15% -> 0.3 diff in steering allowed
# MAX_STEERING_DIFF = 0.15
MAX_STEERING_DIFF = 0.10
# Negative reward for getting off the road
REWARD_CRASH = -10

MIN_THROTTLE = 0.0
MAX_THROTTLE = 0.3
# MAX_THROTTLE = 1

MIN_BRAKE = 0.0
MAX_BRAKE = 0.3
# MAX_BRAKE = 1

Z_SIZE = 32

# Local path
# VAE_PATH = "/Users/matteobiagiola/workspace/my-gym/gym/envs/box2D/vae.json"
# Google colab path
# VAE_PATH = "/content/drive/My Drive/my-gym/gym/envs/box2d/vae.json"

# Symmetric command
MAX_STEERING = 1
MIN_STEERING = -MAX_STEERING

ROAD_COLOR = [0.4, 0.4, 0.4]


class CarExitStatus(Enum):
    CAR_IS_ON_GRASS = 0
    CAR_IS_OUT_OF_PLAYFIELD = 1
    CAR_VISITED_LAST_TILE = 2
    CAR_VISITED_ALL_TILES = 3
    CAR_IS_STILL_TIMEOUT = 4


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
            # print(tile.road_friction, "ADD", len(obj.tiles))
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
                self.env._tile_visited(tile.id)
        else:
            obj.tiles.remove(tile)
            self.env._register_number_of_object_tiles(obj.tiles)
            # print(tile.road_friction, "DEL", len(obj.tiles))  # -- should delete to zero when on grass (this works)
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
                 chk_generator: CheckpointsGenerator = None, track_closed: bool = True, n_command_history: int = 60, 
                 is_vae: bool = False, n_stack: int = 4, discrete_actions: bool = False, num_timesteps_car_allowed_out: int = 15,
                 vae_path: str = None, fixed_track: bool = False, evaluate: bool = False):
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
            self.spline_resolution = 50
            self.track_closed = track_closed
        if import_track and generate_track:
            raise ValueError('Either tracks are imported or generated. Choose one!')

        self.export_tracks_dir = export_tracks_dir
        if not import_track and export_tracks_dir is not None and verbose == 1:
            print('Dir export tracks:', export_tracks_dir)

        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.fixed_track = fixed_track
        self.evaluate = evaluate
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
        self.num_object_tiles = -1
        self.object_tiles_deque = deque(maxlen=4)
        self.nsteps = -1
        self.previous_steering = None
        self.discrete_actions = discrete_actions
        # Max time out car is allowed to be out of the track or still
        self.max_time_out = 1.2
        # self.max_time_out = 4
        self.const_num_timesteps_car_allowed_out = num_timesteps_car_allowed_out

        self.reset_num_timesteps_car_allowed_out()
        self.steer_actions = []
        self.gas_actions = []
        self.brake_actions = []

        # # Save last n commands (steering + gas + brake)
        self.n_commands = 3
        # Save last n commands (steering)
        # self.n_commands = 1
        # Custom frame-stack
        self.n_stack = n_stack

        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_command_history = n_command_history
        self.is_vae = is_vae

        if self.is_vae:
            self.vae = ConvVAE(z_size=Z_SIZE, batch_size=1, is_training=False, reuse=False, gpu_mode=False)
            self.vae.load_json(vae_path)

        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0), (1, 0), (1, -1), (0, -1)]))
        # self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]),
        #                                dtype=np.float32)  # steer, gas, brake
        # self.action_space = spaces.Box(np.array([MIN_STEERING]), np.array([MAX_STEERING]), dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +0.5]),
        #                                dtype=np.float32)  # steer, gas, brake
        # self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +0.5, +0.2]),
        #                                dtype=np.float32)  # steer, gas
        # self.action_space = spaces.Box(np.array([-1, -1]), np.array([+1, +1]),
        #                                dtype=np.float32)  # steer, brake/gas
        # self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
        if self.is_vae:
            self.action_space = spaces.Box(np.array([-1, -1, -1]), np.array([+1, +1, +1]),
                                           dtype=np.float32)  # steer, gas, brake

            if n_stack > 1:
                self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, 
                    shape=(Z_SIZE + self.n_commands * n_command_history,), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, 
                    shape=(1, Z_SIZE + self.n_commands * n_command_history), dtype=np.float32)
            if not self.discrete_actions:
                self.action_space = spaces.Box(np.array([-1, -1, -1]), np.array([+1, +1, +1]),
                                           dtype=np.float32)  # steer, gas, brake
            else:
                self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]),
                                                           dtype=np.float32)  # steer, gas, brake
        else:
            self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]),
                                           dtype=np.float32)  # steer, gas, brake
            self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
            # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.getGroundTruthDim(),), dtype=np.float32)

        if n_stack > 1:
            obs_space = self.observation_space
            low = np.repeat(obs_space.low, self.n_stack, axis=-1)
            high = np.repeat(obs_space.high, self.n_stack, axis=-1)
            self.stacked_obs = np.zeros(low.shape, low.dtype)
            self.observation_space = spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.created_seed = seed
        return [seed]

    @staticmethod
    def getGroundTruthDim():
        return 5

    def getGroundTruth(self):
        return np.array([self.car.hull.position[0], self.car.hull.position[1], 
            self.car.hull.linearVelocity[0], self.car.hull.linearVelocity[1], 
            self.car.hull.angularVelocity])

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

    def _register_number_of_object_tiles(self, object_tiles):
        self.object_tiles_deque.append(len(object_tiles))
        self.num_object_tiles = len(object_tiles)

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
        self.alphas = []
        self.rads = []
        self.checkpoints = []
        self.road = []

        if self.import_track:
            track = self._import_track()
        elif self.generate_track:
            # print('np_random:', self.np_random.get_state()[2])
            track, checkpoints, self.alphas, self.rads = self.road_generator \
                .generate_road(self.num_checkpoints, self.spline_resolution, self.np_random,
                               TRACK_RAD, looped=self.track_closed)
            self.checkpoints = [checkpoint.point for checkpoint in checkpoints]
        else:
            # original code
            CHECKPOINTS = 12
            # Create checkpoints
            checkpoints = []
            for c in range(CHECKPOINTS):
                alpha_random_part = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
                alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
                rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
                if c == 0:
                    alpha = 0
                    alpha_random_part = 0.0
                    rad = 1.5 * TRACK_RAD
                if c == CHECKPOINTS - 1:
                    alpha = 2 * math.pi * c / CHECKPOINTS
                    alpha_random_part = 0.0
                    self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                    rad = 1.5 * TRACK_RAD
                checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
                self.alphas.append(alpha_random_part)
                self.rads.append(rad)

            self.checkpoints = checkpoints
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

        try:
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
                self.road_poly_with_id.append((Polygon((road1_l, road1_r, road2_r, road2_l)), t.id))
                self.road.append(t)
        except AssertionError as e:
            print('AssertionError: trying to re-generate track...')
            return False

        self.track = track[1:]
        # self.track = track

        if self.export_tracks_dir is not None:
            self._export_track(track, 'track')

        return True

    def reset_num_timesteps_car_allowed_out(self):
        self.num_timesteps_car_allowed_out = self.const_num_timesteps_car_allowed_out

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.last_tile_visited = False
        self.last_touch_with_track = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.road_poly_with_id = []
        self.id_tile_visited = -1
        self.num_object_tiles = -1
        self.object_tiles_deque.clear()
        self.nsteps = -1
        self.steer_actions = []
        self.gas_actions = []
        self.brake_actions = []
        self.previous_steering = None
        self.reset_num_timesteps_car_allowed_out()
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        if self.fixed_track:
            self.seed(self.created_seed)

        return self.step(None)[0]

    def _get_safe_index_list(self, start_index, _list):
        assert len(_list) > 0
        result = start_index
        exception = True
        while exception and result > 0:
            try:
                _ = _list[start_index]
                exception = False
            except IndexError:
                result -= 1
        if result < 0:
            raise IndexError('Not possible to find a valid index for this list:', _list)
        return result

    def _is_car_inside(self, x, y, tile_crossed_id):
        # print("Last contact with track", self.t - self.last_touch_with_track)
        if self.t - self.last_touch_with_track > 0.2:
            shapely_point = Point(x, y)
            # polygon, tile_id = self.road_poly_with_id[tile_crossed_id - 1]
            # assert tile_id == tile_crossed_id
            for poly, tile_id in self.road_poly_with_id:
                if poly.contains(shapely_point):
                    self.reset_num_timesteps_car_allowed_out()
                    return True
            if self.num_timesteps_car_allowed_out < 0:
                return False
            else:
                self.num_timesteps_car_allowed_out -= 1
                # print("Num timesteps car allowed out", self.num_timesteps_car_allowed_out)
                return True
        self.reset_num_timesteps_car_allowed_out()
        return True

        # for poly, color in self.road_poly:
        # import matplotlib.pyplot as plt
        # plt.plot(x, y, 'x', color='blue', markersize=20)
        # plt.plot(poly[0][0], poly[0][1], 'x', color='cyan', markersize=20)  # bottom_right
        # plt.plot(poly[1][0], poly[1][1], 'x', color='black', markersize=20)  # bottom_left
        # plt.plot(poly[2][0], poly[2][1], 'x', color='green', markersize=20)  # top_left
        # plt.plot(poly[3][0], poly[3][1], 'x', color='red', markersize=20)  # top_right
        # plt.show()
        # raise ValueError()
        # shapely_polygon = Polygon((poly[0], poly[1], poly[2], poly[3]))
        # if shapely_polygon.contains(shapely_point):
        #     print('Car is inside')
        #     return True
        # return False

    def remap_to_new_range(self, old_value, old_min, old_max, new_min, new_max):
        return ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

    def original_step(self, action):
        if action is not None:
            # Clip steering angle rate to enforce continuity
            if self.n_command_history > 0 and not self.discrete_actions:
                prev_steering = self.command_history[0, -3]
                max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
                diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
                action_0 = prev_steering + diff
                # action_0 = action[0]
            else:
                action_0 = action[0]

            self.car.steer(-action_0)
            self.steer_actions.append(action_0)
            # self.car.steer(-action_0)
            # self.steer_actions.append(action_0)
            if self.is_vae and not self.discrete_actions: 
                self.car.gas(self.remap_to_new_range(action[1], -1, 1, MIN_THROTTLE, MAX_THROTTLE))
                self.gas_actions.append(self.remap_to_new_range(action[1], -1, 1, MIN_THROTTLE, MAX_THROTTLE))
                self.car.brake(self.remap_to_new_range(action[2], -1, 1, MIN_BRAKE, MAX_BRAKE))
                self.brake_actions.append(self.remap_to_new_range(action[2], -1, 1, MIN_BRAKE, MAX_BRAKE))
            else:
                self.car.gas(action[1])
                self.gas_actions.append(action[1])
                self.car.brake(action[2])
                self.brake_actions.append(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        self.nsteps += 1

        # self.state = self.getGroundTruth()
        self.state = self.render("state_pixels")
        if self.is_vae:
            # cv2.imwrite("{}.jpg".format('/Users/matteobiagiola/Desktop/record_original/frame_' + str(self.nsteps)), self.state)
            self.state = self.state.astype(np.float32)/255.0
            self.state = self.state.reshape(1, STATE_W, STATE_H, 3)
            self.state = self.vae.encode(self.state)
            # reconstructed_observation = self.vae.decode(self.state)
            # reconstructed_observation = reconstructed_observation * 255.0
            # reconstructed_observation = reconstructed_observation.astype(np.uint8)
            # cv2.imwrite("{}.jpg".format('/Users/matteobiagiola/Desktop/record_reconstructed/frame_' + str(self.nsteps)), reconstructed_observation[0])

        # Update command history
        if self.is_vae and self.n_command_history > 0 and action is not None:
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            if not self.discrete_actions:
                self.command_history[..., -self.n_commands:] = [action_0, self.remap_to_new_range(action[1], -1, 1, MIN_THROTTLE, MAX_THROTTLE), self.remap_to_new_range(action[2], -1, 1, MIN_BRAKE, MAX_BRAKE)]
            else:
                self.command_history[..., -self.n_commands:] = [action_0, action[1], action[2]]
            self.state = np.concatenate((self.state, self.command_history), axis=-1)
        elif self.is_vae and self.n_command_history > 0 and action is None:
            self.state = np.concatenate((self.state, self.command_history), axis=-1)

        x, y = self.car.hull.position

        car_position = (x, y)

        step_reward = 0
        done = False
        info = {}
        if action is not None:  # First step without action, called from reset()

            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            average_action = [np.array(self.steer_actions).mean(), np.array(self.gas_actions).mean(), np.array(self.brake_actions).mean()]
            min_action = [np.array(self.steer_actions).min(), np.array(self.gas_actions).min(), np.array(self.brake_actions).min()]
            max_action = [np.array(self.steer_actions).max(), np.array(self.gas_actions).max(), np.array(self.brake_actions).max()]
            max_action = [np.array(self.steer_actions).max(), np.array(self.gas_actions).max(), np.array(self.brake_actions).max()]
            std_action = [np.array(self.steer_actions).std(), np.array(self.gas_actions).std(), np.array(self.brake_actions).std()]

            other_info = {
                'per_visited_tiles': self.tile_visited_count/len(self.track), 
                'average_action': average_action,
                'min_action': min_action,
                'max_action': max_action,
                'std_action': std_action,
                'alphas': self.alphas,
                'rads': self.rads
            }

            if self.tile_visited_count == len(self.track):
                if self.verbose == 1:
                    print('All tiles visited')
                done = True
                info = {'car_exit_status': CarExitStatus.CAR_VISITED_ALL_TILES.value, **other_info}

            if self.is_last_tile_visited():
                if self.verbose == 1:
                    print('Done: last tile visited')
                done = True
                info = {'car_exit_status': CarExitStatus.CAR_VISITED_LAST_TILE.value, **other_info}

            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                if self.verbose == 1:
                    print('Car is out of playfield.')
                done = True
                step_reward = -100
                info = {'car_exit_status': CarExitStatus.CAR_IS_OUT_OF_PLAYFIELD.value, **other_info}

            if self.is_outside_or_still():
                if self.verbose == 1:
                    print('Done: car was outside track or still for more than '
                          + str(self.max_time_out) + ' seconds')
                done = True
                info = {'car_exit_status': CarExitStatus.CAR_IS_STILL_TIMEOUT.value, **other_info}

            if not self._is_car_inside(x, y, self.id_tile_visited):
                if self.verbose == 1:
                    print('Car is out.')
                done = True
                step_reward -= REWARD_CRASH
                info = {'car_exit_status': CarExitStatus.CAR_IS_ON_GRASS.value, **other_info}

        if self.n_stack > 1:
            self.stacked_obs = np.roll(self.stacked_obs, shift=-self.state.shape[-1], axis=-1)
            if done:
                self.stacked_obs[...] = 0
            self.stacked_obs[..., -self.state.shape[-1]:] = self.state
            self.state = self.stacked_obs

        if self.evaluate and done == True:
            print('info:', info)

        return self.state, step_reward, done, info

    def step(self, action):
        return self.original_step(action)
        # if action is None:
        #     return self.original_step(action)
        # else:
        #     for _ in range(3):
        #         state, step_reward, done, info = self.original_step(action)
        #     return state, step_reward, done, info

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
        # self.render_indicators(WINDOW_W, WINDOW_H)

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
    # a = np.array([0.0])


    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -0.7
        if k == key.RIGHT: a[0] = +0.7
        if k == key.UP:    a[1] = +0.7
        if k == key.DOWN:  a[2] = +0.7  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -0.7: a[0] = 0
        if k == key.RIGHT and a[0] == +0.7: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0

    # def key_press(k, mod):
    #     global restart
    #     if k == 0xff0d: restart = True
    #     if k == key.LEFT:  a[0] = -0.7
    #     if k == key.RIGHT: a[0] = +0.7
    #     if k == key.UP:    a[1] = +0.7
    #     if k == key.DOWN:  a[2] = +0.7  # set 1.0 for wheels to block to zero rotation


    # def key_release(k, mod):
    #     if k == key.LEFT and a[0] == -0.7: a[0] = 0
    #     if k == key.RIGHT and a[0] == +0.7: a[0] = 0
    #     if k == key.UP:    a[1] = 0
    #     if k == key.DOWN:  a[2] = 0


    # dir_with_tracks = '/Users/matteobiagiola/workspace/carracing/road-generator/tracks_simple_pt_splines'
    # env = CarRacingOut(verbose=0, import_track=True, dir_with_tracks=dir_with_tracks)

    radius = 15.0
    spline = PtSpline(radius)
    # spline = CatmullRomSpline()
    rad_percentage = 0.5
    alpha_percentage = 0.5
    # chk_generator = ThreeCheckpointsGenerator(randomize_alpha=True, randomize_radius=True,
    #                                           randomize_first_curve_direction=True,
    #                                           track_rad_percentage=rad_percentage,
    #                                           alpha_percentage=alpha_percentage)
    chk_generator = CircularCheckpointsGenerator(randomize_alpha=True, randomize_radius=True,
                                              randomize_first_curve_direction=True,
                                              track_rad_percentage=rad_percentage,
                                              alpha_percentage=alpha_percentage)
    env = CarRacingOut(verbose=0, generate_track=True, spline=spline,
                       chk_generator=chk_generator, num_checkpoints=8, track_closed=False, is_vae=True, 
                       n_command_history=60, n_stack=4, discrete_actions=True, 
                       num_timesteps_car_allowed_out=60,
                       vae_path="/Users/matteobiagiola/workspace/my-gym/gym/envs/box2D/vae.json", fixed_track=False)

    # env = CarRacingOut()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, '/tmp/video-test', force=True)
    isopen = True
    # DIR_NAME = '/Users/matteobiagiola/Desktop/record'
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        # recording_obs = []
        # random_generated_int = random.randint(0, 2**31-1)
        # filename = DIR_NAME + "/" + str(random_generated_int) + ".npz"
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            # recording_obs.append(s)
            # if steps % 200 == 0 or done:
            #     print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            #     # import matplotlib.pyplot as plt
            #     # plt.imshow(s)
            #     # plt.savefig("test.jpeg")
            if done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                print('info:', info)
                # recording_obs = np.array(recording_obs, dtype=np.uint8)
                # np.savez_compressed(filename, obs=recording_obs)
                # import matplotlib.pyplot as plt
                # plt.imshow(s)
                # plt.savefig("test.jpeg")
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
