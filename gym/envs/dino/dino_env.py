import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import numpy as np
from PIL import Image
import cv2 #opencv
import sys 
import io
import time
import pandas as pd
import numpy as np
from random import randint
import os

import random
import pickle
from io import BytesIO
import base64
import json
import time

from gym.envs.dino.dino_game import DinoGame

STATE_H = 300
STATE_W = 600

class DinoEnv(gym.Env, utils.EzPickle):

    def __init__(self, chromebrowser_path='/Users/matteobiagiola/Downloads/chromedriver', 
            render=False):

        utils.EzPickle.__init__(self)

        self.seed()
        self.dino_game = DinoGame(chromebrowser_path=chromebrowser_path, render=render)
        self.previous_score = -1.0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 4), dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if action == 0:
            self.dino_game.press_up()
        # elif action == 1:
        #     self.dino_game.press_down()
        elif action == 1:
            # do nothing
            pass

        observation = self._get_obs()
        step_reward = 0.0
        info = {}
        if self.previous_score < 0.0:
            step_reward = self.dino_game.get_score()
        else:
            step_reward = self.dino_game.get_score() - self.previous_score
        self.previous_score = self.dino_game.get_score()
        return observation, step_reward, self.dino_game.is_crashed(), info

    def _get_obs(self):
        image = self.dino_game.get_image()
        image = self.process_img(image) # processing image as required
        return image

    def process_img(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #RGB to Grey Scale
        image = image[:500, :600] #Crop Region of Interest(ROI)
        # image = cv2.resize(image, (84,84))
        image[image > 0] = 255
        # image = np.reshape(image, (84,84,1))
        return image
        # return image[:, :, None]

    def get_displayed_score(self):
        return self.dino_game.get_score()

    def reset(self):
        self.previous_score = -1.0
        self.dino_game.restart()
        return self._get_obs()

    def render(self):
        pass

    def close(self):
        self.dino_game.end()

# def process_img(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #RGB to Grey Scale
#     image = image[:500, :600] #Crop Region of Interest(ROI)
#     image = cv2.resize(image, (84,84))
#     image[image > 0] = 255
#     return image[:, :, None]

if __name__ == '__main__':
    
    # dino_game = DinoGame(chromebrowser_path='/Users/matteobiagiola/Downloads/chromedriver', render=True)
    # image = dino_game.get_image()
    # print(image.shape)
    # cv2.imwrite('/Users/matteobiagiola/Desktop/image.png', image)
    # preprocessed_image = process_img(image)
    # print(preprocessed_image.shape)
    # cv2.imwrite('/Users/matteobiagiola/Desktop/preprocessed_image.png', preprocessed_image)

    env = DinoEnv(chromebrowser_path='/Users/matteobiagiola/Downloads/chromedriver', render=True)
    env.reset()
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        s, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    print(total_reward, env.get_displayed_score())
    env.close()
