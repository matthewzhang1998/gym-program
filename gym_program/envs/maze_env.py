#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:58:55 2018

@author: matthewszhang
"""
import numpy as np

import gym
from gym.spaces import Tuple, Discrete, Box
from gym.utils import seeding

STOCH = 0.5

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.stoch = STOCH
        self.obs = 0
        self.observation_space = Box(low=-1, high=1, shape=(1,))
        self.action_space = Discrete(2)
    
    def step(self, action):
        old_state = self.obs
        self.transition(action)
            
        reward, done = self.get_reward(old_state)
        
        return self.obs, reward, done, {}
                
    def reset(self):
        self.obs = 0
        return self.obs
    
    def get_reward(self, old_state):
        done = 0
        if self.obs >= 1 or self.obs <= -1:
            done = 1
        reward = self.obs - old_state
        return reward, done
        
    def transition(self, action):
        if action == 0:
            left = (self.stoch - np.random.uniform() < 0)
            if left:
                self.obs += 0.2
            else:
                self.obs -= 0.2
        else:
            self.obs -= 0.2        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_curiosity(self, curiosity, model='lstm'):
        pass
    
    def set_visualize(self, visualize):
        pass
    
    def set_path(self, dir):
        pass
    
    def set_hier(self, *args):
        pass
    
    def set_stoch(self, stoch=0.2):
        self.stoch = stoch
    
    def render(self, mode='human', close=False):
        raise NotImplementedError("Env must implement abstract method")