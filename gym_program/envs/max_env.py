#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:38:17 2018

@author: matthewszhang
"""
import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import os.path as osp

import gym
from gym.spaces import Tuple, Discrete, Box
from gym.utils import seeding

from gym_program.icm import make_icm
from gym_program.envs.program_env import AbstractProgramEnv

class MaxEnv(AbstractProgramEnv):
    env_dir = "MaxEnv"
    dim1 = 4
    dim2 = 6
    vis_iteration = 1e4
    init_state = {'state':[1,2,3,4,5], 'ptr':[0], 'comp_flag':[0],
                      'stack':[], 'ptr_stack':[], 'gpr_1':[0], 'gpr_2':[0],
                      'alu_flag':[0], 'return':[0]}
    sequence = ['find_max']
    end_scaling = 1
    def __init__(self):
        self._init_state, self._tokens = self._get_init()
        self._state = copy.deepcopy(self._init_state)
        super(MaxEnv, self).__init__()
        
        try:
            self._sess = tf.get_default_session()
            self._sess.run(tf.global_variables_initializer()) # Non-parallel
            self.obs, _ = self._enc(self._state)
            self._uninitialized = False
        except:
            self._uninitialized = True
        
        self.viewer = None
        self._vis_map = np.zeros((MaxEnv.dim1,MaxEnv.dim2))
    
    def step(self, action):
        if self._uninitialized:
            try:
                self._sess = tf.get_default_session()
                self._sess.run(tf.global_variables_initializer())
                self.obs, _ = self._enc(self._state)
                self._uninitialized = False
            except:
                raise Exception("Unable to initialize encoding variables")
            
        result = super(MaxEnv, self).step(action)
        
        if self.visualize:
            self._visualize(self._state)
        return result
    
    def reset(self):
        if self._uninitialized:
            try:
                self.obs, _ = self._enc(self._state)
                self._sess = tf.get_default_session()
                self._sess.run(tf.global_variables_initializer())
                self._uninitialized = False
            except:
                raise Exception("Unable to initialize encoding variables")
        return super(MaxEnv, self).reset()
    
    def _visualize(self, state):
        self._iterator += 1
        iterator = (self._iterator - 1) % MaxEnv.vis_iteration + 1
        self._vis_map = self._vis_map * (iterator - 1)/iterator
        state = state["ptr"] + state["gpr_1"] + state["gpr_2"] + state["return"]
        for i in range(len(state)):
            self._vis_map[i,state[i]] += 1/iterator 
        
        if self._iterator % MaxEnv.vis_iteration == 0:
            vis_fig = plt.gcf()
            plt.imshow(self._vis_map, cmap='hot', interpolation='nearest')
            vis_fig.savefig(osp.join(self.logdir, 'visitation-{}.png'.format(
                    self._iterator)), dpi = 100)    
            self._vis_map = np.zeros((self.dim1,self.dim2))
    
    def _get_reward(self, new_state, state, action):
        
        feed_state, _ = self._enc(new_state)
        feed_old, _ = self._enc(state)
        if self.hier: 
            _, feed_state = feed_state    
            _, feed_old = feed_old
        
        reward = 0
        
        #reward = 0
        if self.curiosity:
            reward += (self._cur_model.run(self._sess, feed_old, feed_state, action)[0])
        
        reward /= MaxEnv.end_scaling
        done = 0
                    
        if action == self.n_actions - 1:
            if self._state["return"] == np.argmax(np.array(MaxEnv.init_state["state"])):
                reward = 1
                
        if self._episode_length % self.max_iteration == 0:
            done = 1
            
        return reward, done

    def _get_init(self):
        random.shuffle(MaxEnv.init_state)
        return copy.deepcopy(MaxEnv.init_state), copy.deepcopy(list(reversed(MaxEnv.sequence)))