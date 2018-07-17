#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 22:11:15 2018

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

class SwapEnv(AbstractProgramEnv):
    env_dir = "SwapEnv"
    dim1 = 3
    dim2 = 3
    vis_iteration = 1e4
    init_state = {'state':[1, 0], 'ptr':[0], 'comp_flag':[0],
                      'stack':[], 'ptr_stack':[], 'gpr_1':[0], 'gpr_2':[0],
                      'alu_flag':[0]}
    sequence = ["swap"]
    
    def __init__(self):
        self._state, self._tokens = self._get_init()
        super(SwapEnv, self).__init__()
        
        try:
            self._sess = tf.get_default_session()
            self._sess.run(tf.global_variables_initializer()) # Non-parallel
            self.obs, _ = self._enc(self._state)
            if self.hier: self._obs_buffer.append(self.obs[1])
            else: self._obs_buffer.append(self.obs)
            self._uninitialized = False
        except:
            self._uninitialized = True
            
        self.viewer = None
        self._vis_map = np.zeros((2 ** SwapEnv.dim1, 2 ** SwapEnv.dim2))
        self._vis_map2 = np.zeros((self.n_actions, 1))
    
    def step(self, action):
        result = super(SwapEnv, self).step(action)
        
        def get_flattened(state_dict):        
            state = np.array(state_dict["state"])
            state_ptr = np.array(state_dict["ptr"])
            state_gpr_1 = np.array(state_dict["gpr_1"])
            state_gpr_2 = np.array(state_dict["gpr_2"])
            state_comp = np.array(state_dict["comp_flag"])
            return np.concatenate((state, state_ptr, state_gpr_1,
                                   state_gpr_2, state_comp))
        
        if self.visualize:
            state = get_flattened(self._state)
            self._visualize(state, action)
            
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
        return super(SwapEnv, self).reset()
    
    def _visualize(self, state, action):
        self._iterator += 1
        iterator = (self._iterator - 1) % SwapEnv.vis_iteration + 1
        def get_indices(state):
            index1 = index2 = 0
            for i in range(SwapEnv.dim1):
                index1 += state[i] * (2 ** i)
            for j in range(SwapEnv.dim2):
                index2 += state[j + SwapEnv.dim1] * (2 ** j)
            return index1, index2
            
        self._vis_map = self._vis_map * ((iterator - 1)/iterator)
        index1, index2 = get_indices(state)
        self._vis_map[index1, index2] += 1/iterator
        self._vis_map2 = self._vis_map2 * ((iterator - 1)/iterator)
        self._vis_map2[action,0] += 1/iterator
        
        if self._iterator % SwapEnv.vis_iteration == 0:
            vis_fig = plt.gcf()
            plt.imshow(self._vis_map, cmap='hot', interpolation='nearest')
            vis_fig.savefig(osp.join(self.logdir, 'visitation-{}.png'.format(
                    self._iterator)), dpi = 100)
            self._vis_map = np.zeros((2 ** SwapEnv.dim1, 2 ** SwapEnv.dim2))
            vis_fig = plt.gcf()
            plt.imshow(self._vis_map2, cmap='hot', interpolation='nearest')
            vis_fig.savefig(osp.join(self.logdir, 'act-{}.png'.format(
                    self._iterator)), dpi = 100)
            self._vis_map2 = np.zeros((self.n_actions, 1))
    
    def _get_reward(self, new_state, state, action):
        new_list = new_state["state"]
        
        feed_state, _ = self._enc(new_state)
        feed_old, _ = self._enc(state)
        if self.hier:
            _, feed_state = feed_state
            _, feed_old = feed_old
    
        if self.curiosity:
            reward = (self._cur_model.run(self._sess, feed_old, feed_state, action)[0])
        else:
            reward = 0
        done = 0
        if self._episode_length % self.max_iteration == 0:
            if new_list == [0,1]:
                reward = 1
            done = 1
                
        return reward, done
    
    def _get_init(self):
        return copy.deepcopy(SwapEnv.init_state), copy.deepcopy(list(SwapEnv.sequence))
    
    def intermediate_goal(self, num, *args):
        return {'state':[0, 1], 'ptr':[1], 'comp_flag':[1],
                      'stack':[], 'ptr_stack':[], 'gpr_1':[0], 'gpr_2':[1],
                      'alu_flag':[0]}