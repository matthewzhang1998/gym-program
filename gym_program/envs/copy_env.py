#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:21:06 2018

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

class CopyEnv(AbstractProgramEnv):
    env_dir = "CopyEnv"
    copy_length = 3
    vis_iteration = 1e4
    init_state = {'state':list(range(1, copy_length + 1))+[0]*copy_length, 'ptr':[0],
                      'comp_flag':[0],'stack':[], 'ptr_stack':[], 'gpr_1':[0],
                      'gpr_2':[0], 'alu_flag':[0]}
    sequence = ['copy'] * 3
    end_scaling = 1
    
    def __init__(self):
        self._init_state, self._tokens = self._get_init()
        self._state = copy.deepcopy(self._init_state)
        super(CopyEnv, self).__init__()
        
        try:
            self._sess = tf.get_default_session()
            self._sess.run(tf.global_variables_initializer()) # Non-parallel
            self.obs, _ = self._enc(self._state)
            self._uninitialized = False
        except:
            self._uninitialized = True
        
        self.viewer = None
        self._vis_map = np.zeros((CopyEnv.copy_length*2,CopyEnv.copy_length + 1))
    
    def step(self, action):
        if self._uninitialized:
            try:
                self._sess = tf.get_default_session()
                self._sess.run(tf.global_variables_initializer())
                self.obs, _ = self._enc(self._state)
                self._uninitialized = False
            except:
                raise Exception("Unable to initialize encoding variables")
            
        result = super(CopyEnv, self).step(action)
        
        if self.visualize:
            self._visualize(self._state["state"])
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
        return super(CopyEnv, self).reset()
    
    def _visualize(self, state):
        self._iterator += 1
        iterator = (self._iterator - 1) % CopyEnv.vis_iteration + 1
        self._vis_map = self._vis_map * (iterator - 1)/iterator
        for i in range(len(state)):
            self._vis_map[i,state[i]] += 1/iterator 
        
        if self._iterator % CopyEnv.vis_iteration == 0:
            vis_fig = plt.gcf()
            plt.imshow(self._vis_map, cmap='hot', interpolation='nearest')
            vis_fig.savefig(osp.join(self.logdir, 'visitation-{}.png'.format(
                    self._iterator)), dpi = 100)    
            self._vis_map = np.zeros((CopyEnv.copy_length*2,CopyEnv.copy_length + 1))
    
    def _get_reward(self, new_state, state, action):
        new_list = new_state["state"]
        old_list = state["state"]
        
        feed_state, _ = self._enc(new_state)
        feed_old, _ = self._enc(state)
        if self.hier: 
            _, feed_state = feed_state    
            _, feed_old = feed_old
        
        def smeasure(state, slist):
            max_dist = len(slist) - 1
            sortedness = 0
            for i in range(len(slist)):
                min_dist = max_dist
                for j in range(len(state)):
                    if state[j] == slist[i]: min_dist = min(abs(i - j), min_dist)
                sortedness += min_dist
            return 1 - (sortedness/(max_dist * len(slist)))
    
        reward = 0
        #reward = 0
        if self.curiosity:
            reward += (self._cur_model.run(self._sess, feed_old, feed_state, action)[0])
        
        reward /= CopyEnv.end_scaling
        done = 0
                    
        if action == AbstractProgramEnv.n_actions - 1:
            if self.hier:
                self._tokens.pop()
                if len(self._tokens) == 0:
                    reward = (smeasure(new_list, self.final_state)
                                - smeasure(self._init_state["state"], self.final_state))
            else:
                done = 1
                reward = (smeasure(new_list, self.final_state)
                            - smeasure(self._init_state["state"], self.final_state))
                
        if self._episode_length % self.max_iteration == 0:
            done = 1
            
        return reward, done

    def _get_init(self):
        filled = CopyEnv.init_state["state"][0:CopyEnv.copy_length]
        random.shuffle(filled)
        CopyEnv.init_state["state"] = filled + [0]*CopyEnv.copy_length
        print(CopyEnv.init_state["state"])
        self.final_state = [0]*CopyEnv.copy_length + filled
        return copy.deepcopy(CopyEnv.init_state), copy.deepcopy(list(reversed(CopyEnv.sequence)))
    