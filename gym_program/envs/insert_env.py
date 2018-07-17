#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 22:13:57 2018

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

class InsertEnv(AbstractProgramEnv):
    env_dir = "InsertEnv"
    vis_iteration = 1e4
    n_nums = 5
    init_state = {'state':[0] * n_nums, 'ptr':[0], 'comp_flag':[0],
                      'stack':[], 'ptr_stack':[], 'gpr_1':[0], 'gpr_2':[0],
                      'alu_flag':[0]}
    enque = list(range(1,6))
    sequence = ['insert'] * 5
    end_scaling = 1
    
    def __init__(self):
        self._init_state, self._tokens = self._get_init()
        self._state = copy.deepcopy(self._init_state)
        super(InsertEnv, self).__init__()
        
        self._enqueue()
        try:
            self._sess = tf.get_default_session()
            self._sess.run(tf.global_variables_initializer()) # Non-parallel
            self.obs, _ = self._enc(self._state)
            self._uninitialized = False
        except:
            self._uninitialized = True
            
        self.viewer = None
        self._vis_map = np.zeros((InsertEnv.n_nums,InsertEnv.n_nums + 1))
    
    def step(self, action):
        if self._uninitialized:
            try:
                self._sess = tf.get_default_session()
                self._sess.run(tf.global_variables_initializer())
                self.obs, _ = self._enc(self._state)
                self._uninitialized = False
            except:
                raise Exception("Unable to initialize encoding variables")
            
        result = super(InsertEnv, self).step(action)
        
        if action == self.n_actions - 1:
            self._enqueue()
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
        val = super(InsertEnv, self).reset()
        self._enqueue()
        return val
    
    def _visualize(self, state):
        self._iterator += 1
        iterator = (self._iterator - 1) % InsertEnv.vis_iteration + 1
        self._vis_map = self._vis_map * (iterator - 1)/iterator
        for i in range(len(state)):
            self._vis_map[i,state[i]] += 1/iterator 
        
        if self._iterator % InsertEnv.vis_iteration == 0:
            vis_fig = plt.gcf()
            plt.imshow(self._vis_map, cmap='hot', interpolation='nearest')
            vis_fig.savefig(osp.join(self.logdir, 'visitation-{}.png'.format(
                    self._iterator)), dpi = 100)    
            self._vis_map = np.zeros((InsertEnv.n_nums,InsertEnv.n_nums + 1))
    
    def _get_reward(self, new_state, state, action):
        new_list = new_state["state"]
        
        feed_state, _ = self._enc(new_state)
        feed_old, _ = self._enc(state)
        if self.hier: 
            _, feed_state = feed_state    
            _, feed_old = feed_old
        
        reward = 0
        #reward = 0
        
        def smeasure(state, slist):
            if len(slist) == 1:
                val = 1 if state == slist else 0
                return val
            max_dist = len(slist) - 1
            sortedness = 0
            for i in range(len(slist)):
                min_dist = max_dist
                for j in range(len(state)):
                    if state[j] == slist[i]: min_dist = min(abs(i - j), min_dist)
                sortedness += min_dist
            return 1 - (sortedness/(max_dist * len(slist)))
        
        if self.curiosity:
            reward += (self._cur_model.run(self._sess, feed_old, feed_state, action)[0])
        
        reward /= InsertEnv.end_scaling
        done = 0
        
        if action == self.n_actions - 1:
            if self.hier:
                self._tokens.pop()
                if len(self._tokens) == 0:
                    reward += smeasure(new_list, self._squeue)
            else:
                reward += smeasure(new_list, self._squeue)
                #reward = smeasure(new_list, SortEnv.final_state) - smeasure(self._init_state["state"], SortEnv.final_state)
        
        if self._episode_length % self.max_iteration == 0:
            done = 1
            reward += smeasure(new_list, self._squeue)/InsertEnv.n_nums
            
        return reward, done

    def _get_init(self):
        self._unqueue = copy.deepcopy(InsertEnv.enque)
        random.shuffle(self._unqueue)
        self._squeue = []
        return copy.deepcopy(InsertEnv.init_state), copy.deepcopy(list(reversed(InsertEnv.sequence)))
    
    def _enqueue(self):
        if not self._unqueue:
            return
        x_out = self._unqueue.pop()
        self._state["stack"].append(x_out)
        self._squeue.append(x_out)
        self._squeue.sort()