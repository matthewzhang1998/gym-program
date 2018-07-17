#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 22:13:40 2018

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

class SortEnv(AbstractProgramEnv):
    env_dir = "SortEnv"
    dim1 = 5
    dim2 = 6
    vis_iteration = 1e4
    init_state = {'state':[2,1,0], 'ptr':[0], 'comp_flag':[0],
                      'stack':[], 'ptr_stack':[], 'gpr_1':[0], 'gpr_2':[0],
                      'alu_flag':[0]}
    final_state = sorted(init_state['state'])
    sequence = ['bubble'] * 4
    end_scaling = 1
    
    def __init__(self):
        super(SortEnv, self).__init__()
        self._init_state, self._tokens = self._get_init()
        self._state = copy.deepcopy(self._init_state)
        
        try:
            self._sess = tf.get_default_session()
            self._sess.run(tf.global_variables_initializer()) # Non-parallel
            self.obs, _ = self._enc(self._state)
            self._uninitialized = False
        except:
            self._uninitialized = True
            
        self.viewer = None
        self._vis_map = np.zeros((SortEnv.dim1,SortEnv.dim2))
        self._vis_map2 = np.zeros((AbstractProgramEnv.n_actions, 1))
    
    def step(self, action):
        if self._uninitialized:
            try:
                self._sess = tf.get_default_session()
                self._sess.run(tf.global_variables_initializer())
                self.obs, _ = self._enc(self._state)
                self._uninitialized = False
            except:
                raise Exception("Unable to initialize encoding variables")
            
        result = super(SortEnv, self).step(action)
        
        if self.visualize:
            self._visualize(self._state["state"], action)
        return result
    
    def reset(self, **kwargs):
        if self._uninitialized:
            try:
                self.obs, _ = self._enc(self._state)
                self._sess = tf.get_default_session()
                self._sess.run(tf.global_variables_initializer())
                self._uninitialized = False
            except:
                raise Exception("Unable to initialize encoding variables")
        return super(SortEnv, self).reset(**kwargs)
    
    def _visualize(self, state, action):
        self._iterator += 1
        iterator = (self._iterator - 1) % SortEnv.vis_iteration + 1
        self._vis_map = self._vis_map * (iterator - 1)/iterator
        for i in range(len(state)):
            self._vis_map[i,state[i]] += 1/iterator
        
        self._vis_map2 = self._vis_map2 * (iterator - 1)/iterator
        self._vis_map2[action,0] += 1/iterator
        
        if self._iterator % SortEnv.vis_iteration == 0:
            vis_fig = plt.gcf()
            plt.imshow(self._vis_map, cmap='hot', interpolation='nearest')
            vis_fig.savefig(osp.join(self.logdir, 'visitation-{}.png'.format(
                    self._iterator)), dpi = 100)    
            self._vis_map = np.zeros((SortEnv.dim1,SortEnv.dim2))
            
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
        if self.curiosity:
            reward += (self._cur_model.run(self._sess, feed_old, feed_state, action)[0])
        
        reward /= SortEnv.end_scaling
        done = 0
                    
# =============================================================================
#         if action == AbstractProgramEnv.n_actions - 1:
#             if self.hier:
#                 self._tokens.pop()
#                 if len(self._tokens) == 0:
#                     done = 1
#                     reward = (smeasure(new_list, SortEnv.final_state) - smeasure(SortEnv.init_state["state"], SortEnv.final_state)/
#                               (1-smeasure(SortEnv.init_state["state"], SortEnv.final_state)))
#                     #reward = smeasure(new_list, SortEnv.final_state) - smeasure(self._init_state["state"], SortEnv.final_state)
#             else:
#                 done = 1
#                 reward = (smeasure(new_list, SortEnv.final_state) - smeasure(SortEnv.init_state["state"], SortEnv.final_state)/
#                               (1-smeasure(SortEnv.init_state["state"], SortEnv.final_state)))
#                 #reward = smeasure(new_list, SortEnv.final_state) - smeasure(self._init_state["state"], SortEnv.final_state)
#         
# =============================================================================
        
        if self._episode_length % self.max_iteration == 0:
            done = 1
            if new_list == SortEnv.final_state:
                reward = 1
            else:
                reward = 0
                 
        return reward, done

    def _get_init(self):
        ''' Turn off shuffling '''
        TEST_STATE = [[2,0,1]]
        state = copy.deepcopy(SortEnv.init_state)
        
        if self.test:
            i = np.random.randint(0, len(TEST_STATE))
            state['state'] = TEST_STATE[i]
            return state, copy.deepcopy(list(reversed(SortEnv.sequence)))
        
        unshuffled = True
        while(unshuffled):
            random.shuffle(state["state"])
            unshuffled = False
            if state["state"] == SortEnv.final_state:
                unshuffled = True
            if not self.test and state["state"] in TEST_STATE:
                unshuffled = True
        return state, copy.deepcopy(list(reversed(SortEnv.sequence)))
    
    def intermediate_goal(self, state_dict):
        s = copy.deepcopy(state_dict)
        
        gpr_1 = state_dict['gpr_1'][0]
        gpr_2 = state_dict['gpr_2'][0]
        
        for i in range(len(state_dict['state']) - 1):
            if state_dict['state'][i] > state_dict['state'][i+1]:
                # if first is greater than second
                if state_dict['gpr_1'][0] != state_dict['state'][i]:
                    if state_dict['ptr'][0] > i:
                        s['ptr'][0] -= 1
                        return s
                    elif state_dict['ptr'][0] < i:
                        s['ptr'][0] += 1
                        return s
                    else:
                        s['gpr_1'][0] = state_dict['state'][i]
                        return s
                else:
                    if state_dict['ptr'][0] > i+1:
                        s['ptr'][0] -= 1
                        return s
                    elif state_dict['ptr'][0] < i+1:
                        s['ptr'][0] += 1
                        return s
                    elif state_dict['gpr_2'][0] != state_dict['state'][i+1]:
                        s['gpr_2'][0] = state_dict['state'][i+1]
                        return s
                    else:
                        s['state'][i+1] = state_dict['gpr_1'][0]
                        return s
                        
            elif state_dict['state'][i] == state_dict['state'][i+1]:
                # assume first has been swapped with second
                if ((state_dict['gpr_2'][0] < state_dict['state'][i]) and (state_dict['gpr_2'][0] not in state_dict['state'])) or \
                    ((state_dict['gpr_1'][0] < state_dict['state'][i]) and (state_dict['gpr_1'][0] not in state_dict['state'])):
                    if state_dict['ptr'][0] > i:
                        s['ptr'][0] -= 1
                        return s
                    elif state_dict['ptr'][0] < i:
                        s['ptr'][0] += 1
                        return s
                    elif state_dict['gpr_2'][0] < state_dict['state'][i]:
                        s['state'][i] = state_dict['gpr_2'][0]
                        return s
                    else:
                        s['state'][i] = state_dict['gpr_1'][0]
                        return s
                elif ((state_dict['gpr_2'][0] > state_dict['state'][i+1]) and (state_dict['gpr_2'][0] not in state_dict['state'])) or \
                    ((state_dict['gpr_1'][0] > state_dict['state'][i+1]) and (state_dict['gpr_1'][0] not in state_dict['state'])):
                    if state_dict['ptr'][0] > i+1:
                        s['ptr'][0] -= 1
                        return s
                    elif state_dict['ptr'][0] < i+1:
                        s['ptr'][0] += 1
                        return s
                    elif state_dict['gpr_2'][0] > state_dict['state'][i+1]:
                        s['state'][i+1] = state_dict['gpr_2'][0]
                        return s
                    else:
                        s['state'][i+1] = state_dict['gpr_1'][0]
                        return s
        
        s['comp_flag'][0] = 1
        return s        
        
#        if state_dict['state'][0] == state_dict['state'][1]
#            
#        sequence = [{'state':[2,1,0], 'ptr':[0], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[0], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[2,1,0], 'ptr':[0], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[2,1,0], 'ptr':[1], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[2,1,0], 'ptr':[1], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
#                          'alu_flag':[0]},
#                    {'state':[2,2,0], 'ptr':[1], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
#                          'alu_flag':[0]},
#                    {'state':[2,2,0], 'ptr':[0], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
#                          'alu_flag':[0]},
#                    {'state':[1,2,0], 'ptr':[0], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
#                          'alu_flag':[0]},
#                    {'state':[1,2,0], 'ptr':[1], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
#                          'alu_flag':[0]},
#                    {'state':[1,2,0], 'ptr':[2], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
#                          'alu_flag':[0]},
#                    {'state':[1,2,0], 'ptr':[2], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[1,2,2], 'ptr':[2], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[1,2,2], 'ptr':[1], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[1,0,2], 'ptr':[1], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[1,0,2], 'ptr':[0], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[0,0,2], 'ptr':[0], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[0,0,2], 'ptr':[1], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[0,1,2], 'ptr':[1], 'comp_flag':[0],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
#                          'alu_flag':[0]},
#                    {'state':[0,1,2], 'ptr':[1], 'comp_flag':[1],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
#                          'alu_flag':[0]}]
#        
#        if intermediate == 1:
#            if num + nhist >= len(sequence):
#                return sequence[-1]
#            else:
#                return sequence[num + nhist]
#            
#        else:
#            return {'state':[0, 1, 2], 'ptr':[0], 'comp_flag':[1],
#                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
#                          'alu_flag':[0]}
            
    def get_action(self, obs):
        # ignore observation and issue actions based solely on ep length, assuming that
        # this function is called only during parallel step execution
        
        action = [2,1,4,3,0,5,1,1,4,3,0,5,0,2,5,1,3,9]
        if self._episode_length >= len(action):
            return 9
        else:
            return action[self._episode_length]
        