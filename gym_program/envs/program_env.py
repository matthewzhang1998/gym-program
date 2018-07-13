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
# TO DO: abstractify ProgramEnv and implement true env as inherited class
# Implement rendering using some visualization software

TOKENS = ["default", "swap", "bubble", "insert", "copy", "find_max"]

class AbstractProgramEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    main_dir = osp.dirname('/home/matthewszhang/logs/program')

    n_features = 32
    n_actions = 10
    one_hot = 6
    depth = 6
    penalty = 1e-5
    buf_train = 2e3

    def __init__(self):
        self.hier = True
        self.curiosity = True
        self.visualize = True
        self.model = None
        self._sbuf = []
        self._obs_buffer = []
        self._nobs_buffer = []
        self._action_buffer = []
        self._episode_length = 0
        self._iterator = 0
        self.seed()
        self.status = 0
        self.max_iteration = 100
        self.intermediate = 0
        
        self.ob = None
        self.observation_space = None
        self.action_space = Discrete(AbstractProgramEnv.n_actions)
    
    def step(self, action):
        self._episode_length += 1
        new_state = self._action(action)
        
        reward, self.status = self._get_reward(new_state, self._state, action)
        old_obs, _ = self._enc(self._state)
        self._state = new_state
        self.obs, enc = self._enc(self._state)
        
        if self.hier: 
            self._obs_buffer.append(self.obs[1])
            self._nobs_buffer.append(old_obs[1])
        else: 
            self._obs_buffer.append(self.obs)
            self._nobs_buffer.append(old_obs)
        self._action_buffer.append(action)
        self._sbuf.append(self._state)
        
        if len(self._obs_buffer) == AbstractProgramEnv.buf_train:
            self._cur_model.train(self._sess, self._obs_buffer, 
                                  self._nobs_buffer, self._action_buffer)        
            #self.icm_sample()
            self._sbuf = []
            self._obs_buffer = []
            self._nobs_buffer = []
            self._action_buffer = []
        
        return self.obs, reward, self.status, {'state':enc}
                
    def reset(self):
        self._episode_length = 0
        self._state, self._tokens = self._get_init()
        self.obs, _ = self._enc(self._state)
        return self.obs
    
    def _get_reward(self, state, new_state, action):
        raise NotImplementedError("each task should specify own reward")
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
      
    def _enc(self, state_dict):
        '''
        encodes init_state with one hots
        
        PARAMS
        -------------
        init_state: list
            contains unsorted list of n numbers
        
        RETURNS
        -------------
        list
            contains encoded array with extra features
        '''
        assert self.model is not None
        
        token_map = lambda x: TOKENS.index(x)
        try:
            token = token_map(self._tokens[-1])
        except:
            if self.status: token = 0
            else: raise ValueError("No token in task")
        
        state = state_dict["state"]
        state_ptr = state_dict["ptr"]
        state_gpr_1 = state_dict["gpr_1"]
        state_gpr_2 = state_dict["gpr_2"]
        state_comp = state_dict["comp_flag"]
        state_flag = state_dict["alu_flag"]
        
        def one_hot_deep(vector, one_hot_size, depth):
            one_hot = np.zeros((len(vector), one_hot_size))
            one_hot[np.arange(len(vector)), vector] = 1
            return np.concatenate((np.zeros((len(vector), depth)), one_hot), axis = 1)
                
        def norm_reshape(vector):
            def normalized(a, axis=-1, order=2):
                l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
                l2[l2==0] = 1
                return a / np.expand_dims(l2, axis)
            vector = np.reshape(vector, (1, vector.shape[0], AbstractProgramEnv.depth
                                         + AbstractProgramEnv.one_hot))
            vector = normalized(vector)
            return vector
    
        flatten_state = one_hot_deep(state, AbstractProgramEnv.one_hot,
                                     AbstractProgramEnv.depth)
        flatten_state[:,0] = 1
        
        flatten_gpr_1 = one_hot_deep(state_gpr_1, AbstractProgramEnv.one_hot,
                                     AbstractProgramEnv.depth)
        flatten_gpr_1[:,1] = 1
        
        flatten_gpr_2 = one_hot_deep(state_gpr_2, AbstractProgramEnv.one_hot,
                                     AbstractProgramEnv.depth)
        flatten_gpr_2[:,2] = 1
        
        flatten_ptr = one_hot_deep(state_ptr, AbstractProgramEnv.one_hot,
                                     AbstractProgramEnv.depth)
        flatten_ptr[:,3] = 1
        
        flatten_comp = one_hot_deep(state_comp, AbstractProgramEnv.one_hot,
                                     AbstractProgramEnv.depth)
        flatten_comp[:,4] = 1
        
        flatten_flag = one_hot_deep(state_flag, AbstractProgramEnv.one_hot,
                                     AbstractProgramEnv.depth)
        flatten_flag[:,5] = 1
        
        enc = np.concatenate((flatten_state, flatten_ptr, flatten_gpr_1,
                              flatten_gpr_2, flatten_comp))
        
        
        if self.model == 'mlp':
            t_out = enc.flatten()     
        
        elif self.model == 'lstm':
            t_enc = norm_reshape(enc)
            t_out = self._sess.run([self.t_out], {self.t_in:t_enc})[0]
            t_out = t_out.flatten()
        
        if self.hier:
            t_out = (token, t_out)
        
        return t_out, enc

    def _action(self, action):
        new_dict = copy.deepcopy(self._state)
        ptr = new_dict["ptr"][0]
        
        # reset flags
        new_dict["alu_flag"][0] = 0
        
        # left
        if action == 0:
            if new_dict["ptr"][0] > 0:
                new_dict["ptr"][0] -= 1
            
        # right
        elif action == 1:
            if new_dict["ptr"][0] < len(new_dict["state"]) - 1:
                new_dict["ptr"][0] += 1
            
        # register copy
        elif action == 2:
            new_dict["gpr_1"][0] = new_dict["state"][new_dict["ptr"][0]]
        
        # register write
        elif action == 3:
            new_dict["state"][new_dict["ptr"][0]] = new_dict["gpr_1"][0]
        
        # register copy
        elif action == 4:
            new_dict["gpr_2"][0] = new_dict["state"][new_dict["ptr"][0]]
        
        # register write
        elif action == 5:
            new_dict["state"][new_dict["ptr"][0]] = new_dict["gpr_2"][0]
            
        elif action == 6:
            if new_dict["state"][ptr] <= new_dict["gpr_1"][0]:
                new_dict["alu_flag"][0] = 1
            else: new_dict["alu_flag"][0] = 2   
        
        elif action == 7: # push
            new_dict["stack"].append(new_dict["gpr_1"][0])
            
        elif action == 8: # pop
            if new_dict["stack"]:
                new_dict["gpr_1"][0] = new_dict["stack"].pop()
                
        elif action == AbstractProgramEnv.n_actions - 1:
            new_dict["return"] = new_dict["gpr_1"][0]
            new_dict["comp_flag"][0] = 1
        
        return new_dict
            
    def _dir_search(self, dir):
        dirno = 0
        while(1):
            temp_dir = osp.join(dir, 'it{}'.format(dirno))
            if not osp.exists(temp_dir):
                return temp_dir
                break
            dirno += 1
    
    def _lstm_normalize(self, t_enc, dtype = tf.float32):
        t_enc = tf.cast(t_enc, dtype)
        with tf.variable_scope("encoding"):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    self._cell, self._cell, t_enc, dtype = tf.float32)
            forward, backward = outputs
            
            t_out = tf.reduce_mean((forward+backward), axis = 1)
            t_out_activ = tf.nn.tanh(t_out)
        return t_out_activ
        
    def _get_init(self):
        raise NotImplementedError
    
    def set_hier(self, hier): 
        self.hier = hier
        if self.hier: self.observation_space = Tuple((Discrete(len(TOKENS)),self.ob)) 
        else: self.observation_space = self.ob
            
    def set_curiosity(self, curiosity, model='lstm'):
        self.curiosity = True
        self.model = model
        
        n_actions = AbstractProgramEnv.n_actions
        if model == 'mlp':
            n_features = self._enc(self._state)[1].flatten().shape[0]
            
        elif model == 'lstm':
            n_features = AbstractProgramEnv.n_features
            self._cell = tf.contrib.rnn.LSTMCell(n_features)
            self.t_in = tf.placeholder(dtype = tf.float32, shape = (
                None, None, AbstractProgramEnv.depth + AbstractProgramEnv.one_hot))
            self.t_out = self._lstm_normalize(self.t_in)
        
        self._cur_model = curiosity(n_features, n_actions, dir=self.logdir)
        self.ob = Box(low = -1, high = 1, shape = (n_features,))
        
    def set_visualize(self, visualize):
        self.visualize = visualize
    
    def set_path(self, dir):
        assert os.path.isdir(dir)
        self.logdir=dir
        
    def set_stoch(self, stoch):
        return
    
    def set_length(self, max_len):
        self.max_iteration = max_len
    
    def set_intermediate(self, intermediate, nhist=1):
        self.intermediate = intermediate
        self.nhist = nhist
    
    def render(self, mode='human', close=False):
        raise NotImplementedError("Env must implement abstract method")
    
    def icm_sample(self):
        buf_len = len(self._sbuf)
        for _ in range(3):
            rn = random.randint(0,buf_len-2)
            sobs, nobs, act = self._sbuf[rn], self._sbuf[rn+1], self._action_buffer[rn+1]
            esobs, _ = self._enc(sobs)
            enobs, _ = self._enc(nobs)
            reward = (self._cur_model.run(self._sess, esobs, enobs, act)[0])
            print("Init State: {} \n Next State: {} \n Act: {} \n Rew: {}".format(sobs, nobs, act, reward))
    
    def get_goal_state(self, obs):
        return self._enc(self.intermediate_goal(self._episode_length, self.intermediate, self.nhist))[0]
    
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
        self._vis_map2 = np.zeros((AbstractProgramEnv.n_actions, 1))
    
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
            self._vis_map2 = np.zeros((AbstractProgramEnv.n_actions, 1))
    
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
        self._init_state, self._tokens = self._get_init()
        self._state = copy.deepcopy(self._init_state)
        super(SortEnv, self).__init__()
        
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
    
    def reset(self):
        if self._uninitialized:
            try:
                self.obs, _ = self._enc(self._state)
                self._sess = tf.get_default_session()
                self._sess.run(tf.global_variables_initializer())
                self._uninitialized = False
            except:
                raise Exception("Unable to initialize encoding variables")
        return super(SortEnv, self).reset()
    
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
            self._vis_map2 = np.zeros((AbstractProgramEnv.n_actions, 1))
    
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
        
#        unshuffled = True
#        while(unshuffled):
#            random.shuffle(SortEnv.init_state["state"])
#            unshuffled = False
#            if SortEnv.init_state["state"] == SortEnv.final_state:
#                unshuffled = True
        return copy.deepcopy(SortEnv.init_state), copy.deepcopy(list(reversed(SortEnv.sequence)))
    
    def intermediate_goal(self, num, intermediate=0, nhist=1):
        sequence = [{'state':[2,1,0], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[0], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[2,1,0], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[2,1,0], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[2,1,0], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[2,2,0], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[2,2,0], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[1,2,0], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[1,2,0], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[1,2,0], 'ptr':[2], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[1,2,0], 'ptr':[2], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[1,2,2], 'ptr':[2], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[1,2,2], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[1,0,2], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[1,0,2], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[0,0,2], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[0,0,2], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[0,1,2], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[0,1,2], 'ptr':[1], 'comp_flag':[1],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]}]
        
        if intermediate == 1:
            if num + nhist >= len(sequence):
                return sequence[-1]
            else:
                return sequence[num + nhist]
            
        else:
            return {'state':[0, 1, 2], 'ptr':[0], 'comp_flag':[1],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]}
    
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
        
        if action == AbstractProgramEnv.n_actions - 1:
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
        old_list = state["state"]
        
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
        
        if action == AbstractProgramEnv.n_actions - 1:
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
            self._vis_map = np.zeros((SortEnv.dim1,SortEnv.dim2))
    
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
                    
        if action == AbstractProgramEnv.n_actions - 1:
            if self._state["return"] == np.argmax(np.array(MaxEnv.init_state["state"])):
                reward = 1
                
        if self._episode_length % self.max_iteration == 0:
            done = 1
            
        return reward, done

    def _get_init(self):
        random.shuffle(MaxEnv.init_state)
        return copy.deepcopy(MaxEnv.init_state), copy.deepcopy(list(reversed(MaxEnv.sequence)))
    
def unmap(obs, obs_tup=None):
    obs_dict = {}
    idx_dict = {}
    
    def unhot(X):
        argmax = np.argmax(X, axis=-1)
        if isinstance(argmax, int): return {TOKENS[argmax]:np.array([0])}
        else:
            idx_dict = {}
            for i in range(argmax.size):
                if TOKENS[argmax[i]] not in idx_dict:
                    idx_dict[TOKENS[argmax[i]]] = [i]
                else:
                    idx_dict[TOKENS[argmax[i]]].append(i)
            for kw in idx_dict:
                idx_dict[kw] = np.array(idx_dict[kw])
            return idx_dict
            
    unmapped_kws, obs = obs
    idx_dict = unhot(unmapped_kws)
    for kw in idx_dict:
        kw_obs = obs[idx_dict[kw]]
        kw_tup = (kw_obs,)
        if obs_tup is not None:
            for item in obs_tup:
                if item is not None:
                    kw_tup += (item[idx_dict[kw]],)    
        obs_dict[kw] = kw_tup
        
    return obs_dict
        
def token_unmap(token):
    if isinstance(token, int):
        return TOKENS[token]
    else:
        assert isinstance(token, list)
        return [TOKENS[i] for i in token]                
        
def obs_unmap(obs):
    unhot = lambda X: np.argmax(X)

    kw, obs = obs
    length = kw.shape[0]
    assert obs.shape[0] == length
    obs_out = []
    for i in range(length):
        obs_out.append((TOKENS[unhot(kw[i,:])], obs[np.newaxis,i,:]))
    return obs_out

def get_all_tokens():
    return TOKENS # helper method
