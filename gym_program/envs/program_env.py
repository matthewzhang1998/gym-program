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
    n_actions = 6
    one_hot = 6
    depth = 6
    penalty = 1e-5
    buf_train = 2e3

    def __init__(self):
        self.hier = True
        self.curiosity = True
        self.visualize = True
        self.model = None
        self.test=0
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
        self.final_state = self.init_state
        
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
        
        enc = np.concatenate((flatten_state, flatten_ptr, flatten_gpr_1,
                              flatten_gpr_2))
        
        
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
#            
#        elif action == 6:
#            if new_dict["state"][ptr] <= new_dict["gpr_1"][0]:
#                new_dict["alu_flag"][0] = 1
#            else: new_dict["alu_flag"][0] = 2   
#        
#        elif action == 7: # push
#            new_dict["stack"].append(new_dict["gpr_1"][0])
#            
#        elif action == 8: # pop
#            if new_dict["stack"]:
#                new_dict["gpr_1"][0] = new_dict["stack"].pop()
#                
#        elif action == AbstractProgramEnv.n_actions - 1:
#            new_dict["return"] = new_dict["gpr_1"][0]
#            new_dict["comp_flag"][0] = 1
        
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
        self.ob = Box(low = -1, high = 1, shape = (n_features,), dtype=np.float32)
        
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
        
        intermediate = self._init_state
        for _ in range(self.nhist):
            intermediate = self.intermediate_goal(intermediate)
        self.final_state = intermediate
        
    def set_test(self, test=0):
        self.test = test
    
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
        if obs is None:
            obs = self._enc(self._state)[0]
        intermediate = self.decode(obs)
        for _ in range(self.nhist):
            intermediate = self.intermediate_goal(intermediate)
        
        return self._enc(intermediate)[0]
    
    def get_action(self, obs):
        raise NotImplementedError
        
    def intermediate_goal(self, *args):
        raise NotImplementedError
        
    def decode(self, obs):
        DEPTH = self.depth
        ONE_HOT = self.one_hot
        
        re_enc = {'stack':[], 'ptr_stack':[], 'alu_flag':[0]}
        size = DEPTH + ONE_HOT
        n_nums = obs.shape[0] // size
        for i in range(0, n_nums):
            start = i * size
            vec = obs[start:start+size]
            oh_num = vec[DEPTH:]
            oh_variable = vec[:DEPTH]
            un_hot_num = np.argmax(oh_num, axis=0)
            un_hot_var = np.argmax(oh_variable, axis=0)
            variable = ['state', 'gpr_1', 'gpr_2', 'ptr', 'comp_flag', 'alu_flag'][un_hot_var]
            
            if variable in re_enc:
                re_enc[variable].append(un_hot_num)
            else:
                re_enc[variable] = [un_hot_num]
        return re_enc
    
        
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
