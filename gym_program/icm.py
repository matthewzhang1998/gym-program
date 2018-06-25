#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:10:27 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf
from gym_program.helpers import ff, fc

PARAMS = {'alpha_ICM':1e-4,
          'eta':1e0,
          'gamma':0.9,
          'eta_decay':0.9999,
          'eta_floor':2e5,
          'mb':4,
          'mg':0.2,
          'epochs':10}

class ICM():
    def __init__(self, n_features, n_actions):
        self.iterator = 0
        self.params = PARAMS
        self.minibatch = self.params["mb"]
        self.alpha = self.params["alpha_ICM"]
        self.n_actions = n_actions
        
        self.forward_widths, self.forward_activations = layers
        
        self.X = tf.placeholder(dtype = tf.float32,
                                shape = (None, n_inputs), name = 'curr_states')
        self.Y = tf.placeholder(dtype = tf.float32, shape=(None, n_outputs),
                                       name = 'next_states')
        activ = tf.nn.relu
        flatten = tf.layers.flatten            
        with tf.variable_scope('icm'):                   
            pred_1 = activ(fc(flatten(ccsa), 'pred_1', nh=2*n_features, init_scale=np.sqrt(2)))
            pred_2 = activ(fc(pred_1, 'pred_2', nh=2*n_features, init_scale=np.sqrt(2)))
            pred_3 = activ(fc(pred_2, 'pred_3', nh=n_features, init_scale=np.sqrt(2)))
            pred_out = tf.nn.l2_normalize(pred_3)
        
        t_forward_loss = 1/2 * tf.reduce_sum((tf.square(pred_out - labels)), axis=-1)
        
        #eta = min(1/(self.params["eta"]) * (self.params["eta_decay"] ** self.iterator),
        #          1/self.params["eta_floor"])
        eta = 1/self.params["eta"]
        self.internal_reward = eta * t_forward_loss
        
        self.total_loss = tf.reduce_mean(t_forward_loss)
        optimizer = tf.train.AdamOptimizer(self.alpha)
        params = tf.trainable_variables()
        grads = tf.gradients(self.total_loss, params)
        max_grad = self.params["mg"]
        if max_grad is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
        grads = list(zip(grads, params))
        self.train_op = optimizer.apply_gradients(grads)
        
    def _get_transitions(self, t_states):
        return tf.concat([t_states[:-1], t_states[1:]], axis= -1)
        
    def _one_hot(self, t, depth):
        try:
            t_in = np.array(t)
            length = t_in.shape[0]
        except:
            t_in = np.array([t])
            length = t_in.shape[0]
        t_out = np.zeros((length, depth))
        t_out[np.arange(length), t_in.astype(int)] = 1
        return t_out
    
    def run(self, sess, old_state, new_state, action, curiosity = True):
        if curiosity:
            self.iterator += 1
            t_actions = self._one_hot(action, self.n_actions)
            old_state = old_state[np.newaxis]
            new_state = new_state[np.newaxis]
            feed = {self.t_states: old_state,
                    self.t_actions: t_actions,
                    self.n_states: new_state}
            rewards = sess.run([self.internal_reward], feed)[0]
            return rewards
         
    def train(self, sess, t_states, t_old, t_actions):
        nbatch = len(t_states)
        nbatch_train = nbatch//self.minibatch
        assert nbatch % self.minibatch == 0
        assert nbatch == len(t_actions)
        states = np.asarray(t_states, dtype=np.float32)
        olds = np.asarray(t_old, dtype=np.float32)
        pactions = self._one_hot(t_actions, self.n_actions)
        inds = np.arange(nbatch)
        for _ in range(self.params["epochs"]):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                feed = {self.n_states: states[mbinds],
                        self.t_actions: pactions[mbinds],
                        self.t_states: olds[mbinds]}
                _, loss = sess.run([self.train_op, self.total_loss], feed)
        return
   
def make_icm(n_actions, n_features, n_icm = 64, depth = 3, activation = tf.nn.relu):
    widths = np.append(np.repeat(n_icm, depth), n_features)
    activations = np.append(np.repeat(activation, depth), tf.nn.sigmoid)
    layers = (widths, activations)
    return ICM(n_actions, n_features, layers)
    
  