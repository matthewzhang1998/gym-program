#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:33:00 2018

@author: matthewszhang
"""

import numpy as np
import tensorflow as tf
from edward.models import Normal

PARAMS = {'nh':32}

class BNN():
    def __init__(self, nac, nfeat, params = PARAMS):
        self.width = 2 * nfeat + nac
        nh = params["nh"]
        def neural_network(x):
            h = tf.tanh(tf.matmul(x, W_0) + b_0)
            h = tf.tanh(tf.matmul(h, W_1) + b_1)
            h = tf.matmul(h, W_2) + b_2
            return tf.reshape(h, [-1])
        W_0 = Normal(loc=tf.zeros([self.width, nh]), scale=tf.ones([self.width, nh]))
        W_1 = Normal(loc=tf.zeros([nh, nh]), scale=tf.ones([nh, nh]))
        W_2 = Normal(loc=tf.zeros([nh, 1]), scale=tf.ones([nh, 1]))
        b_0 = Normal(loc=tf.zeros(nh), scale=tf.ones(nh))
        b_1 = Normal(loc=tf.zeros(nh), scale=tf.ones(nh))
        b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1))
        
        self.layers = {"layer_1":[W_0, b_0, tf.nn.tanh],
                       "layer_2":[W_1, b_1, tf.nn.tanh],
                       "layer_3":[W_2, b_2, tf.nn.sigmoid]}
        self.sastup = tf.placeholder(dtype=tf.float32, shape=(None,self.width))
        y = Normal(loc=neural_network(self.sastup, self.layers), scale=0.1 * tf.ones(N))