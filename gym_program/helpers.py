#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 12:33:21 2018

@author: matthewszhang
"""
import tensorflow as tf
import numpy as np

INIT_SCALE = 1.43

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

def _linear(t_in, n_out):
    v_w = tf.get_variable(
            "w",
            shape=(t_in.get_shape()[-1], n_out),
            initializer=tf.contrib.layers.xavier_initializer())
    v_b = tf.get_variable(
            "b",
            shape=n_out,
            initializer=tf.constant_initializer(0))
    if len(t_in.get_shape()) == 2:
        return tf.einsum("ij,jk->ik", t_in, v_w) + v_b
    elif len(t_in.get_shape()) == 3:
        return tf.einsum("ijk,kl->ijl", t_in, v_w) + v_b
    else:
        assert False
        
def ff(t_in, widths, activations, name = None):
    assert len(t_in.get_shape()) in (2, 3)
    assert len(widths) == len(activations)
    prev_layer = t_in
    
    scope_base = ""
    if name != None:
        scope_base = name
    
    for i_layer, (width, act) in enumerate(zip(widths, activations)):
        with tf.variable_scope(scope_base + str(i_layer)):
            layer = _linear(prev_layer, width)
            if act is not None:
                layer = act(layer)
        prev_layer = layer
    return prev_layer