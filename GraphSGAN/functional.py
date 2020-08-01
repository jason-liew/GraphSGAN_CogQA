import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import math
import numpy as np
def log_sum_exp(x, axis = 1):
    m = tf.raw_ops.Max(input=x, axis = 1)
    return m + tf.math.log(tf.raw_ops.Sum(input=tf.exp(x - tf.expand_dims(m,1)), axis = axis))
def normalize_infnorm(x, eps=1e-8):
    assert type(x) == np.ndarray
    return x / (abs(x).max(axis = 0) + 1e-8)   

class LinearWeightNorm(keras.layers.Layer):
    def __init__(self, in_features, out_features, bias=True, weight_scale=None, weight_init_stdv=0.1):
        super(LinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = tf.Variable(initial_value=tf.random_normal_initializer(stddev=1)(shape=[out_features, in_features]) * weight_init_stdv)
        self.bias = tf.Variable(tf.zeros(out_features))
        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = tf.Variable(tf.ones(out_features, 1) * weight_scale)
        else:
            self.weight_scale = 1 
    def call(self, x):
        W = self.weight * self.weight_scale / tf.sqrt(tf.raw_ops.Sum(input=(self.weight ** 2), axis = 1, keep_dims = True))
        return tf.matmul(x,tf.transpose(W)) + self.bias

class Linear(keras.layers.Layer):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = tf.Variable(tf.random.uniform([out_features, in_features]))

    def call(self, x):
        return tf.matmul(x,tf.transpose(self.weight))


def pull_away_term(x):
    '''pull-away loss

    Args:
        x: type=> torch Tensor or Variable, size=>[batch_size * feature_dim], generated samples

    Return:
        scalar Loss
    '''
    x = tf.math.l2_normalize(x,axis=1)
    pt = tf.matmul(x,tf.transpose(x)) ** 2
    return (tf.reduce_sum(pt) - tf.reduce_sum(tf.raw_ops.Diag(diagonal=pt)) )/ (len(x) * (len(x) - 1))