import keras.backend as K
import numpy as np

import consts as c

def _normalize(x):
    x_min = K.min(x, axis=[1,2], keepdims=True)
    # this line isn't necessary if last layer is a ReLU
    #x -= (x_min - K.abs(x_min)) / 2
    x /= K.sum(x, axis=[1,2], keepdims=True)
    return x

def kl_divergence(y_true, y_pred):
    y_pred = _normalize(y_pred)
    eps = K.epsilon() # 1.19e-07
    denom = (eps + y_pred)
    ratio = y_true/denom
    return K.sum(K.tf.multiply(y_true,K.log(eps + ratio)))


def cross_correlation(y_true, y_pred):
    return K.mean(K.dot(y_true - K.mean(y_true), y_pred - K.mean(y_pred)))

def kl_vis(y_true, y_pred):
    eps = c.EPS
    denom = (eps + y_true)
    ratio = y_pred/denom
    return y_pred * np.log(eps + ratio)

def cc_vis(y_true, y_pred):
    return y_true * y_pred / np.sqrt(np.sum(y_true**2 + y_pred**2))
