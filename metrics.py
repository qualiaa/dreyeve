import keras.backend as K
import numpy as np

import consts as c

def kl_divergence(y_true, y_pred):
    eps = K.epsilon() # 1.19e-07
    denom = (eps + y_true)
    ratio = y_pred/denom
    return K.sum(K.tf.multiply(y_pred,K.log(eps + ratio)))


def cross_correlation(y_true, y_pred):
    return K.mean(K.dot(y_true - K.mean(y_true), y_pred - K.mean(y_pred)))

def kl_vis(y_true, y_pred):
    eps = c.EPS
    denom = (eps + y_true)
    ratio = y_pred/denom
    return y_pred * np.log(eps + ratio)

def cc_vis(y_true, y_pred):
    return y_true * y_pred / np.sqrt(np.sum(y_true**2 + y_pred**2))
