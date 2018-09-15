import keras.backend as K
import numpy as np

import consts as c

def _normalize(x):
    # these two lines aren't necessary if last layer is a ReLU
    #x_min = K.min(x, axis=[1,2], keepdims=True)
    #x -= (x_min - K.abs(x_min)) / 2
    x /= K.sum(x, axis=[1,2], keepdims=True)
    return x

def kl_divergence(y_true, y_pred):
    Q = y_true; P = _normalize(y_pred)
    eps = K.epsilon() # 1.19e-07
    return K.sum(K.tf.multiply(Q, K.log(eps + Q/(eps + P))), axis=[1,2])


def cross_correlation(y_true, y_pred):
    X = y_true; Y = y_pred
    X_stddev = K.std(X, axis=[1,2])
    Y_stddev = K.std(Y, axis=[1,2])
    X_mean = K.mean(X, axis=[1,2])
    Y_mean = K.mean(Y, axis=[1,2])
    X_mean = K.expand_dims(K.expand_dims(X_mean, 1), 1)
    Y_mean = K.expand_dims(K.expand_dims(Y_mean, 1), 1)

    denom = K.tf.multiply(X_stddev, Y_stddev)
    denom = K.expand_dims(denom, 1)

    return K.mean(K.dot(X-X_mean, Y-Y_mean),axis=[1,2]) / denom

def kl_vis(y_true, y_pred):
    eps = c.EPS
    Q = y_true
    P = y_pred
    return Q * np.log(eps + Q/(eps + P))

def cc_vis(y_true, y_pred):
    return y_true * y_pred / np.sqrt(np.sum(y_true**2 + y_pred**2))
