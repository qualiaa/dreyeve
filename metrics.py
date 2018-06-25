import keras.backend as K

def kl_divergence(y_true, y_pred):
    eps = K.epsilon() # 1.19e-07
    denom = (eps + y_true)
    numer = eps + y_pred
    ratio = numer/denom
    return K.sum(K.tf.multiply(y_pred,K.log(denom)))


def cross_correlation(y_true, y_pred):
    return K.mean(K.dot(y_true - K.mean(y_true), y_pred - K.mean(y_pred)))
