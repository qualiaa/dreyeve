#!/usr/bin/env python3

import random
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

import network
import consts as c
import utils.pkl_xz as pkl_xz
from DreyeveExamples import DreyeveExamples
from metrics import kl_vis, cc_vis
from utils.Examples import KerasSequenceWrapper

seed = 7
random.seed(seed)
np.random.seed(seed)
K.tf.set_random_seed(seed)

video_folders = glob(c.DATA_DIR + "/[0-9][0-9]")

train_split = int(c.TRAIN_SPLIT * len(video_folders))
validation_split = int(c.VALIDATION_SPLIT * train_split)

train_folders = video_folders[:train_split][:-validation_split]
validation_folders = video_folders[:train_split][-validation_split:]
#test_folders = video_folders[train_split:]

train = DreyeveExamples(train_folders)
val = DreyeveExamples(validation_folders)

files = glob("history*pkl.xz")

for f in files:
    match = re.fullmatch("history_(\d)_(\d).pkl.xz",f)
    radius, frames = match.groups()
    hist = pkl_xz.load(f)

    train.gaze_radius = radius
    train.gaze_frames = hist
    val.gaze_radius = radius
    val.gaze_frames = hist

    model = network.model("weights_{:d}_{:d}.h5".format(radius,hist))

    X_train, Y_true = train.get_example(0)
    X_train_c, X_train_f, _ = X_train
    Y_true_c, Y_true_f = Y_true
    Y_pred_c, Y_pred_f = model.predict([X],batch_size=1)
    kl_c = kl_vis(Y_true_c, Y_pred_c)
    kl_f = kl_vis(Y_true_f, Y_pred_f)
    cc_c = cc_vis(Y_true_c, Y_pred_c)
    cc_f = cc_vis(Y_true_f, Y_pred_f)

    fig = plt.figure()

    ax = fig.subplot(221)
    subax = ax.subplot(211)
    subax.imshow(X_train_c)
    subax = ax.subplot(212)
    subax.imshow(X_train_f)

    ax = fig.subplot(222)
    subax = ax.subplot(221)
    subax.imshow(Y_true_c)
    subax = ax.subplot(222)
    subax.imshow(Y_true_f)
    subax = ax.subplot(223)
    subax.imshow(Y_pred_c)
    subax = ax.subplot(224)
    subax.imshow(Y_pred_f)

    ax = fig.subplot(223)
    subax = ax.subplot(211)
    subax.imshow(kl_c)
    subax = ax.subplot(212)
    subax.imshow(kl_f)

    ax = fig.subplot(224)
    subax = ax.subplot(211)
    subax.imshow(cc_c)
    subax = ax.subplot(212)
    subax.imshow(cc_f)

    plt.show()




    

    print(hist.keys())

    """
    fig = plt.figure()
    fig.plot
    """
