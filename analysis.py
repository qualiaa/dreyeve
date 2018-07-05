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
    radius, frames = [int(x) for x in match.groups()]
    hist = pkl_xz.load(f)

    train.gaze_radius = radius
    train.gaze_frames = frames
    val.gaze_radius = radius
    val.gaze_frames = frames

    model = network.model("weights_{:d}_{:d}.h5".format(radius,frames))

    X_train, Y_true = train.get_example(0)
    X_train = [X[None,:] for X in X_train]
    X_train_c, X_train_f, _ = X_train
    Y_true_c, Y_true_f = Y_true
    Y_pred_c, Y_pred_f = model.predict(X_train,batch_size=1)
    kl_c = kl_vis(Y_true_c, Y_pred_c)
    kl_f = kl_vis(Y_true_f, Y_pred_f)
    cc_c = cc_vis(Y_true_c, Y_pred_c)
    cc_f = cc_vis(Y_true_f, Y_pred_f)

    fig = plt.figure()

    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
    ax1.imshow(X_train_c)
    ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
    ax2.imshow(X_train_f)

    ax3 = plt.subplot2grid((4, 4), (0, 2))
    ax3.imshow(Y_true_c)
    ax4 = plt.subplot2grid((4, 4), (0, 3))
    ax4.imshow(Y_true_f)
    ax5 = plt.subplot2grid((4, 4), (1, 2))
    ax5.imshow(Y_pred_c)
    ax6 = plt.subplot2grid((4, 4), (1, 3))
    ax6.imshow(Y_pred_f)

    ax7 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
    ax7.imshow(kl_c)
    ax8 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    ax8.imshow(kl_f)

    ax9 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
    ax9.imshow(cc_c)
    ax10 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    ax10.imshow(cc_f)

    fig.show()




    

    print(hist.keys())

    """
    fig = plt.figure()
    fig.plot
    """
