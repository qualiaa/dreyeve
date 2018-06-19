#!/usr/bin/env python3

import sys
from glob import glob

import numpy as np
import tensorflow as tf
import warnings

import network
import consts as c
import utils.pkl_xz as pkl_xz
from utils.Examples import KerasSequenceWrapper
from DreyeveExamples import DreyeveExamples

def train(gaze_radius = 16, gaze_frames = 16):
    warnings.filterwarnings("ignore")

    print("Loading model...")
    model = network.model()
    opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    model.compile(optimizer='adam',loss='mse',options=opts)

    video_folders = glob(c.DATA_DIR + "/[0-9][0-9]")

    train_split = int(c.TRAIN_SPLIT * len(video_folders))
    validation_split = int(c.VALIDATION_SPLIT * train_split)

    train_folders = video_folders[:train_split][:-validation_split]
    validation_folders = video_folders[:train_split][-validation_split:]

    seq = lambda x: KerasSequenceWrapper(DreyeveExamples,
            c.BATCH_SIZE, x,
            gaze_radius = gaze_radius,
            gaze_frames = gaze_frames)

    train_examples = seq(train_folders)

    validation_examples = seq(validation_folders)

    history = model.fit_generator(train_examples,
                        validation_data=validation_examples,
                        use_multiprocessing=c.USE_MULTIPROCESSING,
                        workers=c.WORKERS)


    print("Saving weights and history")
    model.save_weights("weights_{:d}_{:d}.h5".format(gaze_radius,gaze_frames))
    pkl_xz.save(history,"history_{:d}_{:d}.pkl.xz".format(gaze_radisu,gaze_frames))

if __name__ == "__main__":
    nargs = len(sys.argv) - 1
    if nargs % 2 == 1 or nargs == 0:
        sys.stderr.write(
            "Must provide [[gaze radius], [gaze_frames], ...] as arguments\n")
        sys.exit(1)
    
    for radius, frames in sys.argv[1:]:
        train(int(radius), int(frames))
