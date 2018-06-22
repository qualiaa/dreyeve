#!/usr/bin/env python3

import sys
from glob import glob

import numpy as np
import tensorflow as tf
import warnings

from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard

import network
import consts as c
import utils.pkl_xz as pkl_xz
from utils.Examples import KerasSequenceWrapper
from DreyeveExamples import DreyeveExamples

def train(gaze_radius = 16, gaze_frames = 16):
    #warnings.filterwarnings("ignore")

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

    callbacks = [
        TerminateOnNaN(),
        ModelCheckpoint(
            c.CHECKPOINT_DIR + "/" +
                "{:d}_{:d}".format(gaze_radius,gaze_frames) +
                "_epoch_{epoch:02d}_loss_{val_loss:.2f}.h5",
            save_weights_only=True,
            period=2),
        TensorBoard(
            log_dir=c.LOG_DIR+
                "/{:d}_{:d}".format(gaze_radius,gaze_frames),
            #histogram_freq=10,
            batch_size=c.BATCH_SIZE,write_images=True,
            write_graph=True)
    ]

    history = model.fit_generator(train_examples,
                        steps_per_epoch=c.TRAIN_STEPS,
                        epochs=c.EPOCHS,
                        callbacks=callbacks,
                        validation_data=validation_examples,
                        validation_steps=c.VALIDATION_STEPS,
                        use_multiprocessing=c.USE_MULTIPROCESSING,
                        workers=c.WORKERS)


    print("Saving weights and history")
    model.save_weights("weights_{:d}_{:d}.h5".format(gaze_radius,gaze_frames))
    pkl_xz.save(history.history,"history_{:d}_{:d}.pkl.xz".format(gaze_radius,gaze_frames))

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    nargs = len(sys.argv) - 1
    if nargs % 2 == 1 or nargs == 0:
        sys.stderr.write(
            "Must provide [[gaze radius], [gaze_frames], ...] as arguments\n")
        sys.exit(1)

    args = [int(arg) for arg in sys.argv[1:]]
    args = zip(args[::2],args[1::2])

    for radius, frames in args:
        print(radius,frames)
        train(int(radius), int(frames))
