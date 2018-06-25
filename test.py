#!/usr/bin/env python3

import sys
import re
from glob import glob

import numpy as np
import tensorflow as tf

from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard

import network
import consts as c
from utils.Examples import KerasSequenceWrapper
from DreyeveExamples import DreyeveExamples
from metrics import cross_correlation, kl_divergence


def test(filename):
    gaze_radius, gaze_frames = [int(x) for x in
            re.match(".*_([0-9]+)_([0-9]+).h5",filename).groups()]

    print("Loading model...")
    model = network.model(weights_file=filename)
    opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    model.compile(optimizer='adam',loss='mse',
            metrics=[cross_correlation,kl_divergence],
            options=opts)

    video_folders = glob(c.DATA_DIR + "/[0-9][0-9]")

    train_split = int(c.TRAIN_SPLIT * len(video_folders))
    validation_split = int(c.VALIDATION_SPLIT * train_split)

    test_folders = video_folders[train_split:]

    test_examples = KerasSequenceWrapper(DreyeveExamples,
            c.BATCH_SIZE,
            test_folders,
            gaze_radius=gaze_radius,
            gaze_frames=gaze_frames)

    results = model.evaluate_generator(test_examples,
                        use_multiprocessing=c.USE_MULTIPROCESSING,
                        workers=c.WORKERS)

    print(filename + ":",results)

if __name__ == "__main__":
    import os
    import warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")

    if len(sys.argv) < 2:
        sys.stderr.write("Must provide one or more weight files as arguments")
        sys.exit(1)
    for filename in sys.argv[1:]:
        test(filename)
