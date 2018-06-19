#!/usr/bin/env python3

import sys
from glob import glob

import numpy as np
import tensorflow as tf
import warnings

import network
import consts as c
from utils.Examples import KerasSequenceWrapper
from DreyeveExamples import DreyeveExamples

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
        gaze_radius = GAZE_RADIUS,
        self.gaze_frames = GAZE_FRAMES,

train_examples = seq(train_folders)

validation_examples = seq(validation_folders)

model.fit_generator(train_examples,
                    validation_data=validation_examples,
                    use_multiprocessing=c.USE_MULTIPROCESSING,
                    workers=c.WORKERS)


print("Saving weights")
model.save_weights("weights_gaussian_16.h5")
