#!/usr/bin/env python3

from glob import glob

import numpy as np
import tensorflow as tf
import warnings

import network
import consts as c
from Examples import KerasSequenceWrapper
from DreyeveExamples import DreyeveExamples

warnings.filterwarnings("ignore")

print("Loading model...")
model = network.model()
opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
model.compile(optimizer='adam',loss='mse',options=opts)

video_folders = glob("DREYEVE_DATA/[0-9][0-9]")

train_split = int(c.TRAIN_SPLIT * len(video_folders))
validation_split = int(c.VALIDATION_SPLIT * train_split)

train_folders = video_folders[:train_split][:validation_split]
validation_folders = video_folders[:train_split][validation_split:]

train_examples = KerasSequenceWrapper(DreyeveExamples,c.BATCH_SIZE,video_folders)
validation_examples = KerasSequenceWrapper(DreyeveExamples,c.BATCH_SIZE,video_folders)

model.fit_generator(examples,
                    validation=validation_examples,
                    use_multiprocessing=c.USE_MULTIPROCESSING,
                    workers=c.WORKERS)
