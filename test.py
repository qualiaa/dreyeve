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
model = network.model(weights_file="weights_full_train.h5")
opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
model.compile(optimizer='adam',loss='mse',options=opts)

video_folders = glob(c.DATA_DIR + "/[0-9][0-9]")

train_split = int(c.TRAIN_SPLIT * len(video_folders))
validation_split = int(c.VALIDATION_SPLIT * train_split)

test_folders = video_folders[train_split:]

test_examples = KerasSequenceWrapper(DreyeveExamples,c.BATCH_SIZE,test_folders)

model.evaluate_generator(test_examples,
                    use_multiprocessing=c.USE_MULTIPROCESSING,
                    workers=c.WORKERS)
