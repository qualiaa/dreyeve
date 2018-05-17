#!/usr/bin/env python3

from glob import glob

import numpy as np
import tensorflow as tf

import network
from Examples import KerasSequenceWrapper
from DreyeveExamples import DreyeveExamples

print("Loading model...")
model = network.model()
opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
model.compile(optimizer='adam',loss='mse',options=opts)

video_folders = glob("DREYEVE_DATA/[0-9][0-9]")

examples = KerasSequenceWrapper(DreyeveExamples, 5,video_folders)

model.fit_generator(examples)
#model.fit_generator(examples,
                    #use_multiprocessing=True,
                    #workers=4)
