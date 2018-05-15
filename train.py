#!/usr/bin/env python3

from glob import glob

import numpy as np

import network
from Examples import KerasSequenceWrapper

print("Loading model...")
model = network.model()

video_folders = glob("DREYEVE_DATA/[0-9][0-9]")

examples = KerasSequenceWrapper(5,video_folders)

model.fit_generator(examples)
#model.fit_generator(examples,
                    #use_multiprocessing=True,
                    #workers=4)
