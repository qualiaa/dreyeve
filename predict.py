#!/usr/bin/env python3

import re
import sys
from pathlib import Path
from glob import glob

import numpy as np
import tensorflow as tf

import network
import consts as c
from utils.Examples import KerasSequenceWrapper
from DreyeveExamples import DreyeveExamples
from metrics import cross_correlation, kl_divergence

def predict(weights_filename):
    match = re.fullmatch("weights_(.*).pkl.xz", weights_filename)
    settings.parse_run_name(match.groups()[0])

    # load model
    print("Loading model...")
    model = network.predict_model(weights_file=weights_filename)
    opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    model.compile(optimizer='adam',loss=settings.loss(),
            metrics=["mse",cross_correlation,kl_divergence],
            options=opts)

    # load data
    video_folders = glob(c.DATA_DIR + "/[0-9][0-9]")
    test_folders = video_folders[train_split:]
    test_examples = KerasSequenceWrapper(DreyeveExamples,
            c.BATCH_SIZE,
            test_folders)

    # evaluate data
    results = model.predict(test_examples,
                        use_multiprocessing=c.USE_MULTIPROCESSING,
                        workers=c.WORKERS)


    # write to stdout
    print(settings.run_name() + ":",results)


if __name__ == "__main__":
    import os
    import warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")

    if len(sys.argv) < 2:
        sys.stderr.write("Must provide one or more weight files as arguments")
        sys.exit(1)
    for filename in sys.argv[1:]:
        predict(filename)
