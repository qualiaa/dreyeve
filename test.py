#!/usr/bin/env python3

import csv
import re
import sys
from pathlib import Path
from glob import glob

import numpy as np
import tensorflow as tf

import settings
import network
import consts as c
from utils.Examples import KerasSequenceWrapper
from DreyeveExamples import DreyeveExamples
from metrics import cross_correlation, kl_divergence

def test(filename):
    match = re.fullmatch("weights_(.*).h5",filename)
    settings.parse_run_name(match.groups()[0])

    # load model
    print("Loading model...")
    model = network.predict_model(weights_file=filename)
    opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    model.compile(optimizer='adam',loss=settings.loss(),
            metrics=["mse", cross_correlation, kl_divergence],
            options=opts)

    # load data
    video_folders = glob(c.DATA_DIR + "/[0-9][0-9]")

    train_split = int(c.TRAIN_SPLIT * len(video_folders))
    validation_split = int(c.VALIDATION_SPLIT * train_split)

    test_folders = video_folders[train_split:]

    test_examples = KerasSequenceWrapper(DreyeveExamples,
            c.BATCH_SIZE,
            test_folders,
            train=False)

    # evaluate data
    results = model.evaluate_generator(test_examples,
                        use_multiprocessing=c.USE_MULTIPROCESSING,
                        workers=c.WORKERS)

    # write to stdout
    print(settings.run_name() + ":",results)

    # write to csv
    results_dict = {"run_name": settings.run_name()}
    results_dict.update(
            dict(zip(model.metrics_names, results))
        )

    csv_file = Path("test_results.csv")
    if not csv_file.exists():
        with csv_file.open("w") as f:
            writer = csv.DictWriter(f, results_dict.keys())
            writer.writeheader()
    else:
        with csv_file.open("a") as f:
            writer = csv.DictWriter(f, results_dict.keys())
            writer.writerow(results_dict)

if __name__ == "__main__":
    import os
    import warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")

    if len(sys.argv) < 2:
        sys.stderr.write("Must provide one or more weight files as arguments")
        model = network.model()
        model.compile(optimizer='adam',loss='mse',
                metrics=[cross_correlation,kl_divergence])
        print(model.metrics_names)
        sys.exit(1)
    for filename in sys.argv[1:]:
        test(filename)
