#!/usr/bin/env python3

import shutil
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf

from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard

import network
import metrics
import consts as c
import utils.pkl_xz as pkl_xz
from utils.Examples import KerasSequenceWrapper
from DreyeveExamples import DreyeveExamples

loss_metrics = {
    "kl": metrics.kl_divergence,
    "cc": metrics.cross_correlation,
    "mse": "mse"
}

def train(gaze_settings=(16,16), loss_name="kl"):
    gaze_radius, gaze_frames = gaze_settings
    loss = loss_metrics[loss_name]
    run_name = "{}_{:d}_{:d}".format(loss_name, gaze_radius, gaze_frames)

    tb_path = Path(c.TB_DIR, run_name)
    if tb_path.exists():
        print("TensorBoard logdir", tb_path, "already exists. Overwrite [y/N]")
        r = input()
        if r.lower()[0] == "y":
            shutil.rmtree(str(tb_path))
        else:
            return

    print("Loading model...")
    model = network.model()
    opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    model.compile(optimizer='adam',loss=loss, options=opts)

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
                run_name + 
                "_epoch_{epoch:02d}_loss_{val_loss:.2f}.h5",
            save_weights_only=True,
            period=2),
        TensorBoard(
            log_dir=str(tb_path),
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
    model.save_weights("weights_" + run_name + ".h5")
    pkl_xz.save(history.history,"history_" + run_name + ".pkl.xz")

if __name__ == "__main__":
    import os
    import warnings
    import argparse
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss",default="kl")

    def extract_settings(arg):
        result = tuple(int(x) for x in arg.split(","))
        if len(result) != 2:
            raise argparse.ArgumentError()
        return result

    parser.add_argument("gaze_settings",nargs="+",type=extract_settings)
    args = vars(parser.parse_args())
    loss = args["loss"]
    gaze_settings_list = args["gaze_settings"]

    for gaze_settings in gaze_settings_list:
        print(gaze_settings)
        train(gaze_settings,loss)
