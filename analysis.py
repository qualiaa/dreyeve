#!/usr/bin/env python3

import argparse
import random
import re
from glob import glob
from pathlib import Path

import numpy as np
import keras.backend as K

import network
import settings
import consts as c
import utils.pkl_xz as pkl_xz
from metrics import kl_vis, cc_vis
from utils.Examples import KerasSequenceWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--png", action="store_true")
parser.add_argument("history_files", nargs="*")
args = vars(parser.parse_args())
files = args["history_files"]
if len(files) == 0:
    files = glob("history*.pkl.xz")
save_png = args["png"]
output_path = Path("figures")

if save_png:
    import matplotlib as mpl
    mpl.use("AGG")
    output_path.mkdir(exist_ok=True)
import matplotlib.pyplot as plt

# must set MPL backend before importing DreyeveExamples, as pims imports MPL
from DreyeveExamples import DreyeveExamples

seed = 7
random.seed(seed)
np.random.seed(seed)
K.tf.set_random_seed(seed)

video_folders = glob(c.DATA_DIR + "/[0-9][0-9]")

train_split = int(c.TRAIN_SPLIT * len(video_folders))
validation_split = int(c.VALIDATION_SPLIT * train_split)

def wait():
    print("Press enter to continue...")
    input()

for f in files:
    match = re.fullmatch("history_(.*).pkl.xz",f)
    title = match.groups()[0]

    hist = pkl_xz.load(f)
    # keys dict_keys(['val_loss', 'val_coarse_output_loss',
    # 'val_fine_output_loss', 'loss', 'coarse_output_loss', 'fine_output_loss'])

    loss_c = hist["coarse_output_loss"]
    loss_f = hist["fine_output_loss"]
    loss_c_val = hist["val_coarse_output_loss"]
    loss_f_val = hist["val_fine_output_loss"]

    def plot1(ax):
        ax.plot(loss_c[1:])
        ax.plot(loss_f[1:])
        ax.set_yticklabels("loss")
        ax.set_xticklabels("epoch")

    def plot2(ax):
        ax.plot(loss_c_val[1:])
        ax.plot(loss_f_val[1:])
        ax.set_yticklabels("loss")
        ax.set_xticklabels("epoch")


    if save_png:
        path = output_path/title
        path.mkdir(exist_ok=True)
        train_path, val_path = [
                (path/mode).with_suffix(".png")
                for mode in ["train", "val"]]
        fig = plt.figure()
        fig.suptitle("train")
        plot1(fig.gca())
        fig.savefig(train_path)
        fig = plt.figure()
        fig.suptitle("validation")
        plot2(fig.gca())
        fig.savefig(val_path)

    else:
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(121)
        ax.title.set_text("train")
        plot1(ax)
        ax = fig.add_subplot(122)
        ax.title.set_text("validation")
        plot2(ax)
        fig.show()
        wait()

    plt.close(fig)


train_folders = video_folders[:train_split][:-validation_split]
validation_folders = video_folders[:train_split][-validation_split:]
#test_folders = video_folders[train_split:]

train = DreyeveExamples(train_folders,seed=seed)


for f in files:
    print(f)
    match = re.fullmatch("history_(.*).pkl.xz",f)
    title=match.groups()[0]
    settings.parse_run_name(title)

    model = network.model("weights_"+settings.run_name() +".h5")

    # show at most 5 examples
    for shuffled_index in range(5):
        example_id = train.example_queue[shuffled_index]
        X_train, Y_true = train.get_example(example_id)
        X_train_c, X_train_f = [np.moveaxis(X[:,-1,...],0,2) for X in X_train[:2]]
        X_train_c, X_train_f = [(X - X.min())/(X.max()-X.min()) for X in [X_train_c, X_train_f]]
        X_train = [X[None,:] for X in X_train]
        Y_true_c, Y_true_f = Y_true
        Y_pred_c, Y_pred_f = model.predict(X_train,batch_size=1)
        """
        print(X_train_c.shape, X_train_f.shape)
        print((X_train_f.min(), X_train_f.max()))
        """
        Y_pred_c, Y_pred_f = [np.squeeze(Y) for Y in [Y_pred_c, Y_pred_f]]
        Y_pred_c, Y_pred_f = [Y/Y.sum() for Y in [Y_pred_c, Y_pred_f]]


        kl_c = kl_vis(Y_true_c, Y_pred_c)
        """
        print(kl_c.shape)
        print(kl_c.dtype)
        print(kl_c.max(),kl_c.min())
        """
        kl_f = kl_vis(Y_true_f, Y_pred_f)
        cc_c = cc_vis(Y_true_c, Y_pred_c)
        cc_f = cc_vis(Y_true_f, Y_pred_f)

        fig = plt.figure(figsize=(20,20), dpi=80)
        fig.suptitle("{} example {:d}".format(f, shuffled_index))

        ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
        ax1.imshow(X_train_c)
        ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
        ax2.imshow(X_train_f)

        ax3 = plt.subplot2grid((4, 4), (0, 2))
        ax3.imshow(Y_true_c)
        ax5 = plt.subplot2grid((4, 4), (0, 3))
        ax5.imshow(Y_pred_c)
        ax4 = plt.subplot2grid((4, 4), (1, 2))
        ax4.imshow(Y_true_f)
        ax6 = plt.subplot2grid((4, 4), (1, 3))
        ax6.imshow(Y_pred_f)

        def centred_graph(ax, array):
            x = np.abs(array).max()
            im = ax.imshow(array, vmin=-x,vmax=x,cmap=plt.get_cmap("seismic"))
            fig.colorbar(im,ax=ax)
            plt.annotate("Val: {}".format(array.sum()),
                    xy=(10,10),
                    xycoords="axes points")


        ax7 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
        ax7.title.set_text("Cropped KL Divergence")
        centred_graph(ax7, kl_c)
        ax8 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
        ax8.title.set_text("KL Divergence")
        centred_graph(ax8, kl_f)

        ax9 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
        ax9.title.set_text("Cropped Correlation Coefficient")
        im = ax9.imshow(cc_c)
        fig.colorbar(im,ax=ax9)
        ax10 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
        ax10.title.set_text("Correlation Coefficient")
        im = ax10.imshow(cc_f)
        fig.colorbar(im,ax=ax10)

        if save_png:
            fig.savefig(output_path/title/
                "example_{:d}.png".format(shuffled_index))
            plt.close(fig)
        else:
            fig.show()
            print("Show next? [y/N]")
            ans = input().lower()
            plt.close(fig)
            if ans == "y": break
