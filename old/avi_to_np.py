#!/usr/bin/env python3

import os
import os.path
import glob
import sys

import pims
import numpy as np
import xz_compress as xz

from warnings import warn
from functools import reduce, partial
from itertools import repeat
from operator import mul
from multiprocessing import Pool

from imageio import imwrite
from skimage.transform import resize


TARGET_SHAPE=(448,448)

product = lambda l: reduce(mul,l,1)
def resize_frame_tensor(frame_tensor,target_shape):
    target_shape=(frame_tensor.shape[0],*target_shape,frame_tensor.shape[-1])
    return resize(frame_tensor,target_shape,anti_aliasing=True,mode="constant")

Reader = pims.ImageIOReader

input_folder = "DREYEVE_DATA"
"""
input_path_format_string = input_folder + "/{:02d}/video_garmin.avi"

output_folder = input_folder
output_path_format_string = output_folder + "/{:02d}/garmin_frames/"
"""

def decimate_video(folder,frames_per_batch=16,target_shape=TARGET_SHAPE):
    print(folder)
    input_path = folder+"/video_garmin.avi"
    output_folder = folder+"/garmin_frames"

    video = Reader(input_path)
    os.makedirs(output_folder,exist_ok=True)

    output_shape = (len(video),*target_shape,video.frame_shape[-1])
    n_batches = len(video)//frames_per_batch

    for batch in range(n_batches):
        output_path = output_folder + "/batch_{:04d}.pkl.xz".format(batch)
        if os.path.exists(output_path):
            warn("Skipping batch {:d} - file exists".format(batch))
            continue

        if batch+1 % 100:
            print("[{:d}] Batch {:d} of {:d}".format(os.getpid(), batch,n_batches))
        start_frame = frames_per_batch*batch

        # create a slice for the current frames
        frame_slice = slice(frames_per_batch*batch,frames_per_batch*(batch+1))
        # load the current frames into a list
        frame_list = video[frame_slice]
        # create a numpy tensor from the frame list
        frame_tensor = np.stack(frame_list,axis=0)
        frame_tensor = resize_frame_tensor(frame_tensor,target_shape)
        # resize operation converts to float and normalizes - convert back
        frame_tensor = (frame_tensor*255).astype(np.uint8)

        # save numpy tensor of full batch
        xz.save_pkl_xz(frame_tensor,output_path)

if len(sys.argv) > 1:
    thread_n = sys.argv[1]
    n_threads = sys.argv[2]

folders = glob.glob(input_folder+"/[0-9][0-9]")
with Pool(4) as p:
    p.map(decimate_video,folders)

"""
import matplotlib.pyplot as plt

plt.figure()
plt.subplot(411)
plt.imshow(numpy_video[0,...])
plt.subplot(412)
plt.imshow(numpy_video[1,...])
plt.subplot(413)
plt.imshow(numpy_video[-2,...])
plt.subplot(414)
plt.imshow(numpy_video[-1,...])

plt.show()
"""
