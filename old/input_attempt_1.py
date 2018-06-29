#!/usr/bin/env python3

import csv
import pims
import pims.imageio_reader
import numpy as np

from skimage.transform import resize

Reader = pims.ImageIOReader

dashcam_video = Reader("DREYEVE_DATA/01/video_garmin.avi")

#n_frames = len(dashcam_video)

from collections import defaultdict
eye_positions = defaultdict(list)

with open("DREYEVE_DATA/01/etg_samples.txt") as f:
    eye_data = csv.reader(f, delimiter=' ', dialect="unix")
    next(eye_data)
    for row in eye_data:
        _, frame, x, y, label, _ = row
        frame = int(frame)
        x     = float(x)
        y     = float(y)
        eye_positions[frame].append((label, x, y))

#print(len(dashcam_video)/(60*5))
#print(len(etg)/(60*5))

def generate_attention_map(gt_coords, size):
    attention_map = np.zeros(size)
    if not gt_coords: return attention_map
    for x, y in gt_coords:
        ix = int(x); iy = int(y)
        x2 = x-ix; y2 = y-iy
        x1 = 1-x2; y1 = 1-y2
        value=np.array([[x1*y1, x2*y1],
                        [x1*y2, x2*y2]])
        value/=len(gt_coords)
        attention_map[ix:ix+2, iy:iy+2]+=value

    return attention_map

def _get_scaled_gt_coords(frame, target_shape):
    coords = eye_positions[frame]
    # remove non-fixations
    print(coords)
    coords=filter(lambda c: c[0] == "Fixation", coords)
    # strip label string
    coords=map(lambda c: c[1:], coords)
    # calculate scale amount
    source_shape=dashcam_video.frame_shape
    if source_shape != target_shape:
        scale=[t/s for s, t in zip(source_shape, target_shape)]
        coords=[(x*scale[1], y*scale[0]) for x, y in coords]
    return list(coords)

def get_scaled_attention_map(frame, target_shape):
    gt_coords=_get_scaled_gt_coords(frame, target_shape)
    attention_map=generate_attention_map(gt_coords, target_shape)
    return attention_map
    

def resize_frame_tensor(frame_tensor,target_shape):
    target_shape=(frame_tensor.shape[0],*target_shape,frame_tensor.shape[-1])
    return resize(frame_tensor,target_shape,anti_aliasing=True)

def get_frame_tensor(video, frame_of_interest, num_frames=16):
    """ return N stacked, resized frames from frame [I-N, I] """
    foi = frame_of_interest
    if foi < num_frames - 1:
        raise IndexError("Frame of interest " + foi +
                         " must have 15 preceding frames")
    frame_slice = video[foi-num_frames+1:foi+1]
    frame_tensor = np.stack(frame_slice, axis=0)
    return frame_tensor

def random_crop_slice(input_shape, crop_shape):
    starts = [np.random.randint(s-c) for s,c in zip(input_shape, crop_shape)]
    slices = [slice(s, s+c) for s, c in zip(starts, crop_shape)]
    return slices

def get_single_example(frame,resize_shape=(256,256),crop_shape=(112,112)):
    frame_tensor_full=get_frame_tensor(dashcam_video, frame)
    frame_tensor = resize_frame_tensor(frame_tensor_full, resize_shape)

    crop_slice = random_crop_slice(resize_shape, crop_shape)

    frames_cropped=frame_tensor[[slice(None),*crop_slice,slice(None)]]
    attention_full=get_scaled_attention_map(frame, resize_shape)
    attention_cropped=attention_full[crop_slice]

    frames_resized=resize_frame_tensor(frame_tensor_full, crop_shape)
    attention_resized=get_scaled_attention_map(frame, crop_shape)

    return (frames_cropped,frames_resized,attention_cropped,attention_resized)

"""
frame = 16
resize_shape = (256, 256)
crop_shape = (112, 112)
frame_tensor_full=get_frame_tensor(dashcam_video, frame)
frame_tensor = resize_frame_tensor(frame_tensor_full, resize_shape)

crop_slice = random_crop_slice(resize_shape, crop_shape)

frames_cropped=frame_tensor[[slice(None),*crop_slice,slice(None)]]
attention_map_full, coords_full=get_scaled_attention_map_and_coords(frame, resize_shape)
attention_map_cropped=attention_map_full[crop_slice]

frames_resized=resize_frame_tensor(frame_tensor_full, crop_shape)
attention_map_resized,coords_resized=get_scaled_attention_map_and_coords(
        frame, crop_shape)

"""
"""
import matplotlib.pyplot as plt

plt.figure()
plt.subplot(221)
plt.imshow(frames_cropped[0,...])
plt.subplot(222)
plt.imshow(frames_resized[0,...])
plt.subplot(223)
plt.imshow(attention_map_cropped*255)
plt.subplot(224)
plt.imshow(attention_map_resized*255)
plt.show()


def plot_frames(frame):
    frames = np.concatenate(dashcam_video[frame:frame+5], axis=1)

    frame_offset = 0
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(frames)

    for i in range(frame, frame+5):
        if i in eye_positions:
            for label, x, y in eye_positions[i]:
                color = 'r'
                if label == "Fixation": color = 'g'
                if label == "Saccade": color = 'b'

                x += frame_offset
                ax.add_artist(plt.Circle((x, y), 100, color=color, ec='black'))

        frame_offset += dashcam_video.frame_shape[1]

    plt.show()

#plot_frames(0)
"""
