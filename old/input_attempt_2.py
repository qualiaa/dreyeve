#!/usr/bin/env python3

import csv
import math
import os
import random
import re
import sys
import numpy as np
from warnings import warn
from itertools import accumulate
from functools import reduce
from operator import add

from glob import glob
from skimage.transform import resize

from BatchedVideo import BatchedVideo
from utils.ShuffleQueue import ShuffleQueue

DATA_DIR = "DREYEVE_DATA"

VIDEO_SHAPE=(1080,1920,3)

class Labels:
    FIXATION = 0
    SACCADE = 1
    BLINK = 2

LABEL_NAMES = { s:getattr(Labels,s) for s in Labels.__dict__.keys()
        if not s.startswith("_")}

def _resize_frame_tensor(frame_tensor,target_shape):
    target_shape=(frame_tensor.shape[0],*target_shape,frame_tensor.shape[-1])
    return resize(frame_tensor,target_shape,anti_aliasing=True,mode='reflect')

class Examples:
    def __init__(self,
                 folders,
                 example_shape=(112,112),
                 frames_per_example=16,
                 seed=None):
        self.frames_per_example=frames_per_example
        self.example_shape=example_shape
        print("Loading video files...")
        self.videos = [
                BatchedVideo(folder+"/garmin_frames")
                for folder in folders]
        print("Done")
        lengths = list(map(len,self.videos))
        self._lengths = list(accumulate(lengths))
        self.num_examples = reduce(add,lengths)

        self.seed = seed or random.randrange(sys.maxsize)
        self.example_queue = ShuffleQueue(range(self.num_examples),self._rand)

    def __getitem__(self,n):
        return self.get_example(n)

    def __next__(self):
        return self.next_example()

    def __len__(self):
        return self.num_examples

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self,seed):
        self._seed = seed
        self._rand = random.Random(self._seed)
    
    @property
    def epoch(self):
        return self.example_queue.epoch

    def get_example(self, example_id):
        print("Example {}".format(example_id))
        vid,frame = self._get_video_and_frame_number_from_id(example_id)
        if len(eye_positions[vid][frame]) == 0:
            raise IndexError("Example has no ground truth")
        video = self.videos[vid]
        tensor=self.videos[vid][frame]
        frame_shape = tensor.shape[1:3]

        crop_slice = random_crop_slice(frame_shape,
                self.example_shape, self._rand)

        tensor_cropped=tensor[[slice(None),*crop_slice,slice(None)]]
        tensor_resized = _resize_frame_tensor(tensor, self.example_shape)

        attention = get_scaled_attention_map(vid, frame, frame_shape)
        attention_cropped = attention[crop_slice]
        attention_resized = get_scaled_attention_map(vid, frame,
                self.example_shape)

        return (tensor_cropped,
                tensor_resized,
                attention_cropped,
                attention_resized)

    def next_example(self):
        example = None
        while example is None:
            example_id = next(self.example_queue)
            try:
                example = self.get_example(example_id)
            except (ValueError, IndexError) as e:
                warn("Exception raised for example {:d}: {}".format(
                        example_id,e.args),
                    RuntimeWarning)
                continue
            except Exception as e:
                print("Exception raised for example {:d}".format(example_id))
                raise e

        return example

    def next_batch(self,batch_size):
        examples=[]
        for i in range(batch_size):
            examples.append(self.next_example())
            if len(self.example_queue) == 0:
                break

        examples = map(np.stack,zip(*examples))

        return examples

    def _get_video_and_frame_number_from_id(self,example_id):
        if i >= len(self):
            raise IndexError("Example ID exceeds number of samples")
        start_id = 0
        for video_number, end_id in enumerate(self._lengths):
            if end_id > example_id:
                frame_number = example_id - start_id
                if frame_number+1 < self.frames_per_example:
                    raise IndexError(
                            "Frame {} below frames_per_example "
                            "for example id {}".format(frame_number,example_id))
                return video_number, frame_number
            start_id = end_id
        raise RuntimeError("Frame not found")

video_folders = glob(DATA_DIR+"/[0-9][0-9]")

from collections import defaultdict
eye_positions = list()

print("Reading eye position data...")
for i,folder in enumerate(video_folders):
    eye_pos = defaultdict(list)
    with open(folder+"/etg_samples.txt") as f:
        eye_data = csv.reader(f, delimiter=' ', dialect="unix")
        next(eye_data) # skip csv header
        for row in eye_data:
            _, frame, x, y, label, _ = row
            try:
                label_id = LABEL_NAMES[label.upper()]
            except KeyError:
                continue
            frame = int(frame)
            coords = np.array([float(y),float(x)])
            
            if label_id != Labels.FIXATION: continue
            if any(a > b for a,b in zip(coords,VIDEO_SHAPE)) or any(
                    a < 0 or math.isnan(a) for a in coords):
                continue
            #eye_pos[frame].append((label_id, coords))
            eye_pos[frame].append(coords)
        eye_pos[frame]=np.array(eye_pos[frame])
    eye_positions.append(eye_pos)
print("Done")

def _generate_attention_map(gt_coords, size):
    attention_map = np.zeros(size)
    if len(gt_coords) == 0: return attention_map
    for x, y in gt_coords:
        ix = int(x); iy = int(y)
        x2 = x-ix; y2 = y-iy
        x1 = 1-x2; y1 = 1-y2
        value=np.array([[x1*y1, x2*y1],
                        [x1*y2, x2*y2]])
        value/=len(gt_coords)
        attention_map[ix:ix+2, iy:iy+2]+=value

    return attention_map

def _scale_gt_coords(coords, target_shape):
    # calculate scale amount
    source_shape=VIDEO_SHAPE[0:2]
    if source_shape != target_shape:
        #print("Input: {}".format(coords))
        scale=np.array([t/s for s, t in zip(source_shape, target_shape)])
        #print("Scale: {}".format(scale))
        #coords=[(y*scale[0], x*scale[1]) for y, x in coords]
        coords=coords*scale
        """ print("Source shape: {}".format(source_shape))
        print("Target shape: {}".format(target_shape))
        print("Output: {}".format(coords)) """
    return coords

def get_scaled_attention_map(vid, frame, target_shape):
    coords = eye_positions[vid][frame]
    # remove non-fixations
    if len(coords) == 0: raise ValueError("No coords for attention map")
    # strip label string
    coords=_scale_gt_coords(coords, target_shape)
    attention_map=_generate_attention_map(coords, target_shape)
    return attention_map
    

def random_crop_slice(input_shape, target_shape,rand=random):
    starts = [rand.randrange(s-c) for s,c in zip(input_shape, target_shape)]
    slices = [slice(s, s+c) for s, c in zip(starts, target_shape)]
    return slices

if __name__ == "__main__":
    """
    for i,folder in enumerate(video_folders):
        if i < 7: continue
        print("Testing video {:d}".format(i+1))
        vid = BatchedVideo(folder+"/garmin_frames")
        vid.validate_batches()
    """

    examples = Examples(video_folders)

    batch = 0
    while examples.epoch < 2:
        print("batch {:d}".format(batch))
        examples.next_batch(50)
        batch+=1
