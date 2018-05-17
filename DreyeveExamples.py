from functools import reduce
from itertools import accumulate
from operator import add
from pathlib import Path, PurePath
from warnings import warn

import numpy as np
from imageio.core.format import CannotReadFrameError
from pims import ImageIOReader as Reader
from skimage.transform import resize

from Examples import Examples
from random_crop_slice import random_crop_slice
import eye_data
from consts import *

class DreyeveExamples(Examples):
    def __init__(self,
                 folders,
                 example_shape=(112,112),
                 frames_per_example=16,
                 seed=None):
        self.frames_per_example=frames_per_example
        self.example_shape=example_shape
        print("Loading eye data...")
        self.eye_positions = eye_data.read(folders)
        print("Loading video files...")
        self.videos = {
                x: [Reader(str(Path(folder, "garmin_resized_{:d}.avi".format(x))))
                    for folder in folders]
                for x in [112,448]}
        print("Done")
        lengths = list(map(len,self.videos[112]))
        self._lengths = list(accumulate(lengths))
        self.num_examples = reduce(add,lengths)

        super().__init__(seed)

    def __len__(self):
        return self.num_examples
    
    def get_example(self, example_id):
        vid_id,frame = self._get_video_and_frame_number_from_id(example_id)
        eye_coords = self.eye_positions[vid_id][frame]

        if len(eye_coords) == 0:
            raise ValueError("Example has no ground truth")

        vid448 = self.videos[448][vid_id]
        vid112 = self.videos[112][vid_id]

        tensor = _get_frame_tensor(vid448,frame)

        frame_shape = tensor.shape[2:4]

        crop_slice = random_crop_slice(frame_shape,
                self.example_shape, self._rand)

        tensor_cropped = tensor[[slice(None),slice(None),*crop_slice]]
        tensor_resized = _get_frame_tensor(vid112,frame)

        attention = get_scaled_attention_map(eye_coords,frame_shape)
        attention_cropped = attention[crop_slice]
        attention_resized = get_scaled_attention_map(eye_coords,(448,448))
        attention_cropped = np.expand_dims(attention_cropped,0)
        attention_resized = np.expand_dims(attention_resized,0)

        return ([tensor,
                tensor_cropped,
                tensor_resized],
                [attention_cropped,
                attention_resized])

    def next_example(self):
        try:
            super().next_example()
        except CannotReadFrameError:
            warn("Corrupt frame", RuntimeWarning)


    
    def _get_video_and_frame_number_from_id(self,example_id):
        if example_id >= len(self):
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


"""
def _resize_frame_tensor(frame_tensor,target_shape):
    target_shape=(frame_tensor.shape[0],*target_shape,frame_tensor.shape[-1])
    return resize(frame_tensor,target_shape,anti_aliasing=True,mode='reflect')
"""


def _get_frame_tensor(video, frame_of_interest, num_frames=16):
    """ return N stacked, resized frames from frame [I-N, I] """
    foi = frame_of_interest
    if foi < num_frames - 1:
        raise IndexError("Frame of interest " + foi +
                         " must have 15 preceding frames")
    frame_slice = video[foi-num_frames+1:foi+1]
    frame_tensor = np.stack(frame_slice, axis=0)
    frame_tensor = np.transpose(frame_tensor,(3,0,1,2))
    return frame_tensor.astype(np.float32)


def _generate_attention_map(gt_coords, size):
    attention_map = np.zeros(size,dtype=np.float32)
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


def get_scaled_attention_map(eye_coords, target_shape):
    # remove non-fixations
    if len(eye_coords) == 0: raise ValueError("No coords for attention map")
    # strip label string
    eye_coords=eye_data.scale_to_shape(eye_coords, target_shape)
    attention_map=_generate_attention_map(eye_coords, target_shape)
    return attention_map
    
#video_folders = glob(DATA_DIR+"/[0-9][0-9]")
#training_examples = Examples(video_folders)
