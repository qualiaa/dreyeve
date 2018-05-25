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
        self._lengths = list(map(len,self.videos[112]))
        self._cumulative_lengths = list(accumulate(self._lengths))
        self.total_frames = reduce(add,self._lengths)
        self.valid_indices = self._get_valid_indices()

        eh = {}
        eh[CannotReadFrameError] = lambda e, i: warn(
            "Corrupt frame for video {:d}, frame {:d}".format(
                *self._get_video_and_frame_number_from_id(i)),
            RuntimeWarning)
        eh[ValueError] = lambda e, i: warn(
            "Exception handled for video {:d}, frame{:d}: {}".format(
                *self._get_video_and_frame_number_from_id(i), e.args),
            RuntimeWarning)
        eh[IndexError] = eh[ValueError]
        #self.exception_handlers = eh
        super().__init__(seed)

    def __len__(self):
        return len(self.valid_indices)
    
    def get_example(self, example_id):
        vid_id,frame = self._get_video_and_frame_number_from_id(example_id)

        if frame+1 < self.frames_per_example:
            raise IndexError(
                    "Frame {} below frames_per_example "
                    "for video {}".format(frame,vid_id))

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

        # close reader
        vid112.reader._close()
        vid112.reader._pos = -101
        vid448.reader._close()
        vid448.reader._pos = -101
        """
        self.videos[448][vid_id] = Reader(vid448.filename)
        self.videos[112][vid_id] = Reader(vid112.filename)
        del vid112,vid448
        """


        return ([tensor,
                tensor_cropped,
                tensor_resized],
                [attention_cropped,
                attention_resized])

    def next_example(self):
        try:
            result = super().next_example()
        except Exception as e:
            print("Video {}, frame {}".format(
                *self._get_video_and_frame_number_from_id(e.args[-1])))
            raise e
        return result

    def get_batch(self, batch_size, batch_n):
        try:
            result = super().get_batch(batch_size, batch_n)
        except Exception as e:
            print("Video {}, frame {}".format(
                *self._get_video_and_frame_number_from_id(e.args[-1])))
            raise e
        return result

    def _get_video_and_frame_number_from_id(self,example_id):
        frame_id = self.valid_indices[example_id]
        if frame_id >= self.total_frames:
            raise IndexError("Example ID exceeds number of samples")
        start_id = 0
        for video_number, end_id in enumerate(self._cumulative_lengths):
            if end_id > frame_id:
                frame_number = frame_id - start_id
                return video_number, frame_number
            start_id = end_id
        raise RuntimeError("Frame not found")

    def _get_valid_indices(self):
        num_vids = len(self.videos[112])
        
        indices = []
        counter = 0
        for vid in range(num_vids):
            counter += 15
            for frame in range(15,self._lengths[vid]):
                if len(self.eye_positions[vid][frame]) > 0:
                    indices.append(counter)
                counter += 1
        return indices

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

    for y, x in gt_coords:
        ix = int(x); iy = int(y)
        if any(c+1 >= s for c,s in zip((x,y),size)):
            attention_map[iy,ix] = 1
        else:
            x2 = x-ix; y2 = y-iy
            x1 = 1-x2; y1 = 1-y2
            value=np.array([[y1*x1, y2*x1],
                            [y1*x2, y2*x2]])
            value/=len(gt_coords)
            attention_map[iy:iy+2, ix:ix+2] += value

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
