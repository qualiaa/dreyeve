from functools import reduce
from itertools import accumulate
from operator import add
from pathlib import Path, PurePath
from warnings import warn

import numpy as np
from imageio import imread
from imageio.core.format import CannotReadFrameError
from pims import ImageIOReader as Reader
from skimage.transform import resize

from utils.attention_map import multiframe_attention_map
from utils.Examples import Examples
from utils.random_crop_slice import random_crop_slice
import eye_data
import settings
from consts import *

class DreyeveExamples(Examples):
    frame_shape = (448,448)
    example_shape = (112,112)

    def __init__(self,
                 folders,
                 predict=False,
                 frames_per_example=16,
                 seed=None):
        self.predict = predict
        self.frames_per_example=frames_per_example

        print("Loading eye data...")
        self.eye_positions = eye_data.read(folders)
        print("Loading video files...")
        self.videos = {
                x: [Reader(str(Path(folder, "garmin_resized_{:d}.avi".format(x))))
                    for folder in folders]
                for x in [112,448]}
        self.mean_frame = {
                x: [imread(str(Path(folder, "mean_frame_{:d}.png".format(x))))
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
        crop_slice = random_crop_slice(self.frame_shape,
                self.example_shape, self._rand)
        labels = self.get_labels(example_id, crop_slice)
        if not self.predict:
            while labels[0].sum() < EPS:
                crop_slice = random_crop_slice(self.frame_shape,
                        self.example_shape, self._rand)
                labels = self.get_labels(example_id, crop_slice)

        data = self.get_data(example_id, crop_slice)

        return (data, labels)

    def get_data(self, example_id, crop_slice=None):
        vid_id,frame_number = self._get_video_and_frame_number_from_id(example_id)

        if frame_number+1 < self.frames_per_example:
            raise IndexError(
                    "Frame {} below frames_per_example "
                    "for video {}".format(frame_number,vid_id))

        vid448 = self.videos[448][vid_id]
        vid112 = self.videos[112][vid_id]
        mf448 = self.mean_frame[448][vid_id]
        mf112 = self.mean_frame[112][vid_id]

        clip = _get_frame_tensor(vid448, frame_number, mf448)

        clip_resized = _get_frame_tensor(vid112, frame_number, mf112)
        last_frame = clip[:,-1,...]

        # close reader to prevent reaching process limit
        _close_video(vid112)
        _close_video(vid448)

        if not self.predict:
            if crop_slice is None:
                raise ValueError("crop_slice must be provided")
            clip_cropped = clip[[slice(None),slice(None),*crop_slice]]
            return [clip_cropped,
                    clip_resized,
                    last_frame]
        else:
            return [clip_resized,
                    last_frame]


    def get_labels(self, example_id, crop_slice=None):
        vid_id,frame_number = self._get_video_and_frame_number_from_id(example_id)

        """
        if frame_number+1 < self.frames_per_example:
            raise IndexError(
                    "Frame {} below frames_per_example "
                    "for video {}".format(frame_number,vid_id))
        """

        # retrieve gaze points from multiple frames
        if settings.centre_attention_map_frames:
            assert settings.attention_map_frames % 2 == 1
            start_frame = frame_number - (settings.attention_map_frames-1)//2
        else:
            start_frame = frame_number - settings.attention_map_frames + 1

        eye_coords = eye_data.get_consecutive_frames(self.eye_positions[vid_id],
                start_frame=start_frame,
                num_frames=settings.attention_map_frames)
        assert eye_coords is not []


        # generate attention map from gaze points
        try:
            attention = multiframe_attention_map(
                    eye_coords,
                    output_shape=self.frame_shape,
                    point_radius=settings.gaze_radius,
                    agg_method=settings.attention_map_aggregation())
            #attention = np.expand_dims(attention,0)
        except ValueError:
            raise ValueError("Example has no ground truth")


        # if not used for prediction, need to generate a cropped version
        if not self.predict:
            if crop_slice is None:
                raise ValueError("crop_slice must be provided")

            #attention_cropped = attention[[slice(None),*crop_slice]]
            attention_cropped = attention[crop_slice]
            return [attention_cropped, attention]

        return attention

    """ Helper functions """

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
                try:
                    if type(self.eye_positions[vid][frame]) == np.ndarray:
                        indices.append(counter)
                except KeyError: ""
                counter += 1
        return indices

    """ Wrap parent fns for exception handling """

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

"""
def _resize_frame_tensor(frame_tensor,target_shape):
    target_shape=(frame_tensor.shape[0],*target_shape,frame_tensor.shape[-1])
    return resize(frame_tensor,target_shape,anti_aliasing=True,mode='reflect')
"""


def _get_frame_tensor(video, frame_of_interest, mean_frame, num_frames=16):
    """ return N stacked, resized frames from frame [I-N, I] """
    foi = frame_of_interest
    if foi < num_frames - 1:
        raise IndexError("Frame of interest " + foi +
                         " must have 15 preceding frames")
    frame_slice = video[foi-num_frames+1:foi+1]
    frame_tensor = np.stack(frame_slice, axis=0)
    frame_tensor = frame_tensor.astype(np.float32)
    frame_tensor = (frame_tensor - mean_frame)
    frame_tensor = np.transpose(frame_tensor,(3,0,1,2))
    return frame_tensor

def _close_video(video):
        video.reader._close()
        video.reader._pos = -101

def flatten(l):
    return [x for sublist in l for x in sublist]
