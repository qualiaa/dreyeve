import csv
import math
from collections import defaultdict

import numpy as np

from consts import *

class Labels:
    FIXATION = 0
    SACCADE = 1
    BLINK = 2

LABEL_NAMES = { s:getattr(Labels,s) for s in Labels.__dict__.keys()
        if not s.startswith("_")}

def read(video_folders, desired_labels = [Labels.FIXATION]):
    """ Return a list where each element is a table of eye data for a video.
    Each table is a list of frames in the video, and contains a list of all
    gaze-points recorded for that frame.
    
    Format is [video [frame np.array(y,x coords)]]. """
    eye_positions = list()

    for i, folder in enumerate(video_folders):
        video_data = defaultdict(list)
        with open(folder + "/etg_samples.txt") as f:
            eye_data = csv.reader(f, delimiter=' ', dialect="unix")
            next(eye_data) # skip csv header
            for row in eye_data:
                # extract relevant CSV data and convert to appropriate types
                _, frame, x, y, label_name, _ = row

                try:
                    label = LABEL_NAMES[label_name.upper()]
                except KeyError:
                    # skip invalid labels
                    continue
                
                # filter undesired labels
                if label not in desired_labels: continue

                frame = int(frame)
                coords = [float(y), float(x)]

                # filter invalid gaze-points
                if any(a > b for a, b in zip(coords,VIDEO_SHAPE)) or any(
                        a < 0 or math.isnan(a) for a in coords):
                    continue

                #video_data[frame].append((label_id, coords))
                video_data[frame].append(coords)
            # convert coordinates to numpy array
            video_data = dict((a,np.array(b).reshape(-1,2)) for a, b in video_data.items())
        eye_positions.append(video_data)
    return eye_positions

def scale_to_shape(coords, target_shape):
    # calculate scale amount
    source_shape=VIDEO_SHAPE[0:2]
    if source_shape != target_shape:
        scale=np.array([t/s for s, t in zip(source_shape, target_shape)])
        coords=coords*scale
    return coords

def get_consecutive_frames(video_data, start_frame, num_frames)
    result = []
    for i in range(start_frame, start_frame + num_frames):
        try:
            result.append(video_data[i])
        except KeyError: continue
    return result
