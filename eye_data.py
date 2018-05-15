import csv
import math
from collections import defaultdict

import numpy as np

from consts import *

def read(video_folders):
    """ Return an array of """
    eye_positions = list()

    for i, folder in enumerate(video_folders):
        eye_pos = defaultdict(list)
        with open(folder + "/etg_samples.txt") as f:
            eye_data = csv.reader(f, delimiter=' ', dialect="unix")
            next(eye_data) # skip csv header
            for row in eye_data:
                _, frame, x, y, label, _ = row
                try:
                    label_id = LABEL_NAMES[label.upper()]
                except KeyError:
                    continue
                frame = int(frame)
                coords = np.array([float(y), float(x)])
                
                if label_id != Labels.FIXATION: continue
                if any(a > b for a, b in zip(coords,VIDEO_SHAPE)) or any(
                        a < 0 or math.isnan(a) for a in coords):
                    continue
                #eye_pos[frame].append((label_id, coords))
                eye_pos[frame].append(coords)
            eye_pos[frame]=np.array(eye_pos[frame])
        eye_positions.append(eye_pos)
    return eye_positions

def scale_to_shape(coords, target_shape):
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
