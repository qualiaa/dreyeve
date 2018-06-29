#!/usr/bin/env python3

from glob import glob
import os.path
import pims

Reader = pims.ImageIOReader

folders = glob("DREYEVE_DATA/[0-9][0-9]")

diffs = []
for folder in folders:
    try:
        print(folder)
        original_video_path = os.path.join(folder,"video_garmin.avi")
        resized_448_path = os.path.join(folder,"garmin_resized_448.avi")
        resized_112_path = os.path.join(folder,"garmin_resized_112.avi")
        original_video = Reader(original_video_path)
        resized_448 = Reader(resized_448_path)
        resized_112 = Reader(resized_112_path)

        if len(resized_448) != len(original_video) != len(resized_112):
            if len(resized_112) != len(resized_448):
                raise RuntimeError
            diffs.append(len(original_video)-len(resized_448))
    except TypeError:
        break

print(set(diffs))
