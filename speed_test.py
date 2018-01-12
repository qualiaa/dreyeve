#!/usr/bin/env python3

from itertools import repeat
from glob import glob
import random
import time

import numpy as np
from pims import ImageIOReader as Reader

from BatchedVideo import BatchedVideo
from input import random_crop_slice, _resize_frame_tensor
from old_input import get_frame_tensor


def get_mem():
    results = {"rss": 0, "peak": 0}
    with open("/proc/self/status") as f:
        for line in f:
            parts=line.split()
            key = parts[0][2:-1].lower()
            if key in results:
                results[key]=int(parts[1])
    return results


def profile_fn(f):
    def wrapper(*args,**kargs):
        pp, pr, lp, lr, dp, dr = (0,0,0,0,0,0)


        for n,_ in enumerate(f(*args,**kargs)):
            mem = get_mem()
            r = mem["rss"]
            p = mem["peak"]
            pp = pp if pp > p else p
            pr = pr if pr > r else r
            if lp != 0:
                dp += p - lp
                dr += r - lr
            lp = p
            lr = r
        dr /= n
        dp /= n
        print("Peak peak: {} Peak resident: {}".format(pp,pr))
        print("Delta peak: {} Delta resident: {}".format(dp,dr))
    return wrapper

def time_fn(f):
    def wrapper(*args,**kargs):
        start = time.time()
        f(*args,**kargs)
        duration = time.time() - start
        print("Execution took {}".format(duration))
    return wrapper


@time_fn
@profile_fn
def method1(folder, n):
    vid = BatchedVideo(folder + "/garmin_frames")

    indices = list(range(15,len(vid)))
    random.shuffle(indices)
    for i in indices[:n]:
        tensor = vid[i]
        crop_slice = random_crop_slice(vid.shape[:2],(112,112))
        cropped_tensor = tensor[[None,*crop_slice,None]]
        resized_tensor = _resize_frame_tensor(tensor,(112,112))
        yield


@time_fn
@profile_fn
def method2(folder, n):
    vid112 = Reader(folder + "/garmin_resized_112.avi")
    vid448 = Reader(folder + "/garmin_resized_448.avi")

    indices = list(range(15,len(vid112)))
    random.shuffle(indices)
    a = 0
    for i in indices[:n]:
        a+=1
        tensor = get_frame_tensor(vid448,i)
        crop_slice = random_crop_slice(tensor.shape[1:3],(112,112))
        cropped_tensor = tensor[[None,*crop_slice,None]]
        resized_tensor = get_frame_tensor(vid112,i)
        yield
    print(a)


folder = "DREYEVE_DATA/01/"
n = 100

print("method1")
method1(folder,n)
print("method2")
method2(folder,n)
