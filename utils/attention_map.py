import numpy as np
import eye_data
import consts as c
from scipy import signal

from . import stamp

def _attention_map(frame_coords,
                   output_shape,
                   point_radius=0,
                   stamp_fn=stamp.stamp_max):
    if type(frame_coords) == tuple:
        frame_coords = np.array(frame_coords)

    if type(frame_coords) != np.ndarray or frame_coords.shape[1] != 2:
        raise TypeError("attention_map expects single coordinate pair or "
                "numpy.ndarray of coordinate pairs",
                frame_coords)

    if point_radius >= 1:
        brush = gaussian_2d(point_radius * 2 + 1)

    # scale eye coordinates
    frame_coords = np.array(list(map(lambda c: eye_data.scale_to_shape(c,
        output_shape), frame_coords)))

    attention_map = np.zeros(output_shape)

    for coord_pair in frame_coords:
        if point_radius < 1:
            brush = bilinear_1px(coord_pair)
        stamp_fn(attention_map,
                 coord_pair,
                 brush,
                 out=attention_map)

    return attention_map


def multiframe_attention_map(
        clip_coords_list : list,
        agg_method=np.max,
        decay_fn=None,
        *args, **kargs):
    """
    if type(clip_coords) == np.ndarray:
        clip_coords = [clip_coords]
        """
    if len(clip_coords_list) == 0:
        raise ValueError("No coords for attention map", clip_coords_list)
    if type(clip_coords_list[0]) != np.ndarray:
        raise TypeError("multiframe_attention_map expects list containing one "
                "(?,2) np.ndarray for each frame",
                clip_coords_list)

    attention_maps = []

    if agg_method==np.max: stamp_fn=stamp.stamp_max
    elif agg_method==np.sum: stamp_fn=stamp.stamp_add
    else: raise ValueError("Unexpected aggregation method")

    for i, frame_coords in enumerate(clip_coords_list):
        weight, x = 1, 1

        if decay_fn is not None and len(clip_coords) > 1:
            x = i/(len(clip_coords) - 1)
            weight = decay_fn(x)

        attention_map = _attention_map(frame_coords, stamp_fn=stamp_fn,
                *args, **kargs)
        attention_maps.append(weight * attention_map)

    attention_maps = np.stack(attention_maps)

    attention_map = agg_method(attention_maps,axis=0)

    attention_sum = attention_map.sum()
    if attention_sum > 0:
        attention_map /= attention_sum
    return attention_map


def gaussian_2d(pixel_size):
    sigma_per_side = 4
    assert pixel_size % 2 == 1
    sigma=(pixel_size-1)/(2 * sigma_per_side)
    g = np.expand_dims(signal.gaussian(pixel_size,sigma),0)
    return np.matmul(g.T, g)


def bilinear_1px(coord):
    brush = np.zeros((3,3))
    y,x = coord
    y2, x2 = y-int(y),x-int(x)
    y1 = 1-y2; x1 = 1-x2;
    brush[1:,1:] = np.array([[y1*x1, y1*x2],
                             [y2*x1, y2*x2]])
    return brush


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pims import ImageIOReader as Reader
    canvas = np.zeros((100,100))
    g = gaussian_2d(19)
    video_folder = c.DATA_DIR + "/01"

    gaze_points = eye_data.read([video_folder])[0]
    num_frames = 50
    frame_number = 1014

    video448 = Reader(video_folder + "/garmin_resized_448.avi")
    frame448 = video448[frame_number]
    video112 = Reader(video_folder + "/garmin_resized_112.avi")
    frame112 = video112[frame_number]

    clip_data = eye_data.get_consecutive_frames(gaze_points, frame_number+1-num_frames, num_frames)

    plt.figure()
    plt.subplot(221)
    plt.imshow(_attention_map(clip_data, (448,448),16))
    plt.subplot(222)
    plt.imshow(frame448)
    plt.subplot(223)
    plt.imshow(_attention_map(clip_data, (112,112), 4, np.exp))
    plt.subplot(224)
    plt.imshow(frame112)
    plt.show()
