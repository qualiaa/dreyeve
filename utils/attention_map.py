import numpy as np
import eye_data
import consts as c
from scipy import signal

from stamp import stamp

def attention_map(eye_coords, output_shape, point_radius=0,decay_fn=None):
    return _attention_map([eye_coords], output_shape, point_radius)

def gaussian_2d(size):
    assert size % 2 == 1
    sd=(size-1)/6
    g = np.expand_dims(signal.gaussian(size,sd),0)
    return np.matmul(g.T, g)

def _attention_map(clip_coords, output_shape, point_radius=0, decay_fn=None):
    if type(clip_coords) == tuple:
        if len(clip_coords) != 2:
            raise TypeError("Attention map expects single coordinate pair,"
                    "np.ndarray of coordinate pairs, or list of np.ndarray per frame.",
                clip_coords)
        clip_coords = np.array(clip_coords)
    if type(clip_coords) == np.ndarray:
        clip_coords = [clip_coords]
    if type(clip_coords[0]) != np.ndarray:
        raise TypeError("Attention map expects single coordinate pair,"
                "np.ndarray of coordinate pairs, or list of np.ndarray per frame",
            clip_coords)
    if len(clip_coords) == 0: raise ValueError("No coords for attention map")

    if point_radius >= 1:
        brush = gaussian_2d(point_radius * 2 + 1)

    # scale eye coordinates
    clip_coords = np.array(list(map(lambda c: eye_data.scale_to_shape(c, output_shape), clip_coords)))

    attention_map = np.zeros(output_shape)

    for i, frame_coords in enumerate(clip_coords):
        weight, x = 1, 1

        if decay_fn is not None and len(clip_coords) > 1:
            x = i/(len(clip_coords) - 1)
            weight = decay_fn(x)

        for coord_pair in frame_coords:
            if point_radius < 1:
                brush = bilinear_1px(coord_pair)
            stamp(attention_map, coord_pair, brush * weight, out=attention_map)

    attention_sum = attention_map.sum()
    if attention_sum > 0:
        attention_map /= attention_sum
    return attention_map
            

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
    canvas = np.zeros((100,100))
    g = gaussian_2d(19)
    video_folder = c.DATA_DIR + "/01"

    gaze_points = eye_data.read([video_folder])[0]

    num_frames = 100

    frame = 1014

    clip_data = []
    for i in range(frame,frame-num_frames,-1):
        try:
            clip_data.append(gaze_points[i])
        except KeyError: continue

    plt.figure()
    plt.imshow(_attention_map(clip_data, (100,140),))
    plt.show()
    plt.subplot(121)
    plt.imshow(_attention_map(clip_data, (100,140),5))
    plt.subplot(122)
    plt.imshow(_attention_map(clip_data, (100,140),5, np.exp))
    plt.show()
