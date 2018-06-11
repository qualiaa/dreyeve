import numpy as np
import eye_data
from scipy import signal

from stamp import stamp

def attention_map(eye_coords, output_shape, point_size):
    # remove non-fixations
    if len(eye_coords) == 0: raise ValueError("No coords for attention map")
    eye_coords = eye_data.scale_to_shape(eye_coords, shape)
    attention_map = np.zeros(shape,dtype=np.float32)

    for y, x in eye_coords:
        ix = int(x); iy = int(y)
        if any(c+1 >= s for c,s in zip((x,y),shape)):
            attention_map[iy,ix] = 1
        else:
            x2 = x-ix; y2 = y-iy
            x1 = 1-x2; y1 = 1-y2
            value = np.array([[y1*x1, y2*x1],
                            [y1*x2, y2*x2]])
            value /= len(eye_coords)
            attention_map[iy:iy+2, ix:ix+2] += value

    return attention_map

def gaussian_2d(size):
    assert size % 2 == 1
    sd=(size-1)/6
    g = np.expand_dims(signal.gaussian(size,sd),0)
    return np.matmul(g.T, g)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    canvas = np.zeros((101,101))
    g = gaussian_2d(19)

    [stamp(canvas,(y,x),g,canvas) for y in [4,50,101-5] for x in [4,50,101-5]]
    plt.figure()
    plt.imshow(canvas)
    plt.show()

