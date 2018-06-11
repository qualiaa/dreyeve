import numpy as np

def stamp(array, coord, brush, out=None):
    if out is None:
        out = np.array(array)
    canvas = out

    centre = [int(round(a)) for a in coord]
    # top left and bottom right
    start_coords = [c - brush_shape//2 for c, brush_shape in zip(centre, brush.shape)]
    end_coords = [c + brush_shape//2 for c, brush_shape in zip(centre, brush.shape)]

    brush_start = [0,0]
    brush_end = [e - 1 for e in brush.shape]

    for i, start_coord in enumerate(start_coords):
        if start_coord < 0:
            start_coords[i] = 0
            brush_start[i] -= start_coord
    for i, end_coord in enumerate(end_coords):
        if end_coord >= canvas.shape[i]:
            end_coords[i] = canvas.shape[i]
            brush_end[i] += canvas.shape[i] - end_coord

    stamp = brush[brush_start[0]:brush_end[0],
                  brush_start[1]:brush_end[1]]

    canvas[start_coords[0]:end_coords[0],
           start_coords[1]:end_coords[1]] = stamp
    return canvas

