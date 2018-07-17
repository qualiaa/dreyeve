import numpy as np

def _get_coords(canvas_shape, coords, brush_shape):
    centre = [int(round(a)) for a in coords]
    # top left and bottom right
    start_coords = [c - brush_shape//2
            for c, brush_shape in zip(centre, brush_shape)]
    end_coords = [c + brush_shape//2 + 1
            for c, brush_shape in zip(centre, brush_shape)]

    brush_start = [0,0]
    brush_end = list(brush_shape)

    for i, start_coord in enumerate(start_coords):
        if start_coord < 0:
            start_coords[i] = 0
            brush_start[i] -= start_coord
    for i, end_coord in enumerate(end_coords):
        if end_coord >= canvas_shape[i]:
            end_coords[i] = canvas_shape[i]
            brush_end[i] += canvas_shape[i] - end_coord

    return start_coords, end_coords, brush_start, brush_end

def stamp_add(canvas, coords, brush, out=None):
    if out is None:
        out = np.array(canvas)
    canvas = out

    start_coords, end_coords, brush_start, brush_end

    stamp = brush[brush_start[0]:brush_end[0],
                  brush_start[1]:brush_end[1]]

    canvas[start_coords[0]:end_coords[0],
           start_coords[1]:end_coords[1]] += stamp

def stamp_max(canvas, coords, brush, out=None):
    if out is None:
        out = np.array(canvas)
    canvas = out

    start_coords, end_coords, brush_start, brush_end = _get_coords(
            canvas.shape, coords, brush.shape)

    stamp = brush[brush_start[0]:brush_end[0],
                  brush_start[1]:brush_end[1]]

    old = canvas[start_coords[0]:end_coords[0],
                 start_coords[1]:end_coords[1]]
    canvas[start_coords[0]:end_coords[0],
           start_coords[1]:end_coords[1]] = np.maximum(stamp,old)

def stamp_over(canvas, coords, brush, out=None):
    if out is None:
        out = np.array(canvas)
    canvas = out

    start_coords, end_coords, brush_start, brush_end = _get_coords(
            canvas.shape, coords, brush.shape)

    stamp = brush[brush_start[0]:brush_end[0],
                  brush_start[1]:brush_end[1]]

    canvas[start_coords[0]:end_coords[0],
           start_coords[1]:end_coords[1]] = stamp
