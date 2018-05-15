DATA_DIR = "DREYEVE_DATA"
VIDEO_SHAPE=(1080,1920,3)

class Labels:
    FIXATION = 0
    SACCADE = 1
    BLINK = 2

LABEL_NAMES = { s:getattr(Labels,s) for s in Labels.__dict__.keys()
        if not s.startswith("_")}
