import numpy as np
from glob import glob

import utils.pkl_xz

# files are batches of 16 frames
BATCH_SIZE = 16

class BatchedVideo:
    batch_format_string="/batch_{:04d}.pkl.xz"
    def __init__(self, folder, batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.folder = folder
        file_list = glob(self.folder+"/batch_[0-9][0-9][0-9][0-9].pkl.xz")
        self.n_batches = len(file_list)

        # find the number of frames in the last batch
        final_batch = self._get_batch(self.n_batches-1)
        self._len = self.batch_size*(self.n_batches-1) + final_batch.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self,frame):
        if type(frame) == int:
            frame_slice = slice(frame+1-16,frame+1)
        elif type(frame) == slice:
            frame_slice = frame
        else: raise IndexError()
        return self.get_frame_tensor(frame_slice)

    @property
    def shape(self):
        return (256,256,3)

    def get_frame_tensor(self,frame_slice):
        empty_result = np.array([])

        start_frame = frame_slice.start or 0
        end_frame = frame_slice.stop or len(self)
        if start_frame < 0: start_frame = start_frame % len(self)
        if end_frame < 0: end_frame = end_frame % len(self)
        num_frames = end_frame-start_frame

        if start_frame >= end_frame: return empty_result

        if start_frame < 0:
            raise IndexError("Start frame must be non-negative")
        if end_frame > len(self):
            raise IndexError("End frame exceeds end of video")

        num_batches = ((num_frames-1) // self.batch_size) + 1
        #print(num_batches)

        start_batch = start_frame // self.batch_size
        end_batch = (end_frame-1) // self.batch_size

        """
        print("Frames {:d} - {:d}".format(start_frame,end_frame))
        print("Total frames {:d}".format(num_frames))
        print("Loading batches ", end='')
        """
        
        result = None
        #print("Batches {:d}-{:d}".format(start_batch,end_batch))
        for batch_id in range(start_batch,end_batch+1):
            batch = self._get_batch(batch_id)
            if result is None:
                result = batch
            else:
                result = np.concatenate([result,batch], axis=0)

        if result is None: return empty_result

        start_offset = start_frame % self.batch_size
        #print("Input shape: {}".format(result.shape))
        result = result[start_offset:start_offset+num_frames,...]
        #print("Result shape: {}".format(result.shape))

        return result

    def _get_batch(self,n):
        path = self.folder + self.batch_format_string.format(n)
        return pkl_xz.load(path)

    def validate_batches(self):
        errors = dict()
        for i in range(self.n_batches):
            try:
                #print("Testing batch {:d}/{:d}".format(i,self.n_batches-1))
                a = self._get_batch(i)
                num_frames = a.shape[0]
                frame_shape = a.shape[1:]

                if type(a) != np.ndarray:
                    raise TypeError("Incorrect type: {}".format(type(a)))

                if i+1 != len(self) and num_frames != self.batch_size:
                    raise ValueError("Incorrect length: {}".format(num_frames))
                if frame_shape != self.shape:
                    raise ValueError("Shape {} does not match video".format(frame_shape))
            except (TypeError,ValueError,ChildProcessError) as e:
                errors[i] = e

        if errors:
            print("Errors in {:s}".format(self.folder))
            print(errors)
            for i,e in errors.items():
                print("[Batch {:d}] {}".format(i,e.args))
            return False
        return True
