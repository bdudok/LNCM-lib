import cv2


class LoadAvi:
    __name__ = 'LoadAvi'

    def __init__(self, vid_fn):
        self.im = cv2.VideoCapture(vid_fn)
        self.frame_buffer = {}
        ret, frame = self.im.read()
        self.end = False
        self.index = 0
        self.cache_frame(0, frame)
        self.fps = self.im.get(cv2.CAP_PROP_FPS)
        self.h, self.w = self.frame.shape[:2]
        self.n_frames = int(self.im.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, item):
        self.frame = self.frame_buffer.get(item, None)
        if self.frame is None:
            self.im.set(cv2.CAP_PROP_POS_FRAMES, item - 1)
            ret, frame = self.im.read()
            self.cache_frame(item, frame)
        self.index = item
        return self.frame

    def __next__(self):
        retval = self.frame
        if retval is not None:
            ret, frame = self.im.read()
            self.cache_frame(self.index+1, frame)
            retval = self.frame
        return retval




    def cache_frame(self, index, frame):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer[index] = self.frame

class WriteAvi:
    __name__ = 'WriteAvi'
    def __init__(self, handle, shape, vidformat='avi', framerate=20.0):
        self.outfile_handle = handle
        self.shape = shape
        self.format = vidformat
        self.framerate = framerate
        self.outfile = None
        if self.format == 'avi':
            self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.file_ext = '.avi'
            if not self.outfile_handle.endswith(self.file_ext):
                self.outfile_handle += self.file_ext
            is_color = False
            self.outfile = cv2.VideoWriter(self.outfile_handle, self.fourcc, self.framerate, shape, is_color)
        assert self.outfile is not None

    def write(self, frame):
        self.outfile.write(frame)

    def close(self):
        self.outfile.release()
        self.outfile = None