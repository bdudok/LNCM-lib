import os.path

import cv2


class LoadAvi:
    '''
    Reads frames of a video file as grayscale. vid[i] returns frame i, next(vid) the frame after the last one read.
    Frames are decoded when requested and not kept: reading forward decodes on, other jumps seek.
    '''
    __name__ = 'LoadAvi'
    max_skip = 30  # frames: a jump forward up to this many decodes on instead of seeking

    def __init__(self, vid_fn):
        assert os.path.exists(vid_fn)
        self.im = cv2.VideoCapture(vid_fn)
        self.fps = self.im.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self.im.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = -1  # frame in self.frame; None if the position is unknown
        self.frame = None
        self[0]
        self.h, self.w = self.frame.shape[:2]

    def __getitem__(self, item):
        if item != self.index:
            if self.index is not None and 0 < item - self.index <= self.max_skip:
                for _ in range(item - self.index - 1):
                    self.im.grab()
            else:
                self.im.set(cv2.CAP_PROP_POS_FRAMES, item)
            self.read(item)
        return self.frame

    def __next__(self):
        return self[self.index + 1]

    def read(self, index):
        ret, frame = self.im.read()
        if not ret:
            self.index, self.frame = None, None  # position unknown: the next request seeks
            raise IndexError(f'Frame {index} could not be read')
        self.index = index
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


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
