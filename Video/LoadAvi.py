import cv2


class LoadAvi:
    __name__ = 'LoadAvi'

    def __init__(self, vid_fn):
        self.im = cv2.VideoCapture(vid_fn)
        self.frame_buffer = {}
        ret, frame = self.im.read()
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
        return self.frame

    def cache_frame(self, index, frame):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer[index] = self.frame
