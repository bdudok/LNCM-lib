import numpy, os
import tifffile
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
# import h5py
import json
# from multiprocessing import Process, Queue
# from datetime import datetime
import cv2
from scipy.signal import resample


class CropVideo:
    def __init__(self, cam_file, savepath):
        self.fn = cam_file
        self.im = None
        print(cam_file)
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.mkdir(savepath)

    def open_video(self):
        # look for the raw file:
        if self.fn.endswith('.avi'):
            reader_type = 'cv2'
            self.ftype = 'avi'
        if reader_type == 'cv2':
            self.im = cv2.VideoCapture(self.fn)
            # estimate number of frames
            self.n_frames = int(self.im.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, self.frame = self.im.read()
            # shape to conform w .mat so rest works without mods
            self.buffer_size = min(1000, self.n_frames)
            self.buffer_offset = 0
            self.data = numpy.empty((self.buffer_size, self.frame.shape[0], self.frame.shape[1]),
                                    dtype=self.frame.dtype)
            # read first 100 frames
            self.i = 0
            while self.frame is not None and self.i < min(100, self.n_frames):
                self.next_frame()
        self.eye_rect = None

    def next_frame(self):
        # #to avoid overallocation, this is now limited.
        # Works for previews, but will need to implement rolling the buffer
        assert self.i < self.buffer_size
        self.data[self.i - self.buffer_offset] = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.i += 1
        ret, self.frame = self.im.read()

    def buffer_frames(self):
        while self.frame is not None and self.i < self.buffer_size:
            self.next_frame()

    def crop(self):
        if self.im is None:
            self.open_video()
        f0 = min(100, self.n_frames)
        fr = self.data[f0 - 10:f0, :, :].mean(axis=0)
        self.preview = fr
        fig_w = 9
        fig_h = fig_w * fr.shape[1] / fr.shape[0]
        fig, axes = plt.subplots(3, figsize=(fig_w, fig_h + 0.4), gridspec_kw={'height_ratios': [fig_h, 0.2, 0.2]})
        ax = axes[0]
        ax.imshow(1 - (fr / fr.max()), cmap='Greys')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', right='off', left='off', labelleft='off')
        self.rect = None
        self.rs = RectangleSelector(ax, self.line_select_callback,
                                    drawtype='box', useblit=False, button=[1],
                                    minspanx=5, minspany=5, spancoords='pixels',
                                    interactive=True)
        self.b_save = Button(axes[1], 'Save Eye')
        self.b_save.on_clicked(self.save_eye_crop)
        self.m_save = Button(axes[2], 'Save Motion')
        self.m_save.on_clicked(self.save_motion_crop)
        self.fig = fig
        plt.show(block=True)

    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.rect = (x1, x2, y1, y2)

    def save_eye_crop(self, *args):
        if self.rect is not None:
            with open(self.savepath + '_eye-crop.json', 'w') as f:
                self.eye_rect = self.rect
                json.dump(self.rect, f)
                self.b_save.label.set_text('Eye Saved')
                print(self.rect, 'saved.')
            tifffile.imsave(self.savepath + '_video_preview.tif', self.preview)

    def save_motion_crop(self, *args):
        if self.rect is not None:
            with open(self.savepath + '_motion-crop.json', 'w') as f:
                self.motion_rect = self.rect
                json.dump(self.rect, f)
                self.m_save.label.set_text('Motion Saved')
                print(self.rect, 'saved.')

    def load_eye_crop(self):
        fn = self.savepath + '_eye-crop.json'
        return self.load(fn)

    @staticmethod
    def load(fn):
        if os.path.exists(fn):
            with open(fn, 'r') as f:
                return json.load(f)
        else:
            return False

    def load_motion_crop(self):
        fn = self.savepath + '_motion-crop.json'
        return self.load(fn)

    def compute_eye_trace(self):
        crop = self.load_eye_crop()
        if crop is False:
            print('Motion crop not saved, not computing...')
            return -1
        rect = crop
        eye = self.eye
        # create vertical and horizontal samples through eye
        bbox = numpy.array(rect, dtype='int').reshape((2, 2))
        center = bbox.mean(axis=1).astype('int')
        h_profile = eye[:, 0, slice(*bbox[0]), center[1]]
        v_profile = eye[:, 0, center[0], slice(*bbox[1])]
        # fold lines in 2:
        profiles = numpy.empty((len(eye), numpy.diff(bbox, axis=1).max() // 2, 4), dtype=eye.dtype)
        profiles[:] = numpy.nan
        m = int(numpy.diff(bbox[0]) // 2)
        profiles[:, :m, 0] = h_profile[:, -m:]
        profiles[:, :m, 1] = numpy.flip(h_profile[:, :m], axis=1)
        m = int(numpy.diff(bbox[1]) // 2)
        profiles[:, :m, 0] = v_profile[:, -m:]
        profiles[:, :m, 1] = numpy.flip(v_profile[:, :m], axis=1)
        # average every 2 frames (dropping first)
        m = profiles.mean(axis=2)[2 - len(profiles) % 2:]
        m = m.reshape(len(m) // 2, 2, m.shape[-1]).mean(axis=1)
        # determine location of max drop
        drop = -numpy.diff(m, axis=1)
        drop_loc = numpy.argmax(drop, axis=1)
        numpy.save(self.savepath + 'eye_trace.npy', drop_loc)
        print(self.savepath, 'eye trace saved')


