import numpy, os
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import h5py
import json
from multiprocessing import Process, Queue
from datetime import datetime
import cv2
from scipy.signal import resample


def load_mikko_trace(path, prefix, fps=15.6):
    fn = f'{path}/{prefix}_eye_radius.mat'
    if not os.path.exists(fn):
        return None
    from sbxreader import loadmat
    meye = loadmat(fn)['r'][1:]
    meye[numpy.where(meye == 0)] = numpy.nan
    # drop 1st frame and average every 2
    if len(meye) % 2:
        meye = meye[:-1]
    meye = numpy.nanmean(meye.reshape((len(meye) // 2, 2)), axis=1)
    # revert nan to zero because in some files there are long nan periods
    meye[numpy.isnan(meye)] = 0
    # smooth trace in a half sec window:
    t1 = 1 * fps
    smooth_eye = numpy.empty(len(meye))
    for t in range(len(meye)):
        ti0 = max(0, int(t - t1 * 0.5))
        ti1 = min(len(meye), int(t + t1 * 0.5) + 1)
        smooth_eye[t] = numpy.mean(meye[ti0:ti1])
    return smooth_eye


class EyeTracing:
    def __init__(self, prefix, folder='eyes/', alt_folders=None, prev_only=False):
        self.prefix = prefix
        # look for the raw file:
        incl_folders = ['./', folder]
        if alt_folders is not None:
            for path in alt_folders:
                if path not in incl_folders:
                    incl_folders.append(path)
        incl_suffix = ['_eye.mat', '_eye.mj2']
        reader_type = None
        for path in incl_folders:
            for suffix in incl_suffix:
                eye_path = path + prefix + suffix
                if os.path.exists(eye_path):
                    reader_type = suffix
                    break
            if reader_type is not None:
                break
        if reader_type is None:
            print(f'Eye file not found for {prefix} in these locations:{incl_folders}')
            return -1
        elif '.mat' in reader_type:
            self.eye = h5py.File(eye_path, 'r')['data']
            self.ftype = '.mat'
        elif '.mj2' in reader_type:
            im = cv2.VideoCapture(eye_path)
            # estimate number of frames
            n_frames = int(im.get(cv2.CAP_PROP_FRAME_COUNT))
            if prev_only:
                n_frames = prev_only
            ret, frame = im.read()
            # shape to conform w .mat so rest works without mods
            data = numpy.empty((n_frames, 1, frame.shape[1], frame.shape[0]), dtype=frame.dtype)
            # read all frames
            i = 0
            while frame is not None and i < n_frames:
                data[i, 0] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).transpose()
                i += 1
                ret, frame = im.read()

            self.eye = data
            self.ftype = 'mj2'
        self.eye_rect = None

    def crop(self):
        fr = self.eye[100:110, 0, :, :].mean(axis=0)
        fig_w = 9
        fig_h = fig_w * fr.shape[1] / fr.shape[0]
        fig, axes = plt.subplots(3, figsize=(fig_w, fig_h + 0.4), gridspec_kw={'height_ratios': [fig_h, 0.2, 0.2]})
        ax = axes[0]
        ax.imshow(1 - (fr / fr.max()).transpose(), cmap='Greys')
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
            with open(self.prefix + '_eye-crop.json', 'w') as f:
                self.eye_rect = self.rect
                json.dump(self.rect, f)
                self.b_save.label.set_text('Eye Saved')
                print(self.rect, 'saved.')

    def save_motion_crop(self, *args):
        if self.rect is not None:
            with open(self.prefix + '_motion-crop.json', 'w') as f:
                self.motion_rect = self.rect
                json.dump(self.rect, f)
                self.m_save.label.set_text('Motion Saved')
                print(self.rect, 'saved.')

    def load_eye_crop(self):
        fn = self.prefix + '_eye-crop.json'
        return self.load(fn)

    @staticmethod
    def load(fn):
        if os.path.exists(fn):
            with open(fn, 'r') as f:
                return json.load(f)
        else:
            return False

    def load_motion_crop(self):
        fn = self.prefix + '_motion-crop.json'
        return self.load(fn)

    def compute_motion_map(self):
        movie = self.eye
        crop = self.load_motion_crop()
        if crop is False:
            print('Motion crop not saved, not computing...')
            return -1
        crop = numpy.array(crop).astype('int')
        crop[[1, 3]] += 1
        IM = movie[:, 0, crop[0]:crop[1], crop[2]:crop[3]]
        # downsample IM
        IM = IM[2 - len(IM) % 2:]
        IM = IM.reshape(len(IM) // 2, 2, *IM.shape[1:]).mean(axis=1)
        alpha = 0.3
        BG = numpy.empty(IM.shape)
        BG[0] = IM[:2].mean(axis=0)
        for fr in range(len(IM) - 1):
            i = fr + 1
            BG[i] = alpha * IM[i] + (1 - alpha) * BG[i - 1]
        D = numpy.abs(IM - BG)
        motion_map = numpy.empty(D.shape)
        motion_map[0] = D[:2].mean(axis=0)
        beta = 0.9
        for fr in range(len(D) - 1):
            i = fr + 1
            motion_map[i] = beta * D[i] + (1 - beta) * motion_map[i - 1]
        # normalize 0-1 and save as 8 bits
        motion_map = (motion_map * 255 / motion_map.max()).astype('uint8')
        numpy.save(self.prefix + '_motion_map', motion_map)
        print(self.prefix, 'Motion map saved')

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
        numpy.save(self.prefix + 'eye_trace.npy', drop_loc)
        print(self.prefix, 'eye trace saved')


class Worker(Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue

    def run(self):
        for path, prefix, eye_folder, alt_folders in iter(self.queue.get, None):
            print(datetime.now(), 'Started processing', prefix)
            os.chdir(path)
            e = EyeTracing(prefix, folder=eye_folder, alt_folders=alt_folders)
            e.compute_motion_map()
            e.compute_eye_trace()


def clean_movement_map(prefix, session, t1=15.6, eye_path=None):
    if eye_path is None:
        eye_path = 'eyes/'
    a = session
    # 1) clean up and z score
    if hasattr(a, 'eye_path'):
        efn = a.eye_path + prefix + '_face_trace.npy'
    else:
        efn = 'eyes//' + prefix + '_face_trace.npy'
    mt = numpy.load(efn)
    if len(mt) > a.ca.frames:
        mt = mt[:a.ca.frames]
    smw = numpy.empty(a.ca.frames)
    for t in range(a.ca.frames):
        ti0 = max(0, int(t - t1 * 0.5))
        ti1 = min(len(mt), int(t + t1 * 0.5) + 1)
        smw[t] = numpy.mean(mt[ti0:ti1])
    bsl = numpy.empty(a.ca.frames)
    t2 = int(t1 * 50)
    for t in range(a.ca.frames):
        ti0, ti1 = max(0, t - t2), min(t, a.ca.frames)
        if ti0 < ti1:
            minv = numpy.min(smw[ti0:ti1])
        else:
            minv = smw[ti0]
        bsl[t] = minv
    rel = numpy.maximum(0, (mt - bsl) / bsl)
    ntr = rel / numpy.std(rel[numpy.where(numpy.logical_not(a.pos.gapless))])
    # 2) zero at movement, beginning, end, and sd < 1
    face = numpy.copy(ntr)
    face[:200] = 0
    face[-200:] = 0
    for events, direction in zip(a.startstop(ret_loc='actual'), (1, -1)):
        for t in events:
            while numpy.any(ntr[t - 3:t + 3] > 1):
                face[t + direction] = 0
                t += direction
    face[numpy.where(a.pos.gapless)] = 0
    if a.pos.gapless[-200]:
        t = a.ca.frames - 200
        while numpy.any(ntr[t - 3:t + 3] > 1):
            face[t - 1] = 0
            t -= 1
    return face


def clean_whisker_map(prefix, session, t1=15.6):
    a = session
    # 1) clean up and z score
    if hasattr(a, 'eye_path'):
        efn = a.eye_path + prefix
    else:
        efn = 'eyes//' + prefix
    wh_fn = efn + '_whisker_trace.npy'
    if os.path.exists(wh_fn):
        return numpy.load(wh_fn)
    mm = numpy.load(efn + '_motion_map.npy', mmap_mode='r')
    with open(efn + '_whisker-recrop.json', 'r') as f:
        crop = numpy.array(json.load(f), dtype=numpy.int64).reshape((2, 2))  # [[x0,x1], [y0,y1]]
    mm_crop = mm[:a.ca.frames, slice(*crop[0]), slice(*crop[1])]
    mt = resample(mm_crop.mean(axis=(1, 2)), a.ca.frames)
    smw = numpy.empty(a.ca.frames)
    for t in range(a.ca.frames):
        ti0 = max(0, int(t - t1 * 0.5))
        ti1 = min(len(mt), int(t + t1 * 0.5) + 1)
        smw[t] = numpy.mean(mt[ti0:ti1])
    bsl = numpy.empty(a.ca.frames)
    t2 = int(t1 * 50)
    for t in range(a.ca.frames):
        ti0, ti1 = max(0, t - t2), min(t, a.ca.frames)
        if ti0 < ti1:
            minv = numpy.min(smw[ti0:ti1])
        else:
            minv = smw[ti0]
        bsl[t] = minv
    rel = numpy.maximum(0, (mt - bsl) / bsl)
    ntr = rel / numpy.std(rel[numpy.where(numpy.logical_not(a.pos.gapless))])
    # 2) zero at movement, beginning, end, and sd < 1
    face = numpy.copy(ntr)
    face[:200] = 0
    face[-200:] = 0
    for events, direction in zip(a.startstop(ret_loc='actual'), (1, -1)):
        for t in events:
            while numpy.any(ntr[t - 3:t + 3] > 1):
                face[t + direction] = 0
                t += direction
    face[numpy.where(a.pos.gapless)] = 0
    if a.pos.gapless[-200]:
        t = a.ca.frames - 200
        while numpy.any(ntr[t - 3:t + 3] > 1):
            face[t - 1] = 0
            t -= 1
    numpy.save(wh_fn, face)
    return face


class WhiskCropper:

    def crop_whisker(self, prefix, preview):
        self.prefix = prefix
        fr = preview
        fig_w = 9
        fig_h = fig_w * fr.shape[1] / fr.shape[0]
        fig, axes = plt.subplots(3, figsize=(fig_w, fig_h + 0.4), gridspec_kw={'height_ratios': [fig_h, 0.2, 0.2]})
        ax = axes[0]
        im = 1 - (fr / fr.max())
        ax.imshow(im.transpose(), cmap='inferno')
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
        # self.b_save = Button(axes[1], 'Save Eye')
        # self.b_save.on_clicked(self.save_eye_crop)
        self.m_save = Button(axes[2], 'Save Whisker')
        self.m_save.on_clicked(self.save_whisker_crop)
        self.fig = fig
        plt.show(block=True)

    def save_whisker_crop(self, *args):
        if self.rect is not None:
            with open(self.prefix + '_whisker-recrop.json', 'w') as f:
                self.motion_rect = self.rect
                json.dump(self.rect, f)
                self.m_save.label.set_text('Whisker crop Saved')
                print(self.rect, 'saved.')

    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.rect = (x1, x2, y1, y2)


def parse_triggers(prefix):
    ext = '.txt'
    match = 'trig'
    line_match = 'Trigger: frame'
    l = os.listdir()
    f_in = None
    prefix_found = False
    for fn in l:
        if prefix in fn:
            prefix_found = True
            if fn.endswith(ext):
                if match in fn:
                    f_in = fn
                    break
    if f_in is None:
        if not prefix_found:
            print(f'Files for {prefix} not found in {os.getcwd()}')
        else:
            print(f'Trig files for {prefix} not found with pattern: match {match}, ext:{ext}')
        return None
    with open(fn, 'r') as f:
        lines = f.readlines()
    ts = []
    for l in lines:
        if line_match in l:
            t = l[l.find(line_match) + len(line_match):].strip()
            ts.append(int(''.join([x for x in t if x.isdigit()])))
    return ts


if __name__ == '__main__':
    path = 'X:\Barna_unprocessed/axax-2/'
    eye_folder = '.'
    os.chdir(path)
    pfn = '_included_pflist.txt'
    prefix = 'axax-2_187_452'
    pflist = prefix
    # with open(pfn, 'r') as f:
    #     pflist = [item.strip() for item in f.readlines()]
    suffix = '_eye.mat'

    # Run roi drawer and queue raw movie processing:

    request_queue = Queue()
    Worker(request_queue).start()
    # for prefix in pflist:
    for f in os.listdir('.'):
        if f.endswith(suffix):
            prefix = f[:-len(suffix)]
        else:
            continue
        if not os.path.exists(prefix + 'eye_trace.npy'):
            if not os.path.exists(prefix + '_eye-crop.json'):
                EyeTracing(prefix).crop()
            request_queue.put((path, prefix, eye_folder))

    # Pull trace from movement map:
    # prefix = 'axax_124_481'
    # for prefix in pflist:
    #     print(datetime.now(), prefix)
    #     mm = numpy.load(prefix + '_motion_map.npy', mmap_mode='r')
    #     # load location of eye bottom, to exclude that:
    #     ec = EyeTracing.load(prefix + '_eye-crop.json')
    #     mc = EyeTracing.load(prefix + '_motion-crop.json')
    #     face_trace = mm[:, :, int(ec[3] - mc[2]):mm.shape[-1] // 2].mean(axis=(1, 2))
    #     numpy.save(prefix + '_face_trace.npy', face_trace)
#
# et = numpy.load(prefix + 'eye_trace.npy')
# import pandas
# smet = numpy.array(pandas.DataFrame(et).ewm(span=15).mean()[0])

# '''create motion map as per Powell et al., 2015 Elife'''
# import datetime
# path = 'D://Barna//axax//eyes//'
# prefix = 'axax_124_231'
# os.chdir(path)
# crop = numpy.array([99.6807785554183, 587.9395041397922, 78.19581812359911, 437.20958693563887]).astype('int')
# # t0 = datetime.datetime.now()
# movie = h5py.File(prefix + '_eye.mat', 'r')['data']
# nframes = len(movie)
# print('Cropping movie')
# IM = movie[:, 0, crop[0]:crop[1], crop[2]:crop[3]]
# #downsample IM
# IM = IM[2 - len(IM) % 2:]
# IM = IM.reshape(len(IM) // 2, 2, *IM.shape[1:]).mean(axis=1)
# alpha = 0.3
# BG = numpy.empty(IM.shape)
# BG[0] = IM[:2].mean(axis=0)
# for fr in range(len(IM)-1):
#     i = fr + 1
#     BG[i] = alpha * IM[i] + (1 - alpha) * BG[i-1]
# D = numpy.abs(IM - BG)
# motion_map = numpy.empty(D.shape)
# motion_map[0] = D[:2].mean(axis=0)
# beta = 0.9
# for fr in range(len(D)-1):
#     i = fr + 1
#     motion_map[i] = beta * D[i] + (1 - beta) * motion_map[i-1]
# # normalize 0-1 and save as 8 bits
# motion_map = (motion_map * 255 / motion_map.max()).astype('uint8')
# numpy.save(prefix + '_motion_map', motion_map)

# import cv2
# class play_eye:
#     def __init__(self, M, raw, scaling):
#         eye=M
#         step = 1
#         self.frame = 0
#         cv2.namedWindow('Movie')
#         cv2.namedWindow('Raw')
#         nframes = len(eye)
#         tblen = 512
#         self.factor = tblen / nframes
#         cv2.createTrackbar('Frame', 'Movie', 0, tblen, self.tbonChange)
#         while self.frame < nframes:
#             if cv2.getWindowProperty('Movie', 0) < 0:
#                 break
#             im = (eye[self.frame]*scaling).transpose().astype('uint8').copy()
#             cv2.putText(im, str(self.frame), (0, 40),
#                         fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=128)
#             cv2.imshow('Movie', im)
#             cv2.imshow('Raw', raw[self.frame].transpose().astype('uint8'))
#             if cv2.waitKey(30) & 0xFF == ord('q'):
#                 break
#             self.frame += step
#             cv2.setTrackbarPos('Frame', 'Movie', int(self.frame * self.factor))
#         cv2.destroyAllWindows()
#
#     def tbonChange(self, v):
#         self.frame = int(v / self.factor)
#
# play_eye(motion_map, IM, 20)
# center=400,122
# size=22
#
# result=[]
# for i in range(len(eye)):
#     output = eye[i, 0, int(center[0] - size):int(center[0] + size), int(center[1] - size):int(center[1] + size)]
#     result.append(numpy.mean(output))
# print('Done')
# oa=numpy.array(result, dtype=numpy.float32)
# oa.tofile(t.replace('.mat', '.np'))
