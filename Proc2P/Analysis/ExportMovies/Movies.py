import cv2
from tkinter import *
import time
import numpy
from multiprocessing import Pipe, Process
import os
from collections import namedtuple
# for specific panels:
import h5py
from scipy.signal import decimate, resample
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Analysis.LoadPolys import LoadImage, LoadPolys
from matplotlib.patches import Circle
from Proc2P.Analysis.CaTrace import CaTrace
import json
from Video.LoadAvi import LoadAvi

'''This class is used to export videos of multimodal recordings.
The general idea is that we define "panels": each occupies an area of the video, and has it's own display logic.
panels include: 2P microscope image, video of the mouse, ephys traces, or DF/F traces.
The main window has a time slider, and a Clock object that advances the slider regularly. At each update, each panel's
update method is called.
When constructing the video, init each Panel subclass, (defining their sizes and positions), and then init the movie 
using the list of panels.
Updating the Panel classes is work in progress.
In the previous (Scanbox) implementation the slider units were frames / refresh * fps
In LNCM implementation, the slider is time in seconds.
Expect that every Panel's init and update functions need to be modified and tested.
Currently tested:  
'''


class Movie:
    def __init__(self, panels, refresh_rate, trig, slider=0, session=None, tag=None, add_sliders=(), title='Movie',
                 start=0, render=None, output_filename=None, cell=-1, testing=False, timetoframe_mode='precise',
                 vres=(1280, 720), playback_speed=1, ):
        '''
        Create the main movie window using the list of panels.
        :param panels: a list of initialized Panel classes
        :param refresh_rate: the refresh rate of the rendered movie
        :param trig: Connection object from multiprocessing.Pipe (for playback)
        :param slider: max value of time slider (unit:seconds)
        :param session: pass (path, prefix), an ImagingSession instance will be initialized in this process.
        :param tag: ROI tag of for ImagingSession
        :param add_sliders: list of additional sliders (cell)
        :param title: window title
        :param start: where playback will start (sec)
        :param render: if True, render each frame between start and stop and save to file. otherwise, interactive
        :param output_filename: '.avi'
        :param cell: to set the cell index (ImagingSession)
        :param testing: if True, does not start playback after init.
        :param timetoframe_mode: if 'precise', uses ImagingSession.timetoframe. if 'fps',
         uses 2P fps. suitable for 20 Hz Ca imaging.
        :param vres: resolution of the output (whether on screen or in file)
        :param playback_speed: default is 1: real time. can speed up or slow down the step size by this factor.
        '''
        if not render:
            render = None
        self.playback_speed = playback_speed
        self.refresh_rate = refresh_rate
        self.panels = panels
        self.title = title
        self.video_frame = 0  # this refers to rendered frame
        self.cell = cell
        self.time = 0.0  # this is the real recording time relative to start (ImagingSession RelativeTime)
        self.previous_frame_lookup = None
        self.timetoframe_mode = timetoframe_mode
        self.tag = tag
        self.sliderfunctions = {'Cell': self.onCellChange}
        if slider > 0:
            self.maxtime = slider
            self.slider = True
        else:
            self.maxtime = 0
            self.slider = False
        self.slider_moved = False
        h, w = 0, 0
        if session is not None:
            path, prefix = session
            self.session = ImagingSession(path, prefix, tag=tag)  # we will use this to get conversions
            # from time -> to frame, sample, video frame, etc
            self.fps_precision = 0.5 / self.session.fps
        for p in panels:
            p.set_parent(self)
            if p.need_session:
                p.set_session(self.session)
            p.nonpickle_init()
            w = max(w, p.x + p.w)  # size is determined by assembling panels.
            h = max(h, p.y + p.h)
        self.img = numpy.zeros((h, w, 3), dtype=numpy.uint8)  # container for the displayed image (RGB)
        self.draw()
        if self.slider:
            cv2.createTrackbar('Time', self.title, 0, self.maxtime, self.tbonChange)
        for s in add_sliders:
            cv2.createTrackbar(s.name, self.title, 0, s.range, self.sliderfunctions[s.name])
        cv2.setMouseCallback(self.title, self.mouse)
        cv2.setTrackbarPos('Time', self.title, start)
        self.kill = False

        # start timer to call update
        if not testing and render is None:
            self.sliders_enabled = True
            for _ in iter(trig.recv, None):
                cv2.waitKey(10)
                if cv2.getWindowProperty(self.title, 0) < 0:
                    self.kill = True
                    break
                self.tick()

        # or render all frames in file:
        if render is not None:
            self.sliders_enabled = False
            vrate = self.refresh_rate
            start, stop = render
            if output_filename is None:
                output_filename = prefix + '_movie.avi'
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(output_filename, fourcc, vrate, vres)
            prog = 0
            self.time = start
            while self.time < stop:
                if prog > refresh_rate:
                    cv2.waitKey(3) # to allow rendering occasional frames while exporting
                    prog = 0
                prog += 1
                self.tick()
                out.write(cv2.resize(self.img, vres))
            out.release()
            self.kill = True
            print('Movie saved:', output_filename)
            self.tick()

    def tick(self):
        '''Called every time by the Clock (for screen refresh). how much time will advance depends on playback speed'''
        if not self.kill:
            self.time += self.playback_speed / self.refresh_rate
            for p in self.panels:
                p.update(self.time, self.slider_moved, self.cell)
            self.draw()
            self.slider_moved = False
        else:
            cv2.destroyAllWindows()
            del self

    def get_2p_frame(self):
        '''which 2P frame belongs to the current time'''
        if self.previous_frame_lookup is None or abs(self.time - self.previous_frame_lookup) > self.fps_precision:
            if self.timetoframe_mode == 'precise':
                self.current_2P_frame = self.session.timetoframe(self.time)
            elif self.timetoframe_mode == 'fps':
                self.current_2P_frame = int(self.time * self.session.fps)
            self.previous_frame_lookup = self.time
        return self.current_2P_frame

    def tbonChange(self, v):
        if self.sliders_enabled:
            t = cv2.getTrackbarPos('Time', self.title)
            if int(t) != int(self.time):
                self.time = t
                self.slider_moved = True

    def onCellChange(self, v):
        self.cell = cv2.getTrackbarPos('Cell', self.title) - 1
        self.slider_moved = True

    def put_cell(self, cell):
        self.cell = cell
        cv2.setTrackbarPos('Cell', self.title, cell + 1)

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.pick(x, y)

    def pick(self, x, y):
        '''can get click coords from the movie window - used for selecting cells by clicking on them'''
        for p in self.panels:
            if p.x < x < p.x + p.w and p.y < y < p.y + p.h:
                p.put_pick(x - p.x, y - p.y)

    def draw(self):
        for p in self.panels:
            if p.updated:
                self.img[p.y: p.y + p.h, p.x: p.x + p.w] = p.draw()
        # cv2.waitKey(10)
        cv2.imshow(self.title, self.img)
        cv2.setTrackbarPos('Time', self.title, int(self.time))


class Clock:
    def __init__(self, conn, refresh_rate):
        self.conn = conn
        self.td = 1.0 / refresh_rate
        while True:
            time.sleep(self.td)
            self.conn.send(True)


class Panel:
    '''subclasses should be created. a draw and an update has to be defined.
    draw returns the pixel data. update does whatever necessary inside when time updates.
    Every panel instance should be initialized in the main program, before passing it to Movie.
    NB Movie is in a spearate process and everything passed to it will be pickled.
    So only do picklable, low-IO stuff in panel's init. Anything high IO should go to its nonpickle_init,
    which will be called by the Movie process after everything's set up and right before playback starts.
    '''

    def __init__(self, p):
        '''p is the panel size and position within the movie: (x0, y0, w, h)'''
        self.p = p
        self.x, self.y = p[0], p[1]
        self.w, self.h = p[2], p[3]
        self.im = numpy.zeros((self.h, self.w, 3), dtype=numpy.uint8)
        self.updated = False
        self.time = 0
        self.need_session = False
        self.session = None
        self.slider_moved = False
        self.cell = -1
        self.pick = None
        self.parent = None

    def set_session(self, session):
        '''this is called by the parent if need_session is set. done before calling nonpickle_init.'''
        self.session = session

    def set_parent(self, parent):
        self.parent = parent

    def update(self, time, slider, cell):
        self.cell = cell
        self.time = time
        self.slider_moved = slider
        self.run_update()

    def run_update(self):
        pass

    def nonpickle_init(self):
        '''this is called by the parent movie after setting the Session attribute. can use ImagingSession stuff.'''
        pass

    def put_pick(self, x, y):
        self.pick = (x, y)

    def draw(self):
        self.updated = False
        return self.im


class TestPanel(Panel):
    def draw(self):
        self.updated = False
        im = numpy.copy(self.im)
        cv2.putText(im, str(self.frame), (int(self.h * 0.5), int(self.w * 0.5)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255, 255, 255))
        return im


# class Panel_2PMovie(Panel):
#     def __init__(self, x, y, prefix, fps, refresh=15, smoothing=0, channel=(0,), print_frame=True, print_time=False,
#                  trim_x=70, trim_y=10, load_polys=False, scaling=False, tag=None, bbox=None, extra_top_cut=0,
#                  show_all=False, show_points=None, color=None,
#                  gamma=1.0, dff=False, dff_baseline=None, dff_scale_max=3, mc=None, print_title=(None, 0),
#                  show_scale=0, print_offset=0, time_offset=0):
#         info = loadmat(prefix + '.mat')['info']
#         p = (x, y, info['sz'][1] - 2 * trim_x, info['sz'][0] - 2 * trim_y - extra_top_cut)
#         super().__init__(p)
#         self.etc = extra_top_cut
#         self.gamma = gamma
#         self.refresh = float(fps) / refresh
#         self.fps = fps
#         self.prefix = prefix
#         self.cell_active = -1
#         self.cell_slider = -1
#         self.show_all = show_all
#         self.show_points = show_points
#         self.smoothing = smoothing
#         self.channel = channel
#         self.color=color
#         self.print_frame = print_frame
#         self.print_time = print_time
#         self.print_offset = print_offset
#         self.time_offset = time_offset
#         self.show_scale = show_scale
#         self.title, self.title_pos = print_title  # label and its position
#         self.trim_x = trim_x
#         self.trim_y = trim_y
#         self.bbox = bbox
#         self.load_polys = load_polys
#         self.movie_frame = 0
#         self.scaling = scaling
#         self.tag = tag
#         self.dff = dff
#         self.dff_baseline = dff_baseline
#         self.dff_scale_max = dff_scale_max
#         self.mc_kw = mc
#         # for picklability:
#         self.cam = None
#         self.polys = None
#         self.dff_values = {}
#
#     def nonpickle_init(self):
#         if self.mc_kw == None:
#             self.cam = LoadImage(self.prefix, explicit_need_data=True,)
#         elif self.mc_kw == 'raw':
#             self.cam = LoadImage(self.prefix, explicit_need_data=True, raw=True)
#         else:
#             fn = self.prefix + '_' + self.mc_kw + '.sbx'
#             self.cam = LoadImage(self.prefix, explicit_need_data=True, force=fn)
#         if self.load_polys:
#             self.polys = LoadPolys(self.prefix, tag=self.tag)
#             self.cms = numpy.empty((len(self.polys.data), 2))
#             for i in range(len(self.polys.data)):
#                 self.cms[i, :] = self.polys[i].mean(axis=0)
#         if self.dff:
#             print('Determining baseline for DFF')
#             trim_x, trim_y = self.trim_x, self.trim_y
#             for channel in self.channel:
#                 a = self.cam.data[slice(*self.dff_baseline),
#                     trim_y + self.etc:-trim_y, trim_x:-trim_x, channel].mean(axis=0)
#                 self.dff_values[channel] = 65535 - a
#         if self.show_scale > 0:
#             # calculate number of pixels to show scale
#             self.show_scale = int(self.show_scale / self.cam.pixelsize)
#
#     def run_update(self):
#         m_frame = int(self.frame / self.refresh)
#         trim_x, trim_y = self.trim_x, self.trim_y
#         if self.load_polys:
#             self.pick_cell()
#         if m_frame != self.movie_frame:
#             self.im[:, :, :] = 0
#             self.movie_frame = m_frame
#             for ci, channel in enumerate(self.channel):
#                 if self.smoothing == 0:
#                     a = numpy.array(self.cam.data[m_frame, trim_y + self.etc:-trim_y, trim_x:-trim_x, channel],
#                                     dtype='float')
#                 else:  # perfofrm smoothing if set:
#                     t0 = int(max(0, m_frame - self.smoothing))
#                     t1 = int(min(self.cam.nframes, m_frame + self.smoothing + 1))
#                     a = self.cam.data[t0:t1, trim_y + self.etc:-trim_y, trim_x:-trim_x, channel].mean(axis=0)
#                 a = 65535 - a
#                 if self.dff:
#                     dff = numpy.maximum(a - self.dff_values[channel], 0) / self.dff_values[channel]
#                     dff = numpy.minimum(dff, self.dff_scale_max)
#                     a = dff / self.dff_scale_max
#                 else:
#                     a /= 65535
#                 if self.scaling:
#                     a = numpy.minimum(a * self.scaling, 1)
#                 a = a ** (1.0 / self.gamma)
#                 #add to image
#                 if self.color is None:
#                     self.im[:, :, channel + 1] = (a * 255).astype('uint8')
#                 else:
#                     color = self.color[ci]
#                     for vi, v in enumerate(color):
#                         if v:
#                             self.im[:, :, vi] = (a * 255 * v).astype('uint8')
#             if self.print_frame:
#                 cv2.putText(self.im, str(m_frame), (2, 22 + self.print_offset),
#                             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(250, 255, 250))
#             if self.print_time:
#                 t = self.frame / self.fps
#                 if t >= self.time_offset:
#                     mins, sec = divmod(t - self.time_offset, 60)
#                     ttext = f'{int(mins)}:{int(sec):02}'
#                 else:
#                     mins, sec = divmod(self.time_offset - t, 60)
#                     ttext = f'-{int(mins)}:{int(sec):02}'
#                 cv2.putText(self.im, ttext, (2, 22 + 22 * self.print_frame + self.print_offset),
#                             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(250, 255, 250))
#             if self.title is not None:
#                 cv2.putText(self.im, self.title, (2, self.title_pos),
#                             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(250, 255, 250))
#             if self.load_polys and self.cell_active > -1:
#                 if self.show_all:
#                     for c in range(len(self.polys.data)):
#                         self.cell_active = c
#                         cv2.polylines(self.im, self.getroi(), isClosed=True, color=(255, 255, 255))
#                 else:
#                     cv2.polylines(self.im, self.getroi(), isClosed=True, color=(255, 255, 255))
#             if self.show_points is not None:
#                 for p in self.getpoints():
#                     cv2.polylines(self.im, [numpy.array(p).transpose()], isClosed=True, color=(255, 255, 255))
#             if self.show_scale > 0:
#                 ypos = self.p[-1] - 10
#                 cv2.line(self.im, (10, ypos), (10 + self.show_scale, ypos), (255, 255, 255), 2)
#             self.updated = True
#
#     def pick_cell(self):
#         if self.pick != None:
#             x, y = self.pick
#             x += self.trim_x
#             y += self.trim_y
#             mini, mind = 0, 99999
#             for i, cm in enumerate(self.cms):
#                 d = numpy.sqrt((x - cm[0]) ** 2 + (y - cm[1]) ** 2)
#                 if d < mind:
#                     # print (x, y, cm[0], cm[1], d)
#                     mind = d
#                     mini = i
#             self.pick = None
#             print(f'Cell picked: {mini}')
#             self.parent.put_cell(mini)
#             self.cell_active = mini
#         elif self.slider_moved:
#             self.cell_active = self.cell
#
#     def getroi(self):
#         new = []
#         for p in self.polys[self.cell_active]:
#             x = int(round(p[0])) - self.trim_x
#             y = int(round(p[1])) - self.trim_y
#             np = [x, y]
#             if np not in new:
#                 new.append(np)
#         return [numpy.array(new)]
#
#     def getpoints(self):
#         new = []
#         for p in self.show_points:
#             circle = Circle(p, 5)
#             p = circle.get_verts().transpose()
#             new.append([(p[0] - self.trim_x).round().astype('int'), (p[1] - self.trim_y).round().astype('int')])
#         return new
#

class Panel_MouseCam(Panel):
    def __init__(self, x, y, cam_file, gamma=2.2, scale=1,
                 trim_x=0, trim_y=0, crop=None, available_space=None):
        cam = LoadAvi(cam_file)
        h = cam.h
        w = cam.w
        if crop is None:
            p = (x, y, int((w - 2 * trim_x) * scale), int((h - 2 * trim_y) * scale))
        else:
            p = available_space
            crop_w = abs(crop[1] - crop[0])
            crop_h = abs(crop[3] - crop[2])
            scale = min(p[2] / crop_w, p[3] / crop_h)
            p = (x, y, int(crop_w * scale) + 1, int(crop_h * scale) + 1)
        super().__init__(p)
        self.crop = crop
        self.cam_file = cam_file
        self.need_session = False  # but parent needs session
        self.movie_frame = 0  # frame in the input video file
        self.gamma = numpy.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
        # for picklability:
        self.cam = None
        self.trim_x = trim_x
        self.trim_y = trim_y
        self.scale = scale

    def nonpickle_init(self):
        self.cam = LoadAvi(self.cam_file)
        assert hasattr(self.parent, 'session')
        self.syncframes = self.parent.session.ca.sync.load('cam')

    def run_update(self):
        # if time_to_frame is slow, we can 1) do it in parent once per update, 2) unless necessary to avoid, use time/fps
        next_frame = self.syncframes[self.parent.get_2p_frame()]

        if next_frame != self.movie_frame:
            self.movie_frame = next_frame
            fdat = self.cam[next_frame]
            if self.crop is None:
                im_data = fdat[self.trim_x:fdat.shape[0] - self.trim_x, self.trim_y:fdat.shape[1] - self.trim_y]
            else:
                im_data = self.cam[next_frame][self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
            id = cv2.LUT(im_data, self.gamma)
            if not self.scale == 1:
                id = cv2.resize(id, (0, 0), fx=self.scale, fy=self.scale)
            for i in range(3):
                xmax = min(self.im.shape[0], id.shape[0])
                ymax = min(self.im.shape[1], id.shape[1])
                self.im[:xmax, :ymax, i] = id[:xmax, :ymax]
            self.updated = True


# class Panel_Speed(Panel):
#     def __init__(self, p, prefix, fps, refresh, alt_trace=None, color=None, scaling=1, title=None, normmode='float'):
#         super().__init__(p)
#         self.prefix = prefix
#         self.refresh = float(fps) / refresh
#         self.fs = fps * 60 / self.refresh
#         self.min = 0
#         self.pts = []
#         if color is None:
#             self.color = (128, 64, 64)
#         else:
#             self.color = color
#         self.title = title
#         self.normmode = normmode
#         # for picklability:
#         self.speed = None
#         self.scaling = scaling
#         self.alt_trace = alt_trace
#
#     def nonpickle_init(self):
#         if self.alt_trace is None:
#             q = numpy.maximum(numpy.minimum(SplitQuad(self.prefix).speed, 20), -5)
#             self.speed = -q * 0.8 / q.max()
#         else:
#             s = self.alt_trace
#             if self.normmode == 'float':
#                 s -= numpy.nanmean(s)
#                 s /= numpy.nanpercentile(s, 99)
#                 s[numpy.where(numpy.isnan(s))] = 0
#                 s = numpy.minimum(s, 1)
#             elif self.normmode == 'max':
#                 s /= numpy.nanmax(s)
#             elif self.normmode == 'bin':
#                 pass
#             self.speed = s
#
#             self.speed /= - numpy.nanmax(self.speed) / self.scaling
#         self.getmin()
#
#     def run_update(self):
#         sample = int(self.frame / self.refresh)
#         new_min, curr = divmod(sample, self.fs)
#         # update trace only in new minute
#         if new_min != self.min or self.slider_moved:
#             self.min = new_min
#             self.getmin()
#         curr = int(curr * self.p[2] / self.fs)
#         if self.normmode in ['float', 'max']:
#             linetype = cv2.LINE_AA
#         elif self.normmode == 'bin':
#             linetype = 4
#         cv2.polylines(self.im, [self.pts[:curr]], isClosed=False, thickness=2, lineType=linetype,
#                       color=self.color)
#         self.updated = True
#
#     def getmin(self):
#         h = int(self.p[3] / 2)
#         l = self.p[2]
#         x = numpy.arange(l)
#         self.im = numpy.zeros((2 * h, l, 3), dtype=numpy.uint8)
#         y = self.speed[int(self.min * self.fs): int((self.min + 1) * self.fs)]
#         y = resample(y, l)
#         y = h * y * 1.5 + h * 1.5
#         self.pts = numpy.vstack((x, y)).astype('int32').T
#         cv2.polylines(self.im, [self.pts], isClosed=False, thickness=2, lineType=cv2.LINE_AA,
#                       color=(128, 128, 128))
#         if self.title is not None:
#             cv2.putText(self.im, self.title, (0, 20),
#                         fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=self.color)
#

# class Panel_GCAMP(Panel):
#     def __init__(self, p, prefix, fps, refresh, param=None, title=None, color=None, ch=0, tag=None, set_cell=None):
#         super().__init__(p)
#         self.prefix = prefix
#         self.refresh = float(fps) / refresh
#         self.fs = fps * 60 / self.refresh
#         self.min = 0
#         self.pts = []
#         if set_cell is None:
#             self.cell_active = -1
#             self.set_cell = False
#         else:
#             self.cell_active = set_cell
#             self.set_cell = True
#         self.tag = tag
#         if param is None:
#             param = 'smtr'
#         self.param = param
#         self.title = title
#         if color is None:
#             color = (155, 202, 60)
#         self.color = color
#         self.ch = ch
#         # for picklability:
#         self.ca = None
#         self.trace = None
#
#     def nonpickle_init(self):
#         a = Firing(self.prefix, ch=self.ch, tag=self.tag)
#         a.load()
#         if self.param == 'raw':
#             self.ca = numpy.load(f'{self.prefix}_trace_{self.tag}.npy')[..., self.ch]
#         else:
#             self.ca = getattr(a, self.param)
#
#     def run_update(self):
#         if self.cell > -1:
#             sample = int(self.frame / self.refresh)
#             new_min, curr = divmod(sample, self.fs)
#             if self.cell != self.cell_active:
#                 if not self.set_cell:
#                     self.cell_active = self.cell
#                 self.cell_update()
#                 self.getmin()
#             # update trace only in new minute
#             if new_min != self.min or self.cell != self.cell_active or self.slider_moved:
#                 self.min = new_min
#                 self.getmin()
#             curr = int(curr * self.p[2] / self.fs)
#             cv2.polylines(self.im, [self.pts[:curr]], isClosed=False, thickness=2, lineType=cv2.LINE_AA,
#                           color=self.color)
#             self.updated = True
#
#     def cell_update(self):
#         trace = self.ca[self.cell_active]
#         trace = trace - trace.mean()
#         self.trace = trace / numpy.nanmax(trace)
#
#     def getmin(self):
#         h = int(self.p[3] / 2)
#         l = self.p[2]
#         x = numpy.arange(l)
#         self.im = numpy.zeros((2 * h, l, 3), dtype=numpy.uint8)
#         if self.trace is None:
#             self.cell_update()
#         y = self.trace[int(self.min * self.fs): int((self.min + 1) * self.fs)]
#         y = resample(y, l)
#         y = h * y * -1 + h * 1.5
#         self.pts = numpy.vstack((x, y)).astype('int32').T
#         cv2.polylines(self.im, [self.pts], isClosed=False, thickness=2, lineType=cv2.LINE_AA,
#                       color=(128, 128, 128))
#         if self.title is not None:
#             cv2.putText(self.im, self.title, (0, 20),
#                         fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=self.color)
#

# class Panel_Dual(Panel):
#     def __init__(self, p, prefix, fps, refresh, param=None, title=None, color=None, tag=None, set_cell=None):
#         super().__init__(p)
#         self.prefix = prefix
#         self.refresh = float(fps) / refresh
#         self.fs = fps * 60 / self.refresh
#         self.min = 0
#         self.pts = []
#         if set_cell is None:
#             self.cell_active = -1
#             self.set_cell = False
#         else:
#             self.cell_active = set_cell
#             self.set_cell = True
#         self.tag = tag
#         if param is None:
#             param = 'smtr'
#         self.param = param
#         self.title = title
#         self.color = color
#         # for picklability:
#         self.ca = None
#         self.trace = None
#
#     def nonpickle_init(self):
#         self.ca = []
#         for ch in (0, 1):
#             a = Firing(self.prefix, ch=ch, tag=self.tag)
#             a.load()
#             self.ca.append(getattr(a, self.param))
#
#     def run_update(self):
#         if self.cell > -1:
#             sample = int(self.frame / self.refresh)
#             new_min, curr = divmod(sample, self.fs)
#             if self.cell != self.cell_active:
#                 if not self.set_cell:
#                     self.cell_active = self.cell
#                 self.cell_update()
#                 self.getmin()
#             # update trace only in new minute
#             if new_min != self.min or self.cell != self.cell_active or self.slider_moved:
#                 self.min = new_min
#                 self.getmin()
#             curr = int(curr * self.p[2] / self.fs)
#             for pts, color in zip(self.pts, self.color):
#                 cv2.polylines(self.im, [pts[:curr]], isClosed=False, thickness=2, lineType=cv2.LINE_AA,
#                               color=color)
#             self.updated = True
#
#     def cell_update(self):
#         self.trace = []
#         for trace in self.ca:
#             if not self.cell_active == 'merge':
#                 trace = trace[self.cell_active]
#             trace = trace - trace.mean()
#             self.trace.append(trace / numpy.nanmax(trace))
#
#     def getmin(self):
#         h = int(self.p[3] / 2)
#         l = self.p[2]
#         x = numpy.arange(l)
#         self.im = numpy.zeros((2 * h, l, 3), dtype=numpy.uint8)
#         if self.trace is None:
#             self.cell_update()
#         self.pts = []
#         for trace, color in zip(self.trace, self.color):
#             y = trace[int(self.min * self.fs): int((self.min + 1) * self.fs)]
#             y = resample(y, l)
#             y = h * y * -1 + h * 1.5
#             pts = numpy.vstack((x, y)).astype('int32').T
#             self.pts.append(pts)
#             cv2.polylines(self.im, [pts], isClosed=False, thickness=2, lineType=cv2.LINE_AA,
#                           color=tuple([int(x/2) for x in color]))
#         # if self.title is not None:
#         #     cv2.putText(self.im, self.title, (0, 20),
#         #                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=self.color)
#

# class Panel_Trace(Panel):
#     def __init__(self, p, prefix, fps, refresh, ch_config=(1, 1), title=None, color=None, compress=1, show_scale=False,
#                  add_messages=None):
#         super().__init__(p)
#         self.prefix = prefix
#         self.samplerate = 10000
#         self.refresh = float(fps) / refresh
#         self.sec = 0
#         self.pts = []
#         self.title = title
#         if color is None:
#             color = (112, 195, 237)
#         self.color = color
#         self.compress = compress
#         self.show_scale = show_scale
#         self.add_messages = add_messages  # list of tuples: second, text
#         # for picklability:
#         self.trace = None
#         self.ephysframes = None
#         self.ch_config = ch_config
#
#     def nonpickle_init(self):
#         ch, n_channels = self.ch_config
#         raw_shape = n_channels + 1
#         ep_raw = numpy.fromfile(self.prefix + '.ephys', dtype='float32')
#         n_samples = int(len(ep_raw) / raw_shape)
#         ep_formatted = numpy.reshape(ep_raw, (n_samples, raw_shape))
#         self.trace = -ep_formatted[:, ch]
#         self.ephysframes = ep_formatted[:, 0]
#         self.n = n_samples
#         # self.ephys_all_channels = ep_formatted[:, 1:]
#         self.trace -= self.trace.mean()
#         self.trace /= numpy.percentile(numpy.absolute(self.trace), 99)
#         self.getsec()
#         if self.show_scale > 0:
#             h = int(self.p[3] / 2)
#             self.show_scale = int(h * self.show_scale / self.compress)
#
#     def run_update(self):
#         sample = numpy.argmax(self.ephysframes > (int(self.frame / self.refresh)))
#         new_sec, curr = divmod(sample, self.samplerate)
#         # update trace only in new seconds, redraw up till current sample with different color in each rendered frame
#         if new_sec != self.sec or self.slider_moved:
#             self.sec = new_sec
#             self.getsec()
#         curr = int(curr * self.p[2] / self.samplerate)
#         cv2.polylines(self.im, [self.pts[:curr]], isClosed=False, color=self.color)
#         self.updated = True
#
#     def getsec(self):
#         h = int(self.p[3] / 2)
#         l = self.p[2]
#         x = numpy.arange(l)
#         self.im = numpy.zeros((2 * h, l, 3), dtype=numpy.uint8) * 255
#         y = self.trace[self.sec * self.samplerate: (self.sec + 1) * self.samplerate]
#         y = decimate(y, 10)
#         y = resample(y, l)
#         y = h * y / self.compress + h
#         self.pts = numpy.vstack((x, y)).astype('int32').T
#         cv2.polylines(self.im, [self.pts], isClosed=False,
#                       color=(128, 128, 128))
#         if self.title is not None:
#             cv2.putText(self.im, self.title, (20, 22),
#                         fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=self.color)
#         if self.show_scale > 0:
#             cv2.line(self.im, (10, 10), (10, 10 + self.show_scale), (255, 255, 255), 2)
#         if self.add_messages is not None:
#             for t, mtext in self.add_messages:
#                 if self.sec == t + 1:
#                     cv2.putText(self.im, mtext, (20, 50),
#                                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(250, 255, 250))
#
#
# class Panel_LFP(Panel):
#     '''This doesn't update every second but shows a resampled lfp that is continous bw duration'''
#
#     def __init__(self, p, prefix, fps, refresh, duration, title=None, color=None, ):
#         super().__init__(p)
#         self.prefix = prefix
#         self.samplerate = 10000
#         self.refresh = float(fps) / refresh
#         self.pts = []
#         self.duration = duration
#         self.sample0 = 0
#         self.sample1 = fps * self.samplerate
#         self.title = title
#         if color is None:
#             color = (112, 195, 237)
#         self.color = color
#         # for picklability:
#         self.trace = None
#         self.ephysframes = None
#
#     def nonpickle_init(self):
#         a = numpy.fromfile(self.prefix + '.ephys', dtype='float32')
#         b = numpy.reshape(a, (int(len(a) / 2), 2))
#         self.trace = b[:, 1]
#         self.ephysframes = b[:, 0]
#         self.trace -= self.trace.mean()
#         self.trace /= numpy.percentile(numpy.absolute(self.trace), 99)
#         self.getsec()
#
#     def run_update(self):
#         sample = numpy.argmax(self.ephysframes > (int(self.frame / self.refresh)))
#         curr = (sample - self.sample0) / (self.sample1 - self.sample0)  # point in trace
#         # update trace only in new seconds, redraw up till current sample with different color in each rendered frame
#         curr = int(curr * self.p[2])
#         cv2.polylines(self.im, [self.pts[:curr]], isClosed=False, color=self.color, thickness=1)
#         if self.title is not None:
#             cv2.putText(self.im, self.title, (0, 20),
#                         fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=self.color)
#         self.updated = True
#
#     def getsec(self):
#         h = int(self.p[3] / 2)
#         l = self.p[2]
#         x = numpy.arange(l)
#         self.sample0 = numpy.argmax(self.ephysframes > (int(self.duration[0] / self.refresh)))
#         self.sample1 = numpy.argmax(self.ephysframes > (int(self.duration[1] / self.refresh)))
#         self.im = numpy.ones((2 * h, l, 3), dtype=numpy.uint8) * 255
#         y = numpy.copy(self.trace[self.sample0:self.sample1])
#         factor = len(y) / l
#         while factor > 2:
#             q = min(factor, 10)
#             y = decimate(y, int(q))
#             factor = len(y) / l
#         y = resample(y, l)
#         y = h * y + h
#         self.pts = numpy.vstack((x, y)).astype('int32').T
#         cv2.polylines(self.im, [self.pts], isClosed=False,
#                       color=(250, 215, 177))
#
# class Sliding_Trace(Panel):
#     #TODO started working on this
#     '''This doesn't update every second, continous bw duration'''
#
#     def __init__(self, p, prefix, fps, refresh, session_length, trace, length='minute', title=None, color=None):
#         super().__init__(p)
#         self.prefix = prefix
#         self.trace = trace
#         self.refresh = float(fps) / refresh
#         self.pts = []
#         self.title = title
#         self.session_length = session_length
#         if length == 'minute':
#             self.length = 30 * self.samplerate
#         if color is None:
#             color = (112, 195, 237)
#         self.color = color
#         # for picklability:
#         self.trace = None
#         self.ephysframes = None
#         self.resampling_factor = None
#         self.plot_y = None
#
#     def nonpickle_init(self):
#         a = numpy.fromfile(self.prefix + '.ephys', dtype='float32')
#         b = numpy.reshape(a, (int(len(a) / 2), 2))
#         trace = b[:, 1]
#         self.ephysframes = b[:, 0]
#         trace -= trace.mean()
#         trace /= numpy.percentile(numpy.absolute(trace), 99) * 1.5
#         self.trace = trace
#         y = numpy.copy(trace)
#         l = self.p[2]
#         session_pixels = self.session_length * l
#         factor = len(y) / l
#         while factor > 10:
#             y = decimate(y, 10)
#             factor = len(y) / l
#         y = resample(y, int(session_pixels) + 1)
#         self.plot_y = y
#         self.resampling_factor = len(trace) / len(y)
#
#     def run_update(self):
#         sample = numpy.argmax(self.ephysframes > (int(self.frame / self.refresh)))
#         t0 = int(max(0, sample - self.length) / self.resampling_factor)
#         t1 = int(min((sample + self.length) / self.resampling_factor, len(self.plot_y)))
#
#         h = int(self.p[3] / 2)
#         l = self.p[2]
#         x = numpy.arange(t1 - t0)
#         self.im = numpy.ones((2 * h, l, 3), dtype=numpy.uint8) * 255
#         self.im[:, l // 2, :] = 0
#         y = self.plot_y[t0:t1] * h + h
#         self.pts = numpy.vstack((x, y)).astype('int32').T
#         cv2.polylines(self.im, [self.pts], isClosed=False, color=self.color, lineType=8)
#         if self.title is not None:
#             cv2.putText(self.im, self.title, (0, 20),
#                         fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=self.color)
#         self.updated = True
#

class Sliding_LFP(Panel):
    '''This doesn't update every second but shows a resampled lfp that is continous bw duration'''

    def __init__(self, p, length=30, title=None, color=None):
        super().__init__(p)
        self.pts = []
        self.length = length
        self.title = title
        if color is None:
            color = (112, 195, 237)
        self.color = color
        self.nyquistrate = 2.7

        # for picklability:
        self.trace = None
        self.ephysframes = None
        self.resampling_factor = None
        self.plot_y = None

    def nonpickle_init(self):
        self.samplerate = self.parent.session.si.info['fs']

        # downsample the entire trace at the start.
        trace = numpy.copy(self.parent.session.ephys.trace)
        nsamples = len(trace)
        trace -= trace.mean()
        trace /= numpy.percentile(numpy.absolute(trace), 99) * 1.5
        l = self.p[2]
        session_duration = len(trace) / self.samplerate
        session_pixels = round((session_duration / (2 * self.length)) * l * self.nyquistrate)
        factor = len(trace) / session_pixels
        while factor > 10:
            trace = decimate(trace, 10)
            factor = len(trace) / session_pixels
        trace = resample(trace, session_pixels)
        self.plot_y = trace
        self.resampling_factor = nsamples / len(trace)
        self.halfscreen = int(self.length * self.samplerate / self.resampling_factor)

    def run_update(self):
        sample = self.parent.session.frametosample(self.parent.get_2p_frame()) / self.resampling_factor
        t0 = int(max(0, sample - self.halfscreen))
        t1 = int(min(sample + self.halfscreen, len(self.plot_y)))

        h = int(self.p[3] / 2)
        l = self.p[2]
        xmin = l / 2 - (sample - t0) / self.nyquistrate
        xmax = l / 2 + (t1 - sample) / self.nyquistrate
        x = numpy.linspace(xmin, xmax, t1-t0)
        self.im = numpy.ones((2 * h, l, 3), dtype=numpy.uint8) * 255
        self.im[:, l // 2, :] = 0
        y = self.plot_y[t0:t1] * h + h
        self.pts = numpy.vstack((x, y)).astype('int32').T
        cv2.polylines(self.im, [self.pts], isClosed=False, color=self.color, lineType=8)
        if self.title is not None:
            cv2.putText(self.im, self.title, (0, 20),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=self.color)
        self.updated = True


Slider = namedtuple('Slider', ['name', 'range'])
# range can be number, or use 'ncells' to fill with value during init
# make sure to define an onchange function for the name. Available: 'Cell'

# First, init all the Panel classes, then call movie with the list of panels
if __name__ == '__main__':
    # path = 'C://2pdata//movie//'
    # prefix = 'cckcre_2_062_241'
    session = None
    fps = 25
    refresh = 20
    # # p1 = TestPanel((0,0,256,256))
    # cam_file = r'D:\\Shares\\Data\\_RawData\\Bruker\\LFP-Pupil\\JEDI-Sncg80_2025-04-18_lfp_005-000/JEDI-Sncg80_2025-04-18_lfp_005.avi'
    # cam = Panel_MouseCam(0, 0, cam_file, fps, refresh)
    # cx, cy = cam.p[2:]
    # field = Panel_2PMovie(cx, 0, prefix, fps, refresh)
    # gx, gy = field.p[2:]
    # speed = Panel_Speed((0, cy, cx, 256), prefix, fps, refresh)
    # celltrace = Panel_GCAMP((cx, gy, gx, cy + 256 - gy), prefix, fps, refresh)
    # tx, ty = celltrace.p[2:]
    # trace = Panel_Trace((0, ty, tx, 256), prefix, fps, refresh)
    # # # panels = [p1, cam, trace]
    # panels = [cam, field, speed, celltrace, trace]
    # cellslider = Slider('Cell', 100)
    # sliders = [cellslider]
    # trig, conn = Pipe(duplex=False)
    # testing = False
    # nframes = 5000
    # # '''def __init__(self, panels, fps, trig, slider=0, session=None, title='Movie', testing=False):'''
    # if testing:
    #     tp = Movie(panels, fps, trig, nframes, session, sliders, testing=True)
    # else:
    #     t1 = Process(target=Movie, args=(panels, fps, trig, nframes, session, sliders))
    #     t2 = Process(target=Clock, args=(conn, fps))
    #     t1.start()
    #     t2.start()
