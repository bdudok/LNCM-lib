import tifffile

from Proc2P.Analysis.LoadPolys import FrameDisplay
from Proc2P.utils import lprint, logger
import matplotlib.path as mplpath
import os
import shutil
import scipy
import cv2
import numpy
import zipfile
from shapely.geometry import Polygon, Point
import sima
from multiprocessing import Process
from TkApps import SourceTarget, PickFromList, ShowMessage
import copy


class MouseTest:
    def __init__(self):
        im = numpy.empty((100, 100), dtype='uint8')
        cv2.imshow('Test', im)
        cv2.setMouseCallback('Test', self.drawmouse)

    def drawmouse(self, event, y, x, flags, param):
        print(f'event: {event}, x, y, flag: {flags}')
        if event == cv2.EVENT_RBUTTONDBLCLK:
            print('Double Right!')
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            print('Double Left!')
        elif event == cv2.EVENT_RBUTTONDOWN:
            print('Right!')
        elif event == cv2.EVENT_LBUTTONDOWN:
            print('Left!')


class Worker(Process):
    __name__ = 'RoiWorker'
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue

    def run(self):
        for din in iter(self.queue.get, None):
            # the whole thing goes in a try so worker is not dead on error. remove this for debugging.
            try:
                path, prefix, apps, config = din
                log = logger()
                log.set_handle(path, prefix)
                lprint(self, 'Calling autodetect with:', din, logger=log)
                RoiEditor(path, prefix, ).autodetect(approach=apps, config=config, log=log)
            except Exception as e:
                lprint(self, 'Autodetect failed with error:', e,)


class Lasso:
    def __init__(self, parent, mode='add'):
        self.parent = parent
        im = numpy.array(parent.pic)
        if mode == 'add':
            cv2.polylines(im, parent.psets[parent.current_key], isClosed=True, color=parent.colors['active'])
        if mode == 'remove':
            ps = []
            for poly in parent.saved_rois:
                pts = numpy.array(poly, dtype='int32')
                pts = pts.reshape(-1, 1, 2)
                ps.append(pts)
            cv2.polylines(im, ps, isClosed=True, color=parent.colors['saved'])
        self.im = cv2.LUT(im, parent.lut)
        self.polys = []
        self.coords = []
        self.drawing = False
        cv2.imshow('Lasso', self.im)
        cv2.moveWindow('Lasso', 0, 0)
        cv2.setMouseCallback('Lasso', self.drawmouse)
        self.retval = False
        while self.retval is False:
            if cv2.getWindowProperty('Lasso', 0) < 0:
                break
            k = cv2.waitKey(1) & 0xFF
        cv2.destroyWindow('Lasso')

    def drawmouse(self, event, y, x, flags, param):
        if event == cv2.EVENT_RBUTTONDBLCLK or flags == 33:
            self.retval = -1
        elif event == cv2.EVENT_LBUTTONDBLCLK or flags == 17 or event == cv2.EVENT_FLAG_SHIFTKEY:
            if len(self.polys) > 0:
                self.retval = self.polys
            else:
                self.retval = -1
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.coords) > 2:
                cv2.line(self.im, (y, x), (self.coords[0][1], self.coords[0][0]), color=(255, 255, 0))
                self.clean_poly()
                cv2.imshow('Lasso', self.im)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

        if event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.coords.append([x, y])
            if len(self.coords) > 1:
                cv2.line(self.im, (y, x), (self.coords[-2][1], self.coords[-2][0]), color=(255, 255, 0))
                cv2.imshow('Lasso', self.im)

    def clean_poly(self):
        new = []
        for p in self.coords:
            x = round(p[0])
            y = round(p[1])
            np = [y, x]
            if not np in new:
                new.append(np)
        new.append(new[0])
        self.polys.append(new)
        self.coords = []


class Newroi:
    def __init__(self, parent, x, y):
        self.parent = parent
        parent.pos = (y, x)
        self.im, self.start = parent.zoom('res', ret=True)
        self.polys = []
        self.coords = []
        self.drawing = False
        cv2.imshow('Draw', self.im)
        cv2.moveWindow('Draw', int(800 + x - self.parent.zoomsize / 2),
                       min(int(600 + y - self.parent.zoomsize / 2), int(1080 - self.parent.zoomsize)))
        cv2.setMouseCallback('Draw', self.drawmouse)
        self.retval = False
        while self.retval is False:
            if cv2.getWindowProperty('Draw', 0) < 0:
                break
            k = cv2.waitKey(1) & 0xFF
        cv2.destroyWindow('Draw')

    def drawmouse(self, event, y, x, flags, param):
        if event == cv2.EVENT_RBUTTONDBLCLK or flags == 33:
            self.retval = -1
        elif event == cv2.EVENT_LBUTTONDBLCLK or flags == 17:
            if len(self.polys) > 0:
                self.retval = self.polys
            else:
                self.retval = -1
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.coords) > 2:
                cv2.line(self.im, (y, x), (self.coords[0][1], self.coords[0][0]), color=(255, 255, 0))
                self.clean_poly()
                cv2.imshow('Draw', self.im)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

        if event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.coords.append([x, y])
            if len(self.coords) > 1:
                cv2.line(self.im, (y, x), (self.coords[-2][1], self.coords[-2][0]), color=(255, 255, 0))
                cv2.imshow('Draw', self.im)

    def clean_poly(self):
        new = []
        for p in self.coords:
            x = self.start[0] + round(p[0] / self.parent.zoomfactor)
            y = self.start[1] + round(p[1] / self.parent.zoomfactor)
            np = [y, x]
            if not np in new:
                new.append(np)
        new.append(new[0])
        self.polys.append(new)
        self.coords = []


class Translate:
    def __init__(self, path):
        self.path = path
        st = SourceTarget(path).ret
        print(st)
        self.src, self.tgt = st
        self.translate = [0, 0]
        self.data = None

        os.chdir(path)
        self.rois = RoiEditor(self.src)
        self.sim, self.tim = self.getims()
        self.pset = self.loadrois()

        cv2.polylines(self.sim, self.pset, isClosed=True, color=(0, 255, 255))
        cv2.imshow('Source', self.sim)
        cv2.moveWindow('Source', 0, 0)

        self.draw()
        cv2.moveWindow('Target', 800, 0)

        interactive = True
        while interactive:
            if cv2.getWindowProperty('Target', 0) < 0:
                break
            if cv2.getWindowProperty('Source', 0) < 0:
                break
            k = cv2.waitKey(25) & 0xFF
            if k == ord('w'):
                self.translate[1] -= 1
                self.draw()
            elif k == ord('s'):
                self.translate[1] += 1
                self.draw()
            elif k == ord('a'):
                self.translate[0] -= 1
                self.draw()
            elif k == ord('d'):
                self.translate[0] += 1
                self.draw()
            elif k == ord('r'):
                self.translate = [0, 0]
            elif k == ord('q'):
                ret = PickFromList(['Exit without saving', 'Save to sbx', 'Export ROI', 'Cancel']).ret
                if ret == 0:
                    interactive = False
                elif ret == 1:
                    self.save(sbx=True)
                    interactive = False
                elif ret == 2:
                    self.save(sbx=False)
                    interactive = False
            # elif k == ord('i') or k == ord('o'):
            #         # cycle active key with wheel
            #         i = self.active_keys.index(self.current_key)
            #         if k == ord('i'):
            #             i -= 1
            #         else:
            #             i+= 1
            #         # i = max(0, min(i, len(self.active_keys)-1))
            #         if not i < len(self.active_keys):
            #             i = 0
            #         if i < 0:
            #             i = len(self.active_keys) - 1
            #         self.current_key = self.active_keys[i]
            #         self.calc_sets(self.current_key)
            #         self.draw()
        cv2.destroyAllWindows()

    def save(self, sbx=True):
        exs = [0]
        for f in os.listdir():
            if self.tgt in f and '_saved_roi_' in f:
                try:
                    # ugly way to keep autodetected rois from breaking numbering
                    if f[-6:-4] != '-1':
                        exs.append(int(f[:-4].split('_')[-1]))
                except:
                    pass
        fn = self.tgt + '_saved_roi_' + str(max(exs) + 1)
        RoiEditor.save_roi(self.data, fn, self.rois.img.image.info['sz'], self.translate)
        message = f'{len(self.data)} saved in {fn}'
        print(message)
        if sbx:
            Process(target=save_sbx, args=(self.tgt, self.tim.shape, self.data, self.path)).start()

    def draw(self):
        im = numpy.copy(self.tim)
        pset = copy.deepcopy(self.pset)
        for p in pset:
            p[:, 0, 0] += self.translate[0]
            p[:, 0, 1] += self.translate[1]
        cv2.polylines(im, pset, isClosed=True, color=(255, 0, 255))
        cv2.imshow('Target', im)

    def loadrois(self):
        keys = []
        for f in os.listdir():
            if self.src in f and '_saved_roi_' in f:
                keys.append(f)
        if len(keys) > 1:
            fn = keys[PickFromList(keys).ret]
        elif len(keys) == 1:
            fn = keys[0]
        else:
            print('No source ROIs found')
            return -1
        polys = RoiEditor.load_roi(fn)
        self.data = polys
        ps = []
        for i, poly in enumerate(polys):
            pts = numpy.array(poly, dtype='int32')
            pts = pts.reshape(-1, 1, 2)
            ps.append(pts)
        return ps

    def getims(self):
        return [RoiEditor.get_pic(prefix) for prefix in (self.src, self.tgt)]


class Gui:
    __name__ = 'RoiEditorGUI'
    def __init__(self, path, prefix, exporting=False, preferred_tag='1'):
        self.path = path
        self.preferred_tag = preferred_tag
        self.prefix = prefix
        self.log = logger()
        self.log.set_handle(path, prefix)
        self.opPath = os.path.join(self.path, self.prefix + '/')
        self.psets = {}
        self.paths = {}
        self.incl = {}
        self.data = {}
        self.sizes = {}
        self.brightness = {}
        self.active_keys = []
        self.default_key = None
        # channels for 0 brightness, 1 kutrosis and 2 red:
        self.channels = [1, 0, 2]
        self.colors = {'active': (0, 255, 255), 'excluded': (255, 255, 0), 'saved': (255, 0, 255),
                       'inactive': (128, 128, 128)}
        self.addmode = 'contain'
        self.lut = numpy.arange(0, 256, dtype='uint8')
        self.pos = (256, 398)
        self.zoomsize = 512
        self.zoomfactor = 4
        self.current_key = None
        self.saved_rois = []
        self.saved_paths = []

        os.chdir(path)
        print(path, prefix)
        self.rois = RoiEditor(path, prefix)
        self.load_rois()

        avgfn = self.opPath + self.prefix + '_avgmax.tif'
        if os.path.exists(avgfn):
            preview_rgb = tifffile.imread(avgfn)
        else:
            preview_rgb = self.rois.get_pic()
        self.pic = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)

        self.currim = self.pic
        if len(self.active_keys) > 0:
            self.calc_sizes()
        else:
            self.empty_init()

        cv2.imshow('Rois', self.pic)
        cv2.createTrackbar('Min Size', 'Rois', self.lims[0], self.lims[1], self.slimChange)
        cv2.createTrackbar('Max Size', 'Rois', self.lims[1], self.lims[1], self.slimChange)
        # cv2.createTrackbar('Brightness', 'Rois', self.blims[0], self.blims[1], self.slimChange)
        # cv2.createTrackbar('Max Bright', 'Rois', self.blims[1], self.blims[1], self.slimChange)
        # cv2.createTrackbar('Kurtosis', 'Rois', self.klims[0], self.klims[1], self.slimChange)
        cv2.createTrackbar('Gamma', 'Rois', 10, 20, self.gammaChange)
        self.draw()
        cv2.moveWindow('Rois', 0, 0)
        cv2.setMouseCallback('Rois', self.mouse)

        cv2.namedWindow('Zoom')
        self.zoom_flag = True
        self.zoom()
        cv2.moveWindow('Zoom', 800, 0);

        cv2.namedWindow('Result')
        self.draw_result()
        cv2.moveWindow('Result', 800, 600);
        cv2.setMouseCallback('Result', self.resmouse)

    def empty_init(self):
        self.lims = 0, 100
        # self.blims = 0, 100
        # self.klims = 0, 100

    def loop(self, client=None):
        interactive = True
        while interactive:
            if cv2.getWindowProperty('Rois', 0) < 0:
                interactive = False
                cv2.destroyAllWindows()
            k = cv2.waitKey(25) & 0xFF
            if k == ord('l'):
                self.lasso_callback('add')
            elif k == ord('i'):
                self.lasso_callback('add-dil')
            elif k == ord('g'):
                self.lasso_callback('draw')
            elif k == ord('r'):
                self.lasso_callback('remove')
            elif k == ord('d'):
                self.dilate_callback()
            elif k == ord('e'):
                msg = self.save_callback()
                ShowMessage('Saved', msg=msg)
            elif k == ord('s'):
                if client == 'gui':
                    return True
                else:
                    self.save_callback()
                    self.save_sbx()
            elif k == ord('c'):
                self.clear_callback()
            elif k == ord('o'):
                self.options_callback()
            elif k == ord('q'):
                ret = PickFromList(['Exit without saving', 'Save ROI only', 'Save to sbx', 'Cancel']).ret
                if ret == 0:
                    interactive = False
                elif ret == 1:
                    self.save_callback()
                    interactive = False
                elif ret == 2:
                    self.save_callback()
                    self.save_sbx()
                    interactive = False
            elif k in [ord('-'), ord('+')]:
                i = self.active_keys.index(self.current_key)
                if k == ord('-'):
                    i -= 1
                else:
                    i += 1
                if not i < len(self.active_keys):
                    i = 0
                if i < 0:
                    i = len(self.active_keys) - 1
                self.current_key = self.active_keys[i]
                self.calc_sets(self.current_key)
                self.draw()
    #
    # def save_sbx(self):
    #     print('Exporting masks in background...')
    #     Process(target=save_sbx, args=(self.prefix, self.pic.shape, self.saved_rois, self.path)).start()

    def lasso_callback(self, mode):
        # add rois included in lasso
        ps = Lasso(self, mode).retval
        if ps is -1:
            return
        if 'add' in mode:
            rois = []
            for p in ps:
                p = mplpath.Path(p)
                for i, roi in enumerate(self.data[self.current_key]):
                    if self.incl[self.current_key][i]:
                        if p.contains_point(mplpath.Path(roi).vertices.mean(axis=0)):
                            if not i in rois:
                                rois.append(i)
            self.incl[self.current_key][:] = False
            self.incl[self.current_key][rois] = True
            self.add_current(dil='dil' in mode)
        if mode == 'draw':
            for p in ps:
                if len(p) > 3:
                    self.saved_rois.append(p)
                    self.saved_paths.append(mplpath.Path(p))
        if mode == 'remove':
            rois = []
            for p in ps:
                p = mplpath.Path(p)
                for i, roi in enumerate(self.saved_paths):
                    if p.contains_point(roi.vertices.mean(axis=0)):
                        if i not in rois:
                            rois.append(i)
            r, p = [], []
            for i in range(len(self.saved_rois)):
                if i not in rois:
                    r.append(self.saved_rois[i])
                    p.append(self.saved_paths[i])
            self.saved_rois = r
            self.saved_paths = p
        self.draw_result()

    def save_callback(self):
        return self.save()

    def options_callback(self):
        if self.addmode == 'contain':
            self.addmode = 'overlap'
        elif self.addmode == 'overlap':
            self.addmode = 'preserve'
        elif self.addmode == 'preserve':
            self.addmode = 'contain'
        self.draw()

    def clear_callback(self):
        self.saved_rois = []
        self.saved_paths = []
        self.draw_result()

    def close_callback(self):
        cv2.destroyAllWindows()

    def dilate_callback(self):
        nrs = []
        for roi in self.saved_rois:
            nrs.append(Polygon(roi).buffer(1).exterior.coords)
        self.saved_rois = []
        self.saved_paths = []
        for roi in nrs:
            self.saved_rois.append(roi)
            self.saved_paths.append(mplpath.Path(roi))
        self.draw_result()

    def save(self):
        if len(self.saved_rois) < 1:
            print('0 rois, roi file not saved for', self.prefix)
            return -1
        # code moved to separate function, call that to save after determining file name
        # determine number for next file
        fn = self.prefix + '_saved_roi_' + self.preferred_tag
        if os.path.exists(self.opPath + fn + '.npy'):
            exs = [0]
            for f in os.listdir(self.opPath):
                if self.prefix in f and '_saved_roi_' in f:
                    try:
                        # ugly way to keep autodetected rois from breaking numbering
                        exs.append(int(f[:-4].split('_')[-1]))
                    except:
                        pass
            exi = max(exs)+1
            while os.path.exists(self.opPath + self.prefix + '_saved_roi_' + str(exi) + '.npy'):
                exi += 1
            fn = self.prefix + '_saved_roi_' + str(exi)
        RoiEditor.save_roi(self.saved_rois, self.opPath + fn, self.rois.img.image.info['sz'])
        msg = f'{len(self.saved_rois)} saved in {fn}'
        lprint(self, msg, logger=self.log)
        return msg

    def gammaChange(self, v):
        gamma = 10.0 / (cv2.getTrackbarPos('Gamma', 'Rois') + 1)
        self.lut = numpy.array([((i / 255.0) ** gamma) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
        self.draw()

    def slimChange(self, v):
        self.lims = [cv2.getTrackbarPos('Min Size', 'Rois'), cv2.getTrackbarPos('Max Size', 'Rois')]
        # self.blims = [cv2.getTrackbarPos('Brightness', 'Rois'), cv2.getTrackbarPos('Max Bright', 'Rois')]
        # self.klims[0] = cv2.getTrackbarPos('Kurtosis', 'Rois')
        for i, s in enumerate(self.sizes[self.current_key]):
            sincl = self.lims[0] < s < self.lims[1]
            # bincl = self.blims[0] < self.brightness[self.current_key][i, self.channels[0]] < self.blims[1]
            # kincl = self.klims[0] < self.brightness[self.current_key][i, self.channels[1]]
            self.incl[self.current_key][i] = sincl  # and bincl #and kincl
        self.calc_sets(self.current_key)
        self.draw()

    def mouse(self, event, y, x, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK or flags == 17:
            # self.zoom_flag = not self.zoom_flag
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            self.pick(y, x, 'rois')
        elif event == cv2.EVENT_RBUTTONUP:
            self.add_current()
            self.draw_result()
        if event == cv2.EVENT_MOUSEMOVE and self.zoom_flag:
            self.pos = (x, y)
            self.zoom('rois')

        if event == cv2.EVENT_MOUSEWHEEL:
            # cycle active key with wheel
            i = self.active_keys.index(self.current_key)
            if flags > 0:
                i -= 1
            else:
                i += 1
            # i = max(0, min(i, len(self.active_keys)-1))
            if not i < len(self.active_keys):
                i = 0
            if i < 0:
                i = len(self.active_keys) - 1
            self.current_key = self.active_keys[i]
            self.calc_sets(self.current_key)
            self.draw()

    def resmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK or flags == 17:
            #     self.zoom_flag = not self.zoom_flag
            # elif event == cv2.EVENT_LBUTTONDOWN:
            ps = Newroi(self, x, y).retval
            if not ps is -1:
                for p in ps:
                    self.saved_rois.append(p)
                    self.saved_paths.append(mplpath.Path(p))
                self.draw_result()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pick(x, y, 'remove')

        if event == cv2.EVENT_MOUSEMOVE and self.zoom_flag:
            self.pos = (y, x)
            self.zoom('res')

    def pick(self, x, y, w):
        found = False
        if w == 'rois':
            for key in self.active_keys:
                for i in range(len(self.data[self.current_key])):
                    if self.paths[self.current_key][i].contains_point([x, y]):
                        p = self.data[self.current_key][i]
                        if p not in self.saved_rois:
                            self.saved_rois.append(p)
                            self.saved_paths.append(mplpath.Path(p))
                            self.draw_result()
                            if not self.incl[self.current_key][i]:
                                self.incl[self.current_key][i] = True
                                self.calc_sets(self.current_key)
                                self.draw()
                            found = True
                        break
                if found:
                    break
        elif w == 'remove':
            for i in range(len(self.saved_rois)):
                if self.saved_paths[i].contains_point([x, y]):
                    self.saved_rois.pop(i)
                    self.saved_paths.pop(i)
                    self.draw_result()
                    found = True
                    break

    def zoom(self, src=None, ret=False):
        # find start positions
        size = int(self.zoomsize / self.zoomfactor)
        start = [0, 0]
        for i in [0, 1]:
            start[i] = int(min(self.pic.shape[i] - size, self.pos[i] - size / 2))
            start[i] = max(start[i], 0)
        x, y = start
        if src == 'res':
            im = self.resim[x:x + size, y:y + size, :]
        else:
            im = self.currim[x:x + size, y:y + size, :]
        im = cv2.resize(im, (0, 0), fx=self.zoomfactor, fy=self.zoomfactor)
        if ret:
            return numpy.array(im), start
        m = [0, 0]
        for i in [0, 1]:
            m[i] = int((self.pos[i] - start[i]) * self.zoomfactor)
        cv2.line(im, (m[1], 0), (m[1], self.zoomsize), color=(255, 255, 255))
        cv2.line(im, (0, m[0]), (self.zoomsize, m[0]), color=(255, 255, 255))
        cv2.imshow('Zoom', im)

    def add_current(self, dil=False):
        # print(f'Add mode: {self.addmode}')
        for i, incl in enumerate(self.incl[self.current_key]):
            if incl:
                p = self.data[self.current_key][i]
                if dil:
                    p = Polygon(p).buffer(3).exterior.coords
                if (not p in self.saved_rois) or self.addmode == 'preserve':
                    # check if roi is inside existing roi
                    tp = mplpath.Path(p)
                    cm = tp.vertices.mean(axis=0)
                    point1 = Point(cm)
                    skip = False
                    if self.addmode == 'contain':
                        for ep in self.saved_paths:
                            if ep.contains_point(cm):
                                skip = True
                                break
                    elif self.addmode == 'overlap':
                        p1 = Polygon(p)
                        for oi, ep in enumerate(self.saved_paths):
                            if ep.contains_point(cm):
                                skip = True
                                break
                            else:
                                if point1.distance(Point(ep.vertices.mean(axis=0))) < 50:
                                    if p1.intersects(Polygon(self.saved_rois[oi])):
                                        skip = True
                                        break
                    if not skip:
                        self.saved_rois.append(p)
                        self.saved_paths.append(tp)

    def draw_result(self):
        im = numpy.array(self.pic)
        if len(self.saved_rois) > 0:
            ps = []
            for poly in self.saved_rois:
                pts = numpy.array(poly, dtype='int32')
                pts = pts.reshape(-1, 1, 2)
                ps.append(pts)
            cv2.polylines(im, ps, isClosed=True, color=self.colors['saved'])
        self.resim = numpy.array(im)
        cv2.putText(im, str(len(self.saved_rois)), (0, 25),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255, 255, 255))
        cv2.imshow('Result', im)

    def draw(self):
        im = numpy.array(self.pic)
        keylist = []
        for key in self.active_keys:
            if not key == self.current_key:
                keylist.append(key)
        keylist.append(self.current_key)
        for key in keylist:
            if key is None:
                continue
            if key == self.current_key:
                color = self.colors['active']
                ecolor = self.colors['excluded']
            else:
                color = self.colors['inactive']
                ecolor = self.colors['inactive']
            if key == self.current_key:
                if len(self.psets[key + '_e']) > 0:
                    cv2.polylines(im, self.psets[key + '_e'], isClosed=True, color=ecolor)
                cv2.polylines(im, self.psets[key], isClosed=True, color=color)
        self.currim = cv2.LUT(im, self.lut)
        self.zoom()
        im = numpy.array(self.currim)
        #fill with black the left and bottom of the pic, if it's too small for the GUI.
        # this is done after polys are drawn, and does not effect the image data, just the drawing of the 'Rois' window
        # the zoom will work with the new image coordinates though (top left corner)
        min_h = 200
        min_w = 500
        if im.shape[1] < min_w:
            old_im = numpy.copy(im)
            im = numpy.zeros((im.shape[0], min_w, im.shape[2]), im.dtype)
            im[:, min_w-old_im.shape[1]:, :] = old_im
        if im.shape[0] < min_h:
            old_im = numpy.copy(im)
            im = numpy.zeros((min_h, im.shape[1], im.shape[2]), im.dtype)
            im[:old_im.shape[0],:, :] = old_im
        if len(self.active_keys) > 0:
            cv2.putText(im, str(len(self.psets[self.current_key])) + ' of', (0, 25),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255, 255, 255))
            cv2.putText(im, str(len(self.data[self.current_key])), (0, 50),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255, 255, 255))
            cv2.putText(im, self.addmode, (0, 75),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255, 255, 255))
        for i, key in enumerate(self.active_keys):
            if key == self.current_key:
                color = self.colors['active']
                fsc = 0.8
            else:
                color = (255, 255, 255)
                fsc = 0.5
            cv2.putText(im, key, (0, 25 * (i + 4)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fsc, color=color)
        cv2.imshow('Rois', im)
        cv2.resizeWindow('Rois', max(min_w, im.shape[1]), max(min_h, im.shape[0]))

    def calc_sets(self, key):
        ps, es = [], []
        for i, poly in enumerate(self.data[key]):
            pts = numpy.array(poly, dtype='int32')
            pts = pts.reshape(-1, 1, 2)
            if self.incl[key][i]:
                ps.append(pts)
            else:
                es.append(pts)
        self.psets[key] = ps
        self.psets[key + '_e'] = es

    def calc_sizes(self):
        sizes, bright, kurtos = [], [], []
        print(self.active_keys)
        for key in self.active_keys:
            if len(self.data[key]) < 1:
                self.active_keys.pop(self.active_keys.index(key))
                continue
            self.sizes[key] = []
            self.brightness[key] = numpy.zeros((len(self.data[key]), 3))
            self.paths[key] = []
            for ip, coords in enumerate(self.data[key]):
                poly = RoiEditor.trim_coords(numpy.array(coords), self.rois.img.image.info['sz'])
                p = Polygon(poly)
                mp = mplpath.Path(poly)
                self.paths[key].append(mp)
                # pull intensities from mask
                x0, y0, x1, y1 = [int(i) for i in p.bounds]
                b = numpy.zeros(3)
                n = 0
                for x in range(x0, x1 + 1):
                    for y in range(y0, y1 + 1):
                        if mp.contains_point([x, y]):
                            b += self.pic[y, x, :]
                            n += 1
                if n > 0:
                    self.brightness[key][ip, :] = b / n
                else:
                    self.brightness[key][ip, :] = b
                if numpy.any(numpy.isnan(b)):
                    print(ip, poly, y, x)
                self.sizes[key].append(numpy.sqrt(p.area))
            sizes.extend(self.sizes[key])
            bright.extend(self.brightness[key][:, self.channels[0]])
            kurtos.extend(self.brightness[key][:, self.channels[1]])
        self.lims = [int(min(sizes)), int(max(sizes) + 1)]
        # self.blims = [int(min(bright)), int(max(bright) + 1)]
        # self.klims = [int(min(kurtos)), int(max(kurtos) + 1)]


    def load_rois(self):

        # load previous saves
        exs = []
        for f in os.listdir(self.opPath):
            if self.prefix in f and '_saved_roi_' in f:
                if os.path.getsize(self.opPath + f) > 128:
                    exs.append(f)
        exs.sort()
        for nf, f in enumerate(exs):
            polys = RoiEditor.load_roi(self.opPath + f)
            if len(polys) > 0:
                key = f[:-4].split('_')[-1]
                try:
                    int(key)
                    self.default_key = key
                except ValueError:
                    pass
                self.current_key = key
                self.colors[key] = (int(255.0 / len(exs) * (nf + 1)), int(255 - 255.0 / len(exs) * nf), 0)
                self.data[key] = polys
                self.incl[key] = numpy.ones(len(polys), dtype='bool')
                self.calc_sets(key)
                self.active_keys.append(key)
        if self.default_key is None:
            if 'STICA' in self.active_keys:
                self.default_key = 'STICA'
            elif 'PC' in self.active_keys:
                self.default_key = 'PC'


class RoiEditor(object):
    __name__ = 'RoiEditor'
    def __init__(self, procpath, prefix):
        self.img = FrameDisplay(procpath, prefix, tag='skip')
        self.procpath = procpath
        self.prefix = prefix
        self.opPath = os.path.join(procpath, prefix + '/')

    @staticmethod
    def save_roi(roi_list, fn, image_shape, translate=(0, 0)):
        rois_chunk = 1000
        rois = numpy.empty((rois_chunk, 3), dtype='int32')
        rois[:] = numpy.nan
        roi_counter = 0
        junk_roi = numpy.ones((3, 3))
        translate = numpy.array(translate)
        # convert list of polygons into arrays with each line having and index for polygon ID
        for i, poly in enumerate(roi_list):
            new_array = numpy.empty((len(poly), 3))
            new_array[:, 0] = i
            if len(poly) < 3:
                new_array = junk_roi * [i, 0, 0]
            else:
                for j, p in enumerate(poly):
                    new_array[j, 1:] = p + translate
                # check if new array is outside image region:
                assert not numpy.any(numpy.isnan(new_array))
                coords = numpy.copy(new_array[:, 1:])
                if image_shape is not None:
                    coords = RoiEditor.trim_coords(coords, image_shape)
            # append the new array to container
            new_length = roi_counter + len(new_array)
            if new_length > len(rois):
                rois = numpy.append(rois, numpy.empty((rois_chunk, rois.shape[1])), axis=0)
            rois[roi_counter:new_length, :] = new_array
            roi_counter += len(new_array)
        if not fn.endswith('.npy'):
            fn += '.npy'
        numpy.save(fn, rois[:roi_counter])

    @staticmethod
    def load_roi(fn):
        polys = []
        data = numpy.load(fn)
        n = 0
        poly = []
        for j, x, y in data:
            if n == j:
                poly.append([x, y])
            else:
                polys.append(poly)
                n += 1
                poly = [[x, y]]
        polys.append(poly)
        return polys

    def get_pic(self):
        return self.img.image.show_field()

    @staticmethod
    def trim_coords(old_coords, image_shape):
        coords = numpy.maximum(1, old_coords)
        coords[:, 0] = numpy.minimum(coords[:, 0], image_shape[1] - 1)
        coords[:, 1] = numpy.minimum(coords[:, 1], image_shape[0] - 1)
        return coords

    def autodetect(self, chunk_n=100, chunk_size=50, approach=('iPC', 'PC', 'IN'), config={}, exclude_opto=True, log=None):
        approach = list(approach)
        prefix = self.prefix
        im = self.img.image
        n_channels = len(im.channels)
        maxint = self.img.image.imdat.bitdepth - 1
        force_re = False
        opto_name = self.opPath + prefix + '_bad_frames.npy'
        if exclude_opto and os.path.exists(opto_name):
            have_opto = True
            bad_frames = numpy.load(opto_name)
            lprint(self, f'opto excluding {len(bad_frames)} frames', logger=log)

        else:
            # lprint(self, 'no opto')
            have_opto = False
        # find trim:
        # x0, x1, y0, y1 = 10, 15, 10, 10
        x0, x1, y0, y1 = 1, 1, 1, 1#not needed for bruker
        # create 50 frames which are each an average of 100, in random order
        dsname = self.opPath + prefix + '_avgs' + '.sima'
        idsname = self.opPath + prefix + '_iavgs' + '.sima'
        if 'Start' in config:
            force_re = True
            try:
                s1 = int(config['Start'])
            except:
                s1 = 0
                print(prefix, 'Start defaulting to 0, got', config['Start'])
            try:
                s2 = int(config['Stop'])
            except:
                s2 = im.nframes
            span = s1, s2
        else:
            span = 0, im.nframes
        imlen = span[1] - span[0]
        if force_re:
            for n in [dsname, idsname]:
                if os.path.exists(n):
                    shutil.rmtree(n, ignore_errors=True)
        if all([os.path.exists(dsname), os.path.exists(idsname), not force_re]):
            ds = sima.ImagingDataset.load(dsname)
            ids = sima.ImagingDataset.load(idsname)
        else:
            chunk_n = min(chunk_n, int(imlen / chunk_size) - 2)
            lprint(self, 'Detecting', prefix, span, ', ', chunk_n, 'chunks')
            imshape = (*im.imdat.data.shape, n_channels)
            d = numpy.empty((chunk_n, imshape[1] - y1 - y0, imshape[2] - x1 - x0, n_channels),
                            dtype=im.imdat.data.dtype)
            chlist = numpy.sort(numpy.random.choice(int(imlen / chunk_size) - 2, chunk_n, replace=False))
            for chunk, frame in enumerate(chlist):
                # make ds and trim edges
                # (16453, 512, 796, 1)
                chunkslice = slice(span[0] + frame * chunk_size, span[0] + (frame + 1) * chunk_size)
                # select frames. exclude opto frames if any.
                if have_opto:
                    chunkslice = [i for i in range(chunkslice.start, chunkslice.stop) if i not in bad_frames]
                # include all channels
                for chi, chn in enumerate(im.channels):
                    d[chunk, ..., chi] = im.imdat.get_channel(chn)[chunkslice, y0:-y1, x0:-x1].mean(axis=0)

            # put noise on saturated pixels to prevent CA1PC from crashing
            saturated = numpy.where(d == maxint)
            d[saturated] = maxint - numpy.random.random(saturated[0].size) * 16
            # save a picture of the averaged stuff in mIP
            nicepic = numpy.zeros((imshape[1], imshape[2], 3), dtype='uint8')
            lut = [2, 1, 0]
            for ch in range(n_channels):
                id = numpy.amax(d[..., ch], axis=0).astype('float')
                id -= id.min()
                id /= numpy.percentile(id, 99)
                nicepic[y0:-y1, x0:-x1, lut[ch]] = numpy.minimum(id, 1) * 255
            # add kurtosis image to blue channel
            if n_channels == 1:
                id = scipy.stats.kurtosis(d[..., 0], axis=0)
                id -= id.min()
                id /= numpy.percentile(id, 99)
                nicepic[y0:-y1, x0:-x1, lut[2]] = numpy.minimum(id, 1) * 255
            cv2.imwrite(self.opPath + prefix + '_avgmax.tif', nicepic)
            lprint(self, prefix, 'avgmax.tif saved', logger=log)

            # create SIMA object
            if os.path.exists(dsname) and not force_re:
                ds = sima.ImagingDataset.load(dsname)
            else:
                # (num_frames, num_planes, num_rows, num_columns, num_channels)
                ds = sima.ImagingDataset([sima.Sequence.create('ndarray', numpy.expand_dims(maxint - d, axis=1))],
                                         dsname)
            if os.path.exists(idsname) and not force_re:
                ids = sima.ImagingDataset.load(idsname)
            else:
                ids = sima.ImagingDataset([sima.Sequence.create('ndarray', numpy.expand_dims(d, axis=1))],
                                          idsname)
        # run segmentation
        for item in approach:
            # find index of channel
            ch = 0
            if item.endswith('-R'):
                if 'Ch1' in im.channels:
                    ch = im.channels.index('Ch1')
                else:
                    ch = 1
            if item.endswith('-G') and 'Ch2' in im.channels:
                ch = im.channels.index('Ch2')
            # default params if not specified
            pnames = ('Diameter', 'MinSize', 'MaxSize')
            pdefs = (10, 40, 200)
            itemtag = item
            all_def = True
            for pname, pval in zip(pnames, pdefs):
                if not pname in config:
                    config[pname] = pval
                if not config[pname] == pval:
                    all_def = False
            if 'PC' in item:
                if not all_def:
                    itemtag = item + '-'.join([str(config[pname]) for pname in pnames])
            lprint(self, item, 'Size setting: diam', config['Diameter'], 'min:', config['MinSize'], 'max:', config['MaxSize'], logger=log)

            py_approach = sima.segment.PlaneCA1PC(channel=ch, verbose=False,
                                                  cut_min_size=int(config['MinSize']),
                                                  cut_max_size=int(config['MaxSize']),
                                                  x_diameter=int(config['Diameter']),
                                                  y_diameter=int(config['Diameter']), )
            if 'PC' in item and 'iPC' not in item:
                try:
                    rois = ds.segment(py_approach, 'pc_ROIs')
                    self.saveauto(rois, itemtag, x0, y0)
                except AssertionError:
                    lprint(self, 'CA1PC crashed for', prefix, logger=log)
            if 'iPC' in item:
                try:
                    rois = ids.segment(py_approach, 'ipc_ROIs')
                    self.saveauto(rois, itemtag, x0, y0)
                except AssertionError:
                    lprint(self, 'CA1PC-i crashed for', prefix, logger=log)
            if 'IN' in item or 'STICA' in item:
                # default params if not specified
                if not 'comps' in config:
                    config['comps'] = 75
                if not 'mu' in config:
                    config['mu'] = 0.1
                stica_approach = sima.segment.STICA(channel=ch, components=int(config['comps']), mu=float(config['mu']),
                                                    verbose=False)
                stica_approach.append(sima.segment.SparseROIsFromMasks())
                stica_approach.append(sima.segment.SmoothROIBoundaries())
                stica_approach.append(sima.segment.MergeOverlapping(threshold=0.5))
                rois = ds.segment(stica_approach, 'stica_ROIs')

                retval = self.saveauto(rois, item, x0, y0, filter=50 * 50)
                lprint(self, retval, logger=log)

        # clean up sima folders
        ds, ids = None, None
        for fn in (dsname, idsname):
            try:
                shutil.rmtree(fn + '//')
            except:
                pass

    def saveauto(self, input_rois, tag, x0, y0, filter=None):
        fn = self.opPath + self.prefix + '_saved_roi_' + tag
        nrs = []
        for poly in input_rois:
            roi = []
            for p in poly.coords[0]:
                roi.append([p[0] + x0, p[1] + y0])
            if filter is None:
                nrs.append(roi)
            elif Polygon(roi).area < filter:
                nrs.append(roi)
        RoiEditor.save_roi(nrs, fn, self.img.image.info['sz'])
        return f'{len(nrs)} ROIs saved in {fn}'

    def load_roiset(self):
        archive = zipfile.ZipFile(self.prefix + '_modrois.zip', 'r')
        rois = []
        for fname in archive.filelist:
            archive.extract(fname)
            with open(fname.filename, 'rb') as f:
                d = numpy.fromfile(f, dtype='>h')
            os.remove(fname.filename)
            x0, y0 = d[5], d[4]
            nc = d[8]
            coords = numpy.empty((2, nc), dtype='int')
            for i in range(nc):
                coords[:, i] = [d[32 + i] + x0, d[32 + nc + i] + y0]
            rois.append(coords)
        self.ijrois = rois

    def create_previews(self):
        return self.img.polys.create_previews()


if __name__ == '__main__':
    # test single
    # wdir = 'D:/Shares/Data/_Processed/2P/testing/'
    # prefix = 'SncgTot4_2023-11-09_LFP_002'
    # r = RoiEditor(wdir, prefix, )
    # approach = ('STICA-G', )
    # # r.autodetect(approach=approach)
    #
    # self = r
    # chunk_n = 100
    # chunk_size = 50
    # ('D:/Shares/Data/_Processed/2P/testing//', 'SncgTot4_2023-11-09_LFP_002', ['PC-G', 'STICA-G'],
    #  {'Start': '0', 'Stop': 'end', 'Diameter': '20', 'MinSize': '100', 'MaxSize': '800'})


    wdir = 'D:\Shares\Data\_Processed/2P\JEDI-IPSP/'
    prefix = 'JEDI-Sncg65_2024-12-10_lfp_opto_124'
    r = RoiEditor(wdir, prefix, )

    preview_rgb = r.get_pic()

    # pic = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
