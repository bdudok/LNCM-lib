
from Proc2P.Analysis.LoadPolys import FrameDisplay
from Proc2P.Bruker.LoadRegistered import check_compatible_ops
from Proc2P.utils import lprint, logger
import os
import shutil
from pathlib import Path
import scipy
import cv2
import numpy
from shapely.geometry import Polygon, Point
import sys

if sys.version_info < (3, 7):
    import sima

    sima_available = True
else:
    sima_available = False
    from Proc2P.Analysis.Autodetect_wrapper import roi_detector
    from dataclasses import dataclass


    @dataclass
    class SIMAConfig:
        # fields to be configurable from GUI
        iPC_G: bool = True
        PC_G: bool = True
        STICA_G: bool = True
        iPC_R: bool = True
        PC_R: bool = True
        STICA_R: bool = True
        Start: int = 0  # first frame to include.
        Stop: str | int = 'end'  # last frame to include (or anything that doesn't convert to an int to use all frames)
        Diameter: float = 20  # args passed to SIMA
        MinSize: float = 100  # args passed to SIMA
        MaxSize: float = 200  # args passed to SIMA

from multiprocessing import Process


'''
The RoiEditor class in here is used by the Qt BatchGUI and ROI editor.
'''

class Worker(Process):
    __name__ = 'RoiWorker'

    def __init__(self, queue, res_queue=None, n=0):
        super(Worker, self).__init__()
        self.queue = queue
        self.res_queue = res_queue
        self.n = n

    def run(self):
        for din in iter(self.queue.get, None):
            # the whole thing goes in a try so worker is not dead on error. remove this for debugging.
            try:
                path, prefix, apps, config = din
                if sima_available:
                    # if we're on Python 3.6 and sima installed, call the detector directly
                    log = logger()
                    log.set_handle(path, prefix)
                    lprint(self, 'Calling autodetect with:', din, logger=log)
                    RoiEditor(path, prefix, ).autodetect(approach=apps, config=config, log=log)
                else:
                    # if we're on Python 3.11, call the detector through the 3.6 executable
                    check_compatible_ops(path, prefix)
                    lprint(self, roi_detector(path, prefix, apps, config))
                if self.res_queue is not None:
                    self.res_queue.put((self.__name__ + str(self.n), prefix))
            except Exception as e:
                lprint(self, 'Autodetect failed with error:', e, )


class RoiEditor(object):
    __name__ = 'RoiEditor'

    def __init__(self, procpath, prefix):
        self.procpath = Path(procpath)
        self.img = FrameDisplay(self.procpath, prefix, tag='skip')
        self.prefix = prefix
        self.opPath = self.procpath / prefix

    def find_rois(self):
        exs = []
        for f in os.listdir(self.opPath):
            if self.prefix in f and '_saved_roi_' in f:
                if os.path.getsize(self.opPath / f) > 128:
                    exs.append(f)
        exs.sort()
        return exs

    @staticmethod
    def save_roi(roi_list, fn, image_shape, translate=(0, 0)):
        rois_chunk = 10000
        rois = numpy.zeros((rois_chunk, 3), dtype='int32')
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
                if image_shape is not None:
                    coords = RoiEditor.trim_coords(numpy.copy(new_array[:, 1:]), image_shape)
                    new_array[:, 1:] = coords
            # append the new array to container
            new_length = roi_counter + len(new_array)
            if new_length > len(rois):
                rois = numpy.append(rois, numpy.empty((rois_chunk, rois.shape[1])), axis=0)
            rois[roi_counter:new_length, :] = new_array
            roi_counter += len(new_array)
        fn = Path(fn)
        if fn.suffix != '.npy':
            fn = fn.with_name(fn.name + '.npy')
        numpy.save(fn, rois[:roi_counter])

    @staticmethod
    def dilate_polygon(poly, buffer=1):
        big_poly = Polygon(poly).buffer(buffer)
        if big_poly.type == 'MultiPolygon':
            # weird shaped poly causes output to have holes. use chull instead
            big_poly = Polygon(poly).convex_hull.exterior.buffer(buffer)
        return big_poly.exterior.coords

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

    def autodetect(self, chunk_n=100, chunk_size=50, approach=('iPC', 'PC', 'IN'), config={}, exclude_opto=True,
                   log=None):
        if not sima_available:
            print('For using segmentation, run using 3.6 env (2pcode36)')
            raise EnvironmentError('SIMA not compatible with current Python version.')
        approach = list(approach)
        prefix = self.prefix
        im = self.img.image
        n_channels = len(im.channels)
        maxint = self.img.image.imdat.bitdepth - 1
        force_re = False
        opto_name = self.opPath / f'{prefix}_bad_frames.npy'
        if exclude_opto and os.path.exists(opto_name):
            have_opto = True
            bad_frames = numpy.load(opto_name)
            lprint(self, f'opto excluding {len(bad_frames)} frames', logger=log)

        else:
            # lprint(self, 'no opto')
            have_opto = False
        # find trim:
        # x0, x1, y0, y1 = 10, 15, 10, 10
        x0, x1, y0, y1 = 1, 1, 1, 1  # not needed for bruker
        # create 50 frames which are each an average of 100, in random order
        dsname = str(self.opPath / f'{prefix}_avgs.sima')
        idsname = str(self.opPath / f'{prefix}_iavgs.sima')
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
            cv2.imwrite(str(self.opPath / f'{prefix}_avgmax.tif'), nicepic)
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
            lprint(self, item, 'Size setting: diam', config['Diameter'], 'min:', config['MinSize'], 'max:',
                   config['MaxSize'], logger=log)

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
                shutil.rmtree(fn)
            except:
                pass

    def saveauto(self, input_rois, tag, x0, y0, filter=None):
        fn = self.opPath / f'{self.prefix}_saved_roi_{tag}'
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


    def create_previews(self):
        return self.img.polys.create_previews()


    # preview_rgb = r.get_pic()

    # pic = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
