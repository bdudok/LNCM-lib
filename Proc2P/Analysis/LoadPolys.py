import matplotlib.path as mplpath
import matplotlib.pyplot as plt
import numpy
import scipy
import cv2
import os
import copy, math
import tifffile

from Proc2P.Bruker.LoadRegistered import LoadRegistered, Source
from Proc2P.utils import lprint


class LoadPolys(object):
    __name__ = 'LoadPolys'
    def __init__(self, procpath, prefix, tag=None):
        self.opPath = os.path.join(procpath, prefix + '/')
        self.prefix = prefix
        if '+' in tag:
            tags = tag.split('+')
            sets = []
            for tag in tags:
                roi_name = os.path.join(self.opPath, f'{prefix}_saved_roi_{tag}.npy')
                sets.append(load_roi_file(roi_name))
            polys = []
            for s in sets:
                for p in s:
                    polys.append(p)
            self.data = numpy.array([numpy.array(xi) for xi in polys])
        else:
            roi_name = roi_name = os.path.join(self.opPath, f'{prefix}_saved_roi_{tag}.npy')
            self.data = load_roi_file(roi_name)
        lprint(self, f'Polys loaded from {roi_name}')

    def __getitem__(self, item):
        return self.data[item]

    def bounds(self, c):
        maxes = self.data[c].max(axis=0).astype('int')
        mins = self.data[c].min(axis=0).astype('int')
        return mins[1], maxes[1], mins[0], maxes[0]

    def area(self, c):
        def EucDistance2D(point1, point2):
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        def TriangleArea2D(point1, point2, point3):
            a = EucDistance2D(point1, point2)
            b = EucDistance2D(point2, point3)
            c = EucDistance2D(point3, point1)
            s = (a + b + c) / 2
            return math.sqrt(s * (s - a) * (s - b) * (s - c))

        hull2D = scipy.spatial.ConvexHull(self.data[c])
        area = 0.0
        for k in range(len(hull2D.vertices) - 2):
            area += TriangleArea2D(hull2D.points[hull2D.vertices[-1]],
                                   hull2D.points[hull2D.vertices[k]],
                                   hull2D.points[hull2D.vertices[k + 1]])
        return area


class LoadImage(object):
    def __init__(self, procpath, prefix, source=Source.S2P):
        self.prefix = prefix
        self.opPath = procpath + prefix + '/'
        self.data_loaded = False
        self.imdat = LoadRegistered(procpath, prefix, source)
        self.info = {'sz': (self.imdat.Ly, self.imdat.Lx)}
        self.nframes = self.imdat.n_frames
        self.nplanes = 1
        if not len(self.imdat.channel_keys):
            self.imdat.find_alt_path()
        self.channels = self.imdat.channel_keys

    def get_frame(self, frame, ch=None, zplane=0):
        if self.nplanes == 1:
            return self.imdat.get_channel(ch)[frame, :, :, ]
        # else:
        #     return self.data[zplane][frame, :, :, ch]

    def show_field(self):
        preview_fn = self.opPath + self.prefix + '_preview.tif'
        pic = tifffile.imread(preview_fn)
        if len(pic.shape) < 3:
            rgb_array = numpy.zeros((*pic.shape, 3), dtype='uint8')
            rgb_array[..., 1] = 255 * pic / pic.max()
        else:
            rgb_array = pic
        return rgb_array


class FrameDisplay(object):
    # future: instead of passing these paths, use the session info dict instead

    def __init__(self, procpath, prefix, raw=False, force=None, tag=None, lazy=False):
        if tag not in ('skip', 'off', 'dummy'):
            self.polys = LoadPolys(procpath, prefix, tag)
        self.procpath = procpath
        self.prefix = prefix
        self.image_loaded = False
        if not lazy:
            self.load_image()
        self.preview = None

    def load_image(self):
        if not self.image_loaded:
            self.image = LoadImage(self.procpath, self.prefix, )
            self.image_loaded = True

    def get_preview(self, ch=1):
        if self.preview is None:
            preview_fn = self.procpath + self.prefix + '/' + self.prefix + '_preview.tif'
            print(preview_fn)
            self.preview = tifffile.imread(preview_fn)
        return self.preview

    def get_cell(self, frame, cell, channel=0):
        x1, x2, y1, y2 = self.polys.bounds(cell)
        return self.image.data[frame, x1 - 1:x2 + 1, y1 - 1:y2 + 1]  # , channel]

    def calc_mask(self):
        height, width = self.image.info['sz']
        mask = numpy.zeros((height, width), dtype=numpy.uint8)
        polys = []
        masks = []
        for i, roi in enumerate(self.polys.data):
            c_mask = numpy.zeros((height, width), dtype=numpy.uint8)
            left, top = roi.min(axis=0).astype('int')
            right, bottom = roi.max(axis=0).astype('int')
            polys.append(mplpath.Path(roi))
            for x in range(left, right):
                for y in range(top, bottom):
                    if polys[i].contains_point([x, y]):
                        mask[y, x] = i + 1
                        c_mask[y, x] = 1
            masks.append(c_mask)
        self.c_mask = masks
        # create np_mask
        np_mask = []
        r = 50
        for i, roi in enumerate(self.polys.data):
            left, top = roi.min(axis=0).astype('int')
            right, bottom = roi.max(axis=0).astype('int')
            nmask = numpy.zeros((height, width), dtype=numpy.uint8)
            tree = scipy.spatial.cKDTree(polys[i].vertices)
            for x in range(max(0, left - r), min(width, right + r)):
                for y in range(max(0, top - r), min(height, bottom + r)):
                    if not mask[y, x]:
                        if len(tree.query_ball_point([x, y], r)):
                            nmask[y, x] = 1
            np_mask.append(nmask)
        self.halo_mask = np_mask

    def show_cell(self, cell, itn=100, channel=0):
        x1, x2, y1, y2 = self.polys.bounds(cell)
        if self.preview is None:
            self.preview = self.image.create_previews(ret=True)
        pic = self.preview[x1 - 1:x2 + 1, y1 - 1:y2 + 1]
        return pic

def load_roi_file(roi_name):
    data = numpy.load(roi_name)
    polys = []
    poly = []
    n = 0
    for j, x, y in data:
        if n == j:
            poly.append([x, y])
        else:
            if len(poly) > 2:
                polys.append(poly)
            n += 1
            poly = []
    if len(poly) > 2:
        polys.append(poly)
    return [numpy.array(xi) for xi in polys]


if __name__ == '__main__':
    procpath = 'D:\Shares\Data\_Processed/2P\PVTot\Opto/'
    prefix = 'PVTot5_2023-09-04_opto_023'
    img = FrameDisplay(procpath, prefix, tag='skip')




