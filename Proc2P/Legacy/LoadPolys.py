import matplotlib.path as mplpath
from .sbxreader import *
import numpy
import scipy
import cv2
import copy, math
from tifffile import imwrite, TiffFile
import h5py

class LoadPolys(object):
    def __init__(self, prefix, tag=None):
        if tag is None:
            rfn = prefix + '_nonrigid.segment'
            nrfn = prefix + '_rigid.segment'
            for f in (nrfn, rfn):
                if os.path.exists(f):
                    self.data = loadmat(f)['vert']
                    print(f'Polys loaded from {f}')
                    break
        else:
            if '+' in tag:
                tags = tag.split('+')
                sets = []
                for tag in tags:
                    roi_name = f'{prefix}_saved_roi_{tag}.npy'
                    print(f'Polys loaded from {roi_name}')
                    sets.append(load_roi_file(roi_name))
                polys = []
                for s in sets:
                    for p in s:
                        polys.append(p)
                self.data = numpy.array([numpy.array(xi) for xi in polys])
            else:
                roi_name = f'{prefix}_saved_roi_{tag}.npy'
                print(f'Polys loaded from {roi_name}')
                self.data = load_roi_file(roi_name)

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
    def __init__(self, prefix, raw=False, force=None, matpath='', hdf_source=False, explicit_need_data=False, ):
        self.prefix = prefix
        self.data_loaded = False
        self.found = None
        self.hdf_source = False
        self.tif_source = False
        self.data_field_key = 'data'
        self.is_red = False
        self.sbx_version = 0
        matname = None
        # find input file
        if force is not None:
            file_in = force
            if os.path.exists(file_in):
                self.found = force
        else:
            file_in = prefix + '.sbx'
            if os.path.exists(file_in):
                self.found = 'raw'
            if not raw:
                rgdn = prefix + '_nonrigid.sbx'  # case: sbx, corrected
                if os.path.exists(rgdn):
                    file_in = rgdn
                    self.found = 'nonrigid'
                    if prefix[-3:-1] == '_z':
                        matname = prefix[:-3] + '.mat'
                else:
                    rgdn = prefix + '_rigid.sbx'  # case: sbx corrected with scanbox
                    if os.path.exists(rgdn):
                        file_in = rgdn
                        self.found = 'rigid'
                    else:
                        rgdn = prefix + '_rigid.hdf5'  # case: sbx corrected with miniscope
                        if os.path.exists(rgdn):
                            file_in = rgdn
                            self.hdf_source = True
                            self.found = 'rigid'
        # try if not sbx, is _rigid.hdf5 present:
        if not self.found:
            hdn = prefix + '.hdf5'  # case: miniscope avi converted
            if os.path.exists(hdn):
                self.found = 'raw'
                self.hdf_source = True
                file_in = hdn
            if not raw:
                h5n = prefix + '_rigid.hdf5'  # case: miniscope avi ccorrected with miniscope
                if os.path.exists(h5n):
                    self.found = 'rigid'
                    self.hdf_source = True
                    file_in = h5n
                h5n = prefix + '_caiman.h5'  # case: caiman output
                if os.path.exists(h5n):
                    self.found = 'nonrigid'
                    self.data_field_key = 'mov'
                    self.hdf_source = True
                    file_in = h5n
        # try if .tif file
        if not self.found:
            tn = prefix + '.tif'
            if os.path.exists(tn):
                self.found = 'raw'
                self.tif_source = True
                file_in = tn
        # if no image, use just mat files
        if not self.found:
            f = prefix + '.mat'
            if os.path.exists(f):
                file_in = f
                self.found = True
        if not self.found:
            print('Raw 2p movie not available for', prefix)
        else:
            print(file_in)
            if not self.hdf_source and not self.tif_source:
                if matname is None:
                    matname = prefix + '.mat'
                info = loadmat(matpath + matname)['info']
                self.sbx_version = 2
                self.info = info
                if info['channels'] == 1:
                    info['nChan'] = 2
                    factor = 1
                    self.channels = [0, 1]
                elif info['channels'] == 2:
                    info['nChan'] = 1
                    factor = 2
                    self.channels = [0]
                elif info['channels'] == 3:
                    info['nChan'] = 1
                    factor = 2
                    self.channels = [1]
                    # force channel 0 if one channel so original code works with it:
                    self.channels = [0]
                    self.is_red = True
                elif info['channels'] == -1:
                    self.sbx_version = 3
                    self.nplanes = 1
                    self.channels = [i for i, x in enumerate(info['chan']['save']) if x]
                    info['nChan'] = info['chan']['nchan']
                    factor = info['chan']['nchan']
                    if 'etl_table' in info:
                        nplanes = len(info['etl_table'])
                        if nplanes > 1:
                            self.etl_pos = [a[0] for a in info['etl_table']]
                        if nplanes == 0:
                            nplanes = 1
                        self.nplanes = nplanes
                cfg = info['config']
                self.magnification = numpy.nan
                self.pixelsize = numpy.nan
                if 'magnification_list' in cfg:
                    try:
                        self.magnification = cfg['magnification_list'][cfg['magnification'] - 1]
                        self.pixelsize = info['calibration'][cfg['magnification'] - 1].x
                    except:
                        pass
                if self.sbx_version > 2:
                    self.magnification = cfg['magnification_list'][cfg['magnification'] - 1]
                    if hasattr(info, "dxcal"):
                        self.pixelsize = info['dxcal']
                self.factor = factor
                self.nframes = None
                self.data = None
            self.file_in = file_in
            if explicit_need_data or self.hdf_source or self.tif_source:
                self.load_data()
            else:
                self.data_type = 'mat'

    def load_data(self):
        if self.data_loaded:
            return True
        file_in = self.file_in
        self.data_type = 'sbx'
        if not self.hdf_source and not self.tif_source:
            # Determine number of frames in whole file
            # temp hack to skip bidir files until solved
            fsize = os.path.getsize(file_in)
            info = self.info
            if self.sbx_version == 2:
                max_idx = fsize / info['recordsPerBuffer'] / info['sz'][1] * self.factor / 4
                if info['scanmode'] == 0:
                    max_idx /= 2
                n = int(max_idx)  # Last frame
                self.framebytesize = fsize / n
                self.nframes = n
                # Memory map for access
                self.data = numpy.memmap(file_in, dtype='uint16', mode='r',
                                         shape=(n, int(info['sz'][0]), int(info['sz'][1]), info['nChan']))
            elif self.sbx_version == 3:
                nrows, ncols = info['sz']
                # format is NFRAMES x NPLANES x NCHANNELS x NCOLS x NROWS.
                max_idx = int(fsize / nrows / ncols / info['nChan'] / 2)
                self.framebytesize = fsize / int(max_idx)
                self.nframes = int(max_idx / self.nplanes)
                self.mm = np.memmap(file_in, dtype='uint16', order='F', mode='r',
                                    shape=(info['nChan'], ncols, nrows, self.nplanes, self.nframes,)).transpose(
                    (4, 2, 1, 0, 3))  # to conform v2 shape - everything should work unless multiplane
                if self.nplanes == 1:
                    self.data = self.mm.squeeze(4)
                else:
                    self.data = [self.mm[..., i] for i in range(self.nplanes)]

        elif self.hdf_source:
            f = h5py.File(file_in, 'r')
            self.data = f[self.data_field_key]
            self.data_type = 'avi'
            self.nframes = len(self.data)
            self.found = 'hdf5'
            self.channels = [0]
            self.nplanes = 1
            self.info = {'sz': self.data.shape[1:3], 'scanmode': 0, 'nChan': self.data.shape[-1]}
        elif self.tif_source:
            f = TiffFile(file_in)
            print('Reading Tif file')
            self.data = f.asarray()
            self.data_type = 'avi'
            self.nframes = len(self.data)
            self.found = 'tif'
            self.channels = [0]
            self.nplanes = 1
            self.nframes = len(self.data)
            self.info = {'sz': self.data.shape[1:], 'scanmode': 0, 'nChan': 0}
        self.data_loaded = True
        #print(f'data load complete from {file_in}')

    # def getitem_kw(self, frame=None, row=None, col=None, ch=None, plane=None, reshape=True):
    #     '''returns a view to the memory mapped data. pass None for full slice'''
    #     nf, np, nc, ny, nx = self.data.shape
    #     #change None-s to full slice
    #     frame, row, col, ch, plane = [[x, slice(0, None, 1)][x is None] for x in [frame, row, col, ch, plane]]
    #     if self.sbx_version < 3:
    #         return self.data[frame, row, col, ch]
    #     else:
    #         y = self.data[frame, plane, ch, col, row]
    #         if reshape:
    #             assert np == 1 #for now, planes not supported downstream - export in mc per plane
    #             #not sure if I should have this universal interpretation. just have get_frame now...
    #             x = numpy.empty((nf, nx, ny, nc))
    #             for f in range(nf):
    #                 for c in range(nc):
    #                     x[f, ..., c] = y[f, 0, c].transpose()
    #             y = x
    #         return y

    def __getitem__(self, item):
        # if self.sbx_version < 3:
        return self.data[item, :, :, 0]
        # else:
        #     return self.data[item, 0, 0, :, :].transpose()
        # print(f'Called getitem frame {item}')
        # return self.data[item, 0, 0, :, :].ravel(ordef='C').reshape(self.info['sz'])

    def force_read(self, force=False):
        '''to use with hdf5 data source which can be memory mapped but requires indexing in order'''
        if force or self.data_type == 'avi':
            self.data = numpy.array(self.data[...])

    def get_frame(self, frame, ch=None, zplane=0):
        if ch is None:
            ch = slice(0, None, 1)
        if self.nplanes == 1:
            return self.data[frame, :, :, ch]
        else:
            return self.data[zplane][frame, :, :, ch]
        # else:
        #     y = self.data[frame, 0, ch, :, :]
        #     nc, nx, ny = y.shape
        #     x = numpy.empty((ny, nx, nc))
        #     for c in range(nc):
        #         x[..., c] = y[c].reshape((ny, nx))
        #     return x

    def create_zstack(self, step=5, zrange=200, frames=30, norm='none', save='False'):
        self.load_data()
        stack = numpy.zeros((int(zrange / step), *self.info['sz'], 3))
        lut = [1, 2, 0]
        nch = self.info['nChan']
        trim = int(min(frames / 2, 5))
        for z in range(int(zrange / step)):
            for ch in range(nch):
                dx = numpy.nanmean(self.data[z * frames + trim:(z + 1) * frames - 1, :, :, ch], axis=0)
                d = 65535 - dx
                if norm == 'slice':
                    d -= d.min()
                    d /= d.max()
                stack[z, :, :, lut[ch]] = numpy.nan_to_num(d)
        if norm == 'stack':
            stack -= stack.min()
            stack /= numpy.percentile(stack, 99.7)
            stack = numpy.minimum(stack, 1)
        if norm == 'none':
            stack /= 255
        if norm in ['stack', 'slice']:
            stack *= 255
        stack = numpy.maximum(0, numpy.minimum(255, stack))
        self.stack = stack.astype(numpy.uint8)
        if save:
            sstack = numpy.zeros(stack.shape, dtype=numpy.uint8)
            sstack[:, :, :, 0] = stack[:, :, :, 2]
            sstack[:, :, :, 1] = stack[:, :, :, 1]
            imwrite(self.prefix + '_stack.tif', sstack)
            cv2.imwrite(self.prefix + '_MIP.png', numpy.amax(self.stack, axis=0))
        return z

    def get_slice(self, z):
        return self.stack[z, :, :, :]

    def create_previews(self, ret=False):
        pwn = self.prefix + '_preview.tif'
        if not os.path.exists(pwn):
            self.load_data()
            im = numpy.zeros((*self.info['sz'], 3), dtype='uint8')
            lut = [1, 2, 0]
            frs = min(300, self.nframes)
            nch = self.info['nChan']
            for ch in range(nch):
                if not self.data_type == 'avi':
                    d = 65535 - self.show_field(frs, ch)
                    d -= d.min()
                    d /= d.max()
                    im[:, :, lut[ch]] = d * 255
                else:
                    im[:, :, lut[ch]] = self.show_field(frs, ch).squeeze()
            if nch == 0:
                cv2.imwrite(pwn, im[:, :, 1])
            else:
                cv2.imwrite(pwn, im)
        if ret:
            return cv2.imread(pwn, 1)

    def show_field(self, itn=100, channel=0, force=False):
        self.load_data()
        pic = None
        if not force:
            if os.path.exists(self.prefix + '_preview.tif'):
                pic = cv2.imread(self.prefix + '_preview.tif', 1)
                if len(pic.shape) == 3:
                    pic = pic[:, :, 1]
        if pic is None:
            if not self.data_type == 'avi':
                pic = numpy.mean(
                    self.data[numpy.random.choice(len(self.data), min(itn, len(self.data))), :, :, channel], axis=0)
            else:
                pic = numpy.mean(self.data[:itn, :, :, channel], axis=0)
        return pic


class FrameDisplay(object):
    def __init__(self, prefix, raw=False, force=None, tag=None):
        if tag not in ('skip', 'off'):
            self.polys = LoadPolys(prefix, tag)
        self.prefix = prefix
        self.image = LoadImage(prefix, raw=raw, force=force)
        self.preview = None

    def get_cell(self, frame, cell, channel=0):
        x1, x2, y1, y2 = self.polys.bounds(cell)
        return self.image.data[frame, x1 - 1:x2 + 1, y1 - 1:y2 + 1, channel]

    def get_intensity(self, c, itn=300):
        if not hasattr(self, 'c_mask'):
            self.calc_mask()
        if not hasattr(self, 'bgpic'):
            self.bgpic = self.show_field(itn)
        return (65535 - self.bgpic[numpy.where(self.c_mask[c])].mean()) / (
                65535 - self.bgpic[numpy.where(self.halo_mask[c])].mean())

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

    def export_field(self, itn=500, channel=0):
        itn = min(itn, self.image.nframes)
        dat = numpy.empty((itn, *self.image.info['sz']))
        for i, j in enumerate(numpy.random.choice(self.image.nframes, itn)):
            dat[i, :, :] = self.image.data[j, :, :, channel]
        # invert, normalize and create mean channel
        pic = dat.mean(axis=0)
        pic = 256 - pic / 256
        pic *= 255 / pic.max()
        # create and normalize kurtosis channel
        kpic = scipy.stats.kurtosis(dat, axis=0)
        self.kraw = copy.copy(kpic)
        kpic *= 255.0 / kpic.max()
        hsvimg = numpy.empty((*pic.shape, 3), dtype=numpy.uint8)
        hsvimg[:, :, 0] = numpy.ones(pic.shape) * 180
        hsvimg[:, :, 1] = kpic
        hsvimg[:, :, 2] = pic
        img = cv2.cvtColor(hsvimg, cv2.COLOR_HSV2BGR)
        return img
        # cv2.imwrite(prefix+'_preview.tif', pic)
        # self.save_roiset_ij()


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
    return numpy.array([numpy.array(xi) for xi in polys])


if __name__ == '__main__':
    from ImportFirst import *

    os.chdir(r'X:\Barna_unprocessed\test\xx0_234_567')
    prefix = 'xx0_234_567'

    f = LoadImage(prefix, explicit_need_data=True)
    print(f.data.shape)
    single = (len(f.channels) == 1)
    nframes = f.nframes

    frame = 2
    fr = numpy.zeros((*f.info['sz'], 3), dtype='uint8')
    d = f.get_frame(frame)
    avi = f.data_type == 'avi'
    if avi:
        fr[:, :, 1] = d.squeeze()
    elif single:
        fr[:, :, 1] = 255 - (d / 256).squeeze()
    else:
        fr[:, :, 1:] = 255 - (d / 256)
    #
    # row = None
    # col = None
    # ch = None
    # plane = None
    # reshape = True
    #
    # frame, row, col, ch, plane = [[x, slice(0, None, 1)][x is None] for x in [frame, row, col, ch, plane]]
    # y = f.data[frame, plane, ch, col, row]
    # print(y.shape)
    # z = a.createfram_zstack()
    # plt.imshow(a.stack[8, ..., 1])
    #
    # self = a
    # z = 8
    # frames = 30
    # trim = int(min(frames / 2, 5))
    # ch = 0
