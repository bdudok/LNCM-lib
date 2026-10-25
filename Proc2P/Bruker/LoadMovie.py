import os
import numpy
from tifffile import TiffFile #if looking to import imsave, use imwrite instead.
from Proc2P.utils import lprint, path_to_list
from envs import CONFIG


def get_raw_movies(info):
    '''
    :param info: the dict of a SessionInfo
    :return: a dict of the concatenated arrays for each channel
    '''
    dpath = info['dpath']
    prefix = info.prefix
    # the raw data folder may have been moved to a backup location
    filelist = [fn for fn in os.listdir(dpath) if ((prefix in fn) and ('.ome.tif' in fn))] if os.path.isdir(dpath) else []
    if not len(filelist):
        # look in alternative paths
        path1 = str(dpath)
        key = '_RawData'
        if key not in path1:
            raise FileNotFoundError(f'No raw movie in {dpath}. Alternative paths are only searched for folders in {key}')
        rpath = path1[path1.find(key) + len(key):]
        parent_folders = path_to_list(rpath)
        # check in each alt path and stop if found em
        for alt_path in CONFIG.alt_raw_paths:
            path2 = os.path.join(os.path.realpath(alt_path), *parent_folders)
            if not os.path.exists(path2):
                continue
            filelist = [fn for fn in os.listdir(path2) if ((prefix in fn) and ('.ome.tif' in fn))]
            if len(filelist):
                print(f'Image data loaded from {path2}')
                dpath = path2
                break
    movies = {}
    n_frames = info['n_frames']
    for channel_name in info['channelnames']:
        sz = None
        input_tiffs = [fn for fn in filelist if f'_{channel_name}_' in fn]
        lprint(None, f'Reading {len(input_tiffs)} files for {channel_name}')
        if not len(input_tiffs):
            raise FileNotFoundError(f'No ome.tif files found at {dpath} for {prefix}')
        i = 0
        for fn in input_tiffs:
            print(dpath, fn)
            m = LoadMovie(os.path.join(dpath, fn))
            m.load_data()
            if sz is None:
                sz = m.sz
                height, width = sz
                data = numpy.empty((n_frames, height, width), dtype=m.data.dtype)
            n_take = min(m.nframes, n_frames - i) # to discard spurious frames from malformed tif exports
            # print(data.shape, i, n_take, m.data.shape)
            data[i:i + n_take] = m.data[:n_take]
            i += n_take
        movies[channel_name] = data
        sz = None
    return movies


class LoadMovie():
    __name__ = 'LoadMovie'
    '''
    For loading a raw Buker Tif file.
    #pass filename with full path directly. figure out channels and filenames upstream.
    '''

    def __init__(self, filehandle, preload=False):
        self.filehandle = filehandle

        if not os.path.exists(filehandle):
            lprint(self, filehandle + ' not found.')
            return None
        self.f = TiffFile(filehandle)
        self.data_loaded = False
        if preload:
            self.load_data()

    def load_data(self):
        if self.data_loaded:
            return True

        #calling .asarray on each page is ~100x faster than on the TiffFile object
        mmap = self.f.pages
        n_pages = len(mmap)
        frameshape = mmap[0].shape
        self.data = numpy.empty((n_pages, *frameshape), dtype=mmap[0].dtype)
        for i in range(n_pages):
            self.data[i] = mmap[i].asarray()

        self.nframes = n_pages
        self.sz = frameshape
        self.data_loaded = True
        lprint(self, 'Tif file loaded.')
        return True

    def get_frame(self, frame, ch=None, zplane=0):
        if ch is None:
            ch = slice(0, None, 1)
        if self.nplanes == 1:
            return self.data[frame, :, :, ch]
        else:
            return self.data[zplane][frame, :, :, ch]

    def preview(self, itn=100):  # , channel=0, zplane=0):
        self.load_data()
        indices = numpy.random.choice(numpy.arange(self.nframes), itn, replace=False)
        indices.sort()
        return numpy.mean(self.data[indices, :, :], axis=0)
        # handling of cases with more complex files tbd


if __name__ == '__main__':
    # path = 'D:\Shares\Data\_RawData\Bruker/2P/JEDI-IPSP/'
    # prefix = 'JEDI-Sncg204_2026-01-21_lfp_opto_1765-000'
    fn = r'D:\Shares\Data\_RawData\Bruker\JEDI-IPSP\JEDI-Sncg204_2026-01-21_lfp_opto_1765-000\JEDI-Sncg204_2026-01-21_lfp_opto_1765-000_Cycle00001_Ch2_000002.ome.tif'

    lprint(None, 'started reading')
    f = LoadMovie(fn, preload=False)
    im = f.preview()
    lprint(None, f.data.shape)

    import matplotlib.pyplot as plt

    plt.imshow(im)
    plt.show()
