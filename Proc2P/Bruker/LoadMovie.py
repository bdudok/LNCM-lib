import os
import numpy
from tifffile import TiffFile #if looking to import imsave, use imwrite instead.
from Proc2P.utils import lprint, path_to_list
from envs import CONFIG


def alt_raw_locations(path):
    '''
    :param path: a file or folder in _RawData
    :return: the same path in each alternative raw data path (backup locations, CONFIG.alt_raw_paths). Empty if the
        path is not in _RawData.
    '''
    path1 = str(path)
    key = '_RawData'
    if key not in path1:
        return []
    parent_folders = path_to_list(path1[path1.find(key) + len(key):])
    return [os.path.join(os.path.realpath(alt_path), *parent_folders) for alt_path in CONFIG.alt_raw_paths]


def find_raw_files(info):
    '''
    :param info: the dict of a SessionInfo
    :return: the folder of the raw movie (the backup location if it was moved there) and its sorted .ome.tif files.
        No files if the movie is not found.
    '''
    prefix = info.prefix
    # the raw data folder may have been moved to a backup location
    for dpath in (info['dpath'], *alt_raw_locations(info['dpath'])):
        if os.path.isdir(dpath):
            filelist = [fn for fn in os.listdir(dpath) if ((prefix in fn) and ('.ome.tif' in fn))]
            if len(filelist):
                if dpath != info['dpath']:
                    print(f'Image data loaded from {dpath}')
                return dpath, sorted(filelist)
    return info['dpath'], []


def get_raw_movies(info):
    '''
    :param info: the dict of a SessionInfo
    :return: a dict of the concatenated arrays for each channel
    '''
    dpath, filelist = find_raw_files(info)
    prefix = info.prefix
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


def get_lazy_raw_movies(info):
    '''
    Like get_raw_movies, without loading the movies: frames are read from the files when indexed.
    :param info: the dict of a SessionInfo
    :return: a dict of a LazyRawMovie for each channel. Empty if the raw movie is not found.
    '''
    dpath, filelist = find_raw_files(info)
    movies = {}
    for channel_name in info['channelnames']:
        input_tiffs = [os.path.join(dpath, fn) for fn in filelist if f'_{channel_name}_' in fn]
        if len(input_tiffs):
            movies[channel_name] = LazyRawMovie(input_tiffs, info['n_frames'])
    return movies


class LazyRawMovie:
    '''
    One channel of a raw Bruker movie, split into ome.tif files. Indexing reads frames from the files, so the movie
    is not loaded into memory. Each file is opened when a frame in it is first read.
    Bruker writes the same number of frames into each file except the last, so only the first file's pages are counted.
    '''

    def __init__(self, files, n_frames):
        self.files = files
        self.tiffs = {}  # opened files by index
        first = self.open(0)
        self.frames_per_file = len(first.pages)
        assert (len(files) - 1) * self.frames_per_file < n_frames <= len(files) * self.frames_per_file, \
            f'{n_frames} frames do not fit {len(files)} files of {self.frames_per_file} frames: {files[0]}'
        page = first.pages[0]
        self.shape = (n_frames, *page.shape)
        self.dtype = page.dtype

    def open(self, file_index):
        if file_index not in self.tiffs:
            tif = TiffFile(self.files[file_index], is_ome=False)  # is_ome=False: don't parse the OME-XML
            tif.pages.useframes = True  # light page objects, only offsets
            self.tiffs[file_index] = tif
        return self.tiffs[file_index]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        if numpy.ndim(item):
            return numpy.stack([self[i] for i in item])
        file_index, page = divmod(int(item), self.frames_per_file)
        return self.open(file_index).pages[page].asarray()

    def close(self):
        for tif in self.tiffs.values():
            tif.close()
        self.tiffs = {}


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
