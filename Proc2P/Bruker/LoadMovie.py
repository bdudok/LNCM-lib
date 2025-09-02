import os
import numpy
from tifffile import imsave, TiffFile
from Proc2P.utils import lprint


def get_raw_movies(info):
    '''
    :param info: the dict of a SessionInfo
    :return: a dict of the concatenated arrays for each channel
    '''
    dpath = info['dpath']
    filelist = os.listdir(dpath)
    movies = {}
    n_frames = info['n_frames']
    for channel_name in info['channelnames']:
        sz = None
        input_tiffs = [fn for fn in filelist if f'_{channel_name}_' in fn]
        lprint(None, f'Reading {len(input_tiffs)} files for {channel_name}')
        i = 0
        for fn in input_tiffs:
            m = LoadMovie(os.path.join(dpath, fn))
            m.load_data()
            if sz is None:
                sz = m.sz
                height, width = sz
                data = numpy.empty((n_frames, height, width), dtype=m.data.dtype)
            l = m.nframes
            # print(data.shape, i, l, m.data.shape)
            data[i:i + l] = m.data
            i += l
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
        self.data = self.f.asarray()
        self.nframes = len(self.data)
        self.sz = self.data.shape[1:]
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
    path = 'D:\Shares\Data\_RawData\Bruker/testing/20230810_IMG/nm-test-08092023-1000-scan-029/'
    prefix = 'nm-test-08092023-1000-scan-029_Cycle00001_Ch2_000001.ome.tif'

    f = LoadMovie(path + prefix, preload=True)
    print(f.data.shape)
    im = f.preview()
    import matplotlib.pyplot as plt

    plt.imshow(im)
    plt.show()
