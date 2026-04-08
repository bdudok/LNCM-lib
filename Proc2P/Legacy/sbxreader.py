import os
import scipy.io as spio
import numpy as np


class LoadLFP(object):
    def __init__(self, prefix):
        self.prefix = prefix
        a = np.fromfile(self.prefix + '.ephys', dtype='float32')
        b = np.reshape(a, (int(len(a) / 2), 2))
        self.ephystrace = b[:, 1]
        self.ephysframes = b[:, 0]


class LoadImage_deprecated(object):
    '''this reads v2 and v3 files, but use loadPolys.LoadImage instead for implementing multiplane,
     force read, alternative formats and other methods'''
    def __init__(self, prefix):
        self.prefix = prefix
        info = loadmat(prefix + '.mat')['info']
        self.info = info
        self.sbx_version = 2
        if info['channels'] == 1:
            info['nChan'] = 2
            factor = 1
        elif info['channels'] == 2:
            info['nChan'] = 1
            factor = 2
        elif info['channels'] == 3:
            info['nChan'] = 1
            factor = 2
        elif info['channels'] == -1:
            self.sbx_version = 3
            self.nplanes = 1
            info['nChan'] = info['chan']['nchan']
            factor = info['chan']['nchan']
            if hasattr(info, 'etl_table'):
                nplanes = len(info.etl_table)
                if nplanes > 1:
                    etl_pos = [a[0] for a in info.etl_table]
                if nplanes == 0:
                    nplanes = 1
                self.nplanes = nplanes
        file_in = prefix + '.sbx'
        # Memory map for access
        if self.sbx_version == 2:
            max_idx = os.path.getsize(file_in) / info['recordsPerBuffer'] / info['sz'][1] * factor / 4
            if info['scanmode'] == 0:
                max_idx /= 2  # temp hack to read bidir files until solved
            n = int(max_idx)  # Last frame
            self.nframes = n
            self.data = np.memmap(file_in, dtype='uint16',
                              shape=(n, int(info['sz'][0]), int(info['sz'][1]), info['nChan']))
        elif self.sbx_version == 3:
            nrows, ncols = info['sz']
            #format is NFRAMES x NPLANES x NCHANNELS x NCOLS x NROWS.
            max_idx = int(os.path.getsize(file_in) / nrows / ncols / info['nChan'] / 2)
            self.nframes = int(max_idx/self.nplanes)
            self.data = np.memmap(file_in, dtype='uint16',
                              shape=(self.nframes, self.nplanes, info['nChan'], ncols, nrows))


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''

    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


if __name__ == '__main__':
    os.chdir(r'C:\2pdata\test\xx0\xx0_234_567')
    prefix = 'xx0_234_567'
    a = LoadImage(prefix)