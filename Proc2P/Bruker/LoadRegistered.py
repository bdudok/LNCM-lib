import numpy
import os
from Proc2P.Bruker.ConfigVars import CF
from Proc2P.utils import path_to_list

class LoadRegistered():
    __name__ = 'LoadRegistered'
    '''
    For loading a S2p-processed movie from binary.
    #pass full path to the processed output folder.
    '''

    def __init__(self, procpath, prefix):
        self.path = os.path.join(procpath, prefix)
        self.prefix = prefix
        lp = os.path.join(self.path, 'suite2p/plane0/ops.npy')
        ops = numpy.load(lp, allow_pickle=True).item()
        self.n_frames = ops['frames_per_folder'][0]
        self.Ly = ops['Ly']
        self.Lx = ops['Lx']
        self.find_files_in_folder(self.path)
        self.shape = (self.n_frames, self.Ly, self.Lx)
        self.load()

    def find_files_in_folder(self, path):
        # find available files
        tags = '', '_Ch1', '_Ch2'
        keys = (None, 'Ch1', 'Ch2')
        self.input_files = []
        self.channel_keys = []
        for ti, (tag, key) in enumerate(zip(tags, keys)):
            bp = os.path.join(path, self.prefix + f'_registered{tag}.bin')
            if os.path.exists(bp):
                self.input_files.append(bp)
                self.channel_keys.append(key)
        self.n_channels = len(self.input_files)

    def find_alt_path(self):
        '''
        use the alternative data path specified in the config file (absoulute path to _Processed folder)
        to load the binary data of the motion corrected movie. does not change the saving location
        '''
        #get last subfolders
        path1 = str(self.path)
        key = '_Processed'
        path = path1[path1.find(key)+len(key):]
        parent_folders = path_to_list(path)
        #check in each alt path and stop if found em
        for alt_path in CF.alt_processed_paths:
            path2 = os.path.join(os.path.realpath(alt_path), *parent_folders)
            self.find_files_in_folder(path2)
            if len(self.input_files):
                print(f'Image data loaded from {path2}')
                self.load()
                break

    def load(self):
        if len(self.input_files):
            self.data = numpy.memmap(self.input_files[0], mode='r', dtype='int16', shape=self.shape)
            self.bitdepth = pow(2, 12)
            if len(self.input_files) > 1:
                self.data2 = numpy.memmap(self.input_files[1], mode='r', dtype='int16', shape=self.shape)
        else:
            self.data = None

    def get_channel(self, ch=None):
        '''
        :param ch 1: returns second channel. 'Green': Ch2; 'Red: Ch1. Any other value returns first channel
        :return: memory mapped data corresponding to the selected channel
        '''
        ret_ch = 0
        # print(self.channel_keys)
        if ch == 'Green' or ch == 'Ch2':
            assert 'Ch2' in self.channel_keys
            if self.channel_keys[0] == 'Ch2':
                pass
            elif self.channel_keys[1] == 'Ch2':
                ret_ch = 1
        elif ch == 'Red' or ch == 'Ch1':
            assert 'Ch1' in self.channel_keys
            if self.channel_keys[0] == 'Ch1':
                pass
            elif self.channel_keys[1] == 'Ch1':
                ret_ch = 1
        elif ch in (0, 1):
            assert self.n_channels > ch
            ret_ch = ch
        if ret_ch == 0:
            return self.data
        else:
            return self.data2
