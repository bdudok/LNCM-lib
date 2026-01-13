import numpy
import os
from envs import CONFIG
from Proc2P.utils import path_to_list
from enum import Enum
import json

class Source(Enum):
    S2P = 0
    GEVIReg = 1
    Raw = 2


class LoadRegistered():
    __name__ = 'LoadRegistered'
    '''
    For loading a S2p-processed movie from binary.
    #pass full path to the processed output folder.
    '''

    def __init__(self, procpath, prefix, source=Source.S2P):
        '''
        Open a registered 2P movie (saved by Suite2P or by GEVIReg)
        :param source: 'S2P' or 'GEVIReg'
        '''
        self.path = os.path.join(procpath, prefix)
        self.prefix = prefix
        self.source = source
        if source == Source.S2P:
            lp = os.path.join(self.path, 'suite2p/plane0/ops_compat.json')
            if os.path.exists(lp):
                with open(lp, "r") as f:
                    ops = json.load(f)
                self.n_frames = ops['n_frames']
            else:
                lp = os.path.join(self.path, 'suite2p/plane0/ops.npy')
                ops = numpy.load(lp, allow_pickle=True).item()
                self.n_frames = ops['frames_per_folder'][0]
            self.Ly = ops['Ly']
            self.Lx = ops['Lx']

            self.shape = (self.n_frames, self.Ly, self.Lx)
        elif source == Source.GEVIReg:
            pass # no prep required, npy header contains shape, will be set by load()
        self.find_files_in_folder(self.path)
        self.load()

    def find_files_in_folder(self, path):
        # find available files
        tags = '', '_Ch1', '_Ch2'
        keys = (None, 'Ch1', 'Ch2')
        self.input_files = []
        self.channel_keys = []
        for ti, (tag, key) in enumerate(zip(tags, keys)):
            if self.source == Source.S2P:
                bp = os.path.join(path, self.prefix + f'_registered{tag}.bin')
            elif self.source == Source.GEVIReg:
                bp = os.path.join(path, 'GEVIReg', self.prefix + f'_registered{tag}.npy')
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
        for alt_path in CONFIG.alt_processed_paths:
            path2 = os.path.join(os.path.realpath(alt_path), *parent_folders)
            self.find_files_in_folder(path2)
            if len(self.input_files):
                print(f'Image data loaded from {path2}')
                self.load()
                break

    def load(self):
        if len(self.input_files):
            if self.source == Source.S2P:
                self.data = numpy.memmap(self.input_files[0], mode='r', dtype='int16', shape=self.shape)
                if len(self.input_files) > 1:
                    self.data2 = numpy.memmap(self.input_files[1], mode='r', dtype='int16', shape=self.shape)
            elif self.source == Source.GEVIReg:
                self.data = numpy.load(self.input_files[0], mmap_mode='r')
                self.shape = self.data.shape
                self.n_frames, self.Ly, self.Lx = self.shape
                if len(self.input_files) > 1:
                    self.data2 = numpy.load(self.input_files[1], mmap_mode='r')
            self.bitdepth = pow(2, 13) # all bruker data has 13 effective bits
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

def check_compatible_ops(procpath, prefix):
    path = os.path.join(procpath, prefix)
    in_file = os.path.join(path, 'suite2p/plane0/ops.npy')
    out_file = os.path.join(path, 'suite2p/plane0/ops_compat.json')
    if os.path.exists(in_file) and not os.path.exists(out_file):
        ops = numpy.load(in_file, allow_pickle=True).item()
        op = {
            "n_frames": int(ops['frames_per_folder'][0]),
            "Lx": int(ops['Lx']),
            "Ly": int(ops['Ly'])
        }
        with open(out_file, "w") as f:
            json.dump(op, f, )