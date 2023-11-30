import numpy
import os


class LoadRegistered():
    __name__ = 'LoadRegistered'
    '''
    For loading a S2p-processed movie from binary.
    #pass full path to the processed output folder.
    '''

    def __init__(self, procpath, prefix):
        self.path = os.path.join(procpath, prefix)
        lp = os.path.join(self.path, 'suite2p/plane0/ops.npy')
        ops = numpy.load(lp, allow_pickle=True).item()
        self.n_frames = ops['frames_per_folder'][0]
        self.Ly = ops['Ly']
        self.Lx = ops['Lx']
        # find available files
        tags = '', '_Ch1', '_Ch2'
        keys = (None, 'Ch1', 'Ch2')
        input_files = []
        self.channel_keys = []
        for ti, (tag, key) in enumerate(zip(tags, keys)):
            bp = os.path.join(self.path, prefix + f'_registered{tag}.bin')
            if os.path.exists(bp):
                input_files.append(bp)
                self.channel_keys.append(key)
        self.n_channels = len(input_files)
        self.shape = (self.n_frames, self.Ly, self.Lx)
        # instead of loading the dimensttions, this could be passed from session info
        self.data = numpy.memmap(input_files[0], mode='r', dtype='int16', shape=self.shape)
        self.bitdepth = pow(2, 12)
        if len(input_files) > 1:
            self.data2 = numpy.memmap(input_files[1], mode='r', dtype='int16', shape=self.shape)

    def load(self):
        pass #no need to override memory map with the s2p binary
        # self.data = numpy.array(self.data[...])

    def get_channel(self, ch=None):
        '''
        :param ch 1: returns second channel. 'Green': Ch2; 'Red: Ch1. Any other value returns first channel
        :return: memory mapped data corresponding to the selected channel
        '''
        ret_ch = 0
        print(self.channel_keys)
        if ch == 'Green':
            assert 'Ch2' in self.channel_keys
            if self.channel_keys[0] == 'Ch2':
                pass
            elif self.channel_keys[1] == 'Ch2':
                ret_ch = 1
        elif ch == 'Red':
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
