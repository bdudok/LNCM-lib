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
        bp = os.path.join(self.path, prefix+'_registered.bin')
        self.n_frames = ops['frames_per_folder'][0]
        self.Ly = ops['Ly']
        self.Lx = ops['Lx']
        self.shape = (self.n_frames, self.Ly, self.Lx)
        #instead of loading the dimensttions, this could be passed from parent
        self.data = numpy.memmap(bp, mode='r+', dtype='int16', shape=self.shape)
        self.bitdepth = pow(2, 12)

    def load(self):
        self.data = numpy.array(self.data[...])