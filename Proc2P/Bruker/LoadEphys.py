import os
import numpy
class Ephys:

    def __init__(self, procpath, prefix, channel=0):
        self.trace = (numpy.load(os.path.join(procpath, prefix, prefix + '_ephys.npy'))[channel]).astype('float') / 1000
