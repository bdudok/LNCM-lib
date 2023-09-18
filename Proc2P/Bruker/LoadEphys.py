import os
import numpy
class Ephys:
    '''to interface with spike detector (emulate legacy ripples.trace)'''

    def __init__(self, procpath, prefix, channel=0):
        self.edat = numpy.load(os.path.join(procpath, prefix, prefix + '_ephys.npy'))
        self.frames = self.edat[0]
        if channel is not None:
            self.trace = self.edat[channel+1].astype('float') / 1000