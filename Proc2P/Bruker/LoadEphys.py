import os
import numpy
class Ephys:
    '''to interface with spike detector (emulate legacy ripples.trace)'''

    def __init__(self, procpath, prefix, channel=0):
        epn = os.path.join(procpath, prefix, prefix + '_ephys.npy')
        if os.path.exists(epn):
            self.edat = numpy.load(epn)
            self.frames = self.edat[0]
            if channel is not None:
                self.trace = self.edat[channel+1].astype('float') / 1000
        else:
            self.trace = None