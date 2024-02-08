import os
import numpy
class Ephys:
    '''to interface with spike detector (emulate legacy ripples.trace)'''

    def __init__(self, procpath, prefix, channel=1):
        epn = os.path.join(procpath, prefix, prefix + '_ephys.npy')
        self.path = os.path.join(procpath, prefix, '')
        self.prefix = prefix
        # print(epn)
        if os.path.exists(epn):
            self.edat = numpy.load(epn)
            self.frames = self.edat[0]
            if channel is not None: #indexed from 1, 0 is frame
                self.trace = self.edat[channel].astype('float') / 1000
        else:
            self.trace = None