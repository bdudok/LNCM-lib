import os
import numpy
class Ephys:
    '''to interface with spike detector (emulate legacy ripples.trace)'''

    def __init__(self, procpath, prefix, channel=0):
        edat = numpy.load(os.path.join(procpath, prefix, prefix + '_ephys.npy'))[channel+1]
        self.trace = edat.astype('float') / 1000
