import os

import matplotlib.pyplot as plt
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
            #this can overflow after ~30k frames, because it's stored in int with the microvolt traces
            maxval = numpy.iinfo(self.frames.dtype).max
            wh_turn = numpy.where(self.frames == maxval)[0]
            frames = self.frames.astype(numpy.int64)
            for t_idx in wh_turn:
                if frames[t_idx+1] < frames[t_idx]:
                    frames[t_idx+1:] = frames[t_idx] + (frames[t_idx+1:] - frames[t_idx+1])
            self.frames = frames
            if channel is not None: #indexed from 1, 0 is frame
                self.trace = self.edat[channel].astype('float') / 1000
        else:
            self.trace = None