import numpy
from scipy.signal import bessel, hilbert, filtfilt, decimate
class Filter:
    __name__ = 'Filter'

    def __init__(self, trace, fs):
        self.trace = trace
        self.fs = fs

    def gen_filt(self, lowcut, highcut, order=3):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = min(highcut / nyq, 0.95)
        return bessel(order, [low, high], btype='band')

    def run_filt(self, filter):
        ftr = filtfilt(*filter, self.trace)
        htr = hilbert(ftr)
        envelope = numpy.abs(htr)
        return ftr, envelope
