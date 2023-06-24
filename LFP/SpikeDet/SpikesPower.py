from scipy import signal
import numpy
from LFP.EphysFunctions import butter_bandpass_filter
#detect peaks on a filtered trace

class Detect:
    def __init__(self, trace, fs, lo=80, hi=500):
        '''
        :param trace: array (raw LFP)
        :param fs: sampling rate
        :param lo: filter low cut
        :param hi: filter high cut
        '''

    #resample if too high rate
        max_fs = 2000
        if fs > max_fs*1.5:
            trace = signal.resample(trace, len(trace)/fs*max_fs)
        self.trace = trace
        self.fs = fs


        #filter
        self.HF = butter_bandpass_filter(trace, lo, hi, fs,)
        self.env = numpy.abs(signal.hilbert(self.FH))
        self.stdev = numpy.std(self.HF)

    def get_spikes(self, tr1=3, tr2=7, dur=10, dist=50):
        '''
        :param tr1: lower threshold (for duration)
        :param tr2: higher threshold (for inclusion)
        :param dur: HF duration (ms)
        :param dist: peak separation (ms)
        :return: peak times (s)
        '''

        #transform
        HFO_amp_thresh1 = self.stdev * tr1
        HFO_amp_thresh2 = self.stdev * tr2

        HFOpeaks, _ = signal.find_peaks(self.env, height=HFO_amp_thresh1, distance=dist * self.fs / 1000)
        HFO_duration, _, _, _ = signal.peak_widths(numpy.clip(self.env, HFO_amp_thresh2, max(self.env)),
                                                       peaks=HFOpeaks, rel_height=1)
        return HFOpeaks[HFO_duration / self.fs * 1000 > dur]

