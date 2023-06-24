from scipy import signal
import numpy
from LFP.EphysFunctions import butter_bandpass_filter
from scipy.ndimage import gaussian_filter
#detect peaks on a filtered trace
ms = 1000

class Detect:
    def __init__(self, trace, fs, lo=80, hi=500):
        '''
        :param trace: array (raw LFP)
        :param fs: sampling rate
        :param lo: filter low cut
        :param hi: filter high cut
        '''

    #resample if too high rate
        max_fs = 2 * ms
        if fs > max_fs*1.5:
            trace = signal.resample(trace, int(len(trace)/fs*max_fs))
            self.fs = max_fs
        else:
            self.fs = fs
        self.trace = trace

        #filter
        self.HF = butter_bandpass_filter(trace, lo, hi, fs,)
        self.env = numpy.abs(signal.hilbert(self.HF))
        self.stdev_env = numpy.std(self.HF)
        self.stdev_trace = numpy.std(self.trace)



    def get_spikes(self, tr1=3, tr2=5, dur=10, dist=50):
        '''
        :param tr1: lower threshold (for duration)
        :param tr2: higher threshold (for inclusion)
        :param dur: HF duration (ms)
        :param dist: peak separation (ms)
        :return: peak times (s)
        '''

        #get LFP amplitude peaks
        diff2 = numpy.abs(numpy.diff(self.trace, 2))
        d2s = gaussian_filter(diff2, dur*ms/self.fs)
        peakdet_trace = numpy.zeros(len(self.trace))
        d2tr = diff2.mean() + diff2.std()
        wh = numpy.where(d2s>d2tr)
        peakdet_trace[wh] = numpy.abs(self.trace[wh])
        AMPpeaks, _ = signal.find_peaks(peakdet_trace, height=self.stdev_trace*tr1, distance=dist * self.fs / ms)

        #get additional HF
        HFO_amp_thresh1 = self.stdev_env * tr1
        HFO_amp_thresh2 = self.stdev_env * tr2

        HFOpeaks, _ = signal.find_peaks(self.env, height=HFO_amp_thresh1, distance=dist * self.fs / ms)
        HFO_duration, _, _, _ = signal.peak_widths(numpy.clip(self.env, HFO_amp_thresh2, max(self.env)),
                                                       peaks=HFOpeaks, rel_height=1)

        #filter for short peaks
        HFOpeaks = HFOpeaks[HFO_duration / self.fs * ms > dur]

        #filter for overlap:
        add_peaks = []
        for t in HFOpeaks:
            if numpy.any(numpy.abs(AMPpeaks-t) < (dur / ms)):
                add_peaks.append(t)

        #return sorted
        add_peaks.extend(AMPpeaks)
        ra = numpy.array(add_peaks)
        ra.sort()
        return ra

