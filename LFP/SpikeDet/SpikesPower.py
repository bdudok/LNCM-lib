import pandas
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
        self.stdev_env = numpy.std(self.env)
        self.stdev_trace = numpy.std(self.trace)



    def get_spikes(self, tr1=3, tr2=5, trdiff=2, dur=10, dist=50):
        '''
        :param tr1: lower threshold (for duration)
        :param tr2: higher threshold (for inclusion)
        :param trdiff: threshold for differential-based inclusion (std)
        :param dur: HF duration (ms)
        :param dist: peak separation (ms)
        :return: peak times (s)
        '''

        # print(f'get_spikes called with {tr1}, {tr2}, {trdiff}')


        #get additional HF
        HFO_amp_thresh1 = self.stdev_env * tr1
        HFO_amp_thresh2 = self.stdev_env * tr2

        HFOpeaks, _ = signal.find_peaks(self.env, height=HFO_amp_thresh2, distance=dist * self.fs / ms)
        # HFOpeaks = HFOpeaks[self.env[HFOpeaks] > HFO_amp_thresh2]
        HFO_duration, _, left_ips, right_ips = signal.peak_widths(numpy.clip(self.env, HFO_amp_thresh1, max(self.env)),
                                                       peaks=HFOpeaks, rel_height=1)

        # return HFO_duration, left_ips, right_ips, self.env

        #filter for short peaks
        HFOpeaks = HFOpeaks[HFO_duration / self.fs * ms > dur]

        #generate trace for checking overlap
        overlap = numpy.zeros(len(self.env), dtype='bool')
        for t1, t2 in zip(left_ips.astype('uint64'), right_ips.astype('uint64')):
            overlap[t1:t2] = True

        #detect amplitude swings
        diff2 = numpy.abs(numpy.diff(self.trace, 2))
        d2s = gaussian_filter(diff2, dur*ms/self.fs)
        peakdet_trace = numpy.zeros(len(self.trace))
        d2tr = diff2.mean() + diff2.std() * trdiff
        wh = numpy.where(d2s>d2tr)
        peakdet_trace[wh] = numpy.abs(self.trace[wh])
        AMPpeaks, _ = signal.find_peaks(peakdet_trace, height=self.stdev_trace*2, distance=dist * self.fs / ms)


        #filter for overlap:
        add_peaks = []
        for t in AMPpeaks:
            if overlap[t]:
                continue
            if numpy.min(numpy.abs(HFOpeaks-t) > dist):
                add_peaks.append(t)

        #return sorted
        add_peaks.extend(HFOpeaks)
        ra = numpy.array(add_peaks)
        ra.sort()

        self.spiketimes = ra / self.fs
        self.spikepower = self.env[ra]
        self.spikeamp = self.trace[ra]
        return self.spiketimes

    def spiketimes_to_excel(self, path, prefix, ch=None, savetag=None):
        op = pandas.DataFrame({'SpikeTimes(s)': self.spiketimes, 'SpikeApmlitudes': self.spikeamp,
                               'SpikePower': self.spikepower})
        of = path+prefix
        if savetag is not None:
            of += '_' + savetag
        if ch is None:
            f = of + '_spiketimes.xlsx'
        else:
            f = of + f'_Ch{ch}_spiketimes.xlsx'
        op.to_excel(f)

if __name__ == '__main__':
    processed_path = 'D:\Shares\Data\_Processed/2P\PVTot\LFP/'
    prefix = 'PVTot5_2023-09-15_LFP_025'
    from Proc2P.Bruker import LoadEphys
    fs = 2000
    trace = LoadEphys.Ephys(processed_path, prefix).trace
    spk = Detect(trace, fs=fs)

    t = spk.get_spikes()

    self = spk
    tr1 = 3
    tr2 = 5
    trdiff = 2
    dur = 10
    dist = 50
