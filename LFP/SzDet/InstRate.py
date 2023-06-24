import numpy
from sklearn.neighbors import KernelDensity

def SpikeTrace(spiketimes, framesize:int, length=None):
    '''
    Take a list of spike times (assuming sorted) and convert to a time series of inst spike rate
    :param spiketimes: array (s)
    :param fs: sampling rate (s)
    :param framesize: rate of the continuous output (ms)
    :param length: length of the output time series. if none, 1 s after last spike.
    :return: array
    '''
    spk_times = spiketimes
    if length is None:
        length = int((spk_times[-1] + 1) * 1000 / framesize)

    rec_dur = length * framesize / 1000

    # KDE approach: estimate instantaneous spike rate with the resolution of 'framesize' (ms units)
    # NB indices will have to be recalculated using framesize to use this for anything.
    X = numpy.arange(0, length*framesize+1, framesize) / 1000
    # fitting a gaussian kernel on the event distribution.
    # higher value for bandwidth = more smoothing. the 1/3 sec works very well for
    #  approximating a rolling count of spikes within 1 sec
    kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(spk_times.reshape(-1, 1))
    inst_rate = numpy.exp(kde.score_samples(X.reshape(-1, 1)))[1:]  # this returns log prob density, so we exp it
    # calibrate to Hz. the returned probability density does not have units.
    mean_freq = len(spk_times) / (X[-1] - X[0])
    inst_rate *= mean_freq / inst_rate.mean()

    # detect seizures.
    sz_burden = numpy.empty(length)
    sz_burden[:] = numpy.nan
    sz = []
    fs = 1000/framesize
    spk_counter = numpy.zeros(length)
    spk_n = numpy.zeros(len(inst_rate))
    for t in spk_times:
        spk_n[int(t*fs)] += 1
        if 1 < t < rec_dur-1:
            spk_counter[int((t - 0.5) * fs): int((t + 0.5) * fs)] += 1
    # find szs:
    s = int(fs)
    slim = int(length - fs)
    while s < slim:
        # find sz ON
        if spk_counter[s]:
            # first spike
            s = s + numpy.argmax(spk_n[s:int(s + fs)])
            s1 = s
            while spk_counter[s1 + 1]:
                s1 += 1
                if s1 + 1 > slim:
                    break
            # determine if at least 5 spikes
            if spk_n[s:s1].sum() > 4:
                # extend until it drops to 0
                while inst_rate[s1 + 1] > 0.3 and s1 + 1 < slim:  ##0.3 to match the KDE bandwidth
                    if s1 + fs < slim and inst_rate[int(s1 + fs)] > 0.3:
                        # large step size prevents closing if another seizure starts soon
                        s1 += int(fs)
                    else:
                        s1 += 1
                sz.append(((s-1) / fs, s1 / fs))
                sz_burden[s-1:s1] = inst_rate[s-1:s1]
            s = s1 + 1
        else:
            s += 1

    # add seizures to an array
    sz_times = numpy.empty((len(sz), 2), dtype='int32')
    for i, szt in enumerate(sz):
        sz_times[i] = szt

    return sz_burden, sz_times * fs