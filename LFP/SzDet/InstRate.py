import numpy
from sklearn.neighbors import KernelDensity


def SpikeTrace(spiketimes, framesize:int, length=None, cleanup=5, gap=2):
    '''
    Take a list of spike times (assuming sorted) and convert to a time series of inst spike rate
    :param spiketimes: array (s)
    :param fs: sampling rate (s)
    :param framesize: rate of the continuous output (ms)
    :param length: length of the output time series. if none, 1 s after last spike.
    :param cleanup: length of minimum sz duration (s) for inclusion
    :param gap: duration between seizures for merging (s)
    :return: tuple of (array of seizure burden, array of seizure times in framesize)
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
            s = s + numpy.searchsorted(spk_n[s:int(s + fs)], 1)
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
                sz.append([s / fs, s1 / fs])
                sz_burden[s:s1] = inst_rate[s:s1]
            s = s1 + 1
        else:
            s += 1


    #merge seizures
    merged_sz = []
    i = 0
    while i < (len(sz)-1):
        # print(f'checking sz {i}: {sz[i]}')
        j = i + 1
        t0 = sz[i][0]
        t1 = sz[i][1]
        while j < len(sz) and sz[j][0] - sz[j-1][1] < gap:
            # print(f'Within gap: sz {j}: {sz[j]}')
            t1 = sz[j][1]
            j += 1
        if t1 - t0 > cleanup:
            #merge if any of the cluster members longer
            for k in range(i, min(len(sz)-1, j+1)):
                if sz[k][1] - sz[k][0] > cleanup:
                    merged_sz.append([t0,t1])
                    break
            # print(f'merging {i} to {j-1}')
        i = j


    # add seizures to an array
    sz_times = numpy.empty((len(merged_sz), 2), )#dtype='int32')
    for i, szt in enumerate(merged_sz):
        sz_times[i] = szt

    return sz_burden, sz_times