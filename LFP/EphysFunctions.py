import numpy
from scipy import signal

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    filter = signal.butter(order, [low, high], btype='band', output='sos')
    y = signal.sosfiltfilt(filter, data)
    return y

def z_scored_power(trace, lowcut, highcut, fs):
    ftr = butter_bandpass_filter(trace, lowcut, highcut, fs)
    p = numpy.abs(signal.hilbert(ftr))
    # z score based on baseline
    z1 = p - p.mean()
    mask = z1 < (z1.std() * 2)
    p -= p[mask].mean()
    p /= p[mask].std()
    return p