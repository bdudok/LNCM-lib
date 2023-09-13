import datetime
import numpy
def lprint(obj, message):
    '''Add timestamp and object name to print calls'''
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    print(f'{ts} - {obj.__name__}: {message}')

from Proc2P.Legacy.Batch_Utils import strip_ax

def gapless(trace, gap=5, threshold=0):
    '''makes binary trace closing small gaps
    :param gap: in samples
    '''
    gapless = trace > threshold
    ready = False
    while not ready:
        ready = True
        for t, m in enumerate(gapless):
            if not m:
                if numpy.any(gapless[t - gap:t]) and numpy.any(gapless[t:t + gap]):
                    gapless[t] = 1
                    ready = False
    return gapless

def outlier_indices(values, thresh=3.5):
    '''return the list of indices of outliers in a series'''
    not_nan = numpy.logical_not(numpy.isnan(values))
    median = numpy.median(values[not_nan])
    diff = (values - median) ** 2
    diff = numpy.nan_to_num(numpy.sqrt(diff))
    med_abs_deviation = numpy.median(diff[not_nan])
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return numpy.where(modified_z_score > thresh)[0]