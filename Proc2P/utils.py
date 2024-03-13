import datetime
import numpy
import pandas

def lprint(obj, message, *args):
    '''Add timestamp and object name to print calls'''
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    for x in args:
        message += ' ' + str(x)
    if obj is None:
        print(f'{ts}: {message}')
    else:
        print(f'{ts} - {obj.__name__}: {message}')

from Proc2P.Legacy.Batch_Utils import strip_ax

def gapless(trace, gap=5, threshold=0):
    '''makes binary trace closing small gaps
    :param gap: in samples
    '''
    if trace is None:
        return None
    elif not numpy.count_nonzero(trace):
        return numpy.zeros(len(trace))
    gapless = trace > threshold
    ready = False
    gap = int(gap)
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

def startstop(speed, duration=50, gap=150, ret_loc='actual', span=None, speed_threshold=0.05):
    '''
    :param speed
    :param duration: of run, in samples
    :param gap: of stop, in samples
    :param ret_loc: 'actual' gives frame with zero speed. need to specify speed for this
    :param span: slice of recording to analyze
    :return:
    '''
    # binary gapless trace whether animal is running
    mov = gapless(numpy.nan_to_num(speed), threshold=speed_threshold)
    if gap is None:
        gap = 150
    if span is None:
        span = gap, len(mov) - gap
    # collect stops
    stops, starts = [], []
    t = span[0] + duration
    while t < span[1] - gap:
        if not numpy.any(mov[t:t + gap]) and numpy.all(mov[t - duration:t]):
            t0 = t
            if ret_loc == 'peak':
                while speed[t0] < speed[t0 - 1]:
                    t0 -= 1
                stops.append(t0)
                while mov[t0]:
                    t0 -= 1
                starts.append(t0)
            elif ret_loc == 'stopped':
                stops.append(t)
                while mov[t0 - 1]:
                    t0 -= 1
                starts.append(t0)
            elif ret_loc == 'actual':
                # go back while raw speed is zero
                while speed[t - 1] < speed_threshold and t > 100:
                    t -= 1
                if t > 100:
                    stops.append(t)
                    while mov[t0 - 1] and t0 > 100:
                        t0 -= 1
                    while speed[t0] < speed_threshold:
                        t0 += 1
                    starts.append(t0)
            t += gap
        t += 1
    return starts, stops


def read_excel(*args, **kwargs):
    fn = args[0]
    if fn.endswith('csv'):
        return pandas.read_csv(*args, **kwargs)
    return pandas.read_excel(*args, **kwargs, engine='openpyxl')