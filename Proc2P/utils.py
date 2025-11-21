import os
import datetime
import numpy
import pandas
from scipy import stats


def lprint(obj, message, *args, logger=None):
    '''Add timestamp and object name to print calls'''
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    message = str(message)
    for x in args:
        message += ' ' + str(x)
    if obj is None:
        output = f'{ts}: {message}'
    else:
        output = f'{ts} - {obj.__name__}: {message}'
    print(output)
    if logger is None:
        return output
    else:
        logger.log(output)


class logger:
    def __init__(self, filehandle=None, defer=0):
        self.fn = filehandle
        self.defer = defer
        self.unsaved = 0
        self.message = ''

    def set_handle(self, procpath, prefix,):
        self.fn = os.path.join(procpath, prefix + '/', prefix + f'_{get_user()}_AnalysisLog.txt')

    def log(self, message):
        if not message.endswith('\n'):
            message += '\n'
        self.message += message
        self.unsaved += 1
        if self.unsaved > self.defer:
            with open(self.fn, 'a') as f:
                f.write(self.message)
                self.unsaved = 0
                self.message = ''

    def flush(self):
        if self.unsaved > 0:
            with open(self.fn, 'a') as f:
                f.write(self.message)


def strip_ax(ca, full=True):
    ca.spines['right'].set_visible(False)
    ca.spines['top'].set_visible(False)
    if full:
        ca.spines['bottom'].set_visible(False)
        ca.spines['left'].set_visible(False)
        ca.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='off')
        ca.tick_params(axis='y', which='both', right='off', left='off', labelright='off')
        ca.xaxis.set_visible(False)
        ca.yaxis.set_visible(False)


def get_user():
    return os.environ.get('USERNAME')


def norm(d):
    wh_notna = numpy.logical_not(numpy.isnan(d))
    y = numpy.empty(d.shape)
    y[:] = numpy.nan
    a = d[wh_notna] - numpy.min(d[wh_notna])
    y[wh_notna] = numpy.minimum(a / numpy.percentile(a, 99, axis=0), 1)
    return y


def gapless(trace, gap=5, threshold=0, expand=None):
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
    if expand is not None:
        eg = numpy.copy(gapless)
        for t, m in enumerate(gapless):
            if not m:
                if numpy.any(gapless[t - expand:t]) or numpy.any(gapless[t:t + expand]):
                    eg[t] = 1
        gapless = eg
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


def startstop(speed, duration=50, gap=150, ret_loc='actual', span=None, speed_threshold=0.05, smoothing=None):
    '''
    :param speed
    :param duration: of run, in samples
    :param gap: of stop, in samples
    :param ret_loc: 'actual' gives frame with zero speed. need to specify speed for this
    :param span: slice of recording to analyze
    :return:
    '''
    # binary gapless trace whether animal is running
    if smoothing is None:
        smoothing = 20
    mov = gapless(numpy.nan_to_num(speed), threshold=speed_threshold, gap=smoothing)
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


def read_excel(*args, keep_index=False, **kwargs):
    '''
    A wrapper for opening Excel files using openpyxl
    :param args: passed to pandas reader
    :param keep_index: if True, keep the first column as index (Default: False)
    :param kwargs: passed to pandas reader
    :return: DataFrame
    '''
    fn = args[0]
    keep_index = False
    if fn.endswith('csv'):
        return pandas.read_csv(*args, **kwargs)
    df = pandas.read_excel(*args, **kwargs, engine='openpyxl', index_col=0)
    # index_col=0 makes the first column index. It's not accessible by .loc or by name.
    if keep_index:
        return df
    # if the input file was already saved by pandas, it has an unnamed index column. We don't want to keep that.
    drop_index = df.index.name == "Unnamed: 0"
    # reset_index will change the index column into a regular (named) column (unless it was unnamed, then drops it).
    return df.reset_index(drop=drop_index)


def ewma(trace, period=15):
    return numpy.array(pandas.DataFrame(numpy.nan_to_num(trace)).ewm(span=period).mean()[0])


def p_lookup(p):
    if numpy.isnan(p):
        return 'nan'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def path_to_list(path):
    parent_folders = []
    while path[-1] in ('//', '/', '\\'):
        path = path[:-1]
    while True:
        path, folder = os.path.split(path)
        if folder:
            parent_folders.append(folder)
        else:
            if path and path not in ('//', '/', '\\'):
                parent_folders.append(path)
            break
    parent_folders.reverse()
    return parent_folders


class completed_list:
    '''
    *deprecated, use DataSet*
    keep persistent track of jobs that are complete. usage:
    completed_prefix = completed_list(project_path + '_completed_prefixes.txt')
    for prefix in jobs:
        if prefix in completed_prefix:
            continue
        do stuff
        completed_prefix(prefix)
    '''

    def __init__(self, filehandle):
        self.filehandle = filehandle
        if os.path.exists(filehandle):
            with open(filehandle, 'r') as f:
                completed_prefix = [s.strip() for s in f.readlines()]
        else:
            completed_prefix = []
        self.list = completed_prefix

    def write(self, prefix):
        with open(self.filehandle, 'a') as of:
            of.write(prefix + '\n')

    def __contains__(self, item):
        return item in self.list

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)


def ts(format=None):
    now = datetime.datetime.now().isoformat(timespec='seconds')
    if format is None:
        return now
    else:
        return now.replace(':', '')


def CI_linreg(x, y):
    # fit a curve to the data using a least squares 1st order polynomial fit
    z = numpy.polyfit(x, y, 1)
    p_y = z[0] * x + z[1]

    # calculate the y-error (residuals)
    y_err = y - p_y

    # create series of new test x-values to predict for
    stepsize = (numpy.max(x) - numpy.min(x)) / 100
    p_x = numpy.arange(numpy.min(x), numpy.max(x) + stepsize, stepsize)

    # now calculate confidence intervals for new test x-series
    mean_x = numpy.mean(x)  # mean of x
    n = len(x)  # number of samples in origional fit
    t = stats.t.ppf(1 - 0.025, n - 1)  # appropriate t value (where n=9, two tailed 95%)
    s_err = numpy.sum(numpy.power(y_err, 2))  # sum of the squares of the residuals

    confs = t * numpy.sqrt((s_err / (n - 2)) * (1.0 / n + (numpy.power((p_x - mean_x), 2) /
                                                           ((numpy.sum(numpy.power(x, 2))) - n * (
                                                               numpy.power(mean_x, 2))))))

    # now predict y based on test x-values
    p_y = z[0] * p_x + z[1]

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)
    return p_x, lower, upper


def touch_path(*args):
    '''
    Join all args and create folder if does not exist
    :param args:
    :return:string of the folder
    '''
    path = os.path.realpath(os.path.join(*args))
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'
