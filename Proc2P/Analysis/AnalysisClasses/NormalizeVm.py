import os.path

from scipy.ndimage import gaussian_filter
from statsmodels.tsa.arima.model import ARIMA
import numpy
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.utils import gapless, outlier_indices, lprint
from multiprocessing import Process

def normalize_trace(session: ImagingSession, **kwargs):
    '''
    :param session: an initialized ImagingSession
    :return: nothing, saves traces in CaTrace
    '''
    fps = session.fps
    look_back_frames = int(fps / 0.3)  # time frame for finding the range, 1/firing rate
    gauss_sigma = 3  # seconds
    # filter_order = (1, 1, 1)  # for arima. dorpna not compatible with all models, this works fine but slows signals.
    sliding_step = int(0.1 * fps) #resolution of calculating the range.

    norm_traces = numpy.empty(session.ca.trace.shape)
    norm_traces[:] = numpy.nan

    for c in range(session.ca.cells):
        lprint(None, 'Processing Vm', session.prefix, f'c{c} of {session.ca.cells}')
        # do arima filtfilt for estimating baseline:    Y = trace
        Y = session.ca.trace[c]
        y = numpy.copy(Y)
        # mask outliers and the frames before and after them, as well as running
        exclude_move = gapless(session.pos.speed, int(fps), threshold=1, expand=int(fps))
        diff_indices = [x for x in outlier_indices(numpy.diff(Y)) if x < (len(Y) - 2)]
        y[diff_indices] = numpy.nan
        y[exclude_move] = numpy.nan
        y[[x + 1 for x in diff_indices]] = numpy.nan
        y[[x + 2 for x in diff_indices]] = numpy.nan
        y[:int(fps)] = numpy.nan
        y[-int(fps):] = numpy.nan
        # exclude seizures
        if 'exclude_seizures' in kwargs and kwargs['exclude_seizures']:
            ir = session.map_instrate()
            exclude_sz = gapless(ir, int(fps), threshold=1.5, expand=int(fps))
            if c == 0:
                exc_rate = numpy.count_nonzero(~exclude_move[exclude_sz]) / len(y) * 100
                lprint(None, f'Vm normalization {session.prefix} with "exclude_seizures" option, removing {round(exc_rate)}%')
            y[exclude_sz] = numpy.nan
        baseline = arima_filtfilt(y)
        slo_baseline = gaussian_filter(baseline, int(fps * gauss_sigma))

        corr = Y - slo_baseline

        # find the 1st and 99th percentile of the range looking back
        locq = numpy.empty((2, len(y)))
        locq[:] = numpy.nan
        # nan-ize the corrected for estimating ranges
        cy = numpy.copy(corr)
        cy[numpy.isnan(y)] = numpy.nan
        diffs = numpy.diff(numpy.nan_to_num(cy))
        sds = 3 * numpy.nanstd(numpy.diff(cy))
        # exclude frames with big pos/neg diffs
        cy[numpy.where(diffs > sds)[0] - 1] = numpy.nan
        cy[numpy.where(diffs < -sds)[0] - 1] = numpy.nan

        #look up ranges in steps
        xvals = []
        for i in range(look_back_frames, len(y), sliding_step):
            lookup_slice = slice(i - look_back_frames, i)
            if numpy.count_nonzero(numpy.logical_not(numpy.isnan(cy[lookup_slice]))) > fps:
                locq[:, i] = numpy.nanpercentile(cy[lookup_slice], [1, 99])
                xvals.append(i)

        # filtfilt the locmax:
        fitg = numpy.empty(locq.shape)
        predx = numpy.arange(look_back_frames, len(y))
        fitg[:] = numpy.nan
        for fi in (0, 1):
            rY = locq[fi][xvals]
            if len(rY) < 5:
                continue
            fitq = arima_filtfilt(rY)
            # interpolate and gaussian filter
            fitg[fi, predx] = gaussian_filter(numpy.interp(predx, xvals, fitq), sigma=int(fps * gauss_sigma))

        # normalize the trace
        v0 = fitg[0, predx]
        v1 = fitg[1, predx]
        v = corr[predx]
        vrange = v1 - v0
        norm_traces[c, predx] = (v - v0) / vrange
    numpy.save(session.ca.pf + '//' + 'vm', norm_traces)

class Worker(Process):
    __name__ = 'Vm-Worker'
    def __init__(self, queue, res_queue):
        super(Worker, self).__init__()
        self.queue = queue
        self.res_queue = res_queue

    def run(self):
        for data in iter(self.queue.get, None):
            if type(data[-1])==dict:
                path, prefix, tag, overwrite, kwargs = data
            else:
                path, prefix, tag, overwrite = data
                kwargs = {}
            session = ImagingSession(path, prefix, tag, norip=True)
            if overwrite or not os.path.exists(os.path.join(session.ca.pf, 'vm.npy')):
                normalize_trace(session, **kwargs)
            self.res_queue.put(prefix)

def arima_filtfilt(y, filter_order=(1, 1, 1)):
    # filter
    ma_model = ARIMA(y, order=filter_order, missing='drop', )
    model_fit = ma_model.fit()
    result = model_fit.predict()
    result[0] = y[0]
    # reverse filter
    ma_model = ARIMA(y[::-1], order=filter_order, missing='drop', )
    model_fit = ma_model.fit()
    result_rev = model_fit.predict()
    result_rev[0] = y[-1]
    # average
    return (result + result_rev[::-1]) / 2
