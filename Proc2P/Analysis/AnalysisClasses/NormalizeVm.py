import os.path

from scipy.ndimage import gaussian_filter
from statsmodels.tsa.arima.model import ARIMA
import numpy
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.utils import gapless, outlier_indices, lprint
from multiprocessing import Process
from dataclasses import dataclass

@dataclass
class PullVmConfig:
    #fields to be configurable from GUI
    gauss_sigma: float = 3 # resolution of calculating the range (s).
    monotonic: str | None = 'up' #if 'up' (such as bleaching in inverted gevi trace): baseline is increasing throughout the recording
    percentile: float = 1 #max and min of the range defined as percentile of values. 1 is good for control, 5 for epileptic
    exclude_move: bool = False # if true, periods with running are excluded from fitting the models (but are predicted).
    exclude_seizures_flag: bool = False #if true, periods marked as seizures are excluded from fitting the models (but are predicted).
    overwrite: bool = False #if False, skips when output exists

#{'exclude_seizures_flag': False, "monotonic": 'up', "percentile": 1, "exclude_move": False}

def normalize_trace(session: ImagingSession, cells=None, save=True,
                    gauss_sigma=3, monotonic=None, percentile=1, exclude_move=True,
                    exclude_seizures_flag=False, exclude_sz=None,
                    keep_traces=False):
    '''
    :param session: an initialized ImagingSession
    :return: nothing, saves traces in CaTrace
    :keywords
        monotonic: if 'up' (such as bleaching in inverted gevi trace): baseline is increasing throughout the recording
            can be less accurate than default polynom, but reduces the number of cells that need to be excluded due to
            poor baseline fitting
        return_for_test: (bool) instead of saving, return the results for inspection
        exclude_seizures: (bool)periods marked as seizures are excluded from fitting the models (but are predicted).
        cell: (int) if specified, only compute that cell
    '''
    fps = session.fps
    look_back_frames = int(fps / 0.3)  # time frame for finding the range, 1/firing rate
    # filter_order = (1, 1, 1)  # for arima. dorpna not compatible with all models, this works fine but slows signals.
    sliding_step = int(gauss_sigma * fps / 3)  # resolution of calculating the range.

    norm_traces = numpy.empty(session.ca.trace.shape)
    norm_traces[:] = numpy.nan
    trace_cache = {}

    if exclude_move:
        exclude_move = gapless(session.pos.speed, int(fps), threshold=1, expand=int(fps))
    else:
        exclude_move = numpy.zeros(session.ca.frames, dtype='bool')  # we don't want to exclude cells if mouse is running much

    if exclude_seizures_flag:
        if exclude_sz is None:
            exclude_sz = gapless(session.map_instrate(), int(fps), threshold=1.5, expand=int(fps))
        exc_rate = numpy.count_nonzero(~exclude_move[exclude_sz]) / session.ca.frames * 100
        lprint(None, f'Vm normalization {session.prefix} with "exclude_seizures" option, removing {round(exc_rate)}%')
    else:
        exclude_sz = numpy.zeros(session.ca.frames, 'bool')

    if cells is None:
        cells = range(session.ca.cells)

    for c in cells:
        lprint(None, 'Processing Vm', session.prefix, f'c{c} of {session.ca.cells}')
        # do arima filtfilt for estimating baseline:    Y = trace
        Y = session.ca.trace[c]
        y = numpy.copy(Y)
        # mask outliers and the frames before and after them, as well as running
        diff_indices = [x for x in outlier_indices(numpy.diff(Y)) if x < (len(Y) - 2)]
        y[diff_indices] = numpy.nan
        y[exclude_move] = numpy.nan
        y[[x + 1 for x in diff_indices]] = numpy.nan
        y[[x + 2 for x in diff_indices]] = numpy.nan
        y[:int(fps*2)] = numpy.nan
        y[-int(fps*2):] = numpy.nan
        # exclude seizures
        if exclude_seizures_flag:
            y[exclude_sz] = numpy.nan
        baseline = arima_filtfilt(y)
        slo_baseline = gaussian_filter(baseline, int(fps * gauss_sigma))
        if monotonic is not None:
            if monotonic == 'up':
                slo_baseline = (slo_baseline + numpy.maximum.accumulate(slo_baseline)) / 2
            elif monotonic == 'down':
                slo_baseline = (slo_baseline + numpy.minimum.accumulate(slo_baseline)) / 2
            else:
                raise ValueError('Monotonic should be None, "up" or "down"')

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

        # look up ranges in steps
        xvals = []
        for i in range(look_back_frames, len(y), sliding_step):
            lookup_slice = slice(i - look_back_frames, i)
            if numpy.count_nonzero(numpy.logical_not(numpy.isnan(cy[lookup_slice]))) > fps:
                locq[:, i] = numpy.nanpercentile(cy[lookup_slice], [percentile, 100 - percentile])
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

        if keep_traces:
            trace_cache[c] = [predx, norm_traces[c], slo_baseline, corr, fitg, exclude_move, exclude_sz, ]
    if save:
        numpy.save(session.ca.pf + '//' + 'vm', norm_traces)
    if keep_traces:
        return trace_cache


class Worker(Process):
    __name__ = 'Vm-Worker'

    def __init__(self, queue, res_queue, n=0):
        super(Worker, self).__init__()
        self.queue = queue
        self.res_queue = res_queue
        self.n = n

    def run(self):
        for data in iter(self.queue.get, None):
            if type(data[-1]) == dict:
                path, prefix, tag, overwrite, kwargs = data
            else:
                path, prefix, tag, overwrite = data
                kwargs = {}
            session = ImagingSession(path, prefix, tag, norip=True)
            if overwrite or not os.path.exists(os.path.join(session.ca.pf, 'vm.npy')):
                normalize_trace(session, **kwargs)
            self.res_queue.put((self.__name__ + str(self.n), prefix))


def arima_filtfilt(y, filter_order=(1, 1, 1)):
    filtfilt = numpy.empty((2, len(y)))
    # filter
    ma_model = ARIMA(y, order=filter_order, missing='drop', )
    model_fit = ma_model.fit()
    filtfilt[0] = model_fit.predict()
    # reverse filter
    ma_model = ARIMA(y[::-1], order=filter_order, missing='drop', )
    model_fit = ma_model.fit()
    filtfilt[1] = model_fit.predict()[::-1]
    # average
    return numpy.nanmean(filtfilt, axis=0)
