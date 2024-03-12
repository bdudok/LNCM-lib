from Proc2P.Analysis.AnalysisClasses import PhotoStim, EventMasks, PSTH
from Proc2P.Analysis.ImagingSession import ImagingSession
import numpy

'''
To be used for higher level analysis.
This can use classes from both Core and AnalysisClasses
'''

def get_opto_responses(session:ImagingSession, isi=10, pre_s = (-1, 0),  post_s = (0, 5), param_key='rel', thr=0.5,
                       w=None, fps=20):
    '''
    Return the response of each cell to photostimulation trains
    :param session: instance of ImagingSession
    :param isi: inter-stim interval (within trains) in frames
    :param pre_s: baseline window (from train start) is s
    :param post_s: response window (from train start) is s
    :param thr: stim intensity threshold for including in response analysis (all train starts are returned)
    :return: array of responses (shape ncells)
    '''

    if hasattr(session, 'fps'):
        fps = session.fps
    if w is None:
        w = int(max(10, post_s[1], -pre_s[0])*fps)
    # get pulse trains
    t = PhotoStim.PhotoStim(session)
    train_starts, train_ints = t.get_trains(isi=isi)
    # get response of each cell to each stim
    event, mask = EventMasks.masks_from_list(session, w, train_starts[train_ints>thr])
    resps = PSTH.pull_session_with_mask(session, mask, param_key=param_key)
    # mean response
    pre_slice = slice(*[int(w + x * fps) for x in pre_s])
    post_slice = slice(*[int(w + x * fps) for x in post_s])
    pre = numpy.nanmean(resps[:, pre_slice], axis=1)
    post = numpy.nanmean(resps[:, post_slice], axis=1)
    return train_starts, train_ints, post - pre

def stoprun_scores(session:ImagingSession, param='ntr', ret_loc='actual', mode='mean', loc_input=None):
    '''
    :return: scores shaped (cells, 2), 0 is stop, 1 is run
    '''
    param = session.getparam(param)
    # collect stop events. Criteria: last speed peak of 50 long run event followed by 150 gap.
    if loc_input is None:
        starts, stops = session.startstop(ret_loc=ret_loc)
    else:
        starts, stops = loc_input
    stoprun_scores = numpy.zeros((session.ca.cells, 2))
    for ti in range(len(stops)):
        start, stop = starts[ti], stops[ti]
        # calculate stop and run response score:
        if mode == 'mean':
            stoprun_scores[:, 0] += numpy.nanmean(param[:, stop:min(session.ca.frames, stop + 100)], axis=1) - \
                                    numpy.nanmean(param[:, start:stop], axis=1)
            stoprun_scores[:, 1] += numpy.nanmean(param[:, start:stop], axis=1) - \
                                    numpy.nanmean(param[:, max(0, start - 100):start], axis=1)
        if mode == 'max':
            stoprun_scores[:, 0] += numpy.nanmax(param[:, stop:min(session.ca.frames, stop + 100)], axis=1) - \
                                    numpy.nanmax(param[:, start:stop], axis=1)
            stoprun_scores[:, 1] += numpy.nanmax(param[:, start:stop], axis=1) - \
                                    numpy.nanmax(param[:, max(0, start - 100):start], axis=1)
    stoprun_scores /= len(stops)
    return stoprun_scores