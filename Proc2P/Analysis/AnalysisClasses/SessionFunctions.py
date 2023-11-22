from _LNCM_Analysis import *
from Bruker.AnalysisClasses import PhotoStim, EventMasks, PSTH
from Bruker.ImagingSession import ImagingSession

'''
To be used for higher level analysis.
This can use classes from both Core and AnalysisClasses
'''

def get_opto_responses(session:ImagingSession, isi=10, pre_s = (-1, 0),  post_s = (0, 5), param_key='rel'):
    '''
    Return the response of each cell to photostimulation trains
    :param session: instance of ImagingSession
    :param isi: inter-stim interval (within trains) in frames
    :param pre_s: baseline window (from train start) is s
    :param post_s: response window (from train start) is s
    :return: array of responses (shape ncells)
    '''

    # get pulse trains
    t = PhotoStim.PhotoStim(session)
    train_starts, train_ints = t.get_trains(isi=isi)
    # get response of each cell to each stim
    event, mask = EventMasks.masks_from_list(session, w, train_starts)
    resps = PSTH.pull_session_with_mask(session, mask, param_key=param_key)
    # mean response
    pre_slice = slice(*[int(w + x * fps) for x in pre_s])
    post_slice = slice(*[int(w + x * fps) for x in post_s])
    pre = numpy.nanmean(resps[:, pre_slice], axis=1)
    post = numpy.nanmean(resps[:, post_slice], axis=1)
    return train_starts, train_ints, post - pre