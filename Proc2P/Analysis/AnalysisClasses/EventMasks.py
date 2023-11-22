import numpy
from sklearn import cluster
from Proc2P.utils import read_excel
# from EyeTracking import clean_movement_map, clean_whisker_map, parse_triggers
# from SCA_reader import get_spikes, find_clusters
# from OEphys_reader import get_OEphys_events, get_tonepuff_events


# define functions to detect events for pulling mean traces. keep each version for future reuse and reference
# functions should take the session and window,
# and return the trigger list and pull mask (indices or none for each event and Dt)

def PhotoStimTrain(a, w):
    event_frames = numpy.load(a.get_file_with_suffix('_photostim_trains.npy'))
    return masks_from_list(a, w, event_frames)

def LFPSpikes(a, w):
    df = read_excel(a.get_file_with_suffix('_spiketimes.xlsx'))
    st = df['SpikeTimes(s)'].values
    event_frames = [a.timetoframe(x) for x in st]
    return nonoverlap_from_list(a, w, event_frames, decay=int(a.CF.fps*0.2), eps=int(a.CF.fps), exclude_move=True)

def Seizures(a, w):
    df = read_excel(a.get_file_with_suffix('_seizure_times.xlsx'))
    st = df['Sz.Start(s)'].values
    event_frames = [a.timetoframe(x) for x in st]
    return nonoverlap_from_list(a, w, event_frames, decay=int(a.CF.fps*0.2), eps=int(a.CF.fps), exclude_move=True)


def masks_from_list(a, w, event_list):
    '''return all masks from a custom event list'''
    events = numpy.array(event_list, dtype=numpy.int64)
    mask = numpy.empty((len(events), 2 * w))
    mask[:] = numpy.nan
    trim = min(100, w)
    for ri, rj in enumerate(events):
        current_frame = rj
        i0 = max(trim, rj - w)
        im = current_frame - i0
        i1 = min(rj + w, a.ca.frames - trim)
        w0 = im
        w1 = i1 - current_frame
        # set actual indices
        mask[ri, w - w0:w + w1] = numpy.arange(current_frame - w0, current_frame + w1)
    return events, mask


def nonoverlap_from_list(a, w, event_list, decay=6, eps=16, exclude_move=False, min_n=None,
                         clustloc='first', trace=None):
    '''return masks for a custom event list with no overlap allowed between events.
    decay: minimum gap between masks
     ignores changes in locomotion
     eps: replaces clusters with onset of firsts. retain all if 0.
     clustloc: 'first' - returns first member of a cluster, 'max': returns loc of max in the trace arg
     '''
    events = numpy.array(event_list, dtype=numpy.int64)
    if min_n is None:
        min_n = 2
        exc_single = False
    else:
        exc_single = True
    if eps:
        event_t = numpy.copy(events)
        if len(event_t) < min_n:
            events = event_t
        else:
            clustering = cluster.DBSCAN(eps=eps, min_samples=min_n).fit(event_t.reshape(-1, 1))
            single_event_indices = numpy.where(clustering.labels_ < 0)[0]
            any_clusters = clustering.labels_.max() > 0
            if exc_single:
                if any_clusters:
                    event_number = clustering.labels_.max() + (len(single_event_indices) > 0)
                    events = numpy.empty(event_number, dtype=numpy.int64)
                    for rci in range(clustering.labels_.max() + (
                            len(single_event_indices) > 0)):  # indices are different if there are zero noise events
                        current_cluster = numpy.where(clustering.labels_ == rci)[0]
                        rj = current_cluster[0]
                        events[rci] = event_t[rj]
                else:
                    events = []
            else:
                event_number = len(single_event_indices)
                if any_clusters:
                    event_number += clustering.labels_.max()
                events = numpy.empty(event_number, dtype=numpy.int64)
                ri, rci = 0, 0
                for ri, rj in enumerate(single_event_indices):
                    # try:
                    events[ri] = event_t[rj]
                    # except:
                    #     print(len(event_t), len(events), len(single_event_indices), clustering.labels_.max(), ri, rj)
                    #     assert False
                if any_clusters:
                    for rci in range(clustering.labels_.max() + (
                            len(single_event_indices) > 0)):  # indices are different if there are zero noise events
                        # get list of ripples in cluster
                        current_cluster = numpy.where(clustering.labels_ == rci)[0]
                        if clustloc == 'first':
                            rj = current_cluster[0]
                        elif clustloc == 'max':
                            rj = current_cluster[numpy.argmax(trace[event_t[current_cluster]])]
                        events[ri + rci] = event_t[rj]
    events.sort()
    if exclude_move:
        events = events[~a.pos.movement[events]]
    mask = numpy.zeros((len(events), 2 * w))
    mask[:] = numpy.nan
    # following algorithm taken and modified from single ripples
    for frame_index, current_frame in enumerate(events):
        # trim end to start of next
        if frame_index < len(events) - 1:
            last_frame = min(current_frame + w, max(events[frame_index + 1] - decay, current_frame + decay))
        else:
            last_frame = current_frame + w
        if frame_index > 0:
            first_frame = max(current_frame - w, min(events[frame_index - 1] + decay, current_frame - decay))
        else:
            first_frame = current_frame - w
        if first_frame > w and last_frame < a.ca.frames - w:
            w0 = current_frame - first_frame
            w1 = last_frame - current_frame
            # set actual indices
            mask[frame_index, w - w0:w + w1] = numpy.arange(first_frame, last_frame)
    return events, mask


def StopResponse(a, w):
    starts, stops = a.startstop()
    return masks_from_list(a, w, stops)


def mask_from_list_nosession(w, event_list, trace_len, decay=6, eps=16, min_n=None,
                             clustloc='first', trace=None):
    '''return masks for a custom event list with no overlap allowed between events.
     decay: minimum gap between masks
     eps: replaces clusters with onset of firsts. retain all if 0.
     clustloc: 'first' - returns first member of a cluster, 'max': returns loc of max in the trace arg
     '''
    events = numpy.array(event_list, dtype=numpy.int64)
    if min_n is None:
        min_n = 2
        exc_single = False
    else:
        exc_single = True
    if eps:
        event_t = numpy.copy(events)
        if len(event_t) < min_n:
            events = event_t
        else:
            clustering = cluster.DBSCAN(eps=eps, min_samples=min_n).fit(event_t.reshape(-1, 1))
            single_event_indices = numpy.where(clustering.labels_ < 0)[0]
            any_clusters = clustering.labels_.max() > 0
            if exc_single:
                if any_clusters:
                    event_number = clustering.labels_.max() + (len(single_event_indices) > 0)
                    events = numpy.empty(event_number, dtype=numpy.int64)
                    for rci in range(clustering.labels_.max() + (
                            len(single_event_indices) > 0)):  # indices are different if there are zero noise events
                        current_cluster = numpy.where(clustering.labels_ == rci)[0]
                        rj = current_cluster[0]
                        events[rci] = event_t[rj]
                else:
                    events = []
            else:
                event_number = len(single_event_indices)
                if any_clusters:
                    event_number += clustering.labels_.max()
                events = numpy.empty(event_number, dtype=numpy.int64)
                ri, rci = 0, 0
                for ri, rj in enumerate(single_event_indices):
                    # try:
                    events[ri] = event_t[rj]
                    # except:
                    #     print(len(event_t), len(events), len(single_event_indices), clustering.labels_.max(), ri, rj)
                    #     assert False
                if any_clusters:
                    for rci in range(clustering.labels_.max() + (
                            len(single_event_indices) > 0)):  # indices are different if there are zero noise events
                        # get list of ripples in cluster
                        current_cluster = numpy.where(clustering.labels_ == rci)[0]
                        if clustloc == 'first':
                            rj = current_cluster[0]
                        elif clustloc == 'max':
                            rj = current_cluster[numpy.argmax(trace[event_t[current_cluster]])]
                        events[ri + rci] = event_t[rj]
    events.sort()
    # if exclude_move:
    #     events = events[~a.pos.gapless[events]]
    mask = numpy.zeros((len(events), 2 * w))
    mask[:] = numpy.nan
    # following algorithm taken and modified from single ripples
    for frame_index, current_frame in enumerate(events):
        # trim end to start of next
        if frame_index < len(events) - 1:
            last_frame = min(current_frame + w, max(events[frame_index + 1] - decay, current_frame + decay))
        else:
            last_frame = current_frame + w
        if frame_index > 0:
            first_frame = max(current_frame - w, min(events[frame_index - 1] + decay, current_frame - decay))
        else:
            first_frame = current_frame - w
        if first_frame > w and last_frame < trace_len - w:
            w0 = current_frame - first_frame
            w1 = last_frame - current_frame
            # set actual indices
            mask[frame_index, w - w0:w + w1] = numpy.arange(first_frame, last_frame)
    return events, mask