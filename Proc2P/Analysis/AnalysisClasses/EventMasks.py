import numpy
from sklearn import cluster


from Proc2P.utils import read_excel
from sklearn.cluster import dbscan
# from EyeTracking import clean_movement_map, clean_whisker_map, parse_triggers
# from SCA_reader import get_spikes, find_clusters
# from OEphys_reader import get_OEphys_events, get_tonepuff_events


# define functions to detect events for pulling mean traces. keep each version for future reuse and reference
# functions should take the session and window,
# and return the trigger list and pull mask (indices or none for each event and Dt)

def PhotoStimTrain(a, w):
    event_frames = numpy.load(a.get_file_with_suffix('_photostim_trains.npy'))
    return masks_from_list(a, w, event_frames)

def PhotoStimPulse(a, w, exclude_move=True, exclude_start_seconds=0, mask_stim=None, filter_by_sz=None):
    if a.opto is None:
        event_frames = []
    else:
        stims = numpy.where(a.opto)[0]
        clustering = dbscan(stims.reshape(-1, 1), eps=1, min_samples=1)
        n_trains = clustering[1].max() + 1
        event_frames = numpy.ones(n_trains, dtype='int32') * -1  # start frame
        for i in range(n_trains):
            event_frames[i] = stims[numpy.searchsorted(clustering[1], i)]
        if exclude_move and len(event_frames):
            event_frames = event_frames[~a.pos.movement[event_frames]]
        if exclude_start_seconds is not None:
            tmin = exclude_start_seconds * a.fps
            event_frames = event_frames[event_frames > tmin]
    if filter_by_sz is not None:
        event_frames = Filter_events_by_sz(a, event_frames, filter_by_sz)
    event, mask = masks_from_list(a, w, event_frames)
    if mask_stim is not None:
        mask[:, w:w+int(mask_stim)+1] = numpy.nan
    return event, mask


def Filter_events_by_sz(a, event_frames, filter_by_sz, margin=0.5):
    if filter_by_sz is not None:
        a.map_seizuretimes()
        #create a binary sz signal, extending sz times on both ends by a margin
        is_sz = numpy.zeros(a.ca.frames, dtype='bool')
        sz_margin = int(margin*a.fps)
        for sz_start, sz_stop in zip(*a.sztimes[0]):
            i0 = max(0, sz_start-sz_margin)
            i1 = min(sz_stop+sz_margin, a.ca.frames)
            is_sz[i0:i1] = True
        #use only events that are in/out sz: filter True will return "in", false "out"
        return [t for t in event_frames if is_sz[t] == bool(filter_by_sz)]

def LFPSpikes(a, w, ch=0, from_session=True, exclude_move=True, filter_by_sz=None):
    if not from_session:
        df = read_excel(a.get_file_with_suffix(f'_Ch{ch+1}_spiketimes.xlsx'))
        st = df['SpikeTimes(s)'].values
        event_frames = [a.timetoframe(x) for x in st]
    else:
        a.map_spiketimes()
        event_frames = a.spiketimes[ch]
    if filter_by_sz is not None:
        event_frames = Filter_events_by_sz(a, event_frames, filter_by_sz)
    return nonoverlap_from_list(a, w, event_frames, decay=int(a.fps*0.2), eps=int(a.fps),
                                exclude_move=exclude_move)

def Seizures(a, w, ch=1):
    df = read_excel(a.get_file_with_suffix(f'_ch{ch}_seizure_times.xlsx'))
    st = df['Sz.Start(s)'].values
    event_frames = [a.timetoframe(x) for x in st]
    return nonoverlap_from_list(a, w, event_frames, decay=int(a.fps*0.2), eps=int(a.fps), exclude_move=True)

def curated_sz_times(a, w, ch=0): #ch is 0 indexed
    a.map_seizuretimes()
    event_frames = a.sztimes[ch][0]
    return nonoverlap_from_list(a, w, event_frames, decay=int(a.fps*0.5), eps=int(a.fps), exclude_move=False)


def masks_from_list(a, w, event_list, exclude_movement=False):
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
    if exclude_movement:
        speed_mask = numpy.nan_to_num(mask.astype('int16'))
        mov_mask = a.pos.movement[speed_mask]
        mov_mask[speed_mask == 0] = 0
        mask[mov_mask] = numpy.nan
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


def StopResponse(a, w, **kwargs):
    starts, stops = a.startstop(**kwargs)
    return masks_from_list(a, w, stops)

def StartResponse(a, w, **kwargs):
    starts, stops = a.startstop(**kwargs)
    return masks_from_list(a, w, starts)


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

def all_ripples_immobility(a, w, cluster_eps = 1):
    '''After adding all single ripples, add ripple clusters by cluster
    Ripples are not overlapping. when they are close, split by an empirically set decay (we cut 1/4 sev after ripple)
    '''
    event_t = []
    for ripple_frame in a.ripple_frames:
        if 100 + w < ripple_frame < a.ca.frames - 100 - w:
            if not a.pos.movement[ripple_frame]:
                event_t.append(ripple_frame)  # used for clustering only
    event_t = numpy.array(sorted(event_t))
    clustering = cluster.DBSCAN(eps=int(a.fps * cluster_eps), min_samples=2).fit(event_t.reshape(-1, 1))
    single_ripple_indices = numpy.where(clustering.labels_ < 0)[0]
    filtered_event_t = [event_t[ri] for ri in single_ripple_indices]
    for rci in range(clustering.labels_.max() + 1):
        current_cluster = numpy.where(clustering.labels_ == rci)[0]
        filtered_event_t.append(event_t[current_cluster[0]])
    event_t = sorted(filtered_event_t)
    decay_w = int(a.fps/4)
    gap = int(a.fps)
    event_number = len(event_t)
    mask = numpy.zeros((event_number, 2 * w))
    events = numpy.empty(event_number, dtype=numpy.int64)
    mask[:] = numpy.nan
    trim = min(100, w)
    for ri, current_frame in enumerate(event_t):
        if ri < event_number - 1:
            last_frame = min(current_frame + w, event_t[ri + 1] - decay_w)
        else:
            last_frame = current_frame + w
        if ri > 0:
            first_frame = max(current_frame - w, event_t[ri - 1] + decay_w)
        else:
            first_frame = current_frame - w
        i0 = max(trim, first_frame)
        im = current_frame - i0
        i1 = min(last_frame, a.ca.frames - trim)
        try:
            m = a.pos.movement[i0:i1]
        except:
            print(ri, current_frame, w, event_number)
            assert False
        # check if mouse was still all the time during the pre period and find last movement if not.
        if numpy.any(m[:im - gap]):
            w0 = gap + numpy.argmax(m[:im - gap][::-1])
        else:
            w0 = im
        # find first movement frame after event
        if numpy.any(m[im + gap:]):
            w1 = numpy.argmax(m[im + gap:]) + gap
        else:
            w1 = i1 - current_frame
        # set actual indices. in case of overlap,skip this.
        if w0 > w:
            continue
        try:
            mask[ri, w - w0:w + w1] = numpy.arange(current_frame - w0, current_frame + w1)
        except:
            print(ri, current_frame, w, w0, w1, mask.shape)
            assert False
        events[ri] = current_frame

    return events, mask


if __name__ == '__main__':
    processed_path = 'D:\Shares\Data\_Processed/2P\PVTot/'
    prefix = 'PVTot7_2024-02-07_lfpOpto_178'
    from Proc2P.Analysis.ImagingSession import ImagingSession
    a = ImagingSession(processed_path, prefix, tag='IN')
    event, mask = LFPSpikes(a, 150)
    print(len(event))