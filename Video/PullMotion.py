import cv2
from matplotlib import pyplot as plt
import numpy
from Proc2P.utils import lprint


def crop_hash(crop):
    '''a unique number for each crop setting'''
    return int(sum([abs(hash(x)/4) for x in crop]) / 10 ** 10)

def pull_motion_energy(vid_fn, output_fn, crop=None, keep_map=False, save_preview=True):
    '''
    Computes the motion enerhy (optical flow) in each frame
    Args:
        vid_fn: handle (full path, filename and extension) of the avi file
        output_fn: handle (.npy) of the output files, e.g. session.face_path + '_motion_energy.npy'
        crop: [x0, x1, y0, y1] of an area, relative to the raw movie.
        keep_map: if True, saves the full map, else just the average of each frame.
         a hash of the crop will be added to the name
        save_preview: if True, saves an image
    Returns: the trace
    '''
    pw_n = 20 # preview frame (avoid sing 1st frame as light may be off)

    # read input
    im = cv2.VideoCapture(vid_fn)
    ret, frame = im.read()
    h, w = frame.shape[:2]
    n_frames = int(im.get(cv2.CAP_PROP_FRAME_COUNT))

    #check crop
    if crop is None:
        crop = [0, w, 0, h]
        hw = int(w / 2)
        hh = int(h / 2)
        crop_id = 'full'
    else:
        crop_id = crop_hash(crop)
        crop = numpy.array(crop).astype('int')
        crop[[1, 3]] += 1
        hw = int((crop[1] - crop[0]) / 2)
        hh = int((crop[3] - crop[2]) / 2)
    mm_fn = output_fn[:-4] + f'_map_{crop_id}.npy'

    # init output
    mm_trace = numpy.empty(n_frames)
    if keep_map:
        mm_map = numpy.empty((n_frames, hw, hh), dtype=frame.dtype)

    pw_data = [[], []]
    n = 0
    lprint(None, f'Reading {n_frames} frames of {w} * {h} video')
    assert crop[1] <= w and crop[3] <= h, str(crop) + str(frame.shape)
    while frame is not None:
        IM = cv2.cvtColor(frame[crop[0]:crop[1], crop[2]:crop[3]], cv2.COLOR_BGR2GRAY)
        # compute mm
        # downsample IM
        IM = cv2.resize(IM, (hw, hh), interpolation=cv2.INTER_AREA).astype('float')
        # keep rolling background
        alpha = 0.3
        if n < 1:
            BG = IM
        else:
            BG = alpha * IM + (1 - alpha) * BG
        # rolling absdiff
        D = numpy.abs(IM - BG)
        beta = 0.9
        if n < 1:
            MM = D
        else:
            MM = beta * D + (1 - beta) * MM
        if keep_map:
            mm_map[n] = MM.transpose()
        mm_trace[n] = MM.mean()
        # keep frames of preview
        if n < pw_n:
            pw_data[0].append(numpy.copy(IM))
            pw_data[1].append(numpy.copy(MM))

        n += 1
        ret, frame = im.read()
    mm_trace[0] = mm_trace[1]
    numpy.save(output_fn, mm_trace)
    if keep_map:
        mm_map[0] = mm_map[1]
        lprint(None, f'Saving motion map to {mm_fn}')
        numpy.save(mm_fn, mm_map)

    if save_preview:
        # gamma = 0.33
        # output_gamma = numpy.array([((i / 255.0) ** gamma) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
        fig, ax = plt.subplots(nrows=2, figsize=(12, 9), sharey=True, sharex=True)
        for ca in ax.flat:
            ca.spines['right'].set_visible(False)
            ca.spines['top'].set_visible(False)
            ca.spines['bottom'].set_visible(False)
            ca.spines['left'].set_visible(False)
            ca.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
            ca.tick_params(axis='y', which='both', right='off', left='off', labelleft='off')
            ca.set_xticks([])
            ca.set_yticks([])
        for ci, (ca, cmap) in enumerate(zip(ax, ('Grays', 'plasma'))):
            Y = numpy.array(pw_data[ci])
            ca.imshow(Y.mean(axis=0), cmap=cmap)
        plt.tight_layout()
        fig.savefig(mm_fn.replace('.npy', '.png'), dpi=300)
        plt.close()

    return mm_trace

def recrop_motion_map(output_fn, recrop, orig_crop, crop_tag):
    '''
    measures motion energy in a cropped region of the map
    Args:
        output_fn: handle (.npy) of the output files, e.g. session.face_path + '_motion_energy.npy'
        orig_crop: [x0, x1, y0, y1] used to export the original motion map from which the peaks are
    saves to file
    '''
    crop_id = crop_hash(orig_crop)
    crop = numpy.array(recrop).astype('int')
    crop[[1, 3]] += 1
    mm = numpy.load(output_fn[:-4] + f'_map_{crop_id}.npy')
    assert crop[1] <= mm.shape[1] and crop[3] <= mm.shape[2]
    mm_trace = mm[:, crop[0]:crop[1], crop[2]:crop[3]].mean(axis=(1, 2))
    fn = output_fn[:-4] + f'_recrop_{crop_tag}.npy'
    numpy.save(fn, mm_trace)
    return mm_trace

def peak_averaged_motion_map(output_fn, peaks, orig_crop=None, time_constant=30):
    '''
    Displays the movement map average on included peaks
    Args:
        output_fn: handle (.npy) of the output files, e.g. session.face_path + '_motion_energy.npy'
        orig_crop: [x0, x1, y0, y1] used to export the original motion map from which the peaks are
    Returns: the image
    '''
    #check crop
    if orig_crop is None:
        crop_id = 'full'
    else:
        crop_id = crop_hash(orig_crop)
    mm = numpy.load(output_fn[:-4] + f'_map_{crop_id}.npy', mmap_mode='r')
    peaks = peaks[peaks > time_constant]
    return numpy.maximum(0, numpy.mean(mm[peaks], axis=0) - numpy.mean(mm[peaks-int(time_constant)], axis=0))


def floor_trace(trace, t1):
    frames = len(trace)
    smw = numpy.empty(frames)
    for t in range(frames):
        ti0 = max(0, int(t - t1 * 0.5))
        ti1 = min(frames, int(t + t1 * 0.5) + 1)
        smw[t] = numpy.mean(trace[ti0:ti1])
    bsl = numpy.empty(frames)
    t2 = int(t1 * 50)
    for t in range(frames):
        ti0, ti1 = max(0, t - t2), min(t, frames)
        if ti0 < ti1:
            minv = numpy.min(smw[ti0:ti1])
        else:
            minv = smw[ti0]
        bsl[t] = minv
    rel = numpy.maximum(0, (trace - bsl) / bsl)
    return rel
