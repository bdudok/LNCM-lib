"""Motion energy trace of long videos (vEEG recordings, 2P face camera), computed on CPU.

Same measure as PullMotion.pull_motion_energy: a rolling background (alpha=0.3) is subtracted from each frame and the
absolute difference is smoothed (beta=0.9); the trace is the mean of that map. Differences from pull_motion_energy:
- uses the full frame, downsampled 4x by area averaging, in float32 with OpenCV operations (pull_motion_energy
  crops to the left h columns because the crop indices are swapped, and stretches that to the full width)
- reads the luma plane from the decoder instead of converting to BGR and back to gray
- splits the video into chunks that are processed in parallel threads (OpenCV releases the GIL). Each chunk starts
  WARMUP frames early, so the rolling averages are settled (0.7 ** 30 < 1e-4) and the result matches a single pass.
Values are on a different scale than traces from pull_motion_energy (luma range, more downsampling).

Output of the pull_ functions: <prefix>_motion_energy.npy in the processed folder, shape (2, n_frames):
row 0 is the time of each video frame in EEG seconds (from the start of the EDF or of the Bruker voltage recording),
row 1 is the motion energy.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy

from Proc2P.Bruker.PreProc import SessionInfo
from Proc2P.utils import lprint

ALPHA = 0.3  # background update rate
BETA = 0.9  # weight of the current frame in the smoothed difference
DOWNSAMPLE = 4
WARMUP = 30  # frames read before each chunk
CHUNK = 1800  # frames per parallel job
N_WORKERS = min(8, os.cpu_count())
SUFFIX = '_motion_energy.npy'


def _chunk_trace(fn, start, stop, decoder_threads):
    """Motion energy of frames start..stop-1 (stop=None: to the end of the file)"""
    cap = cv2.VideoCapture(str(fn), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, decoder_threads])
    # without conversion, yuv420p frames come as the luma plane (h, w)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    first = max(0, start - WARMUP)
    if first > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    values = []
    bg, mm, dsize = None, None, None
    i = first
    while stop is None or i < stop:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if dsize is None:
            dsize = (frame.shape[1] // DOWNSAMPLE, frame.shape[0] // DOWNSAMPLE)
        im = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA).astype(numpy.float32)
        if bg is None:
            bg = im
            mm = numpy.zeros_like(im)
        else:
            cv2.accumulateWeighted(im, bg, ALPHA)
        cv2.accumulateWeighted(cv2.absdiff(im, bg), mm, BETA)
        if i >= start:
            values.append(cv2.mean(mm)[0])
        i += 1
    cap.release()
    return numpy.array(values, dtype=numpy.float32)


def motion_energy(fn, start=0, stop=None, n_workers=N_WORKERS):
    """Motion energy of each frame of a video from start to stop (frame indices; stop=None: to the end).
    Returns a float32 array; it is shorter than stop - start if the file ends earlier."""
    if stop is None:
        cap = cv2.VideoCapture(str(fn))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # an estimate for some containers: the last chunk reads to the end
        cap.release()
    else:
        n = stop
    bounds = list(range(start, n, CHUNK)) or [start]
    jobs = [(b, bounds[i + 1] if i + 1 < len(bounds) else stop) for i, b in enumerate(bounds)]
    decoder_threads = 1 if n_workers > 1 else 0  # 0: decoder picks
    log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # 'unsupported format yuv420p' on every frame
    try:
        with ThreadPoolExecutor(n_workers) as ex:
            parts = list(ex.map(lambda job: _chunk_trace(fn, *job, decoder_threads), jobs))
    finally:
        cv2.utils.logging.setLogLevel(log_level)
    trace = numpy.concatenate(parts)
    if start == 0 and len(trace) > 1:
        trace[0] = trace[1]  # no background yet in the first frame
    return trace


def trace_fn(setup, path, prefix):
    """Motion energy file. setup: 'Pinnacle' (path: EDF folder) or 'LNCM' (path: parent of the session folder)"""
    if setup == 'Pinnacle':
        return Path(path) / (prefix + SUFFIX)
    elif setup == 'LNCM':
        return Path(path) / prefix / (prefix + SUFFIX)
    raise ValueError(f'Unknown setup: {setup}')


def load_trace(setup, path, prefix):
    """(time in EEG s, motion energy), or None if the file does not exist"""
    fn = trace_fn(setup, path, prefix)
    if not fn.exists():
        return None
    t, me = numpy.load(fn)
    return t, me


def pull_veeg_motion(vmap, video_dir, out_fn, n_workers=N_WORKERS):
    """Motion energy of the video frames shown during an EEG file, saved with their EEG times.
    vmap: video map of the EEG (LoadVEEG.video_map or load_video_map); video_dir: folder of the video files."""
    k = vmap['frame']
    valid = ~numpy.isnan(k)
    eeg_s = numpy.arange(len(k))[valid]
    k = k[valid]
    r0, r1 = int(numpy.floor(k.min())), int(numpy.ceil(k.max())) + 1  # frames of the joined recording
    first = [f['first_frame'] for f in vmap['files']] + [r1]
    rec_frames, traces = [], []
    for i, f in enumerate(vmap['files']):
        a, b = max(r0, first[i]), min(r1, first[i + 1])
        if a >= b:
            continue
        lprint(None, f'Motion energy: {f["name"]} frames {a - first[i]}-{b - first[i]}')
        tr = motion_energy(Path(video_dir) / f['name'], a - first[i], b - first[i], n_workers)
        rec_frames.append(numpy.arange(a, a + len(tr)))
        traces.append(tr)
    rec_frames = numpy.concatenate(rec_frames)
    t = numpy.interp(rec_frames, k, eeg_s)
    out = numpy.array([t, numpy.concatenate(traces)])
    numpy.save(out_fn, out)
    return out


def bruker_frame_times(procpath, prefix):
    """(camera file, time of each camera frame in EEG (voltage recording) seconds), or None if either is missing.
    Times come from the CamTimers file next to the video, with the first camera frame at the first scope frame as in
    PreProc.parse_cam. Without CamTimers, the time of the matched scope frame (cam_sync_frames) is used, nan where
    there is none."""
    session = Path(procpath) / prefix
    si = SessionInfo(session, prefix)
    si.load()
    if si['cam_file'] is None or not os.path.exists(si['cam_file']):
        return None
    cam_file = Path(si['cam_file'])
    timers_fn = cam_file.with_name(cam_file.stem + '_CamTimers.npy')
    sync_fn = session / (prefix + '_cam_sync_frames.npy')
    if timers_fn.exists():
        timers = numpy.load(timers_fn)  # (frame, ms)
        return cam_file, (timers[:, 1] - timers[0, 1]) / 1000
    if sync_fn.exists():
        scope_frames = numpy.load(sync_fn)
        t = numpy.load(session / (prefix + '_FrameTimes.npy'))[scope_frames, 0]  # relativeTime
        t[1:][scope_frames[1:] == 0] = numpy.nan  # 0: no scope frame matched
        return cam_file, t
    return None


def pull_bruker_motion(procpath, prefix, n_workers=N_WORKERS):
    """Motion energy of the 2P session's camera, saved with frame times in EEG (voltage recording) seconds.
    Returns None (and saves nothing) if the session has no camera file or camera timing."""
    found = bruker_frame_times(procpath, prefix)
    if found is None:
        lprint(None, f'Motion energy: no camera file or camera timing for {prefix}')
        return None
    cam_file, t = found
    lprint(None, f'Motion energy: {cam_file.name}')
    me = motion_energy(cam_file, n_workers=n_workers)
    if len(me) != len(t):
        raise ValueError(f'{prefix}: {len(me)} video frames, {len(t)} camera timers')
    out = numpy.array([t, me])
    numpy.save(trace_fn('LNCM', procpath, prefix), out)
    return out
