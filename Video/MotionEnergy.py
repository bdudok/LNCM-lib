"""Motion energy trace of long videos (vEEG recordings, 2P face camera), computed on CPU.

Same measure as PullMotion.pull_motion_energy: a rolling background (alpha=0.3) is subtracted from each frame and the
absolute difference is smoothed (beta=0.9); the trace is the mean of that map. Differences from pull_motion_energy:
- uses the full frame (vEEG) or the saved motion crop (2P face camera, which shows only part of the face; without a
  crop the trace isn't computed), downsampled 4x by area averaging, in float32 with OpenCV operations
- reads the luma plane from the decoder instead of converting to BGR and back to gray
- splits the video into chunks that are processed in parallel threads (OpenCV releases the GIL). Each chunk starts
  WARMUP frames early, so the rolling averages are settled (0.7 ** 30 < 1e-4) and the result matches a single pass.
Values are on a different scale than traces from pull_motion_energy (luma range, more downsampling).

Output of the pull_ functions: <prefix>_motion_energy.npy in the processed folder, shape (2, n_frames):
row 0 is the time of each video frame in EEG seconds (from the start of the EDF or of the Bruker voltage recording),
row 1 is the motion energy.
The GUIs start pull_in_background when the file is missing. Video/MotionWatcher.py pulls new vEEG recordings ahead of
time; for 2P sessions, FaceVideo.CropVideo starts it when the motion crop is saved.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy

from LFP.Pinnacle import ReadEDF
from LFP.Setup import Setup
from Proc2P.Bruker.PreProc import SessionInfo
from Proc2P.utils import lprint
from Video import LoadVEEG

ALPHA = 0.3  # background update rate
BETA = 0.9  # weight of the current frame in the smoothed difference
DOWNSAMPLE = 4
WARMUP = 30  # frames read before each chunk
CHUNK = 1800  # frames per parallel job
N_WORKERS = min(8, os.cpu_count())
SUFFIX = '_motion_energy.npy'
LOCK_MAX_AGE_S = 3 * 3600  # an older lock file is left over from a process that was killed


def _chunk_trace(fn, start, stop, decoder_threads, crop=None):
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
        if crop is not None:
            frame = frame[crop[2]:crop[3], crop[0]:crop[1]]
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


def motion_energy(fn, start=0, stop=None, n_workers=N_WORKERS, crop=None):
    """Motion energy of each frame of a video from start to stop (frame indices; stop=None: to the end).
    crop: [x0, x1, y0, y1] in pixels, None: full frame.
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
            parts = list(ex.map(lambda job: _chunk_trace(fn, *job, decoder_threads, crop), jobs))
    finally:
        cv2.utils.logging.setLogLevel(log_level)
    trace = numpy.concatenate(parts)
    if start == 0 and len(trace) > 1:
        trace[0] = trace[1]  # no background yet in the first frame
    return trace


def trace_fn(setup, path, prefix):
    """Motion energy file. setup: Setup.PINNACLE (path: EDF folder) or Setup.LNCM (path: parent of the session folder)"""
    if setup == Setup.PINNACLE:
        return Path(path) / (prefix + SUFFIX)
    elif setup == Setup.LNCM:
        return Path(path) / prefix / (prefix + SUFFIX)
    raise ValueError(f'Unknown setup: {setup}')


def load_trace(setup, path, prefix):
    """(time in EEG s, motion energy), or None if the file does not exist"""
    fn = trace_fn(setup, path, prefix)
    if not fn.exists():
        return None
    t, me = numpy.load(fn)
    return t, me


def _save(out, fn):
    """Write to a temporary file and rename it, so readers never see a partly written file"""
    tmp = fn.with_name(fn.stem + '.partial.npy')
    numpy.save(tmp, out)
    os.replace(tmp, fn)


def pull_veeg_motion(vmap, video_dir, out_fn, n_workers=N_WORKERS):
    """Motion energy of the video frames shown during an EEG file, saved with their EEG times.
    vmap: video map of the EEG (LoadVEEG.video_map or video_map_for_edf); video_dir: folder that the file names in the
    map are relative to (the EDF folder for maps from video_map_for_edf)."""
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
    _save(out, Path(out_fn))
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


def bruker_crop(procpath, prefix):
    """Motion crop of a 2P session, [x0, x1, y0, y1] in pixels, saved by FaceVideo.CropVideo in
    <session>/_face/_motion-crop.json. None if not saved."""
    fn = Path(procpath) / prefix / '_face' / '_motion-crop.json'
    if not fn.exists():
        return None
    x1, x2, y1, y2 = json.loads(fn.read_text())  # corners of the selection, in the order they were dragged
    return [int(min(x1, x2)), int(max(x1, x2)) + 1, int(min(y1, y2)), int(max(y1, y2)) + 1]


def pull_bruker_motion(procpath, prefix, n_workers=N_WORKERS):
    """Motion energy of the 2P session's camera in the motion crop, saved with frame times in EEG (voltage recording)
    seconds. Returns None (and saves nothing) if the session has no camera file, camera timing or motion crop."""
    found = bruker_frame_times(procpath, prefix)
    crop = bruker_crop(procpath, prefix)
    if found is None or crop is None:
        lprint(None, f'Motion energy: no camera file, camera timing or motion crop for {prefix}')
        return None
    cam_file, t = found
    while True:
        lprint(None, f'Motion energy: {cam_file.name}, crop {crop}')
        me = motion_energy(cam_file, n_workers=n_workers, crop=crop)
        new_crop = bruker_crop(procpath, prefix)
        if new_crop == crop:
            break
        crop = new_crop  # the crop was saved again during the computation
        if crop is None:
            return None
    if len(me) != len(t):
        raise ValueError(f'{prefix}: {len(me)} video frames, {len(t)} camera timers')
    out = numpy.array([t, me])
    _save(out, trace_fn(Setup.LNCM, procpath, prefix))
    return out


def lock_fn(setup, path, prefix):
    return trace_fn(setup, path, prefix).with_suffix('.lock')


def is_running(setup, path, prefix):
    """True if another process is computing the trace of this recording"""
    fn = lock_fn(setup, path, prefix)
    return fn.exists() and time.time() - fn.stat().st_mtime < LOCK_MAX_AGE_S


def pull_motion(setup, path, prefix, video_dir=None, n_workers=N_WORKERS):
    """Compute and save the motion energy trace of a recording. Returns None if the recording has no video, or if
    another process is computing it.
    setup: Setup.PINNACLE (path: EDF folder) or Setup.LNCM (path: parent of the session folder)
    video_dir: folder of the vEEG recordings (Pinnacle), default: LoadVEEG.default_video_dir(path)"""
    setup = Setup(setup)
    if is_running(setup, path, prefix):
        lprint(None, f'Motion energy of {prefix} is being computed by another process')
        return None
    lock = lock_fn(setup, path, prefix)
    lock.touch()
    try:
        if setup == Setup.LNCM:
            return pull_bruker_motion(path, prefix, n_workers)
        video_dir = LoadVEEG.default_video_dir(path) if video_dir is None else video_dir
        edf = ReadEDF.EDF(path, prefix)
        vmap = LoadVEEG.video_map_for_edf(edf, video_dir, os.path.join(path, prefix + '_video_map.csv'))
        if vmap is None:
            lprint(None, f'Motion energy: no video matches {prefix}')
            return None
        return pull_veeg_motion(vmap, path, trace_fn(setup, path, prefix), n_workers)  # names relative to path
    finally:
        lock.unlink(missing_ok=True)


def pull_in_background(setup, path, prefix, video_dir=None):
    """Start pull_motion in a separate process that keeps running when the GUI is closed, unless the trace exists or
    is being computed. Errors are written to <prefix>_motion_energy.error.txt next to the trace file."""
    if trace_fn(setup, path, prefix).exists() or is_running(setup, path, prefix):
        return
    if setup == Setup.LNCM and bruker_crop(path, prefix) is None:
        return
    root = Path(__file__).resolve().parents[1]  # LNCM-lib
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), os.environ.get('PYTHONPATH', '')]))
    args = [sys.executable, '-m', 'Video.MotionEnergy', str(setup), str(path), prefix]
    if video_dir is not None:
        args.append(str(video_dir))
    # DETACHED_PROCESS (Windows only): no console, and closing the GUI's console doesn't stop it
    subprocess.Popen(args, env=env, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0))
    lprint(None, f'Computing the motion energy of {prefix} in the background')


if __name__ == '__main__':
    # python -m Video.MotionEnergy <setup> <path> <prefix> [<video_dir>], started by pull_in_background
    setup, path, prefix = sys.argv[1:4]
    try:
        pull_motion(setup, path, prefix, *sys.argv[4:5])
    except Exception:
        error_fn = trace_fn(Setup(setup), path, prefix).with_suffix('.error.txt')
        error_fn.write_text(traceback.format_exc())
        raise
