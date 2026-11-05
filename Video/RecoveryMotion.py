"""Files and motion energy of post-surgery recovery videos (MonitorCam, Pinnacle-vEEG repo), for the recovery review
GUI (Video/RecoveryReviewGUI.py).

The files of a video <_RawData>/MonitorCam/<subfolders>/<stem>.mkv are in the same subfolders under _Processed
(processed_dir), named after the video:
    <stem>_roi.json                  analysis ROI for the motion energy, [x0, x1, y0, y1] in pixels
    <stem>_motion_energy.npy         (2, n_frames): row 0 is the time of each frame from the start of the video (s),
                                     row 1 the motion energy in the ROI (same measure as MotionEnergy.motion_energy)
    <stem>_motion_energy.lock        exists while the trace is queued or being computed
    <stem>_motion_energy.error.txt   traceback of a failed computation
    <stem>_racine.xlsx               seizures marked in the GUI

queue_motion starts a background process (python -m Video.RecoveryMotion <video>) that keeps running when the GUI is
closed. The processes of a PC compute one video at a time: each waits for the queue lock in the temp folder.
"""
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy

from Proc2P.utils import lprint

LOCK_MAX_AGE_S = 3 * 3600  # an older lock file is left over from a process that was killed
QUEUE_LOCK = Path(tempfile.gettempdir()) / 'lncm_recovery_motion.lock'
QUEUE_POLL_S = 10


def processed_dir(video):
    """Folder of the video's files: the last _RawData folder in the video's path is replaced by _Processed, keeping
    the levels after it. The video's own folder if its path has no _RawData folder."""
    parts = list(Path(video).parent.parts)
    for i in reversed(range(len(parts))):
        if parts[i].lower() == '_rawdata':
            parts[i] = '_Processed'
            return Path(*parts)
    return Path(video).parent


def _fn(video, suffix):
    return processed_dir(video) / (Path(video).stem + suffix)


def roi_fn(video):
    return _fn(video, '_roi.json')


def trace_fn(video):
    return _fn(video, '_motion_energy.npy')


def lock_fn(video):
    return _fn(video, '_motion_energy.lock')


def error_fn(video):
    return _fn(video, '_motion_energy.error.txt')


def events_fn(video):
    return _fn(video, '_racine.xlsx')


def load_roi(video):
    """[x0, x1, y0, y1] in pixels, or None if not saved"""
    fn = roi_fn(video)
    return json.loads(fn.read_text()) if fn.exists() else None


def save_roi(video, roi):
    """Save the ROI and delete the trace computed in the previous ROI"""
    processed_dir(video).mkdir(parents=True, exist_ok=True)
    roi_fn(video).write_text(json.dumps([int(x) for x in roi]))
    trace_fn(video).unlink(missing_ok=True)


def load_trace(video):
    """(time from the start of the video in s, motion energy), or None if not computed yet"""
    fn = trace_fn(video)
    if not fn.exists():
        return None
    t, me = numpy.load(fn)
    return t, me


def is_queued(video):
    """True if the trace is queued or being computed"""
    fn = lock_fn(video)
    return fn.exists() and time.time() - fn.stat().st_mtime < LOCK_MAX_AGE_S


def start_time(video):
    """Local start time of the recording (timezone-aware datetime) from the START_TIME_LOCAL tag that MonitorCam
    writes into the file, or from the file name (<name>-YYYY-MM-DDTHH-MM-SS, local time without time zone).
    None if neither is found."""
    with open(video, 'rb') as f:
        head = f.read(1 << 16)  # the tags are in the file header
    i = head.find(b'START_TIME_LOCAL')
    if i >= 0:
        j = head.find(b'\x44\x87', i)  # TagString element
        if 0 <= j < i + 40:
            size_byte = head[j + 2]
            n = 8 - size_byte.bit_length() + 1  # EBML variable-size integer: the leading zeros give the length
            size = int.from_bytes(head[j + 2:j + 2 + n], 'big') & ((1 << (7 * n)) - 1)
            text = head[j + 2 + n:j + 2 + n + size].decode(errors='replace')
            try:
                return datetime.datetime.fromisoformat(text)
            except ValueError:
                pass
    match = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})$', Path(video).stem)
    if match:
        return datetime.datetime.fromisoformat(f'{match[1]}T{match[2]}:{match[3]}:{match[4]}')
    return None


def queue_motion(video):
    """Start computing the trace in a background process, unless it exists, is queued, or there is no ROI.
    Returns True if it was started."""
    if trace_fn(video).exists() or is_queued(video) or load_roi(video) is None:
        return False
    lock_fn(video).touch()  # before starting, so a second call doesn't start another process
    error_fn(video).unlink(missing_ok=True)
    root = Path(__file__).resolve().parents[1]  # LNCM-lib
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), os.environ.get('PYTHONPATH', '')]),
               OPENCV_LOG_LEVEL='ERROR')  # OpenCV warns on every frame when reading the luma plane
    # DETACHED_PROCESS (Windows only): no console, and closing the GUI's console doesn't stop it
    subprocess.Popen([sys.executable, '-m', 'Video.RecoveryMotion', str(video)], env=env, cwd=root,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0))
    lprint(None, f'Queued the motion energy of {Path(video).name}')
    return True


def _take_queue_lock(video):
    """Wait until no other video is being computed on this PC. Keeps the video's own lock fresh while waiting."""
    while True:
        if QUEUE_LOCK.exists() and time.time() - QUEUE_LOCK.stat().st_mtime > LOCK_MAX_AGE_S:
            QUEUE_LOCK.unlink(missing_ok=True)
        try:
            os.close(os.open(QUEUE_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return
        except FileExistsError:
            lock_fn(video).touch()
            time.sleep(QUEUE_POLL_S)


def compute_motion(video):
    """Compute and save the trace in the saved ROI (waits for the queue lock)"""
    from Video.MotionEnergy import _save, motion_energy  # imports cv2; not needed by the GUI's file functions
    _take_queue_lock(video)
    try:
        roi = load_roi(video)
        lprint(None, f'Motion energy: {Path(video).name}, ROI {roi}')
        me, t = motion_energy(video, crop=roi, return_times=True)
        _save(numpy.array([t, me]), trace_fn(video))
    finally:
        QUEUE_LOCK.unlink(missing_ok=True)


if __name__ == '__main__':
    # python -m Video.RecoveryMotion <video>, started by queue_motion
    video = sys.argv[1]
    try:
        compute_motion(video)
    except Exception:
        error_fn(video).write_text(traceback.format_exc())
        raise
    finally:
        lock_fn(video).unlink(missing_ok=True)
