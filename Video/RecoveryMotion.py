"""Files and motion energy of post-surgery recovery videos (MonitorCam, Pinnacle-vEEG repo), for the recovery review
GUI (Video/RecoveryReviewGUI.py).

The files of a video <_RawData>/MonitorCam/<subfolders>/<stem>.mkv are in the same subfolders under _Processed
(processed_dir), named after the video:
    <stem>_roi.json                  analysis ROI for the motion energy, [x0, x1, y0, y1] in pixels
    <stem>_motion_energy.npy         (2, n_frames): row 0 is the time of each frame from the start of the video (s),
                                     row 1 the motion energy in the ROI (same measure as MotionEnergy.motion_energy)
    <stem>_motion_energy.lock        exists while the trace is queued or being computed
    <stem>_motion_energy.error.txt   traceback of a failed computation
    <stem>_racine.jsonl              seizures marked in the GUI, one per line, saved on every change
    <stem>_racine.xlsx               the marks exported with the GUI's Save button, escalating marks merged,
                                     with a Length(s) column

queue_motion queues the computation on the job queue (RecoveryMotion jobs, one video at a time for all users). It
runs in a worker process that keeps running when the GUI is closed.
"""
import datetime
import json
import os
import re
import time
import traceback
from pathlib import Path

import numpy

from JobQueue import client as job_queue
from JobQueue.common import QueueError
from Proc2P.utils import lprint

LOCK_MAX_AGE_S = 3 * 3600  # an older lock file is left over from a process that was killed
EVENT_COLUMNS = ['Start(s)', 'End(s)', 'Racine', 'Start time']
STAGES = ('1', '2', '3', '4', '5', '5+')  # Racine stages, from mild to severe
MERGE_GAP_S = 30  # escalating marks closer than this are merged into one seizure


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
    return _fn(video, '_racine.jsonl')


def excel_fn(video):
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


def load_events(video):
    """Marked seizures: dicts with EVENT_COLUMNS keys, Racine as text"""
    fn = events_fn(video)
    if not fn.exists():
        return []
    return [json.loads(line) for line in fn.read_text().splitlines() if line.strip()]


def save_events(video, events):
    """Write the marks to the jsonl file (through a temporary file, so a crash doesn't leave a partial file)"""
    processed_dir(video).mkdir(parents=True, exist_ok=True)
    fn = events_fn(video)
    tmp = fn.with_name(fn.name + '.tmp')
    tmp.write_text(''.join(json.dumps({k: e[k] for k in EVENT_COLUMNS}) + '\n' for e in events))
    os.replace(tmp, fn)


def export_excel(video, events):
    """Write the marks to the Excel file. Raises PermissionError if the file is open in Excel."""
    import pandas
    df = pandas.DataFrame(events, columns=EVENT_COLUMNS)
    df['Racine'] = [int(s) if s.isdigit() else s for s in df['Racine']]  # numbers stay numbers in Excel
    df.insert(2, 'Length(s)', (df['End(s)'] - df['Start(s)']).round(3))  # empty without an end
    df.to_excel(excel_fn(video), index=False)


def merge_escalating(events):
    """Merge marks of one escalating seizure: a mark is merged into the previous one if its stage is the same or
    higher, it starts at most MERGE_GAP_S after the previous mark, and the previous mark has no end. The merged
    seizure has the start (and start time) of the first mark, and the stage and end of the last."""
    merged = []
    last_start = None  # start of the last mark merged into merged[-1]
    for e in sorted(events, key=lambda e: e['Start(s)']):
        if (merged and merged[-1]['End(s)'] is None and e['Start(s)'] - last_start <= MERGE_GAP_S
                and STAGES.index(e['Racine']) >= STAGES.index(merged[-1]['Racine'])):
            merged[-1]['Racine'] = e['Racine']
            merged[-1]['End(s)'] = e['End(s)']
        else:
            merged.append(dict(e))
        last_start = e['Start(s)']
    return merged


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
    """Queue computing the trace on the job queue, unless it exists, is queued, or there is no ROI.
    Returns True if it was queued."""
    if trace_fn(video).exists() or is_queued(video) or load_roi(video) is None:
        return False
    lock_fn(video).touch()  # before queueing, so a second call doesn't queue it again
    error_fn(video).unlink(missing_ok=True)
    try:
        job_queue.submit('RecoveryMotion', str(video), label=Path(video).name, key=str(trace_fn(video)))
    except QueueError as e:
        lock_fn(video).unlink(missing_ok=True)
        lprint(None, f'Motion energy of {Path(video).name} not queued: {e}')
        return False
    lprint(None, f'Queued the motion energy of {Path(video).name}')
    return True


def compute_motion(video):
    """Compute and save the trace in the saved ROI"""
    from Video.MotionEnergy import _save, motion_energy  # imports cv2; not needed by the GUI's file functions
    roi = load_roi(video)
    lprint(None, f'Motion energy: {Path(video).name}, ROI {roi}')
    me, t = motion_energy(video, crop=roi, return_times=True)
    _save(numpy.array([t, me]), trace_fn(video))


def run_job(video):
    """Job queue handler of RecoveryMotion jobs, queued by queue_motion"""
    try:
        compute_motion(video)
    except Exception:
        error_fn(video).write_text(traceback.format_exc())
        raise
    finally:
        lock_fn(video).unlink(missing_ok=True)
    return Path(video).name
