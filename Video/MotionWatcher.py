"""Watch the EEG folders and pull the motion energy trace of new Pinnacle EDF files, so the seizure detection and
curation GUIs show it the first time a recording is opened. Run it with GUIs/launch_MotionEnergyWatcher.py.

Each .edf in the EEG folders (searched recursively) without <prefix>_motion_energy.npy is processed once it has not
been modified for settle_h hours: the video files rotate every hour, so by then the file covering the end of the EDF
is closed. The EDF's video map is saved as well.
Only EDFs modified in the last max_age_days are considered, so old recordings without video are not read again at
each start.
EDFs without a matching video, or that fail, are reported and skipped until the watcher restarts.
2P sessions are not watched: their trace needs a motion crop, and saving the crop starts the computation.
"""
import os
import time
import traceback

from LFP.Setup import Setup
from Proc2P.utils import lprint
from Video.MotionEnergy import N_WORKERS, is_running, pull_motion, trace_fn


def pinnacle_jobs(eeg_dirs, settle_s, max_age_s):
    now = time.time()
    for root in eeg_dirs:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if not fn.endswith('.edf'):
                    continue
                prefix = fn[:-len('.edf')]
                age = now - os.path.getmtime(os.path.join(dirpath, fn))
                if settle_s < age < max_age_s and not trace_fn(Setup.PINNACLE, dirpath, prefix).exists():
                    yield Setup.PINNACLE, dirpath, prefix


def watch(eeg_dirs, video_dir=None, interval_min=10, settle_h=1.5, max_age_days=3, n_workers=N_WORKERS):
    """
    eeg_dirs: folders of Pinnacle EDF files
    video_dir: folder of the vEEG recordings, default: the folder of each EDF (as in the curation GUI)
    interval_min: time between scans of the folders
    """
    skipped = set()
    while True:
        for job in list(pinnacle_jobs(eeg_dirs, settle_h * 3600, max_age_days * 86400)):
            if job in skipped or is_running(*job):  # running: started from a GUI, check again at the next scan
                continue
            setup, path, prefix = job
            lprint(None, f'Pulling motion energy: {setup} {prefix}')
            try:
                if pull_motion(setup, path, prefix, video_dir, n_workers) is None:
                    lprint(None, f'No video found for {prefix}, skipping')
                    skipped.add(job)
            except Exception:
                traceback.print_exc()
                lprint(None, f'Failed: {prefix}, skipping')
                skipped.add(job)
        time.sleep(interval_min * 60)
