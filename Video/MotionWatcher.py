"""Watch the processed data folders and pull the motion energy trace of new recordings, so the seizure detection and
curation GUIs show it the first time a recording is opened. Run it with GUIs/launch_MotionEnergyWatcher.py.

- Pinnacle: each .edf in the EEG folders (searched recursively) without <prefix>_motion_energy.npy. An EDF is
  processed once it has not been modified for settle_eeg_h hours: the video files rotate every hour, so by then
  the file covering the end of the EDF is closed. The EDF's video map is saved as well.
- Bruker: each session folder in the 2P folders that has LFP (<prefix>_ephys.npy) and no <prefix>_motion_energy.npy,
  once its SessionInfo has not been modified for settle_2p_h hours.
Only recordings modified in the last max_age_days are considered, so old recordings without video are not read again
at each start. Recordings without a matching video, or that fail, are reported and skipped until the watcher restarts.
"""
import os
import time
import traceback
from pathlib import Path

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


def bruker_jobs(twop_dirs, settle_s, max_age_s):
    now = time.time()
    for root in twop_dirs:
        for entry in os.scandir(root):
            if not entry.is_dir():
                continue
            prefix = entry.name
            session = Path(entry.path)
            si_fn = session / f'{prefix}_SessionInfo.json'
            if not (si_fn.exists() and (session / f'{prefix}_ephys.npy').exists()):
                continue
            age = now - os.path.getmtime(si_fn)
            if settle_s < age < max_age_s and not trace_fn(Setup.LNCM, root, prefix).exists():
                yield Setup.LNCM, root, prefix


def watch(eeg_dirs=(), twop_dirs=(), video_dir=None, interval_min=10, settle_eeg_h=1.5, settle_2p_h=0.2,
          max_age_days=3, n_workers=N_WORKERS):
    """
    eeg_dirs: folders of Pinnacle EDF files; twop_dirs: folders of processed 2P sessions
    video_dir: folder of the vEEG recordings, default: the folder of each EDF (as in the curation GUI)
    interval_min: time between scans of the folders
    """
    skipped = set()
    while True:
        jobs = [*pinnacle_jobs(eeg_dirs, settle_eeg_h * 3600, max_age_days * 86400),
                *bruker_jobs(twop_dirs, settle_2p_h * 3600, max_age_days * 86400)]
        for job in jobs:
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
