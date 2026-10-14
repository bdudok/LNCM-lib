import datetime

from Video.MotionWatcher import watch

'''
Pulls the motion energy trace of new Pinnacle EDF files, so the seizure GUIs show it without waiting.
Keep it running in a console; it scans the folders every interval_min minutes.
'''

eeg_dirs = [r'D:\Shares\Data\_Processed\EEG']  # searched recursively for .edf files
video_dir = None  # folder of the vEEG recordings; None: the folder of each EDF
# EDFs recorded earlier are skipped: before the vEEG recorder pinned cameras by USB id (Recorder 00ec5b8, 2026-10-08),
# videos could come from the wrong camera
recorded_after = datetime.datetime(2026, 10, 9)

if __name__ == '__main__':
    watch(eeg_dirs=eeg_dirs, video_dir=video_dir, recorded_after=recorded_after, interval_min=10)
