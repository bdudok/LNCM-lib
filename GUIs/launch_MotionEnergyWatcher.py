from Video.MotionWatcher import watch

'''
Pulls the motion energy trace of new Pinnacle EDF files, so the seizure GUIs show it without waiting.
Keep it running in a console; it scans the folders every interval_min minutes.
'''

eeg_dirs = [r'D:\Shares\Data\_Processed\EEG']  # searched recursively for .edf files
video_dir = None  # folder of the vEEG recordings; None: the folder of each EDF

if __name__ == '__main__':
    watch(eeg_dirs=eeg_dirs, video_dir=video_dir, interval_min=10)
