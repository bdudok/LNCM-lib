'''
Constants used across scripts are stored here
'''


class CF:
    pos = '_pos.npy'
    speed = '_spd.npy'
    spd = '_spd.npy'
    smspd = '_smspd.npy'
    opto = '_bad_frames.npy'
    laps = '_laps.npy'
    cam = '_cam_sync_frames.npy'
    fps = 20.0 #default for calcium.
    ledcal = 230 #100 led power in mW
    bands = {
        'ripple': (130, 200),
        'theta': (6, 9),
        'HF': (80, 500)
    }
    alt_processed_paths = ('I:/_Processed/', 'Y:/_Processed/') #can look here for archived files