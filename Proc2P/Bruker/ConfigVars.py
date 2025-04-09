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
    alt_processed_paths = ('E:/ExtDrives/LNCM1/_Processed/', 'Y:/_Processed/') #can look here for archived processed files
    alt_raw_paths = ('E:/ExtDrives/LNCM2/_RawData/', ) # can look here for archived raw files