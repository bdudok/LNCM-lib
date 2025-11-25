'''
Constants used across scripts are stored here
These aren't expected to be installation-specific - for those use envs.site_config
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
    bands = {
        'ripple': (130, 200),
        'theta': (6, 9),
        'HF': (80, 500)
    }