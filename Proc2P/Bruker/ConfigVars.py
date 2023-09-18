'''
Constants used across scripts are stored here
'''


class CF:
    pos = '_pos.npy'
    speed = '_smspd.npy'
    opto = '_bad_frames.npy'
    fps = 20.0
    ledcal = 230 #100 led power in mW
    bands = {
        'ripple': (130, 200),
        'theta': (6, 9),
        'HF': (80, 500)
    }