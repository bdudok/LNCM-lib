import os

import pandas

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
import suite2p
from Proc2P import GetSessions
from Proc2P.Bruker import MotionCorrect


'''
Template for pre-processing and motion correction.
Specify the search criteria and settings below, then run using the suite2p interpreter.
Confirm the list of sessions by hitting Enter after it's displayed. 
'''

'''
To update s2p, use:
conda activate suite2p
pip install git+https://github.com/mouseland/suite2p.git
'''

# Process specified sessions: list them in quotes or set the variable to None
#note that each item in the list will be searched with 'contain', and all results will be included
prefix_list = ['BL6-47_2026-01-09_KA-LFP_112', ]
# prefix_list = None

# If prefix_lisit is None, process sessions based on searching the Session database. Specify search criteria:
project = 'eCB-GRAB'  # the "Project" field
task = 'MotionCorr'  # or None to ignore. default: "MotionCorr"
incltag = 'test'  # 'substring' or None to ignore. default: None

# settings
preset = 'Calcium'  # one of: ('Calcium', 'Voltage') or None to use S2P defaults
ref_ch = 'Ch2'  # reference channel, 'Ch1' or 'Ch2', ignored if the movie has 1 channel. # Ch1 is red; Ch2 is green
pre_only = False  # set True if only pre-process, no motion correction (default: False)
overwrite_previous = True  # set if you want to delete old motion corrected movie (use if re-processing with different settings)
overwrite_preproc = False  # set if you want to delete old pre-processed files (if False, skips to motion correct)
mc_only = True  # set True if only motion correct, not running the rest os Suite2p pipeline (default: True)

# get the list of sessions
db = GetSessions()
if prefix_list is None:
    session_df = db.search(project=project, task=task, incltag=incltag)
else:
    sessions = []
    for prefix in prefix_list:
        sessions.append(db.get_session(prefix))
    session_df = pandas.concat(sessions)
print(session_df[['Image.ID', 'Processed.Path']])
input(f"Press Enter to continue with these {len(session_df)} sessions")
assert len(session_df)


'''Below are general settings, only change if you know what you're doing'''
# pipeline ops
ops = suite2p.default_ops()

# specific settings
ops['do_bidiphase'] = True  # whether or not to compute bidirectional phase offset
ops['bidi_corrected'] = True  # Specifies whether to do bidi correction
ops['two_step_registration'] = True  # run registration twice

# sensor specific settings
if preset == None:
    pass
elif preset == 'Calcium':
    ops['nonrigid'] = True
    ops['block_size'] = [512, 16]  # power of 2 and/or 3 (e.g. 128, 256, 384, etc)
    ops['batch_size'] = 500
elif preset == 'Voltage':
    ops['snr_thresh'] = 1.5  # increased, so that low snr frames don't get corrected
    ops['nonrigid'] = False  # no movement expected within frame
    ops['block_size'] = [512, 48]  # power of 2 and/or 3 (e.g. 128, 256, 384, etc) #use full frame as one block
    ops['batch_size'] = 5000  # use full rec as one batch
    ops['maxregshift'] = 0.05  # reduced, because fow is small
else:
    input(f"No preset specified for {preset}, press Enter to continue with defaults")

# general settings
ops['delete_bin'] = False
ops['move_bin'] = True
ops['keep_movie_raw'] = True

if mc_only:
    ops['roidetect'] = False
    ops['neuropil_extract'] = False
    ops['spikedetect'] = False

switches = {'pre_only': pre_only, 'overwrite_previous': overwrite_previous, 'overwrite_preproc': overwrite_preproc,
            'ref_ch': ref_ch, }

#run the script
MotionCorrect.run(ops, session_df, **switches)

