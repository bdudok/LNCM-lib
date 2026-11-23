import os

import pandas

# threads are not limited: with several jobs running, each is slower but RAM use doesn't change.
# To limit threads, set OMP_NUM_THREADS, MKL_NUM_THREADS and NUMBA_NUM_THREADS to the same value here,
# otherwise numba resets torch to use all cores.
os.environ["MKL_DISABLE_FAST_MM"] = "1"
from Proc2P import GetSessions
from Proc2P.Bruker import MotionCorrect


'''
Template for pre-processing and motion correction.
Specify the search criteria and settings below, then run using the suite2p interpreter.
Confirm the list of sessions by hitting Enter after it's displayed.
The sessions are queued on the job queue (dashboard: http://SVR-LNCM-01:8766). The script waits and prints each result;
if you stop it, the sessions are still processed.
'''

'''
To update s2p, use:
conda activate suite2p
python -m pip install suite2p --upgrade
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
# pipeline settings (suite2p 1.x: nested dict): the lab's settings with the sensor preset, see MotionCorrect.preset_settings
settings = MotionCorrect.preset_settings(preset, mc_only)
reg = settings['registration']  # to change a setting, e.g.: reg['batch_size'] = 1000

switches = {'pre_only': pre_only, 'overwrite_previous': overwrite_previous, 'overwrite_preproc': overwrite_preproc,
            'ref_ch': ref_ch, }

#run the script
MotionCorrect.run(settings, session_df, **switches)

