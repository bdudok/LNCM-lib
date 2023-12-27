from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import os
import suite2p
#run using suite2p conda env but s2p package, not source
from suite2p.io import ome_to_binary

from Proc2P.Bruker import PreProc
from Proc2P import GetSessions
import pandas
from tifffile import imwrite

'''
Template for pre-processing and motion correction.
Specify the search criteria and settings below, then run using the suite2p interpreter.
Confirm the list of sessions by hitting Enter after it's displayed. 
'''

#Process sessions based on searching the Session database. Specify search criteria:
project = 'Voltage' #the "Project" field
task = 'MotionCorr' #or None to ignore. default: "MotionCorr"
incltag = None#'voltage' #or None to ignore. default: None

#settings
# Ch1 is red; Ch2 is green
ref_ch = 'Ch2' #reference channel, 'Ch1' or 'Ch2', only used if the movie has 2 channels.
pre_only = False #set True if only pre-process, no motion correction (default: False)
mc_only = True # True if only motion correct, not running the rest os Suite2p pipeline (default: True)

#the script should display this s2p version:
# Running suite2p version 0.14.2.dev7+g118901a

'''Below are general settings, only change if you know what you're doing'''
#pipeline ops
ops = suite2p.default_ops()

#specific settings
ops['do_bidiphase'] = True #whether or not to compute bidirectional phase offset
ops['bidi_corrected'] = True #Specifies whether to do bidi correction
ops['nonrigid'] = True
ops['block_size']  = [512, 16] #power of 2 and/or 3 (e.g. 128, 256, 384, etc)
ops['two_step_registration'] = True #run registration twice
# ops['reg_tif'] = False #write the registered binary to tiff files
# ops['reg_tif_chan2'] = False #write the registered binary to tiff files

#general settings
ops['batch_size'] = 5000
ops['fast_disk'] = 'E:/S2P/'
ops['delete_bin'] = False
ops['move_bin'] = True
ops['keep_movie_raw'] = True
#
if mc_only:
    ops['roidetect'] = False
    ops['neuropil_extract'] = False
    ops['spikedetect'] = False


if __name__ == '__main__':
    print('Running suite2p version', suite2p.version)

    #get the list of sessions
    db = GetSessions()
    session_df = db.search(project=project, task=task, incltag=incltag)
    print(session_df[['Image.ID', 'Processed.Path']])
    input(f"Press Enter to continue with these {len(session_df)} sessions")


    for i, item in session_df.iterrows():
        dpath = item['Raw.Path']
        processed_path = item['Processed.Path']
        imID = item['Image.ID']
        prefix = imID[:-4]
        btag = imID[-3:]
        print(prefix)
        if not os.path.exists(processed_path):
            os.mkdir(processed_path)

        #preprocess session
        s = PreProc.PreProc(dpath, processed_path, prefix, btag)
        if pre_only:
            if s.is_processed:
                s.preprocess()
            continue

        filelist = [fn for fn in os.listdir(s.dpath) if ((prefix in fn) and ('.ome.tif' in fn))]

        # session ops
        db = {
            'fs': s.si['framerate'],
            'data_path': [s.dpath, ],
            'tiff_list': filelist,
            'save_path0': s.procpath
        }

        dual_channel = len(s.channelnames) > 1
        if dual_channel:
            db['input_format'] = 'bruker'
            db['bruker'] = True
            ops['nchannels'] = 2
            ops['align_by_chan'] = s.channelnames.index(ref_ch)


        #run motion correction
        output_ops = suite2p.run_s2p(ops=ops, db=db)

        #clean up output
        mean1 = output_ops['meanImg'] / output_ops['meanImg'].max() * 255
        previewname = s.procpath + prefix + f'_preview.tif'
        if not dual_channel:
            preview = mean1
        else:
            mean2 = output_ops['meanImg_chan2'] / output_ops['meanImg_chan2'].max() * 255
            preview = numpy.zeros((*output_ops['meanImg'].shape, 3), dtype='uint8')
            preview[..., 0] = mean1
            preview[..., 1] = mean2
        imwrite(previewname, preview)

        #remove raw movie
        os.remove(s.procpath + 'suite2p\plane0/data_raw.bin')
        os.rename(s.procpath + 'suite2p\plane0/data.bin',
                  s.procpath + prefix + f'_registered_{s.channelnames[0]}.bin')
        if dual_channel:
            os.remove(s.procpath + 'suite2p\plane0/data_chan2_raw.bin')
            os.rename(s.procpath + 'suite2p\plane0/data_chan2.bin',
                      s.procpath + prefix + f'_registered_{s.channelnames[1]}.bin')