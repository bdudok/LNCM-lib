import numpy
import os, shutil
import suite2p
# run using suite2p conda env but s2p package, not source

from Proc2P.Bruker import PreProc
from Proc2P import GetSessions
from tifffile import imwrite

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

# Process sessions based on searching the Session database. Specify search criteria:
project = 'PVTot'  # the "Project" field
task = 'MotionCorr'  # or None to ignore. default: "MotionCorr"
incltag = None  # 'substring' or None to ignore. default: None

# settings
preset = 'Calcium'  # one of: ('Calcium', 'Voltage') or None to use S2P defaults
ref_ch = 'Ch2'  # reference channel, 'Ch1' or 'Ch2', ignored if the movie has 1 channel. # Ch1 is red; Ch2 is green
pre_only = False  # set True if only pre-process, no motion correction (default: False)
overwrite_previous = True  # set if you want to delete old motion corrected movie (use if re-processing with different settings)
overwrite_preproc = False  # set if you want to delete old pre-processed files (if False, skips to motion correct)
mc_only = True  # set True if only motion correct, not running the rest os Suite2p pipeline (default: True)

# the script should display this s2p version:
# Running suite2p version 0.14.2.dev7+g118901a

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
    ops['batch_size'] = 5000
elif preset == 'Voltage':
    ops['snr_thresh'] = 1.5  # increased, so that low snr frames don't get corrected
    ops['nonrigid'] = False  # no movement expected within frame
    ops['block_size'] = [512, 48]  # power of 2 and/or 3 (e.g. 128, 256, 384, etc) #use full frame as one block
    ops['batch_size'] = 100000  # use full rec as one batch
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

if __name__ == '__main__':
    username = os.environ.get('USERNAME')
    print(f'Running suite2p version {suite2p.version} as {username}')
    fast_disk = f'E:/S2P/{username}/'
    ops['fast_disk'] = fast_disk

    # get the list of sessions
    db = GetSessions()
    session_df = db.search(project=project, task=task, incltag=incltag)
    print(session_df[['Image.ID', 'Processed.Path']])
    input(f"Press Enter to continue with these {len(session_df)} sessions")
    assert len(session_df)

    for i, item in session_df.iterrows():
        dpath = item['Raw.Path']
        processed_path = item['Processed.Path']
        imID = item['Image.ID']
        prefix = imID[:-4]
        btag = imID[-3:]
        print(prefix)
        if not os.path.exists(processed_path):
            os.mkdir(processed_path)

        # preprocess session
        s = PreProc.PreProc(dpath, processed_path, prefix, btag, overwrite=overwrite_preproc)
        if pre_only:
            if s.is_processed:
                s.preprocess()
            continue

        # clean destination for scratch files
        if os.path.exists(fast_disk):
            if 'E:' in fast_disk:
                shutil.rmtree(fast_disk)

        # delete output of previous run
        if overwrite_previous:
            ofn = os.path.join(s.procpath, 'suite2p')
            if os.path.exists(ofn):
                shutil.rmtree(ofn)
            for suffix in ('_avgmax.tif', '_preview.tif'):
                ofn = os.path.join(s.procpath, prefix + suffix)
                if os.path.exists(ofn):
                    os.remove(ofn)

        filelist = [fn for fn in os.listdir(s.dpath) if ((prefix in fn) and ('.ome.tif' in fn))]

        # session ops
        db = {
            'fs': s.si['framerate'],
            'data_path': [s.dpath, ],
            'tiff_list': filelist,
            'save_path0': s.procpath
        }

        # adjust ops if dual channel
        dual_channel = len(s.channelnames) > 1
        if dual_channel:
            db['input_format'] = 'bruker'
            db['bruker'] = True
            ops['nchannels'] = 2
            ops['align_by_chan'] = s.channelnames.index(ref_ch)

        # run motion correction
        output_ops = suite2p.run_s2p(ops=ops, db=db)

        # clean up output
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

        # remove raw movie and move output to final destination
        os.remove(s.procpath + 'suite2p\plane0/data_raw.bin')
        os.rename(s.procpath + 'suite2p\plane0/data.bin',
                  s.procpath + prefix + f'_registered_{s.channelnames[0]}.bin')
        if dual_channel:
            os.remove(s.procpath + 'suite2p\plane0/data_chan2_raw.bin')
            os.rename(s.procpath + 'suite2p\plane0/data_chan2.bin',
                      s.procpath + prefix + f'_registered_{s.channelnames[1]}.bin')

        print(f'S2P processing completed for {prefix}.')
