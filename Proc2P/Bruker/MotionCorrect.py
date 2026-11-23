import copy
import json
import os, shutil
import time
import warnings
# set before suite2p (torch) is imported, also in the job queue workers that import this module first
os.environ.setdefault("MKL_DISABLE_FAST_MM", "1")
import numpy
import pandas
import suite2p
# run using suite2p conda env but s2p package, not source

from Proc2P.Bruker import Bidiphase, PreProc
from tifffile import imwrite
from Proc2P.utils import lprint
from JobQueue.client import QueueClient

# with keep_movie_raw, suite2p registers from the read-only data_raw.bin memmap and torch warns about it.
# suite2p doesn't write to that tensor, so the warning is harmless.
warnings.filterwarnings('ignore', message='The given NumPy array is not writable')

'''Pre-processing and motion correction of Bruker sessions (suite2p 1.x).
run queues one MotionCorrect job per session on the job queue; the jobs run process_session in the user's workers.
See: Proc2P/Analysis/S2P_template/20261007_BatchRun_MotionCorrect_Template.py
'''


POLL_S = 10  # run checks the queued jobs this often


def run(settings, session_df, pre_only, overwrite_previous, overwrite_preproc, ref_ch, fast_disk=None):
    '''
    Queues pre-processing and motion correction of each session on the job queue and waits until all are done,
    printing each result. Raises RuntimeError at the end if any session failed. If the script is stopped while
    waiting, the jobs stay queued (see the dashboard). Arguments: see process_session.
    '''
    client = QueueClient()
    jobs = {}
    for _, item in session_df.iterrows():
        prefix = item['Image.ID'][:-4]
        entry = json.loads(item.to_json(default_handler=str))  # the Baserow row, as JSON values
        job_id = client.submit('MotionCorrect', (settings, entry, pre_only, overwrite_previous, overwrite_preproc,
                                                 ref_ch, fast_disk), label=prefix, key=prefix)
        jobs[job_id] = prefix
    lprint(None, f'Queued {len(jobs)} sessions. Waiting for them to finish; if you stop the script, they still run.')
    failed = []
    while jobs:
        time.sleep(POLL_S)
        for job in client.job_status(list(jobs)):
            if job['status'] == 'done':
                lprint(None, 'Done:', job['result'])
            elif job['status'] == 'failed':
                lprint(None, f'Failed: {jobs[job["job_id"]]}\n{job["error"]}')
                failed.append(jobs[job['job_id']])
            else:
                continue
            del jobs[job['job_id']]
    if failed:
        raise RuntimeError(f'Motion correction failed in {len(failed)} sessions: {", ".join(failed)}')


def run_job(data):
    '''Job queue handler of MotionCorrect jobs, queued by run'''
    settings, entry, pre_only, overwrite_previous, overwrite_preproc, ref_ch, fast_disk = data
    return process_session(settings, pandas.Series(entry), pre_only, overwrite_previous, overwrite_preproc, ref_ch,
                           fast_disk)


def process_session(settings, item, pre_only, overwrite_previous, overwrite_preproc, ref_ch, fast_disk=None):
    '''
    Pre-process and motion correct one session. Returns a result message.
    :param settings: suite2p settings dict (from suite2p.default_settings()).
        Bidiphase: with settings['registration']['do_bidiphase'] False, the movie is not corrected. With True, the
        offset is settings['registration']['bidiphase'] if nonzero, otherwise estimated from the reference channel
        (Bidiphase.estimate_file) and rounded, or 0 if the estimate is under 1 px. A nonzero offset is corrected in
        a copy of the raw movie in <fast_disk>_bidi, which suite2p registers instead of the raw files (suite2p 1.1.0
        can't correct it, see Bidiphase). suite2p itself runs with do_bidiphase False; the corrected offset is saved
        in the suite2p db as 'bidiphase_corrected'.
    :param item: the session's Baserow row (Image.ID, Raw.Path, Processed.Path, and fields for PreProc.parse_notes)
    :param fast_disk: folder for suite2p scratch files, in a subfolder per session. Default: E:/S2P/<username>/
    '''
    username = os.environ.get('USERNAME')
    print(f'Running suite2p version {suite2p.version} as {username}')
    if fast_disk is None:
        fast_disk = f'E:/S2P/{username}/'

    dpath = item['Raw.Path']
    processed_path = item['Processed.Path']
    imID = item['Image.ID']
    prefix = imID[:-4]
    btag = imID[-3:]
    lprint(None, "Processing", prefix)
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)

    # preprocess session
    s = PreProc.PreProc(dpath, processed_path, prefix, btag, overwrite=overwrite_preproc, db_entry=item)
    if pre_only:
        return f'{prefix}: pre-processed'

    fast_disk = os.path.join(fast_disk, prefix)  # own folder for each session, so parallel jobs don't share it
    if os.path.exists(fast_disk):
        if 'E:' in fast_disk:
            shutil.rmtree(fast_disk)

    if overwrite_previous:
        ofn = os.path.join(s.procpath, 'suite2p')
        if os.path.exists(ofn):
            shutil.rmtree(ofn)
        for dirname in ('_trace_1-ch1', '_trace_1-ch0'):
            ofn = os.path.join(s.procpath, prefix + dirname)
            if os.path.exists(ofn):
                shutil.rmtree(ofn)
        suffix_list = ('_trace_1.npy', '_saved_roi_1.npy', '_saved_roi_STICA-R.npy', '_saved_roi_STICA-G.npy',
                       '_avgmax.tif', '_preview.tif', '_registered_Ch2.bin', '_registered_Ch1.bin')
        for suffix in suffix_list:
            ofn = os.path.join(s.procpath, prefix + suffix)
            if os.path.exists(ofn):
                os.remove(ofn)

    filelist = [fn for fn in os.listdir(s.dpath) if ((prefix in fn) and ('.ome.tif' in fn))]

    # session db (input and output paths, data format)
    db = {
        'data_path': [str(s.dpath), ],
        'file_list': filelist,
        'save_path0': str(s.procpath),
        'fast_disk': fast_disk,
        'keep_movie_raw': True,
        'nchannels': 1,
        'functional_chan': 1,
    }
    # copy settings for each session, so changes don't carry over to the next session in the batch
    session_settings = copy.deepcopy(settings)
    session_settings['fs'] = s.si['framerate']
    session_settings['registration']['align_by_chan2'] = False

    # Bruker files are split into channels by file name: Ch1 goes to data.bin, Ch2 to data_chan2.bin
    dual_channel = len(s.channelnames) > 1
    if dual_channel:
        db['input_format'] = 'bruker'
        db['nchannels'] = 2
        session_settings['registration']['align_by_chan2'] = s.channelnames.index(ref_ch) == 1

    # bidiphase offset, corrected in a copy of the movie
    reg = session_settings['registration']
    offset = 0
    if reg['do_bidiphase']:
        if reg['bidiphase']:
            offset = int(reg['bidiphase'])
            lprint(None, f'Bidiphase offset from settings: {offset} px')
        else:
            align_ch = s.channelnames[1] if reg['align_by_chan2'] else s.channelnames[0]
            fn = sorted(fn for fn in filelist if f'_{align_ch}_' in fn)[0]
            estimate = Bidiphase.estimate_file(os.path.join(s.dpath, fn))
            offset = round(estimate) if abs(estimate) >= 1 else 0  # sub-pixel offsets are not corrected
            lprint(None, f'Bidiphase offset estimated from {align_ch}: {estimate:.1f} px, correcting {offset} px')
    reg['do_bidiphase'] = False
    reg['bidiphase'] = 0
    db['bidiphase_corrected'] = offset
    standin_path = os.path.normpath(fast_disk) + '_bidi'
    if offset:
        Bidiphase.write_corrected(s.dpath, filelist, standin_path, offset)
        db['data_path'] = [standin_path]

    # run motion correction
    try:
        suite2p.run_s2p(db=db, settings=session_settings)
    finally:
        if offset:
            shutil.rmtree(standin_path, ignore_errors=True)
    s2p_folder = s.procpath / 'suite2p' / 'plane0'
    reg_outputs = numpy.load(s2p_folder / 'reg_outputs.npy', allow_pickle=True).item()

    # clean up output
    preview_channels = {'R': None, 'G': None, 'B': None}
    mean1 = reg_outputs['meanImg'] / reg_outputs['meanImg'].max() * 255
    previewname = s.procpath / f'{prefix}_preview.tif'
    imgshape = [*reg_outputs['meanImg'].shape]
    if len(imgshape) < 3:
        imgshape.append(3)
    preview = numpy.zeros(imgshape, dtype='uint8')

    if s.channelnames[0] == 'Ch1':
        preview_channels['R'] = mean1
    else:
        preview_channels['G'] = mean1

    if dual_channel:
        mean2 = reg_outputs['meanImg_chan2'] / reg_outputs['meanImg_chan2'].max() * 255
        preview_channels['R'] = mean1
        preview_channels['G'] = mean2

    for chi, chn in enumerate('RGB'):
        x = preview_channels[chn]
        if x is not None:
            preview[..., chi] = x

    imwrite(previewname, preview)

    # remove raw movie
    os.remove(s2p_folder / 'data_raw.bin')
    os.rename(s2p_folder / 'data.bin',
              s.procpath / f'{prefix}_registered_{s.channelnames[0]}.bin')
    if dual_channel:
        os.remove(s2p_folder / 'data_chan2_raw.bin')
        os.rename(s2p_folder / 'data_chan2.bin',
                  s.procpath / f'{prefix}_registered_{s.channelnames[1]}.bin')
    if 'E:' in fast_disk:
        shutil.rmtree(fast_disk, ignore_errors=True)
    return f'{prefix}: motion corrected, bidiphase {offset} px'
