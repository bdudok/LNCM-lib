'''
Handlers of the BatchGUI job types (registered in JobQueue/common.py).
Each takes the job data that BatchGUI queues (the same tuples as before, as lists) and returns a result message.
They run in the runner's worker processes; imports are inside the functions so each worker loads only its own.
'''
import os
from pathlib import Path


def gevireg(data):
    from Proc2P.Analysis.GEVIReg.Register import register
    processed_path, prefix, config = data
    register(processed_path, prefix, config)
    return prefix


def pull_vm(data):
    from Proc2P.Analysis.ImagingSession import ImagingSession
    from Proc2P.Analysis.NormalizeVm import normalize_trace
    path, prefix, tag, overwrite, kwargs = data
    session = ImagingSession(path, prefix, tag, norip=True)
    if overwrite or not os.path.exists(os.path.join(session.ca.pf, 'vm.npy')):
        normalize_trace(session, **kwargs)
    return prefix


def sima(data):
    from Proc2P.Analysis import RoiEditor as roi_editor
    from Proc2P.utils import logger
    path, prefix, apps, config = data
    if roi_editor.sima_available:
        log = logger()
        log.set_handle(path, prefix)
        roi_editor.RoiEditor(path, prefix).autodetect(approach=apps, config=config, log=log)
        return prefix
    roi_editor.check_compatible_ops(path, prefix)
    return f'{prefix}: {roi_editor.roi_detector(path, prefix, apps, config)}'


def pull_signals(data):
    from Proc2P.Analysis.PullSignals import pull_signals as pull
    path, prefix, roi, ch, szmode, snrw, source = data
    if roi == 'Auto':
        # locate last saved roi file
        exs = [1]
        for f in os.listdir(Path(path) / prefix):
            if prefix in f and '_saved_roi_' in f:
                try:
                    if f[-6:-4] != '-1':
                        exs.append(int(f[:-4].split('_')[-1]))
                except Exception:
                    pass
        tag = str(max(exs))
    else:
        tag = roi
    retval = pull(path, prefix, tag=tag, ch=ch, snr_weighted=snrw, use_movie=source)
    return f'{prefix}: {retval}' if retval else prefix


def ripples(data):
    from Proc2P.Analysis.Ripples import Ripples, RippleConfig, legacy_channel_from_config
    from Proc2P.Bruker.PreProc import SessionInfo
    wdir, prefix, kwargs = data
    cfg = {key: kwargs[key] for key in ('tr1', 'tr2', 'y_scale', 'require_3_cycles')}
    fs = SessionInfo(os.path.join(wdir, prefix), prefix).load(must_exist=True)['fs']
    r = Ripples(wdir, prefix, fs=fs, config=cfg, force=kwargs['overwrite_existing'],
                ephys_channel=kwargs['channel'], tag=kwargs['tag'],
                ask_legacy_channel=legacy_channel_from_config(RippleConfig(**kwargs)))
    return f'{prefix}: {r.count_included()} events saved.'


N_CELL_WORKERS = 24  # ProcessROIs runs the cells of a session in this many processes
_cell_pool = None  # started by the first ProcessROIs job in this worker, reused by later ones


def process_rois(data):
    from multiprocessing import Queue
    from Proc2P.Analysis.CaTrace import CaTrace
    from Proc2P.Analysis.CaTrace import Worker as CellWorker
    global _cell_pool
    if _cell_pool is None:
        _cell_pool = {'queue': Queue(), 'result_queue': Queue(), 'n': 0}
    pool = _cell_pool
    path, prefix, tag, config = data
    bsltype = 'poly' if config.polynom_baseline else 'original'
    invert = [config.invert_first_channel_G, config.invert_second_channel_R]
    excl = (config.exclude_from_frame, config.exclude_until_frame)
    sz_mode = config.seizure_mode
    for ch in (0, 1):
        session = CaTrace(path, prefix, bsltype=bsltype, exclude=excl, peakdet=False, ch=ch, tag=tag,
                          invert=invert[ch], last_bg=config.last_ROI_is_background, ignore_saturation=sz_mode)
        if session.open_raw() == -1:
            continue
        if os.path.exists(session.pf):
            print(f'{session.pf} folder exists, skipping')
            continue
        print('Processing', session.pf)
        if sz_mode:
            session.ol_index = []
        for c in range(session.cells):
            if c >= pool['n'] and pool['n'] < N_CELL_WORKERS:
                CellWorker(pool['queue'], pool['result_queue']).start()
                pool['n'] += 1
            pool['queue'].put(session.pack_data(c))
        for cell_data in iter(pool['result_queue'].get, None):
            if session.unpack_data(cell_data):
                break
    return prefix
