from datetime import datetime

import numpy
import datetime
from LFP.SpikeDet import SpikesPower
from LFP.SzDet import InstRate
from LFP.Pinnacle import ReadEDF
from Proc2P.Bruker import LoadEphys
import pandas


def run_detection(*args, **kwargs):
    '''wrapper to call sz processing for each format'''
    format = kwargs.pop('format')
    if format == 'edf':
        run_detection_edf(*args, **kwargs)
    elif format == 'ephys':
        run_detection_ephys(*args, **kwargs)

def run_detection_edf(edf: ReadEDF, opts, save_envelope=False, savetag=None):
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    path = edf.path
    prefix = edf.prefix
    item = opts
    if savetag is None:
        output_fn = path + prefix
    else:
        output_fn = path + prefix + '_' + savetag
    ch = int(item['Channel'])
    print(ts, 'Processing', prefix)
    kwargs = {}
    for kw, key in (('lo', 'LoCut'), ('hi', 'HiCut'),):
        kwargs[kw] = float(item[key])
    spikedet = SpikesPower.Detect(edf.trace, fs=edf.fs, **kwargs)
    kwargs = {}
    for kw, key in (('tr1', 'Tr1'), ('tr2', 'Tr2'), ('trdiff', 'TrDiff'),
                    ('dur', 'Dur'), ('dist', 'Dist'),):
        kwargs[kw] = float(item[key])
    spikes = spikedet.get_spikes(**kwargs)
    spikedet.spiketimes_to_excel(path, prefix, ch=ch)
    if save_envelope:
        numpy.save(output_fn + f'_Ch{ch}_envelope.npy', spikedet.env)

    # detect seizures
    kwargs = {}
    for kw, key in (('cleanup', 'Sz.MinDur'), ('gap', 'Sz.Gap'),):
        kwargs[kw] = float(item[key])
    sz_burden, sz_times = InstRate.SpikeTrace(spikes, int(item['SzDet.Framesize']), **kwargs)

    numpy.save(output_fn + f'_Ch{ch}_sz_burden.npy', sz_burden)
    op = pandas.DataFrame(sz_times, columns=('Sz.Start(s)', 'Sz.Stop(s)'))
    op.to_excel(output_fn + f'_Ch{ch}_seizure_times.xlsx')
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    print(ts, f'Ch {ch} Done.')


def run_detection_ephys(ephys: LoadEphys, opts, save_envelope=False, savetag=None):
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    path = ephys.path
    prefix = ephys.prefix
    if savetag is None:
        output_fn = path + prefix
    else:
        output_fn = path + prefix + '_' + savetag
    item = opts
    ch = int(item['Channel'])
    print(ts, 'Processing', prefix)
    kwargs = {}
    fs = int(item['fs'])
    for kw, key in (('lo', 'LoCut'), ('hi', 'HiCut'),):
        kwargs[kw] = float(item[key])
    spikedet = SpikesPower.Detect(ephys.trace, fs=fs, **kwargs)
    kwargs = {}
    for kw, key in (('tr1', 'Tr1'), ('tr2', 'Tr2'), ('trdiff', 'TrDiff'),
                    ('dur', 'Dur'), ('dist', 'Dist'),):
        kwargs[kw] = float(item[key])
    spikes = spikedet.get_spikes(**kwargs)
    spikedet.spiketimes_to_excel(path, prefix, ch=ch)
    if save_envelope:
        numpy.save(output_fn + f'_ch{ch}_envelope.npy', spikedet.env)

    # detect seizures
    kwargs = {}
    for kw, key in (('cleanup', 'Sz.MinDur'), ('gap', 'Sz.Gap'),):
        kwargs[kw] = float(item[key])
    sz_burden, sz_times = InstRate.SpikeTrace(spikes, int(item['SzDet.Framesize']), **kwargs)

    numpy.save(output_fn + f'_ch{ch}_sz_burden.npy', sz_burden)
    op = pandas.DataFrame(sz_times, columns=('Sz.Start(s)', 'Sz.Stop(s)'))
    op.to_excel(output_fn + f'_ch{ch}_seizure_times.xlsx')
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    print(ts, f'Ch {ch} Done.')
