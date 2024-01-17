from datetime import datetime

import numpy
import datetime
from LFP.SpikeDet import SpikesPower
from LFP.SzDet import InstRate
from LFP.Pinnacle import ReadEDF
import pandas

def run_detection(edf:ReadEDF, opts, save_envelope=False):
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    path = edf.path
    prefix = edf.prefix
    item=opts
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
        numpy.save(path + prefix + f'_ch{ch}_envelope.npy', spikedet.env)

    # detect seizures
    kwargs = {}
    for kw, key in (('cleanup', 'Sz.MinDur'), ('gap', 'Sz.Gap'),):
        kwargs[kw] = float(item[key])
    sz_burden, sz_times = InstRate.SpikeTrace(spikes, int(item['SzDet.Framesize']), **kwargs)

    numpy.save(path + prefix + f'_ch{ch}_sz_burden.npy', sz_burden)
    op = pandas.DataFrame(sz_times, columns=('Sz.Start(s)', 'Sz.Stop(s)'))
    op.to_excel(path + prefix + f'_ch{ch}_seizure_times.xlsx')
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    print(ts, f'Ch {ch} Done.')
