from types import SimpleNamespace

import numpy
import pandas

from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Bruker.LoadEphys import Ephys

PREFIX = 'Synthetic_001'
FS = 40  # LFP samples per second
SAMPLES_PER_FRAME = 4


def make_session(tmp_path):
    '''session stub with one LFP channel whose int16 frame row overflows, and spike and seizure files'''
    frames = numpy.repeat(numpy.arange(40000), SAMPLES_PER_FRAME)
    stored = numpy.zeros((2, len(frames)), dtype='int16')
    stored[0] = frames.astype('int16')  # wraps from 32767 to -32768
    folder = tmp_path / PREFIX
    folder.mkdir()
    numpy.save(folder / f'{PREFIX}_ephys.npy', stored)
    pandas.DataFrame({'SpikeTimes(s)': [100.0, 3500.0]}).to_excel(folder / f'{PREFIX}_Ch1_spiketimes.xlsx')
    pandas.DataFrame({'Sz.Start(s)': [3500.0], 'Sz.Stop(s)': [3600.0]}).to_excel(
        folder / f'{PREFIX}_Ch1_seizure_times.xlsx')
    return SimpleNamespace(kwargs={}, ephys=Ephys(tmp_path, PREFIX, channel=1), si=SimpleNamespace(info={'fs': FS}),
                           get_file_with_suffix=lambda suffix: folder / f'{PREFIX}{suffix}')


def test_spike_and_seizure_frames_use_corrected_frames(tmp_path):
    session = make_session(tmp_path)
    ImagingSession.map_spiketimes(session)
    ImagingSession.map_seizuretimes(session)
    frames = session.ephys.frames
    samples = [100 * FS, 3500 * FS, 3600 * FS]
    assert frames[samples[1]] > 32767  # after the overflow
    assert list(session.spiketimes[0]) == [frames[samples[0]], frames[samples[1]]]
    assert [list(x) for x in session.sztimes[0]] == [[frames[samples[1]]], [frames[samples[2]]]]
