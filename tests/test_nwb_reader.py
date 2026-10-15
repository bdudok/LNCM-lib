'''
Tests for NWB_converter.NWBImagingSession: export test sessions with the lean config, load them with
NWBImagingSession, and compare with ImagingSession loaded from the processed folder.
Sessions: Dag47 (behavior events), Sncg248 (opto, custom vm trace), BL6-47 (dual channel, LFP, seizures, ripples).
Test data: see test_nwb_export.py. Skipped if the folder is missing.
'''
import os
from pathlib import Path

import numpy
import pytest
import yaml
from pynwb import validate

from NWB_converter.ExportImagingSession import ExportImagingSession
from NWB_converter.NWBExporter import NWBExporter
from NWB_converter.NWBImagingSession import NWBImagingSession
from Proc2P.Analysis.AnalysisClasses import EventMasks
from Proc2P.Analysis.ImagingSession import ImagingSession

DEFAULT_DATA_DIR = Path('/test_data') if os.path.isdir('/test_data') else Path(__file__).resolve().parents[2] / '_test_data'
DATA_DIR = Path(os.environ.get('LNCM_TEST_DATA', DEFAULT_DATA_DIR))
PROCPATH = DATA_DIR / '_Processed'
LEAN_CONFIG = Path(__file__).resolve().parents[1] / 'NWB_converter' / 'export_config_template_2ch-lean.yaml'

RED = {'name': 'Ch1_Red', 'emission_lambda': 595.0, 'indicator': 'jRGECO1a'}
GREEN = {'name': 'Ch2_Green', 'emission_lambda': 525.0, 'indicator': 'green'}
SESSIONS = {
    'behavior': ('Dag47_2026-04-22_goal_128', [RED, GREEN],
                 {'tag': 'CellsRe', 'source': 'S2P', 'weighted': False, 'channels': ['Ch1_Red'],
                  'traces': ['trace', 'rel', 'ntr', 'smtr']}),
    'opto': ('Sncg248_2026-10-01_CRISPR_opto_vehicle_085', [GREEN],
             {'tag': 'CellsGR', 'source': 'GEVIReg', 'weighted': False, 'channels': ['Ch2_Green'],
              'traces': ['trace', 'rel', 'ntr', 'smtr', 'vm']}),
    'lfp': ('BL6-47_2025-10-09_PlaceField_087', [RED, GREEN],
            {'tag': 'SLROI', 'source': 'S2P', 'weighted': False, 'channels': ['Ch2_Green', 'Ch1_Red'],
             'traces': ['trace', 'rel', 'ntr', 'smtr']}),
}

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not all((PROCPATH / prefix).exists() for prefix, _, _ in SESSIONS.values()),
                                 reason=f'test data not found in {DATA_DIR}')]


@pytest.fixture(scope='module')
def nwb_folder(tmp_path_factory):
    '''lean export of the test sessions, with LFP'''
    folder = tmp_path_factory.mktemp('nwb')
    for prefix, channels, roi_set in SESSIONS.values():
        with open(LEAN_CONFIG, 'r') as f:
            config = yaml.safe_load(f)
        config['output'].update(directory=str(folder), filename='{prefix}.nwb')
        config['session'].update(use_db=False, procpath=str(PROCPATH), imaging_session_kwargs={'tag': roi_set['tag']})
        config['imaging']['channels'] = channels
        config['roi_sets'] = [roi_set]
        config['include']['lfp'] = True
        config['include']['abspos'] = False
        config_path = folder / f'{prefix}.yaml'
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        session = ExportImagingSession(PROCPATH, prefix, tag=roi_set['tag'])
        # SessionInfo stores the raw data path of the machine that did the preprocessing
        session.si.info['dpath'] = str(DATA_DIR / '_RawData' / f'{prefix}-000')
        path = NWBExporter(prefix, config_path, session=session).run()
        assert validate(path=path) == []
    return folder


def load_pair(nwb_folder, key, **kwargs):
    prefix, _, roi_set = SESSIONS[key]
    original = ImagingSession(PROCPATH, prefix, tag=roi_set['tag'], **kwargs)
    return original, NWBImagingSession(nwb_folder, prefix, tag=roi_set['tag'], **kwargs)


def assert_same(a, b):
    numpy.testing.assert_array_equal(numpy.asarray(a), numpy.asarray(b))


@pytest.mark.parametrize('key', list(SESSIONS))
def test_session(nwb_folder, key):
    a, b = load_pair(nwb_folder, key)
    assert_same(a.ftimes, b.ftimes)
    assert a.fps == b.fps
    assert (a.ca.cells, a.ca.frames, a.dualch) == (b.ca.cells, b.ca.frames, b.dualch)
    for name in ('trace', 'rel', 'ntr', 'smtr'):
        assert_same(getattr(a.ca, name), getattr(b.ca, name))
    for name in ('speed', 'smspd', 'pos', 'laps', 'movement', 'relpos'):
        assert_same(getattr(a.pos, name), getattr(b.pos, name))
    assert a.has_behavior == b.has_behavior
    assert getattr(a, 'bdat', None) == getattr(b, 'bdat', None)
    assert_same(a.get_mobility_periods(), b.get_mobility_periods())
    assert_same(a.startstop(), b.startstop())
    assert_same(a.frametotime(numpy.arange(0, a.ca.frames, 7), 'absoluteTime'),
                b.frametotime(numpy.arange(0, b.ca.frames, 7), 'absoluteTime'))
    assert a.timetoframe(12.345) == b.timetoframe(12.345)
    assert_same(a.qc(), b.qc())
    assert_same(a.get_preview(), b.get_preview())
    for p, q in zip(a.rois.polys.data, b.rois.polys.data, strict=True):
        assert_same(p, q)
    assert (a.opto is None) == (b.opto is None)
    temp_folder = b.temp_folder
    b.close()
    assert not temp_folder.exists()


def test_behavior_events(nwb_folder):
    a, b = load_pair(nwb_folder, 'behavior')
    assert len(b.bdat['licks']) and len(b.bdat['RZ'])
    assert a.bdat == b.bdat


def test_opto(nwb_folder):
    a, b = load_pair(nwb_folder, 'opto')
    assert_same(a.opto, b.opto)
    assert_same(a.opto_ints, b.opto_ints)
    assert_same(a.getparam('vm'), b.getparam('vm'))
    assert_same(a.getparam('vm', mask_stim=4), b.getparam('vm', mask_stim=4))
    for x, y in zip(EventMasks.PhotoStimPulse(a, 288, mask_stim=(0, 4)),
                    EventMasks.PhotoStimPulse(b, 288, mask_stim=(0, 4)), strict=True):
        assert_same(x, y)


@pytest.mark.parametrize('ch', [0, 1, 'All'])
def test_channels(nwb_folder, ch):
    a, b = load_pair(nwb_folder, 'lfp', ch=ch)
    assert_same(a.ca.rel, b.ca.rel)


@pytest.mark.parametrize('lfp_ch', [1, 2])
def test_lfp(nwb_folder, lfp_ch):
    a, b = load_pair(nwb_folder, 'lfp', lfp_ch=lfp_ch)
    assert_same(a.ephys.trace, b.ephys.trace)
    assert_same(a.ephys.frames, b.ephys.frames)
    samples = numpy.arange(0, len(a.ephys.frames), 1001)
    assert_same(a.sampletoframe(samples), b.sampletoframe(samples))


def test_lfp_events(nwb_folder):
    a, b = load_pair(nwb_folder, 'lfp')
    a.map_seizuretimes()
    b.map_seizuretimes()
    for x, y in zip(a.sztimes, b.sztimes, strict=True):
        assert_same(x, y)
    a.map_spiketimes()
    b.map_spiketimes()
    for x, y in zip(a.spiketimes, b.spiketimes, strict=True):
        assert_same(x, y)
    # the original folder has the rate saved; the NWB session computes it from the spike times
    numpy.testing.assert_allclose(a.map_instrate(), b.map_instrate(), rtol=0, atol=1e-9)
    for function, kwargs in ((EventMasks.curated_sz_times, {}), (EventMasks.LFPSpikes, {})):
        for x, y in zip(function(a, 50, **kwargs), function(b, 50, **kwargs), strict=True):
            assert_same(x, y)


def test_ripples(nwb_folder):
    '''stored ripple events are loaded; filtered LFP and band power are recomputed'''
    a, b = load_pair(nwb_folder, 'lfp', norip=False)
    assert_same([e.p for e in a.ripples.events], [e.p for e in b.ripples.events])
    assert_same(a.ripple_frames, b.ripple_frames)
    for name in ('ripple_power', 'theta_power', 'hf_power'):
        numpy.testing.assert_allclose(getattr(a.ephys, name), getattr(b.ephys, name), rtol=1e-9, atol=1e-12)


def test_psth(nwb_folder, tmp_path):
    '''PSTH pulls the same traces with ImagingSession and with NWBImagingSession (set with session_class)'''
    from Proc2P.Analysis.AnalysisClasses.PSTH import PSTH
    prefix, _, roi_set = SESSIONS['opto']
    results = []
    for name, path, session_kwargs in (('original', PROCPATH, {}),
                                       ('nwb', nwb_folder, {'session_class': NWBImagingSession})):
        (tmp_path / name).mkdir()
        e = PSTH(tmp_path / name, 'test')
        e.add_items(prefix, roi_set['tag'], [0, 1, 2], grouping=(prefix,), group_names=('Prefix',), path=path)
        e.save_items()
        e.ready()
        e.set_session_kwargs(**session_kwargs)
        e.set_pull_params(param_key='vm', window_w=288)
        e.set_triggers(EventMasks.PhotoStimPulse, 'OptoPulse', mask_stim=(0, 4))
        e.pull_traces()
        results.append(e.get_data())
    assert len(results[0])
    numpy.testing.assert_array_equal(*results)
