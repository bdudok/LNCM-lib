'''
PSTH item handling: items are ready to use after saving and after loading a locked PSTH, the number of included items
is printed, and a too long save path gives a clear error.
'''
import json
import os
import zlib
from pathlib import Path

import numpy
import pytest

from Proc2P.Analysis.AnalysisClasses.PSTH import PSTH

GROUP_NAMES = ('Prefix', 'Animal', 'SessionType')


def make_psth(path):
    e = PSTH(path, 'test')
    e.add_items('mouse1_veh_001', 'cells', [0, 1, 2], grouping=('mouse1_veh_001', 'mouse1', 'Veh'),
                group_names=GROUP_NAMES)
    e.add_items('mouse1_drug_002', 'cells', [0, 1], grouping=('mouse1_drug_002', 'mouse1', 'Drug'),
                group_names=GROUP_NAMES)
    e.add_items('mouse2_veh_003', 'cells', [4], grouping=('mouse2_veh_003', 'mouse2', 'Veh'),
                group_names=GROUP_NAMES)
    return e


def test_ready_after_save(tmp_path, capsys):
    e = make_psth(tmp_path)
    e.save_items()
    assert e.group_df is not None
    assert len(e.pull_dict) == 3
    assert e.get_groups_by('Animal') == ['mouse1', 'mouse2']
    assert ('PSTH test: 6 cells in 3 sessions. Unique values per group: Prefix 3, Animal 2, SessionType 2'
            in capsys.readouterr().out)


def test_ready_when_loaded_locked(tmp_path, capsys):
    make_psth(tmp_path).save_items()
    capsys.readouterr()
    e = PSTH(tmp_path, 'test')
    assert e.locked
    assert e.get_groups_by('SessionType') == ['Drug', 'Veh']
    assert 'PSTH test: 6 cells in 3 sessions.' in capsys.readouterr().out


def test_long_path_error(tmp_path, monkeypatch):
    def fail(fn, data):
        raise FileNotFoundError(2, 'No such file or directory', str(fn))
    monkeypatch.setattr(numpy, 'save', fail)
    long_name = tmp_path / ('x' * 260 + '.npy')
    with pytest.raises(FileNotFoundError, match='longer than the Windows path limit'):
        PSTH.save_array(long_name, numpy.zeros(3))
    # short paths raise the original error
    with pytest.raises(FileNotFoundError, match='No such file or directory'):
        PSTH.save_array(tmp_path / 'missing' / 'a.npy', numpy.zeros(3))


def test_single_group_name(tmp_path):
    e = PSTH(tmp_path, 'single')
    e.add_items('mouse1_veh_001', 'cells', 3, grouping='Veh', group_names='SessionType')
    e.save_items()
    assert e.group_names == ['SessionType']
    assert e.group_df['SessionType'].tolist() == ['Veh']


def test_grouping_length_checked(tmp_path):
    e = PSTH(tmp_path, 'test')
    with pytest.raises(ValueError, match='2 values for 3 group names'):
        e.add_items('mouse1_veh_001', 'cells', 0, grouping=('mouse1_veh_001', 'mouse1'), group_names=GROUP_NAMES)


def test_cell_ids_are_stable(tmp_path):
    '''cell ids don't depend on the Python process (crc32, not hash)'''
    e = make_psth(tmp_path)
    assert zlib.crc32(b'mouse1_veh_001cells0') % (10 ** 8) * 10000 + 2 in e.item_dict


def with_data(tmp_path):
    '''PSTH with synthetic pulled traces: window 2 (4 samples), each event row is constant'''
    e = make_psth(tmp_path)
    e.save_items()
    e.window_w = 2
    rows = []
    # cell 0 of the first session: 1 event of 1; cell 1: 3 events of 3; cell 2: no events
    for cell_index, trace_index, values in ((0, 0, [1]), (1, 1, [3, 3, 3])):
        for v in values:
            rows.append([v] * 4 + [cell_index, trace_index])
    e.data = numpy.array(rows, dtype='float')
    return e


def test_group_by_weights_cells(tmp_path):
    e = with_data(tmp_path)
    data = e.get_data(group_by='Prefix')
    # mouse1_drug_002 and mouse2_veh_003 have no events; mouse1_veh_001 is the mean of the cell averages (1, 3)
    groups = e.get_groups_by('Prefix')
    numpy.testing.assert_array_equal(data[groups.index('mouse1_veh_001')], [2, 2, 2, 2])
    assert numpy.isnan(data[groups.index('mouse2_veh_003')]).all()


def test_avg_cells_rows(tmp_path):
    e = with_data(tmp_path)
    data = e.get_data(group_criteria=(('Prefix', 'mouse1_veh_001'),), avg_cells=True)
    numpy.testing.assert_array_equal(data, [[1] * 4, [3] * 4])


def test_empty_group_with_baseline(tmp_path):
    e = with_data(tmp_path)
    assert e.get_data(group_criteria=(('Prefix', 'none'),), avg_cells=True, baseline=slice(0, 1)) is None


def test_fingerprint_check(tmp_path):
    e = make_psth(tmp_path)
    e.save_items()
    e.set_triggers(lambda a, w: None, 'Test', mask_stim=(0, 4))
    session_data, cell_data = next(iter(e.pull_dict.items()))
    fn = tmp_path / 'traces.npy'
    fingerprint = e.get_fingerprint(cell_data, None, None, None, False)
    # traces saved before fingerprints were added are used as before
    e.check_fingerprint(fn, fingerprint)
    with open(e.fingerprint_name(fn), 'w') as f:
        json.dump(fingerprint, f)
    e.check_fingerprint(fn, fingerprint)
    with pytest.raises(ValueError, match='mask_stim: saved None, now 3'):
        e.check_fingerprint(fn, e.get_fingerprint(cell_data, 3, None, None, False))
    with pytest.raises(ValueError, match='the cells differ'):
        e.check_fingerprint(fn, e.get_fingerprint(dict(list(cell_data.items())[:1]), None, None, None, False))


DATA_DIR = Path(os.environ.get('LNCM_TEST_DATA', '/test_data'))
OPTO_PREFIX = 'Sncg248_2026-10-01_CRISPR_opto_vehicle_085'


@pytest.mark.slow
@pytest.mark.skipif(not (DATA_DIR / '_Processed' / OPTO_PREFIX).exists(), reason='test data not found')
def test_pull_and_fingerprint(tmp_path):
    '''pulled traces are saved with a fingerprint; loading them with another setting raises'''
    from Proc2P.Analysis.AnalysisClasses import EventMasks

    def setup():
        e = PSTH(tmp_path, 'opto')
        if not e.locked:
            e.add_items(OPTO_PREFIX, 'CellsGR', [0, 1, 2], grouping=OPTO_PREFIX, group_names='Prefix',
                        path=DATA_DIR / '_Processed')
            e.save_items()
        e.set_pull_params(param_key='vm', window_w=288)
        e.set_triggers(EventMasks.PhotoStimPulse, 'OptoPulse', mask_stim=(0, 4))
        return e

    e = setup()
    e.pull_traces()
    data = e.get_data(avg_cells=True, group_criteria=(('Prefix', OPTO_PREFIX),))
    assert data.shape == (3, 576)
    fn = e.get_numpy_name(OPTO_PREFIX, 'CellsGR', 0)
    assert e.fingerprint_name(fn).exists()
    # same settings: loaded
    e = setup()
    e.pull_traces()
    numpy.testing.assert_array_equal(e.get_data(avg_cells=True, group_criteria=(('Prefix', OPTO_PREFIX),)), data)
    # other mask_stim (not in the file name): raises instead of loading the old traces
    e = setup()
    with pytest.raises(ValueError, match='mask_stim: saved None, now 2'):
        e.pull_traces(mask_stim=2)
