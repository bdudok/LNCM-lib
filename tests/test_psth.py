'''
PSTH item handling: items are ready to use after saving and after loading a locked PSTH, the number of included items
is printed, and a too long save path gives a clear error.
'''
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
