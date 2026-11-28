'''
AnalysisResult.session and dataset: the same script opens the processed sessions, NWB files (nwb_path), or the
shared files (LNCM_SHARED_NWB), and saves the result in a rerun folder (LNCM_RESULTS_ROOT).
'''
import json
import shutil

import h5py
import pandas
import pytest

import NWB_converter.NWBImagingSession
import Proc2P.Analysis.ImagingSession
from Proc2P.Analysis.AnalysisClasses import AnalysisResult as ar
from Proc2P.Analysis.AnalysisClasses.AnalysisResult import AnalysisResult
from Proc2P.Analysis.AnalysisClasses.DataSet import DataSet


class Opened:
    def __init__(self, path, prefix, **kwargs):
        self.path, self.prefix, self.kwargs = path, prefix, kwargs


@pytest.fixture
def stub_sessions(monkeypatch):
    monkeypatch.setattr(Proc2P.Analysis.ImagingSession, 'ImagingSession', Opened)
    monkeypatch.setattr(NWB_converter.NWBImagingSession, 'NWBImagingSession', Opened)
    monkeypatch.delenv(ar.SHARED_NWB_ENV, raising=False)
    monkeypatch.delenv(ar.RESULTS_ROOT_ENV, raising=False)


def write_nwb(fn, session_id):
    fn.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(fn, 'w') as f:
        f.create_dataset('general/session_id', data=session_id)


def make_saved_result(project):
    '''a saved result with a DataSet of two sessions, one included'''
    project.mkdir(parents=True)
    ds = DataSet(project)
    ds.add_row({'Prefix': 's_17', 'Animal': 'm1', 'Incl': True, 'Cells.R.Incl': '[1, 2]'})
    ds.add_row({'Prefix': 's_171', 'Animal': 'm1', 'Incl': False})
    ds.save_df()
    res = AnalysisResult(project, 'Fig1', script=__file__)
    res.set_sessions(['s_17', 's_171'], dataset=ds)
    res.save()


def test_session_opens_processed_folder_and_logs(tmp_path, stub_sessions):
    res = AnalysisResult(tmp_path, 'Fig1', script=__file__)
    s = res.session('proc', 's_17', tag='R', ch=1, spiketag='AE_2', norip=True)
    assert (s.path, s.kwargs) == ('proc', {'tag': 'R', 'ch': 1, 'spiketag': 'AE_2', 'norip': True})
    res.session('proc', 's_17', tag='R', ch=1, spiketag='AE_2')
    assert res.session_log == [{'prefix': 's_17', 'tag': 'R', 'ch': 1, 'spiketag': 'AE_2'}]
    res.set_sessions(['s_17'])
    res.save()
    manifest = json.loads((tmp_path / 'Results' / 'Fig1' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['session_log'] == res.session_log
    assert manifest['nwb_path'] is None and manifest['shared'] is False


def test_session_opens_nwb_file_by_session_id(tmp_path, stub_sessions):
    nwb = tmp_path / 'nwb'
    write_nwb(nwb / 'sub-m1' / 'sub-m1_ses-LNCM-5-s-17_ophys.nwb', 'LNCM-5-s_17')
    write_nwb(nwb / 's_171.nwb', 's_171')
    res = AnalysisResult(tmp_path / 'project', 'Fig1', script=__file__, nwb_path=nwb)
    s = res.session('proc', 's_17', tag='R', spiketag='AE_2')
    assert (s.path, s.kwargs) == (nwb / 'sub-m1', {'nwb_filename': 'sub-m1_ses-LNCM-5-s-17_ophys.nwb', 'tag': 'R'})
    assert res.session('proc', 's_171').kwargs['nwb_filename'] == 's_171.nwb'
    with pytest.raises(FileNotFoundError):
        res.session('proc', 's_1')
    # the lab DataSet is used with nwb_path
    (tmp_path / 'project').mkdir(exist_ok=True)
    assert res.dataset(tmp_path / 'project').project_folder == (tmp_path / 'project').resolve()


def test_shared_rerun_uses_saved_sessions_and_rerun_folder(tmp_path, stub_sessions, monkeypatch):
    project = tmp_path / 'project'
    make_saved_result(project)
    nwb = tmp_path / 'shared_nwb'
    write_nwb(nwb / 's_17.nwb', 's_17')
    monkeypatch.setenv(ar.SHARED_NWB_ENV, str(nwb))
    monkeypatch.setenv(ar.RESULTS_ROOT_ENV, str(tmp_path / 'rerun'))
    res = AnalysisResult(project, 'Fig1', script=__file__, nwb_path=tmp_path / 'lab_nwb')
    assert res.project_folder == tmp_path / 'rerun' / 'project'
    ds = res.dataset(project)
    assert list(ds.get_incl()['Prefix']) == ['s_17']
    assert ds.get_cells('s_17', 'R') == [1, 2]
    assert res.session('proc', 's_17').path == nwb
    res.set_sessions(['s_17'], dataset=ds)
    res.save()
    assert AnalysisResult.load(project, 'Fig1', script=__file__).session_log == [{'prefix': 's_17'}]
    manifest = json.loads((tmp_path / 'rerun' / 'project' / 'Results' / 'Fig1' / 'manifest.json').read_text())
    assert manifest['shared'] is True and manifest['nwb_path'] == str(nwb)
    # the saved result is unchanged
    assert len(pandas.read_feather(project / 'Results' / 'Fig1' / 'sessions.feather')) == 2


def make_shared_repo(tmp_path):
    '''the saved result Fig1 copied to a shared repo (Figures/Figure1/Fig1), with its script in script/'''
    make_saved_result(tmp_path / 'project')
    repo = tmp_path / 'repo' / 'Figures'
    shared = repo / 'Figure1' / 'Fig1'
    shutil.copytree(tmp_path / 'project' / 'Results' / 'Fig1', shared)
    (repo / 'publication.json').write_text('{}')
    script = shared / 'script' / 'fig1.py'
    script.parent.mkdir(exist_ok=True)
    script.write_text('')
    return repo, shared, script


def test_use_nwb_reruns_the_shared_result_next_to_it(tmp_path, stub_sessions):
    '''a reader adds use_nwb to the script in script/ of a shared result'''
    repo, shared, script = make_shared_repo(tmp_path)
    nwb = tmp_path / 'downloads'
    write_nwb(nwb / 'sub-m1' / 'sub-m1_ses-s-17.nwb', 'LNCM-3-s_17')
    lab_folder = tmp_path / 'not_on_this_machine'
    res = AnalysisResult(lab_folder, 'Fig1', script=script, use_nwb=nwb)
    assert res.saved_folder() == shared
    assert res.folder == shared / 'rerun' / 'Results' / 'Fig1'
    ds = res.dataset(lab_folder)
    assert len(ds.df) == 2
    assert res.session('lab path', 's_17', tag='R').path == nwb / 'sub-m1'
    res.set_sessions(['s_17'], dataset=ds)
    res.save()
    assert (shared / 'rerun' / 'Results' / 'Fig1' / 'manifest.json').exists()
    assert not lab_folder.exists()


def test_use_nwb_needs_the_script_of_a_shared_result(tmp_path, stub_sessions):
    with pytest.raises(ValueError, match='script folder'):
        AnalysisResult(tmp_path, 'Fig1', script=__file__, use_nwb=tmp_path)


def test_shared_load_of_another_result_reads_the_shared_copy(tmp_path, stub_sessions, monkeypatch):
    '''a script that loads another result (as 4b loads 3c) reads it from the shared results, its rerun if there is one'''
    repo, shared, script = make_shared_repo(tmp_path)
    other_script = repo / 'Figure2' / 'Fig2' / 'script' / 'fig2.py'
    other_script.parent.mkdir(parents=True)
    monkeypatch.setenv(ar.SHARED_NWB_ENV, str(tmp_path))
    assert AnalysisResult.load(tmp_path / 'lab', 'Fig1', script=other_script).folder == shared
    rerun = AnalysisResult(tmp_path / 'lab', 'Fig1', script=script)
    rerun.set_sessions(['s_17'])
    rerun.save()
    loaded = AnalysisResult.load(tmp_path / 'lab', 'Fig1', script=other_script)
    assert loaded.folder == shared / 'rerun' / 'Results' / 'Fig1'
    assert list(loaded.sessions['Prefix']) == ['s_17']


def test_input_table_is_saved_and_read_back_on_the_shared_files(tmp_path, stub_sessions):
    project = tmp_path / 'project'
    project.mkdir()
    pandas.DataFrame({'Animal': ['m1', 'm2'], 'Dur': [3.5, 4.0]}).to_excel(project / 'triggers.xlsx', index=False)
    res = AnalysisResult(project, 'Fig1', script=__file__)
    table = res.input_table('triggers', project / 'triggers.xlsx', 'trigger table')
    res.set_sessions(['s_17'])
    res.save()
    manifest = json.loads((project / 'Results' / 'Fig1' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['inputs'][0]['rows'] == 2
    shared = tmp_path / 'repo' / 'Figure1' / 'Fig1'
    shutil.copytree(project / 'Results' / 'Fig1', shared)
    (tmp_path / 'repo' / 'publication.json').write_text('{}')
    script = shared / 'script' / 'fig1.py'
    script.write_text('')
    (project / 'triggers.xlsx').unlink()
    rerun = AnalysisResult(project, 'Fig1', script=script, use_nwb=tmp_path)
    assert rerun.input_table('triggers', project / 'triggers.xlsx').equals(table)
