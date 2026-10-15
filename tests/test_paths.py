'''
Path handling of the core classes: a folder given as str, as str with a trailing separator, or as Path gives the
same paths and files.
The real-data tests use the example sessions in <test data>/_Processed, where the test data folder is
LNCM_TEST_DATA, or /test_data, or the _test_data folder next to LNCM-lib. They are skipped if it is missing.
'''
import json
import os
from pathlib import Path

import numpy
import pytest

from Proc2P.Analysis.AnalysisClasses.DataSet import DataSet
from Proc2P.Analysis.AnalysisClasses.PSTH import PSTH
from Proc2P.Analysis.CaTrace import CaTrace
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Bruker.LoadRegistered import LoadRegistered

PREFIX = 'Test1_2026-01-01_paths_001'
DEFAULT_DATA_DIR = Path('/test_data') if os.path.isdir('/test_data') else Path(__file__).resolve().parents[2] / '_test_data'
TEST_DATA = Path(os.environ.get('LNCM_TEST_DATA', DEFAULT_DATA_DIR))
REAL_PROCPATH = TEST_DATA / '_Processed'
REAL_SESSIONS = sorted(p.name for p in REAL_PROCPATH.iterdir() if p.is_dir()) if REAL_PROCPATH.is_dir() else []


def path_forms(folder):
    '''the same folder as str, as str with a trailing separator, and as Path'''
    return [str(folder), str(folder) + os.sep, Path(folder)]


@pytest.fixture
def procpath(tmp_path):
    '''processed folder with one minimal session: session info, frame times, an ROI set and its trace folder'''
    sdir = tmp_path / PREFIX
    sdir.mkdir()
    with open(sdir / f'{PREFIX}_SessionInfo.json', 'w') as f:
        json.dump({'framerate': 10.0, 'n_frames': 50, 'has_opto': False}, f)
    numpy.save(sdir / f'{PREFIX}_FrameTimes.npy', numpy.zeros((50, 2)))
    numpy.save(sdir / f'{PREFIX}_saved_roi_A.npy', numpy.array([[0, 1, 1], [0, 4, 1], [0, 4, 4]]))
    ca = CaTrace(tmp_path, PREFIX, tag='A', ch=0)
    for key in ca.keys:
        setattr(ca, key, numpy.arange(100.0).reshape(2, 50))
    ca.bsltype, ca.fps, ca.cells = 'poly', 10.0, 2
    ca.save()
    return tmp_path


def test_catrace_files(procpath):
    pf = procpath / PREFIX / f'{PREFIX}_trace_A-ch0'
    assert sorted(os.listdir(pf)) == sorted(f'{key}.npy' for key in ('bsl', 'rel', 'ntr', 'trace', 'smtr', 'info'))


@pytest.mark.parametrize('form', range(3))
def test_imaging_session_paths(procpath, form):
    a = ImagingSession(path_forms(procpath)[form], PREFIX, tag='A')
    sdir = procpath / PREFIX
    assert a.path == sdir
    assert a.get_file_with_suffix('_FrameTimes.npy') == sdir / f'{PREFIX}_FrameTimes.npy'
    assert a.get_face_path() == sdir / '_face' and a.face_path.is_dir()
    assert a.ca.pf == sdir / f'{PREFIX}_trace_A-ch0'
    numpy.testing.assert_array_equal(a.ca.rel, numpy.arange(100.0).reshape(2, 50))


@pytest.mark.parametrize('form', range(3))
def test_psth_cache_folder(tmp_path, form):
    e = PSTH(path_forms(tmp_path)[form], 'test')
    assert e.wdir == tmp_path / 'EventTrigger' and e.wdir.is_dir()
    assert e.lock_name == tmp_path / 'EventTrigger' / '_test.lock'


@pytest.mark.parametrize('form', range(3))
def test_dataset_file(tmp_path, form):
    ds = DataSet(path_forms(tmp_path)[form])
    assert ds.get_fn() == tmp_path / '_dataset00.feather'


def trace_tags(prefix):
    '''ROI set tags that have a trace folder for the first channel'''
    start, end = f'{prefix}_trace_', '-ch0'
    return sorted(f[len(start):-len(end)] for f in os.listdir(REAL_PROCPATH / prefix)
                  if f.startswith(start) and f.endswith(end))


@pytest.mark.slow
@pytest.mark.skipif(not REAL_SESSIONS, reason=f'test data not found in {REAL_PROCPATH}')
@pytest.mark.parametrize('prefix', REAL_SESSIONS)
def test_real_session_loads_the_same_from_each_path_form(prefix):
    tag = trace_tags(prefix)[0]
    sessions = [ImagingSession(form, prefix, tag=tag) for form in path_forms(REAL_PROCPATH)]
    for a in sessions:
        assert a.path == REAL_PROCPATH / prefix
        assert a.ca.pf == REAL_PROCPATH / prefix / f'{prefix}_trace_{tag}-ch0'
        numpy.testing.assert_array_equal(a.ca.rel, sessions[0].ca.rel)
        numpy.testing.assert_array_equal(a.pos.speed, sessions[0].pos.speed)


@pytest.mark.slow
@pytest.mark.skipif(not REAL_SESSIONS, reason=f'test data not found in {REAL_PROCPATH}')
@pytest.mark.parametrize('prefix', REAL_SESSIONS)
def test_real_movie_files_found(prefix):
    for form in path_forms(REAL_PROCPATH):
        movie = LoadRegistered(form, prefix)
        assert movie.input_files and all(f.parent == REAL_PROCPATH / prefix for f in movie.input_files)
