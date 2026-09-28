'''
Tests for NWB_converter, run on one real session with the lean export config (no 2P movie, video or LFP).
Run from the LNCM-lib folder: python -m pytest
Test data: the _test_data folder next to LNCM-lib, or set the LNCM_TEST_DATA environment variable.
Skipped if the folder is missing.
'''
import itertools
import json
import os
from pathlib import Path

import numpy
import pandas
import pytest
import yaml
from pynwb import NWBHDF5IO, validate

from BaserowAPI.BaserowRequests import GetSessions
from NWB_converter.ExportImagingSession import ExportImagingSession
from NWB_converter.NWBExporter import NWBExporter
from Proc2P.Analysis.LoadPolys import load_roi_file

PREFIX = 'Dag47_2026-04-22_goal_128'
DATA_DIR = Path(os.environ.get('LNCM_TEST_DATA', Path(__file__).resolve().parents[2] / '_test_data'))
PROCPATH = DATA_DIR / '_Processed'
RAWPATH = DATA_DIR / '_RawData' / f'{PREFIX}-000'
# the session's Baserow row (pickled pandas Series) and the mouse's name and sex from the mice table
DB_ENTRY = DATA_DIR / 'NWB_export' / f'{PREFIX}_baserow_entry.pkl'
DB_MOUSE = DATA_DIR / 'NWB_export' / f'{PREFIX}_baserow_mouse.json'
LEAN_CONFIG = Path(__file__).resolve().parents[1] / 'NWB_converter' / 'export_config_template_2ch-lean.yaml'

pytestmark = pytest.mark.skipif(not all(p.exists() for p in (PROCPATH, RAWPATH, DB_ENTRY, DB_MOUSE)),
                                reason=f'test data not found in {DATA_DIR}')


@pytest.fixture(scope='module')
def mouse():
    with open(DB_MOUSE, 'r') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def db_entry():
    return pandas.read_pickle(DB_ENTRY)


@pytest.fixture(scope='module')
def session(db_entry, mouse):
    '''
    ExportImagingSession of the test data, with the raw data paths pointing to the test data folder,
    and the saved DB row and mice table result instead of DB requests
    '''
    with open(LEAN_CONFIG, 'r') as f:
        kwargs = yaml.safe_load(f)['session']['imaging_session_kwargs']
    session = ExportImagingSession(str(PROCPATH) + os.sep, PREFIX, db_entry=db_entry, **kwargs)
    # SessionInfo stores the absolute raw data paths of the machine that did the preprocessing
    session.si.info['dpath'] = str(RAWPATH)
    session.si.info['cam_file'] = str(RAWPATH / f'{PREFIX}.avi')
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GetSessions, 'get_mouse', lambda self, item=None, mtag=None, ret_sex=False: mouse['sex'])
        yield session


def export(session, output_dir, roi_sets=True, video_storage='external', dandiset_path=None, **include):
    '''
    run the exporter with the lean config, writing to output_dir.
    roi_sets: False to leave out the ROI sets. include: overrides of the include section
    '''
    with open(LEAN_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    config['output']['directory'] = str(output_dir)
    config['session']['use_db'] = False
    config['include'].update(include)
    if not roi_sets:
        config['roi_sets'] = []
    config['video']['storage'] = video_storage
    config['dandi']['dandiset_path'] = dandiset_path
    config_path = Path(output_dir) / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    return NWBExporter(PREFIX, config_path, session=session).run()


@pytest.fixture(scope='module')
def lean_path(session, tmp_path_factory):
    return export(session, tmp_path_factory.mktemp('lean'))


@pytest.fixture(scope='module')
def nwbfile(lean_path):
    with NWBHDF5IO(lean_path, 'r') as io:
        yield io.read()


def test_file_is_valid(lean_path):
    assert validate(path=lean_path) == []


def test_lean_contents(nwbfile):
    assert list(nwbfile.acquisition) == ['ImagingFrameTimes']
    assert set(nwbfile.processing['behavior']['BehavioralTimeSeries'].time_series) == {'speed', 'position'}


def test_session_start_time(nwbfile, session):
    assert nwbfile.session_start_time.isoformat()[:19] == session.get_starttime()


def test_subject(nwbfile, mouse):
    assert nwbfile.subject.subject_id == mouse['mouse_name']
    assert nwbfile.subject.sex == mouse['sex']
    assert nwbfile.subject.species == 'Mus musculus'


def test_notes(nwbfile, session, db_entry):
    notes = json.loads(nwbfile.notes)
    assert set(notes) == {'SessionInfo', 'Baserow'}
    assert notes['SessionInfo']['framerate'] == session.si['framerate']
    assert notes['Baserow'] == json.loads(json.dumps(db_entry.to_dict(), default=str))
    assert nwbfile.session_id == f'LNCM-{db_entry["id"]}-{PREFIX}'


def test_frame_times(nwbfile, session):
    frame_times = nwbfile.acquisition['ImagingFrameTimes']
    numpy.testing.assert_array_equal(frame_times.timestamps[:], session.ftimes[:, 0])
    speed = nwbfile.processing['behavior']['BehavioralTimeSeries']['speed']
    # stored as a link to the frame times, not a copy
    assert speed.timestamps.name == '/acquisition/ImagingFrameTimes/timestamps'
    numpy.testing.assert_array_equal(speed.data[:], session.pos.speed)


def test_traces(nwbfile, session):
    n_rois = len(nwbfile.processing['ophys']['ImageSegmentation']['CellsRe'])
    fluorescence = nwbfile.processing['ophys']['CellsRe']
    assert set(fluorescence.roi_response_series) == {'trace_Ch1_Red', 'rel_Ch1_Red'}
    rel = fluorescence['rel_Ch1_Red'].data[:]
    assert rel.shape == (len(session.ftimes), n_rois)
    numpy.testing.assert_array_equal(rel, session.export_traces('CellsRe', 'rel'))


def test_roi_masks_orientation(nwbfile):
    '''each mask's centroid is at its ROI polygon's centroid, in [row (y), column (x)] indexing'''
    masks = nwbfile.processing['ophys']['ImageSegmentation']['CellsRe']['image_mask'][:]
    fov = nwbfile.processing['ophys']['FieldOfView']['FieldOfView_RGB'].data
    assert masks.shape[1:] == fov.shape[:2]
    polys = load_roi_file(PROCPATH / PREFIX / f'{PREFIX}_saved_roi_CellsRe.npy')
    assert len(polys) == len(masks)
    for mask, poly in zip(masks, polys):
        rows, cols = numpy.nonzero(mask)
        x, y = numpy.mean(poly, axis=0)
        assert abs(rows.mean() - y) < 3
        assert abs(cols.mean() - x) < 3


def test_position(nwbfile, session):
    position = nwbfile.processing['behavior']['BehavioralTimeSeries']['position']
    assert position.timestamps.name == '/acquisition/ImagingFrameTimes/timestamps'
    assert len(position.data) == len(session.ftimes)
    assert numpy.nanmin(position.data[:]) >= 0
    assert numpy.nanmax(position.data[:]) <= 1


def test_external_video_export(session, tmp_path):
    '''the video is linked by file name, and the DANDI upload file lists the paths'''
    dandiset_path = tmp_path / '000000'
    path = export(session, tmp_path, dandiset_path=str(dandiset_path), video=True)
    assert validate(path=path) == []
    with NWBHDF5IO(path, 'r') as io:
        acquisition = io.read().acquisition
        video = acquisition['Video']
        assert list(video.external_file[:]) == [f'{PREFIX}.avi']
        assert list(video.starting_frame[:]) == [0]
        assert video.format == 'external'
        assert video.data is None or len(video.data) == 0
        numpy.testing.assert_array_equal(video.timestamps[:], session.export_video_times())
        video_frames = acquisition['VideoImagingFrame']
        assert video_frames.timestamps.name == '/acquisition/Video/timestamps'
        numpy.testing.assert_array_equal(video_frames.data[:], session.export_video_frames())
    with open(Path(path).parent / (Path(path).stem + '_dandi.json'), 'r') as f:
        upload = json.load(f)
    assert upload['nwb_file'] == str(path)
    assert upload['video_source'] == str(RAWPATH / f'{PREFIX}.avi')
    assert upload['dandiset_path'] == str(dandiset_path)
    assert len(upload['powershell_commands']) == 4


def test_movie_and_video_export(session, tmp_path, monkeypatch):
    '''2P movie and video from the real files, cut to the first 100 frames of each'''
    n = 100
    movies = [movie[:n] for movie in session.export_2p_movie()]
    video = list(itertools.islice(session.export_video(), n))
    video_times = session.export_video_times()[:n]
    video_frames = session.export_video_frames()[:n]
    monkeypatch.setattr(session, 'export_frame_times', lambda: session.ftimes[:n, 0])
    monkeypatch.setattr(session, 'export_2p_movie', lambda: movies)
    monkeypatch.setattr(session, 'export_video', lambda: iter(video))
    monkeypatch.setattr(session, 'export_video_times', lambda: video_times)
    monkeypatch.setattr(session, 'export_video_frames', lambda: video_frames)
    path = export(session, tmp_path, roi_sets=False, video_storage='embedded', speed=False, abspos=False,
                  **{'2p_movie': True, 'video': True})
    assert validate(path=path) == []
    with NWBHDF5IO(path, 'r') as io:
        acquisition = io.read().acquisition
        for name, movie in zip(['TwoPhotonSeries_Ch1_Red', 'TwoPhotonSeries_Ch2_Green'], movies):
            series = acquisition[name]
            assert series.timestamps.name == '/acquisition/ImagingFrameTimes/timestamps'
            numpy.testing.assert_array_equal(series.data[:], movie)
        numpy.testing.assert_array_equal(acquisition['Video'].data[:], numpy.stack(video))
        numpy.testing.assert_array_equal(acquisition['Video'].timestamps[:], video_times)
        numpy.testing.assert_array_equal(acquisition['VideoImagingFrame'].data[:], video_frames)


def test_video_times(session):
    '''video frame times are in seconds and match the imaging frame each video frame was synced to'''
    video_times = session.export_video_times()
    assert numpy.all(numpy.diff(video_times) > 0)
    # PreProc's sync: the nearest imaging frame of each video frame, 0 if none
    sync_frames = session.ca.sync.load('cam')
    assert len(video_times) == len(sync_frames)
    matched = sync_frames > 0
    errors = video_times[matched] - session.ftimes[sync_frames[matched], 0]
    assert numpy.abs(errors).max() < 1 / session.fps


def test_video_frames(session):
    '''each video frame's time is within the imaging frame it's assigned to'''
    video_times = session.export_video_times()
    frames = session.export_video_frames()
    frame_times = session.ftimes[:, 0]
    inside = frames >= 0
    assert inside.sum() > 0.99 * len(frames)
    assert numpy.all(video_times[inside] >= frame_times[frames[inside]])
    assert numpy.all(video_times[inside] < frame_times[frames[inside]] + 1 / session.fps)
    # the nearest imaging frame (PreProc's sync) is the same or the next one
    sync_frames = session.ca.sync.load('cam')
    assert set(numpy.unique(sync_frames[inside] - frames[inside])) <= {0, 1}
