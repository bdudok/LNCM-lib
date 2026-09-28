'''
Tests for NWB_converter, run on one real session with the lean export config (no 2P movie, video or LFP).
Run from the LNCM-lib folder: python -m pytest
Test data: the _test_data folder next to LNCM-lib, or set the LNCM_TEST_DATA environment variable.
Skipped if the folder is missing.
'''
import os
from pathlib import Path

import numpy
import pytest
import yaml
from pynwb import NWBHDF5IO, validate

from NWB_converter.ExportImagingSession import ExportImagingSession
from NWB_converter.NWBExporter import NWBExporter
from Proc2P.Analysis.LoadPolys import load_roi_file

PREFIX = 'Dag47_2026-04-22_goal_128'
DATA_DIR = Path(os.environ.get('LNCM_TEST_DATA', Path(__file__).resolve().parents[2] / '_test_data'))
PROCPATH = DATA_DIR / '_Processed'
RAWPATH = DATA_DIR / '_RawData' / f'{PREFIX}-000'
LEAN_CONFIG = Path(__file__).resolve().parents[1] / 'NWB_converter' / 'export_config_template_2ch-lean.yaml'

pytestmark = pytest.mark.skipif(not (PROCPATH.exists() and RAWPATH.exists()),
                                reason=f'test data not found in {DATA_DIR}')


@pytest.fixture(scope='module')
def session():
    '''ExportImagingSession of the test data, with the raw data paths pointing to the test data folder'''
    with open(LEAN_CONFIG, 'r') as f:
        kwargs = yaml.safe_load(f)['session']['imaging_session_kwargs']
    session = ExportImagingSession(str(PROCPATH) + os.sep, PREFIX, **kwargs)
    # SessionInfo stores the absolute raw data paths of the machine that did the preprocessing
    session.si.info['dpath'] = str(RAWPATH)
    session.si.info['cam_file'] = str(RAWPATH / f'{PREFIX}.avi')
    return session


def export(session, output_dir, **include):
    '''run the exporter with the lean config, writing to output_dir. include: overrides of the include section'''
    with open(LEAN_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    config['output']['directory'] = str(output_dir)
    config['session']['use_db'] = False
    config['include'].update(include)
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


def test_video_export(session, tmp_path, monkeypatch):
    '''video with the real frame times, but tiny frames instead of reading the video file'''
    n_video_frames = len(session.export_video_times())
    monkeypatch.setattr(session, 'export_video', lambda: numpy.zeros((n_video_frames, 2, 2), dtype='uint8'))
    path = export(session, tmp_path, video=True)
    assert validate(path=path) == []
    with NWBHDF5IO(path, 'r') as io:
        acquisition = io.read().acquisition
        video_times = acquisition['Video'].timestamps[:]
        video_frames = acquisition['VideoImagingFrame']
        assert video_frames.timestamps.name == '/acquisition/Video/timestamps'
        numpy.testing.assert_array_equal(video_frames.data[:], session.export_video_frames())
    numpy.testing.assert_array_equal(video_times, session.export_video_times())


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
