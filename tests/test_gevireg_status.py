'''
Record of the movie that a trace was pulled from (pull info json -> CaTrace version info -> ImagingSession), and the
fallbacks for traces pulled before it was recorded: GEVIReg folder and analysis logs.
'''
import json
import shutil

import numpy
import pytest

from Proc2P.Analysis.CaTrace import CaTrace
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Analysis.PullSignals import pull_signals, logged_pull_source
from Proc2P.Bruker.LoadRegistered import Source, GEVIRegStatus, gevireg_version

PREFIX = 'Test1_2026-01-01_gevi_001'
N_FRAMES = 200
N_ROIS = 3


def make_session(path):
    sdir = path / PREFIX
    sdir.mkdir()
    with open(sdir / f'{PREFIX}_SessionInfo.json', 'w') as f:
        json.dump({'framerate': 500.0, 'n_frames': N_FRAMES}, f)
    numpy.save(sdir / f'{PREFIX}_FrameTimes.npy', numpy.zeros((N_FRAMES, 2)))
    numpy.save(sdir / f'{PREFIX}_saved_roi_A.npy',
               numpy.array([[c, x + 4 * c, y] for c in range(N_ROIS) for x, y in ((1, 1), (4, 1), (4, 4))]))
    return sdir


def make_gevireg(sdir, fixed):
    gdir = sdir / 'GEVIReg'
    gdir.mkdir()
    rng = numpy.random.default_rng(0)
    numpy.save(gdir / f'{PREFIX}_registered_Ch2.npy', rng.integers(100, 200, (N_FRAMES, 8, 16), dtype='uint16'))
    config = {'ref_channel': 'Ch2'}
    if fixed:
        config.update({'S2P_offset': [0.0, 0.0], 'bidiphase': 0, 'version': 1})
    with open(gdir / f'{PREFIX}_regconfig.json', 'w') as f:
        json.dump(config, f)


def process(path, tag='A'):
    '''process the raw trace like the CaTrace workers do, with a flat baseline'''
    ca = CaTrace(path, PREFIX, tag=tag, ch=0)
    assert ca.open_raw() is None
    ca.bsl = numpy.repeat(ca.trace.mean(axis=1, keepdims=True), ca.frames, axis=1)
    ca.rel = ca.ntr = ca.smtr = (ca.trace - ca.bsl) / ca.bsl
    ca.fps = 500.0
    ca.save()


def write_log(sdir, entries, user='u1'):
    '''log lines as the pull Worker writes them. The message contains \\r\\n, written in text mode on Windows'''
    with open(sdir / f'{PREFIX}_{user}_AnalysisLog.txt', 'w', newline='') as f:
        for ts, tag, source in entries:
            f.write(f'{ts} - PullSignals: Pulled simple average from {N_FRAMES} frames ({N_ROIS} regions, 1 channels) '
                    f'from {PREFIX} roi {tag}.\r\r\nChannel order is: [\'Ch2\'], {source} movie was used.\r\n')


@pytest.mark.parametrize('fixed', (True, False))
def test_pull_from_gevireg_is_recorded(tmp_path, fixed):
    sdir = make_session(tmp_path)
    make_gevireg(sdir, fixed)
    assert pull_signals(tmp_path, PREFIX, tag='A', ch='Ch2', use_movie=Source.GEVIReg)
    with open(sdir / f'{PREFIX}_trace_A.json') as f:
        pull_info = json.load(f)
    assert pull_info['movie'] == 'GEVIReg'
    assert pull_info['GEVIReg_version'] == (1 if fixed else 0)
    process(tmp_path)
    session = ImagingSession(tmp_path, PREFIX, tag='A')
    assert session.ca.version_info['movie'] == 'GEVIReg'
    assert session.gevireg_status() == (GEVIRegStatus.FIXED if fixed else GEVIRegStatus.OLD)


def test_pull_from_other_movie_is_not_gevireg(tmp_path):
    sdir = make_session(tmp_path)
    make_gevireg(sdir, fixed=False)
    numpy.save(sdir / f'{PREFIX}_trace_A.npy', numpy.random.default_rng(0).uniform(100, 200, (N_ROIS, N_FRAMES, 1)))
    with open(sdir / f'{PREFIX}_trace_A.json', 'w') as f:
        json.dump({'movie': 'S2P'}, f)
    process(tmp_path)
    assert ImagingSession(tmp_path, PREFIX, tag='A').gevireg_status() == GEVIRegStatus.NOT_GEVIREG


@pytest.mark.parametrize('has_gevireg, log, expected', (
        (False, [], GEVIRegStatus.NOT_GEVIREG),
        (True, [], GEVIRegStatus.UNKNOWN),
        (True, [('2026-10-02T10:48:34', 'A', 'Source.GEVIReg')], GEVIRegStatus.OLD),
        (True, [('2026-10-02T10:48:34', 'A', 'Source.S2P')], GEVIRegStatus.NOT_GEVIREG),
        # only the last pull of the tag counts
        (True, [('2026-10-02T10:48:34', 'A', 'Source.GEVIReg'), ('2026-10-03T09:00:00', 'A', 'Source.S2P')],
         GEVIRegStatus.NOT_GEVIREG),
        (True, [('2026-10-02T10:48:34', 'AB', 'Source.GEVIReg')], GEVIRegStatus.UNKNOWN),
))
def test_trace_without_pull_record(tmp_path, capsys, has_gevireg, log, expected):
    sdir = make_session(tmp_path)
    if has_gevireg:
        make_gevireg(sdir, fixed=True)  # rerun after the pull: the trace is still from the old movie
    numpy.save(sdir / f'{PREFIX}_trace_A.npy', numpy.random.default_rng(0).uniform(100, 200, (N_ROIS, N_FRAMES, 1)))
    process(tmp_path)
    write_log(sdir, log)
    capsys.readouterr()
    session = ImagingSession(tmp_path, PREFIX, tag='A')
    assert session.gevireg_status() == expected
    warned = 'Warning' in capsys.readouterr().out
    assert warned == (expected in (GEVIRegStatus.OLD, GEVIRegStatus.UNKNOWN))


def test_logged_pull_source_reads_logs_of_all_users(tmp_path):
    sdir = make_session(tmp_path)
    write_log(sdir, [('2026-10-02T10:48:34', 'A', 'Source.S2P')], user='u1')
    write_log(sdir, [('2026-10-04T10:48:34', 'A', 'Source.GEVIReg')], user='u2')
    assert logged_pull_source(tmp_path, PREFIX, 'A') == 'GEVIReg'
    assert logged_pull_source(tmp_path, PREFIX, 'B') is None


def test_gevireg_version(tmp_path):
    sdir = make_session(tmp_path)
    assert gevireg_version(tmp_path, PREFIX) is None
    (sdir / 'GEVIReg').mkdir()
    assert gevireg_version(tmp_path, PREFIX) == 0  # no regconfig: interrupted run
    (sdir / 'GEVIReg').rmdir()
    make_gevireg(sdir, fixed=False)
    assert gevireg_version(tmp_path, PREFIX) == 0
    shutil.rmtree(sdir / 'GEVIReg')
    make_gevireg(sdir, fixed=True)
    assert gevireg_version(tmp_path, PREFIX) == 1
