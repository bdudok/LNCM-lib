'''
Crosstalk correction on a synthetic two-channel session: a fast source sensor leaks into the channel of a slow sensor.
'''
import json

import numpy
import pytest
from scipy.ndimage import gaussian_filter1d

from Proc2P.Analysis.AnalysisClasses.Crosstalk import Crosstalk
from Proc2P.Analysis.CaTrace import CaTrace
from Proc2P.Analysis.ImagingSession import ImagingSession

PREFIX = 'Test1_2026-01-01_xtalk_001'
FPS = 20.0
K_TRUE = 0.05


def make_session(path, noise, n_rois=14, n_frames=36000, seed=0):
    '''
    Writes the trace folders of channel 0 (slow sensor + leak) and channel 1 (fast sensor). Returns the true rel of
    channel 0 without the leak.
    '''
    rng = numpy.random.default_rng(seed)
    sdir = path / PREFIX
    sdir.mkdir()
    with open(sdir / f'{PREFIX}_SessionInfo.json', 'w') as f:
        json.dump({'framerate': FPS, 'n_frames': n_frames}, f)
    numpy.save(sdir / f'{PREFIX}_FrameTimes.npy', numpy.zeros((n_frames, 2)))
    # one triangle per ROI (rows: roi index, x, y), needed by ImagingSession
    numpy.save(sdir / f'{PREFIX}_saved_roi_A.npy',
               numpy.array([[c, x, y] for c in range(n_rois) for x, y in ((1, 1), (4, 1), (4, 4))]))
    # fast events shared by the ROIs in the source channel, slow independent signal in the target channel
    events = numpy.clip(gaussian_filter1d(rng.normal(0, 1, n_frames), 2), 0, None) * 30
    source = numpy.array([100 * rng.uniform(0.3, 1.5) + events * rng.uniform(0.2, 1) for _ in range(n_rois)])
    target_true = numpy.array([80 + gaussian_filter1d(rng.normal(0, 20, n_frames), 40) for _ in range(n_rois)])
    target = target_true + K_TRUE * source
    if noise:
        source, target = rng.poisson(source).astype(float), rng.poisson(target).astype(float)
    true_bsl = numpy.repeat(target_true.mean(axis=1, keepdims=True), n_frames, axis=1)
    for ch, f in ((0, target), (1, source)):
        ca = CaTrace(path, PREFIX, tag='A', ch=ch)
        bsl = numpy.repeat(f.mean(axis=1, keepdims=True), n_frames, axis=1)
        ca.trace, ca.bsl = f, bsl
        ca.rel = (f - bsl) / bsl
        ca.ntr = ca.smtr = ca.rel
        ca.bsltype, ca.fps, ca.cells = 'poly', FPS, n_rois
        ca.save()
    return (target_true - true_bsl) / true_bsl


def test_recovers_leak_and_true_trace(tmp_path):
    true_rel = make_session(tmp_path, noise=False)
    xt = Crosstalk(tmp_path, PREFIX, 'A', source_ch=1, target_ch=0)
    rel = xt.run()
    assert xt.k == pytest.approx(K_TRUE, abs=1e-3)
    assert numpy.nanmax(numpy.abs(rel - true_rel)) < 1e-3


def test_with_shot_noise(tmp_path):
    make_session(tmp_path, noise=True)
    xt = Crosstalk(tmp_path, PREFIX, 'A', source_ch=1, target_ch=0)
    xt.fit()
    # the instrumental variable is not biased toward 0 by the noise in the ROI's own source trace
    assert xt.k == pytest.approx(K_TRUE, abs=0.015)


def test_saved_files(tmp_path):
    make_session(tmp_path, noise=False)
    rel = Crosstalk(tmp_path, PREFIX, 'A', source_ch=1, target_ch=0).run()
    pf = tmp_path / PREFIX / f'{PREFIX}_trace_A-ch0'
    with open(pf / 'xtalk.json') as f:
        params = json.load(f)
    assert params['source_ch'] == 1 and params['target_ch'] == 0 and params['n_rois_used'] == 14
    assert params['k'] == pytest.approx(K_TRUE, abs=1e-3)
    xt = Crosstalk(tmp_path, PREFIX, 'A', source_ch=1, target_ch=0)
    assert numpy.array_equal(xt.load(), rel)
    assert numpy.array_equal(numpy.load(pf / 'rel_xtalk.npy'), rel)
    with pytest.raises(ValueError, match='parameters differ'):
        Crosstalk(tmp_path, PREFIX, 'A', source_ch=1, target_ch=0, hp_cutoff=2.0).load()


def test_same_channel(tmp_path):
    make_session(tmp_path, noise=False, n_frames=1000)
    with pytest.raises(ValueError, match='must be different'):
        Crosstalk(tmp_path, PREFIX, 'A', source_ch=0, target_ch=0)


def test_imaging_session_getparam(tmp_path):
    make_session(tmp_path, noise=False, n_frames=2000)
    rel = Crosstalk(tmp_path, PREFIX, 'A', source_ch=1, target_ch=0).run()
    assert numpy.array_equal(ImagingSession(tmp_path, PREFIX, tag='A', ch=0).getparam('xtalk'), rel)
    with pytest.raises(FileNotFoundError, match='No crosstalk-corrected trace'):
        ImagingSession(tmp_path, PREFIX, tag='A', ch=1).getparam('xtalk')
    with pytest.raises(ValueError, match='Unknown param: xtlak'):
        ImagingSession(tmp_path, PREFIX, tag='A', ch=0).getparam('xtlak')
