import cv2
import numpy

from LFP.Setup import Setup
from Video import MotionEnergy


def write_video(fn, n_frames=300, moving=(100, 200), size=(160, 120)):
    """Noisy gray video with a bright square that moves during frames moving[0]..moving[1]"""
    rng = numpy.random.default_rng(0)
    out = cv2.VideoWriter(str(fn), cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    x = 20
    for i in range(n_frames):
        frame = rng.integers(60, 70, (size[1], size[0]), dtype=numpy.uint8)
        if moving[0] <= i < moving[1]:
            x = 20 + (i - moving[0]) % 100
        frame[40:80, x:x + 40] = 220
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    out.release()


def test_chunks_match_single_pass(tmp_path, monkeypatch):
    fn = tmp_path / 'test.avi'
    write_video(fn)
    single = MotionEnergy.motion_energy(fn, n_workers=1)
    monkeypatch.setattr(MotionEnergy, 'CHUNK', 70)
    chunked = MotionEnergy.motion_energy(fn, n_workers=4)
    assert len(single) == len(chunked) == 300
    numpy.testing.assert_allclose(chunked, single, rtol=1e-3, atol=1e-3 * single.mean())
    # starting in the middle of the file
    part = MotionEnergy.motion_energy(fn, 150, 250, n_workers=2)
    numpy.testing.assert_allclose(part, single[150:250], rtol=1e-3, atol=1e-3 * single.mean())
    # motion is where the square moves
    assert numpy.median(single[110:190]) > 5 * numpy.median(single[220:290])


def test_pull_veeg_motion_times(tmp_path):
    # a recording joined from two files of 300 frames; the EEG starts 2 s before the video, at 30 frames/s
    for name in ('cam-a.avi', 'cam-b.avi'):
        write_video(tmp_path / name)
    eeg_s = numpy.arange(22)
    frame = (eeg_s - 2) * 30.0
    frame[eeg_s < 2] = numpy.nan
    vmap = {'files': [{'name': 'cam-a.avi', 'first_frame': 0}, {'name': 'cam-b.avi', 'first_frame': 300}],
            'frame': frame, 'fps': 30.0}
    out_fn = tmp_path / 'eeg_motion_energy.npy'
    t, me = MotionEnergy.pull_veeg_motion(vmap, tmp_path, out_fn, n_workers=2)
    assert len(t) == len(me) == 19 * 30 + 1  # frames 0..570
    numpy.testing.assert_allclose(t, 2 + numpy.arange(len(t)) / 30)
    numpy.testing.assert_array_equal(numpy.load(out_fn), [t, me])


def test_pull_motion_skips_locked_recording(tmp_path):
    session = tmp_path / 'mouse_2026-10-14_test_001'
    session.mkdir()
    MotionEnergy.lock_fn(Setup.LNCM, tmp_path, session.name).touch()
    assert MotionEnergy.is_running(Setup.LNCM, tmp_path, session.name)
    assert MotionEnergy.pull_motion('LNCM', tmp_path, session.name) is None
    assert MotionEnergy.lock_fn(Setup.LNCM, tmp_path, session.name).exists()  # left for the process that owns it
