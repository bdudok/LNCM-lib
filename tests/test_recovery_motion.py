import datetime
from pathlib import Path

import cv2
import numpy

from Video import RecoveryMotion


def test_processed_dir():
    video = Path('D:/Shares/Data/_RawData/MonitorCam/Exp1/mouse1-2026-11-05T13-56-59.mkv')
    assert RecoveryMotion.processed_dir(video) == Path('D:/Shares/Data/_Processed/MonitorCam/Exp1')
    assert RecoveryMotion.trace_fn(video) == Path(
        'D:/Shares/Data/_Processed/MonitorCam/Exp1/mouse1-2026-11-05T13-56-59_motion_energy.npy')
    # no _RawData in the path: next to the video
    assert RecoveryMotion.processed_dir(Path('E:/videos/a.mkv')) == Path('E:/videos')


def test_start_time(tmp_path):
    # tag as MonitorCam's ffmpeg writes it into the Matroska header: TagName element, then TagString (0x4487)
    value = b'2026-11-05T13:56:59.586-06:00'
    tag = b'\x45\xa3\x90START_TIME_LOCAL\x44\x87' + bytes([0x80 | len(value)]) + value
    video = tmp_path / 'mouse1-2026-11-05T13-56-59.mkv'
    video.write_bytes(b'\x00' * 300 + tag + b'\x00' * 300)
    t = RecoveryMotion.start_time(video)
    assert t == datetime.datetime(2026, 11, 5, 13, 56, 59, 586000,
                                  tzinfo=datetime.timezone(datetime.timedelta(hours=-6)))
    # without the tag: from the file name, local time
    video.write_bytes(b'\x00' * 1000)
    assert RecoveryMotion.start_time(video) == datetime.datetime(2026, 11, 5, 13, 56, 59)
    other = tmp_path / 'other.mkv'
    other.write_bytes(b'\x00' * 1000)
    assert RecoveryMotion.start_time(other) is None


def test_compute_motion_in_roi(tmp_path, monkeypatch):
    monkeypatch.setattr(RecoveryMotion, 'QUEUE_LOCK', tmp_path / 'queue.lock')
    video = tmp_path / '_RawData' / 'MonitorCam' / 'mouse1.avi'
    video.parent.mkdir(parents=True)
    # a square moves in the right half during frames 100-200; the left half flickers all the time
    rng = numpy.random.default_rng(0)
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (320, 120))
    for i in range(300):
        frame = numpy.full((120, 320, 3), 60, dtype=numpy.uint8)
        frame[:, :160] = rng.integers(0, 255, dtype=numpy.uint8)
        x = 170 + (i - 100) % 100 if 100 <= i < 200 else 170
        frame[40:80, x:x + 40] = 220
        out.write(frame)
    out.release()
    RecoveryMotion.save_roi(video, [160, 320, 0, 120])
    RecoveryMotion.compute_motion(video)
    t, me = RecoveryMotion.load_trace(video)
    assert RecoveryMotion.trace_fn(video).parent == tmp_path / '_Processed' / 'MonitorCam'
    assert len(me) == 300
    numpy.testing.assert_allclose(t, numpy.arange(300) / 30, atol=0.002)
    # the flicker outside the ROI is not measured
    assert numpy.median(me[110:190]) > 5 * numpy.median(me[220:290])
    assert not RecoveryMotion.QUEUE_LOCK.exists()
    # a new ROI deletes the trace
    RecoveryMotion.save_roi(video, [0, 160, 0, 120])
    assert RecoveryMotion.load_trace(video) is None
