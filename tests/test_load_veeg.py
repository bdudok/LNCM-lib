import datetime
import os

import numpy
import pandas

from Video import LoadVEEG
from Video.LoadVEEG import VEEGVideo


def write_recording(stem, rng, n_frames=20 * 60 * 30, period=1 / 29.926, dropped=(5000, 5001, 20000)):
    """Synthetic vEEG recorder output. Returns the PC time of each pulse and of each frame's exposure."""
    seq = numpy.setdiff1d(numpy.arange(n_frames + len(dropped)), dropped)
    exposure_pc = 1.7e9 + seq * period
    arrival_pc = exposure_pc + 0.030 + rng.normal(0, 0.002, len(seq))  # transfer latency and jitter
    pandas.DataFrame({'frame': numpy.arange(len(seq)), 'frame_total': seq + 100, 'cam_seq': seq,
                      'cam_time_us': (seq * period * 1e6).astype('int64'),
                      'pc_time': arrival_pc}).to_csv(f'{stem}_frames.csv', index=False)
    pulse_pc = exposure_pc[0] + 2 + numpy.cumsum(rng.uniform(0.5, 9.5, 230))
    pulse_pc = pulse_pc[pulse_pc < arrival_pc[-1]]
    frame = numpy.searchsorted(arrival_pc, pulse_pc) - 1  # last frame received before the pulse
    pandas.DataFrame({'pulse': numpy.arange(1, len(pulse_pc) + 1), 'pc_time': pulse_pc, 'frame': frame,
                      'frame_total': frame + 100, 'cam_seq': seq[frame]}).to_csv(f'{stem}_ttl.csv', index=False)
    return pulse_pc, exposure_pc


def test_align_to_eeg_ttl(tmp_path):
    rng = numpy.random.default_rng(0)
    stem = str(tmp_path / 'vEEG-2026-10-07T17-03-59')
    pulse_pc, exposure_pc = write_recording(stem, rng)

    # EEG started 600 s before the video, its clock runs 50 ppm fast, TTL edges sampled at 2 kHz.
    # It also has pulses from before the video started (another file of the same recording).
    def to_eeg(pc):
        return numpy.round(((pc - pulse_pc[0]) * (1 + 50e-6) + 600) * 2000) / 2000
    earlier = pulse_pc[0] - numpy.cumsum(rng.uniform(0.5, 9.5, 60))[::-1]
    eeg_ttl = to_eeg(numpy.concatenate([earlier, pulse_pc]))

    v = VEEGVideo(stem)
    frame_times = v.align(eeg_ttl)

    assert v.n_frames == 36000 and v.dropped == 3
    assert abs(v.fps - 29.926) < 0.001
    assert v.matched_pulses > 0.9 * len(v.ttl)
    assert not numpy.isnan(frame_times).any()
    # frame times are in the EEG clock, offset by the transfer latency of 30 ms
    error = frame_times - to_eeg(exposure_pc)
    assert abs(numpy.median(error) - 0.030) < 0.002
    assert numpy.abs(error - numpy.median(error)).max() < 0.010


def split_recording(stem, new_stems, n_first):
    """Split a synthetic recording into two consecutive files, as the recorder does at a segment switch."""
    frames = pandas.read_csv(f'{stem}_frames.csv')
    ttl = pandas.read_csv(f'{stem}_ttl.csv')
    for i, (sl, s) in enumerate(((slice(0, n_first), new_stems[0]), (slice(n_first, None), new_stems[1]))):
        f = frames.iloc[sl].copy()
        f['frame'] -= f['frame'].iloc[0]
        f.to_csv(f'{s}_frames.csv', index=False)
        t = ttl[(ttl['frame'] >= n_first) == bool(i)].copy()
        t['frame'] -= n_first * i
        t.to_csv(f'{s}_ttl.csv', index=False)
        open(f'{s}.mkv', 'w').close()


def test_video_map_spans_files_and_picks_camera(tmp_path):
    rng = numpy.random.default_rng(1)
    pulse_pc, exposure_pc = write_recording(str(tmp_path / 'full'), rng)
    stems = [str(tmp_path / 'camA-2026-10-07T17-00-00'), str(tmp_path / 'camA-2026-10-07T17-10-00')]
    split_recording(str(tmp_path / 'full'), stems, 18000)
    # another camera in the same folder, same times, different pulse sequence
    write_recording(str(tmp_path / 'camB-2026-10-07T17-00-00'), numpy.random.default_rng(2))
    open(tmp_path / 'camB-2026-10-07T17-00-00.mkv', 'w').close()

    # EEG file starts 5 min into the video, lasts 8 min, so it spans the file switch at 10 min
    eeg_t0 = exposure_pc[0] + 300
    eeg_ttl = pulse_pc[(pulse_pc > eeg_t0) & (pulse_pc < eeg_t0 + 480)] - eeg_t0
    eeg_start = datetime.datetime(2026, 10, 7, 17, 5)
    vmap = LoadVEEG.video_map(eeg_ttl, eeg_start, 480, str(tmp_path))

    assert [f['name'] for f in vmap['files']] == [os.path.basename(s) + '.mkv' for s in stems]
    assert vmap['files'][1]['first_frame'] == 18000
    # frame shown at each second: the last exposure before it, plus the 30 ms transfer latency
    t_switch = exposure_pc[18000] - eeg_t0
    for t in (0.0, t_switch - 0.5, t_switch + 0.5, 479.0):
        file_i, frame = LoadVEEG.frame_at(vmap, t)
        expected = numpy.searchsorted(exposure_pc, eeg_t0 + t - 0.030)
        k = vmap['files'][file_i]['first_frame'] + frame
        assert abs(k - expected) <= 1, (t, k, expected)
    assert LoadVEEG.frame_at(vmap, t_switch - 0.5)[0] == 0 and LoadVEEG.frame_at(vmap, t_switch + 0.5)[0] == 1

    vmap['frame'][:3] = numpy.nan  # seconds without video
    LoadVEEG.save_video_map(vmap, tmp_path / 'map.csv')
    loaded = LoadVEEG.load_video_map(tmp_path / 'map.csv')
    assert loaded['files'] == vmap['files']
    assert numpy.allclose(loaded['frame'], vmap['frame'], atol=0.01, equal_nan=True)
    assert abs(loaded['fps'] - vmap['fps']) < 0.01
