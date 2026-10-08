import numpy
import pandas

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
