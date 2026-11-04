import numpy

from Proc2P.Bruker.PreProc import first_affected_frame, led_frames

FS = 5000
PERIOD = 0.05  # 20 Hz
CAPTURE = 0.0323  # 512 lines x 63.15 us
FRAMETIMES = numpy.arange(10) * PERIOD


def led_trace(pulses):
    '''bool LED trace with pulses given as (onset, duration) in s'''
    trace = numpy.zeros(int(FS * 0.6), dtype='bool')
    for onset, duration in pulses:
        trace[int(onset * FS):int((onset + duration) * FS)] = True
    return trace


def test_pulse_at_frame_start_marks_that_frame():
    trace = led_trace([(2 * PERIOD, 0.008)])
    assert list(led_frames(trace, FRAMETIMES, CAPTURE, FS)) == [2]
    assert first_affected_frame(2 * PERIOD + 0.0002, FRAMETIMES, CAPTURE) == 2


def test_pulse_between_frames_marks_no_frame_and_counts_for_the_next():
    # LED on 33-41 ms after frame start, the next frame starts at 50 ms
    trace = led_trace([(2 * PERIOD + 0.033, 0.008)])
    assert len(led_frames(trace, FRAMETIMES, CAPTURE, FS)) == 0
    assert first_affected_frame(2 * PERIOD + 0.033, FRAMETIMES, CAPTURE) == 3


def test_pulse_ending_within_the_gate_margin_marks_the_next_frame():
    # LED off 0.5 ms before the next frame starts: the gate is still on
    trace = led_trace([(2 * PERIOD + 0.0415, 0.008)])
    assert list(led_frames(trace, FRAMETIMES, CAPTURE, FS)) == [3]


def test_pulse_across_frames_marks_both():
    trace = led_trace([(2 * PERIOD + 0.03, 0.03)])
    assert list(led_frames(trace, FRAMETIMES, CAPTURE, FS)) == [2, 3]
    assert first_affected_frame(2 * PERIOD + 0.03, FRAMETIMES, CAPTURE) == 2


def test_onsets_as_array():
    onsets = numpy.array([0.0002, 0.034, 0.1002])
    assert list(first_affected_frame(onsets, FRAMETIMES, CAPTURE)) == [0, 1, 2]


def test_full_period_capture_matches_the_old_rule():
    # without line timing, a pulse anywhere in the frame period marks the frame
    trace = led_trace([(2 * PERIOD + 0.033, 0.008)])
    assert list(led_frames(trace, FRAMETIMES, PERIOD, FS)) == [2]
    assert first_affected_frame(2 * PERIOD + 0.033, FRAMETIMES, PERIOD) == 2
