"""Loader for vEEG recorder output (Recorder repo, VideoEEG package).

Each hourly video file comes with two CSV files sharing its name stem:
    <stem>.mkv          video
    <stem>_frames.csv   one row per frame: frame, frame_total, cam_seq, cam_time_us, pc_time
    <stem>_ttl.csv      one row per TTL pulse: pulse, pc_time, frame, frame_total, cam_seq

Frame times use the camera clock (cam_time_us), so frames dropped before reaching the PC do not
shift the times of later frames. A pulse's time on the camera clock is the time of the last frame
received before the pulse plus the PC time between that frame's arrival and the pulse.
"""
import numpy
import pandas

from Proc2P.Treadmill import rsync


class VEEGVideo:
    __name__ = 'VEEGVideo'

    def __init__(self, stem):
        """stem: path of the video without extension, e.g. 'D:/vid/mouse1-2026-10-07T17-03-59'"""
        self.stem = stem
        self.video_fn = stem + '.mkv'
        self.frames = pandas.read_csv(stem + '_frames.csv')
        self.ttl = pandas.read_csv(stem + '_ttl.csv')
        self.n_frames = len(self.frames)
        cam_us = self.frames['cam_time_us'].values
        self.frame_times = (cam_us - cam_us[0]) / 1e6  # s from the first frame, camera clock
        self.fps = 1 / numpy.median(numpy.diff(self.frame_times))
        self.dropped = int((numpy.diff(self.frames['cam_seq'].values) - 1).clip(0).sum())
        fi = self.ttl['frame'].values
        frame_pc = self.frames['pc_time'].values[fi]
        self.ttl_times = self.frame_times[fi] + (self.ttl['pc_time'].values - frame_pc)  # s, camera clock
        self.alignment = None

    def align(self, eeg_ttl_times):
        """Match the video's TTL pulses to the TTL times recorded with the EEG (in s).
        Returns the time of each frame in EEG seconds. Frames between matched pulses are interpolated;
        frames before the first or after the last matched pulse use a line fitted to all matched pulses.
        Raises rsync.RsyncError if the pulse sequences do not match."""
        self.alignment = rsync.Rsync_aligner(numpy.asarray(eeg_ttl_times), self.ttl_times, units_A=1000,
                                             units_B=1000)
        matched = ~numpy.isnan(self.alignment.cor_times_A)
        video_t = self.ttl_times[matched]
        eeg_t = self.alignment.cor_times_A[matched] / 1000
        self.matched_pulses = int(matched.sum())
        frame_times = numpy.interp(self.frame_times, video_t, eeg_t)
        outside = (self.frame_times < video_t[0]) | (self.frame_times > video_t[-1])
        frame_times[outside] = numpy.polyval(numpy.polyfit(video_t, eeg_t, 1), self.frame_times[outside])
        return frame_times
