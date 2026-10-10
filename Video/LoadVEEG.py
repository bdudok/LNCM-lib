"""Loader for vEEG recorder output (Recorder repo, VideoEEG package).

Each hourly video file comes with two CSV files sharing its name stem:
    <stem>.mkv          video
    <stem>_frames.csv   one row per frame: frame, frame_total, cam_seq, cam_time_us, pc_time
    <stem>_ttl.csv      one row per TTL pulse: pulse, pc_time, frame, frame_total, cam_seq
The stem is <prefix>-<YYYY-MM-DDTHH-MM-SS>, the prefix is set per camera in the recorder.

Frame times use the camera clock (cam_time_us), so frames dropped before reaching the PC do not
shift the times of later frames. A pulse's time on the camera clock is the time of the last frame
received before the pulse plus the PC time between that frame's arrival and the pulse.

Consecutive files of one camera share the camera clock until the recorder restarts (cam_seq starts over),
so they are joined into one recording and aligned together. This covers EEG files that span a video file switch.
"""
import datetime
import os
import re

import numpy
import pandas

from Proc2P.Treadmill import rsync

STEM_PATTERN = re.compile(r'^(?P<prefix>.+)-(?P<start>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$')


class VEEGRecording:
    __name__ = 'VEEGRecording'

    def __init__(self, stems):
        """stems: paths of consecutive video files of one camera, without extension, in recording order"""
        self.stems = list(stems)
        frames, ttls, first = [], [], 0
        for stem in self.stems:
            f = pandas.read_csv(stem + '_frames.csv')
            t = pandas.read_csv(stem + '_ttl.csv')
            t['frame'] += first  # index into the joined frame table
            frames.append(f)
            ttls.append(t)
            first += len(f)
        self.frames = pandas.concat(frames, ignore_index=True)
        self.ttl = pandas.concat(ttls, ignore_index=True)
        self.file_n_frames = numpy.array([len(f) for f in frames])
        self.file_first_frame = numpy.concatenate([[0], numpy.cumsum(self.file_n_frames)[:-1]])
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


class VEEGVideo(VEEGRecording):
    __name__ = 'VEEGVideo'

    def __init__(self, stem):
        """stem: path of the video without extension, e.g. 'D:/vid/mouse1-2026-10-07T17-03-59'"""
        super().__init__([stem])
        self.stem = stem
        self.video_fn = stem + '.mkv'


def find_recordings(folder, start=None, stop=None, margin_s=600):
    """Group the vEEG files in a folder into recordings: consecutive files of one camera without a recorder restart.
    start, stop: datetimes; if given, only files that may overlap this period (+- margin_s, for clock differences
    between PCs) are read. Returns a list of lists of stems."""
    by_prefix = {}
    for fn in os.listdir(folder):
        if not fn.endswith('.mkv'):
            continue
        m = STEM_PATTERN.match(fn[:-4])
        if m is None or not os.path.exists(os.path.join(folder, fn[:-4] + '_frames.csv')):
            continue
        t = datetime.datetime.strptime(m.group('start'), '%Y-%m-%dT%H-%M-%S')
        by_prefix.setdefault(m.group('prefix'), []).append((t, os.path.join(folder, fn[:-4])))
    margin = datetime.timedelta(seconds=margin_s)
    recordings = []
    for files in by_prefix.values():
        files.sort()
        # a file lasts until the next one starts at most; the last one is checked by its start only
        keep = [stem for i, (t, stem) in enumerate(files)
                if (stop is None or t < stop + margin) and
                (start is None or i + 1 == len(files) or files[i + 1][0] > start - margin)]
        group, last_seq = [], None
        for stem in keep:
            seq = pandas.read_csv(stem + '_frames.csv', usecols=['cam_seq'])['cam_seq'].values
            if group and (len(seq) == 0 or seq[0] <= last_seq):
                recordings.append(group)
                group = []
            group.append(stem)
            if len(seq):
                last_seq = seq[-1]
        if group:
            recordings.append(group)
    return recordings


def video_map(eeg_ttl_times, eeg_start, eeg_duration, folder):
    """Map each second of an EEG file to a video frame, using the vEEG recording whose TTL pulses match the EEG.
    eeg_ttl_times: TTL pulse times in s from the EEG start; eeg_start: datetime of the EEG start;
    eeg_duration: s. Returns None if no recording in the folder matches, else a dict:
        files: [{'name', 'first_frame'}]  video files of the recording, first_frame in the joined recording
        frame: float frame index in the joined recording at each EEG second 0, 1, ...; nan where there is no video
        fps
    """
    eeg_stop = eeg_start + datetime.timedelta(seconds=eeg_duration)
    best = None
    for stems in find_recordings(folder, eeg_start, eeg_stop):
        rec = VEEGRecording(stems)
        try:
            frame_times = rec.align(eeg_ttl_times)
        except rsync.RsyncError:
            continue
        if best is None or rec.matched_pulses > best[0].matched_pulses:
            best = rec, frame_times
    if best is None:
        return None
    rec, frame_times = best
    grid = numpy.arange(int(numpy.floor(eeg_duration)) + 1)
    frame = numpy.interp(grid, frame_times, numpy.arange(rec.n_frames), left=numpy.nan, right=numpy.nan)
    # keep the files from the first to the last one shown during the EEG
    used = numpy.searchsorted(rec.file_first_frame, frame[~numpy.isnan(frame)], side='right') - 1
    keep = range(used.min(), used.max() + 1)
    return {'files': [{'name': os.path.basename(rec.stems[i]) + '.mkv', 'first_frame': int(rec.file_first_frame[i])}
                      for i in keep],
            'frame': frame, 'fps': float(rec.fps)}


def save_video_map(vmap, fn):
    """CSV with one row per EEG second: eeg_s, file, frame (in that file), rec_frame (in the joined recording).
    file and frames are empty where there is no video."""
    k = vmap['frame']
    has_video = ~numpy.isnan(k)
    first = [f['first_frame'] for f in vmap['files']]
    i = numpy.searchsorted(first, numpy.where(has_video, k, 0), side='right') - 1
    names = numpy.array([f['name'] for f in vmap['files']], dtype=object)[i]
    pandas.DataFrame({'eeg_s': numpy.arange(len(k)), 'file': numpy.where(has_video, names, ''),
                      'frame': k - numpy.array(first)[i], 'rec_frame': k}).to_csv(fn, index=False, float_format='%.2f')


def load_video_map(fn):
    df = pandas.read_csv(fn)
    first_rows = df.dropna(subset=['file']).drop_duplicates('file')
    files = [{'name': row.file, 'first_frame': int(round(row.rec_frame - row.frame))}
             for row in first_rows.itertuples()]
    k = df['rec_frame'].values.astype(float)
    return {'files': files, 'frame': k, 'fps': float(numpy.nanmedian(numpy.diff(k)))}


def frame_at(vmap, t):
    """Video file index and frame index in that file at EEG time t (s), interpolated between the 1-s map points.
    Returns None where there is no video."""
    k = numpy.interp(t, numpy.arange(len(vmap['frame'])), vmap['frame'])
    if numpy.isnan(k):
        return None
    k = int(round(k))
    first = [f['first_frame'] for f in vmap['files']]
    i = int(numpy.searchsorted(first, k, side='right')) - 1
    return i, k - first[i]
