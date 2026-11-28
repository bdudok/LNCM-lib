import json
import os.path
from pathlib import Path

import numpy
import pandas

from BaserowAPI.BaserowRequests import GetSessions
from envs.site_config import CONFIG
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Analysis.LoadPolys import load_roi_file
from Proc2P.Analysis.PullSignals import pull_signals
from Proc2P.Analysis.CaTrace import CaTrace
from Proc2P.Bruker.LoadEphys import Ephys
from Proc2P.Analysis.Ripples import ripple_folder

import cv2

'''
ImagingSession with a uniform set of export methods, used by NWBExporter.
Each method returns one data type in the format NWBExporter expects (see the docstrings).
Times are always in seconds, relative to the same reference as the imaging frame times.
'''

# LFP event tables exported per LFP channel n: file suffix, NWB name, and the time columns (seconds from the first LFP
# sample). One time column: events; two: intervals (start, stop). NWBImagingSession writes the files back.
LFP_TABLES = (
    ('_Ch{n}_spiketimes.xlsx', 'LFPSpikes_Ch{n}', ('SpikeTimes(s)',)),
    ('_Ch{n}_seizure_times.xlsx', 'Seizures_Ch{n}', ('Sz.Start(s)', 'Sz.Stop(s)')),
    ('_Ch{n}_seizure_times_curated.xlsx', 'SeizuresCurated_Ch{n}', ('Start', 'Stop')),
)
# detection settings saved next to the LFP tables, stored in the notes json
LFP_SETTINGS_FILES = ('_Ch{n}_SpikeSzDet.json',)
# export_position: fewer laps than this give the distance run in default lap lengths instead of the position in the lap
MIN_LAPS_FOR_POSITION = 3
# lfp.event_tags in the export config: the spike tag whose LFP tables were saved last
LATEST_TAG = 'latest'


def lfp_event_tag(path, prefix, n, event_tags):
    '''
    The spike tag of the LFP tables of channel n to export. File names as ImagingSession.map_spiketimes reads them:
    <prefix>_<tag>_Ch<n><suffix>, or <prefix>_Ch<n><suffix> without a tag (see LFP_TABLES).
    :param path: the session folder
    :param event_tags: preferences in order: a tag, None (no tag), or LATEST_TAG (the tag, or no tag, whose tables
        were saved last)
    :return: the tag of the first preference that has tables (None: no tag), or False if none has
    '''
    saved = {}  # tag: time the last of its tables was saved
    for suffix, _, _ in LFP_TABLES:
        end = suffix.format(n=n)
        for fn in Path(path).glob(f'{prefix}*{end}'):
            infix = fn.name[len(prefix):-len(end)]
            if infix and not infix.startswith('_'):  # another session whose prefix starts with this one
                continue
            tag = infix[1:] or None
            saved[tag] = max(saved.get(tag, 0), fn.stat().st_mtime)
    for tag in event_tags:
        if tag == LATEST_TAG and saved:
            return max(saved, key=saved.get)
        if tag in saved:
            return tag
    return False


def read_table_file(fn):
    '''
    An Excel file saved by pandas, with its first column (the index) in column 'xlsx_index'.
    (Proc2P.utils.read_excel drops or renames the index, so it can't be written back as it was.)
    '''
    table = pandas.read_excel(fn, engine='openpyxl', index_col=0)
    table.index.name = 'xlsx_index'
    return table.reset_index()


class ExportImagingSession(ImagingSession):
    __name__ = 'ExportImagingSession'

    def __init__(self, procpath, prefix, db_entry=None, **kwargs):
        '''
        :param procpath: processed data folder (as in ImagingSession)
        :param prefix: session prefix
        :param db_entry: the session's row from the Baserow DB (pandas Series), if available
        :param kwargs: passed to ImagingSession (e.g. tag, lfp_ch)
        '''
        super().__init__(procpath, prefix, **kwargs)
        self.db_entry = db_entry

    def export_metadata(self):
        '''
        Session metadata. The whole dict is saved in the NWB file's notes json, under 'SessionInfo'
        (non-serializable values are converted with str).
        Reserved keys, used to fill NWB fields:
            'experimenter': list of str, optional. Added to the experimenters in the config.
            'laserWavelength_0': float or str (nm), optional. Excitation wavelength, used if not set in the config.
        :return: dict
        '''
        return self.si.info

    def export_subject(self):
        '''
        Subject fields from the DB. Values in the config take precedence.
        :return: dict with keys of pynwb.file.Subject (subject_id, sex). Empty if there's no DB entry.
        '''
        if self.db_entry is None:
            return {}
        mouse_name, sex = GetSessions().get_sex(item=self.db_entry)
        return {'subject_id': mouse_name, 'sex': sex}

    def get_starttime(self):
        '''
        Look up when the recording was started
        Returns:str iso
        '''
        fn = os.path.join(self.si.info['dpath'], self.prefix + '.log.txt')
        with open(fn, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line[20:].strip() == "Recording start":
                    return line[:19]  # e.g. '2026-04-22T17:39:21'
        raise ValueError(f"No 'Recording start' entry in {fn}")


    def export_fov_image(self):
        '''
        RGB image of the field of view.
        :return: numpy array, shape (height, width, 3), dtype uint8
        '''
        return self.get_preview()

    def export_2p_movie(self):
        '''
        Motion corrected (suite2p) 2P movie, one array per channel, in the order of imaging.channels in the config.
        Arrays can be numpy memmaps: they are written to the file in chunks, without loading into memory.
        :return: list of array-like, each shape (n_frames, height, width)
        '''
        self.rois.load_image()
        registered_movie = []
        for ch in self.rois.image.channels:
            registered_movie.append(self.rois.image.imdat.get_channel(ch))
        return registered_movie

    def export_frame_times(self):
        '''
        Timestamp of each imaging frame. This defines the time reference for all other data.
        :return: 1D float numpy array, length n_frames, seconds
        '''
        return self.ftimes[:, 0] #relative times in seconds, frame 0 is time 0

    def export_frame_times_absolute(self):
        '''
        Timestamp of each imaging frame from the start of the acquisition (second column of _FrameTimes.npy).
        :return: 1D float numpy array, length n_frames, seconds
        '''
        return self.ftimes[:, 1]

    def export_speed(self):
        '''
        Running speed of the animal, one value per imaging frame.
        :return: 1D float numpy array, length n_frames, cm/s
        '''
        if self.has_behavior:
            return self.pos.speed
        raise NotImplementedError('export_speed')

    def export_smoothed_speed(self):
        '''
        Smoothed running speed, one value per imaging frame (pos.smspd, used for mobility periods).
        :return: 1D float numpy array, length n_frames, cm/s
        '''
        if self.has_behavior:
            return self.pos.smspd
        raise NotImplementedError('export_smoothed_speed')

    def export_belt_position(self):
        '''
        Position on the belt as recorded by the treadmill (pos.pos), one value per imaging frame.
        :return: 1D float numpy array, length n_frames
        '''
        if self.has_behavior:
            return self.pos.pos
        raise NotImplementedError('export_belt_position')

    def export_laps(self):
        '''
        Lap index, one value per imaging frame (pos.laps).
        :return: 1D float numpy array, length n_frames
        '''
        if self.has_behavior:
            return self.pos.laps
        raise NotImplementedError('export_laps')

    def export_behavior_events(self):
        '''
        Licks, rewards and reward zones from _bdat.json.
        :return: None if the session has no _bdat.json, else a dict with keys:
            'licks', 'rewards': 1D int numpy arrays, imaging frame of each event
            'RZ': numpy array, shape (n_zones, 2), int, start and stop frame of each reward zone
        '''
        if not self.has_behavior:
            return None
        return {'licks': numpy.asarray(self.bdat['licks'], dtype='int64'),
                'rewards': numpy.asarray(self.bdat['rewards'], dtype='int64'),
                'RZ': numpy.asarray(self.bdat['RZ'], dtype='int64').reshape(-1, 2)}

    def export_position(self):
        '''
        Position of the animal on the belt, in laps. With at least MIN_LAPS_FOR_POSITION laps, the position within the
        lap (0-1, pos.relpos). With fewer (the mouse did not pass the lap tag often enough to know the lap), the distance
        run since the first frame (cm) divided by default_lap_length of the site config, starting from 0.
        NaN where the position is not known; all NaN if the session has no treadmill position.
        :return: 1D float numpy array, length n_frames
        '''
        pos = self.ca.sync.load('pos')  # None without treadmill data (map_pos then fills pos.pos with zeros)
        if pos is None or numpy.isnan(pos).all():
            return numpy.full(self.ca.frames, numpy.nan)
        laps = self.pos.laps
        n_laps = 0 if laps is None or numpy.isnan(laps).all() else numpy.nanmax(laps)
        if n_laps >= MIN_LAPS_FOR_POSITION:
            return numpy.asarray(self.pos.relpos, dtype='float64')
        pos = numpy.asarray(pos, dtype='float64')
        return (pos - pos[numpy.isfinite(pos)][0]) / CONFIG.default_lap_length

    def export_lfp(self, channels=None):
        '''
        LFP recording, as stored in _ephys.npy (integer microvolts).
        :param channels: list of channel numbers (rows of _ephys.npy, from 1, as lfp_ch). None: all channels
        :return: None if the session has no LFP, else a dict with keys:
            'data': numpy array, shape (n_samples, n_channels)
            'channels': list of int, the channel numbers of the data columns
            'rate': float, sampling rate in Hz
            'starting_time': float, time of the first sample in seconds (same reference as frame times)
            'conversion': float, factor that converts data to volts (e.g. 1e-6 if data is in microvolts)
            'frames': 1D int numpy array, length n_samples, imaging frame index of each sample (Ephys.frames)
        '''
        ephys = Ephys(self.procpath, self.prefix, channel=None)
        if not hasattr(ephys, 'edat'):
            return None
        if self.si['lfp_units'] != 'microvolts':
            raise ValueError(f"{self.prefix}: LFP units are {self.si['lfp_units']}, export expects microvolts")
        if channels is None:
            channels = list(range(1, ephys.edat.shape[0]))
        # ephys.frames is the imaging frame index of each sample. Align on the first sample of the second frame,
        # the first frame's start may not be recorded
        fs = float(self.si['fs'])
        first_frame = ephys.frames[0] + 1
        first_sample = numpy.searchsorted(ephys.frames, first_frame)
        return {'data': ephys.edat[channels].transpose(),
                'channels': [int(c) for c in channels],
                'rate': fs,
                'starting_time': float(self.ftimes[first_frame, 0] - first_sample / fs),
                'conversion': 1e-6,
                'frames': ephys.frames}

    def export_ripples(self, channels, tags=None):
        '''
        Ripple events saved by Ripples (<tag>.ripples in the ripple_folder of each LFP channel), as start and stop LFP
        sample indices. Filtered LFP and power traces are not exported; Ripples recomputes them from the LFP.
        :param channels: LFP channel numbers
        :param tags: list of ripple tags (file names without .ripples). None: all. Tags a channel doesn't have are
            skipped.
        :return: dict of {channel: {tag: numpy array, shape (n_events, 2), float, start and stop sample}}, for the
            channels that have ripple events
        '''
        if not (self.path / '_ripples').exists():
            return {}
        n_channels = numpy.load(self.path / f'{self.prefix}_ephys.npy', mmap_mode='r').shape[0] - 1
        events = {}
        for channel in channels:
            path = ripple_folder(self.path, channel, n_channels)
            if not path.exists():
                continue
            found = sorted(f.stem for f in path.glob('*.ripples'))
            channel_events = {tag: numpy.fromfile(path / f'{tag}.ripples').reshape(-1, 2) for tag in found
                              if tags is None or tag in tags}
            if channel_events:
                events[channel] = channel_events
        return events

    def export_lfp_tables(self, channels, event_tags=(None,)):
        '''
        Spike and seizure tables of the LFP channels (see LFP_TABLES), for the files that exist. Of each channel, the
        tables of one spike tag are exported (see lfp_event_tag).
        :param channels: list of LFP channel numbers
        :param event_tags: spike tag preferences, see lfp_event_tag. Default: the tables without a tag
        :return: list of dicts with keys:
            'name': NWB table name, 'suffix': file suffix without the tag (NWBImagingSession writes this file),
            'source': name of the exported file, 'tag': its spike tag (None: no tag),
            'table': DataFrame, the file's contents. The file's first column (index) is in column 'xlsx_index'
            'time_columns': tuple of the columns that hold times in seconds from the first LFP sample
        '''
        tables = []
        for n in channels:
            tag = lfp_event_tag(self.path, self.prefix, n, event_tags)
            if tag is False:
                continue
            infix = '' if tag is None else f'_{tag}'
            for suffix, name, time_columns in LFP_TABLES:
                fn = self.get_file_with_suffix(infix + suffix.format(n=n))
                if not fn.exists():
                    continue
                table = read_table_file(fn)
                tables.append({'name': name.format(n=n), 'suffix': suffix.format(n=n), 'source': fn.name, 'tag': tag,
                               'table': table, 'time_columns': time_columns})
        return tables

    def export_settings(self, channels, event_tags=(None,)):
        '''
        Detection settings files of the LFP channels (see LFP_SETTINGS_FILES), for the files that exist, of the spike
        tag whose tables are exported (see export_lfp_tables). Tagged files are <prefix>_Ch<n>_SpikeSzDet_<tag>.json.
        :return: dict of {file suffix without the tag: the file's json contents}
        '''
        settings = {}
        for n in channels:
            tag = lfp_event_tag(self.path, self.prefix, n, event_tags)
            if tag is False:
                continue
            for suffix in LFP_SETTINGS_FILES:
                stem, ext = os.path.splitext(suffix.format(n=n))
                fn = self.get_file_with_suffix(stem + ('' if tag is None else f'_{tag}') + ext)
                if fn.exists():
                    with open(fn, 'r') as f:
                        settings[suffix.format(n=n)] = json.load(f)
        return settings

    def export_opto(self):
        '''
        Photostimulation of the session.
        :return: None if the session has no opto, else a dict with keys:
            'pulses': DataFrame of _StimFrames.xlsx, with columns 'xlsx_index' (voltage recording sample of the
                pulse), 'Intensity', 'ImgFrame', 'Duration' (ms)
            'stim_frames': 1D int numpy array, imaging frames affected by the stimulation (_bad_frames.npy,
                ImagingSession.opto)
        '''
        if not self.si['has_opto']:
            return None
        table = read_table_file(self.get_file_with_suffix('_StimFrames.xlsx'))
        return {'pulses': table, 'stim_frames': self.ca.sync.load('opto')}

    def export_roi_masks(self, roi_tag, weighted, source):
        '''
        ROI masks of one ROI set.
        :param roi_tag: str, the ROI set's tag
        :param weighted: bool, return SNR weights (each ROI's weights sum to 1) instead of binary masks
        :param source: Source of the movie used for the SNR weights
        :return: numpy array, shape (n_rois, height, width), bool, or float weights
        '''
        masks = pull_signals(self.procpath, self.prefix, tag=roi_tag, ch=0, snr_weighted=weighted,
                             enable_alt_path=True, overwrite=True, use_movie=source, mask_only=True)
        # pull_signals indexes masks as [roi, x, y]; images are [row (y), column (x)]
        return masks.transpose(0, 2, 1)

    def export_roi_polygons(self, roi_tag):
        '''
        ROI polygons of one ROI set, as saved by the ROI editor. Same order as export_roi_masks(roi_tag).
        :return: list of int numpy arrays, each shape (n_vertices, 2), columns x, y
        '''
        return load_roi_file(self.get_file_with_suffix(f'_saved_roi_{roi_tag}.npy'))

    def export_traces(self, roi_tag, trace_tag, n_channels=1):
        '''
        One version of the traces pulled from one ROI set. ROI order must match export_roi_masks(roi_tag).
        :param roi_tag: str, the ROI set's tag
        :param trace_tag: str, the trace version (e.g. 'trace', 'rel', 'ntr', or a custom one like 'vm')
        :param n_channels: number of trace channels (CaTrace ch 0, 1, ...)
        :return: numpy array, shape (n_frames, n_rois, n_channels).
            Channels in the order they were pulled (see roi_sets.channels in the config).
        '''
        data = []
        for ch in range(n_channels):
            ca = CaTrace(self.procpath, self.prefix, tag=roi_tag, ch=ch)
            ca.load()
            data.append(ca.get_custom(trace_tag).transpose())
        return numpy.stack(data, axis=-1)

    def export_trace_info(self, roi_tag, n_channels=1):
        '''
        CaTrace version info (info.npy) of each trace channel of one ROI set.
        :return: list of dicts, one per channel
        '''
        info = []
        for ch in range(n_channels):
            ca = CaTrace(self.procpath, self.prefix, tag=roi_tag, ch=ch)
            ca.load()
            info.append(ca.version_info)
        return info

    def export_point_events(self, event_type):
        '''
        Events with a single timestamp (e.g. behavior events).
        :param event_type: str, the event type's name
        :return: dict with keys:
            'times': 1D float numpy array, seconds
            'frames': 1D int numpy array, imaging frame index of each event, same length as 'times'
        '''
        raise NotImplementedError('export_point_events')

    def export_interval_events(self, event_type):
        '''
        Events with start and stop times (e.g. seizures).
        :param event_type: str, the event type's name
        :return: numpy array, shape (n_events, 2), columns: start, stop, seconds
        '''
        raise NotImplementedError('export_interval_events')

    def export_video_file(self):
        '''
        Path of the video file of the scene, in the raw data folder.
        :return: str
        '''
        return os.path.join(self.si.info['dpath'], os.path.basename(self.si.info['cam_file']))

    def export_video(self):
        '''
        Video of the scene, greyscale frames (e.g. read from a cv2.VideoCapture).
        Can be a generator, so the video does not have to fit in memory.
        :return: iterable of 2D uint8 numpy arrays (height, width), or a 3D array (n_frames, height, width)
        '''
        return self._iter_video_frames(self.export_video_file())

    @staticmethod
    def _iter_video_frames(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Could not open {path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (height, width)
        finally:
            cap.release()

    def export_video_times(self):
        '''
        Timestamp of each video frame.
        :return: 1D float numpy array, length n_video_frames, seconds
        '''
        # columns: camera frame, time in ms. As in PreProc.parse_cam, the first camera frame is at imaging time 0
        timers = numpy.load(os.path.join(self.si.info['dpath'], self.prefix + '_CamTimers.npy'))
        return (timers[:, 1] - timers[0, 1]) / 1000

    def export_video_frames(self):
        '''
        Imaging frame during which each video frame was recorded.
        :return: 1D int numpy array, length n_video_frames. -1 for video frames before the first or after the last
            imaging frame
        '''
        video_times = self.export_video_times()
        frame_times = self.ftimes[:, 0]
        frames = numpy.searchsorted(frame_times, video_times, side='right') - 1
        frames[video_times >= frame_times[-1] + 1 / self.fps] = -1
        return frames
