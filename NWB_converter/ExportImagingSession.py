import os.path

import numpy

from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Analysis.PullSignals import pull_signals
from Proc2P.Analysis.CaTrace import CaTrace
from Proc2P.Analysis.AnalysisClasses.Spatial import Spatial

import cv2

'''
ImagingSession with a uniform set of export methods, used by NWBExporter.
Each method returns one data type in the format NWBExporter expects (see the docstrings).
Times are always in seconds, relative to the same reference as the imaging frame times.
'''


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
        Session metadata. The whole dict is saved in the NWB file as a json string (NWBFile.notes),
        so values must be json serializable (non-serializable values are converted with str).
        Reserved keys, used to fill NWB fields:
            'session_start_time': datetime or ISO 8601 str. Required. Naive datetimes are assumed local time.
            'experimenter': list of str, optional. Added to the experimenters in the config.
            'excitation_lambda': float (nm), optional. Used if not set in the config.
            'subject': dict, optional. Keys of pynwb.file.Subject (subject_id, sex, species, age, genotype,
                strain, date_of_birth, description). Values in the config take precedence.
        :return: dict
        '''
        return self.si.info

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

    def export_speed(self):
        '''
        Running speed of the animal, one value per imaging frame.
        :return: 1D float numpy array, length n_frames, cm/s
        '''
        if self.has_behavior:
            return self.pos.speed
        raise NotImplementedError('export_speed')

    def export_position(self):
        '''
        Absolute position of the animal on the belt, relative to lap length
        :return: 1D float numpy array, length n_frames
        '''
        if self.has_behavior:
            s = Spatial()
            s.init_session(self)
            s.set_pos()
            return s.pos
        raise NotImplementedError('export_speed')

    def export_lfp(self):
        '''
        LFP recording (single channel).
        :return: dict with keys:
            'data': 1D numpy array, length n_samples
            'rate': float, sampling rate in Hz
            'starting_time': float, time of the first sample in seconds (same reference as frame times)
            'conversion': float, factor that converts data to volts (e.g. 1e-3 if data is in mV)
        '''
        if self.has_ephys:
            return self.ephys.trace / 1000
        raise NotImplementedError('export_lfp')

    def export_roi_masks(self, roi_tag, weighted, source):
        '''
        ROI masks of one ROI set.
        :param roi_tag: str, the ROI set's tag
        :return: numpy array, shape (n_rois, height, width), bool, or float weights between 0 and 1
        '''
        return pull_signals(self.procpath, self.prefix, tag=roi_tag, ch=0, snr_weighted=weighted, enable_alt_path=True,
                         overwrite=True, use_movie=source, mask_only=True)

    def export_traces(self, roi_tag, trace_tag):
        '''
        One version of the traces pulled from one ROI set. ROI order must match export_roi_masks(roi_tag).
        :param roi_tag: str, the ROI set's tag
        :param trace_tag: str, the trace version (e.g. 'raw', 'dff', 'norm')
        :return: numpy array, shape (n_frames, n_rois, n_channels). Channels in the order of imaging.channels.
        '''
        ca = CaTrace(self.procpath, self.prefix, tag=roi_tag, ch='All')
        ca.load()
        data = ca.__getattribute__(trace_tag)
        if data.data.ndim == 2:
            return data.transpose()
        else:
            return data.transpose(1, 0, 2)

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

    def export_video(self):
        '''
        Video of the scene, greyscale frames (e.g. read from a cv2.VideoCapture).
        Can be a generator, so the video does not have to fit in memory.
        :return: iterable of 2D uint8 numpy arrays (height, width), or a 3D array (n_frames, height, width)
        '''
        return self._iter_video_frames(self.si.info['cam_file'])

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
        return self.ca.sync.load('cam')
