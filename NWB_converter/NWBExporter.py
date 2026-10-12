import os
import json
from datetime import datetime
from enum import Enum
from uuid import uuid4

import numpy
import yaml
from dateutil.tz import tzlocal
from hdmf.data_utils import DataChunkIterator
from pynwb import NWBHDF5IO, NWBFile, TimeSeries, H5DataIO
from pynwb.behavior import BehavioralTimeSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.ogen import OptogeneticStimulusSite
from pynwb.file import Subject
from pynwb.image import ImageSeries, RGBImage, Images
from pynwb.ophys import Fluorescence, ImageSegmentation, OpticalChannel, RoiResponseSeries, TwoPhotonSeries

from BaserowAPI.BaserowRequests import GetSessions
from NWB_converter.DandiUpload import write_upload_file
from NWB_converter.ExportImagingSession import ExportImagingSession
from Proc2P.Bruker.LoadRegistered import Source

'''
Converts an ImagingSession to an NWB file (pynwb 4.2, NWB schema 2.11).
Usage:
    NWBExporter('Dag47_2026-04-22_goal_128', 'path/to/export_config.yaml').run()
See export_config_template_2ch.yaml for the config options, and ExportImagingSession for the data formats.

File layout:
    acquisition/ImagingFrameTimes       frame index vs. frame time. Other per-frame series link to these timestamps
    acquisition/ImagingFrameAbsoluteTimes   frame time from the start of the acquisition
    acquisition/TwoPhotonSeries_<ch>    motion corrected 2P movie, one per channel
    acquisition/LFP                     ElectricalSeries, integer microvolts (conversion 1e-6), one column per channel
    acquisition/LFPImagingFrame         imaging frame index of each LFP sample (Ephys.frames)
    acquisition/Video                   ImageSeries of the scene video, timestamps in seconds. With video.storage external
                                        (default), a link to the video file by name; see DandiUpload for the upload
    acquisition/VideoImagingFrame       imaging frame index of each video frame (-1: outside imaging), links to the
                                        Video timestamps
    processing/ophys/FieldOfView        RGBImage
    processing/ophys/ImageSegmentation  one PlaneSegmentation per ROI set, named with the ROI tag
    processing/ophys/ImageSegmentation  PlaneSegmentations also have a 'polygon' column (ROI editor x, y vertices)
    processing/ophys/<roi_tag>          Fluorescence, with a RoiResponseSeries named <trace_tag>_<ch> per trace version
    processing/behavior/BehavioralTimeSeries/speed, smoothed_speed, position, belt_position, laps
    processing/optogenetics/StimFrames  bool per imaging frame, True in frames affected by the stimulation
    Tables without rows are not written; notes 'Export' lists the files they came from
    events/licks, rewards               EventsTable with timestamp and frame columns
    events/LFPSpikes_Ch<n>              spike times of LFP channel n, with the columns of the spiketimes file
    events/<event_type>                 EventsTable with timestamp and frame columns
    intervals/RewardZones               start_frame and stop_frame columns
    intervals/OptoStim                  photostimulation pulses, with the columns of _StimFrames.xlsx
    intervals/Ripples_<tag>             ripple events, with start_sample and stop_sample (LFP sample) columns
    intervals/Seizures_Ch<n>, SeizuresCurated_Ch<n>   seizures of LFP channel n, with the columns of the files
    intervals/<event_type>              TimeIntervals with start_time and stop_time columns
    notes                               json: {'SessionInfo': session metadata, 'Baserow': the session's DB row,
                                        'Export': what NWBImagingSession needs to rebuild the processed files
                                        (ROI set channels and trace info, LFP channels, detection settings)}
    subject                             subject_id and sex from the DB, other fields from the config
'''


class VideoStorage(Enum):
    '''config video.storage: how the scene video is stored'''
    EXTERNAL = 'external'  # ImageSeries.external_file with the video file name. The video is uploaded to DANDI next to
    # the NWB file (DandiUpload). Keeps the original encoding, no decoding during export
    EMBEDDED = 'embedded'  # decoded greyscale frames in the NWB file


class NWBExporter:
    __name__ = 'NWBExporter'

    def __init__(self, prefix, config_path, session=None):
        '''
        :param prefix: session prefix
        :param config_path: yaml export config (see export_config_template_2ch.yaml)
        :param session: optional, an ExportImagingSession to use instead of loading one based on the config
        '''
        self.prefix = prefix
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        if session is None:
            session = self.load_session()
        self.session = session
        self.video_storage = VideoStorage(self.cfg('video', 'storage', default='external'))
        self.nwbfile = None
        self.lfp = None

    def cfg(self, *keys, default=None):
        '''get a nested config value, or default if missing or null'''
        value = self.config
        for key in keys:
            if not isinstance(value, dict) or value.get(key) is None:
                return default
            value = value[key]
        return value

    def load_session(self):
        procpath = self.cfg('session', 'procpath')
        db_entry = None
        if procpath is None or self.cfg('session', 'use_db', default=True):
            results = GetSessions().get_session(self.prefix)
            if len(results) != 1:
                raise ValueError(f'Expected one DB entry for {self.prefix}, found {len(results)}')
            db_entry = results.iloc[0]
            if procpath is None:
                procpath = db_entry['Processed.Path']
        kwargs = self.cfg('session', 'imaging_session_kwargs', default={})
        return ExportImagingSession(procpath, self.prefix, db_entry=db_entry, **kwargs)

    def get_output_path(self):
        fn = self.cfg('output', 'filename', default='{prefix}.nwb').format(prefix=self.prefix)
        return os.path.join(self.cfg('output', 'directory'), fn)

    def run(self):
        '''Build the NWB file and save it. Returns the output path.'''
        path = self.get_output_path()
        if os.path.exists(path) and not self.cfg('output', 'overwrite', default=False):
            raise FileExistsError(f'{path} exists. Set output.overwrite in the config to replace it.')
        include = self.cfg('include', default={})

        self.create_nwbfile()
        self.add_devices()
        self.add_frame_times()
        if include.get('fov_image', True):
            self.add_fov_image()
        if include.get('2p_movie', True):
            self.add_2p_movie()
        if include.get('speed', True):
            self.add_speed()
        if include.get('abspos', True):
            self.add_position()
        if include.get('treadmill', True):
            self.add_treadmill()
        if include.get('behavior_events', True):
            self.add_behavior_events()
        if include.get('opto', True):
            self.add_opto()
        if include.get('lfp', True):
            self.add_lfp()
        for roi_set in self.cfg('roi_sets', default=[]):
            self.add_roi_set(roi_set)
        for event_type in self.cfg('point_events', default=[]):
            self.add_point_events(event_type)
        for event_type in self.cfg('interval_events', default=[]):
            self.add_interval_events(event_type)
        if include.get('video', True):
            self.add_video()

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        print(f'Writing {path}...')
        with NWBHDF5IO(path, 'w') as io:
            io.write(self.nwbfile)
        if include.get('video', True) and self.video_storage == VideoStorage.EXTERNAL:
            upload_file = write_upload_file(path, self.session.export_video_file(),
                                            self.cfg('dandi', 'dandiset_path'))
            print(f'DANDI upload paths and commands saved in {upload_file}')
        return path

    # file header
    def create_nwbfile(self):
        self.metadata = self.session.export_metadata()
        start_time = self.session.get_starttime()
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=tzlocal())

        lab = self.cfg('nwbfile', 'lab', default='')
        db_entry = self.session.db_entry
        session_id = self.prefix
        notes = {'SessionInfo': self.metadata, 'Export': self.get_export_record()}
        if db_entry is not None:
            session_id = f'{lab}-{db_entry["id"]}-{self.prefix}'
            notes['Baserow'] = db_entry.to_dict()
        experimenter = list(self.cfg('nwbfile', 'experimenter', default=[]))
        for name in self.metadata.get('experimenter', []):
            if name not in experimenter:
                experimenter.append(name)

        self.nwbfile = NWBFile(
            session_description=self.cfg('nwbfile', 'session_description', default=self.prefix),
            identifier=str(uuid4()),
            session_start_time=start_time,
            session_id=session_id,
            experimenter=experimenter,
            lab=lab,
            institution=self.cfg('nwbfile', 'institution'),
            experiment_description=self.cfg('nwbfile', 'experiment_description'),
            keywords=self.cfg('nwbfile', 'keywords'),
            related_publications=self.cfg('nwbfile', 'related_publications'),
            notes=json.dumps(notes, default=str, indent=1),
            subject=self.get_subject(),
        )
        self.ophys_module = self.nwbfile.create_processing_module(
            name='ophys', description='Processed 2P data converted from LNCM ImagingSession format')

    def get_lfp_channels(self):
        '''LFP channels to export: lfp.channels from the config, or all channels of the session if not set'''
        if not self.cfg('include', 'lfp', default=True):
            return []
        if self.lfp is None:
            self.lfp = self.session.export_lfp(self.cfg('lfp', 'channels'))
        return [] if self.lfp is None else self.lfp['channels']

    def get_export_record(self):
        '''what NWBImagingSession needs to rebuild the processed files. Saved in the notes json, under 'Export'.'''
        roi_sets = []
        for roi_set in self.cfg('roi_sets', default=[]):
            traces = [t if isinstance(t, str) else t['tag'] for t in roi_set.get('traces', [])]
            polygons = self.session.export_roi_polygons(roi_set['tag'])
            roi_sets.append({'tag': roi_set['tag'], 'channels': roi_set['channels'], 'traces': traces,
                             'trace_info': self.session.export_trace_info(roi_set['tag'], len(roi_set['channels'])),
                             'polygon_dtype': str(numpy.asarray(polygons[0]).dtype)})
        lfp_channels = self.get_lfp_channels()
        # tables without rows are not written in the NWB file, so the files and their columns are listed here
        lfp_tables = []
        if self.cfg('include', 'lfp_events', default=True):
            for table in self.session.export_lfp_tables(lfp_channels):
                lfp_tables.append({'name': table['name'], 'suffix': table['suffix'],
                                   'columns': [str(c) for c in table['table'].columns]})
        has_behavior_events = (self.cfg('include', 'behavior_events', default=True)
                               and self.session.export_behavior_events() is not None)
        return {'roi_sets': roi_sets, 'lfp_channels': lfp_channels, 'lfp_tables': lfp_tables,
                'behavior_events': has_behavior_events, 'settings': self.session.export_settings(lfp_channels)}

    def get_subject(self):
        subject = self.session.export_subject()
        subject.update(self.cfg('subject', default={}))
        if not subject:
            return None
        return Subject(**subject)

    def add_devices(self):
        dcfg = self.cfg('imaging', 'device', default={})
        model = self.nwbfile.create_device_model(
            name=dcfg.get('model_name', 'Microscope model'),
            description=dcfg.get('description', ''),
            manufacturer=dcfg.get('manufacturer', ''),
            model_number=dcfg.get('model_number'),
        )
        self.microscope = self.nwbfile.create_device(
            name=dcfg.get('name', 'Microscope'),
            description=dcfg.get('description', ''),
            model=model,
            serial_number=dcfg.get('serial_number'),
        )
        excitation_lambda = self.cfg('imaging', 'excitation_lambda', default=self.metadata.get('laserWavelength_0'))
        region = self.cfg('imaging', 'brain_region', default='unknown')
        # one imaging plane per channel
        self.imaging_planes = []
        for ch in self.cfg('imaging', 'channels'):
            optical_channel = OpticalChannel(name=ch['name'], description=ch.get('description', ''),
                                             emission_lambda=float(ch['emission_lambda']))
            self.imaging_planes.append(self.nwbfile.create_imaging_plane(
                name=f'ImagingPlane_{ch["name"]}',
                optical_channel=optical_channel,
                imaging_rate=float(self.session.fps),
                description=region,
                device=self.microscope,
                excitation_lambda=float(excitation_lambda),
                indicator=ch['indicator'],
                location=region,
            ))

    # imaging data
    def add_frame_times(self):
        frame_times = numpy.asarray(self.session.export_frame_times(), dtype='float64')
        self.n_frames = len(frame_times)
        # per-frame series store a link to these timestamps instead of a copy
        self.frame_times = TimeSeries(
            name='ImagingFrameTimes',
            description='Index of each imaging frame, with the frame timestamps',
            data=numpy.arange(self.n_frames),
            unit='frame',
            timestamps=frame_times,
        )
        self.nwbfile.add_acquisition(self.frame_times)
        self.nwbfile.add_acquisition(TimeSeries(
            name='ImagingFrameAbsoluteTimes',
            description='Time of each imaging frame from the start of the acquisition',
            data=numpy.asarray(self.session.export_frame_times_absolute(), dtype='float64'),
            unit='s',
            timestamps=self.frame_times,
        ))

    def add_fov_image(self):
        image = numpy.asarray(self.session.export_fov_image())
        images = Images(name='FieldOfView', description='RGB image of the field of view')
        images.add_image(RGBImage(name='FieldOfView_RGB', data=image,
                                  description='RGB image of the field of view'))
        self.ophys_module.add(images)

    def add_2p_movie(self):
        movies = self.session.export_2p_movie()
        channels = self.cfg('imaging', 'channels')
        if len(movies) > len(channels):
            raise ValueError(f'{len(movies)} 2P movie channels, but only {len(channels)} channels in config')
        level = self.cfg('compression', '2p_movie', default=4)
        for ch, movie, plane in zip(channels, movies, self.imaging_planes):
            self.check_length(movie, f'2P movie channel {ch["name"]}')
            self.nwbfile.add_acquisition(TwoPhotonSeries(
                name=f'TwoPhotonSeries_{ch["name"]}',
                description='Motion corrected (suite2p) 2P movie',
                data=self.wrap_large(movie, level, chunks=(1,) + tuple(movie.shape[1:])),
                imaging_plane=plane,
                unit='a.u.',
                timestamps=self.frame_times,
            ))

    def get_behavioral_time_series(self):
        '''the BehavioralTimeSeries container of the behavior module, created on first use'''
        if 'behavior' not in self.nwbfile.processing:
            behavior_module = self.nwbfile.create_processing_module(name='behavior', description='Behavior data')
            behavior_module.add(BehavioralTimeSeries())
        return self.nwbfile.processing['behavior']['BehavioralTimeSeries']

    def add_speed(self):
        speed = numpy.asarray(self.session.export_speed(), dtype='float64')
        self.check_length(speed, 'speed')
        self.get_behavioral_time_series().create_timeseries(
            name='speed',
            description='Running speed of the animal',
            data=speed,
            unit='cm/s',
            timestamps=self.frame_times,
        )
        smoothed_speed = numpy.asarray(self.session.export_smoothed_speed(), dtype='float64')
        self.check_length(smoothed_speed, 'smoothed speed')
        self.get_behavioral_time_series().create_timeseries(
            name='smoothed_speed',
            description='Smoothed running speed of the animal, used to find mobility periods',
            data=smoothed_speed,
            unit='cm/s',
            timestamps=self.frame_times,
        )

    def add_treadmill(self):
        for name, data, description, unit in (
                ('belt_position', self.session.export_belt_position(), 'Position on the belt, from the treadmill',
                 'a.u.'),
                ('laps', self.session.export_laps(), 'Lap index', 'lap')):
            data = numpy.asarray(data, dtype='float64')
            self.check_length(data, name)
            self.get_behavioral_time_series().create_timeseries(
                name=name, description=description, data=data, unit=unit, timestamps=self.frame_times)

    def add_behavior_events(self):
        events = self.session.export_behavior_events()
        if events is None:
            print(f'{self.prefix}: no behavior events')
            return
        frame_times = self.frame_times.timestamps
        for name, description in (('licks', 'Licks'), ('rewards', 'Reward deliveries')):
            if not len(events[name]):
                continue
            table = self.nwbfile.create_events_table(name=name, description=description)
            table.add_column(name='frame', description='imaging frame index of the event')
            for fr in events[name]:
                table.add_event(timestamp=float(frame_times[fr]), frame=int(fr))
        if not len(events['RZ']):
            return
        table = self.nwbfile.create_time_intervals(name='RewardZones', description='Reward zones')
        table.add_column(name='start_frame', description='imaging frame index of the start')
        table.add_column(name='stop_frame', description='imaging frame index of the stop')
        for start, stop in events['RZ']:
            table.add_interval(start_time=float(frame_times[start]), stop_time=float(frame_times[stop]),
                               start_frame=int(start), stop_frame=int(stop))

    def add_opto(self):
        opto = self.session.export_opto()
        if opto is None:
            print(f'{self.prefix}: no opto')
            return
        ocfg = self.cfg('opto', default={})
        device = self.nwbfile.create_device(name=ocfg.get('device_name') or 'Opto light source',
                                            description=ocfg.get('device_description') or '')
        excitation_lambda = ocfg.get('excitation_lambda')
        site = OptogeneticStimulusSite(name='OptoStimSite', device=device,
                                       description=ocfg.get('description') or '',
                                       excitation_lambda=float('nan') if excitation_lambda is None
                                       else float(excitation_lambda),
                                       location=ocfg.get('location') or 'unknown')
        self.nwbfile.add_ogen_site(site)
        # pulses: start at the frame time of the pulse's imaging frame, duration from the table (ms)
        pulses = opto['pulses']
        frame_times = self.frame_times.timestamps
        starts = frame_times[pulses['ImgFrame'].values]
        self.add_table_intervals('OptoStim', 'Photostimulation pulses (_StimFrames.xlsx). xlsx_index: voltage '
                                             'recording sample of the pulse; Intensity: relative; Duration: ms',
                                 pulses, starts, starts + pulses['Duration'].values / 1000)
        mask = numpy.zeros(self.n_frames, dtype='bool')
        mask[opto['stim_frames']] = True
        if 'optogenetics' not in self.nwbfile.processing:
            self.nwbfile.create_processing_module(name='optogenetics', description='Photostimulation')
        self.nwbfile.processing['optogenetics'].add(TimeSeries(
            name='StimFrames',
            description='True in imaging frames affected by the photostimulation (ImagingSession.opto)',
            data=mask,
            unit='n.a.',
            timestamps=self.frame_times,
        ))

    def add_position(self):
        position = numpy.asarray(self.session.export_position(), dtype='float64')
        self.check_length(position, 'position')
        self.get_behavioral_time_series().create_timeseries(
            name='position',
            description='Position of animal on the belt, relative to lap length',
            data=position,
            unit='lap',
            timestamps=self.frame_times,
        )

    def add_roi_set(self, roi_set):
        '''
        :param roi_set: dict from config: tag, source, weighted, description,
            channels: config channel names of the trace channels, in trace order,
            traces: list of trace tags (str, or dict with tag, description, unit)
        '''
        roi_tag = roi_set['tag']
        channel_names = [ch['name'] for ch in self.cfg('imaging', 'channels')]
        for name in roi_set['channels']:
            if name not in channel_names:
                raise ValueError(f'{roi_tag}: channel {name} is not in imaging.channels {channel_names}')
        if not hasattr(self, 'image_segmentation'):
            self.image_segmentation = ImageSegmentation()
            self.ophys_module.add(self.image_segmentation)
        masks = numpy.asarray(self.session.export_roi_masks(roi_tag, weighted=roi_set['weighted'],
                                                            source=Source[roi_set['source']]))
        plane_segmentation = self.image_segmentation.create_plane_segmentation(
            name=roi_tag,
            description=roi_set.get('description', ''),
            imaging_plane=self.imaging_planes[channel_names.index(roi_set['channels'][0])],
        )
        polygons = self.session.export_roi_polygons(roi_tag)
        if len(polygons) != len(masks):
            raise ValueError(f'{roi_tag}: {len(polygons)} ROI polygons, {len(masks)} masks')
        plane_segmentation.add_column(name='polygon', description='ROI polygon vertices (x, y) from the ROI editor',
                                      index=True)
        for mask, polygon in zip(masks, polygons):
            plane_segmentation.add_roi(image_mask=mask, polygon=numpy.asarray(polygon).tolist())
        n_rois = len(masks)
        rois = plane_segmentation.create_roi_table_region(region=list(range(n_rois)), description=f'all {roi_tag} ROIs')

        fluorescence = Fluorescence(name=roi_tag)
        self.ophys_module.add(fluorescence)
        for trace in roi_set.get('traces', []):
            if isinstance(trace, str):
                trace = {'tag': trace}
            trace_tag = trace['tag']
            data = numpy.asarray(self.session.export_traces(roi_tag, trace_tag, len(roi_set['channels'])))
            self.check_length(data, f'{roi_tag} {trace_tag} traces')
            if data.shape[1] != n_rois:
                raise ValueError(f'{roi_tag} {trace_tag} traces have {data.shape[1]} ROIs, masks have {n_rois}')
            if data.shape[2] != len(roi_set['channels']):
                raise ValueError(f'{roi_tag} {trace_tag} traces have {data.shape[2]} channels, '
                                 f'config roi_sets.channels has {len(roi_set["channels"])}')
            for chi, channel_name in enumerate(roi_set['channels']):
                fluorescence.create_roi_response_series(
                    name=f'{trace_tag}_{channel_name}',
                    description=trace.get('description', trace_tag),
                    data=H5DataIO(data[..., chi], compression='gzip'),
                    rois=rois,
                    unit=trace.get('unit', 'a.u.'),
                    timestamps=self.frame_times,
                )

    # ephys
    def add_lfp(self):
        channels = self.get_lfp_channels()
        if not channels:
            print(f'{self.prefix}: no LFP')
            return
        lfp = self.lfp
        ecfg = self.cfg('lfp', default={})
        device = self.nwbfile.create_device(name=ecfg.get('device_name', 'LFP amplifier'),
                                            description=ecfg.get('device_description', ''))
        group = self.nwbfile.create_electrode_group(name='LFP_electrode', description=ecfg.get('description', ''),
                                                    location=ecfg.get('location', 'unknown'), device=device)
        self.nwbfile.add_electrode_column(name='ephys_channel', description='channel number in the LFP recording '
                                                                            '(lfp_ch of ImagingSession)')
        for channel in channels:
            self.nwbfile.add_electrode(location=ecfg.get('location', 'unknown'), group=group,
                                       filtering=ecfg.get('filtering'), reference=ecfg.get('reference'),
                                       ephys_channel=channel)
        electrodes = self.nwbfile.create_electrode_table_region(region=list(range(len(channels))),
                                                                description='LFP electrodes')
        lfp_series = ElectricalSeries(
            name='LFP',
            description=ecfg.get('description', 'LFP recording'),
            data=H5DataIO(numpy.asarray(lfp['data']), compression='gzip'),
            electrodes=electrodes,
            rate=float(lfp['rate']),
            starting_time=float(lfp['starting_time']),
            conversion=float(lfp['conversion']),
            filtering=ecfg.get('filtering'),
        )
        self.nwbfile.add_acquisition(lfp_series)
        self.nwbfile.add_acquisition(TimeSeries(
            name='LFPImagingFrame',
            description='Index of the imaging frame of each LFP sample',
            data=H5DataIO(numpy.asarray(lfp['frames'], dtype='int64'), compression='gzip', shuffle=True),
            unit='frame',
            rate=float(lfp['rate']),
            starting_time=float(lfp['starting_time']),
        ))
        if self.cfg('include', 'ripples', default=True):
            self.add_ripples()
        if self.cfg('include', 'lfp_events', default=True):
            for table in self.session.export_lfp_tables(channels):
                self.add_lfp_table(table)

    def lfp_time(self, seconds):
        '''seconds from the first LFP sample to session time'''
        return numpy.asarray(seconds, dtype='float64') + self.lfp['starting_time']

    def add_ripples(self):
        ripples = self.session.export_ripples(self.cfg('ripples', 'tags'))
        for tag, events in ripples.items():
            table = self.nwbfile.create_time_intervals(
                name=f'Ripples_{tag}', description=f'Ripple events (Ripples tag {tag}). The detection runs on one '
                                                   f'LFP channel (ImagingSession lfp_ch), which is not recorded')
            table.add_column(name='start_sample', description='LFP sample index of the start')
            table.add_column(name='stop_sample', description='LFP sample index of the stop')
            times = self.lfp_time(events / self.lfp['rate'])
            for (start, stop), (t0, t1) in zip(events, times):
                table.add_interval(start_time=float(t0), stop_time=float(t1),
                                   start_sample=float(start), stop_sample=float(stop))

    def add_lfp_table(self, table):
        '''spike (events) or seizure (intervals) table of an LFP channel, see ExportImagingSession.LFP_TABLES'''
        df, time_columns = table['table'], table['time_columns']
        description = f"{table['name']}: contents of {table['suffix']}. Times in the columns {list(time_columns)} " \
                      f"are seconds from the first LFP sample"
        if len(time_columns) == 1:
            self.add_table_events(table['name'], description, df, self.lfp_time(df[time_columns[0]].values))
        else:
            self.add_table_intervals(table['name'], description, df, self.lfp_time(df[time_columns[0]].values),
                                     self.lfp_time(df[time_columns[1]].values))

    @staticmethod
    def table_columns(df):
        '''DataFrame columns as lists for NWB table columns. Datetimes and other objects are stored as str'''
        columns = {}
        for name in df.columns:
            values = df[name]
            if values.dtype.kind in 'biuf':
                columns[str(name)] = values.to_numpy()
            else:
                columns[str(name)] = values.astype(str).to_numpy()
        return columns

    def add_table_events(self, name, description, df, times):
        '''events table with the DataFrame's columns. Not written if the DataFrame is empty'''
        if not len(df):
            return
        table = self.nwbfile.create_events_table(name=name, description=description)
        columns = self.table_columns(df)
        for column in columns:
            table.add_column(name=column, description=column)
        for i, t in enumerate(times):
            table.add_event(timestamp=float(t), **{k: v[i] for k, v in columns.items()})

    def add_table_intervals(self, name, description, df, starts, stops):
        '''time intervals table with the DataFrame's columns. Not written if the DataFrame is empty'''
        if not len(df):
            return
        table = self.nwbfile.create_time_intervals(name=name, description=description)
        columns = self.table_columns(df)
        for column in columns:
            table.add_column(name=column, description=column)
        for i, (t0, t1) in enumerate(zip(starts, stops)):
            table.add_interval(start_time=float(t0), stop_time=float(t1), **{k: v[i] for k, v in columns.items()})

    # events
    def add_point_events(self, event_type):
        events = self.session.export_point_events(event_type)
        times, frames = numpy.asarray(events['times'], dtype='float64'), numpy.asarray(events['frames'], dtype='int64')
        if len(times) != len(frames):
            raise ValueError(f'{event_type}: {len(times)} times and {len(frames)} frames')
        table = self.nwbfile.create_events_table(name=event_type, description=f'{event_type} events')
        table.add_column(name='frame', description='imaging frame index of the event')
        for t, fr in zip(times, frames):
            table.add_event(timestamp=float(t), frame=int(fr))

    def add_interval_events(self, event_type):
        intervals = numpy.asarray(self.session.export_interval_events(event_type), dtype='float64').reshape(-1, 2)
        table = self.nwbfile.create_time_intervals(name=event_type, description=f'{event_type} events')
        for start, stop in intervals:
            table.add_interval(start_time=float(start), stop_time=float(stop))

    # video
    def add_video(self):
        video_times = numpy.asarray(self.session.export_video_times(), dtype='float64')
        video_frames = numpy.asarray(self.session.export_video_frames(), dtype='int64')
        if self.video_storage == VideoStorage.EXTERNAL:
            video = ImageSeries(
                name='Video',
                description='Video of the scene',
                external_file=[os.path.basename(self.session.export_video_file())],
                starting_frame=[0],
                format='external',
                unit='n.a.',
                timestamps=video_times,
            )
        else:
            video = ImageSeries(
                name='Video',
                description='Greyscale video of the scene',
                data=self.wrap_large(self.session.export_video(), self.cfg('compression', 'video', default=4)),
                unit='n.a.',
                format='raw',
                timestamps=video_times,
            )
        self.nwbfile.add_acquisition(video)
        # the schema only has times for video frames; the imaging frame of each video frame is a separate series
        self.nwbfile.add_acquisition(TimeSeries(
            name='VideoImagingFrame',
            description='Index of the imaging frame during which each video frame was recorded. '
                        '-1: before the first or after the last imaging frame',
            data=video_frames,
            unit='frame',
            timestamps=video,
        ))

    # helpers
    def check_length(self, data, label):
        if len(data) != self.n_frames:
            raise ValueError(f'{label} has {len(data)} frames, frame times have {self.n_frames}')

    def wrap_large(self, data, compression_level, chunks=None):
        '''write large arrays or frame generators in chunks, so they don't have to fit in memory'''
        iterator = DataChunkIterator(data=data, buffer_size=self.cfg('compression', 'buffer_frames', default=100))
        if compression_level:
            return H5DataIO(iterator, compression='gzip', compression_opts=int(compression_level), chunks=chunks)
        return H5DataIO(iterator, chunks=chunks)
