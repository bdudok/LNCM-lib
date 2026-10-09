import json
import shutil
import tempfile
import weakref
from pathlib import Path

import numpy
import pandas
import tifffile
from pynwb import NWBHDF5IO

from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Analysis.Ripples import Ripples

'''
ImagingSession loaded from an NWB file written by NWBExporter.
Usage, same arguments as ImagingSession, with the folder of the NWB files as procpath:
    with NWBImagingSession('path/to/nwb_folder', prefix, tag='CellsGR') as session:
        vm = session.getparam('vm')
    or NWBImagingSession(..., nwb_filename='{prefix}-lean.nwb') if the files were exported with another name.

The NWB contents are written to a temporary processed folder in the layout ImagingSession reads, and ImagingSession
is initialized from that folder. So all ImagingSession methods, and code that reads the session's files
(get_file_with_suffix, EventMasks, PhotoStim, Ripples), work as with the original processed folder.
Derived data that is not in the NWB file (filtered LFP, ripple and band power, LFP spike rate) is computed in the
temporary folder when first used, as ImagingSession does when those files are missing. Stored ripple events are
loaded; without them, map_ripples runs the ripple detection with the default settings.
The temporary folder is deleted by close(), at the end of a with block, or when the object is garbage collected.

Only what was exported is available: e.g. no ROI sets or trace versions other than the exported ones,
no eye or face traces yet.
'''


class NWBImagingSession(ImagingSession):
    __name__ = 'NWBImagingSession'

    def __init__(self, procpath, prefix, tag='dummy', norip=True, opath='.', ch=0, ripple_tag=None, lfp_ch=1,
                 nwb_filename='{prefix}.nwb', temp_root=None, **kwargs):
        '''
        :param procpath: folder of the NWB file
        :param prefix: session prefix
        :param nwb_filename: name of the NWB file in procpath. {prefix} is replaced with the session prefix
        :param temp_root: folder for the temporary processed folder. None: the system's temp folder
        Other arguments are passed to ImagingSession.
        '''
        self.nwb_path = Path(procpath) / nwb_filename.format(prefix=prefix)
        if not self.nwb_path.exists():
            raise FileNotFoundError(f'NWB file not found: {self.nwb_path}')
        self.temp_folder = Path(tempfile.mkdtemp(prefix='NWBImagingSession_', dir=temp_root))
        self._cleanup = weakref.finalize(self, shutil.rmtree, self.temp_folder, ignore_errors=True)
        self.stored_ripples = {}
        self.ripples_written = False
        self.extract(prefix, tag)
        super().__init__(self.temp_folder, prefix, tag=tag, norip=norip, opath=opath, ch=ch, ripple_tag=ripple_tag,
                         lfp_ch=lfp_ch, **kwargs)
        self.version = 'NWB'

    def close(self):
        '''delete the temporary processed folder'''
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def map_ripples(self):
        if self.stored_ripples and not self.ripples_written:
            self.write_ripples()
        super().map_ripples()

    def write_ripples(self):
        '''
        Write the stored ripple events. Ripples only loads events together with the filtered LFP,
        so that is computed first (from lfp_ch, as ImagingSession.map_ripples uses it).
        '''
        Ripples(self.procpath, self.prefix, fs=self.si['fs'], ephys_channel=self.lfp_ch, force=True,
                keep_ripples=True)
        for tag, events in self.stored_ripples.items():
            events.astype('float64').tofile(self.path / '_ripples' / f'{tag}.ripples')
        self.ripples_written = True

    # writing the processed files
    def extract(self, prefix, tag):
        path = self.temp_folder / prefix
        path.mkdir()

        def fn(suffix):
            return path / f'{prefix}{suffix}'

        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwb = io.read()
            notes = json.loads(nwb.notes)
            if 'Export' not in notes:
                raise ValueError(f'{self.nwb_path} was exported without the information NWBImagingSession needs. '
                                 f'Export it again with the current NWBExporter.')
            export = notes['Export']
            with open(fn('_SessionInfo.json'), 'w') as f:
                json.dump(notes['SessionInfo'], f)
            for name, contents in export['settings'].items():
                with open(fn(name), 'w') as f:
                    json.dump(contents, f)

            frame_times = nwb.acquisition['ImagingFrameTimes'].timestamps[:]
            absolute_times = nwb.acquisition['ImagingFrameAbsoluteTimes'].data[:]
            numpy.save(fn('_FrameTimes.npy'), numpy.column_stack((frame_times, absolute_times)))

            self.extract_behavior(nwb, export, fn)
            self.extract_opto(nwb, fn)
            self.extract_rois(nwb, export, path, prefix, tag, fn)
            self.extract_lfp(nwb, export, fn)

    @staticmethod
    def get_table(container, name):
        '''an events or intervals table as a DataFrame, None if the file doesn't have it'''
        if container is None or name not in container:
            return None
        return container[name].to_dataframe()

    def extract_behavior(self, nwb, export, fn):
        if 'behavior' in nwb.processing:
            files = {'speed': '_spd.npy', 'smoothed_speed': '_smspd.npy', 'belt_position': '_pos.npy',
                     'laps': '_laps.npy'}
            series = nwb.processing['behavior']['BehavioralTimeSeries'].time_series
            for name, suffix in files.items():
                if name in series:
                    numpy.save(fn(suffix), series[name].data[:])
        if export['behavior_events']:
            bdat = {}
            for name in ('licks', 'rewards'):
                table = self.get_table(nwb.events, name)
                bdat[name] = [] if table is None else [int(x) for x in table['frame']]
            table = self.get_table(nwb.intervals, 'RewardZones')
            bdat['RZ'] = [] if table is None else [[int(a), int(b)] for a, b in
                                                   zip(table['start_frame'], table['stop_frame'])]
            with open(fn('_bdat.json'), 'w') as f:
                json.dump(bdat, f)

    def extract_opto(self, nwb, fn):
        if 'optogenetics' not in nwb.processing:
            return
        mask = nwb.processing['optogenetics']['StimFrames'].data[:]
        numpy.save(fn('_bad_frames.npy'), numpy.where(mask)[0])
        table = self.get_table(nwb.intervals, 'OptoStim')
        self.write_table_file(table, ['xlsx_index', 'Intensity', 'ImgFrame', 'Duration'], fn('_StimFrames.xlsx'))

    @staticmethod
    def write_table_file(table, columns, path):
        '''write a table as the Excel file it was exported from (see ExportImagingSession.read_table_file)'''
        if table is None:
            table = pandas.DataFrame(columns=columns)
        table = table[columns].set_index('xlsx_index')
        table.index.name = None
        table.to_excel(path)

    def extract_rois(self, nwb, export, path, prefix, tag, fn):
        roi_tags = [roi_set['tag'] for roi_set in export['roi_sets']]
        if tag not in ('skip', 'off', 'dummy') and tag not in roi_tags:
            raise ValueError(f'ROI set {tag} is not in {self.nwb_path}. Available: {roi_tags}')
        ophys = nwb.processing['ophys']
        if 'FieldOfView' in ophys.data_interfaces:
            image = ophys['FieldOfView']['FieldOfView_RGB'].data[:]
            tifffile.imwrite(fn('_preview.tif'), image)
            # movie dimensions, read by LoadRegistered (e.g. for the FOV size in ImagingSession.qc). The movie
            # itself is not extracted, so image data can't be loaded
            ops_folder = path / 'suite2p' / 'plane0'
            ops_folder.mkdir(parents=True)
            with open(ops_folder / 'ops_compat.json', 'w') as f:
                json.dump({'n_frames': len(nwb.acquisition['ImagingFrameTimes'].timestamps),
                           'Ly': image.shape[0], 'Lx': image.shape[1]}, f)
        for roi_set in export['roi_sets']:
            roi_tag = roi_set['tag']
            if 'trace' not in roi_set['traces']:
                raise ValueError(f'{self.nwb_path}: ROI set {roi_tag} was exported without the trace version '
                                 f"'trace', which CaTrace needs")
            # ROI file of the ROI editor: rows of (roi index, x, y)
            polygons = ophys['ImageSegmentation'][roi_tag]['polygon']
            rows = [[i, x, y] for i in range(len(polygons)) for x, y in polygons[i]]
            numpy.save(fn(f'_saved_roi_{roi_tag}.npy'), numpy.array(rows, dtype=roi_set['polygon_dtype']))
            fluorescence = ophys[roi_tag]
            for ch, channel_name in enumerate(roi_set['channels']):
                folder = path / f'{prefix}_trace_{roi_tag}-ch{ch}'
                folder.mkdir()
                for trace_tag in roi_set['traces']:
                    data = fluorescence[f'{trace_tag}_{channel_name}'].data[:]
                    numpy.save(folder / f'{trace_tag}.npy', data.transpose())
                numpy.save(folder / 'info.npy', roi_set['trace_info'][ch])

    def extract_lfp(self, nwb, export, fn):
        if 'LFP' not in nwb.acquisition:
            return
        lfp = nwb.acquisition['LFP']
        channels = [int(c) for c in lfp.electrodes.table['ephys_channel'][:]]
        data = lfp.data[:]
        # rows of _ephys.npy: imaging frame of each sample, then the channels. Channels that were not exported are 0
        edat = numpy.zeros((max(channels) + 1, len(data)), dtype='int64')
        edat[0] = nwb.acquisition['LFPImagingFrame'].data[:]
        for column, channel in enumerate(channels):
            edat[channel] = data[:, column]
        numpy.save(fn('_ephys.npy'), edat)
        for table in export['lfp_tables']:
            name = table['name']
            container = nwb.events if name.startswith('LFPSpikes') else nwb.intervals
            self.write_table_file(self.get_table(container, name), table['columns'], fn(table['suffix']))
        for name in (nwb.intervals or {}):
            if name.startswith('Ripples_'):
                table = nwb.intervals[name].to_dataframe()
                self.stored_ripples[name[len('Ripples_'):]] = table[['start_sample', 'stop_sample']].to_numpy()
