import os.path

import matplotlib.pyplot as plt

from Proc2P.utils import *
import json
import numpy
import datetime
from PlotTools import *

# readers
from LFP.Pinnacle import ReadEDF
from Proc2P.Bruker import LoadEphys
from Proc2P.Bruker.PreProc import SessionInfo
from Video import LoadVEEG, MotionEnergy

# analysis for sz curation
from scipy import signal
# from scipy.ndimage import gaussian_filter #this is slow, don't use it it bogs the gui down
from scipy.signal import bessel, sosfilt
from LFP.EphysFunctions import butter_bandpass_filter
import traceback
from LFP.Setup import Setup
'''
Manually review automatically detected seizures
'''


class SzReviewData:
    __name__ = 'SzReviewData'
    '''
    Used by Sz Review GUI to access data
    '''

    def __init__(self, path, prefix, ch, tag=None, setup=Setup.PINNACLE, skip_gamma=False, video_dir=None,
                 pull_motion=False):
        '''video_dir: folder of the vEEG recordings (Pinnacle setup), default: the EEG folder
        pull_motion: if the motion energy trace is not saved yet, start computing it in the background (it is shown
            when the recording is opened next time)'''
        self.path = path
        self.prefix = prefix
        self.ch = ch
        self.tag = tag
        self.setup = Setup(setup)
        self.video_dir = path if video_dir is None else video_dir
        self.video_map = None
        self.pull_motion = pull_motion
        self.load_data(skip_gamma)
        self.init_output()

    def load_data(self, skip_gamma):
        self.input_sz = read_excel(self.get_fn('sztime'))
        self.spikes = read_excel(self.get_fn('spiketime'))
        with open(self.get_fn('settings'), 'r') as f:
            self.settings = self.read_settings_with_defaults(json.loads(f.read()))
        self.fs = int(float(self.settings['fs']))
        self.spike_samples = (self.spikes['SpikeTimes(s)'].values * self.fs).astype('int')
        self.read_ephys()
        if not skip_gamma:
            self.get_session_gamma()
            self.sos = bessel(N=3, Wn=1.0, btype='lowpass', analog=False, output='sos', fs=self.fs)
        if self.setup == Setup.PINNACLE:
            self.load_video()
        self.load_motion()

    def get_fn(self, suffix, timestamp=False):
        if self.setup == Setup.PINNACLE:
            if self.tag is None:
                tagstr = ''
            else:
                tagstr = f'_{self.tag}'
            if suffix == 'sztime':
                fn = f'.edf{tagstr}_Ch{self.ch}_seizure_times.xlsx'
            elif suffix == 'save':
                if timestamp:
                    datestr = f'_{datetime.datetime.now():%Y%m%d}_'
                else:
                    datestr = ''
                fn = f'.edf{tagstr}_Ch{self.ch}_seizure_times_curated{datestr}.xlsx'
            elif suffix == 'spiketime':
                fn = f'.edf{tagstr}_Ch{self.ch}_spiketimes.xlsx'
            elif suffix == 'settings':
                fn = f'_Ch{self.ch}_SpikeSzDet{tagstr}.json'
            elif suffix == 'gamma':
                fn = f'_Ch{self.ch}_gamma{tagstr}.npy'
            else:
                return -1
            return os.path.join(self.path, self.prefix + fn)
        elif self.setup == Setup.LNCM:
            if suffix == 'sztime':
                fn = f'_Ch{self.ch}_seizure_times.xlsx'
            elif suffix == 'save':
                fn = f'_Ch{self.ch}_seizure_times_curated.xlsx'
            elif suffix == 'spiketime':
                fn = f'_Ch{self.ch}_spiketimes.xlsx'
            elif suffix == 'settings':
                fn = f'_Ch{self.ch}_SpikeSzDet.json'
            elif suffix == 'gamma':
                fn = f'_Ch{self.ch}_gamma.npy'
            else:
                return -1
            if self.tag not in ('', None, 'None'):
                if suffix == 'settings':
                    fn = fn.replace('.json', f'_{self.tag}.json') # unfortunately settings are named like this
                elif suffix != 'gamma':
                    fn = f'_{self.tag}' + fn #no need to keep separate traces for tagged versions
            return os.path.join(self.path, self.prefix, self.prefix + fn)
        else:
            return -1

    def read_ephys(self):
        if self.setup == Setup.PINNACLE:
            self.ephys = ReadEDF.EDF(self.path, self.prefix, ch=int(self.ch) - 1, rejection_ops=self.settings)
            assert int(self.ephys.fs) == int(float(self.settings['fs'])), 'fs setting in config does not match the data'
            # ttls = self.ephys.get_TTL()
            startdate = self.ephys.d[-1]['startdate']
            print(startdate)
            annotations = self.ephys.d[-1]['annotations']
        if self.setup == Setup.LNCM:
            self.ephys = LoadEphys.Ephys(self.path, self.prefix, channel=int(self.ch))
            si = SessionInfo(os.path.join(self.path, self.prefix) + '/', self.prefix)
            si.load()  # "treadmill_fn": "PVTot9_2024-02-20_lfp_181-2024-02-20-141140.txt"
            if "treadmill_fn" in si.info:
                tstamp = si["treadmill_fn"]
                dtstring = tstamp[len(self.prefix) + 1:-4]
                date_format = '%Y-%m-%d-%H%M%S'
                startdate = datetime.datetime.strptime(dtstring, date_format)
            else:
                dtstring = self.prefix.split('_')[1]
                date_format = '%Y-%m-%d'
                startdate = datetime.datetime.strptime(dtstring, date_format)
            annotations = []
        self.rec_info = {
            'startdate': startdate,
            'annotations': annotations
        }

    def read_settings_with_defaults(self, user_settings):
        '''complete user settings with defaults if not specified'''
        defaults = {
            "Curation.MinDur": 10,  # minimum duration for ictal/interictal (s)
            "Curation.MinFreq": 2.5,  # minimum average spike rate for seizure (Hz)
            "Curation.PISBand": (20, 50),  # frequency band for evaluating postictal suppression (Hz)
            "Curation.PISDur": 6,  # postictal window (s)
            "Curation.PISMultiplier": 10,  # max postictal relative to peak during sz
            "PlotMargin": 10,  # time shown before/after each sz
        }
        for key, value in defaults.items():
            if key not in user_settings:
                user_settings[key] = value
        return user_settings

    def get_session_gamma(self):
        gamma_fn = self.get_fn('gamma')
        if os.path.exists(gamma_fn):
            self.gamma_power = numpy.load(gamma_fn)
        else:
            ftr = butter_bandpass_filter(self.ephys.trace, *self.settings["Curation.PISBand"], self.fs)
            g = numpy.abs(signal.hilbert(ftr))
            # z score based on baseline
            g1 = g - g.mean()
            gmask = g1 < g1.std()
            g -= g[gmask].mean()
            g /= g[gmask].std()
            self.gamma_power = g
            numpy.save(gamma_fn, self.gamma_power)

    def init_output(self):
        sznames, sztimes, szdurs = [], [], []
        spkcount = []
        self.plot_margin = self.settings["PlotMargin"]
        if self.setup in (Setup.PINNACLE, Setup.LNCM):
            start_key = 'Sz.Start(s)'
            stop_key = 'Sz.Stop(s)'
            spike_key = 'SpikeTimes(s)'
        self.spiketimes = self.spikes[spike_key].values
        for _, sz in self.input_sz.iterrows():
            sztime = self.rec_info['startdate'] + datetime.timedelta(seconds=sz[start_key])
            sznames.append(f'{sztime:%H:%M:%S(%m%d)}')
            sztimes.append(sztime)
            szdurs.append(sz[stop_key] - sz[start_key])
            incl_spikes = numpy.logical_and(self.spiketimes > sz[start_key], self.spiketimes < sz[stop_key])
            spkcount.append(numpy.count_nonzero(incl_spikes) + 1)

        # compute features for analysis
        self.output_sz = pandas.DataFrame({'Start': self.input_sz[start_key].values,
                                           'Stop': self.input_sz[stop_key].values,
                                           'Edited': False, 'Duration(s)': szdurs,
                                           'SpikeCount': spkcount,
                                           'SpkFreq': numpy.array(spkcount) / numpy.array(szdurs),
                                           'PostIctalSuppression(s)': numpy.nan,
                                           'Time': sztimes, 'Included': pandas.NA, 'Interictal': False,
                                           'BehaviorSz(s)': numpy.nan, },
                                          index=[sznames])
        # load previous exclusions
        fn = self.get_fn('save')
        truthy_values = {1, True, 'TRUE'}
        falsey_values = {0, False, 'FALSE'}
        self.output_sz["Included"] = pandas.Series(
            pandas.NA, index=self.output_sz.index, dtype="boolean")
        if os.path.exists(fn):
            saved = read_excel(fn, keep_index=True)
            for szname in sznames:
                curated_sz = saved[saved.index == szname]
                for fieldname in ('Included', 'Interictal'):
                    if fieldname not in saved.columns:
                        continue
                    x = curated_sz[fieldname].item()
                    if x in truthy_values:
                        x = True
                    elif x in falsey_values:
                        x = False
                    self.set_sz(szname, x, fieldname)
                if 'BehaviorSz(s)' in saved.columns:
                    self.set_sz(szname, curated_sz['BehaviorSz(s)'].item(), 'BehaviorSz(s)')
                if 'Edited' in saved.columns and curated_sz['Edited'].item():
                    for fieldname in (
                            'Start', 'Stop', 'Duration(s)', 'SpikeCount', 'SpkFreq', 'PostIctalSuppression(s)'):
                        self.set_sz(szname, curated_sz[fieldname].item(), fieldname)

        self.szlist = sznames

    def set_sz(self, sz, value, key='Included'):
        try:
            self.output_sz.loc[sz, key] = value
        except Exception as e:
            print('-----Error with setting sz data------')
            print('Dtypes:', self.output_sz.dtypes)
            print(f'{key} type:', type(self.output_sz[key].dtype))
            print(f'Setting with: {value}({type(value)})')
            print("Error:", e)
            traceback.print_exc()

    def get_sz(self, sz, full=False):
        if full:
            return self.output_sz.loc[sz].iloc[0]
        return self.output_sz.loc[sz, 'Included'].array[0]

    def load_video(self):
        """Map the EEG to the vEEG recording whose TTL pulses match it. The map is saved next to the EEG and reused."""
        fn = os.path.join(self.path, self.prefix + '_video_map.csv')
        self.video_map = LoadVEEG.video_map_for_edf(self.ephys, self.video_dir, fn)

    def locate_video(self, video_fn):
        """Match the EEG to the recording of the camera that made video_fn, searching that file's folder. For when the
        automatic search doesn't find the video, e.g. because it is in another folder. Replaces the saved video map.
        Raises ValueError if the file is not from the vEEG recorder or its recording doesn't match the EEG's TTLs."""
        folder, name = os.path.split(video_fn)
        m = LoadVEEG.STEM_PATTERN.match(os.path.splitext(name)[0])
        if m is None:
            raise ValueError(f'{name} is not named like a vEEG recorder file (<camera>-<YYYY-MM-DDTHH-MM-SS>.mkv)')
        fn = os.path.join(self.path, self.prefix + '_video_map.csv')
        vmap = LoadVEEG.build_video_map_for_edf(self.ephys, folder, fn, camera=m.group('prefix'))
        if vmap is None:
            raise ValueError(f'The TTL pulses of camera {m.group("prefix")} in {folder} do not match this EEG '
                             f'(EEG start: {self.rec_info["startdate"]})')
        self.video_map = vmap
        MotionEnergy.trace_fn(self.setup, self.path, self.prefix).unlink(missing_ok=True)  # from a previous match
        self.load_motion()

    def load_motion(self):
        """Motion energy of the video, (time in EEG s, value), or None if it is not saved.
        If pull_motion, a missing trace is computed in a background process."""
        self.motion = MotionEnergy.load_trace(self.setup, self.path, self.prefix)
        if self.motion is None and self.pull_motion and (self.setup == Setup.LNCM or self.video_map is not None):
            MotionEnergy.pull_in_background(self.setup, self.path, self.prefix, self.video_dir)

    def video_frame(self, t):
        """Video file path and frame index at EEG time t (s), or None where there is no video."""
        loc = LoadVEEG.frame_at(self.video_map, t)
        if loc is None:
            return None
        return os.path.join(self.path, self.video_map['files'][loc[0]]['name']), loc[1]  # names relative to the EDF

    def edit_sz_from_gui(self, event_type, xcoord):
        sz = self.get_sz(self.active_sz, True)
        t0, t1 = sz['Start'], sz['Stop']
        new_time = t0 + xcoord / self.fs - self.plot_delta
        if not sz['Edited']:
            self.set_sz(self.active_sz, value=True, key='Edited')
            self.set_sz(self.active_sz, value=t0, key='OriginalStart')
            self.set_sz(self.active_sz, value=t1, key='OriginalStop')
        self.set_sz(self.active_sz, value=new_time, key=event_type)
        # update other sz features
        if event_type == 'Start':
            t0 = new_time
        elif event_type == 'Stop':
            t1 = new_time
        self.set_sz(self.active_sz, value=t1 - t0, key='Duration(s)')
        n_spikes = numpy.count_nonzero(numpy.logical_and(self.spiketimes > t0, self.spiketimes < t1)) + 1
        self.set_sz(self.active_sz, value=n_spikes, key='SpikeCount')
        self.set_sz(self.active_sz, value=n_spikes / (t1 - t0), key='SpkFreq')
        # PostIctalSuppression(s) is computed by plot_sz

    def plot_sz(self, sz_name, axd):
        self.active_sz = sz_name
        for _, ca in axd.items():
            ca.cla()
        sz = self.get_sz(sz_name, True)
        span_sec = self.plot_margin  # plot flanking the sz start and stop
        t0, t1 = sz['Start'], sz['Stop']
        # s0 = sample where plot starts
        if t0 > span_sec:
            s0 = int((t0 - span_sec) * self.fs)
            self.plot_delta = span_sec
        else:
            s0 = 0
            self.plot_delta = t0
        s1 = min(int((t1 + self.plot_delta) * self.fs), len(self.ephys.trace))  # sample where the plot ends
        ds0 = int(t0 * self.fs) - s0  # samples between plot start and sz start
        # ds1 = int((t1 - t0) * self.fs) - ds0  # samples between plot start and sz end
        sx = numpy.arange(s1 - s0)
        secs = numpy.arange(-span_sec, span_sec + t1 - t0 + 1, span_sec, dtype='int')
        window_w = int(0.1 * self.fs)

        y = self.ephys.trace[s0:s1]
        first_plot_sample = int(self.plot_delta * self.fs)
        last_plot_sample = int((self.plot_delta + t1 - t0) * self.fs)
        y_range = numpy.percentile(numpy.absolute(y), 99) * 1.5
        # sz_gamma = gaussian_filter(self.gamma_power[s0:s1], self.fs * 1.0)
        sz_gamma = sosfilt(self.sos, self.gamma_power[s0:s1])

        gamma_mask = sz_gamma < (sz_gamma.max() / self.settings["Curation.PISMultiplier"])
        gamma_mask[:last_plot_sample] = False

        # full sz trace
        ca = axd['top']
        strip_ax(ca, False)
        ca.set_ylabel('LFP(uV)')

        ca.axvline(first_plot_sample, color='red', )  # linestyle=':')
        ca.axvline(last_plot_sample, color='red', )  # linestyle=':')
        ca.plot(sx, y, color='black')
        # plot where suppressed
        if numpy.any(gamma_mask):
            ca.plot(numpy.ma.masked_where(~gamma_mask, sx), numpy.ma.masked_where(~gamma_mask, y), color='blue')
        ca.set_ylim(-y_range, y_range)
        sz_spikes = []
        for s in self.spike_samples:
            if s0 < s < s1:
                ca.axvline(s - s0, color='orange', linestyle=':', zorder=0)
                sz_spikes.append(s)

        pisdur = numpy.count_nonzero(gamma_mask) / self.fs
        self.set_sz(sz_name, pisdur, 'PostIctalSuppression(s)')
        ca.set_xticks((secs + self.plot_delta) * self.fs)
        ca.set_xticklabels(secs)

        # video time marker, moved by the GUI without redrawing the figure (animated: not part of the background)
        self.plot_t0 = t0 - self.plot_delta  # EEG time at x = 0
        self.plot_t1 = s1 / self.fs
        self.marker_line = ca.axvline(self.time_to_x(t0), color='blue', linestyle='--', zorder=100, animated=True)
        if not numpy.isnan(sz['BehaviorSz(s)']):
            ca.axvline(self.time_to_x(sz['BehaviorSz(s)']), color='magenta', linewidth=2, zorder=99)

        # motion energy, x axis shared with the top plot
        ca = axd['motion']
        strip_ax(ca, False)
        ca.set_ylabel('Motion')
        if self.motion is None:
            ca.text(0.5, 0.5, 'No motion energy trace', transform=ca.transAxes, ha='center', va='center')
        else:
            t, me = self.motion
            incl = (t >= self.plot_t0) & (t <= self.plot_t1)
            ca.plot(self.time_to_x(t[incl]), me[incl], color='tab:purple', linewidth=1)
        ca.axvline(first_plot_sample, color='red')
        ca.axvline(last_plot_sample, color='red')

        # sz start example
        ca = axd['lower left']
        strip_ax(ca, False)
        ca.set_ylabel('LFP(uV)')
        zoom_span = span_sec
        zoom_slice = slice(ds0, ds0 + int(zoom_span * self.fs))
        ca.plot(sx[zoom_slice] - ds0, y[zoom_slice], color='black')
        zoom_secs = numpy.arange(zoom_span + 1)
        ca.set_xticks(zoom_secs * self.fs)
        ca.set_xticklabels(zoom_secs)
        ca.set_xlabel('Time(s)')

        # triggered averages
        ca = axd['lower right']
        strip_ax(ca, False)
        ca.axvline(window_w, color='red', linestyle=':')
        spike_array = numpy.zeros((len(sz_spikes), 2 * window_w))

        for si, s in enumerate(sz_spikes):
            if not (window_w / 2) < s < (len(self.ephys.trace) * window_w / 2):
                continue
            y = self.ephys.trace[s - window_w:s + window_w]
            ca.plot(y, color='black', linewidth=0.5, alpha=0.2)
            spike_array[si] = y - numpy.mean(y[:int(window_w / 2)])
        ca.plot(numpy.nanmean(spike_array, axis=0), color='red', linewidth=1)
        ca.set_xlim(0, 2 * window_w)
        ca.set_xticks([0, window_w, 2 * window_w])
        ca.set_xticklabels([-0.1, 0, 0.1])
        # return fig

    def time_to_x(self, t):
        '''x coordinate of an EEG time (s) in the plot of the active seizure'''
        return (t - self.plot_t0) * self.fs

    def x_to_time(self, x):
        return self.plot_t0 + x / self.fs

    def save(self):
        try:
            self.output_sz.to_excel(self.get_fn('save'))
        except PermissionError:
            print('''Permission error writing the output spreadsheet. Saving to new file, please rename later.''')
            self.output_sz.to_excel(self.get_fn('save', timestamp=True))
