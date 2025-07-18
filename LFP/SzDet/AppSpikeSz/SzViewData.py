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

# analysis for sz curation
from scipy import signal
from scipy.ndimage import gaussian_filter
from LFP.EphysFunctions import butter_bandpass_filter

'''
Manually review automatically detected seizures
'''


class SzReviewData:
    __name__ = 'SzReviewData'
    '''
    Used by Sz Review GUI to access data
    '''

    def __init__(self, path, prefix, ch, tag=None, setup='Pinnacle', skip_gamma=False):
        self.path = path
        self.prefix = prefix
        self.ch = ch
        self.tag = tag
        self.setup = setup
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

        # ignore video now. add this after the lfp review features work and can be used in practice
        # get all ttl times from ephys and pass to video class to align all. this should be done once and stored
        # there should be a selector in the gui that allows the user to browse to video path if these are not in config
        # self.vid = self.read_video(self.settings['vid'], self.settings['vidfn'])

    def get_fn(self, suffix, timestamp=False):
        if self.setup == 'Pinnacle':
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
        elif self.setup == 'LNCM':
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
            return os.path.join(self.path, self.prefix, self.prefix + fn)
        else:
            return -1

    def read_ephys(self):
        if self.setup == 'Pinnacle':
            self.ephys = ReadEDF.EDF(self.path, self.prefix, ch=int(self.ch)-1)
            # ttls = self.ephys.get_TTL()
            startdate = self.ephys.d[-1]['startdate']
            annotations = self.ephys.d[-1]['annotations']
        if self.setup == 'LNCM':
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
        if self.setup in ('Pinnacle', 'LNCM'):
            start_key = 'Sz.Start(s)'
            stop_key = 'Sz.Stop(s)'
            spike_key = 'SpikeTimes(s)'
        spiketimes = self.spikes[spike_key].values
        for _, sz in self.input_sz.iterrows():
            sztime = self.rec_info['startdate'] + datetime.timedelta(seconds=sz[start_key])
            sznames.append(f'{sztime:%H:%M:%S(%m%d)}')
            sztimes.append(sztime)
            szdurs.append(sz[stop_key] - sz[start_key])
            incl_spikes = numpy.logical_and(spiketimes > sz[start_key], spiketimes < sz[stop_key])
            spkcount.append(numpy.count_nonzero(incl_spikes) + 1)

        # compute features for analysis
        self.output_sz = pandas.DataFrame({'Start': self.input_sz[start_key].values,
                                           'Stop': self.input_sz[stop_key].values, 'Duration(s)': szdurs,
                                           'SpikeCount': spkcount,
                                           'SpkFreq': numpy.array(spkcount) / numpy.array(szdurs),
                                           'PostIctalSuppression(s)': '',
                                           'Time': sztimes, 'Included': '', 'Interictal': False, }, index=[sznames])
        # load previous exclusions
        fn = self.get_fn('save')
        if os.path.exists(fn):
            saved = read_excel(fn)
            for szname in sznames:
                for fieldname in ('Included', 'Interictal'):
                    self.output_sz.loc[szname, fieldname] = saved.loc[szname][fieldname]

        self.szlist = sznames

    def set_sz(self, sz, value, key='Included'):
        self.output_sz.loc[sz, key] = value

    def get_sz(self, sz, full=False):
        if full:
            return self.output_sz.loc[sz].iloc[0]
        return self.output_sz.loc[sz, 'Included'].values[0]

    # def read_video(self, path, prefix):
    #     #load all ttls so that each sz can be matched to relevant vid
    #     #pull motion energy of the vids
    #     #provide interface for getting a frame by rec time, and getting all frames between two time points
    #     return SyncVid(path, prefix)

    def plot_sz(self, sz_name, axd):
        for _, ca in axd.items():
            ca.cla()
        sz = self.get_sz(sz_name, True)
        span_sec = 10  # plot flanking the sz start and stop
        t0, t1 = sz['Start'], sz['Stop']

        s0 = max(0, int((t0 - span_sec) * self.fs))  # sample where plot starts
        s1 = min(int((t1 + span_sec) * self.fs), len(self.ephys.trace))  # sample where the plot ends
        ds0 = int(t0 * self.fs) - s0  # samples between plot start and sz start
        # ds1 = int((t1 - t0) * self.fs) - ds0  # samples between plot start and sz end
        sx = numpy.arange(s1 - s0)
        secs = numpy.arange(-span_sec, 2 * span_sec + 1 + t1 - t0, span_sec, dtype='int')
        window_w = int(0.1 * self.fs)

        y = self.ephys.trace[s0:s1]
        first_plot_sample = int(span_sec * self.fs)
        last_plot_sample = int((span_sec + t1 - t0) * self.fs)
        y_range = numpy.percentile(numpy.absolute(y), 99) * 1.5
        sz_gamma = gaussian_filter(self.gamma_power[s0:s1], self.fs * 1.0)

        gamma_mask = sz_gamma < (sz_gamma.max() / self.settings["Curation.PISMultiplier"])
        gamma_mask[:last_plot_sample] = False

        # full sz trace
        ca = axd['top']
        strip_ax(ca, False)
        ca.set_ylabel('LFP(uV)')

        ca.axvline(first_plot_sample, color='red', )  # linestyle=':')
        ca.axvline(last_plot_sample, color='red', )  # linestyle=':')
        ca.set_xticks(secs * self.fs)
        ca.set_xticklabels(secs)
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
        spike_array = numpy.empty((len(sz_spikes), 2 * window_w))
        for si, s in enumerate(sz_spikes):
            y = self.ephys.trace[s - window_w:s + window_w]
            ca.plot(y, color='black', linewidth=0.5, alpha=0.2)
            spike_array[si] = y - numpy.mean(y[:int(window_w / 2)])
        ca.plot(numpy.nanmean(spike_array, axis=0), color='red', linewidth=1)
        ca.set_xlim(0, 2 * window_w)
        ca.set_xticks([0, window_w, 2 * window_w])
        ca.set_xticklabels([-0.1, 0, 0.1])
        # return fig

    def save(self):
        try:
            self.output_sz.to_excel(self.get_fn('save'))
        except PermissionError:
            print('''Permission error writing the output spreadsheet. Saving to new file, please rename later.''')
            self.output_sz.to_excel(self.get_fn('save', timestamp=True))


if __name__ == '__main__':
    setup = 'Pinnacle'
    path = 'D:\Shares\Data\_Processed\EEG\Tottering/'
    prefix = 'Tot9_2024-05-20_07_06_20_export'
    setup = 'LNCM'
    path = 'D:\Shares\Data\_Processed/2P\PVTot/'
    prefix = 'PVTot9_2024-02-20_lfp_181'

    szdat = SzReviewData(path, prefix, 2, setup=setup)
    self = szdat
    sz = szdat.get_sz(self.szlist[1], True)
