import json
import os
import copy
import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.metrics import mutual_info_score
# from BehaviorSession import BehaviorSession
from Proc2P.Analysis.LoadPolys import FrameDisplay, LoadPolys
from Proc2P.Analysis.CaTrace import CaTrace
from Proc2P.Bruker.ConfigVars import CF
from Proc2P.Bruker.PreProc import SessionInfo
from Proc2P.Bruker.LoadEphys import Ephys
from LFP import Filter
from Proc2P.utils import outlier_indices, gapless, startstop, lprint, read_excel, strip_ax, ewma
from Proc2P.Analysis.Ripples import Ripples
import time
from scipy import stats
from Proc2P.Bruker.Video import CropVideo
from Video.PullMotion import pull_motion_energy




class Pos:
    '''to hold attributes of movement, previously used by treadmill (Quad) class'''

    def __init__(self, sync, fps):
        self.speed = sync.load('spd')
        self.smspd = sync.load('smspd')
        self.pos = sync.load('pos')
        self.laps = sync.load('laps')
        self.movement = gapless(self.speed, gap=fps, threshold=2)  # cm/s
        if self.laps is not None and numpy.nanmax(self.laps) > 1:
            self.relpos = numpy.maximum(0, self.pos - numpy.nanpercentile(self.pos, 1))
            self.relpos = numpy.minimum(1, self.relpos / numpy.nanpercentile(self.relpos, 99))
        else:
            if self.pos is not None:
                self.relpos = self.pos / numpy.nanmax(self.pos)

    def drop_nan(self):
        self.speed = numpy.nan_to_num(self.speed)
        self.smspd = numpy.nan_to_num(self.smspd)
        self.movement = numpy.nan_to_num(self.movement)
        self.pos = numpy.nan_to_num(self.pos)
        self.laps = numpy.nan_to_num(self.laps)

    def get_total_distance(self):
        tdt = 0
        wh_notna = numpy.where(numpy.logical_not(numpy.isnan(self.laps)))
        if len(wh_notna[0]) > 10:
            pos = self.pos[wh_notna]
            laps = self.laps[wh_notna].astype('int64')
            for l in numpy.unique(laps):
                curr_lap = pos[laps == l]
                tdt += numpy.nan_to_num(numpy.nanmax(curr_lap) - numpy.nanmin(curr_lap))
        if numpy.isnan(tdt):
            tdt = 0
        return int(tdt)


class ImagingSession(object):
    __name__ = 'ImagingSession'

    def __init__(self, procpath, prefix, tag=None, norip=True, opath='.', ch=0,
                 ripple_tag=None, lfp_ch=1, **kwargs):
        self.opath = opath
        self.prefix = prefix
        self.procpath = procpath
        self.tag = tag
        self.path = os.path.join(procpath, prefix + '/')
        self.kwargs = kwargs

        # load info from json file
        self.si = SessionInfo(self.path, prefix)
        self.si.load()
        self.fps = self.si['framerate']
        self.CF = CF()
        self.CF.fps = self.si['framerate']

        # load polygons, option to access image data
        self.rois = FrameDisplay(procpath, prefix, tag=tag, lazy=True)

        # load processed traces
        self.ca = CaTrace(procpath, prefix, tag=tag, ch=ch)
        self.has_ca = self.ca.load() is True
        self.ftimes = numpy.load(self.get_file_with_suffix('_FrameTimes.npy'))

        # load photostimulation if available:
        if self.has_ca:
            self.map_opto()

        #load treadmill speed
        self.has_behavior = False
        self.map_pos()

        # map ephys
        self.lfp_ch = lfp_ch
        self.map_phys()
        if (not norip) and self.has_ephys:
            self.ripple_tag = ripple_tag
            self.map_ripples()
        # self.map_spiketimes()

        # map video
        self.eye = None  # implement later
        self.has_eye = False
        #to map motion energy trace, call self.map_face()

        if tag == 'skip':
            self.ca.frames = self.si['n_frames']
        self.dualch = self.ca.is_dual

        self.pltnum = 0
        self.zscores = {}
        self.rates = {}
        self.colors = {}

        self.preview = None
        self.version = 'LNCM'

    def get_file_with_suffix(self, suffix):
        return os.path.join(self.path, self.prefix + suffix)

    def map_opto(self):
        if self.si['has_opto']:
            opto_frames = self.ca.sync.load('opto')
            self.opto = numpy.zeros(self.ca.frames, dtype='bool')
            self.opto[opto_frames] = 1
            if len(opto_frames) > 3:
                self.opto_ints = numpy.zeros(self.ca.frames)
                intdat = read_excel(self.get_file_with_suffix('_StimFrames.xlsx'))
                self.opto_ints[intdat['ImgFrame'].values] = intdat['Intensity'].values
        else:
            self.opto = None

    def map_pos(self):
        self.pos = Pos(self.ca.sync, self.fps)
        if self.pos.speed is None:  # in case this data is missing
            if self.has_ca:
                self.pos.speed = numpy.zeros(self.ca.frames)
                self.pos.smspd = numpy.zeros(self.ca.frames)
                self.pos.pos = numpy.zeros(self.ca.frames)
                self.pos.movement = numpy.zeros(self.ca.frames, dtype='bool')
        else:
            bdat = self.get_file_with_suffix('_bdat.json')
            if os.path.exists(bdat):
                with open(bdat, 'r') as f:
                    self.bdat = json.load(f)
                self.has_behavior = True

    def map_face(self):
        self.face_path = self.path + '_face/'
        trace_file = self.face_path + '_motion_energy.npy'
        if os.path.exists(trace_file):
            self.mm_trace = numpy.load(trace_file)
        else:
            #open the video and make sure face is cropped.
            cam_file = self.si.info['cam_file']
            movie = CropVideo(cam_file, self.face_path)
            if movie.load_motion_crop() == False:
                movie.crop()
            #then call motion energy trace proceesing
            lprint(self, 'Pulling motion energy from cropped face field',)
            mm_trace = pull_motion_energy(cam_file, trace_file)
            self.mm_trace = mm_trace
        self.camtimes = self.ca.sync.load('cam')
        return self.camtimes, self.mm_trace

    def startstop(self, *args, **kwargs):
        cf = {'gap': int(10*self.fps), 'duration': int(5*self.fps), 'speed_threshold': 2, 'smoothing': int(self.fps)}
        #pycontrol treadmill needs longer smoothing parameter for gapless to bridge "stepping" of the mouse (1 s).
        for key, value in cf.items():
            if key not in kwargs:
                kwargs[key] = value
        return startstop(self.pos.speed, **kwargs)

    def get_preview(self):
        return self.rois.get_preview()

    def get_photons(self):
        fn = self.get_file_with_suffix('_PhotonTransfer.xlsx')
        if os.path.exists(fn):
            return read_excel(fn)

    def getparam(self, param):
        opn = param
        self.disc_param = False
        if type(param) == str:
            # if param == 'spikes':
            #     param = numpy.nan_to_num(self.ca.spikes * numpy.array(self.ca.event, dtype='bool'))
            #     self.disc_param = True
            if param == 'rel':
                param = self.ca.rel
            elif param == 'ntr':
                param = self.ca.ntr
            elif param == 'trace':
                param = self.ca.trace
            # elif param == 'ontr':
            #     param = self.ca.ontr
            elif param == 'smtr':
                param = self.ca.smtr
            elif param == 'nnd':
                if not hasattr(self.ca, 'nnd'):
                    self.ca.deconvolve(batch=1000, tau=0.75, fs=self.fps)
                param = self.ca.nnd

            # elif param == 'peaks':
            #     self.disc_param = True
            #     param = numpy.nan_to_num(self.ca.peaks.astype('bool'))
            # elif param == 'aupeaks':
            #     param = numpy.maximum(0, self.ca.smtr * self.ca.b2)
            # elif param == 'bin_onset':  # onset frame of 3 sd events
            #     param = numpy.zeros(self.ca.rel.shape, dtype='bool')
            #     for event in self.ca.eventarray:
            #         param[event[0], event[1]] = 1
            elif param == 'diff':
                param = numpy.empty(self.ca.rel.shape)
                param[:, 0] = numpy.nan
                param[:, 1:] = numpy.diff(self.ca.rel, axis=1)
            elif param == 'diff3':
                param = numpy.empty(self.ca.rel.shape)
                param[:, 0] = numpy.nan
                param[:, 1:] = numpy.diff(self.ewma_smooth(3, norm=False), axis=1)
            elif 'optomask' in param:
                #pass with a number at the end of the string to mask n frames after stim
                maskn=int(param[-1])+1
                stims = numpy.where(self.opto)[0]
                param=numpy.copy(self.ca.rel)
                for t in stims:
                    param[:, t:t+maskn] = numpy.nan
        elif type(param) == int:
            param = self.ewma_smooth(param)
        try:
            pm = numpy.nanmax(param)
            if not pm > 0:
                param = self.ca.ntr
                self.disc_param = False
                print('All values zero,', opn, ', continuing with ntr')
        except:
            param = self.ca.ntr
            self.disc_param = False
            print('Param not parsed,', opn, ', continuing with ntr')
        return param

    def timetoframe(self, t: float, TimeRef='relativeTime'):
        '''
        get the 2P imaging frame of a time point
        :param t: time (seconds)
        :param TimeRef: frame of reference, relativeTime is from first frame start (e.g. voltage recording),
         absoluteTime is from acquisition start
        :return: frame: int
        '''
        return numpy.searchsorted(self.ftimes[:, int(TimeRef != 'relativeTime')], t)

    def frametotime(self, f, TimeRef='relativeTime'):
        return self.ftimes[f, int(TimeRef != 'relativeTime')]

    def sampletoframe(self, s):
        return self.ephys.frames[s]

    def frametosample(self, f):
        return numpy.searchsorted(self.ephys.frames, f)

    def map_phys(self):
        self.ephys = Ephys(self.procpath, self.prefix, channel=self.lfp_ch)
        self.has_ephys = self.ephys.trace is not None

    def ewma_smooth(self, period):
        if not self.dualch:
            data = numpy.empty((self.ca.cells, self.ca.frames))
            for c in range(self.ca.cells):
                data[c, :] = ewma(self.ca.ntr[c], period)
        else:
            data = numpy.empty((self.ca.cells, self.ca.frames, 2))
            for ch in range(2):
                for c in range(self.ca.cells):
                    data[c, :, ch] = ewma(self.ca.ntr[c, :, ch], period)
        return data
    def map_spiketimes(self):
        '''
        Load existing spiketime excel files and convert to frames
        '''
        suffix = '_spiketimes.xlsx'
        stfield = 'SpikeTimes(s)'
        spiketag = self.kwargs.get('spiketag', None)
        n_channels = self.ephys.edat.shape[0] - 1
        fs = self.si.info['fs']
        self.spiketime_channels = []
        self.spiketimes = []
        for i in range(n_channels):
            if spiketag in (None, 'None'):
                fullsuffix = f'_Ch{i+1}{suffix}'
            else:
                fullsuffix = f'_{spiketag}_Ch{i+1}{suffix}'
            fname = self.get_file_with_suffix(fullsuffix)
            self.spiketime_channels.append(i + 1)
            if os.path.exists(fname):
                stdat = read_excel(fname)
                self.spiketimes.append(self.ephys.edat[0, (stdat[stfield].values * fs).astype('int64')])
            else:
                self.spiketimes.append(numpy.array([]))

    def map_seizuretimes(self):
        '''
        Load existing spiketime excel files and convert to frames
        '''
        stfield = ('Sz.Start(s)', 'Sz.Stop(s)')
        spiketag = self.kwargs.get('spiketag', None)
        n_channels = self.ephys.edat.shape[0] - 1
        fs = self.si.info['fs']
        for suffix, stfield, do_filter in zip(('_seizure_times_curated.xlsx', '_seizure_times.xlsx'),
                                   (('Start', 'Stop'), ('Sz.Start(s)', 'Sz.Stop(s)')), (True, False)):
            found_any = False
            self.sztime_channels = []
            self.sztimes = []
            for i in range(n_channels):
                if spiketag in (None, 'None'):
                    fullsuffix = f'_Ch{i+1}{suffix}'
                else:
                    fullsuffix = f'_{spiketag}_Ch{i+1}{suffix}'
                fname = self.get_file_with_suffix(fullsuffix)
                self.sztime_channels.append(i + 1)
                if os.path.exists(fname): #this is not supposed to be case sensitive on Windows (confirmed). if it fails to open,
                    # fix ch vs Ch in name.
                    found_any = True
                    stdat = read_excel(fname)
                    if do_filter:
                        stdat = stdat.loc[~stdat['Included'].eq(False)]
                    sztimes = [self.ephys.edat[0, (stdat[x].values * fs).astype('int64')] for x in stfield] #start and stop
                    self.sztimes.append(sztimes)
                else:
                    self.sztimes.append([[], []])
            if found_any:
                print(f'using seizures from {fname}')
                break


    def map_ripples(self):
        fs = self.si['fs']
        self.ripples = Ripples(self.procpath, self.prefix, tag=self.ripple_tag, strict_tag=True,
                               ephys_channel=self.lfp_ch, fs=fs)
        nframes = self.ca.frames
        frs = []
        for e in self.ripples.events:
            f = self.sampletoframe(e.p[0])
            if f < nframes and f not in frs:
                if not self.pos.movement[f]:
                    frs.append(f)
        frs.sort()
        self.ripple_frames = numpy.array(frs)
        #store power per frame in files (or load if already exists)

        keys = ('ripple_power', 'theta_power', 'hf_power')
        bands = ('ripple', 'theta', 'HF')
        power_getter = {'ripple': 'envelope', 'theta': 'theta', 'HF': None}
        path = self.ripples.path
        # later extend this to use multiple channels
        for band, key in zip(bands, keys):
            cfn = path + key + '.npy'
            if os.path.exists(cfn):
                setattr(self.ephys, key, numpy.load(path + key + '.npy'))
            else:
                lprint(self, f'Saving {band} power file ...')
                filt = Filter.Filter(self.ephys.trace, fs)
                frame_power = numpy.zeros(nframes)
                power_key = power_getter[band]
                if power_key is not None:
                    envelope = getattr(self.ripples, power_key)
                else:
                    locut, hicut = self.CF.bands[band]
                    ftr, envelope = filt.run_filt(filt.gen_filt(locut, hicut))
                t0, f = 0, 0
                for t1, s in enumerate(self.ephys.frames):
                    if s > f:
                        if f < nframes:
                            frame_power[f] = envelope[t0:t1].mean()
                            t0 = t1
                            f += 1
                setattr(self.ephys, key, frame_power)
                numpy.save(cfn, frame_power)
        lprint(self, f'power available in bands: ' + str(bands))

    def rippletrigger(self, param=None, ch=0):
        # plot averaged time course ripp triggered. using channel
        if param is None:
            param = self.getparam('ntr')
            if self.dualch:
                param = param[..., ch]
        length = int(self.fps)
        nzr = self.ripple_frames
        self.ripplerates = numpy.zeros((length, self.ca.cells))
        brmean = numpy.empty((len(nzr), length))
        for c in range(self.ca.cells):
            brmean[:] = numpy.nan
            for i, frame in enumerate(nzr):
                if (length < frame < (self.ca.frames - length)):
                    i0 = frame - int(length / 2)
                    y = param[c][i0:i0+length]
                    brmean[i, :] = y
            brflat = numpy.nanmean(brmean, axis=0)
            brflat -= brflat.min()
            brflat /= brflat.max()
            self.ripplerates[:, c] = brflat

    def pull_means(self, param, span, cells=None):
        # return mean of param in the bins of self.bin
        if cells is None:
            cells = range(self.ca.cells)
        bins = self.bin.shape[0]
        start, stop = span
        rates = numpy.empty((bins, len(cells)))
        for i in range(bins):
            for k, c in enumerate(cells):
                rates[i, k] = numpy.nanmean(param[c, start:stop] * self.bin[i, start:stop])
        zscores = numpy.empty((bins, len(cells)))
        for c in range(len(cells)):
            x = rates[:, c]
            zscores[:, c] = (x - numpy.nanmean(x)) / numpy.nanstd(x)
        self.rates[self.pltnum] = rates
        self.zscores[self.pltnum] = zscores
        return zscores

    def qc(self, trim=10, nnd=None):
        '''returns a bool mask for cells that pass qc
        trim: if >0, remove cells that are close to the top or bottom row to avoid artefacts
        '''

        qc_pass = numpy.any(numpy.nan_to_num(self.getparam('ntr')) > 3, axis=1)
        if hasattr(self.ca, 'nnd') and nnd is not None:
            nnd_pass = numpy.any(self.ca.nnd > nnd, axis=1)
            qc_pass = numpy.logical_and(nnd_pass, qc_pass)
        if trim:
            if not self.rois.image_loaded:
                self.rois.load_image()
            tops = [self.rois.polys[ci].min(axis=0)[1] for ci in range(self.ca.cells)]
            bottoms = [self.rois.polys[ci].max(axis=0)[1] for ci in range(self.ca.cells)]
            cutoff = self.rois.image.info['sz'][0] - trim
            incl = [tops[ci] > trim and bottoms[ci] < cutoff for ci in range(self.ca.cells)]
            qc_pass = numpy.logical_and(incl, qc_pass)
        return qc_pass

    def calc_MI(self, param='ntr', bins=11, selection='movement', shuffle=30, ):
        param = self.getparam(param)
        self.mi = numpy.zeros(self.ca.cells)
        self.mi_z = numpy.zeros(self.ca.cells)
        explicit_selection = False
        if type(selection) is str:
            if selection == 'all':
                wh = range(8, self.ca.frames)
            if selection == 'movement':
                wh = numpy.where(self.pos.movement)[0][8:-8]
            if selection == 'still':
                wh = numpy.where(self.pos.movement == 0)[0][8:-8]
        elif hasattr(selection, '__iter__'):
            wh = selection
            explicit_selection = True
        x = self.pos.relpos[wh]
        self.statistical_is_placecell = numpy.zeros(self.ca.cells, dtype='bool')
        for c in range(self.ca.cells):
            # skip nan cells if sel is movement
            if numpy.any(numpy.isnan(param[c])):
                if selection == 'movement' or explicit_selection:
                    self.mi[c] = numpy.nan
                    self.mi_z[c] = numpy.nan
                    continue
                else:
                    nwh = wh[numpy.where(numpy.logical_not(numpy.isnan(param[c][wh])))]
                    nx = self.pos.relpos[nwh]
            else:
                nwh = wh
                nx = x
            y = param[c, nwh]
            if numpy.count_nonzero(y) > bins:
                if any(y > 1):
                    y -= y.min()
                    y /= y.max()
                try:
                    c_xy = numpy.histogram2d(nx, y, bins)[0]
                    mis = mutual_info_score(None, None, contingency=c_xy)
                    shuff_mis = []
                    for _ in range(shuffle):
                        numpy.random.shuffle(y)
                        shuff_mis.append(mutual_info_score(None, None, contingency=numpy.histogram2d(y, nx, bins)[0]))
                    self.mi[c] = mis / numpy.mean(shuff_mis)
                    self.mi_z[c] = (mis - numpy.mean(shuff_mis)) / numpy.std(shuff_mis)
                    self.statistical_is_placecell[c] = mis > numpy.nanpercentile(shuff_mis, 95)
                except:
                    self.mi[c] = numpy.nan
                    self.mi_z[c] = numpy.nan
                    print('MI exception', self.prefix, c)

    def reldist(self, p1, p2):
        # distance of 2 relative belt positions
        a = min(p1, p2)
        b = max(p1, p2)
        d1 = b - a
        d2 = 1 - b + a
        return abs(min(d1, d2))

    def placefields_smooth(self, param='ntr', bins=50, silent=False, cmap='inferno', show=False, wh=None,
                           return_raw=False, exp_decay=2, corder=None, aspect='auto', strict=False, gui=False):
        spikes = self.getparam(param)
        intensity = self.getparam('rel')
        if wh is None:
            wh = numpy.where(self.pos.movement)[0][8:-8]
        binsize = 1.0 / bins
        self.placefield_properties = numpy.empty((self.ca.cells, 4))  # contrast and width (in rates and intensities)
        self.placefield_properties[:] = numpy.nan
        # create exponentially decaying weights for each frame based on distance from bin location
        # pull means for each cell and each bin using the exponential weights
        rates = numpy.empty((bins, self.ca.cells))
        intensity_map = numpy.empty((bins, self.ca.cells))  # non-smooth, for quantifying place field
        for bi in range(bins):
            weights = numpy.empty(len(wh))
            bin_mask = numpy.zeros(len(wh), dtype='bool')
            for wi, t in enumerate(wh):
                reldist = self.reldist(bi * binsize, self.pos.relpos[t])
                d = 1 - reldist
                bin_mask[wi] = reldist <= binsize
                if exp_decay is None:
                    weights[wi] = d
                else:
                    weights[wi] = numpy.e ** (1 - 1 / (d ** exp_decay))
            for c in range(self.ca.cells):
                whc = numpy.logical_not(numpy.isnan(spikes[c, wh]))
                rates[bi, c] = numpy.average(spikes[c, wh][whc], weights=weights[whc])
                intensity_map[bi, c] = numpy.nanmean(intensity[c, wh[bin_mask]])
        self.smooth_zscores = numpy.zeros((bins, self.ca.cells))
        self.placecell_intensity_map = intensity_map
        incl = []
        for c in range(self.ca.cells):
            x = rates[:, c]
            x_int = intensity_map[:, c]
            if x.max() > 0:
                self.smooth_zscores[:, c] = (x - numpy.nanmean(x)) / numpy.nanstd(x)
                incl.append(c)
                self.placefield_properties[c, 0] = numpy.nanmax(x) - numpy.nanmean(x)  # contrast in firing rates
                self.placefield_properties[c, 2] = numpy.nanmax(x_int) - numpy.nanmean(x_int)  # contrast in intensity (DF/F)
                pf_width = (self.smooth_zscores[:, c] > 1).sum() / bins  # number of >SD bins
                if pf_width == 0:
                    pf_width = numpy.nan
                self.placefield_properties[c, 1] = pf_width
                pfw_threshold = numpy.nanmean(x_int) + numpy.nanstd(x_int)
                self.placefield_properties[c, 3] = numpy.count_nonzero(x_int > pfw_threshold) / bins
        ni = 0
        self.corder_smoothpf = []
        if corder is None:
            self.smooth_sorted = numpy.empty((bins, len(incl)))
            for c in numpy.argsort(numpy.argmax(self.smooth_zscores, axis=0)):
                if c in incl:
                    if strict:
                        if self.placefield_properties[c, 0] < 0.2:
                            continue
                    self.smooth_sorted[:, ni] = self.smooth_zscores[:, c]
                    self.corder_smoothpf.append(c)
                    ni += 1
        else:
            self.smooth_sorted = numpy.empty((bins, len(corder)))
            for c in corder:
                self.smooth_sorted[:, ni] = self.smooth_zscores[:, c]
                self.corder_smoothpf.append(c)
                ni += 1
        if gui: #setting attributes for displayin each cells rates by lap
            nl = int(numpy.nanmax(self.pos.laps) + 1)
            # find time points for each bin
            self.bin = numpy.empty((bins, len(self.pos.relpos)), dtype='bool')
            binsize = 1.0 / bins
            for i in range(bins):
                self.bin[i] = (self.pos.relpos > i * binsize) * (self.pos.relpos < (i + 1) * binsize)
                self.bin[i] *= self.pos.movement  ##Specify here if selective for movement
            self.perlap_fields = numpy.zeros((bins, self.ca.cells, nl))
            for l in range(nl):
                lap = numpy.where(self.pos.laps == l)[0]
                if len(lap) > 1:
                    self.pull_means(param, (lap[0], lap[-1]))
                    self.perlap_fields[:, :, l] = self.rates[self.pltnum]
        if not silent:
            fig = plt.figure(self.pltnum)
            self.pltnum += 1
            plt.imshow(self.smooth_sorted.transpose(), cmap=cmap, aspect=aspect)
            if show:
                plt.show()
        if return_raw:
            return numpy.argmax(self.smooth_zscores, axis=0), rates
        elif not silent:
            return fig

    def running_placeseq(self, pf_calc_param='ntr', pf_calc_bins=25, display_param='smtr', cmap='plasma'):
        # identify place cells
        self.calc_MI(pf_calc_param)
        pc = []
        qc = self.qc()
        for c in range(self.ca.cells):
            if qc[c] and self.statistical_is_placecell[c]:
                pc.append(c)
        self.placefields_smooth(param=pf_calc_param, silent=True, bins=pf_calc_bins)
        pc_order = [c for c in self.corder_smoothpf if c in pc]
        pc_order = [c for c in pc_order if self.placefield_properties[c, 0] > 0.2]
        fig, ca = plt.subplots()
        param = self.getparam(display_param)
        ca.imshow(param[pc_order][:, self.pos.movement], aspect='auto', cmap=cmap)
        fig.set_size_inches(9, 6)
        ca.set_xlabel('Running time (frames)')
        ca.set_ylabel('Cell # (sorted by place field location)')
        fig.show()
        self.pc_order = pc_order
        return fig

    def export_spreadsheet(self):
        df = pandas.DataFrame(self.ca.rel.transpose(),
                              columns=[f"c{x}" for x in range(self.ca.cells)],
                              index=numpy.arange(self.ca.frames, dtype='int'))
        df['RelTime'] = self.ftimes[:, 0]
        # df['AbsTime'] = self.ftimes[:, 1]
        # treadmill
        df['Position'] = self.pos.pos
        df['Speed'] = self.pos.speed
        ststop = ['' for i in range(self.ca.frames)]
        for start, stop in zip(*self.startstop()):
            ststop[start] = 'start'
            ststop[stop] = 'stop'
        df['StartStop'] = ststop

        # ephys
        epc = -1
        for epc in range(len(self.si.info['lfp_channels'])):
            ephys = Ephys(self.procpath, self.prefix, channel=epc)
            if ephys.trace is not None:
                self.lfp_ch = epc + 1
                self.map_ripples()
                df[f'LFP{epc + 1}RipplePower'] = self.ephys.ripple_power
                df[f'LFP{epc + 1}HFPower'] = self.ephys.hf_power
        if self.opto is not None:
            ofn = self.get_file_with_suffix('_StimFrames.xlsx')
            if os.path.exists(ofn):
                sf = read_excel(ofn)
            stimpow = numpy.zeros(self.ca.frames)
            stimpow[sf['ImgFrame']] = sf['Intensity']
            df['PhotoStim'] = stimpow
        sfn = self.get_file_with_suffix(f'_ROI-{self.tag}_Ch{self.ca.ch}_{epc + 1}LFP.xlsx')
        df.to_excel(sfn)

    def behavior_plot(self):
        # draw pos, draw reward zones, correct licks
        mins = numpy.arange(0, len(self.pos.pos)) / self.fps / 60
        rz = self.bdat['RZ']
        licks = numpy.array(self.bdat['licks'])
        rewards = numpy.array(self.bdat['rewards'])
        has_rewards = numpy.zeros(len(self.pos.pos), dtype='bool')
        r_window = int(0.5*self.fps)
        rlim = len(self.pos.pos) - r_window
        for r in rewards:
            if 0 < r < rlim:
                has_rewards[r:r+r_window] = True

        self.fig, ax = plt.subplots(4, 1, figsize=(9, 6),
                                                           gridspec_kw={'height_ratios': [1, 0.5, 0.5, 0.5, ]},
                                                           sharex=True)
        for ca in ax[1:]:
            strip_ax(ca, full=False)
        (axp, axz, axr, axl,) = ax
        axp.scatter(mins, self.pos.pos, s=2, c=self.pos.movement, cmap='winter')
        axp.text(0, 0, f'{int(numpy.nan_to_num(self.pos.laps).max())} laps')

        for z in rz:
            axz.axvspan(*[x / self.fps / 60 for x in z], color='#e2efda', label='zones')
        axz.text(0, -0.5, f'{len(rz)} zones')
        axz.set_ylim(-0.6, 0.4)

        axr.scatter(rewards / self.fps / 60, numpy.zeros(len(rewards)), marker="|", s=50, label='drops')
        axr.text(0, -0.5, f'{len(rewards)} drops')
        axr.set_ylim(-0.6, 0.4)

        try:
            licks = licks[numpy.where(numpy.nan_to_num(licks) > 0)]
            licks = licks[numpy.where(licks < self.ca.frames)]
        except:
            print(licks)
            assert False
        if len(licks):
            axl.scatter(licks / self.fps / 60, numpy.zeros(len(licks)), marker="|", c=has_rewards[licks], s=50,
                        cmap='bwr_r', label='licks')
            axl.text(0, -0.5, f'{len(licks)} / {numpy.count_nonzero(has_rewards[licks])} licks')
        else:
            axl.text(0, -0.5, f'{0} licks')
        axl.set_ylim(-0.6, 0.4)

        axl.set_xlabel('Minutes')
        for ca in ax[1:]:
            ca.yaxis.set_ticklabels([])
        axp.set_ylabel('Position')
        axz.set_ylabel('Zones')
        axr.set_ylabel('Drops')
        axl.set_ylabel('Licks')
        self.fig.suptitle(self.prefix)
        self.fig.savefig(self.get_file_with_suffix('_behaviorplot.png'), dpi=300)
        self.fig.clf()

if __name__ == '__main__':
    procpath = 'D:\Shares\Data\_Processed/2P\PVTot/'
    prefix = 'PVtot9_2024-03-07_RF_224'
    tag = 'skip'
    a = ImagingSession(procpath, prefix, tag=tag)
    a.behavior_plot()