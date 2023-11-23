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
from Proc2P.utils import outlier_indices, gapless, startstop, lprint
# from Ripples import Ripples
import time
from scipy import stats

class Pos:
    '''to hold attributes of movement, previously used by treadmill (Quad) class'''
    def __init__(self, sync):
        # self.sync = sync
        self.speed = sync.load('speed')
        self.pos = sync.load('pos')
        self.movement = gapless(self.speed, threshold=0.05)
        # self.laps = #implement this

class ImagingSession(object):
    __name__ = 'ImagingSession'
    def __init__(self, procpath, prefix, tag=None, norip=False, opath='.', ch=0,
                 ripple_tag=None,):
        self.opath = opath
        self.prefix = prefix
        self.procpath = procpath
        self.tag = tag
        self.path = os.path.join(procpath, prefix+'/')

        #load info from json file
        self.si = SessionInfo(self.path, prefix)
        self.si.load()
        self.CF = CF()
        self.CF.fps = self.si['framerate']

        #load polygons, option to access image data
        self.rois = FrameDisplay(procpath, prefix, tag=tag, lazy=True)

        #load processed tracces
        self.ca = CaTrace(procpath, prefix, tag=tag, ch=ch)
        self.ca.load()
        self.ftimes = numpy.load(self.get_file_with_suffix('_FrameTimes.npy'))

        # load photostimulation if available:
        self.map_opto()

        #map ephys
        self.map_phys()
        if (not norip) and self.has_ephys:
            self.ripple_tag = ripple_tag
            self.map_ripples()

        #map video
        self.eye = None #implement later
        self.has_eye = False


        if tag == 'skip':
            self.ca.frames = self.si['n_frames']
        self.dualch = self.ca.is_dual
        self.has_behavior = False
        # here, implement reading the behavior events from treamill.
        # self.bdat = BehaviorSession(prefix + '.tdml', silent=True)
        self.pos = Pos(self.ca.sync)

        self.pltnum = 0
        self.zscores = {}
        self.rates = {}
        self.colors = {}

        self.preview = None


        # implement reading the intensities in separate function

    def get_file_with_suffix(self, suffix):
        return os.path.join(self.path, self.prefix+suffix)

    def map_opto(self):
        if self.si['has_opto']:
            opto_frames = self.ca.sync.load('opto')
            self.opto = numpy.zeros(self.ca.frames, dtype='bool')
            self.opto[opto_frames] = 1
        else:
            self.opto = None

    def startstop(self, *args, **kwargs):
        return startstop(self.pos.speed, **kwargs)

    def get_preview(self):
        return self.rois.get_preview()

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
            # elif param == 'ontr':
            #     param = self.ca.ontr
            elif param == 'smtr':
                param = self.ca.smtr
            elif param == 'nnd':
                if hasattr(self.ca, 'nnd'):
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


    def timetoframe(self, t:float, TimeRef='relativeTime'):
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
        self.ephys = Ephys(self.procpath, self.prefix, channel=None)
        self.has_ephys = self.ephys.trace is not None

    def map_ripples(self):
        # self.ripples = Ripples(self.prefix, tag=self.ripple_tag, strict_tag=True, )
        # store power per frame in files (or load if already exists)
        nframes = self.ca.frames
        fs = self.si['fs']
        keys = ('ripple_power', 'theta_power', 'hf_power')
        bands = ('ripple', 'theta', 'HF')
        path = os.path.join(self.path, 'ephys/')
        if not os.path.exists(path):
            os.mkdir(path)
        cfn = path + keys[0] + '.npy'
        #later extend this to use multiple channels
        if os.path.exists(cfn):
            for key in keys:
                setattr(self.ephys, key, numpy.load(path + key + '.npy'))
        else:
            lprint(self, 'Creating ripple power files...')
            filt = Filter.Filter(self.ephys.edat[1], fs)
            for band, key in zip(bands, keys):
                power = numpy.zeros(nframes)
                locut, hicut = self.CF.bands[band]
                ftr, envelope = filt.run_filt(filt.gen_filt(locut, hicut))
                numpy.save(path + key + '_filtered.npy', numpy.array([ftr, envelope]))
                t0, f = 0, 0
                for t1, s in enumerate(self.ephys.frames):
                    if s > f:
                        if f < nframes:
                            power[f] = envelope[t0:t1].mean()
                            t0 = t1
                            f += 1
                setattr(self.ephys, key, power)
                numpy.save(path + key + '.npy', power)
            lprint(self, 'power measured in bands: ' + str(bands))

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

    def qc(self, trim=10):
        '''returns a bool mask for cells that pass qc
        trim: if >0, remove cells that are close to the top or bottom row to avoid artefacts
        '''

        qc_pass = numpy.any(numpy.nan_to_num(self.getparam('ntr')) > 3, axis=1)
        if trim:
            tops = [self.rois.polys[ci].min(axis=0)[1] for ci in range(self.ca.cells)]
            bottoms = [self.rois.polys[ci].max(axis=0)[1] for ci in range(self.ca.cells)]
            cutoff = self.rois.image.info['sz'][0] - trim
            incl = [tops[ci] > trim and bottoms[ci] < cutoff for ci in range(self.ca.cells)]
            qc_pass = numpy.logical_and(incl, qc_pass)
        return qc_pass



if __name__ == '__main__':
    path = 'D:/Shares/Data/_Processed/2P/PVTot/LFP/'
    prefix = 'PVTot3_2023-09-15_LFP_028'
    tag = 'ALL'
    a = ImagingSession(path, prefix, tag=tag, ch=0)
    e = Ephys(a.procpath, a.prefix)
    trace = e.edat[1]
    r = Filter.Filter(trace, 2000)
    a.map_ripples()


    # fig, ax = plt.subplots(nrows=3, sharex=True)
    # ax[0].imshow(a.ca.smtr, aspect='auto')
    # ax[0].set_ylabel('DF/F')
    # ax[1].plot(a.pos.speed)
    # ax[1].set_ylabel('Speed (cm/s)')
    # ax[2].plot(a.opto)
    # ax[2].set_ylabel('Opto stim')