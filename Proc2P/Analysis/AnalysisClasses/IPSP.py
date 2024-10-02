import os.path

import matplotlib.pyplot as plt

from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Analysis.AnalysisClasses import PhotoStim

from Proc2P.utils import logger, lprint

from collections import namedtuple
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

import pandas
import numpy



class IPSP:
    __name__ = 'IPSP'
    '''Detect and fit optically evoked IPSPs in a voltage imaging session'''

    def __init__(self, session: ImagingSession, config=None):
        '''initialize with an ImagingSession'''
        self.session = session
        self.wdir = self.session.path + 'IPSP/'
        self.n_cells = self.session.ca.cells
        if self.is_bg():
            self.n_cells -= 1
        self.set_defaults(config)
        self.log = logger()
        self.log.set_handle(self.session.procpath, self.session.prefix)

    def set_defaults(self, config):
        if not os.path.exists(self.wdir):
            os.mkdir(self.wdir)
        self.get_stimtimes()
        Config = namedtuple('Config', 'pre, post, param')
        pre = int(self.session.fps * 100/1000)  # duration included before stim (frames)
        post = int(self.session.fps * 200/1000)  # duration included after stim (frames)
        param = 'rel'
        self.config = Config(pre, post, param)
        if config is not None:
            for key, value in config.items():
                self.config._replace(key=value)

    def get_matrix(self):
        # pull the individual responses for each cell and stim
        if not hasattr(self, 'raw_resps'):
            mname = self.get_fn('raw_resps.npy')
            if os.path.exists(mname):
                self.raw_resps = numpy.load(mname)
            else:
                mlen = self.config.pre+self.config.post
                Y = numpy.empty((self.n_stims, self.n_cells, mlen))
                Y[:] = numpy.nan
                #generate the pull mask
                mask = numpy.empty((self.n_stims, mlen))
                mask[:] = numpy.nan
                trim = max(int(self.session.fps), self.config.pre)
                maxframe = self.session.ca.frames - max(trim, self.config.post)
                for ri, current_frame in enumerate(self.stimframes):
                    if current_frame < trim or current_frame > maxframe:
                        continue
                    mask[ri] = numpy.arange(current_frame - self.config.pre, current_frame + self.config.post)

                #set the resp values
                param = self.session.getparam(self.config.param)[:self.n_cells]
                for ei, indices in enumerate(mask):
                    loc = numpy.where(numpy.logical_not(numpy.isnan(indices)))[0]
                    if len(loc):
                        Y[ei] = param[:, indices[loc].astype(numpy.int64)]
                numpy.save(mname, Y)
                self.raw_resps = Y

        return self.raw_resps

    def get_stimtimes(self):
        '''
        Identify onsets of photostimulations. This is shared between different tags of the same session
        :return:
        '''
        if not hasattr(self, 'stimframes'):
            stimname = self.wdir + '_StimTimes'
            if os.path.exists(stimname + '.npy'):
                stimframes = numpy.load(stimname + '.npy')
            else:
                t = PhotoStim.PhotoStim(self.session)
                stimframes = t.get_trains(isi=3)[
                    0]  # ignoring intensities. these are not reliable with current PreProc.
                numpy.save(stimname + '.npy', stimframes)
                pandas.DataFrame({'StimFrames': stimframes}).to_excel(stimname + '.xlsx')
            self.stimframes = stimframes
        self.n_stims = len(self.stimframes)
        return self.stimframes

    def is_bg(self):
        '''check this when assuming traces are background subtracted'''
        return self.session.ca.version_info['bg_corr']

    def get_waveform(self):
        '''the average IPSP from all stims and all cells'''
        if not hasattr(self, 'mean'):
            Y = self.get_matrix()
            self.len = Y.shape[-1]
            #identify nan samples (stim artefact)
            self.wh_nan = numpy.where(numpy.average(numpy.isnan(Y), axis=(0, 1))>0.6)
            self.mean = numpy.nanmean(Y, axis=(0, 1))
            self.mean[self.wh_nan] = numpy.nan
        return self.mean

    def fit_model(self, save=False):
        Y = self.get_waveform()[self.config.pre:]
        notna = numpy.ones(len(Y), dtype='bool')
        for x in self.wh_nan[0]:
            if x >= self.config.pre:
                notna[x-self.config.pre] = False
        self.notna = notna
        X = 1000 * numpy.arange(self.config.post) / self.session.fps
        fitX = X[notna]
        fitY = Y[notna]
        bounds = ([0.01, 3, -1, 0], [1, 100, 1, 50])
        guesses = (0.1, 25, 0, 0)
        maxval = numpy.nanmax(Y)
        # invert it for fitting
        popt, pcov = curve_fit(self.alpha_func, fitX, maxval - fitY, p0=guesses, bounds=bounds)
        self.model = popt
        modstring = self.get_modstring()
        if save:
            fname = self.get_fn('model')
            fig, ax = plt.subplots()
            ax.plot(fitX, 100*(maxval - self.alpha_func(fitX, *popt)), color='red')
            ax.scatter(X, Y*100)
            fig.suptitle(self.session.prefix + ' ' + self.session.tag + '\n' + modstring)
            with open(fname+'.txt', 'w') as f:
                f.write(modstring+'\n'+str(popt))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Response (DF/F %)')
            plt.savefig(fname+'.png')
            numpy.save(fname+'.npy', self.model)
            self.fig = fig, ax
        self.X = X #times in ms of the post
        return self.model

    def get_modstring(self):
        popt = self.model
        return f'ampl:{popt[0] * 100:.2f}%, peak:{popt[1]:.1f} ms, bias:{popt[2] * 100:.1f}%, delay:{popt[3]:.1f}ms'

    def get_fn(self, suffix):
        return self.wdir + f'_{self.session.tag}_{suffix}'

    def alpha_func(self, x, A, B, C, D):
        '''
        generate the time-dependent amplitude using alpha function
        Source: Guzman SJ, Schlögl A, Schmidt-Hieber C (2014)
        Stimfit: quantifying electrophysiological data with Python. Front Neuroinform doi: 10.3389/fninf.2014.00016
        But a delay parameter was added.
        use non-negative, positive peak (inverted for IPSP)
        scale, shape needs to be > 0, start with guess 1,1,0
        when input X is in ms and Y in DF/F (inverted but not rescaled), A can be interpreted as peak amplitude,
        B can be interpreted as peak time, D as onset delay in ms.
        :param x: time (ms)
        :param opts: scale, shape, shift, delay
        :return: array (predicted amplitude for each time point)
        '''
        return A * (x - D) / B * numpy.exp(1 - (x - D) / B) + C

    def ampl_func(self, x, A, C):
        '''
        generate the alpha using the shape of the IPSP model (see gen_alpha)
        '''
        B = self.model[1]
        D = self.model[3]
        return A * (x - D) / B * numpy.exp(1 - (x - D) / B) + C

    def fit_event(self, c, event_index):
        '''
        fit a single trial trace to the IPSP model
        :param c:
        :param event_index:
        :return: amplitude and fit
        '''
        Y = self.raw_resps[event_index, c][self.config.pre:]
        bl = gaussian_filter(self.raw_resps[event_index, c][:self.config.pre], int(0.025 * self.session.fps))
        fit = self.fit_response(Y, bl[-1])
        return Y, bl, fit


    def fit_response(self, Y, bl):
        '''
        fit a trace with the IPSP model, constrained to the peak time and delay of the model
        :param Y: same len as self.X
        :param bl: baseline of the response
        :return: opts of the fit
        '''
        bounds = ([0.01, -1], [1, 1])
        guesses = (self.model[0], self.model[2])
        x = self.X[self.notna]
        y = (bl - Y)[self.notna]
        incl = numpy.logical_not(numpy.isnan(x)+numpy.isnan(y))
        if numpy.count_nonzero(incl) < 10:
            return None
        popt, pcov = curve_fit(self.ampl_func, x[incl], y[incl], p0=guesses, bounds=bounds)
        return popt

    def pull_ipsps(self):
        '''
        store baseline and amplitude for each stim in each cell
        using the parameter of the fit as amplitude. NB that alternatively, we could add the bias,
         or compute the diff of where the fit curve min is relative to baseline.
         :return: array of frame, baseline, response for each c, e
        '''
        self.responses = numpy.empty((self.n_cells, self.n_stims, 3)) #frame, baseline, response
        self.responses[:] = numpy.nan
        for ci in range(self.n_cells):
            for ei in range(self.n_stims):
                Y, bl, fit = self.fit_event(ci, ei) #bl is in DF/F actual, response (fit[0]) in DF/F change.
                if fit is None:
                    self.responses[ci, ei] = self.stimframes[ei], bl[-1], 0
                else:
                    self.responses[ci, ei] = self.stimframes[ei], bl[-1], fit[0]

        lprint(self, 'Pulled responses for', self.session.tag, 'with ', self.config, 'Model:',
               self.get_modstring(), logger=self.log)
        fn = self.get_fn('_ipsps.npy')
        numpy.save(fn, self.responses)
        return self.responses

    def get_ipsps(self):
        if not hasattr(self, 'responses'):
            fn = self.get_fn('_ipsps.npy')
            if os.path.exists(fn):
                self.responses = numpy.load(fn)
                self.model = numpy.load(self.get_fn('model.npy'))
                lprint(self, 'Loaded IPSP model from file:', self.get_modstring())
            else:
                self.fit_model(save=True)
                plt.close()
                self.pull_ipsps()
        return self.responses

