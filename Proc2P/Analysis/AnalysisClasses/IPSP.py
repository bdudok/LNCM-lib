import copy
import json
import os.path

import matplotlib.pyplot as plt

from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Analysis.AnalysisClasses import PhotoStim

from Proc2P.utils import logger, lprint

from collections import namedtuple
import scipy
from scipy.optimize import curve_fit
from statsmodels.stats.weightstats import DescrStatsW

import pandas
import numpy


class IPSP:
    __name__ = 'IPSP'
    '''Detect and fit optically evoked IPSPs in a voltage imaging session'''

    def __init__(self, session: ImagingSession, config=None, purge=False, shuffle=False):
        '''initialize with an ImagingSession
        :param config: a dict with optional keys 'pre, post, param, nan'
        :param purge: if True, will delete cached results
        :param shuffle: if True, replace actual stim frames with random frames (picked between the firs and last stim,
         not including post-stim periods). Needs to be set at init. does not overwrite stored stimtimes.
        '''
        self.session = session
        self.wdir = self.session.path + 'IPSP/'
        self.n_cells = self.session.ca.cells
        if self.session.ca.last_bg:
            self.n_cells -= 1
        if purge:
            self.purge()
        self.set_defaults(config, shuffle)
        self.log = logger()
        self.log.set_handle(self.session.procpath, self.session.prefix)
        self.is_saved = os.path.exists(self.get_fn('model.npy'))

    def purge(self):
        '''delete previously saved contents of the IPSP folder'''
        if not os.path.exists(self.wdir):
            return 0
        for f in os.listdir(self.wdir):
            os.remove(self.wdir + f)

    def set_defaults(self, user_config, shuffle=False):
        default_config = {
            'pre': int(self.session.fps * 100 / 1000),  # duration included before stim (frames)
            'post': int(self.session.fps * 200 / 1000),  # duration included after stim (frames)
            'param': 'rel',  # key of param for ImagingSession to use for traces ('rel')
            'nan': round(self.session.fps * 20 / 1000),  # duration excluded after stim (frames)
            'order': 1  # if 1, a single function if fitted, if 2, a second, positive function is added,
            # if 7: a 2-component fit is done (7 parameters).
        }
        self.constraints = {
            # these are important because they constrain the possible waveform fit.
            # for example, with the default, the IPSP peak cannot occur after 50 ms
            'bounds': ([0.01, 3, -1, 0], [1, 100, 1, 50]),
            'guesses': (0.1, 50, 0, 0),
            'bounds.second': ([0.001, 0.1, -1, 0], [1, 200, 1, 200]),
            'guesses.second': (0.1, 100, 0, 0),
            'bounds.dual': ([-1, 0.1, 0], [0.001, 300, 200]),
            'guesses.dual': (-0.1, 100, 25),
        }
        self.shuffle=shuffle
        self.baseline_kernels = {}
        if not os.path.exists(self.wdir):
            os.mkdir(self.wdir)
        if user_config is None:
            config = {}
        else:
            config = user_config
        assert type(config) is dict
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        self.set_config(config)
        with open(self.get_fn('config.json'), 'w') as f:
            f.write(json.dumps(config))
        self.get_stimtimes()

    def set_config(self, config):
        Config = namedtuple('Config', 'pre, post, param, nan, order')
        config = Config(config['pre'], config['post'], config['param'], config['nan'], config['order'])
        # json serialization order not reliable
        h = str(config.pre) + str(config.post) + str(config.param) + str(config.nan)
        if config.order != 1:
            h += 'o' + str(config.order)
        if self.shuffle:
            h+= '_shuffle'
        self.confighash = h
        self.config = config

    def get_matrix(self):
        # pull the individual responses for each cell and stim
        if not hasattr(self, 'raw_resps'):
            mname = self.get_fn('raw_resps.npy')
            if os.path.exists(mname):
                self.raw_resps = numpy.load(mname)
            else:
                mlen = self.config.pre + self.config.post
                Y = numpy.empty((self.n_stims, self.n_cells, mlen))
                Y[:] = numpy.nan

                # generate the pull mask
                mask = numpy.empty((self.n_stims, mlen))
                mask[:] = numpy.nan
                trim = max(int(self.session.fps), self.config.pre)
                maxframe = self.session.ca.frames - max(trim, self.config.post)
                for ri, current_frame in enumerate(self.stimframes):
                    if (current_frame < trim) or (current_frame > maxframe):
                        continue
                    mask[ri] = numpy.arange(current_frame - self.config.pre, current_frame + self.config.post)

                # set the resp values
                param = self.session.getparam(self.config.param)
                for ei, indices in enumerate(mask):
                    loc = numpy.where(numpy.logical_not(numpy.isnan(indices)))[0]
                    if len(loc):
                        res_view = Y[ei, :]  # need to index in 2 steps otherwise
                        res_view[:, loc] = param[:, indices[loc].astype(numpy.int64)]

                # mask the after-stim frames with nan
                Y[:, :, self.config.pre:self.config.pre + self.config.nan] = numpy.nan

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
            if self.shuffle:
                frame_pool = numpy.ones(stimframes[-1], dtype='bool')
                frame_pool[:stimframes[0]] = 0
                for i in stimframes[:-1]:
                    frame_pool[i:i+self.config.post] = 0
                stimframes = numpy.random.choice(numpy.where(frame_pool)[0], len(stimframes), replace=False)
                stimframes.sort()
            self.stimframes = stimframes
        self.n_stims = len(self.stimframes)
        return self.stimframes

    def shuffle_stimtimes(self):
        '''Call after initializing and fitting the average waveforms to work with events at random times
         instead of photostimulation times. follow this by calling pull_ipsps to fit the random responses with the
         '''
        stimframes = self.get_stimtimes()
        frame_pool = numpy.ones(stimframes[-1], dtype='bool')
        frame_pool[:stimframes[0]] = 0
        for i in stimframes[:-1]:
            frame_pool[i:i + self.config.post] = 0
        stimframes = numpy.random.choice(numpy.where(frame_pool)[0], len(stimframes), replace=False)
        stimframes.sort()
        self.stimframes = stimframes
        self.confighash += '_shufflev2'
        del self.raw_resps
        self.get_matrix()

    def get_waveform(self, cells=None):
        '''the average IPSP from all stims and all cells'''
        if not hasattr(self, 'mean'):
            Y = self.get_matrix()
            if cells is not None:
                Y = Y[:, cells, :]
            self.len = Y.shape[-1]
            # identify nan samples (stim artefact)
            self.wh_nan = numpy.where(numpy.average(numpy.isnan(Y), axis=(0, 1)) > 0.2)
            self.mean = numpy.nanmean(Y, axis=(0, 1))
            self.mean[self.wh_nan] = numpy.nan
        return self.mean

    def fit_cell(self, ci):
        '''
        Do the fit separately on one cell (instead on the average)
        :param c: index
        :return: model parameters
        '''
        M = self.get_matrix()
        self.wh_nan = numpy.where(numpy.average(numpy.isnan(M[:, ci, :]), axis=0) > 0.2)
        Y = numpy.nanmean(M[:, ci, self.config.pre:], axis=0)
        notna = numpy.ones(len(Y), dtype='bool')
        for x in self.wh_nan[0]:
            if x >= self.config.pre:
                notna[x - self.config.pre] = False
        self.notna = notna
        X = 1000 * numpy.arange(self.config.post) / self.session.fps
        fitX = X[notna]
        fitY = Y[notna]
        maxval = numpy.nanmax(Y)
        # invert it for fitting
        guesses = self.constraints['guesses']
        bounds = self.constraints['bounds']
        try:
            popt, pcov = curve_fit(self.alpha_func, fitX, maxval - fitY, p0=guesses, bounds=bounds)
        except RuntimeError:
            print(f'Optimization failed for {self.session.prefix} c{ci}')
            popt = [numpy.nan, numpy.nan, 0, 0]
        return popt

    def fit_model(self, save=False, include_cells=None, waveform=None):
        '''
        Fit an alpha function to the average waveform across cells and events
                # if loading a previously saved model, also call get_ipsps after this
        :param save: plot the resulting fit
        :param include_cells: list of indices to include
        :return: None, saves files and sets attributes.
        '''
        if self.config.order == 1:
            self.fit_model_single_order(include_cells=include_cells, waveform=waveform)
            modstring = self.get_modstring()
            fitX = self.X[self.notna]
            plot_Y = self.Y_fit
        elif self.config.order > 1:
            first_model = self.fit_model_single_order(include_cells=include_cells, waveform=waveform)
            fitX = self.X[self.notna]
            fitY = self.Y[self.notna]
            if self.config.order == 2:
                residual = fitY - self.Y_fit
                minval = numpy.nanmin(residual)
                guesses = self.constraints['guesses.second']
                bounds = self.constraints['bounds.second']
                popt, pcov = curve_fit(self.alpha_func, fitX, residual - minval, p0=guesses, bounds=bounds)
                second_fit = self.alpha_func(fitX, *popt) + minval
                self.Y_fit_all = self.Y_fit + second_fit
            elif self.config.order == 7:
                maxval = numpy.nanmax(fitY)
                # constrain the bounds based on a first fit to the negative peak
                #populate with defaults
                guesses = list(copy.copy(self.model))
                bounds = [copy.copy(x) for x in self.constraints['bounds']]
                guesses.extend(self.constraints['guesses.dual'])
                for b_i in (0, 1):
                    bounds[b_i].extend(self.constraints['bounds.dual'][b_i])
                #negative amplitude will be +- 50% from the first fit
                bounds[0][0] = self.model[0] * 0.5
                bounds[1][0] = min(self.constraints['bounds'][1][0], self.model[0] * 1.5)
                #negative peak will be +- 5 ms from the first fit
                bounds[0][1] = self.model[1] - 5
                bounds[1][1] = self.model[1] + 5
                #bias will be no more than the first model
                bounds[1][2] = self.model[2] + 5
                #delay of first peak will be +- 50% of the first fit
                bounds[0][3] = max(self.constraints['bounds'][0][3], self.model[3] * 0.5)
                bounds[1][3] = min(self.constraints['bounds'][1][3], self.model[3] * 1.5)
                #positive peak will be after the negative
                bounds[0][5] = max(self.constraints['bounds.dual'][0][1], self.model[1] + 5)
                #delay of second component will be after the first peak
                bounds[0][6] = max(self.constraints['bounds.dual'][0][2], self.model[1])
                guesses = self.constrain_guesses(guesses, bounds)
                popt, pcov = curve_fit(self.dual_alpha_func, fitX, maxval - fitY, p0=guesses, bounds=bounds)
                second_fit = maxval - self.dual_alpha_func(fitX, *popt)
                self.Y_fit_all = second_fit
            self.model_2 = popt
            self.Y_fit_2 = second_fit
            plot_Y = self.Y_fit_all
            modstring = self.get_modstring(popt)
        if save:
            fname = self.get_fn('model')
            fig, ax = plt.subplots()
            ax.scatter(self.X, self.Y)
            ax.plot(fitX, plot_Y, color='red')
            figure_title = self.session.prefix + ' ' + self.session.tag + '\n' + modstring
            if self.config.order == 2:
                ax.plot(fitX, self.Y_fit, color='red', linestyle=':')
                ax.plot(fitX, self.Y_fit_2, color='green', linestyle=':')
                figure_title += ('\n' + self.get_modstring(self.model_2))
            elif self.config.order == 7:
                ax.plot(fitX, self.Y_fit, color='red', linestyle=':')
            fig.suptitle(figure_title)
            with open(fname + '.txt', 'w') as f:
                f.write(modstring + '\n' + str(self.model))
                if self.config.order == 2:
                    f.write('\n' + self.get_modstring(self.model_2))
                    f.write('\n' + str(self.model_2))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Response (DF/F)')
            plt.savefig(fname + '.png')
            # numpy.save(fname + '.npy', self.model) #this is never used and can be parsed from json in the text file
            plt.tight_layout()
            self.fig = fig, ax

    def constrain_guesses(self, guesses, bounds):
        ret_guesses = []
        for i, x in enumerate(guesses):
            ret_guesses.append(max(min(x, bounds[1][i]), bounds[0][i]))
        return ret_guesses

    def fit_model_single_order(self, include_cells=None, waveform=None):
        if waveform is not None:
            Y = waveform
        else:
            Y = self.get_waveform(cells=include_cells)[self.config.pre:]
        notna = numpy.ones(len(Y), dtype='bool')
        for x in self.wh_nan[0]:
            if x >= self.config.pre:
                notna[x - self.config.pre] = False
        self.notna = notna
        X = 1000 * numpy.arange(self.config.post) / self.session.fps
        fitX = X[notna]
        fitY = Y[notna]
        guesses = self.constraints['guesses']
        bounds = self.constraints['bounds']
        maxval = numpy.nanmax(Y)
        # invert it for fitting
        popt, pcov = curve_fit(self.alpha_func, fitX, maxval - fitY, p0=guesses, bounds=bounds)
        self.model = popt
        self.Y_fit = maxval - self.alpha_func(fitX, *popt)
        self.X = X  # times in ms of the post
        self.Y = Y

    def get_modstring(self, popt=None):
        if popt is None:
            popt = self.model
        if len(popt) > 4:
            return f'ampl1:{popt[0] * 100:.2f}%, peakt1:{popt[1]:.1f} ms, delay1:{popt[3]:.1f}ms, ampl2:{popt[4] * 100:.2f}%, peakt2:{popt[5]:.1f} ms'
        return f'ampl:{popt[0] * 100:.2f}%, peak:{popt[1]:.1f} ms, bias:{popt[2] * 100:.1f}%, delay:{popt[3]:.1f}ms'

    def get_fn(self, suffix):
        return self.wdir + f'_{self.session.tag}_{self.confighash}_{suffix}'

    def alpha_func(self, x, A, B, C, D):
        '''
        generate the time-dependent amplitude using alpha function
        Source: Guzman SJ, Schlögl A, Schmidt-Hieber C (2014)
        Stimfit: quantifying electrophysiological data with Python. Front Neuroinform doi: 10.3389/fninf.2014.00016
        But a delay parameter was added.
        use non-negative trace, positive peak (inverted for IPSP)
        scale, shape needs to be > 0, start with guess 1,1,0
        when input X is in ms and Y in DF/F (inverted but not rescaled), A can be interpreted as peak amplitude,
        B can be interpreted as peak time, D as onset delay in ms.
        :param x: time (ms)
        :param opts: scale, shape, shift, delay
        :return: array (predicted amplitude for each time point)
        '''
        return A * (x - D) / B * numpy.exp(1 - (x - D) / B) + C

    def dual_alpha_func(self, x, A1, B1, C1, D1, A2, B2, D2):
        y1 = A1 * (x - D1) / B1 * numpy.exp(1 - (x - D1) / B1) + C1
        #we clip the (inverted) EPSP so it's not negative, otherwise constraining the initial response is very fragile
        y2 = numpy.clip(A2 * (x - D2) / B2 * numpy.exp(1 - (x - D2) / B2), max=0)
        return y1 + y2

    # def dual_fit(self, include_cells=None):
            #now included in fit_model
    #     '''Try optimizing both functions in 1 shot (7 df)'''
    #     first_model = self.fit_model_single_order(include_cells=include_cells)
    #     fitX = self.X[self.notna]
    #     fitY = self.Y[self.notna]
    #     guesses = list(self.model)
    #     bounds = self.constraints['bounds']
    #     for add_i in (0, 1, 3):
    #         guesses.append(self.constraints['guesses.second'][add_i])
    #         for b_i in (0, 1):
    #             bounds[b_i].append(self.constraints['bounds.second'][b_i][add_i])
    #     maxval = numpy.nanmax(fitY)
    #     popt, pcov = curve_fit(self.dual_alpha_func, fitX, maxval - fitY, p0=guesses, bounds=bounds)
    #     second_fit = maxval - self.dual_alpha_func(fitX, *popt)
    #     self.model_2 = popt
    #     self.Y_fit_2 = second_fit
    #     plot_Y = self.Y_fit_2
    #     plt.plot(fitX, self.Y_fit, color='red', linestyle=':')
    #     plt.plot(fitX, self.Y_fit_2, color='orange')



    def ampl_func(self, x, A, C):
        '''
        fit the alpha using the shape of the IPSP model (see gen_alpha)
        '''
        B = self.model[1]
        D = self.model[3]
        return self.alpha_func(x, A, B, C, D)

    def ampl_func_2(self, x, A, C):
        B = self.model_2[1]
        D = self.model_2[3]
        return self.alpha_func(x, A, B, C, D)

    def ampl_func_7(self, x, A1, C, A2):
        B1 = self.model[1]
        D1 = self.model[3]
        B2 = self.model_2[1]
        D2 = self.model_2[3]
        return self.dual_alpha_func(x, A1, B1, C, D1, A2, B2, D2)

    def fit_event(self, c, event_index, order=1):
        '''
        fit a single trial trace to the IPSP model
        :param c:
        :param event_index:
        :return: the response trace (Y), the baseline, and the fit
        The baseline is determined as the mean pre-stim trace with backwards gaussian weight decay
        '''
        Y = self.raw_resps[event_index, c][self.config.pre:]
        bl = self.raw_resps[event_index, c][:self.config.pre]
        notna_bl = numpy.logical_not(numpy.isnan(bl))
        if (numpy.count_nonzero(numpy.logical_not(numpy.isnan(Y))) < 10) or (numpy.count_nonzero(notna_bl) < 10):
            return Y, numpy.nanmean(bl), None
        weights = self.get_baseline_kernel()
        bl_mean = DescrStatsW(bl[notna_bl], weights[notna_bl], ddof=0)
        fit = self.fit_response(Y, bl_mean.mean, order=order)
        return Y, bl_mean.mean, fit

    def fit_event_second_component(self, Y, bl, first_fit):
        '''call with the output of fit_event to get the ampl of the second component'''
        assert self.config.order == 2
        incl = self.notna & numpy.logical_not(numpy.isnan(Y))
        fitX = self.X[incl]
        first_Y = bl - self.ampl_func(fitX, *first_fit)
        residual = Y[incl] - first_Y
        minval = numpy.nanmin(residual)
        bounds = tuple([[x[0], x[2]] for x in self.constraints['bounds.second']])
        guesses = (self.model_2[0], self.model_2[2])
        popt, pcov = curve_fit(self.ampl_func_2, fitX, residual, p0=guesses, bounds=bounds)
        second_fit = popt
        second_Y = self.ampl_func_2(fitX, *second_fit)
        return fitX, second_fit, first_Y, second_Y

    def fit_event_7(self, Y, bl, first_fit):
        '''call with the output of fit_event to get the ampl of the second (dual) mnodel'''
        assert self.config.order == 7
        incl = self.notna & numpy.logical_not(numpy.isnan(Y))
        fitX = self.X[incl]
        first_Y = bl - self.ampl_func(fitX, *first_fit)
        second_fit = self.fit_response(Y, bl, order=7)
        if second_fit is None:
            second_fit = [x for x in first_fit]
            second_fit.append(0)
        second_Y = bl - self.ampl_func_7(fitX, *second_fit)
        return fitX, second_fit, first_Y, second_Y

    def fit_response(self, Y, bl, order=1):
        '''
        fit a trace with the IPSP model, constrained to the peak time and delay of the model
        :param Y: same len as self.X
        :param bl: baseline of the response
        :return: opts of the fit
        '''
        x = self.X[self.notna]
        y = (bl - Y)[self.notna]
        incl = numpy.logical_not(numpy.isnan(x) + numpy.isnan(y))
        if numpy.count_nonzero(incl) < 10:
            return None
        if order == 1:
            bounds = tuple([[x[0], x[2]] for x in self.constraints['bounds']])
            guesses = (self.model[0], self.model[2])
            popt, pcov = curve_fit(self.ampl_func, x[incl], y[incl], p0=guesses, bounds=bounds)
        elif order == 7:
            bounds = [[x[0], x[2]] for x in self.constraints['bounds']]
            guesses = (self.model_2[0], self.model_2[2], self.model_2[4])
            for b_i in (0, 1):
                bounds[b_i].append(self.constraints['bounds.dual'][b_i][0])
            guesses = self.constrain_guesses(guesses, bounds)
            try:
                popt, pcov = curve_fit(self.ampl_func_7, x[incl], y[incl], p0=guesses, bounds=bounds)
            except RuntimeError:
                return None #if didn't converge
        return popt

    def get_baseline_kernel(self, t=-4, s=10):
        '''
        get weights to measure baseline: an asymmetric, back-looking gaussian
        :param t: gaussian center (ms before to stim)
        :param s: gaussian sigma (ms)
        :return: weights to be used by DescrStatsW
        '''
        key = f'{int(t)}.{int(s)}'
        if key not in self.baseline_kernels:
            fps = self.session.fps
            indices = numpy.arange(-self.config.pre, 0).astype('int64')
            weights = scipy.stats.norm.pdf(indices, loc=-t * fps / 1000, scale=s * fps / 1000)
            self.baseline_kernels[key] = weights
        return self.baseline_kernels[key]

    def pull_ipsps(self, order=1, save_fitforms=False):
        '''
        store baseline and amplitude for each stim in each cell
        using the parameter of the fit as amplitude. NB that alternatively, we could add the bias,
         or compute the diff of where the fit curve min is relative to baseline.
         :param order: 1 to only fit the first, 2 to also fit the second alpha model, 7 to use a dual model
         :param save_fitforms: also save an array of the fitted responses
         :return: array of frame, baseline, response, amplitude for each c, e
         NB don't use "response" for analysis. use "amplitude"-"baseline" to get response values (y at peak)
        '''
        # response is ampl parameter of the fit, compared to where it returns to
        # amplitude is the peak value minus baseline value
        assert order <= self.config.order
        if save_fitforms and (order > 1):
            order_dims = 2
            if order == 7:
                order_dims += 2
            fitforms = numpy.empty((self.n_cells, self.n_stims, self.config.post, order_dims))
            fitforms[:] = numpy.nan
        if order > 1:
            add_cols = 2 #also add fit[0](ampl), peakval for 2nd fit
        else:
            add_cols = 0
        self.responses = numpy.empty((self.n_cells, self.n_stims, 4+add_cols))  # frame, baseline, response, amplitude
        self.responses[:] = numpy.nan
        for ci in range(self.n_cells):
            for ei in range(self.n_stims):
                self.responses[ci, ei, 0] = self.stimframes[ei]
                Y, bl, fit = self.fit_event(ci, ei, order=1)  # bl is in DF/F actual, response (fit[0]) in DF/F change.
                if fit is None:
                    self.responses[ci, ei, 1:] = numpy.nan
                else:
                    # peak value:
                    B = self.model[1]
                    D = self.model[3]
                    y_at_peak = self.alpha_func(B, fit[0], B, fit[1], D)
                    # note because alpha puts the response on top of the baseline, bl-y_at_peak is the actual value.
                    self.responses[ci, ei, 1:4] = bl, fit[0], bl - y_at_peak
                    if order == 2:
                        fitX, second_fit, first_Y, second_Y = self.fit_event_second_component(Y, bl, fit)
                        B = self.model_2[1]
                        D = self.model_2[3]
                        y_at_peak = self.alpha_func(B, second_fit[0], B, second_fit[1], D)
                        self.responses[ci, ei, -2:] = second_fit[0], bl - y_at_peak
                    elif order == 7:
                        fitX, second_fit, first_Y, second_Y = self.fit_event_7(Y, bl, fit)
                        use_model = [x for x in self.model_2]
                        for fi, mi in enumerate((0, 2, 4)): #A1, C, A2
                            use_model[mi] = second_fit[fi]
                        y_at_peak1 = self.dual_alpha_func(self.model_2[1], *use_model) #predict at B1
                        y_at_peak2 = self.dual_alpha_func(self.model_2[5], *use_model) #predict at B2
                        self.responses[ci, ei, -4:] = second_fit[0], bl - y_at_peak1, second_fit[-1], bl - y_at_peak2
                    if save_fitforms:
                        ind_x = (fitX * self.session.fps / 1000).astype('int64')
                        fitforms[ci, ei, ind_x, 0] = first_Y
                        fitforms[ci, ei, ind_x, 1] = second_Y
                        if order == 7:
                            e_model = [x for x in use_model]
                            e_model[0] = 0
                            i_model = [x for x in use_model]
                            i_model[4] = 0
                            fitforms[ci, ei, ind_x, 2] = self.dual_alpha_func(fitX, *i_model)
                            fitforms[ci, ei, ind_x, 3] = self.dual_alpha_func(fitX, *e_model)


        lprint(self, 'Pulled responses for', self.session.tag, 'with ', self.config, 'Model:',
               self.get_modstring(), logger=self.log)
        fn = self.get_fn('ipsps.npy')
        numpy.save(fn, self.responses)
        if save_fitforms:
            numpy.save(fn.replace('s.npy', '_fits.npy'), fitforms)
        return self.responses

    def get_ipsps(self):
        if not hasattr(self, 'responses'):
            fn = self.get_fn('ipsps.npy')
            if os.path.exists(fn):
                self.responses = numpy.load(fn)
                self.model = numpy.load(self.get_fn('model.npy'))
                lprint(self, 'Loaded IPSP model from file:', self.get_modstring())
            else:
                raise FileNotFoundError('fit_model and pull_ipsps first')
                # self.fit_model(save=True)
                # plt.close()
                # self.pull_ipsps()
        return self.responses
