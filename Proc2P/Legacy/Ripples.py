import numpy, scipy, os, pandas
from matplotlib import pyplot as plt
from scipy.signal import bessel, hilbert, filtfilt, decimate, sosfiltfilt
from scipy.io import savemat
from datetime import datetime
from matplotlib.widgets import RectangleSelector
from Batch_Utils import strip_ax
from sklearn import cluster

class Event(object):
    def __init__(self, ripples, p1, p2):
        self.r = ripples
        self.p = [int(p1), int(p2)]
        self.power = self.r.get_power(self.p)
        self.incl = True

    def flip(self):
        self.incl = not self.incl


class UI(object):
    def __init__(self):
        self.pressed = None

    def key_press_callback(self, event):
        self.pressed = event.key


class Ripples(object):
    def __init__(self, prefix, fs=10000, bands=('theta', 'ripple'), config=None, enum=False, force=False,
                 keep_ripples=False, ephys_channels=(1, 1), tag=None, strict_tag=False, load_minimal=False, **kwargs):
        if config is None:
            config = {}
        if type(prefix) is str and prefix.endswith('.ephys'):
            prefix = prefix[:-6]
        self.attrs = []
        if 'ripple' in bands:
            self.attrs.extend(['ftr', 'envelope'])
        if 'theta' in bands:
            self.attrs.extend(['theta', 'tft'])
        if 'sgamma' in bands:
            self.attrs.extend(['sgamma', 'sgft'])
        self.load_attrs = ['ftr', 'theta', 'sgamma']
        self.bands = bands
        self.prefix = prefix
        self.tag = tag
        self.strict_tag = strict_tag
        self.fs = fs
        self.param_defaults(config)
        self.channels = None
        self.ephys_channel_config = ephys_channels
        if prefix is not None:
            self.path = prefix + '_ripples//'
            self.read_phys()
            exist = os.path.exists(self.path)
            if not exist:
                os.makedirs(self.path)
        else:
            exist = False
            load_minimal = True
        if not load_minimal:
            need_sort = False
            if (not exist) or force:
                self.gen_filt()
                if 'ripple' in bands:
                    self.run_filt()
                if 'theta' in bands:
                    self.calc_theta()
                if 'sgamma' in bands:
                    self.calc_sgamma()
                self.save()
            else:
                self.load()
            if exist and not force:
                self.load_ripples(tag)
            elif not keep_ripples:
                self.pick_events()
                need_sort = True
            if enum:
                self.enum_ripples()
                need_sort = True

            if need_sort:
                # sort events based on start time - this is assumed at single ripple mask pulling
                times = numpy.array([x.p[0] for x in self.events])
                self.events = [self.events[i] for i in numpy.argsort(times)]

    def save(self):
        for attr in self.attrs:
            self.__getattribute__(attr).tofile(self.path + attr + '.np')

    def load(self):
        for attr in self.attrs:
            fn = self.path + attr + '.np'
            if os.path.exists(fn):
                self.__setattr__(attr, numpy.fromfile(fn))
        for attr in self.load_attrs:
            if attr not in self.attrs:
                fn = self.path + attr + '.np'
                if os.path.exists(fn):
                    self.__setattr__(attr, numpy.fromfile(self.path + attr + '.np'))
        if hasattr(self, 'ftr'):
            self.std = self.ftr.std()

    def param_defaults(self, config):
        self.lowcut = 130
        self.highcut = 200
        self.filt_order = 3
        self.tr1 = 5
        self.tr2 = 3
        self.minl = self.ms(20)
        self.maxgap = self.ms(15)
        self.mindist = self.ms(50)
        self.window = self.ms(1)
        self.width_lotr = self.ms(50)
        self.y_scale = 1
        for key, value in config.items():
            setattr(self, key, value)

    def ms(self, ms):
        return int(ms * self.fs / 1000)

    def read_phys(self):
        ch, n_channels = self.ephys_channel_config
        raw_shape = n_channels + 1
        ep_raw = numpy.fromfile(self.prefix + '.ephys', dtype='float32')
        n_samples = int(len(ep_raw) / raw_shape)
        ep_formatted = numpy.reshape(ep_raw, (n_samples, raw_shape))
        self.trace = ep_formatted[:, ch]
        self.frame_array = ep_formatted[:, 0]
        self.n = n_samples
        self.ephys_all_channels = ep_formatted[:, 1:]
        # if more channels are available, load it:
        mpn = self.prefix + '.morephys'
        if os.path.exists(mpn):
            self.channels = numpy.fromfile(mpn, dtype='float32')

    def gen_filt(self):
        highcut = self.highcut
        lowcut = self.lowcut
        order = self.filt_order
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = min(highcut / nyq, 0.95)
        self.filter = bessel(order, [low, high], btype='band')
        return self.filter

    def run_filt(self, trace=None, filter=None):
        if trace is None:
            self.ftr = filtfilt(*self.filter, self.trace)
            htr = hilbert(self.ftr)
            self.envelope = numpy.abs(htr)
            self.std = self.ftr.std()
        else:
            ftr = filtfilt(*filter, trace)
            htr = hilbert(ftr)
            envelope = numpy.abs(htr)
            return ftr, envelope

    def calc_theta(self):
        nyq = 0.5 * self.fs
        filter = bessel(self.filt_order, [6.0 / nyq, 9.0 / nyq], btype='band')
        self.tft = filtfilt(*filter, self.trace)
        htr = hilbert(self.tft)
        self.theta = numpy.abs(htr)

    def calc_sgamma(self):
        nyq = 0.5 * self.fs
        filter = bessel(self.filt_order, [20.0 / nyq, 55.0 / nyq], btype='band')
        self.gft = filtfilt(*filter, self.trace)
        htr = hilbert(self.gft)
        self.gamma = numpy.abs(htr)

    def calc_fgamma(self):
        nyq = 0.5 * self.fs
        filter = bessel(self.filt_order, [60.0 / nyq, 100.0 / nyq], btype='band')
        self.gft = filtfilt(*filter, self.trace)
        htr = hilbert(self.gft)
        self.gamma = numpy.abs(htr)

    def calc_bgamma(self):
        nyq = 0.5 * self.fs
        filter = bessel(self.filt_order, [20.0 / nyq, 100.0 / nyq], btype='band')
        self.gft = filtfilt(*filter, self.trace)
        htr = hilbert(self.gft)
        self.gamma = numpy.abs(htr)

    def calc_slo(self):
        nyq = 0.5 * self.fs
        sos = bessel(self.filt_order, [3.0 / nyq, 6.0 / nyq], btype='band', output='sos')
        self.gft = sosfiltfilt(sos, self.trace)
        htr = hilbert(self.gft)
        self.gamma = numpy.abs(htr)

    def set_up_recursive(self, calc_diff):
        self.incl = numpy.ones(self.envelope.shape, dtype='bool')
        self.pend = self.n - self.width_lotr - self.minl - self.window
        self.events = []
        if calc_diff:
            self.trace_diff = numpy.abs(numpy.diff(self.trace))
            self.diff_threshold = self.trace_diff.mean() + self.trace_diff.std() * 3

    def det_next(self, exclude_spikes=False):
        '''exclude_spikes will skip events if there's a 3 SD peak in the raw trace differential'''
        ftr = self.ftr[numpy.where(self.incl)]
        std = ftr.std()
        tr1 = self.tr1 * std
        tr2 = self.tr2 * std
        while self.incl.sum() > 1:  # find new ones until a good one is found or the trace is used up
            # find max envelope not excluded:
            p1 = numpy.argmax(self.envelope * self.incl)
            maxval = self.envelope[p1]
            # filter if second threshold is met
            if maxval < tr1:
                return None
            # walk back until envelope is up:
            # walk through envelope with window steps
            while self.envelope[p1 - self.window] > tr2:
                p1 -= self.window
                if p1 < self.window:
                    break
            # use same logic as in pick_events to detect end of peak
            p2 = p1
            while True:
                p = p2 + self.maxgap
                if self.envelope[p] > tr2:
                    p2 = p
                else:
                    for p in range(p2 + self.maxgap, p2, -self.window):
                        if self.envelope[p] > tr2:
                            p2 = p
                            break
                    break
            # exclude region:
            self.incl[p1 - self.mindist:p2 + self.mindist] = False
            print(
                f'Event {len(self.events)}; peak {(maxval / self.std):.1f} SD;'
                f' {int((p2 - p1) / (self.fs/1000))} ms; {self.count_included()} included')
            # filter if spike is present
            if exclude_spikes:
                passed_spike_test = self.trace_diff[p1:p2].max() < self.diff_threshold
            else:
                passed_spike_test = True
            # filter if minlength is met
            if passed_spike_test and p2 - p1 > self.minl:
                self.events.append(Event(self, p1, p2))
                return p1, p2

    def pick_events(self, skip_save=False):
        # redefine std based on no-noise regions
        self.std = self.ftr[numpy.where(numpy.abs(self.ftr) < self.std * 2)].std()
        # pick regions tr2-tr2 that hit tr1
        tr1 = self.tr1 * self.std
        tr2 = self.tr2 * self.std
        print(f'Treshold set: SD = {self.std}; tr1 = {self.tr1}; tr2 = {self.tr2}')
        self.events = []
        p1, pend = 0, self.n - self.width_lotr - self.minl - self.window
        while True:
            # walk through envelope with window steps
            while self.envelope[p1] < tr2:
                p1 += self.window
                if p1 > pend:
                    break
            p2 = p1
            # extend region with maxgap steps adn then go backwards to find last value above tr
            while True:
                p = p2 + self.maxgap
                if p >= len(self.envelope):
                    break
                if self.envelope[p] > tr2:
                    p2 = p
                else:
                    for p in range(p2 + self.maxgap, p2, -self.window):
                        if self.envelope[p] > tr2:
                            p2 = p
                            break
                    break
            # filter if second treshold is met
            for i in range(p1, p2):
                if self.envelope[i] > tr1:
                    # filter if minlength is met
                    if p2 - p1 > self.minl:
                        self.events.append(Event(self, p1, p2))
                    break
            p1 = p2 + self.window
            if p2 > pend:
                break
        if not skip_save:
            self.calc_powers()
            self.save_ripples()

    def get_HFOs(self, lowcut=80, highcut=500, tr1=7, tr2=5, gap=1000, minl=10, save_envelope=True, use_current=False):
        '''Detect HFOs and return single event/ cluster onsets'''
        if not use_current:
            self.lowcut = lowcut
            self.highcut = highcut
            self.minl=self.ms(minl)
            self.gen_filt()
            self.run_filt()
        self.tr1 = tr1
        self.tr2 = tr2
        self.pick_events(skip_save=True)
        times = []
        if len(self.events) > 0:
            times = single_ripple_onsets(self.events, gap=self.ms(gap))
            times.sort()
            self.events = []
            for t in times:
                self.events.append(Event(self, t, t+self.ms(minl)))
        self.save_ripples(overwrite=True)
        if save_envelope and not use_current:
            self.attrs = ['ftr', 'envelope']
            self.save()
        return times


    def calc_powers(self):
        self.powers = numpy.empty(len(self.events))
        for i, e in enumerate(self.events):
            self.powers[i] = e.power
        self.plist = numpy.argsort(-self.powers)
        print(len(self.events), 'events')
        # self.save_ripples()

    def prep_fig(self):
        if self.extended_figure:
            self.fig = plt.subplots(3, 1, sharex=False)
            axes = self.fig[1]
            axes[0].get_shared_x_axes().join(axes[0], axes[1])
            y = decimate(self.envelope, 10)
            x = numpy.linspace(0, self.n, int(self.n / 10))
            axes[2].plot(x, y, color='black')
            strip_ax(axes[2])
            self.pos_marker_line = self.fig[1][2].axvline(0, color='orange')
            axes[2].axhline(self.tr1 * self.std, color='grey')
            axes[2].axhline(self.tr2 * self.std, color='black')
        else:
            self.fig = plt.subplots(2, 1, sharex=True)
        # plt.get_current_fig_manager().window.showMaximized()
        self.cursor = UI()
        self.exit = False
        self.fig[0].canvas.mpl_connect('key_press_event', self.cursor.key_press_callback)
        self.fig[0].canvas.mpl_connect('close_event', self.on_close)
        self.redraw = False

    def on_close(self, event):
        self.exit = True

    def show_event(self, i, j=None):
        if j is None:
            p1, p2 = self.events[i].p
            start = int(p1 - (self.fs - (p2 - p1)) * 0.5)
            stop = int(start + self.fs)
        else:
            p1, p2 = i, j
            start, stop = int(i), int(j)
        fig, axes = self.fig
        for ax in axes[:2]:
            ax.cla()
        if self.tsource:
            lcolor = 'black'
        elif self.events[i].incl:
            lcolor = 'black'
        else:
            lcolor = 'red'
        axes[0].plot(self.trace[start:stop], color=lcolor, linewidth=0.5)
        # stretch y scale
        ylim = numpy.array(axes[0].get_ylim())
        lim_m = ylim.mean()
        axes[0].set_ylim(lim_m - (lim_m - ylim[0]) * self.y_scale, lim_m + (ylim[1] - lim_m) * self.y_scale)
        axes[1].plot(self.ftr[start:stop])
        axes[1].plot(self.envelope[start:stop])
        if not self.tsource:
            axes[1].axvline(p1 - start)
            axes[1].axvline(p2 - start)
        axes[1].axhline(self.tr1 * self.std, color='grey')
        axes[1].axhline(self.tr2 * self.std, color='black')
        if self.extended_figure:
            self.pos_marker_line.set_xdata(p1)
        if self.redraw:
            plt.draw()
            self.redraw = False
        if hasattr(self, 'frame_array'):
            fig.suptitle(f'{self.prefix} t = {p1/self.fs:.2f} s; frame = {int(self.frame_array[p1])}')
        fig.show()
        try:  # to avoid displaying error message when figure closed by clicking the x
            fig.waitforbuttonpress()
            return self.cursor.pressed
        except:
            self.exit = True

    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.rect = (x1, x2, y1, y2)
        self.redraw = True

    def pick_delay(self, first_sample, last_sample, screensize=None):
        self.delay_fig = plt.subplots(3, 1, sharex=False, figsize=screensize)
        fig, axes = self.delay_fig
        self.cursor = UI()
        self.exit = False
        fig.canvas.mpl_connect('key_press_event', self.cursor.key_press_callback)
        fig.canvas.mpl_connect('close_event', self.on_close)
        for ax in axes:
            ax.cla()
        axes[0].plot(self.trace[first_sample:last_sample], color='black', linewidth=0.5)
        # stretch y scale
        ylim = numpy.array(axes[0].get_ylim())
        lim_m = ylim.mean()
        axes[0].set_ylim(lim_m - (lim_m - ylim[0]) * self.y_scale, lim_m + (ylim[1] - lim_m) * self.y_scale)
        # plot trace for entire stretch, use xlim to adjust wiew
        curr_xlim_pos = 0  # in fs
        max_xlim_pos = int(last_sample / self.fs) + 1
        for ai in (0, 1):
            axes[ai].set_xlim(0, self.fs)
        axes[0].get_shared_x_axes().join(axes[0], axes[1])
        axes[1].plot(self.ftr[first_sample:last_sample])
        for ai in (1, 2):
            axes[ai].plot(self.envelope[first_sample:last_sample])
            axes[ai].axhline(self.tr1 * self.std, color='grey')
            axes[ai].axhline(self.tr2 * self.std, color='black')
        overview_start_marker = axes[2].axvline(0, color='black')
        overview_stop_marker = axes[2].axvline(self.fs, color='black')
        # set up selector
        self.rect = None
        selpos = None
        self.rs = RectangleSelector(axes[0], self.line_select_callback,
                                    drawtype='box', useblit=False, button=[1],
                                    minspanx=100, minspany=0.01, spancoords='data',
                                    interactive=True)

        fig.show()
        while not self.exit:
            self.redraw = False
            try:  # to avoid displaying error message when figure closed by clicking the x
                fig.waitforbuttonpress()
            except:
                self.exit = True
            bp = self.cursor.pressed
            if bp == 'q':
                self.exit = True
            elif bp == 'left':
                if curr_xlim_pos > 0:
                    curr_xlim_pos -= 1
                    self.redraw = True
            elif bp == 'right':
                if curr_xlim_pos < max_xlim_pos:
                    curr_xlim_pos += 1
                    self.redraw = True
            if self.redraw:
                if selpos is None and self.rect is not None:
                    selpos = self.rect[0]
                    vline_hi = axes[1].axvline(selpos, color='orange')
                    vline_lo = axes[2].axvline(selpos, color='orange')
                elif self.rect is not None:
                    selpos = self.rect[0]
                    vline_hi.set_xdata(selpos)
                    vline_lo.set_xdata(selpos)
                startpos = curr_xlim_pos * self.fs
                for ai in (0, 1):
                    axes[ai].set_xlim(startpos, startpos + self.fs)
                overview_start_marker.set_xdata(startpos)
                overview_stop_marker.set_xdata(startpos + self.fs)
                plt.draw()
            self.cursor.pressed = None
        return selpos

    def get_power(self, p):
        p1, p2 = p
        return numpy.sum(self.envelope[p1:p2])

    def disp_ripples(self, lines, i, j=None):
        lines0, lines1, converter = lines
        self.prep_fig()
        if j is None:
            j = i + self.fs
            d = self.fs
        else:
            d = j - i
        self.tsource = True
        while not self.exit:
            bp = self.show_event(i, j)
            if bp == 'q':
                self.exit = True
                for l in lines0:
                    l.remove()
                for l in lines1:
                    l.remove()
            elif bp == 'left':
                i = max(0, i - d)
                j = max(0, j - d)
            elif bp == 'right':
                i = min(len(self.trace), i + d)
                j = min(len(self.trace), j + d)
            for l in lines0:
                l.set_xdata(converter(i))
            for l in lines1:
                l.set_xdata(converter(j))
            self.cursor.pressed = None

    def enum_ripples(self, order='power', no_save=False):
        self.extended_figure = True
        self.prep_fig()
        self.tsource = False
        if order == 'power':
            order = self.plist
        else:
            order = range(len(self.events))
        i = 0
        if len(order) == 0:
            print('No events')
            self.exit = True
        while not self.exit:
            bp = self.show_event(order[i])
            if bp == 'q':
                self.exit = True
                if not no_save:
                    self.save_ripples()
            elif bp == 'left':
                i -= 1
                if i < 0:
                    i = len(self.events) - 1
            elif bp == 'right':
                i += 1
                if i > len(self.events) - 1:
                    i = 0
            elif bp == 'up' or bp == 'down':
                self.redraw = True
                self.events[order[i]].flip()
            self.cursor.pressed = None
            self.current = i

    def rec_enum_ripples(self, exclude_spikes=True):
        self.extended_figure = True
        self.prep_fig()
        self.tsource = False
        self.set_up_recursive(calc_diff=exclude_spikes)
        i = 0
        p = self.det_next(exclude_spikes=exclude_spikes)
        if p is None:
            print('No events')
            self.exit = True
        while not self.exit:
            bp = self.show_event(i)
            if bp == 'q':
                self.exit = True
            if bp == 's':
                self.exit = True
                self.calc_powers()
                self.save_ripples()
            elif bp == 'left':
                i -= 1
                if i < 0:
                    i = len(self.events) - 1
            elif bp == 'right':
                i += 1
                if i > len(self.events) - 1:
                    p = self.det_next()
                    if p is None:
                        print('No more events')
                        i = 0
            elif bp == 'up' or bp == 'down':
                self.redraw = True
                self.events[i].flip()
            self.cursor.pressed = None
            self.current = i

    def count_included(self):
        i = 0
        for event in self.events:
            if event.incl:
                i += 1
        return i

    def save_ripples(self, overwrite=False):
        if self.prefix is None:
            return -1
        if (self.tag is not None) and os.path.exists(self.path + self.tag + '.ripples') and not overwrite:
            print(f'Ripple tag {self.tag} already exists, creating new timestamp')
            self.tag = None
        if self.tag is None:
            ts = str(datetime.now())
            ts = ts[:ts.find(' ')] + '-' + ts[ts.find(' ') + 1:ts.find('.')].replace(':', '-') + '.ripples'
        else:
            ts = self.tag + '.ripples'
        if len(self.events) < 1:
            ea = numpy.zeros((1, 2))
        else:
            events = []
            for e in self.events:
                if e.incl:
                    events.append(e.p)
            ea = numpy.empty((len(events), 2))
            for i, p in enumerate(events):
                ea[i, :] = p
        ea.tofile(self.path + ts)
        print(ts, f'saved with {self.count_included()} events.')

    def export_ripple_times(self):
        if len(self.events) > 0:
            events = []
            for e in self.events:
                if e.incl:
                    events.append(e.p)
            ea = numpy.empty((len(events), 2))
            for i, p in enumerate(events):
                ea[i, :] = p
        pandas.DataFrame(ea, columns=['RippleStart', 'RippleStop']).to_excel(self.prefix + '_ripple_times.xlsx')
        print(self.prefix, 'ripples exported.')

    def load_ripples(self, fn=None):
        if fn is not None and not fn.endswith('ripples'):
            fn += '.ripples'
        if fn is not None and self.strict_tag:
            if not os.path.exists(self.path + fn):
                raise FileNotFoundError(f'{fn} not available')
        if fn is None or not os.path.exists(self.path + fn):
            flist = []
            for f in os.listdir(self.path):
                if f.endswith('.ripples'):
                    flist.append(f)
            if len(flist) == 0:
                print(self.prefix, 'No ripples')
                return -1
            else:
                flist.sort()
                fn = flist[-1]
        print(fn)
        self.ripples_fn = fn
        events = numpy.fromfile(self.path + fn)
        events = events.reshape((int(len(events) / 2), 2))
        self.events = []
        if hasattr(self, 'envelope'):
            for p in events:
                self.events.append(Event(self, *p))
            self.calc_powers()

    def export_matlab(self):
        '''
        {'__globals__': ['fs', 's', 'total_ns', 'gn', 'of'],
         '__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Thu Mar 29 16:13:43 2018',
         '__version__': '1.0',
         'fs': 20000,
         'gn': array([ 0.67108864,  1.048576  ,  0.17592186,  0.11258999]),
         'of': array([ 1.1 , -0.35, -3.  , -0.8 ]),
         's': array([[ 0.0691672 ,  0.06171192, -1.3152473 ,  0.0221665 ],
                [ 0.0623602 ,  0.07273277, -1.3133024 ,  0.02281478],
                [ 0.09185719,  0.0782432 , -1.32205427,  0.02184235],
                ...,
                [-0.07410391,  0.02767693,  0.99264997,  0.02249064],
                [-0.08058677,  0.03027007,  0.98811197,  0.02281478],
                [-0.09225591,  0.02994592,  0.98324984,  0.02313893]], dtype=float32),
         'total_ns': 11994000}
        '''

        header = b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Jan 03 04:43:06 2018'
        version = '1.0'
        globals = ['fs', 's', 'total_ns', 'gn', 'of']
        gn = numpy.ones(4, dtype='<f8')
        s = numpy.zeros((self.n, 4), dtype='<f4')
        s[:, 0] = self.trace

        mat = {'__header__': header, '__version__': version, '__globals__': globals, 'fs': self.fs, 'gn': gn, 'of': gn,
               's': s, 'total_ns': self.n}

        scipy.io.savemat(self.prefix + '_ephysexport.mat', mat)

def single_ripple_onsets(events, gap=4000):
    '''events: event list or Ripples
    gap: samples'''
    event_t = numpy.array([e.p[0] for e in events])
    clustering = cluster.DBSCAN(eps=gap, min_samples=2).fit(event_t.reshape(-1, 1))
    single_ripple_indices = numpy.where(clustering.labels_ < 0)[0]
    event_number = len(single_ripple_indices) + clustering.labels_.max() + 1
    events = numpy.empty(event_number, dtype=numpy.int64)
    ri, rci = 0, 0
    for ri, rj in enumerate(single_ripple_indices):
        events[ri] = event_t[rj]
    for rci in range(clustering.labels_.max() + 1):
        # get list of ripples in cluster
        current_cluster = numpy.where(clustering.labels_ == rci)[0]
        rj = current_cluster[0]
        events[ri+rci] = event_t[rj]
    return events[:ri + rci + 1]

def export_SCA(path, prefix, channels=(1, 1)):
    savepath = path + '_LFP/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    suffix = '_SpikeAnalyzerExport.mat'
    # if os.path.exists(ofn):
    #     print(f'Skipping {prefix}')
    # else:
    for chi in range(channels[1]):

        r = Ripples(prefix, ephys_channels=(chi+1, channels[1]), load_minimal=True)

        header = str.encode(f'Ephys trace from {prefix} ch {chi}')
        version = '1.0'
        globals = ['sbuf', 'fs', 'samples']
        s = r.trace.astype('<f4')

        mat = {'__header__': header, '__version__': version, '__globals__': globals, 'sbuf': s, 'fs': r.fs,
               'samples': len(s)}
        ofn = savepath + prefix + f'_ch{chi}' + suffix

        savemat(ofn, mat)
        print(f'Exported: {prefix}')
# #
# if __name__ == '__main__':
#     os.chdir('//NEURO-GHWR8N2//AnalysisPC-Barna-2PBackup3//cckdlx//')
#     prefix = 'cckdlx_084_461'
#     r = Ripples(prefix)
