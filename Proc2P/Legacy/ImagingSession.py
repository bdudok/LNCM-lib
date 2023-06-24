import copy
from sklearn.metrics import mutual_info_score
from .BehaviorSession import BehaviorSession
from .LoadPolys import FrameDisplay, LoadPolys
from .Firing import Firing
from .Quad import SplitQuad
from .Batch_Utils import mad_based_outlier
from .Ripples import Ripples
import time
from scipy import stats

# very legacy
# from EyeTracking import load_mikko_trace

class ImagingSession(object):
    def __init__(self, prefix, tag=None, norip=False, no_tdml=False, opath='.', ch=0, ephys_channels=(1, 1),
                 ripple_tag=None, force_calc_rpw=False):
        self.opath = opath
        self.norip = norip
        self.prefix = prefix
        self.rois = FrameDisplay(self.prefix, tag=tag)
        self.ca = Firing(prefix, tag=tag, ch=ch)
        self.tag = tag
        self.ripple_tag = ripple_tag
        self.force_calc_rpw = force_calc_rpw
        self.ephys_channel_config = ephys_channels
        if tag == 'skip':
            self.rois.image.load_data()
            self.ca.frames = self.rois.image.nframes
        self.ca.load()
        self.dualch = self.ca.is_dual
        # print(ch, self.ca.channels, self.dualch)
        self.has_behavior = False
        self.has_speed = True
        if os.path.exists(prefix + '.tdml') and not no_tdml:
            self.bdat = BehaviorSession(prefix + '.tdml', silent=True)
            self.mapfiles()
        elif os.path.exists(self.prefix + '_quadrature.mat'):
            self.pos = SplitQuad(self.prefix)
            # populate missing tdml fields with blank data
            self.pos.relpos = self.pos.pos
            self.pos.laps = numpy.zeros(self.ca.frames, dtype='int')
            print('.tdml file not found, using speed only')
        else:
            print('Position data not found, using zero speed')
            self.has_speed = False
            self.pos = SplitQuad(self.ca.frames)
            self.pos.laps = numpy.zeros(self.ca.frames, dtype='int')
        if self.ca.frames == len(self.pos.pos) - 2:
            self.poppos()
        if os.path.exists(prefix + '.ephys'):
            self.map_phys()
        if os.path.exists(prefix + '_ripples') and not norip:
            self.map_ripples()
        self.pltnum = 0
        self.zscores = {}
        self.rates = {}
        self.colors = {}
        self.path = os.path.join(os.getcwd(), '')
        self.eye = load_mikko_trace(self.path, self.prefix)

        # load opto if available:
        ofn = self.prefix + '_opto.npy'
        if os.path.exists(ofn):
            self.opto = numpy.load(ofn)
        else:
            self.opto = None

    def startstop(self, *args, **kwargs):
        return self.pos.startstop(*args, **kwargs)

    def getparam(self, param):
        opn = param
        self.disc_param = False
        if type(param) == str:
            if param == 'spikes':
                param = numpy.nan_to_num(self.ca.spikes * numpy.array(self.ca.event, dtype='bool'))
                self.disc_param = True
            elif param == 'rel':
                param = self.ca.rel
            elif param == 'ntr':
                param = self.ca.ntr
            elif param == 'ontr':
                param = self.ca.ontr
            elif param == 'smtr':
                param = self.ca.smtr
            elif param == 'nnd':
                if hasattr(self.ca, 'nnd'):
                    param = self.ca.nnd
            elif param == 'peaks':
                self.disc_param = True
                param = numpy.nan_to_num(self.ca.peaks.astype('bool'))
            elif param == 'aupeaks':
                param = numpy.maximum(0, self.ca.smtr * self.ca.b2)
            elif param == 'bin_onset':  # onset frame of 3 sd events
                param = numpy.zeros(self.ca.rel.shape, dtype='bool')
                for event in self.ca.eventarray:
                    param[event[0], event[1]] = 1
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

    def poppos(self):
        attrs = 'pos,movement,speed,anyevent,events,laps,qepos,relpos,smspd'
        for attr in attrs.split(','):
            if hasattr(self.pos, attr):
                self.pos.__setattr__(attr, numpy.delete(self.pos.__getattribute__(attr), -1))

    def mapfiles(self):
        '''bdat attribute is coming from BehaviorSession (.tdml). Pos is coming from SplitQuad (quadrature.mat)
        bmpos is calculated in this function, has shape of pos, updated with position form bdat.
        bmpos is then used for abspos.
        laps is determined here from diff of bmpos'''
        if os.path.exists(self.prefix + '_quadrature.mat') and os.path.exists(self.prefix + '.tdml'):
            self.pos = SplitQuad(self.prefix)
            # self.pos.bm = Tdml(self.prefix + '.tdml')
            # implement this as a standalone function that checks and fixes mappability of ttls
            # see if number of syncs adds up with number of frames
            pos = self.pos
            bdat = self.bdat
            theoretical_fps = 15.5
            theoretical_frame_per_ttl = 15
            frames = len(pos.pos)
            expected_duration = frames / theoretical_fps
            expected_syncs = frames / theoretical_frame_per_ttl
            duration = bdat.frametimes[-1] - bdat.frametimes[0]
            # do we have a mismatch in duration or number of signals?
            time = numpy.array(bdat.frametimes)[1:]  # (have to skip first one to avoid offset)
            dur_match = 0.9 < expected_duration / (time[-1] - time[0]) < 1.1
            sync_match = abs(expected_syncs - len(time)) < expected_duration / 60  # 1 per min is OK
            all_ttls = numpy.where(numpy.diff(pos.data[0]))[0]
            if not (dur_match and sync_match):
                # check if multiple scans are included
                dt = numpy.diff(bdat.frametimes)
                if numpy.any(dt > 2.5):
                    # split into sections
                    s_borders = [0]
                    s_borders.extend(numpy.where(dt > 2.5)[0])
                    s_borders.append(len(dt))
                    i = numpy.argmin(numpy.abs(numpy.diff(s_borders) - expected_syncs))  # closest matching_chunk
                    time = numpy.array(bdat.frametimes[s_borders[i]:s_borders[i + 1]])[1:]
                    dur_match = 0.9 < expected_duration / (time[-1] - time[0]) < 1.1
                    sync_match = abs(expected_syncs - len(time)) < 5
            if dur_match:
                self.sbttl = all_ttls
                if sync_match:
                    # collect time to frame sync
                    # 1, detect updates with non-outlier time interval
                    dt = numpy.diff(time)
                    nol = mad_based_outlier(dt, thresh=6)
                    # get point pairs for fitting
                    x = nol + 1  # number of signal
                    y = time[x]
                # if not, check if rec is more than actual TTLs, and attempt to exclude the extra ones.
                elif len(all_ttls) < len(time):
                    # 1, find correct time interval
                    dt = numpy.diff(time)
                    hist_bins = numpy.linspace(0, 1.5, 10)
                    dt_hist = numpy.histogram(dt, bins=hist_bins)
                    best_bin = numpy.argmax(dt_hist[0])
                    nol = numpy.logical_and(dt_hist[1][best_bin] < dt, dt < [best_bin + 1])
                    mean_dt = numpy.array(dt)[nol].mean()
                    # get paired ttls from interval from last one:
                    last_time = time[0]
                    new_time = numpy.empty(len(all_ttls))
                    new_time[0] = last_time
                    x, y = [], []
                    for i in range(len(all_ttls)):
                        time_index = numpy.argmin(numpy.abs(last_time + mean_dt - time))
                        if time_index < len(time) - 1:
                            tstamp = time[time_index]
                            # if delay is precise enough, keep it, otherwise increment with typical dt
                            if abs((tstamp - last_time) - mean_dt) < 0.03:
                                x.append(i)
                                y.append(tstamp)
                                last_time = tstamp
                            else:
                                last_time += mean_dt
                            new_time[i] = tstamp
                        else:
                            new_time = new_time[:i]
                            break
                    time = new_time
                # 2 fit function on non-outlier syncs
                self.bmtime = time
                f1 = numpy.polyfit(x, y, 1)  # ax+b
                # predict time of frames
                x1 = numpy.arange(len(all_ttls))
                y_pred = x1 * f1[0] + f1[1]
                # fit function on frame to time
                f2 = numpy.polyfit(all_ttls, y_pred, 1)  # ax+b
                # 3, create frame coded pos from bm pos updates
                d = self.bdat.data  # 0:timestamp, 1:pos
                # cycle a number of offsets to determine best overlap bw quad speed and bmpos
                # this is necessary because there is no sure way to determine the first matching pair of ttls
                bmpos_list = []
                linpos_list = []
                score_list = []
                offsets = numpy.arange(-3, 4)
                for offset in offsets:
                    bmpos = numpy.zeros(frames)
                    last_update, last_pos = 0, d[1][0]
                    expected_frame = (d[0] - f2[1]) / f2[0] + all_ttls[0]
                    for i in range(len(d[0])):
                        # for each timestamp, find the frame by interpolating between prev and next ttl
                        e_f = expected_frame[i]
                        if 0 < e_f < frames:
                            loc = numpy.searchsorted(all_ttls, e_f) + offset
                            if 0 < loc < len(time) - 1:
                                t = d[0][i]
                                t0, t1 = time[loc - 1:loc + 1]
                                if t1-t0 < 0.1:
                                    if len(time) > loc + 1:
                                        t1 = time[loc+1]
                                    else:
                                        t1 = t0+1
                                interp = (t - t0) / (t1 - t0)
                                fr0, fr1 = all_ttls[loc - 1:loc + 1]
                                try:
                                    est_fr = int(fr0 + (fr1 - fr0) * interp)
                                except:
                                    print(len(all_ttls), e_f, t0, t1, interp, fr0, fr1)
                                    assert False
                                bmpos[last_update:est_fr] = last_pos
                                last_update = est_fr
                        last_pos = d[1][i]
                    bmpos[last_update:] = last_pos
                    bmpos_list.append(bmpos)
                    lin_pos = numpy.copy(bmpos)
                    for t in numpy.where(numpy.diff(bmpos) < -bmpos.max() / 2)[0]:
                        lin_pos[t + 1:] += bmpos[t]
                    lin_pos -= lin_pos.min()
                    lin_pos /= lin_pos.max()
                    linpos_list.append(lin_pos)
                    score_list.append(
                        stats.spearmanr(numpy.diff(lin_pos), numpy.diff(self.pos.qepos))[0])
                sel_offset = numpy.argmax(score_list)
                if not sync_match:
                    overlay_fn = self.prefix + '_position_overlay.png'
                    if not os.path.exists(overlay_fn):
                        plt.figure()
                        plt.plot(linpos_list[sel_offset], label='behaviormate')
                        plt.plot(self.pos.qepos, label='scanbox')
                        plt.gca().legend()
                        plt.savefig(overlay_fn)
                        print(self.prefix,
                              'too many TTLs logged. attempting to detect correct ones. Check position overlay.')
                        plt.show(block=True)
                        # plt.close()
                self.bmpos = bmpos_list[sel_offset]
                # update sync function with correct offset
                x1 = numpy.arange(len(all_ttls)) + offsets[sel_offset]
                y_pred = x1 * f1[0] + f1[1]
                f2 = numpy.polyfit(all_ttls, y_pred, 1)  # ax+b
                self.sync_spline = numpy.arange(frames) * f2[0] + f2[1]

                # add needed attrs
                self.pos.pos = self.bmpos
                self.pos.relpos = (self.bmpos - self.bmpos.min()) / (self.bmpos.max() - self.bmpos.min())
                self.pos.laps = numpy.zeros(self.pos.pos.shape, dtype='int')
                # update laps based on position difference. keep track of last update to avoid issues from double tags
                tag_dist_treshold = 100  # in frames
                last_tag_frame = -1 * tag_dist_treshold
                for t in numpy.where(numpy.diff(self.pos.pos) < -self.pos.pos.max() / 2)[0]:
                    if t - last_tag_frame > tag_dist_treshold:
                        self.pos.laps[t + 1:] += 1
                        last_tag_frame = t
                self.has_behavior = True
            else:
                print(self.prefix, 'tdml and quad encoder does not match')
        # this commented out, how would we analyze a session without syncs to the treadmill?
        # elif os.path.exists(self.prefix + '.tdml'):
        #     self.pos = Quad(self.prefix)
        #     # add needed attrs
        #     self.pos.pos = self.pos.relpos
        #     for i, t in enumerate(self.pos.laptimes):
        #         self.pos.laptimes[i] = self.timetoframe(t)

    def timetoframe(self, t):
        if t < self.bdat.frametimes[0] or t <= self.sync_spline[0]:
            return 0
        elif t > self.bdat.frametimes[-1]:
            return len(self.pos.qepos)
        else:
            return numpy.where(self.sync_spline <= t)[0][-1]

    def timetoframe_fast(self, t):
        '''should be faster, but not foolproof'''
        ig = min(int(t * 15.17 - 15), self.ca.frames - 15)
        while self.sync_spline[ig] > t and ig > 0:
            ig -= 15
        ig = max(0, ig)
        for i, x in enumerate(self.sync_spline[ig:]):
            if x > t:
                return max(0, ig + i - 1)

    def frametotime(self, f):
        return self.sync_spline[int(f)]

    def map_phys(self):
        ch, n_channels = self.ephys_channel_config
        raw_shape = n_channels + 1
        ep_raw = numpy.fromfile(self.prefix + '.ephys', dtype='float32')
        n_samples = int(len(ep_raw) / raw_shape)
        ep_formatted = numpy.reshape(ep_raw, (n_samples, raw_shape))
        self.ephystrace = ep_formatted[:, ch]
        self.ephysframes = ep_formatted[:, 0]
        self.ephys_all_channels = ep_formatted[:, 1:]
        # a = numpy.fromfile(self.prefix + '.ephys', dtype='float32')
        # b = numpy.reshape(a, (int(len(a) / 2), 2))
        # self.ephystrace = b[:, 1]
        # self.ephysframes = b[:, 0]
        # DISABLING THIS, TAKES SECONDS AND NEVER ACTUALLY USED. IF NEEDED, MAKE IT MORE EFFICIENT OR SAVE IN FILE
        # x = numpy.zeros(self.ca.frames, dtype='int')
        # x2 = numpy.zeros(self.ca.frames, dtype='int')
        # t0, f = 0, 0
        # for t1, s in enumerate(self.ephysframes):
        #     if s > f:
        #         if f < self.ca.frames:
        #             x[f] = (t0 + t1) / 2
        #             x2[f] = t0
        #             t0 = t1
        #             f += 1
        # if f < self.ca.frames:
        #     x[f] = (t0 + t1) / 2
        # self.ephysx = x
        # self.ephys_framestarts = x2

    def map_ripples(self):
        self.ripples = Ripples(self.prefix, tag=self.ripple_tag, ephys_channels=self.ephys_channel_config,
                               strict_tag=True, )
        # determine maximum number of frames
        if self.ca.frames is not None:
            nframes = self.ca.frames
        else:
            nframes = int(self.ephysframes[-1])
        self.nframes = nframes
        # update to store this info in file and reload
        # tic = time.time()
        keys = ('ripple_power', 'theta_power', 'ripple_frames')
        path = self.prefix + '_ripples//'
        ffn = path + 'fast_load.txt'
        if os.path.exists(ffn) and not self.force_calc_rpw:
            print('Loading processed ephys trace from file created ',
                  time.ctime(min(os.path.getctime(ffn), os.path.getmtime(ffn))))
            for key in keys:
                setattr(self, key, numpy.load(path + key + '.npy'))
        else:
            self.ripple_power = numpy.zeros(nframes)
            self.theta_power = numpy.zeros(nframes)
            theta = hasattr(self.ripples, 'theta')
            t0, f = 0, 0
            for t1, s in enumerate(self.ephysframes):
                if s > f:
                    if f < nframes:
                        self.ripple_power[f] = self.ripples.envelope[t0:t1].mean()
                        if theta:
                            self.theta_power[f] = self.ripples.theta[t0:t1].mean()
                        t0 = t1
                        f += 1
            frs = []
            for e in self.ripples.events:
                f = self.sampletoframe(e.p[0])
                if f < nframes and f not in frs:
                    if not self.pos.movement[f]:
                        frs.append(f)
            frs.sort()
            self.ripple_frames = numpy.array(frs)
            for key in keys:
                numpy.save(path + key, getattr(self, key))
                with open(ffn, 'w') as f:
                    f.write('True')
        # print('Map ripples run in :', time.time() - tic, 's')

    def sampletoframe(self, s):
        return int(self.ephysframes[int(s)])

    def frametosample(self, f):
        wh = numpy.where(self.ephysframes == f)[0]
        return wh[0], wh[-1], wh.mean()

    def frametosample_fast(self, f):
        return numpy.argmax(self.ephysframes > f - 1)

    def export_raw(self):
        colnames = ['ThetaPower', 'Speed']
        data = numpy.empty((self.ca.frames, self.ca.cells + len(colnames)))
        colnames.extend(['C' + str(x) for x in range(self.ca.cells)])
        if hasattr(self, 'theta_power'):
            data[:, 0] = self.theta_power
        else:
            data[:, 0] = numpy.nan
        data[:, 1] = self.pos.speed
        data[:, 2:] = self.ca.trace.transpose()
        print(f'Writing excel file:{self.prefix}-raw_F.xlsx ...')
        pandas.DataFrame(data, columns=colnames).to_excel(self.prefix + '-raw_F.xlsx')
        print('Done.')

    def qc(self, trim=10):
        '''returns a bool mask for cells that pass qc
        trim: if >0, remove cells that are close to the top or bottom row to avoid artefacts
        '''
        nnd = numpy.any(numpy.nan_to_num(self.ca.nnd) > 0, axis=1)
        ppm = numpy.any(numpy.nan_to_num(self.ca.peaks) > 0, axis=1)
        qc_pass = numpy.logical_and(nnd, ppm)
        if trim:
            tops = [self.rois.polys[ci].min(axis=0)[1] for ci in range(self.ca.cells)]
            bottoms = [self.rois.polys[ci].max(axis=0)[1] for ci in range(self.ca.cells)]
            cutoff = self.rois.image.info['sz'][0] - trim
            incl = [tops[ci] > trim and bottoms[ci] < cutoff for ci in range(self.ca.cells)]
            qc_pass = numpy.logical_and(incl, qc_pass)
        return qc_pass

    def RUNseq(self, param=None, span=None, duration=50, cmap='hot'):
        '''
        sort cells to find  sequences on movement periods and plot entire session
        '''
        if span is None:
            span = 8, len(self.pos.pos) - duration
        param = self.getparam(param)
        # find run periods to use for sorting
        starts = []
        t1 = span[0]
        while t1 < span[1] - duration:
            t2 = int(t1 + duration)
            if self.pos.movement[t1:t2].min() == 1:
                while self.pos.movement[t2 + 1] == 1 and t2 < span[1]:
                    t2 += 1
                loc = numpy.argmax(self.pos.speed[t1 - 7:t1 + 8]) - 7
                t1 += loc
                t2 = min(t2, t2 + loc)
                starts.append((t1, t2))
                t1 = t2
            else:
                t1 += 1
        # collect ranks for all periods
        ranks = numpy.empty((self.ca.cells, len(starts)))
        for i, (t1, t2) in enumerate(starts):
            maxlocs = numpy.argmax(param[:, t1:t2], axis=1)
            for c in range(self.ca.cells):
                if self.ca.event[c, maxlocs[c] + t1]:
                    ranks[c, i] = maxlocs[c]
                else:
                    ranks[c, i] = numpy.nan
        self.corder = numpy.argsort(numpy.nanmean(ranks, axis=1))
        self.plot_session(param=param, corder=self.corder, cmap=cmap)

    def running_placeseq(self, pf_calc_param='nnd', pf_calc_bins=25, display_param='smtr', cmap='plasma'):
        # identify place cells
        self.calc_MI(pf_calc_param)
        pc = []
        nnd = numpy.nan_to_num(self.ca.nnd) > 0
        ppm = numpy.nan_to_num(self.ca.peaks) > 0
        for c in range(self.ca.cells):
            if self.statistical_is_placecell[c]:
                if numpy.any(nnd[c]) and numpy.any(ppm[c]):
                    if numpy.nanpercentile(self.ca.ntr[c], 99) > 1:
                        pc.append(c)
        self.placefields_smooth(param=pf_calc_param, silent=True, bins=pf_calc_bins)
        pc_order = [c for c in self.corder_smoothpf if c in pc]
        fig, ca = plt.subplots()
        param = self.getparam(display_param)
        ca.imshow(param[pc_order][:, self.pos.gapless], aspect='auto', cmap=cmap)
        fig.set_size_inches(9, 6)
        ca.set_xlabel('Running time (frames)')
        ca.set_ylabel('Cell # (sorted by place field location)')
        fig.show()
        self.pc_order = pc_order
        return fig

    def plot_session(self, param=None, offset=None, scatter=False, spec=[], hm=True, corder=None, hlines=None,
                     riplines=False, cmap='hot', rate='mean', silent=False, axtitle=None, unit='frames', vmax=None):
        self.pltsessionplot = plt.subplots(4, 1, gridspec_kw={'height_ratios': [4, 1, 1, 1]}, sharex=True)
        fig, (axf, axr, axs, axp) = self.pltsessionplot
        param = self.getparam(param)
        if corder is None:
            corder = range(self.ca.cells)
        self.plotsession_corder = corder
        if hm:
            sortedhm = numpy.empty((len(corder), self.ca.frames))
            for i, c in enumerate(corder):
                if self.disc_param:
                    sortedhm[i] = param[c]
                else:
                    if numpy.nanmax(param[c]) > 0:
                        sortedhm[i] = param[c] / numpy.nanstd(param[c])
                    else:
                        sortedhm[i] = param[c]
            if not hasattr(self.ca, 'version_info'):
                sortedhm -= max(0, numpy.nanmin(sortedhm))
            elif self.ca.version_info['bsltype'] == 'original':
                sortedhm -= max(0, numpy.nanmin(sortedhm))
            if vmax is None:
                sortedhm /= numpy.nanmax(sortedhm)
                vmax = 0.9
            axf.imshow(sortedhm, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
            if hlines is not None:
                y = 0
                for i in hlines:
                    y += i
                    axf.axhline(y - 0.5, color='black', linewidth=0.5)
        else:
            if offset is None:
                offset = -numpy.nanstd(param)
            for i, c in enumerate(corder):
                if c in spec:
                    continue
                if scatter:
                    x = numpy.nonzero(param[c])
                    axf.scatter(x, [i * offset] * len(x[0]), marker="|")
                else:
                    axf.plot(param[c] + i * offset)
        if rate == 'spikes':
            r = numpy.copy(self.ca.rate)
        elif rate == 'mean':
            r = numpy.nanmean(param, axis=0)
        r[:5] = numpy.nanmean(r[5:10])
        axr.plot(r, color='black')
        for c in spec:
            axr.plot(param[c] / numpy.nanmax(param[c]), alpha=0.8 / len(spec))
        self.behavplot(axp)

        if unit == 'frames':
            axp.set_xlabel('Frame')
        else:
            try:
                unit = int(unit)
            except:
                unit = 60
            axp.set_xlabel('Seconds')
            t0 = int(self.frametotime(0))
            seconds = numpy.array(range(t0, int(self.frametotime(self.ca.frames - 1)), unit))
            axp.set_xticklabels(seconds - t0)
            axp.set_xticks([self.timetoframe(x) for x in seconds])

        # plot ripples if available:
        if hasattr(self, 'ripples'):
            if riplines:
                for t in self.ripple_frames:
                    axs.axvline(t - 0.5, color='black')
                    axr.axvline(t - 0.5, color='black')
                    axf.axvline(t - 0.5, color='black')
            if hasattr(self.ripples, 'theta'):
                rpsm = pandas.DataFrame(self.theta_power).ewm(span=3).mean()
                rpsm -= rpsm.min()
                rpsm /= rpsm.max()
                axs.plot(rpsm, label='ThetaPower', alpha=0.8)
            rpsm = pandas.DataFrame(self.ripple_power).ewm(span=3).mean()
            rpsm -= rpsm.min()
            rpsm /= rpsm.max()
            axs.plot(rpsm, label='RipplePower', alpha=0.8)

        # plot eye if available
        eyefn = self.prefix + '_eye.np'
        if os.path.exists(eyefn):
            eye = numpy.fromfile(eyefn, dtype=numpy.float32)
            if round(len(eye) / len(self.pos.smspd)) == 2:
                l = int(len(eye) / 2)
                eye1 = numpy.empty(l)
                for i in range(l):
                    eye1[i] = eye[i * 2] + eye[i * 2 + 1]
                eye = eye1 / 2
            eyesm = pandas.DataFrame(eye).ewm(span=15).mean()
            eyesm -= eyesm.min()
            eyesm /= eyesm.max()
            axs.plot(eyesm, label='Eye')
        if self.eye is not None:
            axs.plot(self.eye, color='blue', alpha=0.8, label='Eye')
        movfn = self.prefix + '_eye.move'
        if os.path.exists(movfn):
            eye = numpy.fromfile(movfn, dtype=numpy.float32)
            if round(len(eye) / len(self.pos.smspd)) == 2:
                l = int(len(eye) / 2)
                eye1 = numpy.empty(l)
                for i in range(l):
                    eye1[i] = eye[i * 2] + eye[i * 2 + 1]
                eye = eye1 / 2
            eye -= eye.min()
            eye /= eye.max()
            axs.plot(eye, label='AbsDiff', alpha=0.75)

        # plot opto if available
        if self.opto is not None:
            axs.plot(self.opto, color='dodgerblue', label='light', alpha=0.8)

        # put separator line for multiple groups:
        if self.tag is not None:
            if '+' in self.tag:
                for subtag in self.tag.split('+')[:-1]:
                    axf.axhline(len(LoadPolys(self.prefix, subtag).data) - 0.5, color='grey', linewidth=0.5)

        # plot speed
        speed_trim = copy.copy(self.pos.speed)
        speed_trim[:10] = 0
        if speed_trim.max() > 0:
            spdval = numpy.array(pandas.DataFrame(speed_trim).ewm(span=15).mean())
            spdval /= spdval.max()
        else:
            spdval = speed_trim
        axs.plot(spdval, label='Speed')
        for t in self.get_tones():
            axs.axvline(t, color='red')

        axs.legend(loc='upper right', framealpha=0.1)

        axf.yaxis.set_ticklabels([])
        axp.yaxis.set_ticklabels([])
        axs.yaxis.set_ticklabels([])
        axr.yaxis.set_ticklabels([])
        if axtitle is None:
            axtitle = u"\N{GREEK CAPITAL LETTER DELTA}" + 'F/F'
        axf.set_ylabel(axtitle)
        axp.set_ylabel('Position')
        axs.set_ylabel('Speed')
        axr.set_ylabel('Mean Firing')
        if not silent:
            fig.show()

            def button_press_callback(event):
                if event.inaxes == axf:
                    cell_index = self.plotsession_corder[int(event.ydata + 0.5)]
                    stamp = f'Cell:{cell_index}, Frame:{int(event.xdata - 0.5)}'
                    if hasattr(self, 'sync_spline'):
                        stamp += f', Time:{self.frametotime(event.xdata) - int(self.frametotime(0)):.2f}'
                    if hasattr(self, 'ripples'):
                        stamp += f', EPhys:{self.frametosample(int(event.xdata - 0.5))[0] / self.ripples.fs:.2f}'
                    print(stamp)

            fig.canvas.mpl_connect('button_press_event', button_press_callback)
        return fig

    def export_timeseries(self):
        pandas.DataFrame(self.ca.rate).ewm(span=15).mean().to_csv(self.prefix + '_mean_rates.csv')
        tdf = pandas.DataFrame(self.ca.rel.mean(axis=0)).ewm(span=15).mean()
        tdf['speed'] = self.pos.speed
        if hasattr(self, 'theta_power'):
            tdf['thetaPower'] = self.theta_power
        tdf.to_csv(self.prefix + '_mean_DFF.csv')

    def export_current_cell(self):
        pass

    def display_activity_image(self, cell, savingonly=False):
        im = self.display_event_image(numpy.nonzero(self.ca.peaks[cell])[0])
        r = max(abs(im.min()), abs(im.max()))
        plt.figure()
        plt.imshow(im, cmap='bwr', vmin=-r, vmax=r)
        plt.suptitle(self.prefix + 'Cell' + str(cell))
        if savingonly:
            plt.savefig('plots//' + self.prefix + '_Cell_' + str(cell) + '.png', dpi=300)
            plt.close()
        else:
            return im

    def display_event_image(self, peaks, windows=None, channel=0):
        im = numpy.zeros(self.rois.image.info['sz'])
        for frame in peaks:
            im += self.pull_img_ripple(frame, windows=windows, channel=channel)
        return im

    def show_ripple(self, frame):
        t0 = int(self.frametosample(frame)[-1] - self.ripples.fs / 2)
        t1 = t0 + self.ripples.fs
        fig, axes = self.pltsessionplot
        lines0 = [a.axvline(self.sampletoframe(t0), color='grey') for a in axes]
        lines1 = [a.axvline(self.sampletoframe(t1), color='grey') for a in axes]
        self.ripples.disp_ripples([lines0, lines1, self.sampletoframe], t0, t1)

    def pull_img_ripple(self, frame, windows=None, channel=0):
        # defines averaging widows
        if windows is None:
            windows = ((-10, 5), (0, 5))
        im = self.rois.image.data
        pre = im[frame + windows[0][0]:frame + windows[0][0] + windows[0][1], :, :, channel].mean(axis=0)
        pos = im[frame + windows[1][0]:frame + windows[1][0] + windows[1][1], :, :, channel].mean(axis=0)
        return pre - pos

    def show_cam(self, f):
        pass

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

    def ewma_smooth(self, period, norm=True):
        data = numpy.empty((self.ca.cells, self.ca.frames))
        for c in range(self.ca.cells):
            if norm:
                data[c, :] = pandas.DataFrame(self.ca.ntr[c]).ewm(span=period).mean()[0]
            else:
                data[c, :] = pandas.DataFrame(self.ca.rel[c]).ewm(span=period).mean()[0]
        return data

    def behavplot(self, ax):
        ax.plot(self.pos.pos, color='grey')
        if hasattr(self, 'bdat'):
            # add reward zones to plot
            rz = self.bdat.data[2]
            td = self.bdat.data[0]
            l = len(self.pos.pos)
            m = max(self.pos.pos)
            x = numpy.empty(l)
            for f in range(l):
                bf = numpy.where(td <= self.frametotime(f))[0]
                if len(bf) > 0 and rz[bf[-1]]:
                    x[f] = m * 100
                else:
                    x[f] = -m * 100
            ax.fill_between(range(l), -m * 100, x, color=self.colors.get('rewardzone', '#e2efda'))
            ax.plot(self.pos.pos, color='grey')
            # add licks to plot
            x, y = [], []
            for l in self.bdat.choices[0]:
                if l < self.bmtime[-1] and l > self.bmtime[0]:
                    t = self.timetoframe(l)
                    x.append(t)
                    y.append(self.pos.pos[t])
            ax.scatter(x, y, marker="o", color='green', s=50)
            x, y = [], []
            for l in self.bdat.choices[1]:
                if l < self.bmtime[-1] and l > self.bmtime[0]:
                    t = self.timetoframe(l)
                    if t < self.ca.frames:
                        x.append(t)
                        y.append(self.pos.pos[t])
            ax.scatter(x, y, marker="|", color='red', s=50)
            x, y = [], []
            for l in self.bdat.rewards:
                if l < self.bmtime[-1] and l > self.bmtime[0]:
                    t = self.timetoframe(l)
                    if t < self.ca.frames:
                        x.append(t)
                        y.append(self.pos.pos[t])
            ax.scatter(x, y, marker="|", color='blue', s=25)
            # plot extra pin info
            x1, x2, y1, y2 = [], [], [], []
            for pin, l, is_open in self.bdat.other_events:
                if l < self.bmtime[-1] and l > self.bmtime[0]:
                    t = self.timetoframe(l)
                    if t < self.ca.frames:
                        if is_open:
                            x1.append(t)
                            y1.append(self.pos.pos[t])
                        else:
                            x2.append(t)
                            y2.append(self.pos.pos[t])
            ax.scatter(x1, y1, marker="<", color='black', s=50)
            ax.scatter(x2, y2, marker=">", color='black', s=50)
            # add tones to plot
            tones = self.get_tones()
            for t in tones:
                ax.axvline(t, color='red')

            ax.set_ylim(-m * 0.1, m * 1.1)

    def calc_MI(self, param='nnd', bins=11, selection='movement', shuffle=30, ):
        param = self.getparam(param)
        self.mi = numpy.zeros(self.ca.cells)
        self.mi_z = numpy.zeros(self.ca.cells)
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
                    self.mi_z[c] = mis / numpy.std(shuff_mis)
                    self.statistical_is_placecell[c] = mis > numpy.nanpercentile(shuff_mis, 95)
                except:
                    self.mi[c] = numpy.nan
                    self.mi_z[c] = numpy.nan
                    print('MI exception', self.prefix, c)

    def MI_speed(self, param, bins=11, shuffle=30):
        param = self.getparam(param)
        self.mispeed = numpy.zeros(self.ca.cells)
        self.speedcorr = numpy.zeros(self.ca.cells)
        wh = range(8, self.ca.frames - 8)
        x = self.pos.smspd[wh]
        for c in range(self.ca.cells):
            if numpy.any(numpy.isnan(param[c])):
                self.mispeed[c] = numpy.nan
            else:
                y = param[c][wh]
                if numpy.count_nonzero(y) > bins:
                    y -= y.min()
                    y /= y.max()
                    try:
                        c_xy = numpy.histogram2d(x, y, bins)[0]
                        mis = mutual_info_score(None, None, contingency=c_xy)
                        shuff_mis = []
                        for _ in range(shuffle):
                            numpy.random.shuffle(y)
                            shuff_mis.append(
                                mutual_info_score(None, None, contingency=numpy.histogram2d(y, x, bins)[0]))
                        self.mispeed[c] = mis / numpy.mean(shuff_mis)
                    except:
                        self.mispeed[c] = numpy.nan
                        print('MI exception', self.prefix, c)
                    self.speedcorr[c] = stats.pearsonr(x, y)[0]

    def MImatrix(self, param, bins=11, saving=False):
        scores = numpy.zeros((self.ca.cells, self.ca.cells))
        wh = range(8, self.ca.frames - 8)
        for c1 in range(self.ca.cells):
            x = param[c1][wh]
            for c2 in range(c1):
                y = param[c2][wh]
                y /= y.max()
                try:
                    c_xy = numpy.histogram2d(x, y, bins)[0]
                    mis = mutual_info_score(None, None, contingency=c_xy)
                except:
                    mis = 0
                scores[c1, c2] = mis
                scores[c2, c1] = mis
        self.crossMI = scores
        rank = numpy.argsort(numpy.sum(-scores, axis=0))
        sortscores = numpy.zeros((self.ca.cells, self.ca.cells))
        for i in range(self.ca.cells):
            for j in range(i):
                sc = scores[rank[i], rank[j]]
                sortscores[i, j] = sc
                sortscores[j, i] = sc
        if saving:
            scores.tofile(self.prefix + 'MImatrix.np')
            plt.imshow(scores, cmap='inferno')
            plt.colorbar()
            plt.savefig(self.prefix + 'Mimatrix.png', dpi=600)
            plt.imshow(sortscores, cmap='inferno')
            plt.savefig(self.prefix + 'Mimatrix_sorted.png', dpi=600)
            plt.close('all')

    def frame_bins(self, bins):
        # 'lists of frames in each bin of relpos'
        bin = []
        binsize = 1.0 / bins
        for i in range(bins):
            bin.append(numpy.where((self.pos.relpos > i * binsize) * (self.pos.relpos < (i + 1) * binsize))[0])

    # def cell_bins(self, bins):
    #     '''list of cells with pf in each bin of relpos. has to be called after placefields,
    #     returns indices of original cells'''
    #     bin = []
    #     binsize = 1.0 / bins
    #     for i in range(bins):
    #         bin.append(numpy.where((self.args[0] > i * binsize) * (self.pos.relpos < (i + 1) * binsize))[0])

    def reldist(self, p1, p2):
        # distance of 2 relative belt positions
        a = min(p1, p2)
        b = max(p1, p2)
        d1 = b - a
        d2 = 1 - b + a
        return abs(min(d1, d2))

    def opp_pos(self, p1):
        '''return opposite position on belt'''
        p2 = p1 + 0.5
        if p2 > 1:
            p2 -= 1
        return p2

    def placefields_smooth(self, param='spikes', bins=50, silent=False, cmap='inferno', show=False, wh=None,
                           return_raw=False, exp_decay=2, corder=None, aspect='auto'):
        spikes = self.getparam(param)
        if wh is None:
            wh = numpy.where(self.pos.movement)[0][8:-8]
        binsize = 1.0 / bins
        self.placefield_properties = numpy.empty((self.ca.cells, 2))  # store contrast and width
        self.placefield_properties[:] = numpy.nan
        # create exponentially decaying weights for each frame based on distance from bin location
        # pull means for each cell and each bin using the exponential weights
        rates = numpy.empty((bins, self.ca.cells))
        for bi in range(bins):
            weights = numpy.empty(len(wh))
            for wi, t in enumerate(wh):
                d = 1 - self.reldist(bi * binsize, self.pos.relpos[t])
                if exp_decay is None:
                    weights[wi] = d
                else:
                    weights[wi] = numpy.e ** (1 - 1 / (d ** exp_decay))
            for c in range(self.ca.cells):
                rates[bi, c] = numpy.average(spikes[c, wh], weights=weights)
        self.smooth_zscores = numpy.zeros((bins, self.ca.cells))
        incl = []
        for c in range(self.ca.cells):
            x = rates[:, c]
            if x.max() > 0:
                self.smooth_zscores[:, c] = (x - numpy.nanmean(x)) / numpy.nanstd(x)
                incl.append(c)
                self.placefield_properties[c, 0] = x.max() - x.mean()  # contrast in firing rates
                pf_width = (self.smooth_zscores[:, c] > 1).sum() / bins  # number of >SD bins
                if pf_width == 0:
                    pf_width = numpy.nan
                self.placefield_properties[c, 1] = pf_width
        ni = 0
        self.corder_smoothpf = []
        if corder is None:
            self.smooth_sorted = numpy.empty((bins, len(incl)))
            for c in numpy.argsort(numpy.argmax(self.smooth_zscores, axis=0)):
                if c in incl:
                    self.smooth_sorted[:, ni] = self.smooth_zscores[:, c]
                    self.corder_smoothpf.append(c)
                    ni += 1
        else:
            self.smooth_sorted = numpy.empty((bins, len(corder)))
            for c in corder:
                self.smooth_sorted[:, ni] = self.smooth_zscores[:, c]
                self.corder_smoothpf.append(c)
                ni += 1
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

    def placefields(self, param=None, bins=50, ztresh=2, span=None, silent=False, cells=None, movement=True,
                    cmap='inferno', show=False, aspect='auto', corder_in=None):
        param = self.getparam(param)
        if span is None:
            span = 10, min(self.ca.frames, len(self.pos.pos)) - 10
        if cells is None:
            cells = range(self.ca.cells)
        # find time points for each bin
        self.bin = numpy.empty((bins, len(self.pos.relpos)), dtype='bool')
        binsize = 1.0 / bins
        for i in range(bins):
            self.bin[i] = (self.pos.relpos > i * binsize) * (self.pos.relpos < (i + 1) * binsize)
            if movement > 0:
                self.bin[i] *= self.pos.movement  ##Specify here if selective for movement
            elif movement < 0:
                self.bin[i] *= [not m for m in self.pos.movement]
        zscores = self.pull_means(param, span, cells=cells)
        self.pf_zscores = zscores
        # sort cells 0 position 1 max z 2 rank of cell
        self.args = numpy.empty((3, len(cells)))
        for c in range(len(cells)):
            x = zscores[:, c]
            self.args[:2, c] = numpy.argmax(x), x.max()

        # plotting strong z cells sorted by pos
        rest, ploti = [], 0
        sorted_fields = numpy.empty((bins, numpy.count_nonzero(self.args[1] > ztresh)))
        self.corder_pf = []
        for c in numpy.argsort(self.args[0]):
            if self.args[1, c] > ztresh:
                sorted_fields[:, ploti] = zscores[:, c]
                self.args[2, c] = ploti
                ploti += 1
                self.corder_pf.append(c)
            else:
                self.args[2, c] = -1

        # create per lap view
        nl = self.pos.laps[-1] + 1
        self.perlap_fields = numpy.zeros((bins, self.ca.cells, nl))
        if not numpy.any(numpy.isnan(param[c])):
            for l in range(nl):
                lap = numpy.where(self.pos.laps == l)[0]
                self.pull_means(param, (lap[0], lap[-1]))
                self.perlap_fields[:, :, l] = self.rates[self.pltnum]

        # display plot
        if not silent:
            plt.figure(self.pltnum)
            self.pltnum += 1
            plt.imshow(numpy.nan_to_num(sorted_fields.transpose()), cmap=cmap, interpolation='nearest', aspect=aspect)
            if show:
                plt.show()

    def get_place_cell_order(self, pf_calc_param='nnd', pf_calc_bins=25, loc=False, force=False):
        assert self.has_behavior
        pf_fn = self.path + self.prefix + f'_placecell_order-{pf_calc_param}-{pf_calc_bins}.npy'
        if os.path.exists(pf_fn) and not force:
            pc_order = numpy.load(pf_fn)
        else:
            self.calc_MI(pf_calc_param)
            pc = []
            nnd = numpy.nan_to_num(self.ca.nnd) > 0
            ppm = numpy.nan_to_num(self.ca.peaks) > 0
            for c in range(self.ca.cells):
                if self.statistical_is_placecell[c]:
                    if numpy.any(nnd[c]) and numpy.any(ppm[c]):
                        if numpy.nanpercentile(self.ca.ontr[c], 99) > 1:
                            pc.append(c)
            # save place field plot and loc of each place cell
            self.placefields_smooth(param=pf_calc_param, silent=True, bins=pf_calc_bins)
            locdat = numpy.argmax(self.smooth_zscores, axis=0)
            pc_order = [c for c in self.corder_smoothpf if c in pc]
            pc_loc = [locdat[c] for c in self.corder_smoothpf if c in pc]
            numpy.save(pf_fn, pc_order)
            numpy.save(pf_fn.replace('order', 'loc'), pc_loc)
        if loc:
            pc_loc = numpy.load(pf_fn.replace('order', 'loc'))
            return pc_order, pc_loc
        else:
            return pc_order

    def movement_traces(self, cell, param=None, span=None, duration=100):
        if param is None:
            param = self.ca.ntr
        if span is None:
            span = duration, len(self.pos.pos) - duration
        fig, (axstart, axstop) = plt.subplots(2, 1, sharex=True, sharey=True)
        # axstop.plot(self.pos.speed)
        # axstop.plot(self.pos.movement*25)
        if not hasattr(self, 'mov_stops'):
            # collect stops
            stops = []
            t1 = span[0]
            while t1 < span[1] - duration:
                t2 = int(t1 + duration / 2)
                t0 = int(t1 - duration / 2)
                if self.pos.movement[t1:t2].max() == 0 and self.pos.movement[t0:t1].min() == 1:
                    while self.pos.movement[t2 + 1] == 0 and t2 < t1 + duration:
                        t2 += 1
                    while self.pos.movement[t0 - 1] == 1 and t0 > t1 - duration:
                        t0 -= 1
                    loc = numpy.argmin(self.pos.speed[t1 - 7:t1 + 8]) - 7
                    t1 += loc
                    t0 = max(t0, t0 + loc)
                    t2 = min(t2, t2 + loc)
                    stops.append((t0, t1, t2))
                    while self.pos.movement[t2 + 1] == 0 and t2 < span[1]:
                        t2 += 1
                    t1 = t2
                else:
                    t1 += 1
            # collect starts
            starts = []
            t1 = span[0]
            while t1 < span[1] - duration:
                t2 = int(t1 + duration / 2)
                t0 = int(t1 - duration / 2)
                if self.pos.movement[t1:t2].min() == 1 and self.pos.movement[t0:t1].max() == 0:
                    while self.pos.movement[t2 + 1] == 1 and t2 < t1 + duration:
                        t2 += 1
                    while self.pos.movement[t0 - 1] == 0 and t0 > t1 - duration:
                        t0 -= 1
                    loc = numpy.argmax(self.pos.speed[t1 - 7:t1 + 8]) - 7
                    t1 += loc
                    t0 = max(t0, t0 + loc)
                    t2 = min(t2, t2 + loc)
                    starts.append((t0, t1, t2))
                    while self.pos.movement[t2 + 1] == 1 and t2 < span[1]:
                        t2 += 1
                    t1 = t2
                else:
                    t1 += 1
            self.mov_stops = stops
            self.mov_starts = starts
        else:
            stops = self.mov_stops
            starts = self.mov_starts
        # plot starts
        startmean = numpy.empty((duration * 2, len(starts)))
        startmean[:, :] = numpy.nan
        selstartmean = numpy.empty(startmean.shape)
        selstartmean[:, :] = numpy.nan
        for i, (t0, t1, t2) in enumerate(starts):
            x = range(t0 - t1, t2 - t1)
            y = param[cell, t0:t2]
            axstart.plot(x, y, color='black', alpha=0.4)
            startmean[duration + t0 - t1:t2 - t1 + duration, i] = y
            if self.ca.peaks[cell, t0:t2].max() > 0:
                selstartmean[duration + t0 - t1:t2 - t1 + duration, i] = y
        axstart.plot(range(-duration, duration), numpy.nanmean(startmean, axis=1), color='blue')
        # axstart.plot(range(-duration, duration), numpy.nanmean(selstartmean, axis=1), color='blue')
        # plot stops
        stopmean = numpy.empty((duration * 2, len(stops)))
        stopmean[:, :] = numpy.nan
        selstopmean = numpy.empty(stopmean.shape)
        selstopmean[:, :] = numpy.nan
        for i, (t0, t1, t2) in enumerate(stops):
            x = range(t0 - t1, t2 - t1)
            y = param[cell, t0:t2]
            axstop.plot(x, y, color='black', alpha=0.4)
            stopmean[duration + t0 - t1:t2 - t1 + duration, i] = y
            if self.ca.peaks[cell, t0:t2].max() > 0:
                selstopmean[duration + t0 - t1:t2 - t1 + duration, i] = y
        axstop.plot(range(-duration, duration), numpy.nanmean(stopmean, axis=1), color='blue')
        # axstop.plot(range(-duration, duration), numpy.nanmean(selstopmean, axis=1), color='blue')
        axstart.set_ylabel('Start')
        axstop.set_ylabel('Stop')
        axstart.axvline(0, color='green')
        axstop.axvline(0, color='red')
        plt.show()

    def replot(self, param, span=None, corr=False):
        param = self.getparam(param)
        if span is None:
            span = 10, self.ca.frames - 10
        bins = self.bin.shape[0]
        zscores = self.pull_means(param, span)
        sorted_fields = numpy.empty((bins, numpy.count_nonzero(self.args[2] > -1)))
        for c, p in enumerate(self.args[2]):
            if p > -1:
                sorted_fields[:, int(p)] = zscores[:, c]
        plt.figure(self.pltnum)
        self.pltnum += 1
        plt.imshow(numpy.nan_to_num(sorted_fields.transpose()), cmap='hot', interpolation='nearest')
        if not corr is False:
            self.corrs = numpy.empty((self.ca.cells))
            for c in range(self.ca.cells):
                self.corrs[c] = numpy.correlate(self.zscores[corr][:, c], zscores[:, c])
            nc = numpy.count_nonzero(self.corrs[numpy.where(self.args[2] > -1)] > 0)
            sorted_fields = numpy.zeros((bins * 2 + 1, nc))
            ploti = 0
            for c in numpy.argsort(self.args[0]):
                if self.args[2, c] > -1 and self.corrs[c] > 0:
                    sorted_fields[:bins, ploti] = zscores[:, c]
                    sorted_fields[bins + 1:, ploti] = self.zscores[corr][:, c]
                    ploti += 1
            sorted_fields[bins, :] = sorted_fields.min()

            plt.imshow(numpy.nan_to_num(sorted_fields.transpose()), cmap='hot', interpolation='nearest')

    def placefield(self, param, c, bins):
        plt.figure(self.pltnum)
        self.pltnum += 1
        l = min(len(param[c]), len(self.pos.movement))
        x = param[c, :l] * self.pos.movement[:l]
        m = max(x)
        plt.plot(x)
        plt.plot(-self.pos.relpos[:l] * m * 0.2, color='grey')
        for b in bins:
            plt.fill_between(range(l), 0, self.bin[b, :l] * m, color='#ddebf7')
            plt.fill_between(range(l), 0, self.bin[b, :l] * -self.pos.relpos[:l] * m * 0.2, color='#ddebf7')

    def runspeed(self, param=None, bins=5, binsize=5, span=None):
        param = self.getparam(param)
        if span is None:
            span = (0, self.ca.frames)
        # find time points for each bin
        self.bin = numpy.empty((bins + 1, len(self.pos.pos)), dtype='bool')
        self.bin[0] = numpy.invert(self.pos.movement)
        for i in range(bins):
            self.bin[i + 1] = (self.pos.speed > i * binsize) * (self.pos.speed < (i + 1) * binsize) * self.pos.movement
        zscores = self.pull_means(param, span)
        self.speedrates = copy.copy(self.rates[self.pltnum])

    def rippletrigger(self, param=None, ch=0):
        # plot averaged time course ripp triggered. using channel
        if param is None:
            param = self.getparam('ntr')
            if self.dualch:
                param = param[..., ch]
        length = 30
        nzr = []
        for r in self.ripples.events:
            t = self.sampletoframe(r.p[0])
            if t > 100 and t < self.ca.frames - 100:
                if not self.pos.movement[t]:
                    nzr.append(t)
        self.ripplerates = numpy.zeros((length, self.ca.cells))
        for c in range(self.ca.cells):
            brmean = numpy.zeros((len(nzr), length))
            for i, frame in enumerate(nzr):
                y = param[c][frame - int(length / 2):frame + int(length / 2)]
                brmean[i, :] = y
            brmean = numpy.nanmean(brmean, axis=0)
            brmean -= brmean.min()
            brmean /= brmean.max()
            self.ripplerates[:, c] = brmean

    def licktrigger(self, param=None):
        param = self.getparam(param)
        span = (100, self.ca.frames - 100)
        # will calc 'lick triggered average' for all licks, lick with reward and lick without reward. 'lick info is in secs from bdat.choices[correct,incorrect]
        self.lickrates = {}

        # add all licks
        if len(self.bdat.choices[0]) + len(self.bdat.choices[1]) > 3:
            self.bin = numpy.zeros((30, len(self.pos.pos)), dtype='bool')
            for l in self.bdat.choices[0]:
                l = self.timetoframe(l)
                for i in range(max(0, l - 15), min(self.ca.frames, l + 15)):
                    self.bin[i - l, i] = True
            for l in self.bdat.choices[1]:
                l = self.timetoframe(l)
                for i in range(max(0, l - 15), min(self.ca.frames, l + 15)):
                    self.bin[i - l, i] = True
            self.pull_means(param, span)
            self.lickrates['all'] = copy.copy(self.rates[self.pltnum])
        else:
            self.lickrates['all'] = numpy.empty((30, self.ca.cells))

        # add reward licks
        if len(self.bdat.choices[0]) > 3:
            self.bin = numpy.zeros((30, len(self.pos.pos)), dtype='bool')
            for l in self.bdat.choices[0]:
                l = self.timetoframe(l)
                for i in range(max(0, l - 15), min(self.ca.frames, l + 15)):
                    self.bin[i - l, i] = True
            self.pull_means(param, span)
            self.lickrates['reward'] = copy.copy(self.rates[self.pltnum])
        else:
            self.lickrates['reward'] = numpy.empty((30, self.ca.cells))

        # add no reward licks
        if len(self.bdat.choices[1]) > 3:
            self.bin = numpy.zeros((30, len(self.pos.pos)), dtype='bool')
            for l in self.bdat.choices[1]:
                l = self.timetoframe(l)
                for i in range(max(0, l - 15), min(self.ca.frames, l + 15)):
                    self.bin[i - l, i] = True
            self.pull_means(param, span)
            self.lickrates['no_reward'] = copy.copy(self.rates[self.pltnum])
        else:
            self.lickrates['no_reward'] = numpy.empty((30, self.ca.cells))

    def movetrigger(self, param=None, trsh=1):
        # scatterplot of peak times:
        param = self.getparam(param)
        delays, movingcells = [], []
        for c in range(self.ca.cells):
            if (numpy.nanmean(param[c] * self.pos.movement[:-1]) /
                numpy.nanmean(param[c] * numpy.invert(self.pos.movement[:-1]))) < trsh:
                continue
            cdel = []
            movingcells.append(c)
            for t in self.ca.event_times[c]:
                if self.pos.movement[t]:
                    t0 = t - 1
                    while self.pos.movement[t0]:
                        t0 -= 1
                    if t - t0 < 1000:
                        cdel.append(t - t0)
            delays.append(numpy.array(cdel))
        ds = []
        for d in delays:
            ds.append(numpy.median(d[numpy.where(d < 150)[0]]))
        x, y = [], []
        for i, c in enumerate(numpy.argsort(ds)):
            for t in delays[c]:
                x.append(t)
                y.append(i)
        plt.scatter(x, y, marker="|", s=10)
        plt.gca().invert_yaxis()
        # density plot of rates:
        mtr = numpy.zeros((len(movingcells), 1000))
        td = numpy.empty(self.pos.movement.shape)
        t0 = 0
        for t, m in enumerate(self.pos.movement):
            if not m:
                t0 = t
                td[t] = -1
            else:
                td[t] = t - t0
        wh = numpy.where(numpy.logical_and(td > 0, td < 1000))
        rnk, frn = [], numpy.array(range(1000))
        for i, c in enumerate(movingcells):
            for t in wh[0]:
                mtr[i, int(td[t])] += param[c, t]
            rnk.append(numpy.average(frn, weights=mtr[i]))
        srt = numpy.empty(mtr.shape)
        for i, c in enumerate(numpy.argsort(rnk)):
            srt[i] = mtr[c]
        plt.figure()
        self.tim = mtr, srt, rnk
        plt.imshow(srt, cmap='hot')

    def sync_events(self, data_only=False, trace_only=False, thr=None):
        # find percent of cells active:
        p_a = numpy.zeros(self.ca.frames)
        for i in range(self.ca.frames):
            p_a[i] = numpy.count_nonzero(self.ca.smtr[:, i] > 1)
        p_a /= self.ca.cells
        if trace_only:
            return p_a
        mean_immo = p_a[numpy.where(self.pos.gapless[:self.ca.frames])].mean()
        if thr is None:
            thr = mean_immo
        else:
            thr = mean_immo * thr
        # find frames where more cells are active and keep ratio of active cells
        imm = numpy.where(numpy.logical_not(self.pos.gapless[:self.ca.frames]))[0]
        peaks = []
        endpoint = len(p_a) - 11
        b_a = p_a > thr
        t = 10
        while t < endpoint:
            if b_a[t]:
                t1 = t + 1
                while b_a[t1] and t1 < endpoint:
                    t1 += 1
                if not any(self.pos.gapless[t - 10:t1 + 10]):
                    peak = t + numpy.argmax(p_a[t:t1])
                    peaks.append([peak, p_a[peak]])
                    t = t1
            t += 1
        if data_only:
            return peaks
        fig, ax = plt.subplots()
        ax.plot(p_a)
        if len(peaks) > 0:
            peaks = numpy.array(peaks)
            ax.scatter(peaks[:, 0], peaks[:, 1], color='red')
        ax.plot(self.pos.gapless * thr, color='orange')
        fig.savefig(self.prefix + '_synchronous_activity.png', dpi=240)
        plt.close()
        with open(self.prefix + '_synchronous_activity.txt', 'w') as f:
            s = 'Number Of Events\tFrequency of Events during immobility (event\minute)\tMean Amplitude (ratio of cells active)\n'
            if len(peaks) > 0:
                s += '\t'.join([str(x) for x in (
                    len(peaks), (len(peaks) / len(imm)) * (60 * self.ca.frames / self.frametotime(self.ca.frames)),
                    peaks[:, 1].mean())])
                s += '\nFrame\tAmplitude\n'
                for p in peaks:
                    s += '\t'.join([str(x) for x in p]) + '\n'
            else:
                s += '0\t0\tNo peaks'
            f.write(s)

    def get_hifreq(self, lo=80, hi=500):
        '''
        Return hf filetered power averaged in each frame
        :param lo:
        :param hi:
        :return:
        '''
        y = self.ephystrace
        pfn = self.prefix
        nframes = self.ca.frames
        hf_fn = self.path + self.prefix + f'_hf{lo}-{hi}.npy'
        if not os.path.exists(hf_fn):
            r = Ripples(None)
            r.lowcut = lo
            r.highcut = hi
            filt = r.gen_filt()
            ftr, envelope = r.run_filt(trace=y, filter=filt)

            hf_power = numpy.zeros(nframes)
            t0, f = 0, 0
            for t1, s in enumerate(self.ephysframes):
                if s > f:
                    if f < nframes:
                        hf_power[f] = envelope[t0:t1].mean()
                        t0 = t1
                        f += 1

            numpy.save(hf_fn, hf_power)
            return hf_power
        else:
            return numpy.load(hf_fn)

    def get_ripple_number_trace(self, period=15):
        rn_trace = numpy.zeros(self.ca.frames)
        rn_trace[self.ripple_frames] = 1
        return rn_trace  # pandas.DataFrame(rn_trace).ewm(span=period).mean()[0] * period

    def get_tones(self, pin=11):
        if self.has_behavior:
            return [self.timetoframe(t) for t in self.bdat.get_events(pin=pin)]
        else:
            return []
