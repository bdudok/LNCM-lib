import multiprocessing
from multiprocessing import Process, Queue
from scipy import signal
import shutil
from .Batch_Utils import outlier_indices
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Firing(object):
    def __init__(self, prefix, verbose=False, bsltype='poly', exclude=(0, 0), peakdet=True, ch=0, tag=None):
        if verbose:
            print(f'Firing called with ch={ch}, tag={tag}')
        self.peakdet = peakdet
        self.prefix = prefix
        self.ch = ch
        self.is_dual = False
        self.tag = tag
        self.pf = prefix + '.np'  # defaults to this
        # figure out folder name and number of channels. string channel names to be used only for loading, not for processing
        # ch is used to specify index in signal for processing, but used for name of saved folder when loading
        if tag is not None:
            pftag = '-' + tag
            dualpf = prefix + pftag + '-dual.np'
            if ch == 'Both' and not os.path.exists(dualpf):
                ch = 0
            if ch == 0 or ch == 'First':
                self.pf = prefix + pftag + '-ch0.np'
            elif ch == 1 or ch == 'Second':
                self.pf = prefix + pftag + '-ch1.np'
            elif ch == 'dual' or ch == 'Both' or ch == 'All':
                self.pf = dualpf
                self.is_dual = True
            else:
                raise ValueError(f'Cannot parse channel: {ch}')
        else:
            dualpf = self.prefix + '-dual.np'
            if ch == 'Both' and not os.path.exists(dualpf):
                ch = 0
            if ch == 0 or ch == 'First':
                self.pf = prefix + '.np'
                if not os.path.exists(self.pf):
                    self.pf = prefix + '-ch0.np'
            elif ch == 1 or ch == 'Second':
                self.pf = prefix + '-ch1.np'
            elif ch == 'dual' or ch == 'Both':
                self.pf = dualpf
                self.is_dual = True
            else:
                self.pf = prefix + '-' + ch + '.np'
        # the key 'event' is reserved in previous version and heavily used in analysis code. so leaving it in, be careful
        #  not to confuse with 'events' or local variable during peak detection
        print(self.pf)
        self.keys = ['bsl', 'rel', 'ntr', 'trace', 'spikes', 'rate', 'ontr', 'smtr', 'event', 'nnd']
        if peakdet:
            self.keys.extend(['b2', 'peaks'])
        self.verbose = verbose
        self.bsltype = bsltype
        self.exclude = exclude
        if verbose:
            print(prefix, 'firing init')
        self.version_info = {}  # should contain only single strings as values
        self.channels = 0
        self.cells, self.frames = None, None

    def open_trace(self):
        self.open_raw(trace=numpy.load(f'{self.prefix}_trace_{self.tag}.npy'))

    def open_raw(self, trace=None):
        # init containers and pool
        if trace is None:
            if self.tag is None:
                rfn = self.prefix + '_rigid.signals'
                nrfn = self.prefix + '_nonrigid.signals'
                if os.path.exists(nrfn):
                    f = loadmat(nrfn)
                elif os.path.exists(rfn):
                    f = loadmat(rfn)
                else:
                    return -1
                if self.verbose:
                    print('Loading trace')
                self.trace = f['sig'].transpose()
                self.spikes = f['spks'].transpose()
            else:
                file_name = f'{self.prefix}_trace_{self.tag}.npy'
                if os.path.exists(file_name):
                    self.trace = numpy.load(file_name)
                    self.spikes = numpy.zeros(self.trace.shape)
                else:
                    print('Missing file:', file_name)
                    return -1
        else:
            self.trace = trace
            self.spikes = numpy.zeros(self.trace.shape)
        # to work with 2 ch data:
        if len(self.trace.shape) == 3:
            self.channels = self.trace.shape[-1]
        elif len(self.trace.shape) == 2:
            self.channels = 1
        if self.channels < self.ch + 1:
            return -1  # stop if specified channel no is higher than channels in data
        if len(self.trace.shape) > 2:
            self.trace = self.trace[:, :, self.ch]
            self.spikes = self.spikes[:, :, self.ch]
        self.cells, self.frames = self.trace.shape[:2]
        if self.verbose:
            print(self.prefix, 'Traces loaded', str(self.cells))
        self.bsl = numpy.empty(self.trace.shape)
        self.ntr = numpy.empty(self.trace.shape)
        self.ontr = numpy.empty(self.trace.shape)
        self.smtr = numpy.empty(self.trace.shape)
        self.rel = numpy.empty(self.trace.shape)
        self.event = numpy.empty(self.trace.shape)
        self.nnd = numpy.empty(self.trace.shape)
        if self.peakdet:
            self.b2 = numpy.empty(self.trace.shape)
            self.peaks = numpy.empty(self.trace.shape)
            self.events = {}
        self.computed_cells = 0
        if os.path.exists(self.prefix + '_quadrature.mat'):
            self.movement = SplitQuad(self.prefix).gapless
        else:
            print('movement not found')
            self.movement = numpy.zeros(self.frames, dtype=numpy.bool)
        # save outlier data points to exclude from analysis:
        self.ol_index = outlier_indices(numpy.nanmean(self.trace, axis=0), thresh=12)

    def pack_data(self, c):
        return c, self.trace[c], self.bsltype, self.movement, self.exclude, self.peakdet, self.ol_index

    def unpack_data(self, data):
        c, data = data
        if self.verbose:
            print('Saving cell ' + str(c))
        for att in self.keys:
            if att not in ['rate', 'spikes']:
                self.__getattribute__(att)[c] = data[att]
        if self.peakdet:
            self.events[c] = data['events']
        self.computed_cells += 1
        if self.computed_cells == self.cells:
            if self.verbose:
                print('All cells received')
            self.save()
            return (True)

    def save(self):
        if os.path.exists(self.pf):
            shutil.rmtree(self.pf)
        os.mkdir(self.pf)
        if self.peakdet:
            s = self.peaks
        else:
            s = self.spikes
        self.rate = numpy.array(pandas.DataFrame(numpy.nanmean(s, 0)).ewm(span=5).mean())
        # catch nan cells
        # n = []
        # for c in range(self.cells):
        #     if numpy.isnan(self.ntr[c, 0]):
        #         n.append(c)
        #         for key in self.keys:
        #             if key not in ['rate', 'spikes']:
        #                 self.__getattribute__(key)[c, :] = 0
        # if len(n) > 0:
        #     print('Nan cells:', *[str(i) for i in n])
        for key in self.keys:
            numpy.save(self.pf + '//' + key, self.__getattribute__(key))
        self.version_info = {'v': '9', 'bsltype': self.bsltype,
                             'channels': str(self.channels)}  # should contain only single strings as values
        numpy.save(self.pf + '//' + 'info', self.version_info)
        # build and save event array of c, onset, start, peak, stop, decay from the events
        if self.peakdet:
            lengths = 0
            for c, es in self.events.items():
                lengths += len(es)
            eventarray = numpy.empty((lengths, 6), dtype='int')
            i = 0
            for c, es in self.events.items():
                for e in es:
                    eventarray[i, 0] = c
                    eventarray[i, 1:] = e
                    i += 1
            numpy.save(self.pf + '//eventarray', eventarray)
        print(self.prefix, self.cells, 'cells saved.')

    def load(self):
        if not os.path.exists(self.pf):
            if self.tag == None:
                self.tag = ''
            pf_try = [self.tag + '.np', self.tag + '-ch0.np', '1' + '.np', '1' + '-ch0.np', self.tag + 'dual.np',
                      '1-dual.np']
            for x in pf_try:
                pf = self.prefix + '-' + x
                if os.path.exists(pf):
                    self.pf = pf
                    print('Loading traces from', pf)
                break
        if not os.path.exists(self.pf):
            if self.tag not in ('skip', 'off'):
                raise FileNotFoundError(f'Make sure the .np folder exists: {self.pf}\nPath is: {os.getcwd()}')
            else:
                print('skipped loading ca')
                return -1
        for key in self.keys:
            fn = self.pf + '//' + key + '.npy'
            if os.path.exists(fn):
                self.__setattr__(key, numpy.load(fn))
            else:
                print(key, 'not available')
        if self.peakdet:
            fn = self.pf + '//eventarray.npy'
            if os.path.exists(fn):
                self.eventarray = numpy.load(fn)
        self.cells, self.frames = self.trace.shape[:2]
        if len(self.trace.shape) == 3:
            self.channels = self.trace.shape[-1]
        elif len(self.trace.shape) == 2:
            self.channels = 1
        vfn = self.pf + '//' + 'info' + '.npy'
        # load version info text into dict
        if os.path.exists(vfn):
            vtx = str(numpy.load(vfn, allow_pickle=True))
            for pair in vtx[1:-1].split(','):
                key, value = [x.strip()[1:-1] for x in pair.strip().split(':')]
                self.version_info[key] = value
        else:
            self.version_info = {'v': '<4', 'bsltype': 'original'}

    def repair_outliers(self):
        pass

    def get_npc(self, components=1):
        cells = []
        nnd = numpy.nan_to_num(self.nnd) > 0
        ppm = numpy.nan_to_num(self.peaks) > 0
        for c in range(self.cells):
            if numpy.any(nnd[c]) and numpy.any(ppm[c]):
                if numpy.nanpercentile(self.ntr[c], 99) > 1:
                    cells.append(c)
        fval = numpy.nan_to_num(self.rel[cells].transpose())
        fval = StandardScaler().fit_transform(fval)
        pca = PCA(n_components=components)
        return pca.fit_transform(fval)


class Worker(Process):
    def __init__(self, queue, res_queue, verbose=False):
        super(Worker, self).__init__()
        self.queue = queue
        self.res_queue = res_queue
        self.verbose = verbose

    def run(self):
        for data in iter(self.queue.get, None):
            c, data, bsltype, movement, exclude, peakdet, ol_index = data
            if self.verbose:
                print('Starting cell ' + str(c))
            t1, t2, frames = 50, 500, len(data)
            smw = numpy.empty(frames)
            bsl = numpy.empty(frames)
            rel = numpy.empty(frames)
            ntr = numpy.empty(frames)
            ontr = numpy.empty(frames)
            smtr = numpy.empty(frames)
            event = numpy.zeros(frames)
            oasis_r = numpy.zeros(frames)
            trim = min(20, int(frames / 50))
            data[0] = data[1:3].mean()  # modify first sample to make cells with zero value in first frame look OK.
            # fix outliers in trace:
            this_is_nancell = False
            nan_samples = []
            for t in ol_index:
                if any([t == 0, t - 1 in ol_index, t + 1 in ol_index, t == frames - 1]):
                    data[t] = numpy.nan
                    nan_samples.append(t)
                else:
                    data[t] = (data[t - 1] + data[t + 1]) / 2
            # skip cells that are translated to the edge and have zero trace
            for t in range(frames):
                ti0 = max(0, int(t - t1 * 0.5))
                ti1 = min(frames, int(t + t1 * 0.5) + 1)
                smw[t] = numpy.nanmean(data[ti0:ti1])
            if numpy.count_nonzero(numpy.isnan(smw)) > len(nan_samples):
                this_is_nancell = True
            if numpy.nanmax(smw) == 0:
                this_is_nancell = True
            if this_is_nancell:
                ntr[:] = numpy.nan
            else:
                if bsltype == 'poly':
                    # instead of original sliding window minimum, use a polynom fit for baseline
                    # iteratively remove negative peaks from the smoothed trace
                    for _ in range(5):
                        sneg = []
                        for chunk in range(trim, frames - trim, trim):
                            pks = signal.find_peaks_cwt(-smw[chunk:chunk + trim],
                                                        numpy.array([5, 10, 20, 50, 100]), min_length=10, min_snr=2)
                            sneg.extend(numpy.array(pks) + chunk)
                        if len(sneg) > 1:
                            nsmw = numpy.copy(smw)
                            for t in sneg:
                                start, stop = max(0, t - 100), min(frames, t + 100)
                                tc1, tc2 = numpy.nanargmax(smw[start:t]) + start, numpy.nanargmax(smw[t:stop]) + t
                                inds = list(numpy.arange(start, max(tc1, start + 15)))
                                inds.extend(numpy.arange(min(tc2, stop - 15), stop))
                                nsmw[tc1:tc2] = numpy.nanmean(smw[inds])
                            smw = nsmw
                        else:
                            break
                    # fit polynom (for periods outside movement
                    x = []
                    for t in numpy.arange(trim, frames - trim):
                        if not movement[t]:
                            # exclude excluded part
                            if not exclude[0] < t <= exclude[1]:
                                x.append(t)
                    y = numpy.nan_to_num(smw[x])
                    poly = numpy.polyfit(x, y, 3)
                    for i in range(frames):
                        bsl[i] = numpy.polyval(poly, i)
                    if numpy.any(bsl < 0):
                        print(f'Baseline flips 0 in cell {c}, running sliding minimum baseline')
                        bsltype = 'original'
                if bsltype in ['original', 'both']:  # both not implemented
                    for t in range(frames):
                        ti0, ti1 = max(0, t - t2), min(t, frames)
                        if ti0 < ti1:
                            minv = numpy.nanmin(smw[ti0:ti1])
                        else:
                            minv = numpy.nan_to_num(smw[ti0])
                        bsl[t] = minv
                rel = (data - bsl) / bsl
                ntr = rel / numpy.nanstd(rel)
                # additional iteration for more accurate noise levels
                ntr = numpy.nan_to_num(rel / numpy.nanstd(rel[numpy.where(ntr < 2)]))
                ontr = numpy.copy(ntr)
                # ewma
                smtr = numpy.array(pandas.DataFrame(ntr).ewm(span=15).mean()[0])
                # remove run periods if very neg
                excl_run = False
                if True in movement:
                    whr = numpy.where(movement[:frames])[0]
                    whi = numpy.where(movement[:frames] == 0)[0]
                    # check if any neg peak in the ntr
                    if numpy.argmax(smtr[trim:-trim] < -1) > 0:
                        # see if move is very negative
                        if numpy.nanmean(smtr[whr]) < -0.5:
                            ntr[whr] = numpy.nan
                            excl_run = True
                            # smtr[whr] = numpy.nan
                    # find peaks in bg subtracted signal, noise level based on sd (exclude nan values)
                    # find negative peaks to find noise level of detection (exclude nan values)
                    nperc = 100 * len(numpy.where(smtr < 0)[0]) / len(whi)
                else:
                    nperc = 100 * len(numpy.where(smtr < 0)[0]) / frames
                # deconvolve with oasis
                try:
                    from oasis.functions import deconvolve
                    oasis_r = deconvolve(ontr)[1]
                    if excl_run:
                        oasis_r[whr] = numpy.nan
                    oasis_r[nan_samples] = numpy.nan
                except:
                    print('Oasis spike deconvolution skipped.')
                nperc = max(10, nperc)
                nperc = min(90, nperc)
                if not excl_run:
                    s = signal.find_peaks_cwt(smtr, range(3, 60), min_length=3, min_snr=1, noise_perc=nperc)
                    sneg = signal.find_peaks_cwt(-smtr, range(3, 60), min_length=3, min_snr=1, noise_perc=nperc)
                else:
                    s = signal.find_peaks_cwt(smtr[whi], range(3, 60), min_length=2, min_snr=1, noise_perc=nperc)
                    s = whi[s]
                    sneg = signal.find_peaks_cwt(-smtr[whi], range(3, 60), min_length=2, min_snr=1, noise_perc=nperc)
                    sneg = whi[sneg]
                if len(sneg) > 0:
                    valneg = sorted(-smtr[sneg])
                    threshold = valneg[min(len(valneg) - 1, round(0.95 * len(valneg)))]
                else:
                    threshold = 0
                threshold = max(threshold, 1)
                for t in s:
                    if smtr[t] > threshold:
                        dur = []
                        # extend events and filter for duration
                        for i in range(t):
                            if smtr[t - i] > threshold:
                                dur.append(t - i)
                            else:
                                break
                        for i in range(1, frames - t):
                            if smtr[t + i] > threshold:
                                dur.append(t + i)
                            else:
                                break
                        if len(dur) > 9:
                            for i in dur:
                                event[i] = rel[i]
                # detect peaks------------------------------------------------------------------------------------------
                if peakdet:
                    s = smtr
                    shortwindow = 50
                    longwindow = 300
                    gap = 5
                    duration = 10
                    p1 = numpy.zeros(len(s))
                    p1[5:] = s[5:] - s[:-5]
                    p1 = numpy.maximum(p1, 0)
                    # sliding positive auc minus negative auc
                    l3 = numpy.zeros(frames)
                    for t in range(frames - longwindow):
                        t = int(t + longwindow / 2)
                        # positive auc in 50 over 20 percentile in 300: nice for local, not sensitive to bsl loc
                        l3[t] = numpy.maximum(0, s[t:t + shortwindow] -
                                              numpy.percentile(s[int(t - longwindow / 2):int(t + longwindow / 2)],
                                                               20)).sum() / shortwindow
                    b1 = numpy.logical_and(p1, l3 > 1)  # b1 looks good for peaks and excludes decays nicely
                    # fill gaps in b1 to allow burst stats later
                    gapless = numpy.copy(b1)
                    ready = False
                    while not ready:
                        ready = True
                        for t, m in enumerate(gapless):
                            if not m:
                                if numpy.any(gapless[t - gap:t]) and numpy.any(gapless[t:t + gap]):
                                    gapless[t] = 1
                                    ready = False
                    # chop off sections with decrease to split overlapping peaks. inverse, fast, centered version of l3
                    flong, fshort = 90, 15
                    l3if = numpy.zeros(frames)
                    for t in range(frames - flong):
                        t = int(t + flong / 2)
                        l3if[t] = numpy.maximum(0, -s[int(t - fshort / 2):int(t + fshort / 2)] -
                                                numpy.percentile(-s[int(t - flong / 2):int(t + flong / 2)],
                                                                 50)).sum() / fshort
                    b2 = numpy.logical_and(gapless, l3if < s)
                    # detect individual peaks
                    events = []
                    peaks = numpy.zeros(frames)
                    t = duration
                    offset = int(t + shortwindow / 2)
                    endpoint = frames - 8
                    while t < endpoint - gap:
                        # find next point which has at least a duration after
                        if numpy.all(b2[t:t + duration]) and t < endpoint - duration:
                            start = t
                            # go ahead while signal is over threshold
                            t1 = t + duration
                            while b2[t1] and t1 < endpoint:
                                t1 += 1
                            stop = t1
                            peak = start + numpy.argmax(s[start:stop])
                            # expand while l3 is in decay
                            if t1 < endpoint - offset:
                                while l3[t1] > l3[t1 + offset] and t1 < endpoint - offset:
                                    t1 += 1
                            decay = t1
                            # to find onset, go back until l3 is in decline, unless we hit the previous event
                            t0 = t
                            while l3[t0] > l3[t0 - offset] and not numpy.all(b2[t0 - duration: t0]) and t0 > duration:
                                t0 -= 1
                            # then find highest increase until peak
                            if peak - t0 < 2:
                                t0 -= 5
                            onset = t0 + numpy.argmax(p1[t0:peak]) - 1
                            events.append([onset, start, peak, stop, decay])
                            peaks[peak] = 1
                            t = stop + gap
                        t += 1
            result = {'bsl': bsl, 'rel': rel, 'ntr': ntr, 'trace': data, 'ontr': ontr, 'smtr': smtr, 'event': event,
                      'nnd': oasis_r}
            if peakdet:
                if this_is_nancell:
                    events = []
                    peaks = numpy.zeros(frames)
                    b2 = peaks
                result['peaks'] = peaks
                result['events'] = events
                result['b2'] = b2
            self.res_queue.put((c, result))


def proc_rel(trace, framerate=15.6):
    '''process a trace as in the Firing.rel
    input is cells x samples shape'''
    # fix outliers in trace:
    t1 = int(50 / 15.6 * framerate)
    trim = int(100 / 15.6 * framerate)
    t2 = 10 * t1
    this_is_nancell = False
    nan_samples = []
    ol_index = outlier_indices(numpy.nanmean(trace), thresh=12)
    frames = len(trace)
    smw = numpy.empty(frames)
    bsl = numpy.empty(frames)
    rel = numpy.empty(frames)
    data = trace
    for t in ol_index:
        if any([t == 0, t - 1 in ol_index, t + 1 in ol_index, t == frames - 1]):
            data[t] = numpy.nan
            nan_samples.append(t)
        else:
            data[t] = (data[t - 1] + data[t + 1]) / 2
    # skip cells that are translated to the edge and have zero trace
    for t in range(frames):
        ti0 = max(0, int(t - t1 * 0.5))
        ti1 = min(frames, int(t + t1 * 0.5) + 1)
        smw[t] = numpy.nanmean(data[ti0:ti1])
    if numpy.count_nonzero(numpy.isnan(smw)) > len(nan_samples):
        this_is_nancell = True
    if numpy.nanmax(smw) == 0:
        this_is_nancell = True
    # instead of original sliding window minimum, use a polynom fit for baseline
    # iteratively remove negative peaks from the smoothed trace
    for _ in range(5):
        sneg = []
        for chunk in range(trim, frames - trim, trim):
            pks = signal.find_peaks_cwt(-smw[chunk:chunk + trim],
                                        numpy.array([5, 10, 20, 50, 100]), min_length=10, min_snr=2)
            sneg.extend(numpy.array(pks) + chunk)
        if len(sneg) > 1:
            nsmw = numpy.copy(smw)
            for t in sneg:
                start, stop = max(0, t - 100), min(frames, t + 100)
                tc1, tc2 = numpy.nanargmax(smw[start:t]) + start, numpy.nanargmax(smw[t:stop]) + t
                inds = list(numpy.arange(start, max(tc1, start + 15)))
                inds.extend(numpy.arange(min(tc2, stop - 15), stop))
                nsmw[tc1:tc2] = numpy.nanmean(smw[inds])
            smw = nsmw
        else:
            break
    # fit polynom (for periods outside movement
    x = numpy.arange(trim, frames - trim)
    y = numpy.nan_to_num(smw[x])
    poly = numpy.polyfit(x, y, 3)
    for i in range(frames):
        bsl[i] = numpy.polyval(poly, i)
    if numpy.any(bsl < 0):
        print(f'Baseline flips 0, skipping')
        this_is_nancell = True
    rel = (data - bsl) / bsl
    if this_is_nancell:
        rel[:] = numpy.nan
    return rel


if __name__ == '__main__':
    # process trace created by PullSignals:
    path = 'G://Barna//axax//'
    os.chdir(path)
    prefix = 'axax_124_230'
    tr = numpy.load(prefix + '.rawtrace.npy')
    request_queue = Queue()
    result_queue = Queue()
    nworker = 0
    ncpu = multiprocessing.cpu_count()
    a = Firing(prefix, verbose=True, ch=0, tag='PullSig')
    a.open_raw(trace=tr)
    for c in range(a.cells):
        if nworker < ncpu:
            Worker(request_queue, result_queue).start()
            nworker += 1
        request_queue.put(a.pack_data(c))
    for data in iter(result_queue.get, None):
        finished = a.unpack_data(data)
        if finished:
            break

    # process trace created by sbxsegmenttool
    # path = 'G://Barna//axax//'
    # os.chdir(path)
    # files = ['cckcre_2_063_291']
    # request_queue = Queue()
    # result_queue = Queue()
    # nworker = 0
    # ncpu = multiprocessing.cpu_count()
    # for t in files:
    #     print(t)
    #     a = Firing(t, verbose=True)
    #     a.open_raw()
    #     for c in range(a.cells):
    #         if nworker < ncpu:
    #             Worker(request_queue, result_queue).start()
    #             nworker += 1
    #         request_queue.put(a.pack_data(c))
    #     for data in iter(result_queue.get, None):
    #         finished = a.unpack_data(data)
    #         if finished:
    #             break
