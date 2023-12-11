import numpy
import multiprocessing
from multiprocessing import Process, Queue
import os
from scipy import signal
import pandas
import shutil
from Proc2P.Bruker.SyncTools import Sync
from Proc2P.utils import gapless, outlier_indices
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Proc2P.Bruker.ConfigVars import CF
from matplotlib import pyplot as plt


class CaTrace(object):
    def __init__(self, path, prefix, verbose=False, bsltype='poly', exclude=(0, 0), peakdet=False, ch=0, tag=None):
        if verbose:
            print(f'Firing called with ch={ch}, tag={tag}')
        self.peakdet = peakdet  # removed peakdet from old Firing class, this doesn't do anything.
        self.path = path
        self.prefix = prefix
        self.opPath = os.path.join(self.path, self.prefix + '/')
        self.is_dual = False
        # parse channel
        if ch in (0, 'Ch2', 'First', 'Green'):
            ch = 0
        elif ch in (1, 'Ch1', 'Second', 'Red'):
            ch = 1
        elif ch in ('Both', 'All', 'Dual'):
            ch = 0
            self.is_dual = True
        else:
            print('Channel not parsed:', ch)
            assert False
        self.ch = ch
        self.tag = tag
        self.pf = self.opPath + f'{prefix}_trace_{tag}-ch{ch}'  # folder name
        self.sync = Sync(path, prefix)
        self.keys = ['bsl', 'rel', 'ntr', 'trace', 'smtr', ]
        self.verbose = verbose
        self.bsltype = bsltype
        self.exclude = exclude
        if verbose:
            print(prefix, 'firing init')
        self.version_info = {}  # should contain only single strings as values

    # def open_trace(self):
    #     self.open_raw(trace=numpy.load(self.pf))

    def open_raw(self, trace=None):
        # init containers and pool
        if trace is not None:
            self.trace = trace
        else:
            file_name = self.opPath + f'{self.prefix}_trace_{self.tag}.npy'
            if os.path.exists(file_name):
                self.trace = numpy.load(file_name)
            else:
                print('Missing file:', file_name)
                return -1
        if len(self.trace.shape) == 3:
            # print(self.ch, self.trace.shape)
            if not self.ch < self.trace.shape[2]:
                return -1
            self.trace = self.trace[..., self.ch]
        elif self.ch > 0:
            return -1
        self.cells, self.frames = self.trace.shape
        if self.verbose:
            print(self.prefix, 'Traces loaded', str(self.cells))
        self.bsl = numpy.empty(self.trace.shape)
        self.ntr = numpy.empty(self.trace.shape)
        self.smtr = numpy.empty(self.trace.shape)
        self.rel = numpy.empty(self.trace.shape)
        self.computed_cells = 0
        self.movement = gapless(self.sync.load('speed'))
        # else:
        #     print('movement not found')
        #     self.movement = numpy.zeros(self.frames, dtype=numpy.bool)
        # save outlier data points to exclude from analysis:
        # mask with bad frames
        self.ol_index = []
        bad_frames = self.sync.load('opto')
        if bad_frames is not None:
            self.ol_index.extend(bad_frames)
        self.ol_index.extend(outlier_indices(numpy.nanmean(self.trace, axis=0), thresh=12))

    def pack_data(self, c):
        return c, self.trace[c], self.bsltype, self.movement, self.exclude, self.peakdet, self.ol_index

    def unpack_data(self, data):
        channel = 0
        c, data = data
        if self.verbose:
            print('Saving cell ' + str(c))
        for att in self.keys:
            # print('saving', att, self.__getattribute__(att).shape, data[att].shape)
            self.__getattribute__(att)[c] = data[att]

        self.computed_cells += 1
        if self.computed_cells == self.cells:
            if self.verbose:
                print('All cells received')
            self.save()
            return True

    def save(self):
        if os.path.exists(self.pf):
            shutil.rmtree(self.pf)
        os.mkdir(self.pf)
        for key in self.keys:
            numpy.save(self.pf + '//' + key, self.__getattribute__(key))
        self.version_info = {'v': '11', 'bsltype': self.bsltype,
                             'channel': str(self.ch),
                             'fps': str(CF.fps)}  # should contain only single strings as values
        numpy.save(self.pf + '//' + 'info', self.version_info)
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
        self.cells, self.frames = self.trace.shape[:2]
        if len(self.trace.shape) == 3:
            self.channels = self.trace.shape[-1]
            self.is_dual = self.channels > 1
        elif len(self.trace.shape) == 2:
            self.channels = 1
        # keeping the version above for backwards compatibility with 'dual' np folders.
        # if the existing data is dual, we're good. if not, read second single and append.
        if self.channels == 1 and self.is_dual:
            second_folder = self.opPath + f'{self.prefix}_trace_{self.tag}-ch{1}'  # folder name
            found_second = False
            for key in self.keys:
                fn = second_folder + '//' + key + '.npy'
                if os.path.exists(fn):
                    found_second = True
                    tr1 = self.__getattribute__(key)
                    tr2 = numpy.load(fn)
                    self.__setattr__(key, numpy.dstack((tr1, tr2)))
            if not found_second:
                self.is_dual = False
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
        for c in range(self.cells):
            if numpy.nanpercentile(self.ntr[c], 99) > 3:
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
            smtr = numpy.empty(frames)
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
                smtr = numpy.array(pandas.DataFrame(ntr).ewm(span=CF.fps).mean()[0])

            result = {'trace': data, 'bsl': bsl, 'rel': rel, 'ntr': ntr, 'smtr': smtr}
            self.res_queue.put((c, result))


if __name__ == '__main__':
    # test dual - load processed
    wdir = 'D:/Shares/Data/_Processed/2P/testing/'
    prefix = 'SncgTot4_2023-10-23_movie_000'
    tag = '1'
    # b = CaTrace(wdir, prefix, verbose=True, ch=0, tag=tag)
    # c = CaTrace(wdir, prefix, verbose=True, ch=1, tag=tag)
    # b.open_raw()
    # c.open_raw()
    # print(b.pack_data(0)[1][:2], c.pack_data(0)[1][:2])
    a = CaTrace(wdir, prefix, verbose=True, ch='Dual', tag=tag)
    self = a
    a.load()
    # a.open_raw()
    # for c in range(a.cells):
    #     if nworker < ncpu * 0.8:
    #         Worker(request_queue, result_queue).start()
    #         nworker += 1
    #     request_queue.put(a.pack_data(c))
    # for data in iter(result_queue.get, None):
    #     finished = a.unpack_data(data)
    #     if finished:
    #         break
