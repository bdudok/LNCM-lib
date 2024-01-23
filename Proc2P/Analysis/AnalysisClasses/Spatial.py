from matplotlib.patches import Polygon
from Proc2P.Bruker.ImagingSession import ImagingSession
import scipy
from sklearn.metrics import mutual_info_score
from sklearn import cluster
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter
from CommonFunc import ewma



def match_cells(a: ImagingSession, b: ImagingSession, cells: list):
    '''
    takes a list of cells in a, returns the indices of the same cells from b
    robust to missing cells and mismatched indices.
    '''
    cms = [(a.rois.polys[pi].min(axis=0) + a.rois.polys[pi].max(axis=0)) / 2 for pi in cells]
    target_cms = [(b.rois.polys[pi].min(axis=0) + b.rois.polys[pi].max(axis=0)) / 2 for pi in range(b.ca.cells)]
    targets = scipy.spatial.cKDTree(target_cms)
    found = numpy.zeros(len(cells), dtype='bool')
    used = numpy.zeros(b.ca.cells, dtype='bool')
    polys = {}
    matches = numpy.empty(len(cells), dtype='int')
    # best case: index matches
    for ci, c in enumerate(cells):
        if c >= b.ca.cells:
            continue
        if c not in polys:
            polys[c] = Polygon(b.rois.polys[c])
        if polys[c].contains_point(cms[ci]):
            matches[ci] = c
            found[ci] = True
            used[c] = True
    f1 = found.sum()
    print(f'Round 1: {f1}')
    # second: use closest
    if found.sum() < len(cells):
        for ci, c in enumerate(cells):
            if found[ci]:
                continue
            # nearest cm
            d, nn = targets.query(cms[ci])
            if used[nn]:
                continue
            if nn not in polys:
                polys[nn] = Polygon(b.rois.polys[nn])
            if polys[nn].contains_point(cms[ci]):
                matches[ci] = nn
                found[ci] = True
                used[nn] = True
    f2 = found.sum()
    print(f'Round 2: {f2 - f1}, total:{f2}')
    # third: progressively rebuild tree to find any that matches
    if found.sum() < len(cells):
        rebuild = True
        for ci, c in enumerate(cells):
            if found[ci]:
                continue
            # rebuild tree from remaining cells
            if rebuild:
                targets = scipy.spatial.cKDTree([target_cms[pi] for pi in range(b.ca.cells) if not used[pi]])
                rebuild = False
            # nearest cm
            d, nn = targets.query(cms[ci])
            if d > 20:
                continue
            if nn not in polys:
                polys[nn] = Polygon(b.rois.polys[nn])
            if polys[nn].contains_point(cms[ci]):
                matches[ci] = nn
                found[ci] = True
                used[nn] = True
                rebuild = True
    f3 = found.sum()
    print(f'Round 3: {f3 - f2}, total:{f3}')
    return numpy.array(cells)[found], matches[found]


def exclude_overlap(a: ImagingSession, b: ImagingSession, cells: list, dmax=20):
    '''
    takes a list of cells in a, returns the subset of these that are not overlapping with any cells in b.
    '''
    cms = [(a.rois.polys[pi].min(axis=0) + a.rois.polys[pi].max(axis=0)) / 2 for pi in cells]
    target_cms = [(b.rois.polys[pi].min(axis=0) + b.rois.polys[pi].max(axis=0)) / 2 for pi in range(b.ca.cells)]
    targets = scipy.spatial.cKDTree(target_cms)
    incl = []
    for ci, c in enumerate(cells):
        d, nn = targets.query(cms[ci])
        if d > dmax:
            incl.append(c)
    return numpy.array(incl)


class Spatial:
    '''Compute and store place cell-specific data of ImagingSession'''

    def __init__(self):
        self.path: str = None
        self.prefix: str = None
        self.tag: str = None
        self.session: ImagingSession = None
        self.force_mode = False
        self.hash = None
        self.setname = None
        self.cells: numpy.ndarray = None
        self.loc: numpy.ndarray = None
        self.bins = 25
        self.resolution = 200  # divide lap into this many. 200 = 1 cm resolution with 2 m lap
        self.pos = None
        self.cache = {}

    def set_bins(self, bins):
        self.bins = bins

    def init_session(self, session: ImagingSession):
        self.path = session.path
        self.prefix = session.prefix
        self.tag = session.tag
        self.session = session
        self.getdir()
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

    def init_files(self, path, prefix, tag):
        self.path = path
        self.prefix = prefix
        self.tag = tag
        self.getdir()

    def getdir(self):
        self.dir = f'{self.path + self.prefix}-{self.tag}-Spatial/'

    def set_PCs(self, cells, loc=None):
        self.cells = numpy.array(cells)
        self.hash = hash(self.cells.tostring())
        if loc is not None:
            if loc == 'pull':
                loc = self.locmax(self.resolution)
            self.loc = numpy.array(loc)

    def auto_PC(self, bins=25):
        pc, loc = self.session.get_place_cell_order(pf_calc_bins=bins, loc=True)
        self.set_PCs(pc, loc)

    def det_PC(self):
        assert self.session is not None
        # pick place cells
        bins = 9
        shuffle = 30
        a = self.session
        wh = numpy.where(a.pos.gapless[100:-100])[0] + 100
        x = a.pos.pos[wh]
        x = numpy.maximum(0, x)
        x = numpy.minimum(1, x / numpy.percentile(x, 99))
        statistical_is_placecell = numpy.zeros(a.ca.cells, dtype='bool')
        for c in range(a.ca.cells):
            y = numpy.nan_to_num(a.ca.nnd[c][wh])
            if y.sum() < 3:
                continue
            p = numpy.percentile(y, 99)
            if p < 1:
                continue
            y = numpy.minimum(1, y / p)
            c_xy = numpy.histogram2d(x, y, bins)[0]
            mis = mutual_info_score(None, None, contingency=c_xy)
            shuff_mis = []
            for _ in range(shuffle):
                numpy.random.shuffle(y)
                shuff_mis.append(mutual_info_score(None, None, contingency=numpy.histogram2d(x, y, bins)[0]))
            statistical_is_placecell[c] = mis > numpy.nanpercentile(shuff_mis, 95)
        # measure location dependent activity
        incl = numpy.where(statistical_is_placecell)[0]
        rates = numpy.empty((bins, len(incl)))
        binsize = 1.0 / bins
        for bi in range(bins):
            weights = numpy.empty(len(wh))
            for wi, t in enumerate(wh):
                d = 1 - a.reldist(bi * binsize, x[wi])
                weights[wi] = numpy.e ** (1 - 1 / (d ** 2))
            for ci, c in enumerate(incl):
                rates[bi, ci] = numpy.average(a.ca.rel[c, wh], weights=weights)
        locs = numpy.argmax(rates, axis=0)
        corder = numpy.argsort(locs)
        self.set_PCs(incl[corder], locs[corder])

    def save_named_set(self, tag):
        assert self.cells is not None
        nfn = f'{self.prefix}-{self.tag}-{tag}'
        if os.path.exists(self.dir + nfn + '.hash'):
            if not self.force_mode:
                raise FileExistsError('tag exists. set force mode to overwrite')
            else:
                print(self.prefix, 'Overwriting content for', tag)
                for f in os.listdir(self.dir):
                    if nfn in f:
                        os.remove(self.dir + f)
        with open(self.dir + nfn + '.hash', 'w') as f:
            f.write(str(self.hash))
        hfn = self.dir + nfn + f'_{self.hash}'
        numpy.save(hfn + '.cells', self.cells)
        if self.loc is not None:
            numpy.save(hfn + '.locs', self.loc)
        self.setname = tag

    def load_named_set(self, tag):
        nfn = f'{self.prefix}-{self.tag}-{tag}'
        hfn = self.dir + nfn + '.hash'
        if not os.path.exists(hfn):
            return -1
        with open(hfn, 'r') as f:
            self.hash = f.read()
        hfn = self.dir + nfn + f'_{self.hash}'
        self.cells = numpy.load(hfn + '.cells.npy')
        lfn = hfn + '.locs.npy'
        if os.path.exists(lfn):
            self.loc = numpy.load(lfn)
        self.setname = tag

    def set_pos(self, x=None):
        '''store or load position. norm. 0-1, for each frame'''
        pfn = self.dir + self.prefix + '.pos.npy'
        if os.path.exists(pfn):
            self.pos = numpy.load(pfn)
        else:
            if x == 'auto':
                x = self.session.pos.pos
                x = numpy.maximum(0, x)
                x = numpy.minimum(1, x / numpy.percentile(x, 99))
            self.pos = x
            numpy.save(pfn, x)

    def locmax(self, bins):
        '''hires estimation of max location with weighted distances'''
        rates = numpy.empty((bins, len(self.cells)))
        wh = numpy.where(self.session.pos.gapless[100:-100])[0] + 100
        x = self.pos[wh]
        binsize = 1.0 / bins
        for bi in range(bins):
            weights = numpy.empty(len(wh))
            for wi, t in enumerate(wh):
                d = 1 - self.session.reldist(bi * binsize, x[wi])
                weights[wi] = numpy.e ** (1 - 1 / (d ** 8))
            for ci, c in enumerate(self.cells):
                y = self.session.ca.rel[c, wh]
                incl = numpy.logical_not(numpy.isnan(y))
                rates[bi, ci] = numpy.average(y[incl], weights=weights[incl])
        return numpy.argmax(rates, axis=0)

    def reldist_array(self):
        nfn = f'{self.prefix}-{self.tag}-{self.setname}'
        hfn = self.dir + nfn + f'_{self.hash}'
        rdfn = hfn + '.reldists.npy'
        if os.path.exists(rdfn):
            rd = numpy.load(rdfn)
        else:
            rd = numpy.empty((len(self.cells), len(self.pos),))
            # construct a LUT of cell loc, mouse pos and abs dist. (x: cell, y: mouse)
            lut = self.get_lut()
            # for each cell and frame, store lut value
            for ci, x in enumerate(self.loc):
                for yi, yr in enumerate(self.pos):
                    rd[ci, yi] = lut[x, int(yr * self.resolution)]
            numpy.save(rdfn, rd)
        self.cache['rd'] = rd

    def get_lut(self):
        lfn = f'{self.dir + self.prefix}-{self.resolution}_distances.lut.npy'
        if os.path.exists(lfn):
            lut = numpy.load(lfn)
        else:
            lut = numpy.empty((self.resolution, self.resolution + 1))
            for xi, x in enumerate(numpy.linspace(0, 1, self.resolution)):
                for yi, y in enumerate(numpy.linspace(0, 1, self.resolution + 1)):
                    # reldist
                    lut[xi, yi] = self.signeddist(x, y)
            numpy.save(lfn, lut)
        return lut

    @staticmethod
    def signeddist(x, y):
        a = min(x, y)
        b = max(x, y)
        d1 = b - a
        d2 = 1 - b + a
        d = abs(min(d1, d2))
        # sign
        if y > x:
            d3 = y - x
        else:
            d3 = 1 - x + y
        if d3 > 0.5:
            d *= -1
        return d

    @staticmethod
    def calc_MI(x, y, bins=9, shuffle=30, incl=None, ret_bool=False):
        '''
        :param x: dependent variable (e.g. position, speed)
        :param y: trace from cell
        :param bins: number of bins
        :param shuffle: iterations for normalizing
        :param incl: None for all, otherwise bool same shape as x and y
        :return: MI normalized to expected
        '''
        if incl is None:
            incl = numpy.ones(len(x), dtype='bool')
            incl[:100] = 0
            incl[-100:] = 0
        incl = numpy.logical_and(incl, numpy.logical_not(numpy.isnan(x)))
        incl = numpy.logical_and(incl, numpy.logical_not(numpy.isnan(y)))
        wh = numpy.where(incl)[0]
        x = x[wh]
        x = numpy.maximum(0, x)
        x = numpy.minimum(1, x / numpy.percentile(x, 99))
        y = y[wh]
        if y.sum() < 3:
            return numpy.nan
        y = numpy.minimum(1, y / numpy.percentile(y, 99))
        c_xy = numpy.histogram2d(x, y, bins)[0]
        mis = mutual_info_score(None, None, contingency=c_xy)
        shuff_mis = numpy.empty(shuffle)
        shuff_mis[:] = numpy.nan
        for i in range(shuffle):
            numpy.random.shuffle(y)
            shuff_mis[i] = mutual_info_score(None, None, contingency=numpy.histogram2d(x, y, bins)[0])
        if ret_bool:
            return mis > numpy.nanpercentile(shuff_mis, 95)
        return mis / numpy.nanmean(shuff_mis)

    def pull_mean(self, param='rel', res=None, pull_session=None, pull_cells=None, save_tag=None, nan_policy='omit',
                  where='run', overwrite=False, trim=100, norm_trace=False):
        '''distance dependent mean activity for all cells. Leave session and cells None to pull on current.
        pass session and index list to pull from matched cells in a second ImagingSession'''
        if res is None:
            res = self.resolution
        if pull_session is None:
            pull_session = self.session
        if pull_cells is None:
            pull_cells = self.cells
        if save_tag is None:
            save_tag = 'self'
        nfn = f'{self.prefix}-{self.tag}-{self.setname}'
        hfn = self.dir + nfn + f'_{self.hash}'
        if trim == 100:
            rdfn = hfn + f'.binnedsignals-{param}-{res}-{save_tag}-{where}.npy'
        else:
            rdfn = hfn + f'.binnedsignals-{param}-{res}-{save_tag}-{where}-{trim}.npy'
        if norm_trace:
            rdfn = rdfn.replace('.npy', '-norm.npy')
        if os.path.exists(rdfn) and not overwrite:
            print('Loading', rdfn)
            ba = numpy.load(rdfn)
        else:
            print(self.prefix, rdfn, nfn, hfn)
            Y = pull_session.getparam(param)
            rd = self.cache['rd']

            if where is None:
                rdfn = hfn + f'.binnedsignals-{param}-{res}-{save_tag}.npy'
            else:
                if where == 'run':
                    wh = numpy.where(self.session.pos.gapless[trim:-trim])[0] + trim
                elif where == 'stop':
                    starts, stops = self.session.startstop()
                    wh = []
                    for t in stops:
                        wh.extend(numpy.arange(t + int(fps), min(t + int(5 * fps), self.session.ca.frames - 100)))
                    wh = numpy.array(wh)
                elif where == 'immo':
                    wh = numpy.where(numpy.logical_not(self.session.pos.gapless[trim:-trim]))[0] + trim
                else:
                    wh = numpy.copy(where)
                    where = 'man'

            ba = numpy.empty((len(pull_cells), res))
            bins = numpy.linspace(-0.5, 0.5, res + 1)
            for ci, c in enumerate(pull_cells):
                xdat = rd[ci, wh]
                ydat = Y[c, wh]
                if nan_policy == 'omit':
                    incldat = numpy.logical_not(numpy.isnan(ydat))
                    xdat = xdat[incldat]
                    ydat = ydat[incldat]
                if nan_policy == 'tozero':
                    ydat = numpy.nan_to_num(ydat)
                if norm_trace:
                    ymeas = gaussian_filter(ydat, sigma=15)
                    ymeas -= numpy.percentile(ymeas, 50)
                    ymeas /= numpy.percentile(ymeas, 95)
                    ymeas = numpy.maximum(0, ymeas)
                    ymeas = numpy.minimum(1, ymeas)
                else:
                    ymeas = ydat
                binned, edges, _ = binned_statistic(xdat, ymeas, 'mean', bins=bins)
                ba[ci] = binned
            numpy.save(rdfn, ba)
        self.cache['ba-' + save_tag] = ba

    def get_event_mask(self, w, save_tag=None, overwrite=False):
        '''masks for each event, and distance form preferred loc'''
        if save_tag is None:
            save_tag = self.setname
        nfn = f'{self.prefix}-{self.tag}-{self.setname}'
        hfn = self.dir + nfn + f'_{self.hash}'
        mfn = hfn + f'.eventmasks-{w}-{save_tag}.npy'
        # -1 of mask is the cell index (index in list)
        # -2 of mask is the signed distance
        lut = self.get_lut()
        clust_w = 15
        trim = max(w, 100) + clust_w
        if os.path.exists(mfn) and not overwrite:
            masks = numpy.load(mfn)
        else:
            masks = numpy.empty((1000, 2 * w + 2))
            data_counter = 0
            Y = self.session.ca.nnd
            F = self.session.ca.rel
            for ci, c in enumerate(self.cells):
                # cluster frames for onset detection
                event_t = numpy.where(Y[c, trim:-trim] > 0.5)[0] + trim
                if len(event_t) < fps:
                    continue
                clustering = cluster.DBSCAN(eps=clust_w, min_samples=2).fit(event_t.reshape(-1, 1))
                if clustering.labels_.max() < 1:
                    continue
                filtered_event_t = []
                for rci in range(clustering.labels_.max() + 1):
                    current_cluster = numpy.where(clustering.labels_ == rci)[0]
                    t0 = event_t[current_cluster[0]] - clust_w
                    t1 = event_t[current_cluster[-1]] + clust_w
                    p = ewma(F[c, t0:t1], 5)
                    filtered_event_t.append(t0 + numpy.argmax(p))
                event_t = sorted(filtered_event_t)
                lines = numpy.empty((len(event_t), 2 * w), dtype=numpy.int64)
                lines[:] = numpy.nan
                dists = numpy.empty(len(event_t))
                for ti, t in enumerate(event_t):
                    lines[ti] = numpy.arange(t - w, t + w)
                    dists[ti] = lut[self.loc[ci], int(self.pos[t] * self.resolution)]
                new_length = data_counter + len(lines)
                if new_length > len(masks):
                    masks = numpy.append(masks, numpy.empty((max(1000, len(lines)), masks.shape[1])), axis=0)
                masks[data_counter:new_length, :lines.shape[1]] = lines
                masks[data_counter:new_length, -1] = ci
                masks[data_counter:new_length, -2] = dists
                data_counter = new_length
            masks = masks[:data_counter]
            numpy.save(mfn, masks)
        return masks

    def pull_event_mean(self, param='rel', w=None, pull_session=None, pull_cells=None, save_tag=None, overwrite=False):
        '''mean transients, distance dependent. Leave session and cells None to pull on current.
        pass session and index list to pull from matched cells in a second ImagingSession'''
        if w is None:
            w = int(5 * fps)
        if pull_session is None:
            pull_session = self.session
        if pull_cells is None:
            pull_cells = self.cells
        if save_tag is None:
            save_tag = 'self'
        nfn = f'{self.prefix}-{self.tag}-{self.setname}'
        hfn = self.dir + nfn + f'_{self.hash}'
        rdfn = hfn + f'.binnedsignals-event-{param}-{w}-{save_tag}.npy'
        if os.path.exists(rdfn) and not overwrite:
            ba = numpy.load(rdfn)
        else:
            Y = pull_session.getparam(param)
            masks = self.get_event_mask(w, overwrite=overwrite)
            ba = numpy.empty((len(masks), 2 * w + 1))
            for i, l in enumerate(masks):
                ind = l[:w * 2].astype(numpy.int64)
                ba[i, -1] = l[-2]
                c = pull_cells[int(l[-1])]
                ba[i, :-1] = Y[c, ind]
            numpy.save(rdfn, ba)
        self.cache['events-' + save_tag] = ba

    def pull_laps(self, param='rel', res=None, pull_session=None, pull_cells=None, save_tag=None,
                  where='run', overwrite=False, nan_policy='omit'):
        '''mean activity in bins along lap'''
        if res is None:
            res = self.resolution
        if pull_session is None:
            pull_session = self.session
        if pull_cells is None:
            pull_cells = self.cells
        if save_tag is None:
            save_tag = 'self'
        nfn = f'{self.prefix}-{self.tag}-{self.setname}'
        hfn = self.dir + nfn + f'_{self.hash}'
        rdfn = hfn + f'.laps-{param}-{res}-{save_tag}-{where}.npy'
        if os.path.exists(rdfn) and not overwrite:
            ba = numpy.load(rdfn)
        else:
            if where == 'run':
                wh = numpy.where(self.session.pos.gapless[100:-100])[0] + 100
            Y = pull_session.getparam(param)
            laps = numpy.unique(self.session.pos.laps[wh])
            ba = numpy.empty((len(laps), len(pull_cells), res))
            ba[:] = numpy.nan
            bins = numpy.linspace(0, 1, res + 1)
            for li, lap in enumerate(laps):
                retained_frames = wh[numpy.where(self.session.pos.laps[wh] == lap)]
                if len(retained_frames) < 100:
                    continue
                for ci, c in enumerate(pull_cells):
                    xdat = self.pos[retained_frames]
                    ydat = Y[c, retained_frames]
                    if nan_policy == 'omit':
                        incldat = numpy.logical_not(numpy.isnan(ydat))
                        xdat = xdat[incldat]
                        ydat = ydat[incldat]
                    if nan_policy == 'tozero':
                        ydat = numpy.nan_to_num(ydat)
                    binned, edges, _ = binned_statistic(xdat, ydat, 'mean', bins=bins)
                    ba[li, ci] = binned
            numpy.save(rdfn, ba)
            self.cache['laps-' + save_tag] = ba

    def pull_loc_rate(self, bins=50):
        a = self.session
        wh = numpy.where(a.pos.gapless[100:-100])[0] + 100
        x = a.pos.pos[wh]
        x = numpy.maximum(0, x)
        x = numpy.minimum(1, x / numpy.percentile(x, 99))
        incl = self.cells
        rates = numpy.empty((bins, len(incl)))
        binsize = 1.0 / bins
        for bi in range(bins):
            weights = numpy.empty(len(wh))
            for wi, t in enumerate(wh):
                d = 1 - a.reldist(bi * binsize, x[wi])
                weights[wi] = numpy.e ** (1 - 1 / (d ** 2))
            for ci, c in enumerate(incl):
                rates[bi, ci] = numpy.average(a.ca.rel[c, wh], weights=weights)
        return rates

# testing
if __name__ == '__main__':
    # match test
    path = 'X:/Barna/ECB-Sncg/'
    os.chdir(path)
    prefix = 'ECB-Sncg_007_570'
    cells_tag = 'R'

    # init and save cells
    a = ImagingSession(prefix, tag=cells_tag, ch=1)
    s = Spatial()
    s.init_session(a)
    # s.det_PC()
    # s.save_named_set('PC')

    # init and load
    # s = Spatial()
    # s.init_files(path, prefix, cells_tag)
    s.load_named_set('mPC')
    s.set_pos()
    s.pull_event_mean()

    # # match test
    # path = 'X:/Barna/ECB-Sncg/'
    # os.chdir(path)
    # prefix = 'ECB-Sncg_007_550'
    # cells_tag = 'R'
    # dil_tag = 'D'
    # a = ImagingSession(prefix, tag=cells_tag, ch=1)
    # b = ImagingSession(prefix, tag=dil_tag, ch=0)
    # cells = numpy.where(a.qc())[0]
    # c_a, c_b = match_cells(a, b, cells)
