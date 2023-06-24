import re
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from scipy import stats
from Batch_Utils import outlier_indices
from Batch_Utils import strip_ax
from Alignment import Mouse
from Ripples import single_ripple_onsets
from sklearn import cluster
from scipy.signal import resample


''''Don't import CommonFunc in Core files (dependency loop). IF you need a function, move it to Batch_Utils'''

def read_excel(*args, **kwargs):
    return pandas.read_excel(*args, **kwargs, engine='openpyxl')

def filter_stim_area(session: ImagingSession, cutoff=250):
    'return sub list of cells that are below the last line affected by opto stimulus'
    cms = [(session.rois.polys[pi].min(axis=0) + session.rois.polys[pi].max(axis=0)) / 2 for pi in
           range(session.ca.cells)]
    return numpy.array([i for i, x in enumerate(cms) if x[1] > cutoff])


def nonstim_running_frames(session: ImagingSession, raw_opto=False):
    '''return set of frames for calculating place fields in opto sessions
    raw_opto: if enabled, includes single frames during 'thea' stim. by default, loads gapless opto'''
    wh = numpy.where(session.pos.gapless)[0][50:-8]
    if session.opto is not None:
        if raw_opto:
            opto = numpy.load(session.prefix + '_opto.orig.npy')
        else:
            opto = session.opto
        wh = [i for i in wh if not opto[i]]
    return wh


def ewma(trace, period=15):
    return numpy.array(pandas.DataFrame(numpy.nan_to_num(trace)).ewm(span=period).mean()[0])


def restrict_selection(df, column, criteria):
    return df.loc[df[column] == criteria]

def include_tag(df, tag):
    return df.loc[df['Included'].str.contains(tag, na=False)]

def get_groups(m: str):
    '''specific for this experiment, return indicator and actuator groups'''
    groups = (
        ('GECO_YFP', [227, 228, 229]), ('GECO_CHR', [207, 208, 209, 210, 211, 212]), ('RCAMP_YFP', [230, 231, 232]),
        ('RCAMP_CHR', [223, 224, 225, 226]), ('GCAMP_CHR', [236, 238, 240, 242]), ('GCAMP_mCherry', [237, 239, 241]))
    numset = re.compile('([0-9]+)')
    n = [int(text) for text in numset.split(m) if text.isdigit()][0]
    for g_n, g_l in groups:
        if n in g_l:
            return g_n.split('_')


def get_pvsncg_sexes():
    return {'BD155': 'm', 'BD170': 'f', 'BD169': 'f', 'BD172': 'f', 'BD171': 'f', 'BD220': 'f', 'BD246': 'm',
            'BD247': 'm', 'BD248': 'm', 80: 'm', 82: 'm', 84: 'm', }


def get_sex_axax(m: int):
    '''specific for this experiment'''
    #     numset = re.compile('([0-9]+)')
    #     m = [int(text) for text in numset.split(m) if text.isdigit()][0]
    return ['f', 'm'][m > 126]

def get_sex_axax_swap(m):
    '''specific for this experiment'''
    numset = re.compile('([0-9]+)')
    m = [int(text) for text in numset.split(m) if text.isdigit()][0]
    return ['f', 'm'][m < 327]

def get_sex_ecbdlx(m):
    '''specific for this experiment'''
    numset = re.compile('([0-9]+)')
    m = [int(text) for text in numset.split(m) if text.isdigit()][0]
    return ['f', 'm'][m < 432]


def get_sex_axax_ds(m):
    '''specific for this experiment'''
    numset = re.compile('([0-9]+)')
    m = [int(text) for text in numset.split(m) if text.isdigit()][0]
    return ['f', 'm'][m > 370]


def get_groups_pvsncg(m: str):
    '''specific for this experiment, return indicator and actuator groups'''
    groups = (
        ('mCherry', [233, 234, 235]), ('CHR', [220, 221, 222, 274, 276, 278]), ('YFP', [275, 277]))
    numset = re.compile('([0-9]+)')
    n = [int(text) for text in numset.split(m) if text.isdigit()][0]
    for g_n, g_l in groups:
        if n in g_l:
            return g_n


def estimate_optocutoff(prefix: str, min_delay=300, max_delay=400, pmt_delay=8, save_plot=True):
    '''find the line after which signal is safe to analyze (PMT back from gating and LED light decayed
    min delay: minimum lines to wait after PMT is back
    pmt_delay: always exclude so many lines when switching detector (frame start, stop, gating off)
    requires opto triggered average file, saved by ExportStopFromRaw
    '''
    # see if opto trig avg images can be used to determine cutoff line for including cells:
    fn = prefix + '_OptoActivity.tif'
    if not os.path.exists(fn):
        raise FileNotFoundError('Opto Triggered Average image missing: ', fn)
    opto_im = cv2.imread(fn, 1)  # ch1 is positive response, ch2 is negative response
    resp_line = opto_im.mean(axis=1)
    n_lines = len(resp_line)
    pmt_back = numpy.argmin(numpy.diff(resp_line[pmt_delay * 3:n_lines - pmt_delay, 2])) + pmt_delay * 4
    stim_residual = resp_line[:, 1][pmt_back:n_lines - pmt_delay]
    # fit exponential decay on stim artefact residue
    x = numpy.arange(len(stim_residual))
    try:
        fit = numpy.polyfit(x, numpy.log(stim_residual), 1, w=numpy.sqrt(stim_residual))
        # meaning of parameter: drops to 1/e every -1/fit[0] lines. waiting 2 times this
        decay_rate = -1 / fit[0]
        if decay_rate > 0:
            cutoff_y = pmt_back + decay_rate * 2
        else:
            cutoff_y = pmt_back
    except:
        cutoff_y = max_delay
        save_plot = False
    plot_fn = prefix + '_stim_cutoff_estimate.png'
    if save_plot:
        # plot tresholding for sanity check
        fig, ca = plt.subplots()
        ca.plot(resp_line[:, 1], color='green')
        ca.plot(resp_line[:, 2], color='red')
        fit_y = numpy.exp(fit[1]) * numpy.exp(fit[0] * x)
        ca.plot(x + pmt_back, fit_y, color='black', linestyle=':')
        ca.axvline(cutoff_y)
        fig.savefig(plot_fn, dpi=300)
        plt.close()
    return int(min(max(cutoff_y, min_delay), max_delay))


def decoding_R(session, cell_list=None, frames_list=None, n_bins=25, pf_calc_param='smtr', clf=None, ret_raw=False,
               resp_matrix=None):
    '''decode position in each lap trained by other laps, returns r of all frames from all laps'''
    if cell_list is None:
        cell_list = numpy.arange(session.ca.cells)
    else:
        cell_list = numpy.array(cell_list)

    if frames_list is None:
        all_frames = nonstim_running_frames(session)
    else:
        all_frames = numpy.array(frames_list)

    if clf is None:
        clf = MultinomialNB()

    # calcium in place cells during running
    if resp_matrix is None:
        ca = session.getparam(pf_calc_param)[cell_list][:, all_frames].transpose()
        ca = numpy.nan_to_num(ca)
    else:
        ca = resp_matrix.transpose()
    X = StandardScaler().fit_transform(ca)
    X = numpy.maximum(0, X)

    # train and test with each lap retained
    laps = numpy.unique(session.pos.laps[all_frames])
    actual, predicted = [], []
    Y = (n_bins * session.pos.relpos[all_frames]).astype(int)  # binned position
    for li, lap in enumerate(laps):
        training_frames = numpy.where(session.pos.laps[all_frames] != lap)
        retained_frames = numpy.where(session.pos.laps[all_frames] == lap)

        clf.fit(X[training_frames], Y[training_frames])
        actual.extend(Y[retained_frames])
        predicted.extend(clf.predict(X[retained_frames]))
    actual = numpy.array(actual)
    predicted = numpy.array(predicted)
    r_good = numpy.where(numpy.abs(predicted - actual) < n_bins * 0.75)  # discard corners
    rval = stats.spearmanr(actual[r_good], predicted[r_good])[0]
    if ret_raw:
        return rval, actual, predicted
    fig, ca = plt.subplots()
    ca.hist2d(actual, predicted, bins=n_bins)
    ca.set_aspect('equal')
    ca.set_xlabel('Actual position (bin)')
    ca.set_ylabel('Decoded position (bin)')
    ca.set_title(f'r={rval:.2f}')
    return rval, fig, ca


def decoding_AB(session, frames_list_A, frames_list_B, cell_list=None, n_bins=25, pf_calc_param='smtr', clf=None,
                resp_matrix=None):
    '''decode position in frame list B trained on frame list A'''
    if cell_list is None:
        cell_list = numpy.arange(session.ca.cells)
    else:
        cell_list = numpy.array(cell_list)

    if clf is None:
        clf = MultinomialNB()

    frames_list_A = numpy.array(frames_list_A)
    frames_list_B = numpy.array(frames_list_B)
    all_frames = numpy.append(frames_list_A, frames_list_B)

    # calcium in place cells during running
    if resp_matrix is None:
        ca = session.getparam(pf_calc_param)[cell_list][:, all_frames].transpose()
        ca = numpy.nan_to_num(ca)
    else:
        ca = resp_matrix.transpose()
    X = numpy.maximum(0, StandardScaler().fit_transform(ca))

    # train with A, test wit B
    Y = (n_bins * session.pos.relpos[all_frames]).astype(int)  # binned position

    clf.fit(X[:len(frames_list_A)], Y[:len(frames_list_A)])
    actual = Y[len(frames_list_A):]
    predicted = clf.predict(X[len(frames_list_A):])

    fig, ca = plt.subplots()
    ca.hist2d(actual, predicted, bins=n_bins)
    ca.set_aspect('equal')
    ca.set_xlabel('Actual position (bin)')
    ca.set_ylabel('Decoded position (bin)')
    actual = numpy.array(actual)
    predicted = numpy.array(predicted)
    # r_good = numpy.logical_and(numpy.logical_and(0 < actual, actual < n_bins - 1),
    #                            numpy.logical_and(0 < predicted, predicted < n_bins - 1))
    r_good = numpy.where(numpy.abs(predicted - actual) < n_bins * 0.75)
    rval = stats.spearmanr(actual[r_good], predicted[r_good])[0]
    ca.set_title(f'r={rval:.2f}')
    return rval, fig, ca


def decay_of_response(line, maxwindow=None):
    '''input a time series, return decay from maximum (constant, fit)'''
    if maxwindow is None:
        maxwindow = len(line)
    max_loc = numpy.argmax(line[:maxwindow])
    y = numpy.copy(line[max_loc:])
    if numpy.any(y <= 0):
        last_loc = numpy.argmax(y <= 0)
    else:
        last_loc = len(y) - max_loc
    # fit exponential decay on stim artefact residue
    y = y[:last_loc]
    x = numpy.arange(len(y))
    # transform so it's non-negative
    try:
        fit = numpy.polyfit(x, numpy.log(y), 1, w=numpy.sqrt(y))
        # meaning of parameter: drops to 1/e every -1/fit[0] samples.
        fit_y = numpy.exp(fit[1]) * numpy.exp(fit[0] * x)
        ret_line = numpy.empty(len(line))
        ret_line[:] = numpy.nan
        ret_line[max_loc:max_loc + last_loc] = fit_y
        return -1 / fit[0], ret_line
    except:
        print('Exponential decay fit failed')
        return None, None

def decay_of_response_full(line):
    '''input a time series (from peak to end), return decay from start (constant, fit)'''
    y = numpy.nan_to_num(line)
    # fit exponential decay on stim artefact residue
    # transform so it's non-negative
    last_loc = numpy.argmin(y)
    y = y[:last_loc]
    shiftval = line[last_loc]
    x = numpy.arange(len(y))
    try:
        fit = numpy.polyfit(x, numpy.log(y-shiftval), 1, w=numpy.sqrt(y-shiftval))
        # meaning of parameter: drops to 1/e every -1/fit[0] samples.
        fit_y = numpy.exp(fit[1]) * numpy.exp(fit[0] * x)
        ret_line = numpy.empty(len(line))
        ret_line[:] = numpy.nan
        ret_line[:last_loc] = fit_y+shiftval
        return -1 / fit[0], ret_line
    except:
        print('Exponential decay fit failed')
        return None, None



def rise_of_response(baseline, line):
    '''input a time series, return 10 to 90 rise time'''
    max_loc = numpy.argmax(line)
    baseline = numpy.nanmean(baseline)
    i10 = baseline + (line[max_loc] - baseline) * 0.1
    i90 = baseline + (line[max_loc] - baseline) * 0.9
    t90 = numpy.argmax(line > i90)
    return numpy.argmax(line[:t90][::-1] < i10)


def time_to_half(baseline, line):
    '''input a time series, return 10 to 90 rise time'''
    max_loc = numpy.argmax(line)
    baseline = numpy.nanmean(baseline)
    i50 = (line[max_loc] + baseline) * 0.5
    if numpy.any(line[max_loc:] < i50):
        return numpy.argmax(line[max_loc:] < i50)
    else:
        return numpy.nan


def load_eye(path, prefix):
    et = numpy.load(path + prefix + 'eye_trace.npy')
    for i in numpy.where(numpy.abs(numpy.diff(et[1:])) > 2)[0]:
        j = i + 1
        et[j] = numpy.mean([et[j - 1], et[j + 1]])
    smet = numpy.array(pandas.DataFrame(et).ewm(span=15).mean()[0])
    return smet / numpy.nanpercentile(smet, 99)

def load_corr(a: ImagingSession, path, prefix, w=1, suffix='_excess_mi.npy', mod_kw=None, test_mode=False,
              output_path='_decorrelation/'):
    if 'wm' in mod_kw:
        trace_fn = f'{output_path}{prefix}-{mod_kw}-trace.npy'
        if os.path.exists(trace_fn):
            y = numpy.load(trace_fn)
        else:
            y = numpy.empty(a.ca.frames)
            y[:] = numpy.nan
        return y
    emia_fn = path + output_path + prefix + str(w) + suffix
    print(mod_kw, emia_fn)
    if os.path.exists(emia_fn):
        emi = numpy.load(emia_fn)
    else:
        if not test_mode:
            y = numpy.empty(a.ca.frames)
            y[:] = numpy.nan
            return y
        raise FileNotFoundError(emia_fn)
    if mod_kw is None:
        # print('RETURNING MEAN FOR', prefix)
        return numpy.nanmean(emi, axis=0)
    elif mod_kw == 'pos':
        pos_cells = numpy.nanmean(emi, axis=1) > 0
        return numpy.nanmean(emi[pos_cells], axis=0)
    elif mod_kw == 'count':
        return numpy.nanmean(emi > 0, axis=0)
    elif mod_kw == 'full':
        return emi
    elif mod_kw == '2d_mean':
        return numpy.nanmean(emi, axis=(0,1))





def load_face(a: ImagingSession, path, prefix, blank_movement=False):
    '''single function to allow PSTH to get face movement trace from motion maps pulled by EyeTracing
    processed with smoothing, z score, blank on run.
    '''
    fn = path + prefix + '_face_trace.npy'
    if os.path.exists(fn):
        face = numpy.load(fn)
    else:
        mm = numpy.load(path + prefix + '_motion_map.npy', mmap_mode='r')
        line = mm.mean(axis=(1, 2))
        # mt = numpy.zeros(a.ca.frames)
        # for i, k in enumerate(numpy.linspace(0, len(line)-0.01, a.ca.frames)):
        #     mt[i] = line[int(k)]
        mt = resample(line, a.ca.frames)

        # clean it up
        t1 = fps
        smw = numpy.empty(a.ca.frames)
        for t in range(a.ca.frames):
            ti0 = max(0, int(t - t1 * 0.5))
            ti1 = min(len(mt), int(t + t1 * 0.5) + 1)
            smw[t] = numpy.mean(mt[ti0:ti1])
        bsl = numpy.empty(a.ca.frames)
        t2 = int(t1 * 50)
        for t in range(a.ca.frames):
            ti0, ti1 = max(0, t - t2), min(t, a.ca.frames)
            if ti0 < ti1:
                minv = numpy.min(smw[ti0:ti1])
            else:
                minv = smw[ti0]
            bsl[t] = minv
        rel = numpy.maximum(0, (mt - bsl) / bsl)
        ntr = rel / numpy.std(rel[numpy.where(numpy.logical_not(a.pos.gapless))])
        face = numpy.array(pandas.DataFrame(ntr).ewm(span=15).mean()[0])
        # 2) zero at movement, beginning, end, and sd < 1
        if blank_movement:
            face[:100] = 0
            face[-100:] = 0
            for events, direction in zip(a.startstop(ret_loc='actual'), (1, -1)):
                for t in events:
                    while numpy.any(ntr[t - 3:t + 3] > 1):
                        face[t + direction] = 0
                        t += direction
            face[numpy.where(a.pos.gapless)] = 0
            if a.pos.gapless[-100]:
                t = a.ca.frames - 100
                while numpy.any(ntr[t - 3:t + 3] > 1):
                    face[t - 1] = 0
                    t -= 1
        numpy.save(fn, face)
    return face


def get_cell_dict_from_alignment(mouse_path, mouse_id, prefix):
    '''return a dict with IDd cells in prefix'''
    mouse = Mouse(mouse_path, mouse_id)
    m_cells = {}
    for i, c in mouse.data['cells'].items():
        for fn in c['positions'].keys():
            if prefix in fn:
                m_cells[i] = c['positions'][fn]
    return mouse, m_cells


def p_lookup(p):
    if numpy.isnan(p):
        return 'nan'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

def jitter(x, y, s=1.0):
    j = numpy.maximum(numpy.random.normal(0, s/5, len(y)), s * -0.5)
    j = numpy.minimum(j, s * 0.5)
    return numpy.ones(len(y))*x + j

def normalize_phase(angle, shift=0):
    '''take a list of angles (in degrees), shifts it, and normalizes it between -180 and 180'''
    assert hasattr(angle, '__len__')
    newAngle = numpy.array(angle) - shift
    newAngle[numpy.where(newAngle <= -180)] += 360
    newAngle[numpy.where(newAngle > 180)] -= 360
    return newAngle

def single_from_list(events, gap):
    event_t = numpy.copy(events)
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


class Lasso:
    '''display an image draw polygons on it.
    polygons can be accessed as coord lists in the polys attribute'''
    def __init__(self, im):
        self.im = im
        self.polys = []
        self.coords = []
        self.drawing = False
        # if len(im.shape) == 3:
        self.linecolor = (255, 255, 0)
        # elif len(im.shape) == 2:
        #     self.linecolor = (255,)
        cv2.imshow('Lasso', self.im)
        cv2.moveWindow('Lasso', 0, 0)
        cv2.setMouseCallback('Lasso', self.drawmouse)
        self.retval = False
        while self.retval is False:
            if cv2.getWindowProperty('Lasso', 0) < 0:
                break
            k = cv2.waitKey(1) & 0xFF
        cv2.destroyWindow('Lasso')

    def drawmouse(self, event, y, x, flags, param):
        if event == cv2.EVENT_RBUTTONDBLCLK:
            self.retval = -1
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if len(self.polys) > 0:
                self.retval = self.polys
            else:
                self.retval = -1
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.coords) > 2:
                cv2.line(self.im, (y, x), (self.coords[0][1], self.coords[0][0]), color=self.linecolor)
                self.clean_poly()
                cv2.imshow('Lasso', self.im)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

        if event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.coords.append([x, y])
            if len(self.coords) > 1:
                cv2.line(self.im, (y, x), (self.coords[-2][1], self.coords[-2][0]), color=self.linecolor)
                cv2.imshow('Lasso', self.im)

    def clean_poly(self):
        new = []
        for p in self.coords:
            x = round(p[0])
            y = round(p[1])
            np = [y, x]
            if not np in new:
                new.append(np)
        new.append(new[0])
        self.polys.append(new)
        self.coords = []

def norm_minmax(y, trim=False):
    a = y-numpy.nanpercentile(y, 1)
    if trim:
        a = numpy.maximum(0, numpy.nan_to_num(a))
        return numpy.minimum(1, a / numpy.percentile(a, 99))
    else:
        return a / numpy.nanpercentile(a, 99)