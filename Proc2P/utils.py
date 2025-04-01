import os
import datetime
import numpy
import pandas
import json
from scipy import stats


def lprint(obj, message, *args, logger=None):
    '''Add timestamp and object name to print calls'''
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    for x in args:
        message += ' ' + str(x)
    if obj is None:
        output = f'{ts}: {message}'
    else:
        output = f'{ts} - {obj.__name__}: {message}'
    print(output)
    if logger is None:
        return output
    else:
        logger.log(output)


class logger:
    def __init__(self, filehandle=None):
        self.fn = filehandle

    def set_handle(self, procpath, prefix):
        self.fn = os.path.join(procpath, prefix + '/', prefix + f'_{get_user()}_AnalysisLog.txt')

    def log(self, message):
        if not message.endswith('\n'):
            message += '\n'
        with open(self.fn, 'a') as f:
            f.write(message)


from Proc2P.Legacy.Batch_Utils import strip_ax


def get_user():
    return os.environ.get('USERNAME')


def norm(d):
    wh_notna = numpy.logical_not(numpy.isnan(d))
    y = numpy.empty(d.shape)
    y[:] = numpy.nan
    a = d[wh_notna] - numpy.min(d[wh_notna])
    y[wh_notna] = numpy.minimum(a / numpy.percentile(a, 99, axis=0), 1)
    return y


def gapless(trace, gap=5, threshold=0, expand=None):
    '''makes binary trace closing small gaps
    :param gap: in samples
    '''
    if trace is None:
        return None
    elif not numpy.count_nonzero(trace):
        return numpy.zeros(len(trace))
    gapless = trace > threshold
    ready = False
    gap = int(gap)
    while not ready:
        ready = True
        for t, m in enumerate(gapless):
            if not m:
                if numpy.any(gapless[t - gap:t]) and numpy.any(gapless[t:t + gap]):
                    gapless[t] = 1
                    ready = False
    if expand is not None:
        eg = numpy.copy(gapless)
        for t, m in enumerate(gapless):
            if not m:
                if numpy.any(gapless[t - expand:t]) or numpy.any(gapless[t:t + expand]):
                    eg[t] = 1
        gapless = eg
    return gapless


def outlier_indices(values, thresh=3.5):
    '''return the list of indices of outliers in a series'''
    not_nan = numpy.logical_not(numpy.isnan(values))
    median = numpy.median(values[not_nan])
    diff = (values - median) ** 2
    diff = numpy.nan_to_num(numpy.sqrt(diff))
    med_abs_deviation = numpy.median(diff[not_nan])
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return numpy.where(modified_z_score > thresh)[0]


def startstop(speed, duration=50, gap=150, ret_loc='actual', span=None, speed_threshold=0.05, smoothing=None):
    '''
    :param speed
    :param duration: of run, in samples
    :param gap: of stop, in samples
    :param ret_loc: 'actual' gives frame with zero speed. need to specify speed for this
    :param span: slice of recording to analyze
    :return:
    '''
    # binary gapless trace whether animal is running
    if smoothing is None:
        smoothing = 20
    mov = gapless(numpy.nan_to_num(speed), threshold=speed_threshold, gap=smoothing)
    if gap is None:
        gap = 150
    if span is None:
        span = gap, len(mov) - gap
    # collect stops
    stops, starts = [], []
    t = span[0] + duration
    while t < span[1] - gap:
        if not numpy.any(mov[t:t + gap]) and numpy.all(mov[t - duration:t]):
            t0 = t
            if ret_loc == 'peak':
                while speed[t0] < speed[t0 - 1]:
                    t0 -= 1
                stops.append(t0)
                while mov[t0]:
                    t0 -= 1
                starts.append(t0)
            elif ret_loc == 'stopped':
                stops.append(t)
                while mov[t0 - 1]:
                    t0 -= 1
                starts.append(t0)
            elif ret_loc == 'actual':
                # go back while raw speed is zero
                while speed[t - 1] < speed_threshold and t > 100:
                    t -= 1
                if t > 100:
                    stops.append(t)
                    while mov[t0 - 1] and t0 > 100:
                        t0 -= 1
                    while speed[t0] < speed_threshold:
                        t0 += 1
                    starts.append(t0)
            t += gap
        t += 1
    return starts, stops


def read_excel(*args, **kwargs):
    fn = args[0]
    if fn.endswith('csv'):
        return pandas.read_csv(*args, **kwargs)
    return pandas.read_excel(*args, **kwargs, engine='openpyxl')


def ewma(trace, period=15):
    return numpy.array(pandas.DataFrame(numpy.nan_to_num(trace)).ewm(span=period).mean()[0])


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


def path_to_list(path):
    parent_folders = []
    while path[-1] in ('//', '/', '\\'):
        path = path[:-1]
    while True:
        path, folder = os.path.split(path)
        if folder:
            parent_folders.append(folder)
        else:
            if path and path not in ('//', '/', '\\'):
                parent_folders.append(path)
            break
    parent_folders.reverse()
    return parent_folders


class completed_list:
    '''
    *deprecated, use DataSet*
    keep persistent track of jobs that are complete. usage:
    completed_prefix = completed_list(project_path + '_completed_prefixes.txt')
    for prefix in jobs:
        if prefix in completed_prefix:
            continue
        do stuff
        completed_prefix(prefix)
    '''

    def __init__(self, filehandle):
        self.filehandle = filehandle
        if os.path.exists(filehandle):
            with open(filehandle, 'r') as f:
                completed_prefix = [s.strip() for s in f.readlines()]
        else:
            completed_prefix = []
        self.list = completed_prefix

    def write(self, prefix):
        with open(self.filehandle, 'a') as of:
            of.write(prefix + '\n')

    def __contains__(self, item):
        return item in self.list

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)


def ts(format=None):
    now = datetime.datetime.now().isoformat(timespec='seconds')
    if format is None:
        return now
    else:
        return now.replace(':', '')


def CI_linreg(x, y):
    # fit a curve to the data using a least squares 1st order polynomial fit
    z = numpy.polyfit(x, y, 1)
    p_y = z[0] * x + z[1]

    # calculate the y-error (residuals)
    y_err = y - p_y

    # create series of new test x-values to predict for
    stepsize = (numpy.max(x) - numpy.min(x)) / 100
    p_x = numpy.arange(numpy.min(x), numpy.max(x) + stepsize, stepsize)

    # now calculate confidence intervals for new test x-series
    mean_x = numpy.mean(x)  # mean of x
    n = len(x)  # number of samples in origional fit
    t = stats.t.ppf(1 - 0.025, n - 1)  # appropriate t value (where n=9, two tailed 95%)
    s_err = numpy.sum(numpy.power(y_err, 2))  # sum of the squares of the residuals

    confs = t * numpy.sqrt((s_err / (n - 2)) * (1.0 / n + (numpy.power((p_x - mean_x), 2) /
                                                           ((numpy.sum(numpy.power(x, 2))) - n * (
                                                               numpy.power(mean_x, 2))))))

    # now predict y based on test x-values
    p_y = z[0] * p_x + z[1]

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)
    return p_x, lower, upper


def touch_path(*args):
    '''
    Join all args and create folder if does not exist
    :param args:
    :return:string of the folder
    '''
    path = os.path.realpath(os.path.join(*args))
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


class DataSet:
    __name__ = 'DataSet'

    def __init__(self, project_folder, ver=None):
        '''
        :param project_folder: a unique place (in Processed, backed up) for analysis outputs related to the dataset
        :param ver: an int. if None, last available is read.
        '''

        self.project_folder = os.path.realpath(project_folder)
        self.prefix = '_dataset'
        self.ext = '.feather'
        if ver is not None:
            self.ver = int(ver)
        else:
            self.get_current_ver()
        self.load_df()
        self.mod_flag = False
        self.readonly_flag = False
        self.db = None

    def __contains__(self, item):
        return item in self.df["Prefix"].values

    def __getitem__(self, item):
        return self.df.loc[self.df["Prefix"].eq(item)].iloc[0]

    def _changed(func):

        def wrapper(self, *args, **kwargs):
            self.mod_flag = True
            func(self, *args, **kwargs)

        return wrapper

    def get_fn(self, ext=None):
        if ext is None:
            ext = self.ext
        return os.path.join(self.project_folder, f'{self.prefix}{self.ver:02}{ext}')

    def get_current_ver(self, incr=0):
        flist = os.listdir(self.project_folder)
        ds = [x for x in flist if x.endswith(self.ext) and x.startswith(self.prefix)]
        if not len(ds):
            self.ver = 0
        else:
            vs = [int(x[len(self.prefix):-len(self.ext)]) for x in ds]
            self.ver = max(vs) + incr
        if incr:
            self.mod_flag = True

    def load_df(self):
        fn = self.get_fn()
        if os.path.exists(fn):
            self.df = pandas.read_feather(fn)
            # self.readonly_flag = True
        else:
            self.new_df()

    def save_df(self, ver='current'):
        if ver == 'excel':
            self.df.to_excel(self.get_fn(ext='.xlsx'))
        if ver == 'current':
            assert not self.readonly_flag
        elif ver == 'next':
            self.get_current_ver(incr=1)
        if self.mod_flag:
            self.df.to_feather(self.get_fn())
            self.mod_flag = False

    def new_df(self):
        self.df = pandas.DataFrame(columns=["Prefix", "Animal", "Sex", "Incl", "Excl"])

    def report(self):
        rtext = f'Dataset v{self.ver}: {len(self.df)} sessions, {len(self.get_incl())} included\n'
        rtext += f'Project folder: {self.project_folder}'
        print(rtext)
        return rtext

    @_changed
    def new_record(self, prefix):
        db = self.check_db()
        animal, sex = db.get_sex(prefix)
        self.df.loc[len(self.df)] = {"Prefix": prefix, "Animal": animal, "Sex": sex, "Incl": True}

    def check_db(self):
        if self.db is None:
            self.db = GetSessions()
        return self.db

    @_changed
    def include(self, prefix, tag=None, value=True, add_fields=None):
        '''
        :param prefix: add/mark a prefix (or a list of prefixes) to be included in the dataset
        :param tag: optional. use a string if only want to include in a subset
        :return:
        '''
        inclfield = 'Incl'
        if tag is not None:
            inclfield += f'.{tag}'
        new_prefix = self.listify(prefix)
        for pf in new_prefix:
            if pf not in self.df["Prefix"].values:
                self.new_record(pf)
        self.set_field(prefix, inclfield, value)

    @_changed
    def exclude(self, prefix, tag=None):
        '''
        Mark excluded.
        :param prefix: a prefix (or a list of prefixes)
        :param tag: optional alternative excl tag
        '''
        exclfield = 'Excl'
        if tag is not None:
            exclfield += f'.{tag}'
        self.set_field(prefix, exclfield, True)

    @_changed
    def set_field(self, prefix, key, value=True):
        prefix = self.listify(prefix)
        for pf in prefix:
            index = self.df.loc[self.df["Prefix"].eq(pf)].index
            if not len(index) == 1:
                raise ValueError(f'Prefix should have exactly one match, {pf} had {len(index)}')
            self.df.loc[index[0], key] = value

    @_changed
    def set_by_dict(self, prefix, add_fields):
        index = self.df.loc[self.df["Prefix"].eq(prefix)].index
        if not len(index) == 1:
            raise ValueError(f'Prefix should have exactly one match, {pf} had {len(index)}')
        for key, value in add_fields.items():
            self.check_key(key)
            self.df.loc[index[0], key] = value

    def listify(self, prefix):
        if not type(prefix) == list:
            list_prefix = [prefix, ]
        else:
            list_prefix = list(set(prefix))
            list_prefix.sort()
        return list_prefix

    def get_field(self, prefix, key):
        return self.df.loc[self.df["Prefix"].eq(prefix)].iloc[0][key]

    def check_key(self, key, value=None):
        if key not in self.df.columns:
            self.df[key] = value

    @_changed
    def include_cells(self, prefix, roi_tag, cells, alt_tag=None, excl=False):
        '''
        Adds the specified cells to a list of included cells.
        :param prefix:
        :param roi_tag:
        :param cells: list of indices
        :param alt_tag: if set, can use multiple lists within a roi set
        :param excl: If True, adds the specified cells to a list of excluded cells. Possible values are:
        '''
        suffix = ('Incl', 'Excl')[excl]
        cellfield = f'Cells.{roi_tag}.{suffix}'
        if alt_tag is not None:
            cellfield += f'.{alt_tag}'
        if cellfield not in self.df.columns:
            self.check_key(cellfield, value=json.dumps(None))
            clist = cells
        else:
            if cells == None:
                clist = cells
            else:
                old_list = json.loads(self.get_field(prefix, cellfield))
                if old_list == None:
                    clist = cells
                elif type(old_list) == list:
                    old_list.extend(cells)
                    clist = old_list
                else:
                    raise ValueError(f'Not sure what to do with this cell list: {cells}, current is {old_list}')
        if type(clist) == list:
            clist = [int(x) for x in set(clist)]
            clist.sort()
        self.set_field(prefix, cellfield, json.dumps(clist))

    def get_cells(self, prefix, roi_tag, alt_tag=None):
        rets = {}
        for suffix in ('Incl', 'Excl'):
            cellfield = f'Cells.{roi_tag}.{suffix}'
            if alt_tag is not None:
                cellfield += f'.{alt_tag}'
            if cellfield in self.df.columns:
                rets[suffix] = json.loads(self.get_field(prefix, cellfield))
        if 'Incl' not in rets:
            return None
        if 'Excl' not in rets:
            return rets['Incl']
        else:
            incl = rets['Incl']
            excl = rets['Excl']
            if incl is None:
                return (incl, excl)  # we don't have the max n of cells, need to look up externally
            if type(excl) == list:
                return [c for c in incl if c not in excl]
            else:
                return incl

    def get_incl(self, key="Incl", excl=None):
        '''
        Return the df with the included sessions. Always exlcudes ones that have the global Excl set.
        :param key: specify alternative column name
        :param excl: specify additional alternative exclusion column
        :return:
        '''
        sub = self.df.loc[~self.df["Excl"].eq(True)]
        if excl is not None and excl in self.df.columns:
            sub = sub.loc[~sub[excl].eq(True)]
        return sub.loc[sub[key].eq(True)]
