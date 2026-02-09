import os.path
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Analysis.AnalysisClasses.NormalizeVm import arima_filtfilt
from Proc2P.utils import lprint, outlier_indices
from datetime import datetime
import pandas
import numpy
class PSTH:
    __name__ = 'PSTH'
    '''Create a set of sessions, cells and categories that is used to pull event triggered averages'''

    def __init__(self, path, identifier):
        ''' Init only, data should be added or loaded from file
        path: project working folder
        identifier: unique name of data set, used to store data in files'''
        self.path = path
        self.identifier = identifier
        self.wdir = path + 'EventTrigger/'
        self.lock_name = self.wdir + '_' + identifier + '.lock'
        if not os.path.exists(self.wdir):
            os.mkdir(self.wdir)
        self.items_filename = self.wdir + '_' + self.identifier + '.items'
        self.groups_filename = self.wdir + '_' + self.identifier + '.groups'
        # prepare container attributes
        self.field_names = ['Path', 'Prefix', 'ROItag', 'Channel', 'CellIndex',
                            'TraceIndex']  # fields in item_dict tuples
        self.n_of_session_fields = 4
        self.group_names = []
        self.item_dict = {}
        self.group_dict = {}
        self.group_df = None
        self.pull_dict = {}
        self.cell_counter = 0
        self.event_counter = 0
        self.data = None
        self.weights = None
        self.extra_columns = 2  # number of additional info fields in numpy arrays. original index, unique index.
        # reserve attribute names, these can be modified with setter function
        self.uses_ripples = False
        self.uses_behav = False
        self.uses_weights = False
        self.param_key = 'rel'
        self.window_w = 156
        self.eyepath = ''
        self.session_kwargs = {}
        # reserve mask function names, this has to be specified with setter function
        self.mask_function = None
        self.mask_function_name = None
        self.mask_function_kwargs = None
        self.pull_function_kwargs = {}
        # lock metadata once assembled, so cell indices can be relied on to group them
        if os.path.exists(self.lock_name):
            self.locked = True
            self.load_items()
        else:
            self.locked = False

    def add_items(self, prefix, tag, cells, grouping=None, group_names=None, path=None, ch=0):
        '''create an identifier for each item and store in item dict
        prefix: session prefix
        tag: session tag
        cells: cell index or list of indices. each will be added to item dict and assigned a unique ID
        grouping: group ID, a tuple of group IDs for all cells, or list or ndarray same length as cells.
        can have multiple group identifiers for each cell.
        group_names: string handle for each group
        groups can be modified after finalize'''
        if self.locked:
            raise Exception('Cell set locked, delete lock and existing traces to add data')
        # check if cells is a list
        if not hasattr(cells, '__iter__'):
            cells = [cells]
        # check if grouping is the same for all cells:
        if not hasattr(grouping, '__iter__'):
            simple_group = True
            grouping = (grouping,)
        elif type(grouping) is tuple:
            simple_group = True
        elif len(grouping) == len(cells):
            simple_group = False
        else:
            raise ValueError('Error with parsing grouping')
        if hasattr(group_names, '__iter__') and type(group_names) is not str:
            multiple_names = True
        else:
            multiple_names = False
        for g_n in group_names:
            if g_n not in self.group_names:
                self.group_names.append(g_n)
        if path is None:
            path = os.curdir
        # generate ID of session and roi set
        session_hash = abs(hash(prefix + str(tag) + str(ch))) % (10 ** 8) * 10000
        # add each cell to dict
        for ci, c in enumerate(cells):
            cell_id = session_hash + c
            self.item_dict[cell_id] = (path, prefix, tag, ch, c, self.cell_counter)
            self.cell_counter += 1
            self.group_dict[cell_id] = {}
            if simple_group:
                group = grouping
            else:
                group = grouping[ci]
            if not multiple_names:
                self.group_dict[cell_id][group_names] = group
            else:
                for gi, g_n in enumerate(group_names):
                    self.group_dict[cell_id][g_n] = group[gi]


    def sanitize_filename(self, fn):
        d, f = os.path.split(os.path.realpath(fn))
        return os.path.join(d, f.replace('.', '_'))
    def save_items(self):
        '''stores items and groups dicts in pickled binary pandas dataframes
        also saved as excel for future proof'''
        # store items
        df = pandas.DataFrame.from_dict(self.item_dict, orient='index', dtype=None, columns=self.field_names)
        df.to_pickle(self.items_filename)
        df.to_excel(self.sanitize_filename(self.items_filename) + '.xlsx')
        # store groups
        df = pandas.DataFrame.from_dict(self.group_dict, orient='index', dtype=None, columns=self.group_names)
        df.to_pickle(self.groups_filename)
        df.to_excel(self.sanitize_filename(self.groups_filename) + '.xlsx')
        # create lock
        self.locked = True
        with open(self.lock_name, 'w') as f:
            f.write(f'Data for {self.cell_counter} cells saved on {datetime.now().isoformat()}.')

    def load_items(self):
        '''restores items and groups dicts from pickle file, overwriting any existing items in instance attribute'''
        # load items
        self.item_dict = {}
        df = pandas.read_pickle(self.items_filename)
        for _, item in df.iterrows():
            self.item_dict[item.name] = tuple([item[cn] for cn in self.field_names])
        # load groups
        df = pandas.read_pickle(self.groups_filename)
        self.group_dict = df.to_dict(orient='index')
        self.group_names = list(df.columns)

    def set_pull_params(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def set_triggers(self, mask_function, trigger_name, **kwargs):
        '''specify the function (from EventMasks) to pull events, with additional arguments for the function
         (other than a, w). The function will be called with the sessions and window width set in the instance.'''
        self.mask_function = mask_function
        self.trigger_name = trigger_name
        self.mask_function_kwargs = kwargs

    def set_session_kwargs(self, **kwargs):
        '''Specify kwargs that will be passed to the init of the ImagingSession by pull_traces'''
        self.session_kwargs = kwargs

    def set_pull_function_kwargs(self, *args, **kwargs):
        '''Specify kwargs that will be passed to any loader function called by pull_traces
        NB this is not for mask functions! for that, pass kwargs to set_triggers'''
        if None in args:
            self.pull_function_kwargs = {}
        self.pull_function_kwargs = kwargs

    def ready(self):
        '''rebuild dict of items to allow getting all cells for each session'''
        if not self.locked:
            raise Exception('Save and lock before pulling.')
        sessions = {}
        for key, item in self.item_dict.items():
            unique_session = item[:self.n_of_session_fields]
            if unique_session not in sessions:
                sessions[unique_session] = {}
            sessions[unique_session][key] = item[self.n_of_session_fields:]
        self.pull_dict = sessions
        self.group_df = pandas.DataFrame.from_dict(self.group_dict, orient='index', dtype=None,
                                                   columns=self.group_names)

    def pull_traces(self, other_tag=None, other_ch=None, other_match=False, mask_stim=None):
        '''open each session and save event triggered average'''

        # cycle through sessions
        for session_data, cell_data in self.pull_dict.items():
            path, prefix, tag, ch = session_data
            fn = self.get_numpy_name(prefix, tag, ch)
            if os.path.exists(fn):
                print('Loading', prefix)
                continue
            lprint(self, ' '.join(('Pulling:', prefix, tag, self.trigger_name, self.param_key)))
            data = numpy.empty((1000, 2 * self.window_w + self.extra_columns))
            data_counter = 0
            os.chdir(path)
            a = ImagingSession(path, prefix, tag=tag, norip=not self.uses_ripples, ch=ch, **self.session_kwargs)
            if self.uses_ripples:
                rt = self.session_kwargs.get('ripple_tag')
                if rt is not None:
                    assert a.ripples.tag == rt
            if not self.eyepath == '':
                a.eye_path = self.eyepath
            events = self.mask_function(a, self.window_w, **self.mask_function_kwargs)
            if self.uses_weights:
                events, mask, weights = events
            else:
                events, mask = events
            fill_line = False
            if self.param_key == 'pupil':
                line = a.map_eye(model_name=self.pull_function_kwargs.get('model_name', 'final'))
                fill_line = True
            # elif self.param_key == 'face':
            #     # line = load_face(a, self.path + self.eyepath, prefix)
            #     line = load_face(a, self.eyepath, prefix)
            #     fill_line = True
            elif self.param_key == 'speed':
                line = a.pos.speed
                fill_line = True
            elif self.param_key == 'licks':
                line = numpy.zeros(a.ca.frames)
                for t in a.bdat.licks:
                    fr = a.timetoframe(t)
                    if 0 < fr < a.ca.frames:
                        line[fr] += 1
                fill_line = True
            elif self.param_key == 'npc1':
                line = a.ca.get_npc(1)[:, 0]
                fill_line = True
            elif self.param_key == 'rpw':
                if hasattr(a, 'ripple_power'):
                    line = a.ripple_power
                else:
                    line = a.ephys.ripple_power
                fill_line = True
            elif self.param_key == 'rpwz':
                if hasattr(a, 'ripple_power'):
                    line = a.ripple_power
                else:
                    line = a.ephys.ripple_power
                fill_line = True
                line /= numpy.nanstd(line)
            elif self.param_key == 'ripp-n':
                line = a.get_ripple_number_trace() * a.fps
                fill_line = True
            elif self.param_key == 'maxenv':
                nframes = a.ca.frames
                line = numpy.empty(nframes)
                t0, f = 0, 0
                for t1, s in enumerate(a.ephysframes):
                    if s > f:
                        if f < nframes:
                            line[f] = a.ripples.envelope[t0:t1].max()
                            t0 = t1
                            f += 1
                fill_line = True
            elif self.param_key == 'thetapw':
                line = a.ephys.theta_power
                fill_line = True
            elif self.param_key == 'hfpw':
                line = a.ephys.hf_power
                fill_line = True
            elif self.param_key == 'LFP':
                line = a.ephys.edat[1]
                fill_line = True
            elif self.param_key == 'LFPabs':
                line = numpy.abs(a.ephys.edat[1])
                fill_line = True
            # elif self.param_key == 'inst_ev_freq':
            #     np_fn = a.prefix+'_inst_sz_freq_1116.npy'# adjust between sz and epi events as needed
            #     if os.path.exists(np_fn):
            #         line = numpy.load(a.prefix+'_inst_sz_freq_1116.npy')
            #     else:
            #         line = numpy.empty(a.ca.frames)
            #         line[:] = numpy.nan
            #     fill_line = True
            # elif 'corr' in self.param_key:
            #     line = load_corr(a, path, prefix, **self.pull_function_kwargs)
            #     fill_line = True
            elif 'gpw' in self.param_key:
                if not hasattr(self, 'gamma_band'):
                    raise ValueError('Specify gamma band s, f, or b with set_pull_params')
                filt_func = {'s': a.ripples.calc_sgamma,
                             'f': a.ripples.calc_fgamma,
                             'b': a.ripples.calc_bgamma,
                             'slo': a.ripples.calc_slo}
                filt_func[self.gamma_band]()
                line = numpy.zeros(a.ca.frames)
                t0, f = 0, 0
                for t1, s in enumerate(a.ephysframes):
                    if s > f:
                        if f < a.ca.frames:
                            line[f] = a.ripples.gamma[t0:t1].mean()
                            t0 = t1
                            f += 1
                fill_line = True
            elif type(self.param_key) is dict:  # read any attr of imagingsession
                if 'attr' in self.param_key:
                    line = getattr(a, self.param_key['attr'])
                    fill_line = True
                else:
                    raise ValueError('''paramkey not understood. use {'attr': 'ripple_power'} format''')
            else:
                has_other = not ((other_ch is None) and (other_tag is None))
                if not has_other:
                    param = a.getparam(self.param_key, mask_stim=mask_stim)
                else:
                    # other has to be same cells, but can have different tag and ch. masks from a will beused to pull b
                    if other_tag is None:
                        other_tag = a.ca.tag
                    if other_ch is None:
                        other_ch = a.ca.ch
                    b = ImagingSession(prefix, tag=other_tag, norip=not self.uses_ripples, ch=other_ch,
                                       no_tdml=not self.uses_behav, **self.session_kwargs)
                    a_param = a.getparam(self.param_key)
                    b_param = b.getparam(self.param_key)
                    if other_match:  # re-fill param so original indices are replaced with match
                        param = numpy.empty(a_param.shape)
                        param[:] = numpy.nan
                        incl = numpy.where(a.qc())[0]
                        c_a, c_b = match_cells(a, b, incl)
                        for ci, cj in zip(c_a, c_b):
                            param[ci] = b_param[cj]
                    else:
                        param = b_param
            if self.pull_function_kwargs.get('fillnan'):
                line = arima_filtfilt(line)
            if fill_line:
                param = numpy.zeros(a.ca.rel.shape)
                l_dat = min(len(line), a.ca.frames)
                param[:, :l_dat] = line[:l_dat]
            # sort original cell indices and unique cell indices into array
            cells = numpy.empty((2, len(cell_data)), dtype='int')
            for i, (key, value) in enumerate(cell_data.items()):
                cells[:, i] = value
            # pull values
            for frame, indices in zip(events, mask):
                lines = numpy.empty((len(cell_data), 2 * self.window_w))
                lines[:] = numpy.nan
                loc = numpy.where(numpy.logical_not(numpy.isnan(indices)))[0]
                try:
                    lines[:, loc] = param[cells[0]][:, indices[loc].astype(numpy.int64)]
                except:
                    print(prefix, tag, a.ca.cells, cells[0])
                    assert False
                new_length = data_counter + len(lines)
                if new_length > len(data):
                    data = numpy.append(data, numpy.empty((max(1000, len(lines)), data.shape[1])), axis=0)
                data[data_counter:new_length, :lines.shape[1]] = lines
                data[data_counter:new_length, -2] = cells[0]
                data[data_counter:new_length, -1] = cells[1]
                data_counter = new_length
            numpy.save(fn, data[:data_counter])
            # save weights
            if self.uses_weights:
                n_e = len(events)
                n_c = len(cells[0])
                weight_of_cells = numpy.empty(n_e * n_c)
                for i in range(len(events)):
                    weight_of_cells[i * n_c: (i + 1) * n_c] = weights[i, cells[0]]
                numpy.save(self.get_numpy_name(prefix, tag, ch, weight=True), weight_of_cells)
        self.summary()

    # make sure this indexes into the right shape
    def get_unique_indices(self):
        return self.data[:, -1]

    def get_numpy_name(self, prefix, tag, ch, weight=False):
        str_builder = [self.identifier, prefix, tag, ch, self.trigger_name, self.param_key, self.window_w]
        if self.mask_function_kwargs is not None:
            stripchars = r'"{:} ' + "'"
            str_builder.append(''.join([x for x in str(self.mask_function_kwargs) if x not in stripchars]))
        fn = self.wdir + '_'.join([str(s) for s in str_builder])
        if weight:
            return fn + '_weights.npy'
        else:
            return fn + '.npy'

    def summary(self):
        '''load traces and collect in a single array'''
        os.chdir(self.path)
        data = numpy.empty((1000, 2 * self.window_w + self.extra_columns))
        weights = numpy.empty(1000)
        data_counter = 0
        self.event_counter = 0
        for session_data, cell_data in self.pull_dict.items():
            path, prefix, tag, ch = session_data
            os.chdir(path)
            lines = numpy.load(self.get_numpy_name(prefix, tag, ch))
            new_length = data_counter + len(lines)
            if new_length > len(data):
                data = numpy.append(data, numpy.empty((len(lines) + 1000, data.shape[1])), axis=0)
                if self.uses_weights:
                    weights = numpy.append(weights, numpy.empty(len(lines) + 1000))
            data[data_counter:new_length] = lines
            # weights: add each weight to the corresponding line
            if self.uses_weights:
                wdat = numpy.load(self.get_numpy_name(prefix, tag, ch, weight=True))
                weights[data_counter:new_length] = wdat
            data_counter = new_length
        self.data = data[:data_counter]
        if self.uses_weights:
            self.weights = weights[:data_counter]

    def get_data(self, group_criteria=None, group_by=None, avg_cells=False, return_indices=False, keep_all=False,
                 weighted=False, baseline=None, mask_outliers=False):
        '''group criteria provided as a list of key-value pairs applied sequentially
        group_by: average items that have the same value in the specified parameter
        return_indices: also return the original cell indices for each line
        if group_by is specified, returns group keys instead of indices
        If baseline is specified as a slice, this period is subtracted from each individual event'''
        if group_criteria is None and group_by is None and avg_cells is False:
            data = self.data[:, :2 * self.window_w]
            if weighted:
                return numpy.copy(data), numpy.copy(self.weights)
        else:
            index_pool = self.group_dict.keys()
            if group_criteria is not None:
                # find unique IDs of cells in group
                for key, value in group_criteria:
                    indices = []
                    for i in index_pool:
                        if value is numpy.nan:
                            try:
                                if numpy.isnan(self.group_dict[i][key]):
                                    indices.append(i)
                            except:
                                pass
                        elif self.group_dict[i][key] == value:
                            indices.append(i)
                    index_pool = indices  # these are dict keys of cells
            trace_indexer = self.field_names.index('TraceIndex')
            if not group_by:
                # build list of array indices from included items:
                indices = numpy.array([self.item_dict[i][trace_indexer] for i in
                                       index_pool])  # to access traces belonging to unique cells
                if len(indices) > 0:
                    # get line indexes of all traces belonging to selected units
                    element_included = numpy.isin(self.get_unique_indices(),
                                                  indices)  # traces that belong to filtered cells
                    data = self.data[element_included, :2 * self.window_w]  # values
                    if weighted:
                        return data, self.weights[element_included]
                    cell_indices = self.data[element_included, -2]  # cell identifiers for averaging later
                    if avg_cells:
                        # compute mean of lines belonging to the same unique cell. maintain cells that have no events as nan
                        unique_cell_ids = self.data[element_included, -1]
                        data, cell_indices = self.collapse_cells(data, unique_cell_ids, cell_indices, indices,
                                                                 keep_all=keep_all, mask_outliers=mask_outliers)
                else:
                    data = None
                    cell_indices = None
            else:
                groups = self.get_groups_by(group_by)
                data = numpy.empty((len(groups), 2 * self.window_w))
                data[:] = numpy.nan
                for gi, grp in enumerate(groups):
                    sub_indices = []
                    for i in index_pool:
                        if self.group_dict[i][group_by] == grp:
                            sub_indices.append(i)
                    if len(sub_indices) < 1:
                        continue
                    indices = numpy.array([self.item_dict[i][trace_indexer] for i in sub_indices])
                    element_included = numpy.isin(self.get_unique_indices(), indices)
                    data[gi] = numpy.nanmean(self.data[element_included, :2 * self.window_w], axis=0)
                cell_indices = groups
        if baseline is not None:
            nd = numpy.empty(data.shape, data.dtype)
            for i in range(len(data)):
                nd[i] = data[i] - numpy.nanmean(data[i][baseline])
            data = numpy.copy(nd)
        if data is None:
            if return_indices:
                return None, None
            return None
        if return_indices:
            return numpy.copy(data), cell_indices
        else:
            return numpy.copy(data)

    def collapse_cells(self, data, unique_cell_ids, cell_indices, indices, keep_all, mask_outliers=False):
        '''input pre-selected lines, and array of unique ids and original indices
        returns avg trace by cell, and the orig index for each
        if keep_all: cells that are included but have no events are included as nans
            - allows comparing multiple pulls from the same PSTH
        '''
        if keep_all:
            groups = indices
        else:
            groups = numpy.unique(unique_cell_ids)
        mean_data = numpy.empty((len(groups), 2 * self.window_w))
        mean_indices = numpy.empty(len(groups))
        mean_data[:] = numpy.nan
        mean_indices[:] = numpy.nan
        for gi, grp in enumerate(groups):
            wh = numpy.where(unique_cell_ids == grp)[0]
            if len(wh) > 0:
                Y = data[wh, :2 * self.window_w]
                #set outliers to nan
                if mask_outliers:
                    for t in range(Y.shape[1]):
                        ol_index = outlier_indices(Y[:, t])
                        Y[ol_index, t] = numpy.nan
                mean_data[gi] = numpy.nanmean(Y, axis=0)
                mean_indices[gi] = cell_indices[wh[0]]
        return mean_data, mean_indices

    def get_groups_by(self, group_name):
        if group_name not in self.group_names:
            raise ValueError('Group name invalid')
        name_list = list(self.group_df[group_name].unique())
        try:
            name_list.sort()
        except:
            pass
        return name_list

    def return_weighted_mean(self, group_criteria=None, multiply=1, err_type='sem'):
        '''compute the weighted mean and SEM
        Note that group_by and avg_cells makes no sense for this'''
        assert self.uses_weights
        data, weights = self.get_data(group_criteria=group_criteria, group_by=None, avg_cells=False, weighted=True)
        mid = numpy.empty(2 * self.window_w)
        err = numpy.empty(2 * self.window_w)
        if not numpy.any(numpy.logical_not(numpy.isnan(data))):
            mid[:] = numpy.nan
            return mid, mid, mid
        data *= multiply
        # incl_nan = numpy.logical_not(numpy.isnan(data[:, self.window_w])) this gives incomplete traces
        incl_nan = numpy.logical_not(numpy.any(numpy.isnan(data), axis=1))
        weights -= weights[incl_nan].min()
        weights /= weights[incl_nan].max()
        incl_w = weights > 0.1
        incl = numpy.logical_and(incl_nan, incl_w)
        for t in range(2 * self.window_w):
            weighted_stats = DescrStatsW(data[incl, t], weights=weights[incl], ddof=0)
            mid[t] = weighted_stats.mean
            if err_type == 'sem':
                err[t] = weighted_stats.std_mean
            elif err_type == 'SD':
                err[t] = weighted_stats.std
        return mid, mid - err, mid + err

    def return_mean(self, group_criteria=None, spread_mode='SEM', mid_mode='mean', group_by=None, avg_cells=True,
                    multiply=1, print_n=False, scale=False, use_n=None, baseline=None, mask_outliers=False):
        '''# filter data according to criteria, and calculate average of all lines.
        if group_by is specified, first average by that parameter.
        if avg cells, cell averages instead of all individual trace. has no effect with group_by'''
        data = self.get_data(group_criteria=group_criteria, group_by=group_by,
                             avg_cells=avg_cells, baseline=baseline, mask_outliers=mask_outliers)
        if data is None:
            return None, None, None
        return self.calc_mean(data=data, spread_mode=spread_mode, mid_mode=mid_mode,
                    multiply=multiply, print_n=print_n, scale=scale, use_n=use_n)

    @staticmethod
    def calc_mean(data, spread_mode='SEM', mid_mode='mean', multiply=1, scale=False, use_n=None,
                  print_n=False, group_criteria=None):
        '''separated out from return_mean so can be used as static method'''
        data *= multiply
        if scale:
            data /= numpy.nanmax(numpy.nanmean(data, axis=0))
        # determine mid values:
        if mid_mode == 'median':
            mid = numpy.nanmedian(data, axis=0)
        elif mid_mode == 'mean':
            mid = numpy.nanmean(data, axis=0)
        # determine spread values
        if spread_mode == 'IQR':
            lower = numpy.nanpercentile(data, 25, axis=0)
            upper = numpy.nanpercentile(data, 75, axis=0)
        elif spread_mode == 'SD':
            sd = numpy.nanstd(data, axis=0)
            lower = mid - sd
            upper = mid + sd
        elif spread_mode == 'SEM':
            if use_n is None:
                use_n = numpy.count_nonzero(numpy.logical_not(numpy.isnan(data)), axis=0)
            if print_n:
                print(group_criteria, ', n =', use_n)
            sem = numpy.nanstd(data, axis=0) / numpy.sqrt(use_n)
            lower = mid - sem
            upper = mid + sem
        else:
            lower, upper = None, None

        return mid, lower, upper



def pull_session_with_mask(session:ImagingSession, mask, param_key='rel', ext_line=None):
    '''
    Get the mean response of all cells in a session
    :param session:
    :param event: output of EventMasks
    :param mask:
    :param ext_line: if supplied, use this trace instead of getparam
    :return: response array, ncells x 2*w
    '''
    w = mask.shape[1] // 2
    data = numpy.empty((len(mask), session.ca.cells, 2 * w))
    if ext_line is None:
        param = session.getparam(param_key)
    else:
        param = numpy.zeros(session.ca.rel.shape)
        l_dat = min(len(ext_line), session.ca.frames)
        param[:, :l_dat] = ext_line[:l_dat]
    # pull values
    for ei, indices in enumerate(mask):
        lines = numpy.empty((session.ca.cells, 2 * w))
        lines[:] = numpy.nan
        loc = numpy.where(numpy.logical_not(numpy.isnan(indices)))[0]
        lines[:, loc] = param[:, indices[loc].astype(numpy.int64)]
        data[ei] = lines
    return numpy.nanmean(data, axis=0)

# testing
if __name__ == '__main__':
    path = 'X:/Barna/pvsncg/'
    prefix = 'pvsncg_222_380'
    tag = 'A'
    os.chdir(path)
    session = ImagingSession(prefix, tag='A')
    cells = [2, 3]
    grouping = [('control', 'cell'), ('control', 'ax')]
    group_names = ('drug', 'type')
    # grouping = ['cell', 'ax']
    identifier = 'stop'
    e = PSTH(path, identifier)
    e.add_items(session, cells, grouping=grouping, group_names=group_names)
