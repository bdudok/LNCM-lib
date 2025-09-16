import json
import os

import pandas

from BaserowAPI.BaserowRequests import GetSessions


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
            self.report()
        if ver == 'current':
            assert not self.readonly_flag
        elif ver == 'next':
            self.get_current_ver(incr=1)
            self.report()
        if self.mod_flag:
            self.df.to_feather(self.get_fn(), compression="uncompressed")
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
            raise ValueError(f'Prefix should have exactly one match, {prefix} had {len(index)}')
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

    def _get_cells(self, prefix, roi_tag, alt_tag=None):
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

    def get_cells(self, *args, **kwargs):
        cells = self._get_cells(*args, **kwargs)
        if not type(cells) is list:
            return None
        if not len(cells):
            return None
        return cells

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
