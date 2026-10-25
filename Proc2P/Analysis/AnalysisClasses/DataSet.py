import json
import os
import ast
import datetime
from pathlib import Path

import numpy
import pandas

from BaserowAPI.BaserowRequests import GetSessions


# Column types. Each column holds one type and allows missing values (pandas.NA, or NaT for dates).
# The first non-missing value written to a column sets its type.
DTYPES = {bool: 'boolean', int: 'Int64', float: 'Float64', str: 'string', pandas.Timestamp: 'datetime64[ns]'}


def is_missing(value):
    if value is None or value is pandas.NA or value is pandas.NaT:
        return True
    return isinstance(value, (float, numpy.floating)) and numpy.isnan(value)


def value_type(value):
    '''The column type that a single value belongs to, or None if DataSet can't store it.'''
    if isinstance(value, (bool, numpy.bool_)):
        return bool
    if isinstance(value, (int, numpy.integer)):
        return int
    if isinstance(value, (float, numpy.floating)):
        return float
    if isinstance(value, str):
        return str
    if isinstance(value, (datetime.datetime, numpy.datetime64)):
        return pandas.Timestamp
    return None


def type_name(t):
    return 'Timestamp' if t is pandas.Timestamp else t.__name__


def convert(value, to):
    '''
    Explicit conversion used when a setter gets type=...
    Missing values (and '' for non-str types) become missing. bool accepts 0/1 and 'true'/'false' (any case).
    int accepts floats with no fractional part and numeric strings.
    '''
    if to not in DTYPES:
        raise ValueError(f'type must be one of {", ".join(type_name(t) for t in DTYPES)}, got {to!r}')
    if is_missing(value) or (to is not str and isinstance(value, str) and not value.strip()):
        return pandas.NA if to is not pandas.Timestamp else pandas.NaT
    if to is bool:
        if isinstance(value, (bool, numpy.bool_)):
            return bool(value)
        if isinstance(value, (int, float, numpy.number)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ('true', 'false', '1', '0'):
            return value.strip().lower() in ('true', '1')
    elif to is int:
        if isinstance(value, (float, numpy.floating)):
            if float(value).is_integer():
                return int(value)
        elif isinstance(value, str):
            return int(value.strip())
        else:
            return int(value)
    elif to is float:
        return float(value)
    elif to is str:
        return str(value)
    elif to is pandas.Timestamp:
        return pandas.Timestamp(value)
    raise ValueError(f'Cannot convert {value!r} ({type(value).__name__}) to {type_name(to)}')


def column_type(series):
    '''
    Returns the column's type (a key of DTYPES), None if the column has no values yet,
    or a set of types if the column holds mixed types.
    '''
    dtype = series.dtype
    if pandas.api.types.is_bool_dtype(dtype):
        return bool
    if pandas.api.types.is_integer_dtype(dtype):
        return int
    if pandas.api.types.is_float_dtype(dtype):
        return float if series.notna().any() else None
    if pandas.api.types.is_string_dtype(dtype) and not pandas.api.types.is_object_dtype(dtype):
        return str
    if pandas.api.types.is_datetime64_any_dtype(dtype):
        return pandas.Timestamp
    types = set()
    for x in series:
        if not is_missing(x):
            types.add(value_type(x))
    if not types:
        return None
    if types == {int, float}:
        return float
    if len(types) == 1 and None not in types:
        return types.pop()
    return types


class DataSet:
    '''
    A table of sessions (one row per prefix) saved as a versioned feather file in the project folder.
    Each column holds a single type: boolean, Int64, Float64, string or datetime, with missing values allowed.
    The first value written to a column sets its type. Writing a value of another type raises a TypeError;
    pass type=... to the setter to convert the value (and the column, if needed).
    '''
    __name__ = 'DataSet'

    def __init__(self, project_folder, ver=None):
        '''
        :param project_folder: a unique place (in Processed, backed up) for analysis outputs related to the dataset
        :param ver: an int. if None, last available is read.
        '''

        self.project_folder = Path(os.path.realpath(project_folder))
        self.prefix = '_dataset'
        self.ext = '.feather'
        if ver is not None:
            self.ver = int(ver)
        else:
            self.get_current_ver()
        self.load_df()
        self.readonly_flag = False
        self.db = None

    def __contains__(self, item):
        return item in self.df["Prefix"].values

    def __getitem__(self, item):
        return self.df.loc[self.df["Prefix"].eq(item)].iloc[0]

    def get_fn(self, ext=None):
        if ext is None:
            ext = self.ext
        return self.project_folder / f'{self.prefix}{self.ver:02}{ext}'

    def get_current_ver(self, incr=0):
        flist = os.listdir(self.project_folder)
        ds = [x for x in flist if x.endswith(self.ext) and x.startswith(self.prefix)]
        if not len(ds):
            self.ver = 0
        else:
            vs = [int(x[len(self.prefix):-len(self.ext)]) for x in ds]
            self.ver = max(vs) + incr

    def load_df(self):
        fn = self.get_fn()
        if os.path.exists(fn):
            self.df = pandas.read_feather(fn)
            self.normalize_df(strict=False)
        else:
            self.new_df()

    def normalize_column(self, key):
        '''Convert a column to its nullable dtype. Covers older files and columns that scripts set on self.df directly.'''
        ct = column_type(self.df[key])
        if type(ct) is set:
            names = ', '.join(sorted(type_name(t) if t is not None else 'unsupported' for t in ct))
            raise TypeError(f"Column '{key}' holds mixed types ({names}). "
                            f"Call ds.set_type('{key}', <type>) to convert it.")
        if ct is None:
            if not (pandas.api.types.is_object_dtype(self.df[key].dtype) and self.df[key].map(lambda x: x is None).all()):
                self.df[key] = pandas.Series([None] * len(self.df), index=self.df.index, dtype=object)
        elif str(self.df[key].dtype) != DTYPES[ct]:
            self.df[key] = self.typed_values(key, ct)

    def normalize_df(self, strict=True):
        for key in self.df.columns:
            try:
                self.normalize_column(key)
            except TypeError as e:
                if strict:
                    raise
                print(f'DataSet warning: {e}')

    def typed_values(self, key, to):
        values = []
        for pf, x in zip(self.df["Prefix"], self.df[key]):
            try:
                values.append(convert(x, to))
            except ValueError as e:
                raise TypeError(f"Column '{key}', prefix {pf}: {e}") from None
        return pandas.Series(pandas.array(values, dtype=DTYPES[to]), index=self.df.index)

    def set_type(self, key, to):
        '''Convert all values of a column to a type (bool, int, float, str or pandas.Timestamp).'''
        self.df[key] = self.typed_values(key, to)

    @staticmethod
    def asdict(literal):
        return ast.literal_eval(literal)

    def save_df(self, ver='current'):
        if ver == 'excel':
            self.df.to_excel(self.get_fn(ext='.xlsx'))
            self.report()
        if ver == 'current':
            assert not self.readonly_flag
        elif ver == 'next':
            self.get_current_ver(incr=1)
            self.report()
        self.normalize_df()
        # write to a temporary file first, so an interrupted save does not leave a broken dataset
        fn = self.get_fn()
        tmp_fn = fn.with_name(fn.name + '.tmp')
        self.df.reset_index(drop=True).to_feather(tmp_fn, compression="uncompressed")
        os.replace(tmp_fn, fn)

    def new_df(self):
        self.df = pandas.DataFrame({"Prefix": pandas.Series(dtype=DTYPES[str])})
        for key in ("Animal", "Sex", "Incl", "Excl"):
            self.check_key(key)

    def report(self):
        rtext = f'Dataset v{self.ver}: {len(self.df)} sessions, {len(self.get_incl())} included\n'
        rtext += f'Project folder: {self.project_folder}'
        print(rtext)
        return rtext

    def new_record(self, prefix):
        db = self.check_db()
        animal, sex = db.get_sex(prefix)
        self.add_row({"Prefix": prefix, "Animal": animal, "Sex": sex, "Incl": True})

    def add_row(self, fields):
        # reindex fills the new row with missing values and keeps the column dtypes
        self.normalize_df()
        index = self.df.index.max() + 1 if len(self.df) else 0
        self.df = self.df.reindex(self.df.index.append(pandas.Index([index])))
        for key, value in fields.items():
            self.set_value(index, key, value)

    def check_db(self):
        if self.db is None:
            self.db = GetSessions()
        return self.db

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

    def exclude(self, prefix, tag=None, value=True):
        '''
        Mark excluded.
        :param prefix: a prefix (or a list of prefixes)
        :param tag: optional alternative excl tag
        '''
        exclfield = 'Excl'
        if tag is not None:
            exclfield += f'.{tag}'
        self.set_field(prefix, exclfield, value)

    def set_field(self, prefix, key, value=True, type=None):
        '''
        :param type: optional (bool, int, float, str or pandas.Timestamp). Converts the value, and the column if it
        has a different type.
        '''
        prefix = self.listify(prefix)
        for pf in prefix:
            index = self.df.loc[self.df["Prefix"].eq(pf)].index
            if not len(index) == 1:
                raise ValueError(f'Prefix should have exactly one match, {pf} had {len(index)}')
            self.set_value(index[0], key, value, type)

    def set_by_dict(self, prefix, add_fields, type=None):
        '''
        :param type: optional, a type for all fields, or a dict of {key: type} for some of them. See set_field.
        '''
        index = self.df.loc[self.df["Prefix"].eq(prefix)].index
        if not len(index) == 1:
            raise ValueError(f'Prefix should have exactly one match, {prefix} had {len(index)}')
        for key, value in add_fields.items():
            key_type = type.get(key) if isinstance(type, dict) else type
            self.set_value(index[0], key, value, key_type)

    def set_value(self, index, key, value, to=None):
        self.check_key(key)
        if to is None:
            self.normalize_column(key)
        ct = column_type(self.df[key])
        if to is not None:
            if ct != to:
                self.set_type(key, to)
            try:
                value = convert(value, to)
            except ValueError as e:
                raise ValueError(f"Column '{key}': {e}") from None
        elif is_missing(value):
            value = pandas.NaT if ct is pandas.Timestamp else (pandas.NA if ct is not None else None)
        else:
            vt = value_type(value)
            if vt is None:
                raise TypeError(f"Column '{key}': DataSet cannot store {value!r} ({type(value).__name__}). "
                                f"Pass type=str to store it as text.")
            if ct is None:
                self.df[key] = pandas.Series(pandas.array([pandas.NA] * len(self.df), dtype=DTYPES[vt]),
                                             index=self.df.index)
            elif not (vt == ct or (ct is float and vt is int)):
                raise TypeError(f"Column '{key}' has type {type_name(ct)}, the value {value!r} has type {type_name(vt)}. "
                                f"Pass type={type_name(ct)} to convert the value, or type={type_name(vt)} to convert the column.")
            if vt is pandas.Timestamp:
                value = pandas.Timestamp(value)
            elif vt is not str:
                value = vt(value)
        self.df.at[index, key] = value

    def listify(self, prefix):
        if type(prefix) is not list:
            list_prefix = [prefix, ]
        else:
            list_prefix = list(set(prefix))
            list_prefix.sort()
        return list_prefix

    def get_field(self, prefix, key):
        if key not in self.df.columns:
            return False
        return self.df.loc[self.df["Prefix"].eq(prefix)].iloc[0][key]

    def check_key(self, key, value=None):
        if key not in self.df.columns:
            if is_missing(value):
                self.df[key] = pandas.Series([None] * len(self.df), index=self.df.index, dtype=object)
            else:
                self.df[key] = pandas.Series(pandas.array([value] * len(self.df), dtype=DTYPES[value_type(value)]),
                                             index=self.df.index)

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
            if cells is None:
                clist = cells
            else:
                old_val = self.get_field(prefix, cellfield)
                if isinstance(old_val, str):
                    old_list = json.loads(old_val)
                else:
                    old_list = None
                if old_list is None:
                    clist = cells
                elif type(old_list) is list:
                    old_list.extend(cells)
                    clist = old_list
                else:
                    raise ValueError(f'Not sure what to do with this cell list: {cells}, current is {old_list}')
        if type(clist) is list:
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
                value = self.get_field(prefix, cellfield)
                # rows added after the column was made hold a missing value instead of 'null'
                rets[suffix] = None if is_missing(value) else json.loads(value)
        if 'Incl' not in rets:
            return None
        if 'Excl' not in rets:
            return rets['Incl']
        else:
            incl = rets['Incl']
            excl = rets['Excl']
            if incl is None:
                return (incl, excl)  # we don't have the max n of cells, need to look up externally
            if type(excl) is list:
                return [c for c in incl if c not in excl]
            else:
                return incl

    def get_cells(self, *args, **kwargs):
        cells = self._get_cells(*args, **kwargs)
        if type(cells) is not list:
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
        sub = self.df.loc[~self.is_true(self.df["Excl"])]
        if excl is not None and excl in self.df.columns:
            sub = sub.loc[~self.is_true(sub[excl])]
        return sub.loc[self.is_true(sub[key])]

    @staticmethod
    def is_true(series):
        # missing values count as not True
        return series.eq(True).fillna(False).astype(bool)
