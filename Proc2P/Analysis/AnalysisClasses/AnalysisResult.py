import base64
import hashlib
import html
import inspect
import json
import marshal
import os
import shutil
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy
import pandas
import requests

from BaserowAPI.BaserowRequests import GetSessions
from BaserowAPI.config import config as baserow_config

LIB_ROOT = Path(__file__).resolve().parents[3]
CONFIRMED_ENV = 'LNCM_UNCOMMITTED_CONFIRMED'  # set for the plot script that save() runs: already confirmed
REPORT_MAX_ROWS = 50  # rows of a data table shown in the report; the file has all rows
REPORT_MAX_CHARS = 80  # characters of a text cell shown in the report tables (e.g. the cell lists of the DataSet)

REPORT_STYLE = '''
body {font-family: Arial, sans-serif; margin: 2em auto; max-width: 1100px; padding: 0 1em; color: #222;}
h1 {font-size: 1.5em;} h2 {font-size: 1.2em; border-bottom: 1px solid #ccc; margin-top: 2em;}
h3 {font-size: 1em; margin-bottom: 0.3em;}
img {max-width: 100%;}
table {border-collapse: collapse; font-size: 0.8em;}
th, td {border: 1px solid #ccc; padding: 2px 6px; text-align: right;}
th {background: #eee;}
pre {background: #f4f4f4; padding: 0.5em; overflow-x: auto;}
.scroll {overflow-x: auto;}
.note {color: #666; font-size: 0.85em;}
.warn {color: #b00; font-weight: bold;}
'''


def git_state(path):
    '''
    Commit of the repo that contains path, and if path has uncommitted changes; None if it can't be read.
    A file: 'dirty' if that file is modified or not tracked; other files of the repo don't count.
    A folder: 'dirty' if any tracked file of the repo is modified; untracked files don't count.
    '''
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    status_args = ('--untracked-files=no',) if path.is_dir() else ('--', str(path))

    def git(*args):
        return subprocess.run(['git', '-C', str(folder), *args], capture_output=True, text=True, check=True).stdout

    try:
        root = Path(git('rev-parse', '--show-toplevel').strip())
        return {'repo': root.name, 'commit': git('rev-parse', 'HEAD').strip(),
                'dirty': bool(git('status', '--porcelain', *status_args).strip()), 'root': root}
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def plain_value(value):
    '''A json value that stands for value in a fingerprint (arrays and tables by their hash); TypeError if value is
    not data (e.g. an object, a module or a function).'''
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (list, tuple)):
        return [plain_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain_value(v) for k, v in value.items()}
    if isinstance(value, numpy.ndarray):
        data = str(value.tolist()).encode() if value.dtype == object else numpy.ascontiguousarray(value).tobytes()
        return {'ndarray': sha256(data), 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, (pandas.DataFrame, pandas.Series)):
        return {'pandas': sha256(pandas.util.hash_pandas_object(value).values.tobytes())}
    raise TypeError(f'not a data value: {type(value).__name__}')


def source_hash(obj, codes):
    '''hash of the source of a function or class; of its compiled code if the source can't be read (e.g. defined in
    the console), which also changes when its lines move'''
    try:
        return sha256(inspect.getsource(obj).encode())
    except (OSError, TypeError):
        return 'code ' + sha256(b''.join(marshal.dumps(c) for c in codes))


def code_names(code):
    '''names that a code object and the code objects inside it (comprehensions, inner functions) look up'''
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= code_names(const)
    return names


def code_fingerprint(func):
    '''
    What the result of func depends on in its script:
        sources: hash of the source of func and of the functions and classes of its module that it uses, recursively
        values: the data values (numbers, strings, paths, lists, dicts, arrays, tables) of the globals they use
        fingerprints: fingerprint() of the objects they use that have one (e.g. a DataSet)
        library: names of modules, and functions and classes from other modules (e.g. LNCM-lib): not tracked
        untracked: other objects they use, which are not tracked
    '''
    module = func.__module__
    fp = {'sources': {}, 'values': {}, 'fingerprints': {}, 'library': set(), 'untracked': set()}
    todo = [func]
    while todo:
        f = todo.pop()
        if f.__qualname__ in fp['sources']:
            continue
        if inspect.isclass(f):
            codes = [m.__code__ for m in vars(f).values() if inspect.isfunction(m)]
            scope, module_globals = {}, vars(sys.modules[module])
        else:
            codes = [f.__code__]
            scope = {name: cell.cell_contents for name, cell in zip(f.__code__.co_freevars, f.__closure__ or ())}
            module_globals = f.__globals__
        fp['sources'][f.__qualname__] = source_hash(f, codes)
        names = set().union(*(code_names(c) for c in codes)) | set(scope)
        for name in sorted(names):
            if name in scope:
                value = scope[name]
            elif name in module_globals:
                value = module_globals[name]
            else:
                continue  # a builtin, or an attribute name
            if (inspect.isfunction(value) or inspect.isclass(value)) and value.__module__ == module:
                todo.append(value)
            elif inspect.ismodule(value) or inspect.isroutine(value) or inspect.isclass(value):
                fp['library'].add(name)
            elif callable(getattr(value, 'fingerprint', None)):
                fp['fingerprints'][name] = value.fingerprint()
            else:
                try:
                    fp['values'][name] = plain_value(value)
                except TypeError:
                    fp['untracked'].add(name)
    fp['library'] = sorted(fp['library'])
    fp['untracked'] = sorted(fp['untracked'])
    return fp


def fingerprint_changes(old, new):
    '''names of the sources, values and fingerprints that differ between two fingerprints'''
    changes = []
    for section in ('sources', 'values', 'fingerprints'):
        a, b = old.get(section, {}), new.get(section, {})
        for name in sorted(set(a) | set(b)):
            if json.dumps(a.get(name), sort_keys=True) != json.dumps(b.get(name), sort_keys=True):
                changes.append(f'{section[:-1]} {name}')
    return changes


# column of the mouse table: field in Baserow
MOUSE_FIELDS = {'Sex': 'Sex*', 'Strain': 'Strain*', 'DOB': 'DOB*', 'FirstSurgery': baserow_config['surgery_field']}
AGE_MAX = 1000  # days; a larger age, or one that is not positive, gets a warning


def read_mouse_db(animals, prefixes):
    '''
    Read the mouse entries and the dates of the sessions from Baserow.
    :return: {animal: {'Sex', 'Strain', 'DOB', 'FirstSurgery'}}, {prefix: date}. None for the entries that are not
        found and for the fields that are empty.
    '''
    db = GetSessions(timeout=30)
    mice = {}
    for animal in animals:
        rows = db.get_mouse(mtag=animal)
        mice[animal] = {key: empty_to_none(rows.iloc[0].get(field)) for key, field in MOUSE_FIELDS.items()} \
            if len(rows) else None
    dates = {}
    for prefix in prefixes:
        rows = db.get_session(prefix)  # Image.ID is the prefix with a suffix (-000)
        dates[prefix] = rows.iloc[0]['Date'] if len(rows) else None
    return mice, dates


def empty_to_none(value):
    '''A field of a Baserow row: None if it is empty (None, '', or 'None' as text).'''
    return None if value is None or str(value).strip() in ('', 'None') else value


def age_warnings(mice):
    '''Text for each age column of the mouse table that has a missing, not positive or too large value.'''
    labels = {'AgeFirstSurgery': 'Age at the first surgery', 'AgeFirstSession': 'Age at the first session'}
    warnings = []
    for key, label in labels.items():
        days = mice[key]
        for problem, wh in (('missing', days.isna()), ('0 or negative', (days <= 0).fillna(False)),
                            (f'over {AGE_MAX} days', (days > AGE_MAX).fillna(False))):
            if wh.any():
                warnings.append(f'{label} is {problem} for: {", ".join(mice.loc[wh.astype(bool), "Animal"])}.')
    return warnings


class AnalysisResult:
    '''
    The record of one result (usually a figure): the figures, the tables behind them, the statistics, the list of
    sessions used, and the code version. Saved in <project_folder>/Results/<identifier>/:
        manifest.json: script, commits of the script's repo and of LNCM-lib, DataSet version, parameters, contents
        sessions.feather: one row per session used (the DataSet rows, if a DataSet was given)
        mice.feather: one row per animal of the sessions (if a DataSet was given): sex, strain, date of birth,
            and the age at the first surgery and at the first session used in the result. Read from Baserow.
        figures/, data/, stats/: the items added
        script/: a copy of the script as it was run
        <identifier>.html: the report, all of the above on one page
    Usage:
        res = AnalysisResult(project_folder, 'Fig3_RunStop', steps=['3c_PSTHRunStop.py'])
        res.set_sessions(prefixes, dataset=ds)
        res.add_data('psth', df); res.add_stat('mixed_model', table); res.add_figure('psth', fig)
        res.add_test('run vs rest', 'paired t-test', 't', 2.31, df=11, n='12 sessions, 4 animals', p=0.041)
        res.save()
    Items are written when added; save() writes the manifest and the report, and removes items of earlier runs.
    If the script, the plot script or LNCM-lib has uncommitted changes, creating the result waits for enter.

    Data that takes long to compute (e.g. opening the sessions) can be cached, so that changing the statistics or
    figures after it does not compute it again (see cached_data):
        def compute_responses():  # the session loop; uses ds and the settings at the top of the script
            ...
            return responses
        responses = res.cached_data('responses', compute_responses)

    The figures can be made by a separate plot script, so that a plot can be changed without running the analysis
    again. The analysis script names it (plot_script='..._plot.py') and adds no figures; its save() runs the plot
    script. The plot script uses only what is stored in the result:
        res = AnalysisResult.load(project_folder, 'Fig3_RunStop')
        df = res.get_data('psth'); window = res.params['window']
        res.add_figure('psth', fig)
        res.save()
    '''
    __name__ = 'AnalysisResult'

    def __init__(self, project_folder, identifier, script=None, steps=None, title=None, description=None,
                 plot_script=None, processing_steps=None):
        '''
        :param project_folder: the project folder of the analysis (as for DataSet)
        :param identifier: unique name of the result within the project, used as the folder name
        :param script: path of the script that makes the result. Default: the script being run.
        :param plot_script: optional script that makes the figures from the saved result (see the class docstring).
            A file name is looked up in the folder of script. save() runs it.
        :param steps: file names of the analysis scripts that have to be run before this one, in run order.
            They are shared with the result: they run on the processed sessions.
        :param processing_steps: file names of the scripts that processed the sessions before the analysis (e.g.
            registration, pulling traces, cell screening), in run order. Recorded only; they are not shared.
        :param title, description: shown in the report
        '''
        self.project_folder = Path(project_folder)
        self.identifier = identifier
        self.folder = self.project_folder / 'Results' / identifier
        if script is None:
            script = getattr(sys.modules['__main__'], '__file__', None)
        self.script = Path(script).resolve() if script is not None else None
        if plot_script is not None and self.script is not None:
            plot_script = self.script.parent / plot_script  # stays as given if plot_script is an absolute path
        self.plot_script = Path(plot_script).resolve() if plot_script is not None else None
        self.loaded_manifest = None
        self.steps = list(steps) if steps is not None else []
        self.processing_steps = list(processing_steps) if processing_steps is not None else []
        self.title = title if title is not None else identifier
        self.description = description
        self.params = {}
        self.items = {'figures': [], 'data': [], 'stats': []}
        self.tests = []
        self.sessions = None
        self.mice = None
        self.mice_error = None
        self.upload_sessions = True
        self.dataset_info = None
        self.cache_records = {}
        self.folder.mkdir(parents=True, exist_ok=True)
        self.confirm_uncommitted()

    @classmethod
    def load(cls, project_folder, identifier, script=None):
        '''
        Open a saved result in a plot script. The data, stats, tests and parameters are those of the saved result;
        the figures are the ones this script adds. save() then updates the figures and the report.
        :param script: path of the plot script. Default: the script being run.
        '''
        fn = Path(project_folder) / 'Results' / identifier / 'manifest.json'
        if not fn.exists():
            raise FileNotFoundError(f'Result {identifier} is not saved: {fn} not found. Run its analysis script first.')
        with open(fn, encoding='utf-8') as f:
            m = json.load(f)
        res = cls(project_folder, identifier, script=script, steps=m['steps'], title=m['title'],
                  description=m['description'], processing_steps=m['processing_steps'])
        res.loaded_manifest = m
        res.params = m['params']
        res.items = {'figures': [], 'data': m['data'], 'stats': m['stats']}
        res.tests = m['tests']
        res.sessions = pandas.read_feather(res.get_fn('sessions'))
        if res.get_fn('mice').exists():
            res.mice = pandas.read_feather(res.get_fn('mice'))
        res.upload_sessions = m['upload_sessions']
        res.dataset_info = m['dataset']
        res.cache_records = m.get('cache', {})
        return res

    def confirm_uncommitted(self):
        '''Wait for enter if the script, the plot script or LNCM-lib has uncommitted changes.'''
        if os.environ.get(CONFIRMED_ENV):
            return
        dirty = [path.name for path in (self.script, self.plot_script, LIB_ROOT)
                 if path is not None and (git_state(path) or {}).get('dirty')]
        if dirty:
            input(f'Uncommitted changes in analysis code ({", ".join(dirty)}); press enter to continue')

    def get_fn(self, kind, name=None, ext=''):
        '''
        :param kind: 'manifest', 'sessions', 'mice', 'report', or the folder of an item: 'figures', 'data', 'stats', 'script'
        '''
        fixed = {'manifest': 'manifest.json', 'sessions': 'sessions.feather', 'mice': 'mice.feather',
                 'report': f'{self.identifier}.html'}
        if kind in fixed:
            return self.folder / fixed[kind]
        (self.folder / kind).mkdir(exist_ok=True)
        return self.folder / kind / f'{name}{ext}'

    def set_params(self, **kwargs):
        '''Analysis parameters to record in the manifest (values that json can't store are stored as text).'''
        self.params.update(kwargs)

    def set_sessions(self, prefixes, dataset=None, upload=True):
        '''
        The sessions that the result is computed from.
        :param prefixes: session prefixes
        :param dataset: optional DataSet. Its rows of these sessions are stored, with the dataset version, and
            the table of the animals is made (see mouse_table).
        :param upload: False if publishing should not export and upload these sessions (e.g. the entries are
            confocal images, not imaging sessions). Stored in the manifest as 'upload_sessions'.
        '''
        self.upload_sessions = bool(upload)
        prefixes = sorted(set(prefixes))
        if dataset is None:
            self.sessions = pandas.DataFrame({'Prefix': pandas.array(prefixes, dtype='string')})
            self.dataset_info = None
        else:
            missing = [pf for pf in prefixes if pf not in dataset]
            if missing:
                raise ValueError(f'Sessions not in the DataSet: {", ".join(missing)}')
            df = dataset.df.loc[dataset.df['Prefix'].isin(prefixes)]
            self.sessions = df.sort_values('Prefix').reset_index(drop=True)
            self.dataset_info = {'project_folder': str(dataset.project_folder), 'version': int(dataset.ver),
                                 'fingerprint': dataset.fingerprint()}
        self.sessions.to_feather(self.get_fn('sessions'))
        self.mice, self.mice_error = None, None
        self.get_fn('mice').unlink(missing_ok=True)
        if dataset is not None:
            try:
                self.mice = self.mouse_table()
                self.mice.to_feather(self.get_fn('mice'))
                for warning in age_warnings(self.mice):
                    print(f'AnalysisResult {self.identifier} WARNING: {warning}')
            except requests.RequestException as e:
                self.mice_error = f'{type(e).__name__}: {e}'
                print(f'AnalysisResult {self.identifier}: the animals could not be read from Baserow. '
                      f'{self.mice_error}')

    def mouse_table(self):
        '''
        One row per animal of the sessions. The DataSet is the pin on the metadata: Sex, Strain, DOB and FirstSurgery
        of an animal and Date of a session are taken from its columns of these names. Baserow is read only for the
        animals and sessions that have no value there. FirstSession is the
        date of the first session of the animal used in this result, nSessions the number of its sessions used.
        AgeFirstSession and AgeFirstSurgery are in days.
        '''
        sessions = self.sessions.dropna(subset='Animal')
        animals = sorted(sessions['Animal'].unique())

        def pinned(rows, key):
            values = rows[key].dropna() if key in rows else []
            return values.iloc[0] if len(values) else None

        pins = {animal: {key: pinned(sessions.loc[sessions['Animal'] == animal], key)
                         for key in MOUSE_FIELDS} for animal in animals}
        session_dates = pandas.to_datetime(sessions['Date']) if 'Date' in sessions else \
            pandas.Series(pandas.NaT, index=sessions.index)
        look_up_animals = [animal for animal in animals if any(pins[animal][key] is None for key in MOUSE_FIELDS)]
        look_up_prefixes = sessions.loc[session_dates.isna(), 'Prefix']
        entries, dates = {}, {}
        if look_up_animals or len(look_up_prefixes):
            entries, dates = read_mouse_db(look_up_animals, look_up_prefixes)
        session_dates = session_dates.fillna(pandas.to_datetime(sessions['Prefix'].map(dates)))
        rows = []
        for animal in animals:
            entry = entries.get(animal) or {}
            pin = pins[animal]
            wh = sessions['Animal'] == animal
            row = {'Animal': animal}
            for key in MOUSE_FIELDS:
                row[key] = pin[key] if pin[key] is not None else entry.get(key)
            row['FirstSession'] = session_dates[wh].min()
            row['nSessions'] = int(wh.sum())
            rows.append(row)
        mice = pandas.DataFrame(rows)
        for key in ('Animal', 'Sex', 'Strain'):
            mice[key] = mice[key].astype('string')
        for key in ('DOB', 'FirstSurgery', 'FirstSession'):
            mice[key] = pandas.to_datetime(mice[key])
        for key in ('FirstSurgery', 'FirstSession'):
            mice[f'Age{key}'] = (mice[key] - mice['DOB']).dt.days.astype('Int64')
        return mice

    def mice_html(self):
        '''The summary and the table of the animals for the report.'''
        mice = self.mice

        def counts(key):
            return ', '.join(f'{n} {value}' for value, n in mice[key].value_counts().items()) or 'not available'

        def age(key):
            days = mice[key].dropna().astype(float)
            if not len(days):
                return 'not available'
            if len(days) == 1:
                return f'{days.iloc[0]:.0f} days (n = 1)'
            return (f'mean ± SD = {days.mean():.4g} ± {days.std():.4g} days '
                    f'(range {days.min():.0f}–{days.max():.0f}, n = {len(days)})')

        summary = (f'Sex: {counts("Sex")}. Strain: {counts("Strain")}. Age at the first session used in this result: '
                   f'{age("AgeFirstSession")}. Age at the first surgery: {age("AgeFirstSurgery")}.')
        table = mice.dropna(axis=1, how='all')
        for key in table.select_dtypes('datetime').columns:
            table = table.assign(**{key: table[key].dt.strftime('%Y-%m-%d')})
        table = table.astype(object).where(table.notna(), '')  # to_html writes missing strings and Int64 as <NA>
        warnings = ''.join(f'<p class="warn">{html.escape(w)}</p>' for w in age_warnings(mice))
        return f'<p>{html.escape(summary)}</p>' + warnings + self.table_html(table)

    def register(self, kind, name, files, description, **info):
        entries = self.items[kind]
        entries[:] = [e for e in entries if e['name'] != name]  # adding a name again replaces it
        entries.append({'name': name, 'files': [fn.name for fn in files], 'description': description, **info})

    def add_figure(self, name, fig, description=None, formats=('png', 'svg'), dpi=300, text_as_font=True):
        '''
        Save a matplotlib figure. The png is shown in the report.
        :param text_as_font: text in the svg stays text in the font of the figure (12 pt is written as
            'font-size: 12px' in a drawing whose unit is the point). False saves text as outlines.
        '''
        files = []
        with matplotlib.rc_context({'svg.fonttype': 'none' if text_as_font else 'path'}):
            for ext in formats:
                fn = self.get_fn('figures', name, f'.{ext}')
                fig.savefig(fn, dpi=dpi)
                files.append(fn)
        self.register('figures', name, files, description)

    def add_data(self, name, data, description=None):
        '''Save the values behind a figure or a statistic: a DataFrame (feather) or a numpy array (npy).'''
        if isinstance(data, pandas.DataFrame):
            fn = self.get_fn('data', name, '.feather')
            data.reset_index(drop=True).to_feather(fn)
            self.register('data', name, [fn], description, rows=len(data), columns=[str(c) for c in data.columns])
        elif isinstance(data, numpy.ndarray):
            fn = self.get_fn('data', name, '.npy')
            numpy.save(fn, data, allow_pickle=False)
            self.register('data', name, [fn], description, shape=list(data.shape), dtype=str(data.dtype))
        else:
            raise TypeError(f'data should be a DataFrame or a numpy array, got {type(data).__name__}')

    def cached_data(self, name, compute, descriptions=None, recompute=False):
        '''
        Data that takes long to compute (e.g. from opening the sessions), computed only when what it depends on
        changed, so that the statistics and figures after it can be changed and rerun without computing it again.
        :param name: name of the cache record
        :param compute: function without arguments that returns a DataFrame or numpy array, or a dict of them. Its
            result depends on its source, the functions and classes of the script it uses, the data values of the
            globals they use, and the fingerprint() of objects they use (a DataSet: construct it outside compute).
            Modules and LNCM-lib code are not tracked.
        :param descriptions: description of the data item (single result) or {item name: description} (dict result)
        :param recompute: compute even if the saved data matches, e.g. after a change in LNCM-lib
        :return: what compute returned; the items are added as data items (a single result is named name)
        The fingerprint and the code version that computed the data are saved in cache/<name>.json, and in the
        manifest. The data is computed again if the record or a data file is missing, or the fingerprint differs.
        '''
        fp = json.loads(json.dumps(code_fingerprint(compute)))
        record_fn = self.folder / 'cache' / f'{name}.json'
        old = json.loads(record_fn.read_text(encoding='utf-8')) if record_fn.exists() else None
        if old is None:
            reasons = ['no saved data']
        else:
            reasons = fingerprint_changes(old['fingerprint'], fp)
            missing = [item for item, fn in old['items'].items() if not (self.folder / 'data' / fn).exists()]
            reasons += [f'missing file of {item}' for item in missing]
            if recompute:
                reasons.append('recompute=True')
        if not reasons:
            single = not old['dict']
            result = {}
            for item, fn in old['items'].items():
                fn = self.folder / 'data' / fn
                result[item] = numpy.load(fn, allow_pickle=False) if fn.suffix == '.npy' else pandas.read_feather(fn)
                self.add_data(item, result[item], old['descriptions'].get(item))
            lib_now = self.library_record().get('commit')
            script = old['script'] or {}
            print(f'AnalysisResult {self.identifier}: {name} loaded, computed {old["computed"]} with script commit '
                  f'{script.get("commit", "unknown")[:10]}{" (uncommitted changes)" if script.get("dirty") else ""}')
            if old['library'].get('commit') != lib_now:
                print(f'    LNCM-lib changed since then ({old["library"].get("commit", "unknown")[:10]} -> '
                      f'{(lib_now or "unknown")[:10]}): use recompute=True if the change affects {name}')
            elif old['library'].get('dirty'):
                print(f'    LNCM-lib had uncommitted changes when {name} was computed: use recompute=True if they '
                      f'have changed since')
            if fp['untracked']:
                print(f'    not tracked: {", ".join(fp["untracked"])}')
            self.cache_records[name] = {k: v for k, v in old.items() if k != 'fingerprint'}
            return result[name] if single else result

        print(f'AnalysisResult {self.identifier}: computing {name} ({"; ".join(reasons)})')
        value = compute()
        is_dict = isinstance(value, dict)
        result = value if is_dict else {name: value}
        if not isinstance(descriptions, dict):
            descriptions = {name: descriptions}
        files = {}
        for item, data in result.items():
            self.add_data(item, data, descriptions.get(item))
            files[item] = next(e['files'][0] for e in self.items['data'] if e['name'] == item)
        script = git_state(self.script) if self.script is not None else None
        record = {'computed': datetime.now().isoformat(timespec='seconds'), 'dict': is_dict, 'items': files,
                  'descriptions': {k: v for k, v in descriptions.items() if k in files},
                  'script': None if script is None else {k: script[k] for k in ('repo', 'commit', 'dirty')},
                  'library': self.library_record(), 'fingerprint': fp}
        record_fn.parent.mkdir(exist_ok=True)
        record_fn.write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')
        if fp['untracked']:
            print(f'    not tracked: {", ".join(fp["untracked"])}')
        self.cache_records[name] = {k: v for k, v in record.items() if k != 'fingerprint'}
        return value

    def get_data(self, name):
        '''Read a data item of the result: a DataFrame or a numpy array, as it was added.'''
        entries = [e for e in self.items['data'] if e['name'] == name]
        if not entries:
            raise KeyError(f'Result {self.identifier} has no data item {name}. '
                           f'It has: {", ".join(e["name"] for e in self.items["data"])}')
        fn = self.get_fn('data', entries[0]['files'][0])
        if fn.suffix == '.npy':
            return numpy.load(fn, allow_pickle=False)
        return pandas.read_feather(fn)

    def add_stat(self, name, stat, description=None):
        '''Save a statistical result: a DataFrame (feather) or text (txt).'''
        if isinstance(stat, pandas.DataFrame):
            fn = self.get_fn('stats', name, '.feather')
            stat.reset_index(drop=True).to_feather(fn)
        elif isinstance(stat, str):
            fn = self.get_fn('stats', name, '.txt')
            fn.write_text(stat, encoding='utf-8')
        else:
            raise TypeError(f'stat should be a DataFrame or a str, got {type(stat).__name__}')
        self.register('stats', name, [fn], description)

    def add_test(self, comparison, test, statistic, value, df, n, p):
        '''
        Add a statistical test to the 'Statistics reporting' table of the report: what a paper has to state.
        :param comparison: what is compared (e.g. 'Cnr1 vs Rosa26, CB1 intensity')
        :param test: name of the test
        :param statistic: name of the test statistic (e.g. 't', 'U', 'chi2')
        :param value: value of the test statistic
        :param df: degrees of freedom (None if the test has none)
        :param n: sample size, as text if there are several levels (e.g. '14 stacks, 5 animals')
        :param p: exact p value
        '''
        self.tests.append({'comparison': comparison, 'test': test, 'statistic': statistic, 'value': float(value),
                           'df': df, 'n': n, 'p': float(p)})

    def copy_script(self, clear=True):
        '''
        Copy the script into the result folder, and return its record for the manifest.
        :param clear: remove the script copies of earlier runs first
        '''
        if clear:
            shutil.rmtree(self.folder / 'script', ignore_errors=True)
        if self.script is None:
            return None
        shutil.copy2(self.script, self.get_fn('script', self.script.name))
        record = {'file': self.script.name}
        state = git_state(self.script)
        if state is not None:
            root = state.pop('root')
            record.update(state, path=self.script.relative_to(root).as_posix())
        return record

    @staticmethod
    def library_record():
        state = git_state(LIB_ROOT)
        record = {} if state is None else {k: state[k] for k in ('repo', 'commit', 'dirty')}
        lock = LIB_ROOT / 'uv.lock'
        if lock.exists():
            record['uv_lock_sha256'] = hashlib.sha256(lock.read_bytes()).hexdigest()
        return record

    def remove_old_items(self):
        '''Delete files in the item folders that were not added in this run.'''
        for kind, entries in self.items.items():
            keep = {fn for e in entries for fn in e['files']}
            folder = self.folder / kind
            if folder.exists():
                for fn in folder.iterdir():
                    if fn.is_file() and fn.name not in keep:
                        fn.unlink()

    def save(self):
        '''Write the manifest and the report. Call after all items are added.'''
        if self.sessions is None:
            raise ValueError('Set the sessions of the result with set_sessions before saving.')
        self.remove_old_items()
        now = datetime.now().isoformat(timespec='seconds')
        if self.loaded_manifest is None:
            self.manifest = {
                'identifier': self.identifier,
                'title': self.title,
                'description': self.description,
                'created': now,
                'script': self.copy_script(),
                'plot_script': None,
                'plotted': None,
                'steps': self.steps,
                'processing_steps': self.processing_steps,
                'library': self.library_record(),
                'python': sys.version.split()[0],
                'dataset': self.dataset_info,
                'cache': self.cache_records,
                'n_sessions': len(self.sessions),
                'n_animals': None if self.mice is None else len(self.mice),
                'mice_error': self.mice_error,
                'upload_sessions': self.upload_sessions,
                'tests': self.tests,
                'params': self.params,
                **self.items,
            }
        else:
            # opened by a plot script: the record of the analysis is kept, the figures are replaced
            self.manifest = {**self.loaded_manifest, 'plot_script': self.copy_script(clear=False), 'plotted': now,
                             'tests': self.tests, 'params': self.params, **self.items}
        with open(self.get_fn('manifest'), 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        self.get_fn('report').write_text(self.report_html(), encoding='utf-8')
        print(f'AnalysisResult {self.identifier}: {len(self.sessions)} sessions, {len(self.items["figures"])} figures, '
              f'{len(self.items["data"])} data tables, {len(self.items["stats"])} stats saved in {self.folder}')
        if self.loaded_manifest is None and self.plot_script is not None:
            subprocess.run([sys.executable, str(self.plot_script)], check=True, cwd=self.plot_script.parent,
                           env={**os.environ, CONFIRMED_ENV: '1'})

    @staticmethod
    def table_html(df, max_rows=None):
        note = ''
        if max_rows is not None and len(df) > max_rows:
            note = f'<p class="note">First {max_rows} of {len(df)} rows.</p>'
            df = df.head(max_rows)
        df = df.copy()
        for key in df.columns:
            if df[key].dtype == object or isinstance(df[key].dtype, pandas.StringDtype):
                df[key] = df[key].map(lambda v: f'{v[:REPORT_MAX_CHARS]}… ({len(v)} characters)'
                                      if isinstance(v, str) and len(v) > REPORT_MAX_CHARS else v)
        return f'<div class="scroll">{df.to_html(index=False, na_rep="", float_format="{:.4g}".format)}</div>{note}'

    @staticmethod
    def tests_html(tests, first_columns=()):
        '''The 'Statistics reporting' table of a list of tests (see add_test).'''
        tests = pandas.DataFrame(tests)
        tests['statistic'] = [f'{s} = {v:.4g}' for s, v in zip(tests['statistic'], tests['value'])]
        tests['df'] = ['' if d is None else str(d) for d in tests['df']]
        tests['p'] = [f'{p:.4g}' for p in tests['p']]
        tests = tests[[*first_columns, 'comparison', 'test', 'statistic', 'df', 'n', 'p']]
        return f'<div class="scroll">{tests.to_html(index=False, escape=True)}</div>'

    def report_html(self):
        m = self.manifest
        esc = html.escape
        out = [f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{esc(self.title)}</title>'
               f'<style>{REPORT_STYLE}</style></head><body>', f'<h1>{esc(self.title)}</h1>']
        if self.description:
            out.append(f'<p>{esc(self.description)}</p>')

        def item_head(entry):
            text = f'<h3>{esc(entry["name"])}</h3>'
            if entry['description']:
                text += f'<p>{esc(entry["description"])}</p>'
            return text + f'<p class="note">{esc(", ".join(entry["files"]))}</p>'

        if m['tests']:
            out.append('<h2>Statistics reporting</h2>')
            out.append(self.tests_html(m['tests']))
        if m['figures']:
            out.append('<h2>Figures</h2>')
        for entry in m['figures']:
            out.append(item_head(entry))
            png = [fn for fn in entry['files'] if fn.endswith('.png')]
            if png:
                data = base64.b64encode(self.get_fn('figures', png[0]).read_bytes()).decode()
                out.append(f'<img src="data:image/png;base64,{data}" alt="{esc(entry["name"])}">')
        if m['stats']:
            out.append('<h2>Statistics</h2>')
        for entry in m['stats']:
            out.append(item_head(entry))
            fn = self.get_fn('stats', entry['files'][0])
            if fn.suffix == '.txt':
                out.append(f'<pre>{esc(fn.read_text(encoding="utf-8"))}</pre>')
            else:
                out.append(self.table_html(pandas.read_feather(fn)))
        if m['data']:
            out.append('<h2>Data</h2>')
        for entry in m['data']:
            out.append(item_head(entry))
            if 'shape' in entry:
                out.append(f'<p class="note">Array, shape {tuple(entry["shape"])}, {esc(entry["dtype"])}</p>')
            else:
                out.append(self.table_html(self.get_data(entry['name']), REPORT_MAX_ROWS))

        if self.mice is not None:
            out.append(f'<h2>Animals ({len(self.mice)})</h2>')
            out.append(self.mice_html())
        elif m.get('mice_error'):  # results saved before the animals were added don't have the key
            out.append('<h2>Animals</h2>')
            out.append(f'<p class="warn">The animals could not be read from Baserow. {esc(m["mice_error"])}</p>')

        out.append(f'<h2>Sessions ({len(self.sessions)})</h2>')
        if not m['upload_sessions']:
            out.append('<p class="note">These sessions are not exported or uploaded when the result is published.</p>')
        if m['dataset'] is not None:
            out.append(f'<p class="note">DataSet version {m["dataset"]["version"]}, '
                       f'{esc(m["dataset"]["project_folder"])}</p>')
        # all-missing columns are left out of the report; sessions.feather has every column
        out.append(self.table_html(self.sessions.dropna(axis=1, how='all')))

        out.append('<h2>Provenance</h2>')
        rows = [('Created', m['created']), ('Python', m['python'])]
        if m['plotted']:
            rows.append(('Figures made', m['plotted']))
        records = [('Script', m['script']), ('Library', m['library'])]
        if m['plotted']:
            records.insert(1, ('Plot script', m['plot_script']))
        for label, record in records:
            if not record:
                rows.append((label, 'not recorded'))
                continue
            if 'file' in record:
                rows.append((label, esc(record.get('path', record['file']))))
            if 'commit' in record:
                state = ' <span class="warn">with uncommitted changes</span>' if record['dirty'] else ''
                rows.append((f'{label} commit', f'{esc(record["repo"])} {esc(record["commit"])}{state}'))
            else:
                rows.append((f'{label} commit', '<span class="warn">not in a git repository</span>'))
        for name, record in m.get('cache', {}).items():
            script, library = record.get('script') or {}, record.get('library') or {}
            dirty = ' <span class="warn">with uncommitted changes</span>' if script.get('dirty') else ''
            rows.append((f'Cached data: {esc(name)}',
                         f'computed {esc(record["computed"])} with script commit {esc(script.get("commit", "unknown"))}'
                         f'{dirty}, library commit {esc(library.get("commit", "unknown"))}'))
        if m['steps']:
            rows.append(('Steps run before', '<br>'.join(esc(str(s)) for s in m['steps'])))
        if m['processing_steps']:
            rows.append(('Processing before (not shared)', '<br>'.join(esc(str(s)) for s in m['processing_steps'])))
        for key, value in m['params'].items():
            rows.append((f'Parameter: {key}', esc(json.dumps(value, default=str))))
        out.append('<table>' + ''.join(f'<tr><th>{k}</th><td style="text-align:left">{v}</td></tr>'
                                       for k, v in rows) + '</table>')
        out.append('</body></html>')
        return '\n'.join(out)
