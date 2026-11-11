import base64
import hashlib
import html
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas

LIB_ROOT = Path(__file__).resolve().parents[3]
REPORT_MAX_ROWS = 50  # rows of a data table shown in the report; the file has all rows

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
    '''Commit and uncommitted-changes flag of the repo that contains path, or None if it can't be read.'''
    path = Path(path)
    folder = path if path.is_dir() else path.parent

    def git(*args):
        return subprocess.run(['git', '-C', str(folder), *args], capture_output=True, text=True, check=True).stdout

    try:
        root = Path(git('rev-parse', '--show-toplevel').strip())
        return {'repo': root.name, 'commit': git('rev-parse', 'HEAD').strip(),
                'dirty': bool(git('status', '--porcelain').strip()), 'root': root}
    except (OSError, subprocess.CalledProcessError):
        return None


class AnalysisResult:
    '''
    The record of one result (usually a figure): the figures, the tables behind them, the statistics, the list of
    sessions used, and the code version. Saved in <project_folder>/Results/<identifier>/:
        manifest.json: script, commits of the script's repo and of LNCM-lib, DataSet version, parameters, contents
        sessions.feather: one row per session used (the DataSet rows, if a DataSet was given)
        figures/, data/, stats/: the items added
        script/: a copy of the script as it was run
        report.html: all of the above on one page
    Usage:
        res = AnalysisResult(project_folder, 'Fig3_RunStop', steps=['3c_PSTHRunStop.py'])
        res.set_sessions(prefixes, dataset=ds)
        res.add_data('psth', df); res.add_stat('mixed_model', table); res.add_figure('psth', fig)
        res.save()
    Items are written when added; save() writes the manifest and the report, and removes items of earlier runs.
    '''
    __name__ = 'AnalysisResult'

    def __init__(self, project_folder, identifier, script=None, steps=None, title=None, description=None):
        '''
        :param project_folder: the project folder of the analysis (as for DataSet)
        :param identifier: unique name of the result within the project, used as the folder name
        :param script: path of the script that makes the result. Default: the script being run.
        :param steps: file names of the scripts that have to be run before this one, in run order
        :param title, description: shown in the report
        '''
        self.project_folder = Path(project_folder)
        self.identifier = identifier
        self.folder = self.project_folder / 'Results' / identifier
        if script is None:
            script = getattr(sys.modules['__main__'], '__file__', None)
        self.script = Path(script).resolve() if script is not None else None
        self.steps = list(steps) if steps is not None else []
        self.title = title if title is not None else identifier
        self.description = description
        self.params = {}
        self.items = {'figures': [], 'data': [], 'stats': []}
        self.sessions = None
        self.dataset_info = None
        self.folder.mkdir(parents=True, exist_ok=True)

    def get_fn(self, kind, name=None, ext=''):
        '''
        :param kind: 'manifest', 'sessions', 'report', or the folder of an item: 'figures', 'data', 'stats', 'script'
        '''
        fixed = {'manifest': 'manifest.json', 'sessions': 'sessions.feather', 'report': 'report.html'}
        if kind in fixed:
            return self.folder / fixed[kind]
        (self.folder / kind).mkdir(exist_ok=True)
        return self.folder / kind / f'{name}{ext}'

    def set_params(self, **kwargs):
        '''Analysis parameters to record in the manifest (values that json can't store are stored as text).'''
        self.params.update(kwargs)

    def set_sessions(self, prefixes, dataset=None):
        '''
        The sessions that the result is computed from.
        :param prefixes: session prefixes
        :param dataset: optional DataSet. Its rows of these sessions are stored, with the dataset version.
        '''
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
            self.dataset_info = {'project_folder': str(dataset.project_folder), 'version': int(dataset.ver)}
        self.sessions.to_feather(self.get_fn('sessions'))

    def register(self, kind, name, files, description, **info):
        entries = self.items[kind]
        entries[:] = [e for e in entries if e['name'] != name]  # adding a name again replaces it
        entries.append({'name': name, 'files': [fn.name for fn in files], 'description': description, **info})

    def add_figure(self, name, fig, description=None, formats=('png', 'svg'), dpi=300):
        '''Save a matplotlib figure. The png is shown in the report; svg keeps text editable.'''
        files = []
        with matplotlib.rc_context({'svg.fonttype': 'none'}):
            for ext in formats:
                fn = self.get_fn('figures', name, f'.{ext}')
                fig.savefig(fn, dpi=dpi)
                files.append(fn)
        self.register('figures', name, files, description)

    def add_data(self, name, df, description=None):
        '''Save a table of the values behind a figure or a statistic (feather).'''
        fn = self.get_fn('data', name, '.feather')
        df.reset_index(drop=True).to_feather(fn)
        self.register('data', name, [fn], description, rows=len(df), columns=[str(c) for c in df.columns])

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

    def copy_script(self):
        '''Copy the script into the result folder, and return its record for the manifest.'''
        if self.script is None:
            return None
        shutil.rmtree(self.folder / 'script', ignore_errors=True)
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
        self.manifest = {
            'identifier': self.identifier,
            'title': self.title,
            'description': self.description,
            'created': datetime.now().isoformat(timespec='seconds'),
            'script': self.copy_script(),
            'steps': self.steps,
            'library': self.library_record(),
            'python': sys.version.split()[0],
            'dataset': self.dataset_info,
            'n_sessions': len(self.sessions),
            'params': self.params,
            **self.items,
        }
        with open(self.get_fn('manifest'), 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        self.get_fn('report').write_text(self.report_html(), encoding='utf-8')
        print(f'AnalysisResult {self.identifier}: {len(self.sessions)} sessions, {len(self.items["figures"])} figures, '
              f'{len(self.items["data"])} data tables, {len(self.items["stats"])} stats saved in {self.folder}')

    @staticmethod
    def table_html(df, max_rows=None):
        note = ''
        if max_rows is not None and len(df) > max_rows:
            note = f'<p class="note">First {max_rows} of {len(df)} rows.</p>'
            df = df.head(max_rows)
        return f'<div class="scroll">{df.to_html(index=False, na_rep="", float_format="{:.4g}".format)}</div>{note}'

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
            out.append(self.table_html(pandas.read_feather(self.get_fn('data', entry['files'][0])), REPORT_MAX_ROWS))

        out.append(f'<h2>Sessions ({len(self.sessions)})</h2>')
        if m['dataset'] is not None:
            out.append(f'<p class="note">DataSet version {m["dataset"]["version"]}, '
                       f'{esc(m["dataset"]["project_folder"])}</p>')
        # all-missing columns are left out of the report; sessions.feather has every column
        out.append(self.table_html(self.sessions.dropna(axis=1, how='all')))

        out.append('<h2>Provenance</h2>')
        rows = [('Created', m['created']), ('Python', m['python'])]
        for label, record in (('Script', m['script']), ('Library', m['library'])):
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
        if m['steps']:
            rows.append(('Steps run before', '<br>'.join(esc(str(s)) for s in m['steps'])))
        for key, value in m['params'].items():
            rows.append((f'Parameter: {key}', esc(json.dumps(value, default=str))))
        out.append('<table>' + ''.join(f'<tr><th>{k}</th><td style="text-align:left">{v}</td></tr>'
                                       for k, v in rows) + '</table>')
        out.append('</body></html>')
        return '\n'.join(out)
