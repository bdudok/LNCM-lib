import html
import json
from datetime import datetime
from pathlib import Path

import pandas

from Proc2P.Analysis.AnalysisClasses.AnalysisResult import REPORT_STYLE, AnalysisResult

LUT_COLUMNS = ['Figure', 'Panel', 'Result', 'FinalFigure', 'FinalPanel']


class Publication:
    '''
    The record of a paper: which AnalysisResults it contains, and which figure and panel each of them is.
    The results stay in their project folders; the paper folder holds:
        publication.json: the results, with their figure names, panels, scripts and statistics
        <name>.html: the same as a page, with links to the report of each result
        figure_numbers.xlsx: optional table for the final figure and panel numbers (write_figure_lut)
    Usage (the script lists all results of the paper and is rerun when one is added or a panel changes):
        pub = Publication(paper_folder, title='...')
        pub.add_result(project_folder, 'CB1immuno_ratio_boxplot', figure='NF1-CRISPR-IPSP', panel='B')
        pub.save()
    Figures have working names and panels until the paper is final. For the final version, call write_figure_lut(),
    fill FinalFigure and FinalPanel in figure_numbers.xlsx, and run the script again: save() reads the table.
    '''
    __name__ = 'Publication'

    def __init__(self, folder, name=None, title=None):
        '''
        :param folder: the folder of the paper
        :param name: name of the paper, used as the file name of the report. Default: the name of the folder.
        :param title: shown in the report
        '''
        self.folder = Path(folder)
        self.name = name if name is not None else self.folder.name
        self.title = title if title is not None else self.name
        self.results = []
        self.folder.mkdir(parents=True, exist_ok=True)

    def get_fn(self, kind):
        '''
        :param kind: 'manifest', 'report' or 'lut'
        '''
        return self.folder / {'manifest': 'publication.json', 'report': f'{self.name}.html',
                              'lut': 'figure_numbers.xlsx'}[kind]

    def add_result(self, project_folder, identifier, figure, panel=''):
        '''
        :param project_folder, identifier: of the AnalysisResult. It has to be saved before the paper is saved.
        :param figure: working name of the figure (e.g. 'NF1-CRISPR-IPSP')
        :param panel: current panel of the result in the figure (e.g. 'B'); changes are made here until the
            paper is final
        '''
        if any(r['identifier'] == identifier for r in self.results):
            raise ValueError(f'Result {identifier} is already in the paper.')
        self.results.append({'identifier': identifier, 'project_folder': str(project_folder), 'figure': figure,
                             'panel': panel, 'final_figure': '', 'final_panel': ''})

    def lut_table(self):
        return pandas.DataFrame([[r['figure'], r['panel'], r['identifier'], r['final_figure'], r['final_panel']]
                                 for r in self.results], columns=LUT_COLUMNS)

    def read_figure_lut(self):
        '''Set the final figure and panel of the results from figure_numbers.xlsx, if the file exists.'''
        fn = self.get_fn('lut')
        if not fn.exists():
            return
        lut = pandas.read_excel(fn, dtype=str).fillna('')
        missing = [c for c in ('Result', 'FinalFigure', 'FinalPanel') if c not in lut.columns]
        if missing:
            raise ValueError(f'{fn} has no column {", ".join(missing)}')
        final = {row['Result']: (row['FinalFigure'].strip(), row['FinalPanel'].strip()) for _, row in lut.iterrows()}
        for r in self.results:
            if r['identifier'] in final:
                r['final_figure'], r['final_panel'] = final[r['identifier']]

    def write_figure_lut(self):
        '''
        Write figure_numbers.xlsx: one row per result with its working figure name and panel. Fill FinalFigure and
        FinalPanel, then save the paper again. Values that are already filled in are kept.
        '''
        self.read_figure_lut()
        fn = self.get_fn('lut')
        try:
            self.lut_table().sort_values(['Figure', 'Panel']).to_excel(fn, index=False)
        except PermissionError:
            raise PermissionError(f'Close {fn} in Excel and run again.') from None
        print(f'Publication {self.name}: fill FinalFigure and FinalPanel in {fn}')

    def save(self):
        '''Read the manifests of the results, and write the manifest and the report of the paper.'''
        self.read_figure_lut()
        entries = []
        for r in self.results:
            fn = Path(r['project_folder']) / 'Results' / r['identifier'] / 'manifest.json'
            if not fn.exists():
                raise FileNotFoundError(f'Result {r["identifier"]} is not saved: {fn} not found. Run its script first.')
            with open(fn, encoding='utf-8') as f:
                m = json.load(f)
            entries.append({**r, 'title': m['title'], 'result_created': m['created'], 'script': m['script'],
                            'steps': m['steps'], 'n_sessions': m['n_sessions'],
                            'upload_sessions': m.get('upload_sessions', True), 'tests': m.get('tests', [])})
        self.manifest = {'name': self.name, 'title': self.title,
                         'updated': datetime.now().isoformat(timespec='seconds'), 'results': entries}
        with open(self.get_fn('manifest'), 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        self.get_fn('report').write_text(self.report_html(), encoding='utf-8')
        print(f'Publication {self.name}: {len(entries)} results saved in {self.folder}')

    @staticmethod
    def figure_label(entry):
        '''Final figure and panel if set, the working name and panel otherwise.'''
        if entry['final_figure']:
            return f'{entry["final_figure"]}{entry["final_panel"]}'
        return f'{entry["figure"]} {entry["panel"]}'.strip()

    def report_html(self):
        m = self.manifest
        esc = html.escape
        entries = sorted(m['results'], key=lambda e: (e['final_figure'] or e['figure'], e['final_panel'] or e['panel']))
        out = [f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{esc(self.title)}</title>'
               f'<style>{REPORT_STYLE} td {{text-align: left;}}</style></head><body>', f'<h1>{esc(self.title)}</h1>',
               f'<p class="note">Updated {m["updated"]}</p>', '<h2>Figures</h2>']
        head = ('Figure', 'Panel', 'Final figure', 'Final panel', 'Result', 'Script', 'Result saved', 'Sessions')
        rows = []
        for e in entries:
            report = (Path(e['project_folder']) / 'Results' / e['identifier'] / f'{e["identifier"]}.html').resolve().as_uri()
            script = e['script']['file'] if e['script'] else 'not recorded'
            sessions = str(e['n_sessions']) if e['upload_sessions'] else f'{e["n_sessions"]}, not uploaded'
            cells = [esc(e['figure']), esc(e['panel']), esc(e['final_figure']), esc(e['final_panel']),
                     f'<a href="{esc(report)}">{esc(e["identifier"])}</a>', esc(script), esc(e['result_created']),
                     sessions]
            rows.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
        out.append('<table><tr>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr>' + ''.join(rows) + '</table>')

        tests = [{'figure': self.figure_label(e), **t} for e in entries for t in e['tests']]
        if tests:
            out.append('<h2>Statistics reporting</h2>')
            out.append(AnalysisResult.tests_html(tests, first_columns=('figure',)))
        out.append('</body></html>')
        return '\n'.join(out)
