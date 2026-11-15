'''
Publication: results are listed with figure names and panels; final numbers come from the table the user fills.
'''
import json

import pandas
import pytest

from Proc2P.Analysis.AnalysisClasses.AnalysisResult import AnalysisResult
from Proc2P.Analysis.AnalysisClasses.Publication import Publication


def make_result(project, identifier, upload=True):
    res = AnalysisResult(project, identifier, script=__file__)
    res.set_sessions(['s1', 's2'], upload=upload)
    res.add_test('a vs b', 'paired t-test', 't', 2.5, df=1, n='2 sessions', p=0.04)
    res.save()


def make_pub(tmp_path, panel='B'):
    pub = Publication(tmp_path / 'paper-x')
    pub.add_result(tmp_path / 'proj', 'ratio', figure='NF1-CRISPR', panel=panel)
    pub.add_result(tmp_path / 'proj', 'psth', figure='NF1-CRISPR', panel='A')
    return pub


def test_save_lists_results_with_working_names(tmp_path):
    make_result(tmp_path / 'proj', 'ratio', upload=False)
    make_result(tmp_path / 'proj', 'psth')
    make_pub(tmp_path).save()

    folder = tmp_path / 'paper-x'
    manifest = json.loads((folder / 'publication.json').read_text(encoding='utf-8'))
    ratio = manifest['results'][0]
    assert (ratio['figure'], ratio['panel'], ratio['final_figure']) == ('NF1-CRISPR', 'B', '')
    assert ratio['upload_sessions'] is False and manifest['results'][1]['upload_sessions'] is True
    assert ratio['script']['file'] == 'test_publication.py'
    report = (folder / 'paper-x.html').read_text(encoding='utf-8')
    assert 'ratio/ratio.html' in report and '2, not uploaded' in report
    assert 'NF1-CRISPR B' in report and 't = 2.5' in report


def test_final_numbers_come_from_the_filled_table(tmp_path):
    make_result(tmp_path / 'proj', 'ratio')
    make_result(tmp_path / 'proj', 'psth')
    pub = make_pub(tmp_path)
    pub.write_figure_lut()
    fn = tmp_path / 'paper-x' / 'figure_numbers.xlsx'
    lut = pandas.read_excel(fn, dtype=str)
    assert list(lut['Result']) == ['psth', 'ratio']
    lut.loc[lut['Result'].eq('ratio'), ['FinalFigure', 'FinalPanel']] = ['3', 'C']
    lut.to_excel(fn, index=False)

    # the panel changes in the script after the table was filled; the table is written again
    pub = make_pub(tmp_path, panel='D')
    pub.write_figure_lut()
    pub.save()
    lut = pandas.read_excel(fn, dtype=str).fillna('')
    assert lut.loc[lut['Result'].eq('ratio')].values.tolist() == [['NF1-CRISPR', 'D', 'ratio', '3', 'C']]
    manifest = json.loads((tmp_path / 'paper-x' / 'publication.json').read_text(encoding='utf-8'))
    assert [(r['final_figure'], r['final_panel']) for r in manifest['results']] == [('3', 'C'), ('', '')]
    assert '<td>3C</td>' in (tmp_path / 'paper-x' / 'paper-x.html').read_text(encoding='utf-8')


def test_result_has_to_be_saved_and_unique(tmp_path):
    pub = make_pub(tmp_path)
    with pytest.raises(ValueError, match='already in the paper'):
        pub.add_result(tmp_path / 'proj', 'ratio', figure='F2')
    with pytest.raises(FileNotFoundError, match='Run its script first'):
        pub.save()
