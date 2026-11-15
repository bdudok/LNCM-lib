'''
AnalysisResult: items, the session snapshot and the code version are saved; a rerun removes items it did not add.
'''
import json
import os
from pathlib import Path

import matplotlib
import numpy
import pandas
import pytest

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402

from Proc2P.Analysis.AnalysisClasses.AnalysisResult import AnalysisResult  # noqa: E402
from Proc2P.Analysis.AnalysisClasses.DataSet import DataSet  # noqa: E402


def make_ds(path):
    ds = DataSet(path)
    for pf, animal in (('s1', 'm1'), ('s2', 'm1'), ('s3', 'm2')):
        ds.add_row({'Prefix': pf, 'Animal': animal, 'Incl': True})
    ds.save_df()
    return ds


def add_items(res, figure_name='psth'):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    res.add_figure(figure_name, fig, description='a line')
    plt.close(fig)
    res.add_data('values', pandas.DataFrame({'Prefix': ['s1', 's2'], 'y': [0.5, 1.5]}))
    res.add_stat('model', pandas.DataFrame({'Term': ['a'], 'p': [0.04]}))
    res.add_stat('summary', 't-test: p = 0.04 <n=2>')


def test_save_writes_items_snapshot_and_manifest(tmp_path):
    ds = make_ds(tmp_path)
    res = AnalysisResult(tmp_path, 'Fig1', script=__file__, steps=['1_construct.py', '3_pull.py'],
                         processing_steps=['0_register.py'])
    res.set_params(window=(-3, -1), min_stops=2)
    res.set_sessions(['s2', 's1', 's1'], dataset=ds)
    add_items(res)
    res.save()

    folder = tmp_path / 'Results' / 'Fig1'
    for fn in ('figures/psth.png', 'figures/psth.svg', 'data/values.feather', 'stats/model.feather',
               'stats/summary.txt', 'script/test_analysis_result.py', 'Fig1.html'):
        assert (folder / fn).exists(), fn
    sessions = pandas.read_feather(folder / 'sessions.feather')
    assert list(sessions['Prefix']) == ['s1', 's2']
    assert list(sessions['Animal']) == ['m1', 'm1']

    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['steps'] == ['1_construct.py', '3_pull.py']
    assert manifest['processing_steps'] == ['0_register.py']
    assert manifest['dataset']['version'] == ds.ver
    assert manifest['params'] == {'window': [-3, -1], 'min_stops': 2}
    assert manifest['script']['path'] == 'tests/test_analysis_result.py'
    assert len(manifest['script']['commit']) == 40
    assert manifest['library']['commit'] == manifest['script']['commit']
    assert [e['name'] for e in manifest['stats']] == ['model', 'summary']

    report = (folder / 'Fig1.html').read_text(encoding='utf-8')
    assert 'data:image/png;base64' in report
    assert '<text' in (folder / 'figures' / 'psth.svg').read_text(encoding='utf-8')  # text is not outlined
    assert 'p = 0.04 &lt;n=2&gt;' in report
    assert 'Processing before (not shared)' in report and '0_register.py' in report


def test_rerun_removes_items_not_added_again(tmp_path):
    for figure_name in ('old_name', 'new_name'):
        res = AnalysisResult(tmp_path, 'Fig1', script=__file__)
        res.set_sessions(['s1'])
        add_items(res, figure_name)
        res.save()
    figures = sorted(fn.name for fn in (tmp_path / 'Results' / 'Fig1' / 'figures').iterdir())
    assert figures == ['new_name.png', 'new_name.svg']


def test_sessions_are_required_and_checked(tmp_path):
    ds = make_ds(tmp_path)
    res = AnalysisResult(tmp_path, 'Fig1', script=__file__)
    with pytest.raises(ValueError, match='not in the DataSet: s9'):
        res.set_sessions(['s1', 's9'], dataset=ds)
    with pytest.raises(ValueError, match='set_sessions'):
        res.save()


def test_reporting_table_and_upload_flag(tmp_path):
    res = AnalysisResult(tmp_path, 'Fig1', script=__file__)
    res.set_sessions(['246_1', '246_2'], upload=False)
    res.add_test('KO vs control', 'LMM likelihood ratio test', 'chi2', 7.0926, df=1, n='14 stacks, 5 animals',
                 p=0.00774)
    res.add_test('a vs b', 'Mann-Whitney U, two-sided', 'U', 46, df=None, n='8, 6', p=0.0027)
    res.save()

    folder = tmp_path / 'Results' / 'Fig1'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['upload_sessions'] is False
    assert manifest['tests'][0] == {'comparison': 'KO vs control', 'test': 'LMM likelihood ratio test',
                                    'statistic': 'chi2', 'value': 7.0926, 'df': 1, 'n': '14 stacks, 5 animals',
                                    'p': 0.00774}
    report = (folder / 'Fig1.html').read_text(encoding='utf-8')
    assert report.index('Statistics reporting') < report.index('<h2>Sessions')
    for text in ('chi2 = 7.093', '0.00774', '14 stacks, 5 animals', 'not exported or uploaded'):
        assert text in report, text


def test_sessions_are_uploaded_by_default(tmp_path):
    res = AnalysisResult(tmp_path, 'Fig1', script=__file__)
    res.set_sessions(['s1'])
    res.save()
    manifest = json.loads((tmp_path / 'Results' / 'Fig1' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['upload_sessions'] is True and manifest['tests'] == []
    assert 'Statistics reporting' not in (tmp_path / 'Results' / 'Fig1' / 'Fig1.html').read_text(encoding='utf-8')


PLOT_SCRIPT = '''
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from Proc2P.Analysis.AnalysisClasses.AnalysisResult import AnalysisResult

res = AnalysisResult.load(sys.argv[1] if len(sys.argv) > 1 else r'{project}', 'Fig1')
df = res.get_data('values')
fig, ax = plt.subplots()
ax.plot(df['y'] * res.params['gain'], res.get_data('trace').mean(axis=0)[:len(df)])
res.add_figure('{figure}', fig)
res.save()
'''


def test_plot_script_makes_the_figures_from_the_saved_result(tmp_path, monkeypatch):
    # the plot script runs in a new process and has to find the library
    monkeypatch.setenv('PYTHONPATH', str(Path(__file__).resolve().parents[1]) + os.pathsep
                       + os.environ.get('PYTHONPATH', ''))
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    analysis = scripts / '3_analysis.py'
    analysis.write_text('# analysis', encoding='utf-8')
    plot = scripts / '3_analysis_plot.py'
    plot.write_text(PLOT_SCRIPT.format(project=tmp_path, figure='first'), encoding='utf-8')

    res = AnalysisResult(tmp_path, 'Fig1', script=analysis, plot_script=plot.name, processing_steps=['0_register.py'])
    res.set_params(gain=2)
    res.set_sessions(['s1'])
    res.add_data('values', pandas.DataFrame({'y': [0.5, 1.5]}))
    res.add_data('trace', numpy.arange(6.0).reshape(2, 3))
    res.add_test('a vs b', 'paired t-test', 't', 2.5, df=1, n='2 sessions', p=0.04)
    res.save()

    folder = tmp_path / 'Results' / 'Fig1'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['script']['file'] == '3_analysis.py' and manifest['plot_script']['file'] == '3_analysis_plot.py'
    assert [e['name'] for e in manifest['figures']] == ['first']
    assert [e['name'] for e in manifest['data']] == ['values', 'trace'] and manifest['data'][1]['shape'] == [2, 3]
    assert len(manifest['tests']) == 1 and manifest['created'] <= manifest['plotted']
    assert sorted(fn.name for fn in (folder / 'script').iterdir()) == ['3_analysis.py', '3_analysis_plot.py']
    report = (folder / 'Fig1.html').read_text(encoding='utf-8')
    assert 'data:image/png;base64' in report and 'Plot script' in report and 'Array, shape (2, 3)' in report

    # the plot script is changed and run alone: the figure is replaced, the analysis record and data stay
    plot.write_text(PLOT_SCRIPT.format(project=tmp_path, figure='second'), encoding='utf-8')
    res = AnalysisResult.load(tmp_path, 'Fig1', script=plot)
    assert numpy.array_equal(res.get_data('trace'), numpy.arange(6.0).reshape(2, 3))
    fig, ax = plt.subplots()
    res.add_figure('second', fig)
    plt.close(fig)
    res.save()
    again = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert sorted(fn.name for fn in (folder / 'figures').iterdir()) == ['second.png', 'second.svg']
    assert again['created'] == manifest['created'] and again['script'] == manifest['script']
    assert again['processing_steps'] == ['0_register.py']
    assert (folder / 'data' / 'values.feather').exists() and (folder / 'data' / 'trace.npy').exists()


def test_load_needs_a_saved_result(tmp_path):
    with pytest.raises(FileNotFoundError, match='Run its analysis script first'):
        AnalysisResult.load(tmp_path, 'Fig1', script=__file__)
