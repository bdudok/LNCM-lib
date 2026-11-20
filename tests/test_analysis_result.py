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


def test_git_state_dirty_only_for_the_path(tmp_path):
    import subprocess

    from Proc2P.Analysis.AnalysisClasses.AnalysisResult import git_state

    def git(*args):
        subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=t', '-c', 'user.email=t@t', *args], check=True,
                       capture_output=True)

    script, other = tmp_path / 'script.py', tmp_path / 'other.py'
    script.write_text('a = 1\n')
    other.write_text('b = 1\n')
    git('init')
    git('add', '.')
    git('commit', '-m', 'init')
    assert not git_state(script)['dirty']
    assert not git_state(tmp_path)['dirty']

    # changes of other files don't count for a file; untracked files don't count for a folder
    (tmp_path / 'new.py').write_text('c = 1\n')
    assert not git_state(tmp_path)['dirty']
    assert git_state(tmp_path / 'new.py')['dirty']
    other.write_text('b = 2\n')
    assert not git_state(script)['dirty']
    assert git_state(tmp_path)['dirty']

    script.write_text('a = 2\n')
    assert git_state(script)['dirty']


def test_uncommitted_script_waits_for_enter(tmp_path, monkeypatch):
    import subprocess

    from Proc2P.Analysis.AnalysisClasses import AnalysisResult as module

    script = tmp_path / 'script.py'
    script.write_text('a = 1\n')
    for args in (['init'], ['commit', '--allow-empty', '-m', 'init']):  # the script is untracked
        subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=t', '-c', 'user.email=t@t', *args], check=True,
                       capture_output=True)
    prompts = []
    monkeypatch.setattr('builtins.input', prompts.append)
    monkeypatch.setattr(module, 'LIB_ROOT', tmp_path / 'not_a_repo')

    AnalysisResult(tmp_path, 'Fig1', script=script)
    assert prompts == []  # confirmed already: the plot script run by save()

    monkeypatch.delenv(module.CONFIRMED_ENV)
    AnalysisResult(tmp_path, 'Fig1', script=script)
    assert len(prompts) == 1 and 'script.py' in prompts[0]


def test_mouse_table_has_the_age_at_the_first_session_used(tmp_path, monkeypatch):
    import requests

    from Proc2P.Analysis.AnalysisClasses import AnalysisResult as module

    ds = make_ds(tmp_path)
    ds.set_field('s1', 'Sex', 'F')
    ds.save_df()
    dates = {'s1': '2025-04-11', 's2': '2025-04-01', 's3': '2025-05-01'}

    def read_mouse_db(animals, prefixes):
        assert list(animals) == ['m1', 'm2']
        return ({'m1': {'Sex': 'M', 'Strain': 'C57BL/6J', 'DOB': '2025-01-01'}, 'm2': None},
                {p: dates[p] for p in prefixes})

    monkeypatch.setattr(module, 'read_mouse_db', read_mouse_db)
    res = AnalysisResult(tmp_path, 'Fig1', script=__file__)
    res.set_sessions(['s1', 's2', 's3'], dataset=ds)
    res.save()
    folder = tmp_path / 'Results' / 'Fig1'
    mice = pandas.read_feather(folder / 'mice.feather').set_index('Animal')
    assert mice.loc['m1', 'Sex'] == 'F'  # the DataSet value is used, not the one in the db
    assert mice.loc['m1', 'AgeFirstSession'] == 90 and mice.loc['m1', 'nSessions'] == 2
    assert pandas.isna(mice.loc['m2', 'AgeFirstSession']) and mice['AgeFirstSurgery'].isna().all()
    report = (folder / 'Fig1.html').read_text(encoding='utf-8')
    for text in ('Animals (2)', 'Sex: 1 F', 'Strain: 1 C57BL/6J', '90 days (n = 1)',
                 'Age at the first surgery: not available', '2025-04-01'):
        assert text in report, text
    animals = report[report.index('Animals (2)'):report.index('Sessions (3)')]
    assert 'NA' not in animals and '00:00' not in animals

    # only the sessions used count; without the db the result is saved with a warning
    res.set_sessions(['s1', 's3'], dataset=ds)
    assert res.mice.set_index('Animal').loc['m1', 'AgeFirstSession'] == 100

    def no_connection(animals, prefixes):
        raise requests.ConnectionError('no route')

    monkeypatch.setattr(module, 'read_mouse_db', no_connection)
    res.set_sessions(['s1', 's3'], dataset=ds)
    res.save()
    assert not (folder / 'mice.feather').exists()
    assert 'could not be read from Baserow' in (folder / 'Fig1.html').read_text(encoding='utf-8')


def test_mouse_table_reads_baserow_only_for_what_the_dataset_lacks(tmp_path, monkeypatch):
    from Proc2P.Analysis.AnalysisClasses import AnalysisResult as module

    ds = make_ds(tmp_path)
    for pf, date in (('s1', '2025-04-11'), ('s2', '2025-04-01')):
        ds.set_by_dict(pf, {'Sex': 'F', 'Strain': 'PV-Cre'})
        ds.set_by_dict(pf, {'DOB': '2025-01-01', 'FirstSurgery': '2025-03-02', 'Date': date}, type=pandas.Timestamp)
    ds.save_df()
    calls = []

    def read_mouse_db(animals, prefixes):
        calls.append((list(animals), list(prefixes)))
        return {'m2': {'Sex': 'M', 'Strain': 'C57BL/6J', 'DOB': '2025-02-01'}}, {'s3': '2025-05-02'}

    monkeypatch.setattr(module, 'read_mouse_db', read_mouse_db)
    res = AnalysisResult(tmp_path, 'Fig1', script=__file__)
    res.set_sessions(['s1', 's2'], dataset=ds)  # everything is in the DataSet
    assert calls == []
    m1 = res.mice.set_index('Animal').loc['m1']
    assert (m1['Sex'], m1['Strain'], m1['AgeFirstSession'], m1['AgeFirstSurgery']) == ('F', 'PV-Cre', 90, 60)

    res.set_sessions(['s1', 's2', 's3'], dataset=ds)  # m2 and s3 are not
    assert calls == [(['m2'], ['s3'])]
    mice = res.mice.set_index('Animal')
    assert mice.loc['m1', 'Strain'] == 'PV-Cre' and mice.loc['m2', 'Strain'] == 'C57BL/6J'
    assert mice.loc['m2', 'AgeFirstSession'] == 90 and pandas.isna(mice.loc['m2', 'AgeFirstSurgery'])


def test_first_surgery_falls_back_to_baserow_and_ages_are_checked(tmp_path, monkeypatch, capsys):
    from Proc2P.Analysis.AnalysisClasses import AnalysisResult as module

    ds = make_ds(tmp_path)
    dates = {'s1': '2025-04-11', 's2': '2025-04-01', 's3': '2025-01-01'}

    def read_mouse_db(animals, prefixes):
        return ({'m1': {'Sex': 'F', 'Strain': 'C57BL/6J', 'DOB': '2025-01-01', 'FirstSurgery': '2025-03-02'},
                 'm2': {'Sex': 'M', 'Strain': 'C57BL/6J', 'DOB': '2025-01-01', 'FirstSurgery': None}},
                {p: dates[p] for p in prefixes})

    monkeypatch.setattr(module, 'read_mouse_db', read_mouse_db)
    res = AnalysisResult(tmp_path, 'Fig1', script=__file__)
    res.set_sessions(['s1', 's2', 's3'], dataset=ds)
    mice = res.mice.set_index('Animal')
    assert mice.loc['m1', 'AgeFirstSurgery'] == 60 and mice.loc['m2', 'AgeFirstSession'] == 0
    printed = capsys.readouterr().out
    res.save()
    report = (tmp_path / 'Results' / 'Fig1' / 'Fig1.html').read_text(encoding='utf-8')
    for text in ('Age at the first surgery is missing for: m2.', 'Age at the first session is 0 or negative for: m2.'):
        assert text in printed and text in report, text
    assert 'for: m1' not in report

    ds.set_field('s3', 'Date', '2028-01-01', type=pandas.Timestamp)  # the DataSet value is used for the session date
    res.set_sessions(['s3'], dataset=ds)
    assert 'Age at the first session is over 1000 days for: m2.' in module.age_warnings(res.mice)


def test_report_tables_cap_long_text():
    '''A long text cell (e.g. a cell list of the DataSet) is cut in the report; short ones and numbers are kept.'''
    cells = str(list(range(1000)))
    df = pandas.DataFrame({'Prefix': ['s1'], 'Cells.R.Incl': pandas.array([cells], dtype='string'), 'n': [3]})
    out = AnalysisResult.table_html(df)
    assert cells not in out
    assert f'{cells[:80]}… ({len(cells)} characters)' in out
    assert '>s1<' in out and '>3<' in out
    assert df['Cells.R.Incl'].iloc[0] == cells  # the table itself is not changed
