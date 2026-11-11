'''
AnalysisResult: items, the session snapshot and the code version are saved; a rerun removes items it did not add.
'''
import json

import matplotlib
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
    res = AnalysisResult(tmp_path, 'Fig1', script=__file__, steps=['1_construct.py', '3_pull.py'])
    res.set_params(window=(-3, -1), min_stops=2)
    res.set_sessions(['s2', 's1', 's1'], dataset=ds)
    add_items(res)
    res.save()

    folder = tmp_path / 'Results' / 'Fig1'
    for fn in ('figures/psth.png', 'figures/psth.svg', 'data/values.feather', 'stats/model.feather',
               'stats/summary.txt', 'script/test_analysis_result.py', 'report.html'):
        assert (folder / fn).exists(), fn
    sessions = pandas.read_feather(folder / 'sessions.feather')
    assert list(sessions['Prefix']) == ['s1', 's2']
    assert list(sessions['Animal']) == ['m1', 'm1']

    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['steps'] == ['1_construct.py', '3_pull.py']
    assert manifest['dataset']['version'] == ds.ver
    assert manifest['params'] == {'window': [-3, -1], 'min_stops': 2}
    assert manifest['script']['path'] == 'tests/test_analysis_result.py'
    assert len(manifest['script']['commit']) == 40
    assert manifest['library']['commit'] == manifest['script']['commit']
    assert [e['name'] for e in manifest['stats']] == ['model', 'summary']

    report = (folder / 'report.html').read_text(encoding='utf-8')
    assert 'data:image/png;base64' in report
    assert 'p = 0.04 &lt;n=2&gt;' in report


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
