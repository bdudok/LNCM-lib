'''
AnalysisResult.cached_data: data is computed again only when the code, the settings or the DataSet it depends on
changed. DataSet.fingerprint changes with the contents, also without a new version.
'''
import importlib.util
import json
import sys

import numpy

from Proc2P.Analysis.AnalysisClasses.AnalysisResult import AnalysisResult
from Proc2P.Analysis.AnalysisClasses.DataSet import DataSet

SCRIPT = '''
import types

import pandas

calls = types.SimpleNamespace(n=0)  # not data: not tracked
offset = {offset}
window = (1, 2)


def helper(x):
    return x + {helper_add}


def compute():
    calls.n += 1
    return pandas.DataFrame({{'Prefix': list(ds.df['Prefix']), 'y': [helper(offset)] * len(ds.df)}})


def compute_two():
    calls.n += 1
    return {{'table': pandas.DataFrame({{'y': [offset]}}), 'array': numpy.arange(3)}}
'''


def make_ds(path):
    ds = DataSet(path)
    for pf in ('s1', 's2'):
        ds.add_row({'Prefix': pf, 'Animal': 'm1', 'Incl': True})
    ds.save_df()
    return ds


def load_script(tmp_path, ds, name='script_a', offset=1, helper_add=10):
    '''import a script with the given settings as a module, with ds as its DataSet'''
    fn = tmp_path / f'{name}.py'
    fn.write_text('import numpy\n' + SCRIPT.format(offset=offset, helper_add=helper_add))
    spec = importlib.util.spec_from_file_location(name, fn)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.ds = ds
    return module


def run(tmp_path, module, **kwargs):
    res = AnalysisResult(tmp_path, 'Cached', script=module.__file__)
    data = res.cached_data('values', module.compute, descriptions='values per session', **kwargs)
    res.set_sessions(list(data['Prefix']))
    res.save()
    return res, data


def test_dataset_fingerprint_follows_contents(tmp_path):
    ds = make_ds(tmp_path)
    fp = ds.fingerprint()
    assert DataSet(tmp_path).fingerprint() == fp
    ds.set_field('s1', 'Cohort', 'a')
    ds.save_df()  # same version
    assert DataSet(tmp_path).ver == ds.ver
    assert DataSet(tmp_path).fingerprint() != fp


def test_computed_once_then_loaded(tmp_path):
    ds = make_ds(tmp_path)
    module = load_script(tmp_path, ds)
    res, data = run(tmp_path, module)
    assert list(data['y']) == [11, 11]
    res, data = run(tmp_path, module)
    assert module.calls.n == 1
    assert list(data['y']) == [11, 11]
    manifest = json.loads((tmp_path / 'Results' / 'Cached' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['cache']['values']['items'] == {'values': 'values.feather'}
    assert (tmp_path / 'Results' / 'Cached' / 'data' / 'values.feather').exists()


def test_changes_that_recompute(tmp_path):
    ds = make_ds(tmp_path)
    module = load_script(tmp_path, ds)
    run(tmp_path, module)

    # a setting the function uses
    module = load_script(tmp_path, ds, name='script_b', offset=2)
    _, data = run(tmp_path, module)
    assert module.calls.n == 1 and list(data['y']) == [12, 12]

    # the source of a helper it calls
    module = load_script(tmp_path, ds, name='script_c', offset=2, helper_add=20)
    _, data = run(tmp_path, module)
    assert module.calls.n == 1 and list(data['y']) == [22, 22]

    # the DataSet, saved under the same version
    ds.set_field('s1', 'Cohort', 'a')
    ds.save_df()
    run(tmp_path, module)
    assert module.calls.n == 2

    # forced, and a missing data file
    run(tmp_path, module, recompute=True)
    assert module.calls.n == 3
    (tmp_path / 'Results' / 'Cached' / 'data' / 'values.feather').unlink()
    run(tmp_path, module)
    assert module.calls.n == 4
    run(tmp_path, module)
    assert module.calls.n == 4


def test_unused_setting_does_not_recompute(tmp_path):
    ds = make_ds(tmp_path)
    module = load_script(tmp_path, ds)
    run(tmp_path, module)
    module.window = (5, 6)  # not used by compute
    run(tmp_path, module)
    assert module.calls.n == 1


def test_dict_result(tmp_path):
    ds = make_ds(tmp_path)
    module = load_script(tmp_path, ds)
    for _ in range(2):
        res = AnalysisResult(tmp_path, 'Two', script=module.__file__)
        data = res.cached_data('both', module.compute_two)
        res.set_sessions(['s1'])
        res.save()
    assert module.calls.n == 1
    assert list(data['table']['y']) == [1]
    assert numpy.array_equal(data['array'], numpy.arange(3))
    assert sorted(e['name'] for e in res.items['data']) == ['array', 'table']
