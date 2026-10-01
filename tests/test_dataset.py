'''
DataSet storage: each column keeps one type through set_field, saving, and reloading.
'''
import json

import numpy
import pandas
import pytest

from Proc2P.Analysis.AnalysisClasses.DataSet import DataSet


def make_ds(path, prefixes=('s1', 's2', 's3')):
    ds = DataSet(path)
    for pf in prefixes:
        ds.add_row({'Prefix': pf, 'Incl': True})
    ds.save_df()
    return ds


def field(ds, prefix, key):
    return ds.df.loc[ds.df['Prefix'].eq(prefix), key].iloc[0]


def test_types_survive_partial_and_full_columns(tmp_path):
    ds = make_ds(tmp_path)
    ds.set_field('s1', 'Done')
    ds.save_df()
    ds = DataSet(tmp_path)
    assert ds.df['Done'].dtype == 'boolean'
    ds.set_field(['s2', 's3'], 'Done')
    ds.save_df()
    ds = DataSet(tmp_path)
    assert ds.df['Done'].dtype == 'boolean'
    # a fully set bool column still accepts a new session and False
    ds.add_row({'Prefix': 's4', 'Incl': True})
    ds.set_field('s1', 'Done', False)
    ds.save_df()
    ds = DataSet(tmp_path)
    assert ds.df['Done'].dtype == 'boolean'
    assert ds.df['Done'].tolist() == [False, True, True, pandas.NA]


def test_first_value_sets_type(tmp_path):
    ds = make_ds(tmp_path)
    ds.set_by_dict('s1', {'Fps': numpy.float64(30.1), 'Frames': numpy.int64(9000), 'Note': 'x', 'Ephys': True,
                          'Date': pandas.Timestamp('2026-10-01'), 'Drug': numpy.nan})
    ds.set_by_dict('s2', {'Fps': 30, 'Frames': 100, 'Note': 'y', 'Ephys': numpy.False_, 'Drug': 'KA'})
    ds.save_df()
    ds = DataSet(tmp_path)
    assert {k: str(ds.df[k].dtype) for k in ('Fps', 'Frames', 'Note', 'Ephys', 'Date', 'Drug')} == {
        'Fps': 'Float64', 'Frames': 'Int64', 'Note': 'string', 'Ephys': 'boolean', 'Date': 'datetime64[ns]',
        'Drug': 'string'}
    assert field(ds, 's2', 'Fps') == 30.0
    assert field(ds, 's3', 'Frames') is pandas.NA


@pytest.mark.parametrize('value', [0, 1, 'TRUE', 1.0])
def test_mismatch_raises_with_types(tmp_path, value):
    ds = make_ds(tmp_path)
    ds.set_field('s1', 'Done')
    with pytest.raises(TypeError, match=r"Column 'Done' has type bool.*Pass type=bool"):
        ds.set_field('s2', 'Done', value)


@pytest.mark.parametrize('value, expected', [(0, False), (1, True), ('TRUE', True), ('false', False), (1.0, True),
                                             ('', pandas.NA), (numpy.nan, pandas.NA)])
def test_type_kwarg_converts_value(tmp_path, value, expected):
    ds = make_ds(tmp_path)
    ds.set_field('s1', 'Done')
    ds.set_field('s2', 'Done', value, type=bool)
    assert ds.df['Done'].dtype == 'boolean'
    got = field(ds, 's2', 'Done')
    assert got is pandas.NA if expected is pandas.NA else got == expected


def test_type_kwarg_converts_column(tmp_path):
    ds = make_ds(tmp_path)
    ds.set_field('s1', 'Done', 1)  # column started as int by mistake
    ds.set_field('s2', 'Done', True, type=bool)
    assert ds.df['Done'].dtype == 'boolean'
    assert ds.df['Done'].tolist() == [True, True, pandas.NA]
    # a column value that can't be converted is reported with its prefix
    ds.set_field('s2', 'Frames', 2.5)
    with pytest.raises(TypeError, match="Column 'Frames', prefix s2"):
        ds.set_field('s1', 'Frames', 3, type=int)


def test_set_by_dict_type_per_key(tmp_path):
    ds = make_ds(tmp_path)
    ds.set_by_dict('s1', {'Wavelength': '920', 'Frames': '9000'}, type={'Frames': int})
    assert ds.df['Wavelength'].dtype == 'string'
    assert ds.df['Frames'].dtype == 'Int64'


def test_progress_pattern(tmp_path):
    ds = make_ds(tmp_path)
    for pf in ds.get_incl(excl='Done')['Prefix']:
        ds.set_field(pf, 'Done')
        ds.save_df()
        break
    ds = DataSet(tmp_path)
    assert list(ds.get_incl(excl='Done')['Prefix']) == ['s2', 's3']
    ds.set_field('s2', 'Done')
    ds.exclude('s3')
    ds.save_df()
    ds = DataSet(tmp_path)
    assert list(ds.get_incl(excl='Done')['Prefix']) == []
    assert list(ds.get_incl()['Prefix']) == ['s1', 's2']


def test_direct_df_assignment(tmp_path):
    ds = make_ds(tmp_path)
    ds.df['Done'] = False  # numpy bool column set by a script
    ds.set_field('s1', 'Done')
    ds.df['Count'] = numpy.arange(3)
    ds.save_df()
    ds = DataSet(tmp_path)
    assert ds.df['Done'].dtype == 'boolean' and ds.df['Done'].tolist() == [True, False, False]
    assert ds.df['Count'].dtype == 'Int64'
    ds.df['Mixed'] = pandas.Series([True, 'x', None], dtype=object)
    with pytest.raises(TypeError, match=r"mixed types \(bool, str\).*set_type"):
        ds.save_df()
    ds.set_type('Mixed', str)
    ds.save_df()


def test_cell_lists(tmp_path):
    ds = make_ds(tmp_path)
    ds.include_cells('s1', 'R', [3, 1])
    ds.include_cells('s1', 'R', [2])
    ds.save_df()
    ds = DataSet(tmp_path)
    assert ds.get_cells('s1', 'R') == [1, 2, 3]
    assert ds.get_cells('s2', 'R') is None


def test_legacy_file(tmp_path):
    legacy = pandas.DataFrame({
        'Prefix': ['s1', 's2'],
        'Incl': [True, True],
        'Excl': [numpy.nan, numpy.nan],
        'Done': pandas.Series([True, None], dtype=object),
        'Fps': [30.0, numpy.nan],
        'Cells.R.Incl': [json.dumps([1, 2]), json.dumps(None)],
    })
    legacy.to_feather(tmp_path / '_dataset00.feather')
    ds = DataSet(tmp_path)
    assert {k: str(v) for k, v in ds.df.dtypes.items()} == {
        'Prefix': 'string', 'Incl': 'boolean', 'Excl': 'object', 'Done': 'boolean', 'Fps': 'Float64',
        'Cells.R.Incl': 'string'}
    assert ds.get_cells('s1', 'R') == [1, 2]
    assert list(ds.get_incl(excl='Done')['Prefix']) == ['s2']
    ds.exclude('s1')
    ds.save_df()
    ds = DataSet(tmp_path)
    assert ds.df['Excl'].dtype == 'boolean'
    assert list(ds.get_incl()['Prefix']) == ['s2']


def test_unsupported_value(tmp_path):
    ds = make_ds(tmp_path)
    with pytest.raises(TypeError, match='Pass type=str'):
        ds.set_field('s1', 'Bad', [1, 2])
