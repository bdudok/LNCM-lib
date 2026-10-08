'''
Proc2P.utils.read_excel: the default index that pandas writes is dropped, other first columns are kept.
'''
import pandas

from Proc2P.utils import read_excel


def test_default_index_dropped(tmp_path):
    fn = tmp_path / 'a.xlsx'
    pandas.DataFrame({'Prefix': ['a', 'b'], 'k': [1.0, 2.0]}).to_excel(fn)
    df = read_excel(fn)
    assert df.columns.tolist() == ['Prefix', 'k']
    # saving and reading again does not add columns
    df.to_excel(fn)
    assert read_excel(fn).columns.tolist() == ['Prefix', 'k']


def test_labeled_index_kept(tmp_path):
    fn = tmp_path / 'a.xlsx'
    pandas.DataFrame({'Photons': [1.0, 2.0]}, index=['Ch1', 'Ch2']).to_excel(fn)
    df = read_excel(fn)
    assert df.columns.tolist() == ['index', 'Photons']
    assert df['index'].tolist() == ['Ch1', 'Ch2']


def test_saved_without_index(tmp_path):
    fn = tmp_path / 'a.xlsx'
    pandas.DataFrame({'Prefix': ['a', 'b'], 'k': [1.0, 2.0]}).to_excel(fn, index=False)
    assert read_excel(fn).columns.tolist() == ['Prefix', 'k']


def test_keep_index(tmp_path):
    fn = tmp_path / 'a.xlsx'
    pandas.DataFrame({'Photons': [1.0, 2.0]}, index=['Ch1', 'Ch2']).to_excel(fn)
    df = read_excel(fn, keep_index=True)
    assert df.columns.tolist() == ['Photons']
    assert df.index.tolist() == ['Ch1', 'Ch2']
