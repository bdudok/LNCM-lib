import pytest

from Proc2P.Analysis.AnalysisClasses import AnalysisResult as analysis_result


@pytest.fixture(autouse=True)
def uncommitted_confirmed(monkeypatch):
    '''AnalysisResult does not wait for enter when the library has uncommitted changes.'''
    monkeypatch.setenv(analysis_result.CONFIRMED_ENV, '1')


@pytest.fixture(autouse=True)
def no_mouse_db(monkeypatch):
    '''AnalysisResult does not read Baserow: no animal or session is found.'''
    monkeypatch.setattr(analysis_result, 'read_mouse_db',
                        lambda animals, prefixes: ({a: None for a in animals}, {p: None for p in prefixes}))
