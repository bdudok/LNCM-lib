import pytest

from Proc2P.Analysis.AnalysisClasses.AnalysisResult import CONFIRMED_ENV


@pytest.fixture(autouse=True)
def uncommitted_confirmed(monkeypatch):
    '''AnalysisResult does not wait for enter when the library has uncommitted changes.'''
    monkeypatch.setenv(CONFIRMED_ENV, '1')
