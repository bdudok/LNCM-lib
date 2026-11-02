"""GetSessions passes its timeout to the requests that the Recorder depends on."""
from types import SimpleNamespace
from unittest import mock

import pytest

from BaserowAPI import BaserowRequests
from BaserowAPI.BaserowRequests import GetSessions


@pytest.mark.parametrize('timeout', [None, 5])
def test_timeout_is_passed_to_requests(monkeypatch, timeout):
    fake = mock.MagicMock()
    fake.get.return_value.json.return_value = {'results': [{'Image.ID': 'x'}]}
    fake.post.return_value = SimpleNamespace(status_code=200)
    monkeypatch.setattr(BaserowRequests, 'requests', fake)
    db = GetSessions(token='fake', timeout=timeout)
    db.get_session('x')
    db.put_new({})
    assert fake.get.call_args.kwargs['timeout'] == timeout
    assert fake.post.call_args.kwargs['timeout'] == timeout
