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


def test_first_surgery_setter_edits_only_its_field(monkeypatch):
    from BaserowAPI.BaserowRequests import SetFirstSurgery

    fake = mock.MagicMock()
    monkeypatch.setattr(BaserowRequests, 'requests', fake)
    monkeypatch.setitem(BaserowRequests.config, 'surgery_token', '')
    with pytest.raises(ValueError, match='BASEROW_SURGERY_TOKEN'):
        SetFirstSurgery()

    monkeypatch.setitem(BaserowRequests.config, 'surgery_token', 'edit')
    setter = SetFirstSurgery()
    setter.set(624, '2025-09-03 10:00')
    call = fake.patch.call_args
    assert call.args[0].split('?')[0].endswith('/table/492/624/')
    assert call.kwargs['json'] == {'FirstSurgeryDate': '2025-09-03'}
    assert call.kwargs['headers']['Authorization'] == 'Token edit'
    with pytest.raises(AssertionError, match='Only FirstSurgeryDate'):
        setter.patch(624, {'FirstSurgeryDate': '2025-09-03', 'DOB*': '2025-01-01'})
    assert fake.patch.call_count == 1  # the refused request was not sent


def test_get_mice_reads_all_pages(monkeypatch):
    pages = [{'results': [{'id': 1}], 'next': 'http://next'}, {'results': [{'id': 2}], 'next': None}]
    fake = mock.MagicMock()
    fake.get.return_value.json.side_effect = pages
    monkeypatch.setattr(BaserowRequests, 'requests', fake)
    mice = GetSessions(token='fake').get_mice()
    assert list(mice['id']) == [1, 2] and fake.get.call_args.args[0] == 'http://next'
