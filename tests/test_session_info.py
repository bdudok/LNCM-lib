'''SessionInfo.load: None for a missing file by default, an error naming the missing folder or file with must_exist.'''
import json

import pytest

from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Bruker.PreProc import SessionInfo

PREFIX = 'Test1_2026-01-01_sessioninfo_001'


def test_missing_file_returns_none(tmp_path):
    assert SessionInfo(tmp_path, PREFIX).load() is None


def test_missing_folder_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='Session folder'):
        SessionInfo(tmp_path / PREFIX, PREFIX).load(must_exist=True)


def test_missing_file_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='no SessionInfo file'):
        SessionInfo(tmp_path, PREFIX).load(must_exist=True)


def test_load(tmp_path):
    with open(tmp_path / f'{PREFIX}_SessionInfo.json', 'w') as f:
        json.dump({'framerate': 30.0}, f)
    assert SessionInfo(tmp_path, PREFIX).load(must_exist=True)['framerate'] == 30.0


def test_imaging_session_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match='Session folder'):
        ImagingSession(tmp_path, PREFIX)
