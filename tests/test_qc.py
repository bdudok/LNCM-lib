'''
ImagingSession.qc on a dual channel session (BL6-47, SLROI): with ch='All', a cell passes if it passes in either channel.
Test data: see test_nwb_export.py. Skipped if the session is missing.
'''
import os
from pathlib import Path

import numpy
import pytest

from Proc2P.Analysis.ImagingSession import ImagingSession

DEFAULT_DATA_DIR = Path('/test_data') if os.path.isdir('/test_data') else Path(__file__).resolve().parents[2] / '_test_data'
PROCPATH = Path(os.environ.get('LNCM_TEST_DATA', DEFAULT_DATA_DIR)) / '_Processed'
PREFIX = 'BL6-47_2025-10-09_PlaceField_087'


@pytest.mark.skipif(not (PROCPATH / PREFIX).exists(), reason=f'test data not found in {PROCPATH}')
def test_qc_dual_channel():
    single = [ImagingSession(PROCPATH, PREFIX, tag='SLROI', ch=ch).qc() for ch in (0, 1)]
    dual = ImagingSession(PROCPATH, PREFIX, tag='SLROI', ch='All').qc()
    assert dual.shape == single[0].shape
    numpy.testing.assert_array_equal(dual, numpy.logical_or(*single))
