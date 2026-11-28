'''Position of the NWB export (ExportImagingSession.export_position) with and without enough laps'''
from types import SimpleNamespace

import numpy

from envs.site_config import CONFIG
from NWB_converter.ExportImagingSession import ExportImagingSession


def fake_session(pos, laps, relpos=None):
    sync = SimpleNamespace(load=lambda key: None if pos is None else numpy.asarray(pos, dtype='float64'))
    return SimpleNamespace(ca=SimpleNamespace(sync=sync, frames=4),
                           pos=SimpleNamespace(laps=None if laps is None else numpy.asarray(laps, dtype='float64'),
                                               relpos=relpos))


def test_position_within_the_lap_with_enough_laps():
    relpos = numpy.array([0.1, 0.5, 0.9, 0.2])
    session = fake_session([10, 50, 90, 20], [0, 1, 2, 3], relpos)
    numpy.testing.assert_array_equal(ExportImagingSession.export_position(session), relpos)


def test_distance_in_default_laps_with_few_laps():
    session = fake_session([numpy.nan, 30, 130, 430], [numpy.nan, 0, 0, 1])
    expected = numpy.array([numpy.nan, 0, 100, 400]) / CONFIG.default_lap_length
    numpy.testing.assert_array_equal(ExportImagingSession.export_position(session), expected)


def test_no_position_is_nan():
    for pos, laps in ((None, None), ([numpy.nan] * 4, [numpy.nan] * 4)):
        assert numpy.isnan(ExportImagingSession.export_position(fake_session(pos, laps))).all()
