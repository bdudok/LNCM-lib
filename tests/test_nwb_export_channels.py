'''The NWB export does not depend on how the session is opened: ripples of every LFP channel, no session options'''
from types import SimpleNamespace

import numpy
import pytest
import yaml

from NWB_converter.ExportImagingSession import ExportImagingSession
from NWB_converter.NWBExporter import NWBExporter

PREFIX = 's_1'


def test_ripples_of_each_channel(tmp_path):
    numpy.save(tmp_path / f'{PREFIX}_ephys.npy', numpy.zeros((4, 10)))  # frame row and 3 LFP channels
    events = {1: {'a': [[1, 2], [5, 6]], 'b': [[3, 4]]}, 3: {'a': [[7, 9]]}}
    for channel, tags in events.items():
        folder = tmp_path / '_ripples' / f'Ch{channel}'
        folder.mkdir(parents=True)
        for tag, x in tags.items():
            numpy.asarray(x, dtype='float64').tofile(folder / f'{tag}.ripples')
    session = SimpleNamespace(path=tmp_path, prefix=PREFIX)
    exported = ExportImagingSession.export_ripples(session, [1, 2, 3])
    assert sorted(exported) == [1, 3]
    numpy.testing.assert_array_equal(exported[1]['a'], events[1]['a'])
    numpy.testing.assert_array_equal(exported[3]['a'], events[3]['a'])
    assert list(ExportImagingSession.export_ripples(session, [1, 2, 3], tags=['b'])) == [1]


@pytest.mark.parametrize('section, option', [('session', {'imaging_session_kwargs': {'lfp_ch': 1}}),
                                             ('lfp', {'channels': [1]})])
def test_session_options_are_rejected(tmp_path, section, option):
    fn = tmp_path / 'config.yaml'
    fn.write_text(yaml.safe_dump({section: option}))
    with pytest.raises(ValueError, match='not used'):
        NWBExporter(PREFIX, fn, session=object())
