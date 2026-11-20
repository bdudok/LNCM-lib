'''Ripples on a synthetic two-channel LFP: per-channel folders and reading files saved without a channel'''
import numpy
import pytest

from Proc2P.Analysis.Ripples import Ripples, RippleConfig, ripple_folder, legacy_channel_from_config, LEGACY_CHANNEL_FILE

PREFIX = 'synthetic'
FS = 5000


@pytest.fixture
def procpath(tmp_path):
    '''session with frame row and two LFP channels (microvolts); 160 Hz bursts only on channel 2'''
    rng = numpy.random.default_rng(0)
    n = FS * 20
    t = numpy.arange(n) / FS
    burst = numpy.zeros(n)
    for start in range(2, 18, 2):
        w = (t > start) & (t < start + 0.05)
        burst[w] = 200 * numpy.sin(2 * numpy.pi * 160 * t[w])
    edat = numpy.stack([numpy.arange(n) // 100, rng.normal(0, 20, n), rng.normal(0, 20, n) + burst]).astype('int32')
    (tmp_path / PREFIX).mkdir()
    numpy.save(tmp_path / PREFIX / f'{PREFIX}_ephys.npy', edat)
    return tmp_path


def write_legacy(session_path):
    root = session_path / '_ripples'
    root.mkdir()
    numpy.zeros(10).tofile(root / 'ftr.np')
    numpy.zeros((1, 2)).tofile(root / 'curated.ripples')
    return root


def never_ask(prefix, n_channels):
    raise AssertionError('asked for the channel')


def test_channels_kept_apart(procpath):
    r1 = Ripples(procpath, PREFIX, fs=FS, ephys_channel=1)
    r2 = Ripples(procpath, PREFIX, fs=FS, ephys_channel=2)
    assert r1.path != r2.path
    assert not numpy.array_equal(r1.ftr, r2.ftr)
    assert len(r2.events) == 8
    # opened again without force, each channel loads its own filtered trace
    assert numpy.array_equal(Ripples(procpath, PREFIX, fs=FS, ephys_channel=2).ftr, r2.ftr)


def test_cycle_criterion(procpath):
    r = Ripples(procpath, PREFIX, fs=FS, ephys_channel=2, force=True)
    assert len(r.events) == 8
    # 50 ms bursts of 160 Hz, the tr2 crossings are a bit wider
    assert all(8 <= r.count_cycles(*e.p) <= 14 for e in r.events)
    r.min_cycles = 100
    r.pick_events(skip_save=True)
    assert len(r.events) == 0
    r.require_3_cycles = False
    r.pick_events(skip_save=True)
    assert len(r.events) == 8


def test_empty_tag(procpath):
    r = Ripples(procpath, PREFIX, fs=FS, ephys_channel=2, tag='')
    saved = [f.name for f in r.path.glob('*.ripples')]
    assert len(saved) == 1 and saved[0] != '.ripples'


def test_legacy_single_channel(tmp_path):
    root = write_legacy(tmp_path)
    assert ripple_folder(tmp_path, 1, 1, ask=never_ask) == root
    assert not (root / LEGACY_CHANNEL_FILE).exists()


@pytest.mark.parametrize('answer', [2, None])
def test_legacy_multi_channel(tmp_path, answer):
    root = write_legacy(tmp_path)
    asked = []
    folder = ripple_folder(tmp_path, 2, 2, ask=lambda prefix, n: asked.append(n) or answer)
    assert asked == [2]
    assert folder == (root if answer == 2 else root / 'Ch2')
    # the answer is stored: not asked again, and the other channel gets its own folder
    assert ripple_folder(tmp_path, 1, 2, ask=never_ask) == root / 'Ch1'


def test_legacy_channel_from_config(tmp_path):
    root = write_legacy(tmp_path)
    with pytest.raises(ValueError, match='old_files_channel'):
        ripple_folder(tmp_path, 1, 2, ask=legacy_channel_from_config(RippleConfig()))
    assert not (root / LEGACY_CHANNEL_FILE).exists()
    assert ripple_folder(tmp_path, 1, 2, ask=legacy_channel_from_config(RippleConfig(old_files_channel=1))) == root
    assert legacy_channel_from_config(RippleConfig(old_files_channel=1, discard_old_files=True))('x', 2) is None


def test_no_prefix():
    Ripples('.', None)
