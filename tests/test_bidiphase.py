'''
Bidiphase offset estimation on synthetic movies with tilted structures, where suite2p's estimator counts the tilt as
offset, and the corrected copy of a tif movie.
'''
import inspect

import numpy
import pytest
from tifffile import TiffFile, TiffWriter, imwrite

from Proc2P.Bruker import Bidiphase


def tilted_movie(n_frames=50, Ly=48, Lx=256, tilt=1.6, noise=0.5, seed=0):
    '''smooth random texture with each line shifted by tilt px from the one above, plus noise per frame'''
    rng = numpy.random.default_rng(seed)
    kx = numpy.fft.rfftfreq(Lx)
    spectrum = numpy.fft.rfft(rng.normal(size=Lx)) * numpy.exp(-(kx / 0.05) ** 2)
    # same line profile, shifted along x by tilt per line, with slowly varying amplitude across lines
    lines = numpy.fft.irfft(spectrum * numpy.exp(-2j * numpy.pi * kx * tilt * numpy.arange(Ly)[:, None]), n=Lx)
    image = 100 + 20 * lines / lines.std() * (1 + 0.3 * numpy.sin(numpy.arange(Ly) / 5))[:, None]
    return image + noise * rng.normal(size=(n_frames, Ly, Lx))


@pytest.mark.parametrize('offset', [-3, -1, 0, 2])
@pytest.mark.parametrize('tilt', [0, 1.6, -2.5])
def test_estimate(offset, tilt):
    movie = [Bidiphase.shift(frame, -offset) for frame in tilted_movie(tilt=tilt)]  # odd lines displaced
    assert Bidiphase.estimate(movie) == pytest.approx(offset, abs=0.3)


def test_shift_round_trip():
    frame = tilted_movie(n_frames=1)[0]
    restored = Bidiphase.shift(Bidiphase.shift(frame, 3), -3)
    assert numpy.array_equal(restored[:, :-3], frame[:, :-3])
    assert numpy.array_equal(restored[::2], frame[::2])


@pytest.mark.skipif('ome' not in inspect.signature(TiffWriter).parameters,
                    reason='tifffile too old; MotionCorrect runs in the suite2p env')
def test_write_corrected(tmp_path):
    movie = tilted_movie(n_frames=20).astype(numpy.uint16)
    src, dst = tmp_path / 'raw', tmp_path / 'corrected'
    src.mkdir()
    fn = 'test_Ch2_000001.ome.tif'
    imwrite(src / fn, movie)
    numpy.save(src / 'bad_frames.npy', numpy.array([3]))
    Bidiphase.write_corrected(src, [fn], dst, -2)
    with TiffFile(dst / fn) as tif:
        corrected = numpy.stack([page.asarray() for page in tif.pages])
    assert numpy.array_equal(corrected, numpy.stack([Bidiphase.shift(frame, -2) for frame in movie]))
    assert (dst / 'bad_frames.npy').exists()
