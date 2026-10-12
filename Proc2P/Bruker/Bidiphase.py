import shutil
from itertools import islice
from pathlib import Path

import numpy
from tifffile import TiffFile, TiffWriter

'''
Bidirectional scan phase offset of the odd lines: estimation, and correction in a copy of the raw movie, which
MotionCorrect registers instead of the raw files. suite2p 1.1.0 can't apply the correction (it crashes on CPU,
MouseLand/suite2p issue #1239, and corrupts odd lines otherwise, #1265).

suite2p's estimator (bidiphase.compute) measures the x shift between odd lines and the even lines above them. Tilted
structures shift by a constant amount from line to line, so it counts the tilt as offset (e.g. -3 px estimated on
GEVI movies with diagonal membranes and an offset under 1 px). Here the shift is measured twice: from even lines to
the odd lines below them (tilt + offset) and from odd lines to the even lines below them (tilt - offset). Half the
difference is the offset without the tilt.

Offsets use suite2p's convention: shift(frame, offset) corrects the movie.
'''


def line_shift(cross_power, max_shift=10, upsample=10):
    '''
    x shift (px) at the peak of the cross-correlation, from the summed cross-power spectrum of line pairs.
    Not whitened (phase correlation): with whitening, frequencies without signal get the same weight as the
    structures, which biased the estimate by up to 1.4 px on synthetic movies.
    '''
    n = 2 * (len(cross_power) - 1) * upsample
    cc = numpy.fft.irfft(cross_power, n=n)
    lags = numpy.arange(-max_shift * upsample, max_shift * upsample + 1)
    return lags[numpy.argmax(cc[lags])] / upsample


def estimate(frames, max_shift=10):
    '''
    Offset of the odd lines (px, subpixel) in a sequence of frames (Ly, Lx).
    :param frames: iterable of 2D frames, read one at a time
    '''
    down = up = 0  # summed cross-power: even -> odd line below, odd -> even line below
    for frame in frames:
        lines = numpy.fft.rfft(numpy.asarray(frame, dtype=float), axis=1)
        m = (len(lines) - 1) // 2 * 2  # pairs (0, 1), (2, 3), ... and (1, 2), (3, 4), ... with the same count
        down = down + (lines[1:m:2] * numpy.conj(lines[0:m:2])).sum(axis=0)
        up = up + (lines[2:m + 1:2] * numpy.conj(lines[1:m:2])).sum(axis=0)
    # shift from line a to line b below it: tilt + d for even -> odd, tilt - d for odd -> even, where d is the
    # displacement of the odd lines. Correcting it means shifting them by -d.
    return -(line_shift(down, max_shift) - line_shift(up, max_shift)) / 2


def estimate_file(fn, n_frames=1000):
    '''offset estimated from the first frames of a tif file'''
    with TiffFile(fn) as tif:
        # iterate instead of len(tif.pages): counting parses every page header, and the link after the last
        # page of Bruker files is invalid, which makes tifffile warn
        return estimate(page.asarray() for page in islice(tif.pages, n_frames))


def shift(frame, offset):
    '''frame with the odd lines shifted by offset (int) px, as suite2p.registration.bidiphase.shift'''
    out = frame.copy()
    if offset > 0:
        out[1::2, offset:] = frame[1::2, :-offset]
    elif offset < 0:
        out[1::2, :offset] = frame[1::2, -offset:]
    return out


def write_corrected(src_path, file_list, dst_path, offset):
    '''copy the tif files from src_path to dst_path with the odd lines shifted by offset. Also copies bad_frames.npy'''
    src_path, dst_path = Path(src_path), Path(dst_path)
    shutil.rmtree(dst_path, ignore_errors=True)
    dst_path.mkdir(parents=True)
    for fn in file_list:
        with TiffFile(src_path / fn) as src, TiffWriter(dst_path / fn, bigtiff=True, ome=False) as dst:
            for page in src.pages:
                dst.write(shift(page.asarray(), offset), contiguous=True)
    if (src_path / 'bad_frames.npy').exists():
        shutil.copy2(src_path / 'bad_frames.npy', dst_path)
