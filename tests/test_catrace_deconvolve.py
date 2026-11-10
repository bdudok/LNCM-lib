'''
CaTrace.deconvolve on synthetic traces: a cell's nnd must not depend on the other cells,
and must scale with the cell's own trace (to float32 precision, which oasis uses).
'''
import numpy

from Proc2P.Analysis.CaTrace import CaTrace


def make_trace(rel):
    ca = CaTrace.__new__(CaTrace)
    ca.rel = rel
    ca.version_info = {'fps': '30'}
    return ca


def synthetic_rel(n_cells=4, n_frames=3000, seed=0):
    rng = numpy.random.default_rng(seed)
    kernel = numpy.exp(-numpy.arange(90) / 30 / 0.75)
    spikes = (rng.random((n_cells, n_frames)) < 0.005).astype(float)
    rel = numpy.array([numpy.convolve(s, kernel)[:n_frames] for s in spikes])
    return rel + rng.normal(0, 0.05, rel.shape)


def deconvolved(rel):
    ca = make_trace(rel)
    ca.deconvolve()
    return ca.nnd


def test_nnd_independent_of_other_cells():
    rel = synthetic_rel()
    alone = deconvolved(rel[:1])[0]
    other = rel.copy()
    other[1:] = other[1:] * 5 - 2  # change the other cells only
    numpy.testing.assert_allclose(deconvolved(rel)[0], alone, rtol=1e-5, atol=1e-6)
    numpy.testing.assert_allclose(deconvolved(other)[0], alone, rtol=1e-5, atol=1e-6)


def test_nnd_scales_with_cell():
    rel = synthetic_rel()
    scaled = rel.copy()
    scaled[0] *= 1.5
    numpy.testing.assert_allclose(deconvolved(scaled)[0], 1.5 * deconvolved(rel)[0], rtol=1e-5, atol=1e-6)


def test_dual_channel_matches_single():
    rel = synthetic_rel()
    dual = numpy.stack([rel, 2 * rel], axis=2)
    nnd = deconvolved(dual)
    numpy.testing.assert_allclose(nnd[..., 0], deconvolved(rel), rtol=1e-5, atol=1e-6)
