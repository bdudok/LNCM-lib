import numpy

from Proc2P.Analysis.LoadPolys import load_roi_file


def test_load_roi_file_keeps_all_vertices(tmp_path):
    # rows of the saved ROI file: roi index, x, y
    squares = [[[0, 0], [0, 4], [4, 4], [4, 0]],
               [[10, 10], [10, 14], [14, 14], [14, 10]],
               [[20, 0], [20, 4], [24, 4], [24, 0]]]
    rows = [[i, x, y] for i, square in enumerate(squares) for x, y in square]
    roi_name = tmp_path / 'saved_roi_test.npy'
    numpy.save(roi_name, numpy.array(rows, dtype='float64'))

    polys = load_roi_file(roi_name)

    assert len(polys) == len(squares)
    for poly, square in zip(polys, squares):
        numpy.testing.assert_array_equal(poly, square)
