import os
import sys

import pytest

if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    # importing the BatchGUI package sets matplotlib to QtAgg, which matplotlib refuses without a display
    pytest.skip('needs a display (the BatchGUI package sets the QtAgg backend)', allow_module_level=True)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy  # noqa: E402
import tifffile  # noqa: E402
from PySide6 import QtCore, QtTest, QtWidgets  # noqa: E402

from Proc2P.Analysis.BatchGUI import RoiEditorQt  # noqa: E402
from Proc2P.Analysis.RoiEditor import RoiEditor  # noqa: E402

PREFIX = 'Mouse1_2026-01-01_test_001'


def square(x, y, size):
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]


@pytest.fixture
def editor(tmp_path, monkeypatch):
    '''Editor on a synthetic session: a 100 x 100 preview and an input set of 4 squares (2 of them in the top half)'''
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / 'settings.ini'), QtCore.QSettings.IniFormat)
    monkeypatch.setattr(RoiEditorQt.RoiEditorWindow, 'settings', settings)
    folder = tmp_path / PREFIX
    folder.mkdir()
    tifffile.imwrite(folder / f'{PREFIX}_avgmax.tif', numpy.zeros((100, 100, 3), dtype='uint8'))
    rois = [square(10, 10, 6), square(40, 10, 10), square(10, 60, 6), square(60, 60, 10)]
    RoiEditor.save_roi(rois, folder / f'{PREFIX}_saved_roi_auto', (100, 100))
    window = RoiEditorQt.open_editor(tmp_path, PREFIX, 'out')
    app.processEvents()
    yield window
    window.unsaved = False
    window.close()
    app.processEvents()


def test_click_lasso_and_save(editor, tmp_path):
    assert editor.roi_list.currentText() == 'auto'
    assert len(editor.input_polys) == 4

    # left click on an input ROI adds it; again does nothing (contain mode)
    for _ in range(2):
        editor.on_click(editor.input_view, QtCore.QPointF(13, 13), QtCore.Qt.LeftButton)
    assert len(editor.output_polys) == 1
    # left click on an output ROI removes it
    editor.on_click(editor.output_view, QtCore.QPointF(13, 13), QtCore.Qt.LeftButton)
    assert len(editor.output_polys) == 0

    # lasso around the top half adds the 2 ROIs whose centroid is inside
    editor.set_tool(RoiEditorQt.Tool.LASSO)
    editor.on_stroke(editor.input_view, [(0, 0), (100, 0), (100, 40), (0, 40)])
    assert len(editor.output_polys) == 2
    # only the lassoed ROIs stay selected: right click adds only these
    assert editor.included.tolist() == [True, True, False, False]
    assert editor.tool == RoiEditorQt.Tool.NONE  # a tool ends after one stroke
    editor.clear_output()
    editor.on_click(editor.input_view, QtCore.QPointF(1, 1), QtCore.Qt.RightButton)
    assert len(editor.output_polys) == 2

    # moving a size slider resets the selection; right click then adds the large ROIs
    editor.min_size.setValue(8)
    assert editor.included.tolist() == [False, True, False, True]
    editor.clear_output()
    editor.on_click(editor.input_view, QtCore.QPointF(1, 1), QtCore.Qt.RightButton)
    assert len(editor.output_polys) == 2

    fn = editor.save()
    assert fn == f'{PREFIX}_saved_roi_out'
    assert len(RoiEditor.load_roi(tmp_path / PREFIX / f'{fn}.npy')) == 2
    # the tag exists now: the next save gets a number
    assert editor.save() == f'{PREFIX}_saved_roi_1'


def test_remembers_input_set(editor, tmp_path):
    RoiEditor.save_roi([square(5, 5, 5)], tmp_path / PREFIX / f'{PREFIX}_saved_roi_other', (100, 100))
    editor.close()
    window = RoiEditorQt.open_editor(tmp_path, PREFIX, 'out')
    window.roi_list.setCurrentIndex(window.roi_list.findText('other'))
    window.close()
    window = RoiEditorQt.open_editor(tmp_path, PREFIX, 'out')
    assert window.roi_list.currentText() == 'other'
    window.close()


def test_draw_with_mouse(editor):
    '''A stroke drawn with the mouse starts at the first point (not at 0, 0), and the tool ends after it'''
    view = editor.output_view
    editor.on_click(view, QtCore.QPointF(50, 50), QtCore.Qt.RightButton)  # right click on the output: draw
    assert editor.tool == RoiEditorQt.Tool.DRAW
    points = [view.mapFromScene(QtCore.QPointF(x, y)) for x, y in ((20, 20), (40, 20), (40, 40), (20, 40), (15, 30))]
    QtTest.QTest.mousePress(view.viewport(), QtCore.Qt.LeftButton, pos=points[0])
    for point in points[1:]:
        QtTest.QTest.mouseMove(view.viewport(), point)
    shown = view._stroke_item.path().elementAt(0)  # the line on the screen
    assert abs(shown.x - 20) < 2 and abs(shown.y - 20) < 2
    QtTest.QTest.mouseRelease(view.viewport(), QtCore.Qt.LeftButton, pos=points[-1])
    assert len(editor.output_polys) == 1
    assert editor.tool == RoiEditorQt.Tool.NONE
