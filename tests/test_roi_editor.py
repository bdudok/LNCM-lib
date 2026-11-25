import os
import sys

import pytest

if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    # importing the BatchGUI package sets matplotlib to QtAgg, which matplotlib refuses without a display
    pytest.skip('needs a display (the BatchGUI package sets the QtAgg backend)', allow_module_level=True)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy  # noqa: E402
import tifffile  # noqa: E402
from PySide6 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402

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
    assert editor.tool == RoiEditorQt.Tool.LASSO  # the tool stays on
    editor.set_tool(RoiEditorQt.Tool.NONE)
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
    '''Left clicks add vertices, a drag adds points, right click closes the polygon; the tool stays on until a
    right click with no polygon started.
    The line on the screen starts at the first vertex (not at 0, 0).'''
    view = editor.output_view
    port = view.viewport()
    editor.on_click(view, QtCore.QPointF(50, 50), QtCore.Qt.RightButton)  # right click on the output: draw
    assert editor.tool == RoiEditorQt.Tool.DRAW
    to_view = [view.mapFromScene(QtCore.QPointF(x, y)) for x, y in ((20, 20), (40, 20), (40, 40), (20, 40))]
    for point in to_view[:2]:
        QtTest.QTest.mouseClick(port, QtCore.Qt.LeftButton, pos=point)
    shown = view._stroke_item.path().elementAt(0)
    assert abs(shown.x - 20) < 2 and abs(shown.y - 20) < 2
    # drag with the left button held from the 3rd to the 4th vertex
    QtTest.QTest.mousePress(port, QtCore.Qt.LeftButton, pos=to_view[2])
    move = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(to_view[3]), QtCore.QPointF(to_view[3]),
                             QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(port, move)
    QtTest.QTest.mouseRelease(port, QtCore.Qt.LeftButton, pos=to_view[3])
    assert len(view._stroke) == 4 and not editor.output_polys  # releasing does not close the polygon
    QtTest.QTest.mouseClick(port, QtCore.Qt.RightButton, pos=to_view[3])
    assert len(editor.output_polys) == 1
    assert editor.tool == RoiEditorQt.Tool.DRAW
    # right click with no polygon started ends the tool, and starts it again on the output pane
    QtTest.QTest.mouseClick(port, QtCore.Qt.RightButton, pos=to_view[0])
    assert editor.tool == RoiEditorQt.Tool.NONE
    QtTest.QTest.mouseClick(port, QtCore.Qt.RightButton, pos=to_view[0])
    assert editor.tool == RoiEditorQt.Tool.DRAW
    assert len(editor.output_polys) == 1


def test_start_selection(editor, tmp_path):
    '''input_tag and preview select the input set and preview at start, instead of the ones used last time'''
    RoiEditor.save_roi([square(5, 5, 5)], tmp_path / PREFIX / f'{PREFIX}_saved_roi_other', (100, 100))
    tifffile.imwrite(tmp_path / PREFIX / f'{PREFIX}_preview.tif', numpy.zeros((100, 100, 3), dtype='uint8'))
    editor.close()
    window = RoiEditorQt.open_editor(tmp_path, PREFIX, 'out', input_tag='other', preview='preview')
    assert window.roi_list.currentText() == 'other'
    assert window.preview_list.currentText() == 'preview'
    window.close()


def test_pan_and_wheel(editor):
    '''Left drag pans the zoomed view without adding ROIs, a left click adds one; sideways wheel scrolls, not zoom'''
    view = editor.input_view
    port = view.viewport()
    view.scale(8, 8)
    view.centerOn(50, 50)
    QtWidgets.QApplication.processEvents()
    h0, scale0 = view.horizontalScrollBar().value(), view.transform().m11()

    start = view.mapFromScene(QtCore.QPointF(13, 13))  # on the first input ROI
    end = start + QtCore.QPoint(60, 0)
    QtTest.QTest.mousePress(port, QtCore.Qt.LeftButton, pos=start)
    move = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(end), QtCore.QPointF(end),
                             QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(port, move)
    QtTest.QTest.mouseRelease(port, QtCore.Qt.LeftButton, pos=end)
    assert view.horizontalScrollBar().value() != h0
    assert not editor.output_polys

    QtTest.QTest.mouseClick(port, QtCore.Qt.LeftButton, pos=view.mapFromScene(QtCore.QPointF(13, 13)))
    assert len(editor.output_polys) == 1

    h1 = view.horizontalScrollBar().value()
    pos = QtCore.QPointF(port.rect().center())
    wheel = QtGui.QWheelEvent(pos, view.mapToGlobal(pos.toPoint()), QtCore.QPoint(), QtCore.QPoint(-120, 0),
                              QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(port, wheel)
    assert view.transform().m11() == scale0
    assert view.horizontalScrollBar().value() != h1
