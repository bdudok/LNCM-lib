import os
import sys

import pytest

if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    # importing the BatchGUI package sets matplotlib to QtAgg, which matplotlib refuses without a display
    pytest.skip('needs a display (the BatchGUI package sets the QtAgg backend)', allow_module_level=True)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy  # noqa: E402
import tifffile  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

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

    # the size filter limits right click (add all) to the large ROIs
    editor.min_size.setValue(8)
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
