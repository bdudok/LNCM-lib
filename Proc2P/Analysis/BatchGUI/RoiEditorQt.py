import os
import sys
from enum import Enum
from pathlib import Path

import numpy
import tifffile
from PySide6 import QtCore, QtGui, QtWidgets
from shapely.geometry import Point, Polygon

from Proc2P.Analysis.RoiEditor import RoiEditor
from Proc2P.AppIcon import set_app_icon
from Proc2P.utils import logger, lprint

'''
Gui for selecting ROIs from ROI sets (e.g. SIMA output), drawing ROIs, and saving them as a new ROI set.
Left pane: the input ROI set on the preview image. Right pane: the output ROIs. The panes zoom and pan together.
    Mouse wheel: zoom; sideways wheel: scroll sideways. Left or middle button drag: pan.
    f (Reset view): fit the whole image.
    Input pane: left click adds the ROI under the cursor, right click adds all ROIs within the size range.
    Output pane: left click removes the ROI under the cursor, right click starts drawing a ROI (as g).
    l / i: lasso on either pane, adds the input ROIs within the size range whose centroid is inside
        (i: dilated by 3 pixels). Only these stay selected (yellow), so a right click adds only these, until a size
        slider is moved.
    g: draw ROIs. r: lasso that removes the output ROIs whose centroid is inside.
        Left click adds a vertex (drag with the left button to add more), right click closes the polygon.
        The tool stays on for more polygons. Right click with no polygon started, or Esc, ends the tool.
    d: dilate all output ROIs by 1 pixel. c: clear the output. e: save.
Only the selected input ROI set is loaded; the image size is taken from the preview.
Open from another Qt app with open_editor (non-modal, in the same process), or run this file with the processed
path, prefix and tag as arguments.
'''

LINE_WIDTH = 1.5  # cosmetic pen width: pixels on the screen at any zoom (as 1 pt lines in matplotlib)
DRAG_PX = 4  # a left button press that moves more than this pans the view instead of clicking
COLORS = {'excluded': QtGui.QColor('gray'), 'included': QtGui.QColor('yellow'), 'output': QtGui.QColor('magenta'),
          'stroke': QtGui.QColor('cyan')}


class AddModes(Enum):
    CONTAIN = 0  # skip ROIs whose centroid is inside an output ROI
    OVERLAP = 1  # also skip ROIs that overlap an output ROI
    PRESERVE = 2  # add all


class Tool(Enum):
    NONE = 0
    LASSO = 1  # l
    LASSO_DILATE = 2  # i
    DRAW = 3  # g
    REMOVE = 4  # r


def make_pen(color):
    pen = QtGui.QPen(color, LINE_WIDTH)
    pen.setCosmetic(True)
    return pen


def to_qpolygon(coords):
    return QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in coords])


def centroid(coords):
    return numpy.asarray(coords, dtype=float).mean(axis=0)


class ImageView(QtWidgets.QGraphicsView):
    '''Zoom (wheel) and pan (middle button) a scene with the image; reports clicks and freehand strokes in image
    coordinates.'''
    clicked = QtCore.Signal(object, QtCore.QPointF, QtCore.Qt.MouseButton)  # view, scene position, button
    stroke_done = QtCore.Signal(object, list)  # view, [(x, y), ...]
    view_changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.SmartViewportUpdate)
        self.setBackgroundBrush(QtGui.QColor('black'))
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.pixmap_item = self.scene().addPixmap(QtGui.QPixmap())
        self.pixmap_item.setZValue(-1)
        self.drawing = False  # set by the window when a stroke tool is active
        self._stroke = []
        self._stroke_path = None  # kept here: a path item drops a path that has only its start point
        self._stroke_item = None
        self._pan_start = None
        self._left_press = None  # (view position, scene position) of a left press outside the drawing tools
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(lambda _: self.view_changed.emit(self))

    def set_image(self, qimage):
        self.pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.scene().setSceneRect(QtCore.QRectF(0, 0, qimage.width(), qimage.height()))

    def fit(self):
        self.fitInView(self.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.view_changed.emit(self)

    def wheelEvent(self, event):
        dx, dy = event.angleDelta().x(), event.angleDelta().y()
        if dy == 0:  # sideways wheel or tilt: scroll sideways
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
        else:
            factor = 1.25 if dy > 0 else 1 / 1.25
            self.scale(factor, factor)
        self.view_changed.emit(self)

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        if event.button() == QtCore.Qt.MiddleButton:
            self.start_pan(event.position())
        elif not self.drawing and event.button() == QtCore.Qt.LeftButton:
            # a click (sent on release) unless the mouse is dragged: then pan
            self._left_press = (event.position(), pos)
        elif self.drawing and event.button() == QtCore.Qt.LeftButton:
            # left click adds a vertex (dragging adds more), right click closes the polygon
            if self._stroke_item is None:
                self._stroke = [(pos.x(), pos.y())]
                self._stroke_path = QtGui.QPainterPath()
                self._stroke_path.moveTo(pos)
                self._stroke_item = self.scene().addPath(self._stroke_path, make_pen(COLORS['stroke']))
            else:
                self.add_stroke_point(pos)
        elif self.drawing and event.button() == QtCore.Qt.RightButton and self._stroke_item is not None:
            self.finish_stroke()
        else:
            self.clicked.emit(self, pos, event.button())

    def add_stroke_point(self, pos):
        last = self._stroke[-1]
        if max(abs(pos.x() - last[0]), abs(pos.y() - last[1])) >= 1:  # one point per image pixel at most
            self._stroke.append((pos.x(), pos.y()))
            self._stroke_path.lineTo(pos)
            self._stroke_item.setPath(self._stroke_path)

    def finish_stroke(self):
        '''Close the polygon. With 3 points or fewer it is discarded, and the tool stays on.'''
        if self._stroke_item is None:
            return
        self.cancel_stroke()
        stroke, self._stroke = self._stroke, []
        if len(stroke) > 3:
            self.stroke_done.emit(self, stroke)

    def cancel_stroke(self):
        if self._stroke_item is not None:
            self.scene().removeItem(self._stroke_item)
        self._stroke_item, self._stroke_path = None, None

    def start_pan(self, view_pos):
        self._pan_start = view_pos
        self.setCursor(QtCore.Qt.ClosedHandCursor)

    def end_pan(self):
        self._pan_start = None
        self.setCursor(QtCore.Qt.CrossCursor) if self.drawing else self.unsetCursor()

    def mouseMoveEvent(self, event):
        if self._left_press is not None and self._pan_start is None:
            moved = event.position() - self._left_press[0]
            if max(abs(moved.x()), abs(moved.y())) > DRAG_PX:
                self.start_pan(self._left_press[0])
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
        elif self._stroke_item is not None and event.buttons() & QtCore.Qt.LeftButton:
            self.add_stroke_point(self.mapToScene(event.position().toPoint()))

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton and self._pan_start is not None:
            self.end_pan()
        elif event.button() == QtCore.Qt.LeftButton and self._left_press is not None:
            if self._pan_start is not None:
                self.end_pan()
            else:
                self.clicked.emit(self, self._left_press[1], QtCore.Qt.LeftButton)
            self._left_press = None

    def keyPressEvent(self, event):
        # keys are handled by the window's shortcuts, not by scrolling the view
        event.ignore()


class RoiEditorWindow(QtWidgets.QMainWindow):
    __name__ = 'ROIEditor'  # for lprint
    settings = QtCore.QSettings('LNCM', 'RoiEditor')  # remembers the last input ROI set and preview

    def __init__(self, procpath, prefix, preferred_tag='1', input_tag=None, preview=None, parent=None):
        '''
        :param preferred_tag: tag of the saved ROI set (the next free number if it exists)
        :param input_tag: input ROI set to select at start (e.g. 'iPC-R20-100-200'). Default: the one used last time
        :param preview: preview to select at start (e.g. 'avgmax'). Default: the one used last time
        '''
        super().__init__(parent)
        self.procpath = Path(procpath)
        self.prefix = prefix
        self.opPath = self.procpath / prefix
        self.preferred_tag = preferred_tag
        self.start_selection = {'preview': preview, 'input_tag': input_tag}
        self.log = logger()
        self.log.set_handle(self.procpath, prefix)
        self.setWindowTitle(f'RoiEditor: {prefix}')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.input_polys = []  # numpy arrays (n, 2) of the current input set
        self.input_items = []
        self.input_paths = []  # QPainterPath per input ROI, for hit tests
        self.sizes = numpy.zeros(0)  # sqrt(area) per input ROI
        self.included = numpy.zeros(0, dtype=bool)
        self.output_polys = []
        self.output_items = []
        self.loaded_sets = {}
        self.preview_cache = {}
        self.image_shape = None
        self.tool = Tool.NONE
        self.addmode = AddModes.CONTAIN
        self.unsaved = False
        self._syncing = False

        self.make_widgets()
        self.find_files()
        self.resize(1400, 800)

    # ----- layout
    def make_widgets(self):
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel('Preview:'))
        self.preview_list = QtWidgets.QComboBox()
        self.preview_list.setMinimumWidth(150)
        controls.addWidget(self.preview_list)
        controls.addWidget(QtWidgets.QLabel('Input ROI:'))
        self.roi_list = QtWidgets.QComboBox()
        self.roi_list.setMinimumWidth(250)
        controls.addWidget(self.roi_list)
        self.gamma = self.add_slider(controls, 'Gamma', 1, 30, 15)
        self.min_size = self.add_slider(controls, 'Min', 0, 1, 0)
        self.max_size = self.add_slider(controls, 'Max', 0, 1, 1)
        self.channel_boxes = {}
        for channel in 'RGB':
            box = QtWidgets.QCheckBox(channel)
            box.setChecked(True)
            box.toggled.connect(self.update_image)
            self.channel_boxes[channel] = box
            controls.addWidget(box)
        controls.addStretch()

        panes = QtWidgets.QHBoxLayout()
        buttons = QtWidgets.QVBoxLayout()
        actions = (('Lasso [l]', 'L', lambda: self.set_tool(Tool.LASSO)),
                   ('Lasso - dilate [i]', 'I', lambda: self.set_tool(Tool.LASSO_DILATE)),
                   ('Draw ROI [g]', 'G', lambda: self.set_tool(Tool.DRAW)),
                   ('Remove [r]', 'R', lambda: self.set_tool(Tool.REMOVE)),
                   ('Dilate [d]', 'D', self.dilate_output),
                   ('Clear output [c]', 'C', self.clear_output),
                   ('Save [e]', 'E', self.save),
                   ('Reset view [f]', 'F', self.fit),
                   ('End tool [Esc]', 'Esc', lambda: self.set_tool(Tool.NONE)))
        top_row = ('Reset view [f]', 'End tool [Esc]')  # right end of the top row, the others in the left column
        for label, key, callback in actions:
            button = QtWidgets.QPushButton(label)
            button.setFocusPolicy(QtCore.Qt.NoFocus)
            button.clicked.connect(callback)
            (controls if label in top_row else buttons).addWidget(button)
            action = QtGui.QAction(label, self)
            action.setShortcut(QtGui.QKeySequence(key))
            action.triggered.connect(callback)
            self.addAction(action)
        layout.addLayout(controls)
        buttons.addWidget(QtWidgets.QLabel('Add mode:'))
        self.mode_group = QtWidgets.QButtonGroup(self)
        for mode in AddModes:
            radio = QtWidgets.QRadioButton(mode.name.title())
            radio.setFocusPolicy(QtCore.Qt.NoFocus)
            radio.setChecked(mode == self.addmode)
            self.mode_group.addButton(radio, mode.value)
            buttons.addWidget(radio)
        self.mode_group.idClicked.connect(lambda i: setattr(self, 'addmode', AddModes(i)))
        buttons.addStretch()
        panes.addLayout(buttons)

        self.input_view = ImageView()
        self.output_view = ImageView()
        for view in (self.input_view, self.output_view):
            view.clicked.connect(self.on_click)
            view.stroke_done.connect(self.on_stroke)
            view.view_changed.connect(self.sync_views)
            panes.addWidget(view, stretch=1)
        layout.addLayout(panes, stretch=1)
        self.setCentralWidget(central)

        self.preview_list.currentIndexChanged.connect(self.update_image)
        self.roi_list.currentIndexChanged.connect(self.load_input_set)
        self.gamma.valueChanged.connect(self.update_image)
        self.min_size.valueChanged.connect(self.update_included)
        self.max_size.valueChanged.connect(self.update_included)

    def add_slider(self, layout, name, vmin, vmax, value):
        label = QtWidgets.QLabel(name)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(vmin, vmax)
        slider.setValue(value)
        slider.setFixedWidth(120)
        slider.setFocusPolicy(QtCore.Qt.NoFocus)
        slider.valueChanged.connect(lambda v: label.setText(f'{name}: {v}'))
        label.setText(f'{name}: {value}')
        label.setMinimumWidth(label.fontMetrics().horizontalAdvance(f"{name}: 000") + 4)
        layout.addWidget(label)
        layout.addWidget(slider)
        return slider

    # ----- files
    def find_files(self):
        '''List the previews and ROI sets of the session; select the ones given, or else the ones used last time.'''
        roi_files, previews = [], []
        for fn in sorted(os.listdir(self.opPath)):
            if fn.startswith(f'{self.prefix}_saved_roi_') and fn.endswith('.npy'):
                if os.path.getsize(self.opPath / fn) > 128:
                    roi_files.append(fn)
            elif fn.startswith(self.prefix) and fn.endswith('.tif'):
                previews.append(fn)
        for combo in (self.preview_list, self.roi_list):
            combo.blockSignals(True)
        for fn in previews:
            self.preview_list.addItem(fn[len(self.prefix) + 1:-4], fn)
        for fn in roi_files:
            self.roi_list.addItem(fn[len(f'{self.prefix}_saved_roi_'):-4], fn)
        for combo, key, default in ((self.preview_list, 'preview', 'avgmax'), (self.roi_list, 'input_tag', None)):
            name = self.start_selection[key] or self.settings.value(key, default)
            index = combo.findText(name) if name else -1
            if index < 0 and combo is self.preview_list:
                index = next((i for i in range(combo.count()) if 'avgmax' in combo.itemText(i)), 0)
            combo.setCurrentIndex(max(index, 0))
            combo.blockSignals(False)
        if not previews:
            self.set_blank_image()
        self.update_image()
        self.load_input_set()
        QtCore.QTimer.singleShot(0, self.fit)  # after the window is shown

    def set_blank_image(self):
        '''No preview tif: a black image of the movie size'''
        image = RoiEditor(self.procpath, self.prefix).img.image
        self.preview_cache[None] = numpy.zeros((*image.info['sz'], 3), dtype='uint8')

    def preview_array(self):
        fn = self.preview_list.currentData()
        if fn not in self.preview_cache:
            image = tifffile.imread(self.opPath / fn).squeeze()
            if image.ndim == 2:
                image = numpy.stack([numpy.zeros_like(image), image, numpy.zeros_like(image)], axis=-1)
            if image.dtype != numpy.uint8:
                image = (255 * image.astype(float) / max(image.max(), 1)).astype('uint8')
            self.preview_cache[fn] = numpy.ascontiguousarray(image[..., :3])
        return self.preview_cache[fn]

    # ----- display
    def update_image(self):
        if not self.preview_cache and self.preview_list.count() == 0:
            return
        image = self.preview_array()
        self.image_shape = image.shape[:2]
        self.settings.setValue('preview', self.preview_list.currentText())
        gamma = 15.0 / (self.gamma.value() + 1)
        lut = (((numpy.arange(256) / 255.0) ** gamma) * 255).astype('uint8')
        image = lut[image]
        for i, channel in enumerate('RGB'):
            if not self.channel_boxes[channel].isChecked():
                image[..., i] = 0
        image = numpy.ascontiguousarray(image)
        h, w = image.shape[:2]
        qimage = QtGui.QImage(image.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        for view in (self.input_view, self.output_view):
            view.set_image(qimage)

    def fit(self):
        self.input_view.fit()

    def sync_views(self, source):
        if self._syncing:
            return
        self._syncing = True
        target = self.output_view if source is self.input_view else self.input_view
        target.setTransform(source.transform())
        target.horizontalScrollBar().setValue(source.horizontalScrollBar().value())
        target.verticalScrollBar().setValue(source.verticalScrollBar().value())
        self._syncing = False

    def load_input_set(self):
        fn = self.roi_list.currentData()
        for item in self.input_items:
            self.input_view.scene().removeItem(item)
        self.input_items, self.input_paths, self.input_polys = [], [], []
        if fn is None:
            self.sizes = numpy.zeros(0)
            self.included = numpy.zeros(0, dtype=bool)
            return
        self.settings.setValue('input_tag', self.roi_list.currentText())
        if fn not in self.loaded_sets:
            self.loaded_sets[fn] = [numpy.array(p, dtype=float) for p in RoiEditor.load_roi(self.opPath / fn)
                                    if len(p) >= 3]
        self.input_polys = self.loaded_sets[fn]
        pen = make_pen(COLORS['included'])
        for poly in self.input_polys:
            qpoly = to_qpolygon(poly)
            item = self.input_view.scene().addPolygon(qpoly, pen)
            self.input_items.append(item)
            path = QtGui.QPainterPath()
            path.addPolygon(qpoly)
            self.input_paths.append(path)
        self.sizes = numpy.array([numpy.sqrt(Polygon(p).area) for p in self.input_polys])
        self.included = numpy.ones(len(self.input_polys), dtype=bool)
        lims = (int(self.sizes.min()), int(self.sizes.max()) + 1) if len(self.sizes) else (0, 1)
        for slider, value in ((self.min_size, lims[0]), (self.max_size, lims[1])):
            slider.blockSignals(True)
            slider.setRange(*lims)
            slider.setValue(value)
            slider.blockSignals(False)
            slider.valueChanged.emit(value)
        self.statusBar().showMessage(f'{len(self.input_polys)} ROIs in {self.roi_list.currentText()}')

    def update_included(self):
        self.included = (self.sizes >= self.min_size.value()) & (self.sizes <= self.max_size.value())
        self.color_input()

    def color_input(self):
        pens = {True: make_pen(COLORS['included']), False: make_pen(COLORS['excluded'])}
        for item, incl in zip(self.input_items, self.included):
            item.setPen(pens[bool(incl)])

    # ----- output
    def add_output(self, poly):
        poly = numpy.asarray(poly, dtype=float)
        item = self.output_view.scene().addPolygon(to_qpolygon(poly), make_pen(COLORS['output']))
        self.output_polys.append(poly)
        self.output_items.append(item)
        self.unsaved = True

    def remove_output(self, indices):
        for i in sorted(indices, reverse=True):
            self.output_view.scene().removeItem(self.output_items.pop(i))
            self.output_polys.pop(i)
        if indices:
            self.unsaved = True

    def add_inputs(self, indices, dilate=0):
        '''Add input ROIs to the output, following the add mode'''
        n0 = len(self.output_polys)
        for i in indices:
            poly = self.input_polys[i]
            if dilate:
                poly = numpy.array(Polygon(poly).buffer(dilate).exterior.coords)
            if self.skip(poly):
                continue
            self.add_output(poly)
        self.statusBar().showMessage(f'{len(self.output_polys) - n0} ROIs added, {len(self.output_polys)} in output')

    def skip(self, poly):
        if self.addmode == AddModes.PRESERVE:
            return False
        if any(numpy.array_equal(poly, p) for p in self.output_polys):
            return True
        cm = QtCore.QPointF(*centroid(poly))
        for item in self.output_items:
            if item.polygon().containsPoint(cm, QtCore.Qt.OddEvenFill):
                return True
        if self.addmode == AddModes.OVERLAP:
            new = Polygon(poly)
            point = Point(centroid(poly))
            for p in self.output_polys:
                if point.distance(Point(centroid(p))) < 50 and new.intersects(Polygon(p)):
                    return True
        return False

    def dilate_output(self):
        polys = [numpy.array(RoiEditor.dilate_polygon(p, buffer=1)) for p in self.output_polys]
        self.remove_output(list(range(len(self.output_polys))))
        for p in polys:
            self.add_output(p)

    def clear_output(self):
        self.remove_output(list(range(len(self.output_polys))))

    # ----- mouse
    def set_tool(self, tool):
        self.tool = tool
        drawing = tool != Tool.NONE
        for view in (self.input_view, self.output_view):
            view.drawing = drawing
            view.setCursor(QtCore.Qt.CrossCursor) if drawing else view.unsetCursor()
            if not drawing:
                view.cancel_stroke()
        messages = {Tool.NONE: '', Tool.LASSO: 'Lasso: draw around input ROIs to add',
                    Tool.LASSO_DILATE: 'Lasso: draw around input ROIs to add, dilated by 3 pixels',
                    Tool.DRAW: 'Draw: each stroke is a new ROI', Tool.REMOVE: 'Remove: draw around output ROIs'}
        self.statusBar().showMessage(messages[tool] + ('. Left click: add a vertex, right click: close the polygon; '
                                                       'right click before drawing or Esc: end the tool.' if drawing else ''))

    def on_click(self, view, pos, button):
        if self.tool != Tool.NONE:
            # a tool is on and no polygon is being drawn: right click ends the tool
            if button == QtCore.Qt.RightButton:
                self.set_tool(Tool.NONE)
            return
        if view is self.input_view:
            if button == QtCore.Qt.LeftButton:
                hit = [i for i, path in enumerate(self.input_paths) if path.contains(pos)]
                if hit:
                    self.included[hit[0]] = True
                    self.input_items[hit[0]].setPen(make_pen(COLORS['included']))
                    self.add_inputs(hit[:1])
            elif button == QtCore.Qt.RightButton:
                self.add_inputs(numpy.flatnonzero(self.included))
        else:
            if button == QtCore.Qt.LeftButton:
                hit = [i for i, item in enumerate(self.output_items)
                       if item.polygon().containsPoint(pos, QtCore.Qt.OddEvenFill)]
                self.remove_output(hit[-1:])
            elif button == QtCore.Qt.RightButton:
                self.set_tool(Tool.DRAW)

    def on_stroke(self, view, points):
        '''Apply the stroke with the current tool. The tool stays on.'''
        lasso = to_qpolygon(points)
        if self.tool in (Tool.LASSO, Tool.LASSO_DILATE):
            inside = [i for i, poly in enumerate(self.input_polys)
                      if self.included[i] and lasso.containsPoint(QtCore.QPointF(*centroid(poly)),
                                                                  QtCore.Qt.OddEvenFill)]
            # only the lassoed ROIs stay selected, until a size slider is moved
            self.included[:] = False
            self.included[inside] = True
            self.color_input()
            self.add_inputs(inside, dilate=3 if self.tool == Tool.LASSO_DILATE else 0)
        elif self.tool == Tool.DRAW:
            h, w = self.image_shape
            self.add_output([(min(max(x, 0), w), min(max(y, 0), h)) for x, y in points])
        elif self.tool == Tool.REMOVE:
            self.remove_output([i for i, poly in enumerate(self.output_polys)
                                if lasso.containsPoint(QtCore.QPointF(*centroid(poly)), QtCore.Qt.OddEvenFill)])

    # ----- save
    def output_name(self):
        '''the preferred tag, or the next free number if that set exists'''
        suffix = '_saved_roi_'
        fn = f'{self.prefix}{suffix}{self.preferred_tag}'
        if not (self.opPath / f'{fn}.npy').exists():
            return fn
        numbers = [0]
        for f in os.listdir(self.opPath):
            if f.startswith(self.prefix + suffix) and f.endswith('.npy'):
                roi_id = f[:-4].split('_')[-1]
                if roi_id.isdigit():
                    numbers.append(int(roi_id))
        n = max(numbers) + 1
        while (self.opPath / f'{self.prefix}{suffix}{n}.npy').exists():
            n += 1
        return f'{self.prefix}{suffix}{n}'

    def save(self):
        if not self.output_polys:
            self.statusBar().showMessage('No output ROIs, nothing saved')
            return None
        fn = self.output_name()
        RoiEditor.save_roi(self.output_polys, self.opPath / fn, self.image_shape)
        msg = f'{len(self.output_polys)} saved in {fn}'
        lprint(self, msg, logger=self.log)
        self.statusBar().showMessage(msg)
        self.unsaved = False
        new_tag = fn[len(f'{self.prefix}_saved_roi_'):]
        if self.roi_list.findText(new_tag) < 0:
            self.roi_list.addItem(new_tag, f'{fn}.npy')
        return fn

    def closeEvent(self, event):
        if self.unsaved:
            answer = QtWidgets.QMessageBox.question(
                self, 'Unsaved ROIs', f'{len(self.output_polys)} output ROIs are not saved. Close anyway?')
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        event.accept()


_open_windows = []  # references keep the windows open when opened from another app


def open_editor(procpath, prefix, preferred_tag='1', input_tag=None, preview=None):
    '''Open the editor as a non-modal window in the running Qt app (arguments: see RoiEditorWindow)'''
    window = RoiEditorWindow(procpath, prefix, preferred_tag, input_tag=input_tag, preview=preview)
    _open_windows.append(window)
    window.destroyed.connect(lambda: _open_windows.remove(window) if window in _open_windows else None)
    window.show()
    return window


def main():
    app = QtWidgets.QApplication(sys.argv)
    set_app_icon(app, 'Roi', '#1E90FF')
    app.setStyle('Fusion')
    open_editor(*sys.argv[1:4])  # path, prefix, tag
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
