import copy

from PySide6 import QtGui, QtCore, QtWidgets
import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
import numpy
import os
import json
import sys
from enum import Enum
from Proc2P.utils import logger, lprint
from Proc2P.Analysis.BatchGUI.Config import *
from Proc2P.Analysis.BatchGUI.utils import *
from Proc2P.Analysis.RoiEditor import RoiEditor
import tifffile
from shapely.geometry import Polygon, Point
import matplotlib.path as mplpath
import cv2
from subprocess import Popen

'''
Gui for manually drawing ROIs and editing ROI sets.
'''


class AddModes(Enum):
    CONTAIN = 0
    OVERLAP = 1
    PRESERVE = 2


class FreehandModes(Enum):
    ADD = 0
    NEW = 1

class GUI_widget(QtWidgets.QWidget):
    __name__ = 'ROIEditor'
    #This doesn't actually work - the ide would be to sub this from GUI-main and
    # have it live in a widget of the calling BATCHGUI app. Using as a modal instead

    def __init__(self, parent, parent_widget, config=None):
        super().__init__(parent)
        if config is None:
            config = GuiConfig()
        self.config = config
        self.widget = parent_widget

        self.state = State.SETUP
        self.make_widgets()


class GUI_main(QtWidgets.QMainWindow):
    __name__ = 'ROIEditor'
    def __init__(self, app, title='Editor', config=None):
        super().__init__()
        self.setWindowTitle(title)
        self.app = app
        if config is None:
            config = GuiConfig()
        self.config = config
        self.widget = QtWidgets.QWidget(self)

        self.state = State.SETUP
        self.make_widgets()

        self.setCentralWidget(self.widget)
        self.setGeometry(*self.config.MainWindowGeometry)
        self.show()

    def eventFilter(self, obj, event):
        if self.state == State.LIVE:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                global_pos = event.globalPosition().toPoint()
                child_pos = obj.mapFromGlobal(global_pos)
                if obj is self.input_window:
                    if event.button() == QtCore.Qt.LeftButton:
                        self.pick(child_pos, widget=obj, mode='add')
                    elif event.button() == QtCore.Qt.RightButton:
                        self.add_current()
                    return True
                elif obj is self.result_window:
                    if event.button() == QtCore.Qt.LeftButton:
                        self.pick(child_pos, widget=obj, mode='remove')
                    elif event.button() == QtCore.Qt.RightButton:
                        self.draw_new_ROI(child_pos, widget=obj)
                    return True
        return super().eventFilter(obj, event)

    def make_widgets(self):
        self.widget.layout = QtWidgets.QVBoxLayout()

        self.control_window = QtWidgets.QWidget(self)
        self.control_window.layout = QtWidgets.QHBoxLayout()
        self.control_window.setFixedHeight(self.config.TextWidgetHeight)

        label = QtWidgets.QLabel('Previews:')
        label.setFixedWidth(self.config.ButtonLabelWidth)
        self.control_window.layout.addWidget(label)
        self.preview_list = QtWidgets.QComboBox(self)
        self.preview_list.setFixedWidth(self.config.PrefixFieldWidth)
        self.preview_list.currentIndexChanged.connect(self.update_preview)
        self.control_window.layout.addWidget(self.preview_list)

        label = QtWidgets.QLabel('Input ROI:')
        label.setFixedWidth(self.config.ButtonLabelWidth)
        self.control_window.layout.addWidget(label)
        self.ROI_list = QtWidgets.QComboBox(self)
        self.ROI_list.setFixedWidth(self.config.PrefixFieldWidth)
        self.ROI_list.currentIndexChanged.connect(self.update_preview)
        self.control_window.layout.addWidget(self.ROI_list)
        apply_layout(self.control_window)
        self.widget.layout.addWidget(self.control_window)

        self.slider_window = QtWidgets.QWidget(self)
        self.slider_window.layout = QtWidgets.QHBoxLayout()
        self.slider_window.setFixedHeight(self.config.TextWidgetHeight)

        self.min_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_size_label = QtWidgets.QLabel(label)
        self.max_size_label = QtWidgets.QLabel(label)
        self.gamma_label = QtWidgets.QLabel(label)
        for slider, label, labeltext in zip((self.min_size, self.max_size, self.gamma),
                                            (self.min_size_label, self.max_size_label, self.gamma_label),
                                            ('MinSize', 'MaxSize', 'Gamma')):
            label.setText(labeltext)
            label.setFixedWidth(self.config.ButtonLabelWidth)
            self.slider_window.layout.addWidget(label)
            slider.setMinimum(1)
            slider.setMaximum(30)
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            self.slider_window.layout.addWidget(slider)
        for slider in (self.min_size, self.max_size):
            slider.valueChanged.connect(self.slider_update)
        self.gamma.valueChanged.connect(self.update_preview)
        self.min_size.setValue(1)
        self.max_size.setValue(30)
        self.gamma.setValue(15)
        apply_layout(self.slider_window)
        self.widget.layout.addWidget(self.slider_window)

        self.ROI_windows = QtWidgets.QWidget(self)
        self.ROI_windows.layout = QtWidgets.QHBoxLayout()
        self.widget.layout.addWidget(self.ROI_windows)

        self.input_controls = QtWidgets.QWidget()
        self.input_controls.setFixedWidth(self.config.ButtonLabelWidth * 1.5)
        self.input_controls.layout = QtWidgets.QVBoxLayout()

        # Callbacks to drawing functions
        buttons = {
            'Lasso [l]': self.add_lasso_callback,
            'Lasso - dilate [i]': self.dil_lasso_callback,
            'Draw large ROI [g]': self.draw_lasso_callback,
            'Remove [r]': self.remove_lasso_callback,
            'Dilate [d]': self.dilate_callback,
            'Save [e]': self.save_callback,
            'Clear output [c]': self.clear_callback,
        }

        for key, value in buttons.items():
            b = QtWidgets.QPushButton(key)
            b.clicked.connect(value)
            self.input_controls.layout.addWidget(b)
            action = QtGui.QAction(key[:-4], self)
            action.setShortcut(QtGui.QKeySequence(key[-2]))
            action.triggered.connect(value)
            self.addAction(action)

        # Add mode selector
        self.input_controls.layout.addWidget(QtWidgets.QLabel('Add mode:'))
        self._button_to_mode = {}
        self.modegroup = QtWidgets.QButtonGroup(self)
        self.modegroup.setExclusive(True)
        for mode in AddModes:
            btn = QtWidgets.QRadioButton(mode.name.title(), self)
            self.input_controls.layout.addWidget(btn)
            self.modegroup.addButton(btn)
            self._button_to_mode[btn] = mode

        self.modegroup.buttonClicked.connect(self.on_mode_changed)
        list(self._button_to_mode.keys())[0].setChecked(True)
        self.addmode = list(AddModes)[0]

        apply_layout(self.input_controls)

        self.input_window = QtWidgets.QLabel(self)
        self.input_window.installEventFilter(self)
        self.ROI_windows.layout.addWidget(self.input_controls)
        self.ROI_windows.layout.addWidget(self.input_window)
        self.result_window = QtWidgets.QLabel(self)
        self.result_window.installEventFilter(self)
        self.ROI_windows.layout.addWidget(self.result_window)
        apply_layout(self.ROI_windows)

        self.tooltip_widget = QtWidgets.QWidget(self)
        self.widget.layout.addWidget(self.tooltip_widget)
        apply_layout(self.widget)

    def open_session(self, procpath, prefix, preferred_tag='1'):
        self.state = State.SETUP
        self.previews = {}
        self.rois = RoiEditor(procpath, prefix)
        self.preferred_tag = preferred_tag
        self.saved_rois = []
        self.saved_paths = []
        self.log = logger()
        self.log.set_handle(procpath, prefix)
        roi_tags = self.rois.find_rois()
        for f in roi_tags:
            polys = RoiEditor.load_roi(os.path.join(self.rois.opPath, f))
            if len(polys):
                self.ROI_list.addItem(f, polys)  # store polys as userdata in the combobox
        for fn in os.listdir(self.rois.opPath):
            if prefix in fn and fn.endswith('.tif'):
                self.preview_list.addItem(fn[len(prefix) + 1:-4], fn)  # store preview filenames as userdata
        for i in range(self.preview_list.count()):
            if self.config.ROI_preview_default in self.preview_list.itemText(i):
                self.preview_list.setCurrentIndex(i)

        self.calc_sizes()
        self.state = State.LIVE
        self.update_preview()

    def slider_update(self):
        if self.state == State.LIVE:
            min_size = self.min_size.value()
            self.min_size_label.setText(f'Min: {min_size}')
            max_size = self.max_size.value()
            self.max_size_label.setText(f'Max: {max_size}')
            key = self.ROI_list.currentText()
            for i, s in enumerate(self.sizes[key]):
                self.incl[key][i] = not ((s < min_size) or (s > max_size))
            self.update_preview()

    def update_preview(self):
        if self.state == State.LIVE:
            self.current_key = self.ROI_list.currentText()
            self.polys = self.ROI_list.currentData()
            preview_fn = self.preview_list.currentData()
            if preview_fn not in self.previews:
                img = tifffile.imread(os.path.join(self.rois.opPath, preview_fn))
                self.previews[preview_fn] = numpy.copy(img)
            gamma = 15.0 / (self.gamma.value() + 1)
            self.lut = numpy.array([((i / 255.0) ** gamma) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
            img = cv2.LUT(self.previews[preview_fn].squeeze(), self.lut)
            self.img = img
            h, w, ch = img.shape
            self.image_h, self.image_w = h, w
            qimg = QtGui.QImage(img, w, h, ch * w, QtGui.QImage.Format_RGB888)

            # Resize while preserving aspect ratio to fit QLabel width
            target_width = self.input_window.width()
            aspect_ratio = h / w
            target_height = int(target_width * aspect_ratio)
            self.rescale_factor = target_width / w
            qimg_resized = qimg.scaled(target_width, target_height,
                                       QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.output_image = copy.copy(qimg_resized)
            self.lasso_image = copy.copy(qimg_resized)

            # Draw polygons on input
            painter = self.get_painter(qimg_resized)
            for poly, incl in zip(self.polys, self.incl[self.current_key]):
                color = (QtCore.Qt.gray, QtCore.Qt.yellow)[int(incl)]
                painter.setPen(QtGui.QPen(color, 1))
                self.paint_poly(poly, painter)
            painter.end()
            self.input_window.setPixmap(QtGui.QPixmap.fromImage(qimg_resized))

            # Draw polygons on output
            painter = self.get_painter(self.output_image)
            painter.setPen(QtGui.QPen(QtCore.Qt.magenta, 1))
            for poly in self.saved_rois:
                self.paint_poly(poly, painter)
            painter.end()
            self.result_window.setPixmap(QtGui.QPixmap.fromImage(self.output_image))

    def get_painter(self, image):
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        return painter

    def paint_poly(self, poly, painter):
        poly_scaled = [(x * self.rescale_factor, y * self.rescale_factor) for (x, y) in poly]
        points = [QtCore.QPointF(x, y) for x, y in poly_scaled]
        painter.drawPolygon(QtGui.QPolygonF(points))

    def on_mode_changed(self, button):
        self.addmode = self._button_to_mode[button]

    def add_lasso_callback(self):
        self.lasso('add')

    def dil_lasso_callback(self):
        self.lasso('add-dil')

    def draw_lasso_callback(self):
        self.lasso('draw')

    def remove_lasso_callback(self):
        self.lasso('remove')

    def dilate_callback(self):
        nrs = []
        for roi in self.saved_rois:
            nrs.append(RoiEditor.dilate_polygon(roi, buffer=1))
        self.saved_rois = []
        self.saved_paths = []
        for roi in nrs:
            self.saved_rois.append(roi)
            self.saved_paths.append(mplpath.Path(roi))
        self.update_preview()

    def save_callback(self):
        suffix = '_saved_roi_'
        opPath = self.rois.opPath
        prefix = self.rois.prefix
        if len(self.saved_rois) < 1:
            print('0 rois, roi file not saved for', prefix)
            return -1
        fn = prefix + suffix + self.preferred_tag
        if os.path.exists(opPath + fn + '.npy'):
            exs = [0]
            for f in os.listdir(opPath):
                if prefix in f and suffix in f:
                    if '_' in f[:-5]:
                        # keep autodetected rois from breaking numbering
                        roi_id = f[:-4].split('_')[-1]
                        if roi_id.isdigit():
                            exs.append(int())
            exi = max(exs) + 1
            while os.path.exists(opPath + prefix + suffix + str(exi) + '.npy'):
                exi += 1
            fn = prefix + suffix + str(exi)
        RoiEditor.save_roi(self.saved_rois, opPath + fn, self.rois.img.image.info['sz'])
        msg = f'{len(self.saved_rois)} saved in {fn}'
        lprint(self, msg, logger=self.log)
        QtWidgets.QMessageBox(QtWidgets.QMessageBox.NoIcon,  # no standard icon → no system chime
                              'Saved', msg, QtWidgets.QMessageBox.Ok, self).exec()
        # QtWidgets.QMessageBox.information(self, 'Saved', msg)

    def clear_callback(self):
        self.saved_rois = []
        self.saved_paths = []
        self.update_preview()

    def draw_new_ROI(self, pos, widget):
        self.zoomfactor = self.config.zoomfactor  # display size over actual image size
        # clicked point in image coordinates
        point = [x / self.rescale_factor for x in self.coord_from_click(pos, widget)]
        # desired size of the cropped out image region
        xsize = self.ROI_windows.width() / self.zoomfactor
        ysize = self.ROI_windows.height() / self.zoomfactor
        # corners of the crop (image coordinates
        x0 = int(max(0, min(self.image_w - xsize, point[0] - xsize / 2)))
        y0 = int(max(0, min(self.image_h - ysize, point[1] - ysize / 2)))
        x1 = int(min(x0 + xsize, self.image_w))
        y1 = int(min(y0 + ysize, self.image_h))
        # scaled up image
        img = numpy.ascontiguousarray(self.img[y0:y1, x0:x1])
        h, w, ch = img.shape
        qimg = QtGui.QImage(img, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.zoom_w = (x1 - x0) * self.zoomfactor
        self.zoom_h = (y1 - y0) * self.zoomfactor
        self.zoom_image = qimg.scaled(self.zoom_w, self.zoom_h,
                                      QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        # draw the existing ROIs on the image
        painter = self.get_painter(self.zoom_image)
        painter.setPen(QtGui.QPen(QtCore.Qt.magenta, 1))
        for poly in self.saved_rois:
            scaled_poly = []
            for p in poly:
                x = (p[0] - x0) * self.zoomfactor
                y = (p[1] - y0) * self.zoomfactor
                if (0 < x < self.zoom_w) and (0 < y < self.zoom_h):
                    scaled_poly.append(QtCore.QPointF(x, y))
            if len(scaled_poly) > 3:
                painter.drawPolygon(QtGui.QPolygonF(scaled_poly))
        painter.end()

        self.zoom_corner = [x0, y0]
        self.zoom_point = point
        self.lasso('new')

    def lasso(self, mode):
        if mode == 'new':
            freehand_mode = FreehandModes.NEW
        else:
            freehand_mode = FreehandModes.ADD
        lasso = FreehandPolygonWidget(self, mode=freehand_mode)
        lasso.exec()
        ps = lasso.polys
        if 'add' in mode:
            rois = []
            for p in ps:
                p = mplpath.Path(p)
                for i, roi in enumerate(self.polys):
                    if self.incl[self.current_key][i]:
                        if p.contains_point(mplpath.Path(roi).vertices.mean(axis=0)):
                            if not i in rois:
                                rois.append(i)
            self.incl[self.current_key][:] = False
            self.incl[self.current_key][rois] = True
            self.add_current(dil='dil' in mode)
        elif mode == 'draw':
            for p in ps:
                if len(p) > 3:
                    self.saved_rois.append(p)
                    self.saved_paths.append(mplpath.Path(p))
        elif mode == 'remove':
            rois = []
            for p in ps:
                p = mplpath.Path(p)
                for i, roi in enumerate(self.saved_paths):
                    if p.contains_point(roi.vertices.mean(axis=0)):
                        if i not in rois:
                            rois.append(i)
            r, p = [], []
            for i in range(len(self.saved_rois)):
                if i not in rois:
                    r.append(self.saved_rois[i])
                    p.append(self.saved_paths[i])
            self.saved_rois = r
            self.saved_paths = p
        elif mode == 'new':
            for p in ps:
                if len(p) > 3:
                    self.saved_rois.append(p)
                    self.saved_paths.append(mplpath.Path(p))
        self.update_preview()

    def add_current(self, dil=False):
        for i, incl in enumerate(self.incl[self.current_key]):
            if incl:
                p = self.polys[i]
                if dil:
                    p = Polygon(p).buffer(3).exterior.coords
                if (not p in self.saved_rois) or self.addmode == AddModes.PRESERVE:
                    # check if roi is inside existing roi
                    tp = mplpath.Path(p)
                    cm = tp.vertices.mean(axis=0)
                    point1 = Point(cm)
                    skip = False
                    if self.addmode == AddModes.CONTAIN:
                        for ep in self.saved_paths:
                            if ep.contains_point(cm):
                                skip = True
                                break
                    elif self.addmode == AddModes.OVERLAP:
                        p1 = Polygon(p)
                        for oi, ep in enumerate(self.saved_paths):
                            if ep.contains_point(cm):
                                skip = True
                                break
                            else:
                                if point1.distance(Point(ep.vertices.mean(axis=0))) < 50:
                                    if p1.intersects(Polygon(self.saved_rois[oi])):
                                        skip = True
                                        break
                    if not skip:
                        self.saved_rois.append(p)
                        self.saved_paths.append(tp)
        self.update_preview()

    def coord_from_click(self, pos, widget):
        # transform to image coordinate from widget reference
        offset_x = (widget.width() - self.image_w * self.rescale_factor) / 2
        offset_y = (widget.height() - self.image_h * self.rescale_factor) / 2
        pix_x = (pos.x() - offset_x) / self.rescale_factor
        pix_y = (pos.y() - offset_y) / self.rescale_factor
        return [pix_x, pix_y]

    def pick(self, pos, widget, mode):
        point = self.coord_from_click(pos, widget)
        if mode == 'add':
            for i, poly in enumerate(self.polys):
                if self.paths[self.current_key][i].contains_point(point):
                    if poly not in self.saved_rois:
                        self.saved_rois.append(poly)
                        self.saved_paths.append(mplpath.Path(poly))
                        if not self.incl[self.current_key][i]:
                            self.incl[self.current_key][i] = True
                            self.update_preview()
                        else:
                            self.draw_result(poly)
                        break
        elif mode == 'remove':
            for i in range(len(self.saved_rois)):
                if self.saved_paths[i].contains_point(point):
                    self.saved_rois.pop(i)
                    self.saved_paths.pop(i)
                    self.update_preview()
                    break

    def draw_result(self, poly):
        # draw a poly on the result without updating everything
        painter = self.get_painter(self.output_image)
        painter.setPen(QtGui.QPen(QtCore.Qt.magenta, 1))
        self.paint_poly(poly, painter)
        painter.end()
        self.result_window.setPixmap(QtGui.QPixmap.fromImage(self.output_image))

    def calc_sizes(self):
        self.sizes = {}
        self.paths = {}
        self.incl = {}
        all_sizes = []
        items = [(self.ROI_list.itemText(i), self.ROI_list.itemData(i)) for i in range(self.ROI_list.count())]
        for key, polys in items:
            self.sizes[key] = []
            self.paths[key] = []
            self.incl[key] = numpy.ones(len(polys), dtype='bool')
            for ip, coords in enumerate(polys):
                poly = RoiEditor.trim_coords(numpy.array(coords), self.rois.img.image.info['sz'])
                p = Polygon(poly)
                self.sizes[key].append(numpy.sqrt(p.area))
                mp = mplpath.Path(poly)
                self.paths[key].append(mp)
            all_sizes.extend(self.sizes[key])
        self.size_lims = [int(min(all_sizes)), int(max(all_sizes) + 1)]

        sliders = (self.min_size, self.max_size)
        for slider in sliders:
            slider.setMinimum(self.size_lims[0])
            slider.setMaximum(self.size_lims[1])
        for l, s in zip(self.size_lims, sliders):
            s.setValue(l)


class FreehandPolygonWidget(QtWidgets.QDialog):
    def __init__(self, parent=None, mode=FreehandModes.ADD):
        super().__init__(parent)
        self.setWindowTitle("Lasso")
        self.mode = mode

        if mode == FreehandModes.ADD:
            self.image = parent.lasso_image
            self.image_h = parent.image_h
            self.image_w = parent.image_w
            self.rescale_factor = parent.rescale_factor
        elif mode == FreehandModes.NEW:
            self.image = parent.zoom_image
            self.image_h = parent.zoom_h
            self.image_w = parent.zoom_w
            self.zoom_offset = parent.zoom_corner
            self.zoom_point = parent.zoom_point
            self.rescale_factor = parent.zoomfactor

        self.widget = QtWidgets.QWidget(self)
        self.widget.layout = QtWidgets.QVBoxLayout()
        self.widget.setFixedWidth(self.image.width())
        self.widget.setFixedHeight(self.image.height())
        self.label = QtWidgets.QLabel()
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.widget.layout.addWidget(self.label)
        apply_layout(self.widget)
        self.first_pos = None

        self.setMouseTracking(True)
        self._drawing = False
        self._points = []  # list of QPointF
        self.polys = []
        # if mode == FreehandModes.NEW:
        #     self.move_cursor_to_clicked()

    def mousePressEvent(self, event):
        global_pos = event.globalPosition().toPoint()
        child_pos = self.label.mapFromGlobal(global_pos)
        if event.button() == QtCore.Qt.LeftButton:
            self._drawing = True
            self._points.append(QtCore.QPointF(child_pos))
            if self.first_pos is None:
                self.first_pos = child_pos
            self.update()
        elif event.button() == QtCore.Qt.RightButton:
            self.finishPoly()

    # def move_cursor_to_clicked(self):
    #     p = [round((self.zoom_point[i] - self.zoom_offset[i]) * self.rescale_factor) for i in (0, 1)]
    #     QtGui.QCursor.setPos(self.label.mapToGlobal(QtCore.QPoint(*p)))
    #     #this does not seem to do anything.

    def mouseMoveEvent(self, event):
        if self._drawing and (event.buttons() & QtCore.Qt.LeftButton):
            global_pos = event.globalPosition().toPoint()
            child_pos = self.label.mapFromGlobal(global_pos)
            self._points.append(QtCore.QPointF(child_pos))
            self.update()

    # def mouseReleaseEvent(self, event):
    #     if event.button() == QtCore.Qt.LeftButton and self._drawing:
    #         global_pos = event.globalPosition().toPoint()
    #         child_pos = self.label.mapFromGlobal(global_pos)
    #         self._points.append(QtCore.QPointF(child_pos))

    def finishPoly(self):
        if len(self._points) > 3:
            # close poly and draw it
            self._drawing = True
            self._points.append(QtCore.QPointF(self.first_pos))
            self.paintEvent(None)

            rescaled = []
            # correct with image position in parent widget
            if self.mode == FreehandModes.ADD:
                offset_x = (self.label.width() - self.image_w * self.rescale_factor) / 2
                offset_y = (self.label.height() - self.image_h * self.rescale_factor) / 2
            elif self.mode == FreehandModes.NEW:
                offset_x = (self.label.width() - self.image_w) / 2
                offset_y = (self.label.height() - self.image_h) / 2
            for p in self._points:
                # scale back to image coordinates
                pix_x = max(0, min(self.image_w, (p.x() - offset_x) / self.rescale_factor))
                pix_y = max(0, min(self.image_h, (p.y() - offset_y) / self.rescale_factor))
                rescaled.append([pix_x, pix_y])
            if self.mode == FreehandModes.NEW:
                # factor in that zoomed image is cropped
                rescaled = [[p[0] + self.zoom_offset[0], p[1] + self.zoom_offset[1]] for p in rescaled]
            simplified = [[round(x, 1) for x in rescaled[0]]]
            for p in rescaled[1:]:
                xdiff = abs(p[0] - simplified[-1][0])
                ydiff = abs(p[1] - simplified[-1][1])
                if max(xdiff, ydiff) > 1:
                    simplified.append([round(x, 1) for x in p])
            self.polys.append(simplified)
        self._points = []
        self._drawing = False
        self.first_pos = None

    def paintEvent(self, event):
        if event is not None:
            super().paintEvent(event)
        if not self._points:
            return
        painter = QtGui.QPainter(self.image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtGui.QPen(QtCore.Qt.magenta, 2))

        path = QtGui.QPainterPath()
        path.moveTo(self._points[0])
        for p in self._points[1:]:
            path.lineTo(p)
        painter.drawPath(path)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.image))


def main():
    app = QtWidgets.QApplication()
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app)
    gui_main.open_session(*sys.argv[1:])
    sys.exit(app.exec())

def test_launcher():
    #
    wdir = 'D:\Shares\Data\_Processed/2P\JEDI-IPSP/'
    prefix = 'JEDI-Sncg124_2025-05-06_opto_burst_665'
    # # wdir = 'D:\Shares\Data\_Processed/2P\CCK/'
    # # prefix = 'Sncg146_2025-07-29_optostim_127'
    #
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui = GUI_main(app, )
    gui.open_session(wdir, prefix, preferred_tag='test')

    sys.exit(app.exec())

def launch_in_subprocess(*args, **kwargs):
    Popen([sys.executable, Path(__file__), *args])

if __name__ == '__main__':
    main()

    # test_launcher()
