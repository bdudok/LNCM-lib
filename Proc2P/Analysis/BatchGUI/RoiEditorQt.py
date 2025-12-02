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

from Proc2P.utils import logger, lprint
from Proc2P.Analysis.BatchGUI.Config import *
from Proc2P.Analysis.BatchGUI.utils import *
from Proc2P.Analysis.RoiEditor import RoiEditor
import tifffile

'''
Gui for manually drawing ROIs and editing ROI sets.
'''

class GUI_main(QtWidgets.QMainWindow):
    __name__= 'ROIEditor'
    def __init__(self, app, title='Editor', parent_widget=None):
        super().__init__()
        self.setWindowTitle(title)
        self.app = app
        self.config = GuiConfig()
        self.widget = parent_widget
        if parent_widget is None:
            self.widget = QtWidgets.QWidget(self)

        self.state = State.SETUP
        self.make_widgets()

        if parent_widget is None:
            self.setCentralWidget(self.widget)
            self.setGeometry(*self.config.MainWindowGeometry)
            self.show()

    def eventFilter(self, obj, event):
        if self.state == State.LIVE:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if obj is self.input_window:
                    if event.button() == QtCore.Qt.LeftButton:
                        print("Input Widget clicked at:", event.position())
                        # call your callback here
                    return True
                elif obj is self.result_window:
                    if event.button() == QtCore.Qt.LeftButton:
                        print("Result Widget clicked at:", event.position())
                        # call your callback here
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
        self.ROI_windows = QtWidgets.QWidget(self)
        self.ROI_windows.layout=QtWidgets.QHBoxLayout()
        self.widget.layout.addWidget(self.ROI_windows)

        self.input_window = QtWidgets.QLabel(self)
        self.input_window.installEventFilter(self)
        self.ROI_windows.layout.addWidget(self.input_window)
        self.result_window = QtWidgets.QLabel(self)
        self.result_window.installEventFilter(self)
        self.ROI_windows.layout.addWidget(self.result_window)
        apply_layout(self.ROI_windows)

        self.tooltip_widget = QtWidgets.QWidget(self)
        self.widget.layout.addWidget(self.tooltip_widget)
        apply_layout(self.widget)

    def open_session(self, procpath, prefix):
        self.state = State.SETUP
        self.previews = {}
        self.rois = RoiEditor(procpath, prefix)
        roi_tags = self.rois.find_rois()
        for f in roi_tags:
            polys = RoiEditor.load_roi(os.path.join(self.rois.opPath, f))
            if len(polys):
                self.ROI_list.addItem(f, polys) #store polys as userdata in the combobox
        for fn in os.listdir(self.rois.opPath):
            if prefix in fn and fn.endswith('.tif'):
                self.preview_list.addItem(fn[len(prefix)+1:-4], fn) #store preview filenames as userdata
        for i in range(self.preview_list.count()):
            if self.config.ROI_preview_default in self.preview_list.itemText(i):
                self.preview_list.setCurrentIndex(i)

        self.state = State.LIVE
        self.update_preview()

    def update_preview(self):
        if self.state == State.LIVE:
            self.polys = self.ROI_list.currentData()
            preview_fn = self.preview_list.currentData()
            if preview_fn not in self.previews:
                img = tifffile.imread(os.path.join(self.rois.opPath, preview_fn))
                self.previews[preview_fn] = numpy.copy(img)
            img = self.previews[preview_fn].squeeze()
            ch = 3
            if len(img.shape) == ch:
                h, w, ch = img.shape
            else:
                h, w = img.shape
                g = numpy.zeros((h, w, ch), img.dtype)
                g[..., 1] = img
                img = g
            qimg = QtGui.QImage(img, w, h, ch * w, QtGui.QImage.Format_RGB888)



            # Resize while preserving aspect ratio to fit QLabel width
            target_width = self.input_window.width()
            aspect_ratio = h / w
            target_height = int(target_width * aspect_ratio)
            self.rescale_factor = target_width / w

            qimg_resized = qimg.scaled(target_width, target_height,
                                       QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

            self.output_image = copy.copy(qimg_resized)
            self.result_window.setPixmap(QtGui.QPixmap.fromImage(self.output_image))

            # Draw polygons
            painter = QtGui.QPainter(qimg_resized)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 1))
            for poly in self.polys:
                poly_scaled = [(x*self.rescale_factor, y*self.rescale_factor) for (x, y) in poly]
                points = [QtCore.QPointF(x, y) for x, y in poly_scaled]
                painter.drawPolygon(QtGui.QPolygonF(points))
            painter.end()

            self.input_window.setPixmap(QtGui.QPixmap.fromImage(qimg_resized))









def launch_GUI(*args, **kwargs):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app, *args, **kwargs)
    sys.exit(app.exec())


if __name__ == '__main__':
    # launch_GUI()

    wdir = 'D:\Shares\Data\_Processed/2P\JEDI-IPSP/'
    prefix = 'JEDI-Sncg124_2025-05-06_opto_burst_665'
    # wdir = 'D:\Shares\Data\_Processed/2P\CCK/'
    # prefix = 'Sncg146_2025-07-29_optostim_127'

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui = GUI_main(app, )
    gui.open_session(wdir, prefix)

    sys.exit(app.exec())