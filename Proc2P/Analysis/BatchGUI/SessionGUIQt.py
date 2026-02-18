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
from Proc2P.Analysis.BatchGUI.Config import *
from Proc2P.Analysis.BatchGUI.utils import *
from Proc2P.Analysis.RoiEditor import RoiEditor
from Proc2P.Analysis.ImagingSession import ImagingSession
from PlotTools.Formatting import strip_ax
import tifffile
from shapely.geometry import Polygon, Point
import matplotlib.path as mplpath
import cv2
from subprocess import Popen
from enum import Enum
from LFP.EphysFunctions import butter_bandpass_filter

'''
Gui for inspecting traces from 2P imaging.
'''

class TraceCodes(Enum):
    trace = 'Raw'
    rel = 'DF/F'
    ntr = 'DF/F(z)'
    smtr = 'EWMA(z)'
    nnd = 'NND'
    vm = 'vm'

class GUI_main(QtWidgets.QMainWindow):
    __name__ = 'SessionViewer'
    def __init__(self, app, title='Viewer', config=None):
        super().__init__()
        self.setWindowTitle(title)
        self.app = app
        if config is None:
            config = GuiConfig()
        self.config = config
        self.central_widget = QtWidgets.QWidget(self)
        self.central_widget.layout = QtWidgets.QVBoxLayout()

        self.state = State.SETUP
        # we don't do anything until a session is opened. open_session will call widget makers.

    def open_session(self, procpath, prefix, tag='1'):
        self.state = State.SETUP
        self.session = ImagingSession(procpath, prefix, tag, ch='Both')

        self.make_control_widget()
        self.make_traces_widget()

        self.X = numpy.load(self.session.get_file_with_suffix('_FrameTimes.npy'))[:, 0]
        for start, stop in zip(*self.session.startstop()):
            for ca in self.TraceCanvas.ax[:2]:
                ca.axvspan(self.X[start], self.X[stop], facecolor=self.config.RunColor, edgecolor=None, alpha=0.4)
        ca = self.TraceCanvas.ax[1]
        if self.session.pos.laps[-1] > 0:
            ax_right = ca.twinx()
            ax_right.plot(self.X, self.session.pos.relpos, color='black')
            ax_right.set_ylabel('Position')
        ca.plot(self.X, self.session.pos.smspd, color=self.config.RunColor)
        ca.set_ylim(-2, 20)
        ca.set_ylabel('Speed (cm/s)')
        self.lines = []

        if self.session.has_ephys:
            ca = self.TraceCanvas.ax[2]
            self.session.map_seizuretimes()
            sztimes = self.session.sztimes[self.session.sztime_channels.index(1)]
            for start, stop in zip(*sztimes):
                for ai in (0, 2):
                    self.TraceCanvas.ax[ai].axvspan(self.X[start], self.X[stop], facecolor=self.config.SzColor,
                                                    edgecolor=None, alpha=0.4)
            Y = butter_bandpass_filter(self.session.ephys.trace, 1, 500, self.session.si['fs'])
            ca.plot(numpy.linspace(0, self.X[-1], len(Y)), Y, color='black')
            ca.set_ylabel('LFP (mV)')
        self.TraceCanvas.ax[-1].set_xlabel('Time (s)')
        self.TraceCanvas.fig.tight_layout()

        apply_layout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        self.setGeometry(*self.config.MainWindowGeometry)
        self.show()
        self.state = State.LIVE
        self.cell_update()

    def eventFilter(self, obj, event):
        if self.state == State.LIVE:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                global_pos = event.globalPosition().toPoint()
                child_pos = obj.mapFromGlobal(global_pos)
                if obj is self.preview_label:
                    if event.button() == QtCore.Qt.LeftButton:
                        self.pick(child_pos, widget=obj)
                    return True
        return super().eventFilter(obj, event)

    def make_control_widget(self):
        #top row for ROI preview and controls
        self.control_window = QtWidgets.QWidget(self)
        self.control_window.layout = QtWidgets.QHBoxLayout()
        self.control_window.setFixedHeight(self.config.SessionControlsHeight)

        # preview of full FOV
        self.preview_label = QtWidgets.QLabel(self)
        # self.zoom_label = QtWidgets.QLabel(self)
        for label in (self.preview_label, ):#self.zoom_label):
            label.setFrameStyle(QtWidgets.QFrame.NoFrame)
            label.setMargin(0)
            label.setContentsMargins(0, 0, 0, 0)
            label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.preview_w = self.config.SessionPreviewWidth
        self.preview_h = self.config.SessionControlsHeight - 5
        img = self.session.get_preview()
        lut = numpy.array([((i / 255.0) ** 0.6) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
        img = cv2.LUT(img.squeeze(), lut)
        if len(img.shape) == 2:
            h, w = img.shape
            ch = 1
        else:
            h, w, ch = img.shape
        qimg = QtGui.QImage(img, w, h, ch * w, QtGui.QImage.Format_RGB888)

        # Resize while preserving aspect ratio to fit QLabel width
        aspect_ratio = h / w
        self.rescale_factor = min((self.preview_w - 5) / w, (self.preview_h - 5) / h)
        self.preview_w = int(w * self.rescale_factor)
        self.preview_h = int(self.preview_w * aspect_ratio)

        self.img_resized = qimg.scaled(self.preview_w, self.preview_h,
                                   QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(self.img_resized))
        self.preview_label.installEventFilter(self)
        self.control_window.layout.addWidget(self.preview_label)
        # self.control_window.layout.addWidget(self.zoom_label)

        #channel selector
        if self.session.dualch:
            widget = QtWidgets.QWidget()
            ch_selector = QtWidgets.QFormLayout()
            ch_selector.addRow(QtWidgets.QLabel('Channel'))
            self.ch_boxes = []
            for chn in ('Green', 'Red'):
                w = QtWidgets.QCheckBox()
                w.setChecked(True)
                w.stateChanged.connect(self.cell_update)
                ch_selector.addRow(chn, w)
                self.ch_boxes.append(w)
            widget.layout = ch_selector
            apply_layout(widget)
            self.control_window.layout.addWidget(widget)

        #trace selector
        widget = QtWidgets.QWidget()
        widget.layout = QtWidgets.QVBoxLayout()
        widget.setFixedWidth(self.config.ButtonLabelWidth * 2)
        widget.setFixedHeight(self.config.TextWidgetHeight * 2)
        label = QtWidgets.QLabel('Trace:')
        label.setFixedHeight(self.config.TextWidgetHeight)
        widget.layout.addWidget(label)
        self.trace_selector = QtWidgets.QComboBox(self)
        for m in TraceCodes:
            self.trace_selector.addItem(m.name, m)
        self.trace_selector.setCurrentText(TraceCodes.rel.name)
        self.trace_selector.currentIndexChanged.connect(self.trace_update)
        widget.layout.addWidget(self.trace_selector)

        button = QtWidgets.QPushButton('Rescale Y')
        button.clicked.connect(self.rescale_callback)
        widget.layout.addWidget(button)

        apply_layout(widget)
        self.control_window.layout.addWidget(widget)

        #cell slider
        widget = QtWidgets.QWidget()
        widget.layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel('Cell:')
        label.setFixedHeight(self.config.TextWidgetHeight)
        label.setFixedWidth(self.config.ButtonLabelWidth)
        widget.layout.addWidget(label)
        self.cell_label = QtWidgets.QLabel('0')
        self.cell_label.setFixedHeight(self.config.TextWidgetHeight)
        self.cell_label.setFixedWidth(self.config.ButtonLabelWidth)
        widget.layout.addWidget(self.cell_label)
        self.cell_selector = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cell_selector.setMinimum(0)
        self.cell_selector.setMaximum(self.session.ca.cells-1)
        self.cell_selector.setValue(0)
        self.cell_selector.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.cell_selector.valueChanged.connect(self.cell_update)
        widget.layout.addWidget(self.cell_selector)
        apply_layout(widget)
        self.control_window.layout.addWidget(widget)


        apply_layout(self.control_window)
        self.central_widget.layout.addWidget(self.control_window)
        self.central_widget.layout.addWidget(QtWidgets.QLabel("<hr>"))

    def make_traces_widget(self):
        #traces with shared X
        self.traces_window = QtWidgets.QWidget(self)
        self.traces_window.layout = QtWidgets.QVBoxLayout()

        # real time plot area
        figsize = (self.config.plot_canvas_size[0]*1.25, self.config.plot_canvas_size[1]*3)
        self.TraceCanvas = SubplotsCanvas(nrows=2+self.session.has_ephys, sharex=True, figsize=figsize)
        for ca in self.TraceCanvas.ax:
            strip_ax(ca, False)
        toolbar = NavigationToolbar2QT(self.TraceCanvas, self)
        self.traces_window.layout.addWidget(toolbar)
        self.traces_window.layout.addWidget(self.TraceCanvas)

        apply_layout(self.traces_window)
        self.central_widget.layout.addWidget(self.traces_window)

    def cell_update(self):
        if self.state != State.LIVE:
            return 0
        self.c = int(self.cell_selector.value())
        self.cell_label.setText(str(self.c))
        self.polys = self.session.rois.polys
        self.paths = [mplpath.Path(poly) for poly in self.polys]

        img = copy.copy(self.img_resized)
        painter = self.get_painter(img)
        for c in range(self.session.ca.cells):
            poly = self.session.rois.polys[c]
            color = (QtCore.Qt.gray, QtCore.Qt.yellow)[c==self.c]
            painter.setPen(QtGui.QPen(color, 1))
            self.paint_poly(poly, painter)
        painter.end()
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(img))

        self.trace_update()

    def trace_update(self):
        if self.state != State.LIVE:
            return 0
        param_key = self.trace_selector.currentData()
        Y = self.session.getparam(param_key.name)[self.c]
        ca = self.TraceCanvas.ax[0]
        for line in self.lines:
            line.remove()
        self.TraceCanvas.fig.canvas.draw_idle()
        self.lines = []
        ca.set_ylabel(param_key.value)
        if not self.session.dualch:
            lines = self.TraceCanvas.ax[0].plot(self.X, Y, color=self.config.GreenColor)
            self.lines.extend(lines)
        else:
            for chi, color in zip((0, 1), (self.config.GreenColor, self.config.RedColor)):
                if self.ch_boxes[chi].isChecked():
                    lines = ca.plot(self.X, Y[:, chi], color=color)
                    self.lines.extend(lines)

    def rescale_callback(self):
        ca = self.TraceCanvas.ax[0]
        ca.relim()
        ca.autoscale_view(scaley=True, scalex=False)
        ca.figure.canvas.draw_idle()

    def get_painter(self, image):
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        return painter

    def paint_poly(self, poly, painter):#, widget):
        offset_x = 0#(widget.width() - self.preview_w) / 2
        offset_y = 0#(widget.height() - self.preview_h) / 2
        poly_scaled = [(x * self.rescale_factor - offset_x, y * self.rescale_factor - offset_y) for (x, y) in poly]
        points = [QtCore.QPointF(x, y) for x, y in poly_scaled]
        painter.drawPolygon(QtGui.QPolygonF(points))

    def coord_from_click(self, pos, widget):
        # transform to image coordinate from widget reference
        offset_x = (widget.width() - self.preview_w) / 2
        offset_y = (widget.height() - self.preview_h) / 2
        pix_x = (pos.x() - offset_x) / self.rescale_factor
        pix_y = (pos.y() - offset_y) / self.rescale_factor
        return [pix_x, pix_y]

    def pick(self, pos, widget):
        point = self.coord_from_click(pos, widget)
        dists = []
        for c, poly in enumerate(self.polys):
            if self.paths[c].contains_point(point):
                self.cell_selector.setValue(c)
                return 0
            dists.append(numpy.linalg.norm(poly.mean(axis=0) - point))
        self.cell_selector.setValue(numpy.argmin(dists))


def main():
    app = QtWidgets.QApplication()
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app)
    gui_main.open_session(*sys.argv[1:]) #pass path, prefix, tag
    sys.exit(app.exec())

def test_launcher():
    #
    wdir = r'D:\Shares\Data\_Processed\2P\eCB-GRAB/'
    prefix = 'BL6-58_2026-01-08_KA-LFP_214'
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui = GUI_main(app, title=prefix)
    gui.open_session(wdir, prefix, tag='area')

    sys.exit(app.exec())

def launch_in_subprocess(*args, **kwargs):
    #can be called with args specifying the session to launch a standalone window
    Popen([sys.executable, Path(__file__), *args])

if __name__ == '__main__':
    main()
    # the main block should only call main() for this to work as a subprocess - for testing, use test_launcher instead
    # test_launcher()
