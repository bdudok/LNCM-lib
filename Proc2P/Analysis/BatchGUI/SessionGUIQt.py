import copy

from PySide6 import QtGui, QtCore, QtWidgets
import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
import numpy
import sys
import time
from enum import Enum
from pathlib import Path
from Proc2P.Analysis.BatchGUI.Config import GuiConfig, State
from Proc2P.Analysis.BatchGUI.utils import apply_layout, SubplotsCanvas
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Bruker.LoadRegistered import LoadRegistered, Source
from PlotTools.Formatting import strip_ax
import matplotlib.path as mplpath
import cv2
from subprocess import Popen
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
        self.player = None  # made when Play is first clicked, so the movie is not opened on load

        self.make_control_widget()
        self.make_traces_widget()

        self.X = numpy.load(self.session.get_file_with_suffix('_FrameTimes.npy'))[:, 0]
        for start, stop in zip(*self.session.get_mobility_periods()):
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

        if self.session.opto is not None:
            self.TraceCanvas.ax[1].plot(self.X, self.session.opto, color=self.config.OptoColor)
            for e in numpy.where(numpy.diff(self.session.opto > 0.5))[0] + 1:
                for ai in (0, 1):
                    self.TraceCanvas.ax[ai].axvline(self.X[e], color=self.config.OptoColor, alpha=0.4)

        self.TraceCanvas.ax[-1].set_xlabel('Time (s)')
        self.TraceCanvas.fig.tight_layout()
        self.marker_lines = [ca.axvline(0, color='black', lw=1, animated=True, visible=False)
                             for ca in self.TraceCanvas.ax]
        self.marker_background = None
        self.dragging = False
        self.TraceCanvas.mpl_connect('draw_event', self.on_draw)
        self.TraceCanvas.mpl_connect('button_press_event', self.marker_press)
        self.TraceCanvas.mpl_connect('motion_notify_event', self.marker_motion)
        self.TraceCanvas.mpl_connect('button_release_event', self.marker_release)

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
        self.preview_img = self.session.get_preview().squeeze()
        h, w = self.preview_img.shape[:2]

        # Resize while preserving aspect ratio to fit QLabel width
        aspect_ratio = h / w
        self.rescale_factor = min((self.preview_w - 5) / w, (self.preview_h - 5) / h)
        self.preview_w = int(w * self.rescale_factor)
        self.preview_h = int(self.preview_w * aspect_ratio)
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

        #movie
        widget = QtWidgets.QWidget()
        widget.layout = QtWidgets.QVBoxLayout()
        widget.setFixedWidth(self.config.ButtonLabelWidth * 2)
        widget.setFixedHeight(self.config.TextWidgetHeight * 2)
        label = QtWidgets.QLabel('Movie:')
        label.setFixedHeight(self.config.TextWidgetHeight)
        widget.layout.addWidget(label)
        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.play_callback)
        widget.layout.addWidget(self.play_button)
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
        self.cell_selector.setMinimumWidth(self.config.ButtonLabelWidth * 2)
        self.cell_selector.valueChanged.connect(self.cell_update)
        widget.layout.addWidget(self.cell_selector)
        apply_layout(widget)
        column = QtWidgets.QWidget()
        column.layout = QtWidgets.QVBoxLayout()
        column.layout.addWidget(widget)

        #gamma slider, for the preview and the movie
        widget = QtWidgets.QWidget()
        widget.layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel('Gamma:')
        label.setFixedHeight(self.config.TextWidgetHeight)
        label.setFixedWidth(self.config.ButtonLabelWidth)
        widget.layout.addWidget(label)
        self.gamma_label = QtWidgets.QLabel('')
        self.gamma_label.setFixedHeight(self.config.TextWidgetHeight)
        self.gamma_label.setFixedWidth(self.config.ButtonLabelWidth)
        widget.layout.addWidget(self.gamma_label)
        self.gamma = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma.setMinimum(1)
        self.gamma.setMaximum(30)
        self.gamma.setValue(24)  # gamma 0.6
        self.gamma.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.gamma.setMinimumWidth(self.config.ButtonLabelWidth * 2)
        self.gamma.valueChanged.connect(self.gamma_update)
        widget.layout.addWidget(self.gamma)
        apply_layout(widget)
        column.layout.addWidget(widget)
        apply_layout(column)
        self.control_window.layout.addWidget(column)
        self.gamma_update()


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
        self.toolbar = NavigationToolbar2QT(self.TraceCanvas, self)
        self.traces_window.layout.addWidget(self.toolbar)
        self.traces_window.layout.addWidget(self.TraceCanvas)

        apply_layout(self.traces_window)
        self.central_widget.layout.addWidget(self.traces_window)

    def gamma_update(self):
        # same scale as the BatchGUI preview tab
        gamma = 15.0 / (self.gamma.value() + 1)
        self.gamma_label.setText(f'{gamma:.2f}')
        self.lut = numpy.array([((i / 255.0) ** gamma) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
        img = cv2.LUT(self.preview_img, self.lut)
        if len(img.shape) == 2:
            h, w = img.shape
            ch = 1
        else:
            h, w, ch = img.shape
        qimg = QtGui.QImage(img, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.img_resized = qimg.scaled(self.preview_w, self.preview_h,
                                       QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        if self.state == State.LIVE:
            self.paint_preview()
        if self.player is not None:
            self.player.lut = self.lut
            self.player.show_frame(self.player.frame)

    def cell_update(self):
        if self.state != State.LIVE:
            return 0
        self.c = int(self.cell_selector.value())
        self.cell_label.setText(str(self.c))
        self.paint_preview()
        self.trace_update()

    def paint_preview(self):
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

    def play_callback(self):
        if self.player is None:
            movie = LoadRegistered(self.session.procpath, self.session.prefix, Source.S2P)
            if movie.data is None:
                movie.find_alt_path()
            if movie.data is None:
                QtWidgets.QMessageBox.warning(self, 'Movie', f'No registered movie found for {self.session.prefix}')
                return
            self.player = MoviePlayer(movie, self.X, self.lut)
            self.player.frame_changed.connect(self.move_marker)
            self.player.playing_changed.connect(lambda playing: self.play_button.setText(('Play', 'Pause')[playing]))
            dock = QtWidgets.QDockWidget('Movie', self)
            dock.setWidget(self.player)
            dock.visibilityChanged.connect(lambda visible: visible or self.player.pause())
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            self.player.seek(self.TraceCanvas.ax[0].get_xlim()[0])  # start at the left edge of the plot
        self.player.parent().show()
        self.player.toggle_play()

    def on_draw(self, event):
        # the figure was redrawn: store it without the marker, then draw the marker on top
        canvas = self.TraceCanvas
        self.marker_background = canvas.copy_from_bbox(canvas.fig.bbox)
        if self.player is not None:
            self.draw_marker()

    def draw_marker(self):
        t = self.X[self.player.frame]
        for line in self.marker_lines:
            line.set_xdata([t, t])
            line.set_visible(True)
            line.axes.draw_artist(line)

    def move_marker(self):
        canvas = self.TraceCanvas
        if self.marker_background is None:
            return
        canvas.restore_region(self.marker_background)
        self.draw_marker()
        canvas.blit(canvas.fig.bbox)

    def marker_press(self, event):
        # click on the plot to seek, drag to scrub. Not while zooming or panning
        if self.player is None or event.inaxes is None or event.button != 1 or self.toolbar.mode:
            return
        self.dragging = True
        self.player.seek(event.xdata)

    def marker_motion(self, event):
        if self.dragging and event.inaxes is not None:
            self.player.seek(event.xdata)

    def marker_release(self, event):
        self.dragging = False

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
        # transform to image coordinate from widget reference. the preview is aligned to the top left of the label
        pix_x = pos.x() / self.rescale_factor
        pix_y = pos.y() / self.rescale_factor
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


class MoviePlayer(QtWidgets.QWidget):
    '''
    Plays the registered movie at real speed: the frame on display is the one recorded at the elapsed time since
    play started, frames in between are skipped if display is slower than the frame rate.
    '''
    frame_changed = QtCore.Signal()
    playing_changed = QtCore.Signal(bool)
    poll_interval = 15  # ms

    rgb_index = {'Ch1': [0], 'Ch2': [1], None: [0, 1, 2]}  # Ch1 red, Ch2 green, untagged movie gray

    def __init__(self, movie, frame_times, lut):
        super().__init__()
        self.movie = movie
        self.X = frame_times
        self.lut = lut  # gamma, applied to the 8-bit image
        self.channels = [movie.get_channel(chi) for chi in range(movie.n_channels)]
        self.rgb = [self.rgb_index[key] for key in movie.channel_keys]
        self.levels = [self.get_levels(data) for data in self.channels]
        self.frame = 0
        self.t_play_start, self.frame_play_start = None, None

        layout = QtWidgets.QVBoxLayout()
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(256, 256)
        # Ignored: the label follows the layout instead of growing to the size of the frame
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        layout.addWidget(self.image_label, stretch=1)
        self.info_label = QtWidgets.QLabel('')
        layout.addWidget(self.info_label)
        self.setLayout(layout)
        self.image = None

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.poll_interval)
        self.timer.timeout.connect(self.next_frame)

    def get_levels(self, data, n=20):
        # display range of a channel from frames spread over the movie
        sample = data[numpy.linspace(0, len(data) - 1, n).astype('int')]
        return numpy.percentile(sample, (1, 99.9))

    def show_frame(self, frame):
        self.frame = frame
        h, w = self.channels[0].shape[1:]
        img = numpy.zeros((h, w, 3), dtype='uint8')
        for data, levels, rgb in zip(self.channels, self.levels, self.rgb):
            img[..., rgb] = self.scale(data[frame], levels)[..., None]
        img = cv2.LUT(img, self.lut)
        self.image = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self.update_pixmap()
        self.info_label.setText(f'{self.X[frame]:.2f} s, frame {frame}')
        self.frame_changed.emit()

    @staticmethod
    def scale(frame, levels):
        lo, hi = levels
        return (numpy.clip((frame - lo) / (hi - lo), 0, 1) * 255).astype('uint8')

    def update_pixmap(self):
        if self.image is not None:
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(self.image).scaled(
                self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def seek(self, t):
        frame = int(numpy.clip(numpy.searchsorted(self.X, t), 0, len(self.X) - 1))
        if self.timer.isActive():  # keep playing from the new time
            self.t_play_start, self.frame_play_start = time.perf_counter(), frame
        self.show_frame(frame)

    def toggle_play(self):
        if self.timer.isActive():
            self.pause()
            return
        if self.frame >= len(self.X) - 1:
            self.frame = 0
        self.t_play_start, self.frame_play_start = time.perf_counter(), self.frame
        self.timer.start()
        self.playing_changed.emit(True)

    def pause(self):
        self.timer.stop()
        self.playing_changed.emit(False)

    def next_frame(self):
        t = self.X[self.frame_play_start] + time.perf_counter() - self.t_play_start
        frame = numpy.searchsorted(self.X, t, side='right') - 1
        if frame >= len(self.X) - 1:
            self.pause()
            frame = len(self.X) - 1
        if frame != self.frame:
            self.show_frame(int(frame))


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
