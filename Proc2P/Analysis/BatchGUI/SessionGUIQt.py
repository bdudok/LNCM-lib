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
from Proc2P.Analysis.CaTrace import STARTUP_S
from Proc2P.Bruker.LoadRegistered import LoadRegistered, Source
from Proc2P.Bruker.LoadMovie import find_raw_files, get_lazy_raw_movies
from Video.LoadAvi import LoadAvi
from Video.MotionEnergy import bruker_frame_times
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
        widget.setFixedHeight(self.config.TextWidgetHeight * 3)
        label = QtWidgets.QLabel('Movie:')
        label.setFixedHeight(self.config.TextWidgetHeight)
        widget.layout.addWidget(label)
        self.source_selector = QtWidgets.QComboBox(self)  # all sources until the movie is first opened, then the available ones
        for source in Source:
            self.source_selector.addItem(source.name, source)
        self.source_selector.setCurrentText(Source.S2P.name)
        self.source_selector.currentIndexChanged.connect(self.source_update)
        widget.layout.addWidget(self.source_selector)
        boxes = QtWidgets.QHBoxLayout()
        self.scope_box = QtWidgets.QCheckBox('2P')
        self.video_box = QtWidgets.QCheckBox('Video')
        for box in (self.scope_box, self.video_box):
            box.setChecked(True)
            box.stateChanged.connect(self.panels_update)
            boxes.addWidget(box)
        widget.layout.addLayout(boxes)
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

        #gamma sliders of the red (Ch1) and green (Ch2) channel, for the preview and the 2P movie
        self.gamma_sliders, self.gamma_labels = [], []
        for name in ('Red', 'Green'):
            widget = QtWidgets.QWidget()
            widget.layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(f'{name} γ:')
            label.setFixedHeight(self.config.TextWidgetHeight)
            label.setFixedWidth(self.config.ButtonLabelWidth)
            widget.layout.addWidget(label)
            label = QtWidgets.QLabel('')
            label.setFixedHeight(self.config.TextWidgetHeight)
            label.setFixedWidth(self.config.ButtonLabelWidth)
            widget.layout.addWidget(label)
            self.gamma_labels.append(label)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(1)
            slider.setMaximum(30)
            slider.setValue(24)  # gamma 0.6
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            slider.setMinimumWidth(self.config.ButtonLabelWidth * 2)
            slider.valueChanged.connect(self.gamma_update)
            widget.layout.addWidget(slider)
            self.gamma_sliders.append(slider)
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
        # same scale as the BatchGUI preview tab. luts: red, green
        self.luts = []
        for slider, label in zip(self.gamma_sliders, self.gamma_labels):
            gamma = 15.0 / (slider.value() + 1)
            label.setText(f'{gamma:.2f}')
            self.luts.append(numpy.array([((i / 255.0) ** gamma) * 255 for i in numpy.arange(0, 256)]).astype('uint8'))
        # the preview is RGB with Ch1 in red and Ch2 in green
        img = cv2.LUT(self.preview_img, numpy.dstack([*self.luts, self.luts[1]]))
        h, w = img.shape[:2]
        qimg = QtGui.QImage(img, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.img_resized = qimg.scaled(self.preview_w, self.preview_h,
                                       QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        if self.state == State.LIVE:
            self.paint_preview()
        if self.player is not None:
            self.player.set_luts(self.luts)

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
        if param_key != TraceCodes.trace:
            # hide the acquisition start transient, also in traces processed before CaTrace masked it (as CaTrace:
            # smtr 1 s longer, until the smoothing settles)
            Y = Y.astype('float')  # copy, the session data is not changed
            Y[self.X < STARTUP_S + (param_key == TraceCodes.smtr)] = numpy.nan
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

    def open_movies(self):
        # 2P movies of the session by source, as frames by channel. Raw is None until selected: opening reads its files
        self.movies = {}
        for source in (Source.S2P, Source.GEVIReg):
            try:
                movie = LoadRegistered(self.session.procpath, self.session.prefix, source)
            except FileNotFoundError:  # S2P: no suite2p ops
                continue
            if movie.data is None:
                movie.find_alt_path()
            if movie.data is not None:
                self.movies[source] = {key: movie.get_channel(chi) for chi, key in enumerate(movie.channel_keys)}
        if len(find_raw_files(self.session.si)[1]):
            self.movies[Source.Raw] = None
        # camera video and the time of its frames
        self.video = bruker_frame_times(self.session.procpath, self.session.prefix)

    def get_movie(self, source):
        if self.movies[source] is None:
            self.movies[source] = get_lazy_raw_movies(self.session.si)
        return self.movies[source]

    def source_update(self):
        if self.player is not None:
            self.player.scope_label.show_message('Switching source...')  # opening a raw movie takes a few seconds
            self.player.set_scope(self.get_movie(self.source_selector.currentData()))

    def panels_update(self):
        if self.player is not None:
            self.player.show_panels(self.scope_box.isChecked(), self.video_box.isChecked())

    def play_callback(self):
        if self.player is None:
            self.open_movies()
            if not self.movies and self.video is None:
                QtWidgets.QMessageBox.warning(self, 'Movie', f'No movie or video found for {self.session.prefix}')
                return
            # list the available sources; keep the selected one if available, otherwise S2P or the first available
            selected = self.source_selector.currentData()
            if selected not in self.movies:
                selected = Source.S2P if Source.S2P in self.movies else next(iter(self.movies), None)
            self.source_selector.blockSignals(True)
            self.source_selector.clear()
            for source in self.movies:
                self.source_selector.addItem(source.name, source)
            if selected is not None:
                self.source_selector.setCurrentText(selected.name)
            self.source_selector.blockSignals(False)
            self.source_selector.setEnabled(len(self.movies) > 0)
            for box, available in ((self.scope_box, len(self.movies)), (self.video_box, self.video is not None)):
                box.blockSignals(True)
                box.setChecked(box.isChecked() and available)
                box.setEnabled(available)
                box.blockSignals(False)
            self.player = MoviePlayer(self.X, self.luts)
            if self.video is not None:
                cam_file, cam_times = self.video
                self.player.set_video(LoadAvi(str(cam_file)), cam_times)
            self.player.time_changed.connect(self.move_marker)
            self.player.playing_changed.connect(lambda playing: self.play_button.setText(('Play', 'Pause')[playing]))
            dock = QtWidgets.QDockWidget('Movie', self)
            dock.setWidget(self.player)
            dock.visibilityChanged.connect(lambda visible: visible or self.player.pause())
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            QtCore.QTimer.singleShot(0, lambda: self.widen_dock(dock, 1.5))  # after the dock is laid out
            self.panels_update()
            if self.movies:
                self.app.processEvents()  # show the dock, so the message is visible while a raw movie opens
                self.source_update()
            self.player.seek(self.TraceCanvas.ax[0].get_xlim()[0])  # start at the left edge of the plot
        self.player.parent().show()
        self.player.toggle_play()

    def widen_dock(self, dock, factor):
        # the window grows by the extra width, the plots keep theirs
        extra = int((factor - 1) * dock.width())
        self.resize(self.width() + extra, self.height())
        self.resizeDocks([dock], [dock.width() + extra], QtCore.Qt.Horizontal)

    def on_draw(self, event):
        # the figure was redrawn: store it without the marker, then draw the marker on top
        canvas = self.TraceCanvas
        self.marker_background = canvas.copy_from_bbox(canvas.fig.bbox)
        if self.player is not None:
            self.draw_marker()

    def draw_marker(self):
        t = self.player.t
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
    Plays the 2P movie and the camera video at real speed, one above the other. The time on display (s, on the time
    axis of the 2P frames) is the elapsed time since play started; each panel shows its frame recorded at that time,
    frames in between are skipped if display is slower than the frame rate.
    '''
    time_changed = QtCore.Signal()
    playing_changed = QtCore.Signal(bool)
    poll_interval = 15  # ms

    rgb_index = {'Ch1': [0], 'Ch2': [1], None: [0, 1, 2]}  # Ch1 red, Ch2 green, untagged movie gray

    def __init__(self, frame_times, luts):
        super().__init__()
        self.X = frame_times
        self.luts = luts  # gamma of the red and green channel, applied to the 8-bit 2P images
        self.t = self.X[0]
        self.t_play_start, self.t_at_play_start = None, None
        self.channels = None
        self.video = None
        self.scope_frame, self.video_frame = None, None  # frames on display
        self.show_scope, self.show_video = False, False  # set by show_panels

        layout = QtWidgets.QVBoxLayout()
        self.scope_label = FrameLabel()
        layout.addWidget(self.scope_label, stretch=1)
        self.video_label = FrameLabel()
        layout.addWidget(self.video_label, stretch=1)
        self.info_label = QtWidgets.QLabel('')
        layout.addWidget(self.info_label)
        self.setLayout(layout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.poll_interval)
        self.timer.timeout.connect(self.next_frame)

    def set_scope(self, channels):
        '''
        :param channels: dict of the frames of each channel, by channel key (Ch1, Ch2, or None for an untagged movie)
        '''
        self.channels = list(channels.values())
        self.rgb = [self.rgb_index[key] for key in channels]
        self.levels = [self.get_levels(data) for data in self.channels]
        self.redraw()

    def set_video(self, video, frame_times):
        '''
        :param video: LoadAvi of the camera
        :param frame_times: time of each camera frame on the time axis of the 2P frames, nan if not known
        '''
        self.video = video
        self.video_frames = numpy.where(numpy.logical_not(numpy.isnan(frame_times)))[0]
        self.video_times = frame_times[self.video_frames]
        self.redraw()

    def show_panels(self, scope, video):
        self.show_scope = scope
        self.show_video = video and self.video is not None
        self.scope_label.setVisible(self.show_scope)
        self.video_label.setVisible(self.show_video)
        self.redraw()

    def set_luts(self, luts):
        self.luts = luts
        if self.show_scope and self.channels is not None:
            self.show_scope_frame(self.scope_frame)

    def redraw(self):
        self.scope_frame, self.video_frame = None, None
        self.show_time(self.t)

    def get_levels(self, data, n=100):
        # display range of a channel from frames in the middle of the first raw file (already open; bleaching is
        # strongest at the start), or of the registered movie
        center = getattr(data, 'frames_per_file', len(data)) // 2
        start = max(0, min(center - n // 2, len(data) - n))
        sample = data[numpy.arange(start, min(start + n, len(data)))]
        return numpy.percentile(sample, (1, 99.9))

    def show_time(self, t):
        '''Display the frames recorded at time t. Returns whether a frame changed.'''
        self.t = t
        changed = False
        info = [f'{t:.2f} s']
        if self.show_scope and self.channels is not None:
            frame = int(max(numpy.searchsorted(self.X, t, side='right') - 1, 0))
            if frame != self.scope_frame:
                self.show_scope_frame(frame)
                changed = True
            info.append(f'2P frame {frame}')
        if self.show_video:
            i = max(numpy.searchsorted(self.video_times, t, side='right') - 1, 0)
            frame = int(self.video_frames[i])
            if frame != self.video_frame:
                self.video_frame = frame
                self.video_label.set_image(self.video[frame])
                changed = True
            info.append(f'video frame {frame}')
        self.info_label.setText(', '.join(info))
        return changed

    def show_scope_frame(self, frame):
        self.scope_frame = frame
        h, w = self.channels[0].shape[1:]
        img = numpy.zeros((h, w, 3), dtype='uint8')
        for data, levels, rgb in zip(self.channels, self.levels, self.rgb):
            lut = self.luts[0] if rgb == [0] else self.luts[1]  # untagged gray movie: green
            img[..., rgb] = lut[self.scale(data[frame], levels)][..., None]
        self.scope_label.set_image(img)

    @staticmethod
    def scale(frame, levels):
        lo, hi = levels
        return (numpy.clip((frame - lo) / (hi - lo), 0, 1) * 255).astype('uint8')

    def seek(self, t):
        t = min(max(t, self.X[0]), self.X[-1])
        if self.timer.isActive():  # keep playing from the new time
            self.t_play_start, self.t_at_play_start = time.perf_counter(), t
        self.show_time(t)
        self.time_changed.emit()

    def toggle_play(self):
        if self.timer.isActive():
            self.pause()
            return
        if self.t >= self.X[-1]:
            self.t = self.X[0]
        self.t_play_start, self.t_at_play_start = time.perf_counter(), self.t
        self.timer.start()
        self.playing_changed.emit(True)

    def pause(self):
        self.timer.stop()
        self.playing_changed.emit(False)

    def next_frame(self):
        t = self.t_at_play_start + time.perf_counter() - self.t_play_start
        if t >= self.X[-1]:
            t = self.X[-1]
            self.pause()
        if self.show_time(t):  # the marker moves with the frames, no need to redraw it on every tick
            self.time_changed.emit()


class FrameLabel(QtWidgets.QLabel):
    '''Shows an 8-bit image (gray or RGB) scaled to the label, keeping the aspect ratio.'''

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(256, 192)
        # Ignored: the label follows the layout instead of growing to the size of the frame
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image = None

    def set_image(self, img):
        img = numpy.ascontiguousarray(img)
        h, w = img.shape[:2]
        if img.ndim == 2:
            self.image = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        else:
            self.image = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self.update_pixmap()

    def show_message(self, text):
        self.image = None
        self.setText(text)
        self.repaint()  # now: the caller blocks the event loop

    def update_pixmap(self):
        if self.image is not None:
            self.setPixmap(QtGui.QPixmap.fromImage(self.image).scaled(
                self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()


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
