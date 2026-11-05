"""Review post-surgery recovery videos (MonitorCam) and mark seizures on the Racine scale.

The window shows the video, the motion energy trace in the analysis ROI, and the marked seizures. A video without a
saved ROI asks for one; saving the ROI queues the motion energy computation in the background
(Video/RecoveryMotion.py), and the trace appears when it is done. Marks are saved on every change.
Launch it with GUIs/launch_Review_Recovery.py.
"""
import datetime
import os
import sys
from pathlib import Path

import cv2
import numpy
import pandas
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from Video import RecoveryMotion

RAW_DIR = 'D:/Shares/Data/_RawData/MonitorCam'
RACINE = {  # stage saved in the Racine column: (key, description). Stages 1-5 are from Racine (1972)
    '1': ('1', 'Mouth and facial movements'),
    '2': ('2', 'Head nodding'),
    '3': ('3', 'Forelimb clonus'),
    '4': ('4', 'Rearing'),
    '5': ('5', 'Rearing and falling'),
    '5+': ('6', 'More severe than 5'),
}
STAGE_COLORS = {'1': '#fdd835', '2': '#fb8c00', '3': '#e53935', '4': '#8e24aa', '5': '#1e3a8a', '5+': '#000000'}
SPEEDS = (1, 2, 4, 8, 16)
STEP_S = 5  # arrow keys
STATUS_POLL_MS = 5000
EVENT_COLUMNS = ['Start(s)', 'End(s)', 'Racine', 'Start time']


class VideoLabel(QtWidgets.QLabel):
    """Shows a color frame scaled to the label with the ROI on it; draws a new ROI with the mouse in draw mode."""
    roi_drawn = QtCore.Signal(list)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setStyleSheet('background-color: black')
        self.frame = None  # BGR
        self.roi = None
        self.drawing = False
        self.drag_start = None
        self.rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

    def set_frame(self, frame):
        self.frame = frame
        self.update_pixmap()

    def update_pixmap(self):
        if self.frame is None:
            self.clear()
            return
        img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        if self.roi is not None:
            x0, x1, y0, y1 = self.roi
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), max(2, img.shape[1] // 400))
        h, w = img.shape[:2]
        image = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.size(), Qt.KeepAspectRatio,
                                                             Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def to_frame(self, pos):
        """Label position to frame pixel, clipped to the frame"""
        h, w = self.frame.shape[:2]
        scale = min(self.width() / w, self.height() / h)
        x0, y0 = (self.width() - w * scale) / 2, (self.height() - h * scale) / 2
        return (int(min(max((pos.x() - x0) / scale, 0), w)), int(min(max((pos.y() - y0) / scale, 0), h)))

    def mousePressEvent(self, event):
        if self.drawing and self.frame is not None:
            self.drag_start = event.position().toPoint()
            self.rubber_band.setGeometry(QtCore.QRect(self.drag_start, QtCore.QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            self.rubber_band.setGeometry(QtCore.QRect(self.drag_start, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event):
        if self.drag_start is None:
            return
        (xa, ya), (xb, yb) = self.to_frame(self.drag_start), self.to_frame(event.position().toPoint())
        self.drag_start = None
        self.rubber_band.hide()
        if abs(xb - xa) > 10 and abs(yb - ya) > 10:
            self.roi_drawn.emit([min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, raw_dir=RAW_DIR):
        super().__init__()
        self.raw_dir = raw_dir
        self.setWindowTitle('Review recovery video')
        self.resize(1600, 1000)
        self.video = None
        self.cap = None
        self.cap_next = None  # frame index that cap.read() returns next
        self.fps = 30.0
        self.frame = 0
        self.n_frames = None
        self.seekable = True
        self.t0 = None  # start time of the recording
        self.trace = None  # (times, motion energy)
        self.events = []  # dicts with EVENT_COLUMNS keys
        self.speed = 1

        # top row: file and processing status
        open_button = QtWidgets.QPushButton('Open video')
        open_button.clicked.connect(self.open_callback)
        self.file_label = QtWidgets.QLabel('No video')
        self.roi_button = QtWidgets.QPushButton('Draw ROI')
        self.roi_button.clicked.connect(self.start_roi)
        self.status_label = QtWidgets.QLabel()
        top = QtWidgets.QHBoxLayout()
        top.addWidget(open_button)
        top.addWidget(self.file_label, stretch=1)
        top.addWidget(self.status_label)
        top.addWidget(self.roi_button)

        # video and controls
        self.video_label = VideoLabel()
        self.video_label.roi_drawn.connect(self.roi_drawn)
        self.play_button = QtWidgets.QPushButton('Play [space]')
        self.play_button.clicked.connect(self.play_callback)
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems([f'{s}x' for s in SPEEDS])
        self.speed_box.currentIndexChanged.connect(lambda i: setattr(self, 'speed', SPEEDS[i]))
        self.time_label = QtWidgets.QLabel()
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(QtWidgets.QLabel('Speed:'))
        controls.addWidget(self.speed_box)
        controls.addWidget(QtWidgets.QLabel(f'[\u2190/\u2192] {STEP_S} s'))
        controls.addWidget(self.time_label, stretch=1)
        video_box = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(video_box)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.video_label, stretch=1)
        vbox.addLayout(controls)

        # motion energy plot
        self.fig = Figure(figsize=(12, 2.5), layout='constrained')
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.plot_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.marker = None
        self.background = None
        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.plot_click)
        plot_box = QtWidgets.QWidget()
        pbox = QtWidgets.QVBoxLayout(plot_box)
        pbox.setContentsMargins(0, 0, 0, 0)
        pbox.addWidget(self.plot_toolbar)
        pbox.addWidget(self.canvas)
        splitter = QtWidgets.QSplitter(Qt.Vertical)
        splitter.addWidget(video_box)
        splitter.addWidget(plot_box)
        splitter.setSizes([700, 300])

        # seizure marks
        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel('Mark seizure at the current time:'))
        for stage, (key, text) in RACINE.items():
            button = QtWidgets.QPushButton(f'[{key}] {stage}: {text}' if stage != key else f'[{key}] {text}')
            button.setStyleSheet('text-align: left; padding: 4px')
            button.clicked.connect(lambda _=False, s=stage: self.mark_callback(s))
            side.addWidget(button)
        end_button = QtWidgets.QPushButton('[e] End of selected seizure')
        end_button.clicked.connect(self.end_callback)
        side.addWidget(end_button)
        self.event_list = QtWidgets.QListWidget()
        self.event_list.itemClicked.connect(self.event_clicked)
        side.addWidget(self.event_list, stretch=1)
        delete_button = QtWidgets.QPushButton('Delete selected')
        delete_button.clicked.connect(self.delete_callback)
        side.addWidget(delete_button)
        side_box = QtWidgets.QWidget()
        side_box.setLayout(side)
        side_box.setFixedWidth(330)

        main = QtWidgets.QHBoxLayout()
        main.addWidget(splitter, stretch=1)
        main.addWidget(side_box)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(main, stretch=1)
        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        shortcuts = [('Space', self.play_callback), ('Left', lambda: self.step(-STEP_S)),
                     ('Right', lambda: self.step(STEP_S)), ('e', self.end_callback)]
        shortcuts += [(key, lambda s=s: self.mark_callback(s)) for s, (key, _) in RACINE.items()]
        for key, callback in shortcuts:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(callback)

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self.play_tick)
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(STATUS_POLL_MS)
        self.draw_plot()

    # --- files ---
    def open_callback(self):
        start = self.raw_dir if os.path.isdir(self.raw_dir) else ''
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open video', start, 'Videos (*.mkv *.mp4 *.avi)')
        if fn:
            self.open_video(fn)

    def open_video(self, fn):
        self.pause()
        if self.cap is not None:
            self.cap.release()
        self.video = Path(fn)
        self.cap = cv2.VideoCapture(str(fn))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # a file the recorder did not close has no duration or index, and OpenCV can't seek in it
        self.seekable = n > 0
        self.n_frames = int(n) if n > 0 else None
        self.cap_next = 0
        self.t0 = RecoveryMotion.start_time(fn)
        self.file_label.setText(f'{self.video.name}  |  start: {self.t0.isoformat(sep=" ", timespec="seconds") if self.t0 else "unknown"}')
        self.video_label.roi = RecoveryMotion.load_roi(fn)
        self.load_events()
        self.trace = RecoveryMotion.load_trace(fn)
        self.show_frame(0)
        self.draw_plot()
        if not self.seekable:
            QtWidgets.QMessageBox.warning(
                self, 'Video not closed properly',
                'This file has no index (the recorder did not finish it), so jumping to a time does not work. '
                'Playback from the start works.\nTo fix the file, remux it with ffmpeg: '
                'ffmpeg -i <file> -c copy <new file>.mkv')
        if self.video_label.roi is None:
            QtWidgets.QMessageBox.information(
                self, 'Draw ROI', 'Draw the analysis ROI for the motion energy on the video: drag a rectangle '
                'around the area where the mouse can move.')
            self.start_roi()
        else:
            RecoveryMotion.queue_motion(fn)  # if the trace is missing (no-op otherwise)
        self.update_status()

    def load_events(self):
        fn = RecoveryMotion.events_fn(self.video)
        self.events = []
        if fn.exists():
            df = pandas.read_excel(fn)
            self.events = [{k: (None if pandas.isna(row[k]) else row[k]) for k in EVENT_COLUMNS}
                           for _, row in df.iterrows()]
            for e in self.events:
                e['Racine'] = str(e['Racine'])
        self.update_event_list()

    def save_events(self):
        RecoveryMotion.processed_dir(self.video).mkdir(parents=True, exist_ok=True)
        self.events.sort(key=lambda e: e['Start(s)'])
        df = pandas.DataFrame(self.events, columns=EVENT_COLUMNS)
        df['Racine'] = [int(s) if s.isdigit() else s for s in df['Racine']]  # numbers stay numbers in Excel
        df.to_excel(RecoveryMotion.events_fn(self.video), index=False)

    # --- ROI and processing ---
    def start_roi(self):
        if self.video is None:
            return
        if RecoveryMotion.is_queued(self.video):
            QtWidgets.QMessageBox.information(self, 'Draw ROI', 'The motion energy is being computed. Draw the ROI '
                                              'again when it is done.')
            return
        self.pause()
        self.video_label.drawing = True
        self.video_label.setCursor(Qt.CrossCursor)
        self.status_label.setText('<b>Drag a rectangle on the video</b>')

    def roi_drawn(self, roi):
        self.video_label.drawing = False
        self.video_label.unsetCursor()
        old = self.video_label.roi
        self.video_label.roi = roi
        self.video_label.update_pixmap()
        answer = QtWidgets.QMessageBox.question(self, 'Save ROI', 'Use this ROI and compute the motion energy?')
        if answer != QtWidgets.QMessageBox.Yes:
            self.video_label.roi = old
            self.video_label.update_pixmap()
            self.update_status()
            return
        RecoveryMotion.save_roi(self.video, roi)
        self.trace = None
        self.draw_plot()
        RecoveryMotion.queue_motion(self.video)
        self.update_status()

    def update_status(self):
        if self.video is None or self.video_label.drawing:
            return
        if self.trace is None:
            self.trace = RecoveryMotion.load_trace(self.video)
            if self.trace is not None:
                self.draw_plot()
        if self.trace is not None:
            text = 'Motion energy: done'
        elif RecoveryMotion.error_fn(self.video).exists():
            text = f'<span style="color:#c62828">Motion energy failed: see {RecoveryMotion.error_fn(self.video)}</span>'
        elif RecoveryMotion.is_queued(self.video):
            text = 'Motion energy: computing in the background (queued)'
        elif self.video_label.roi is None:
            text = 'No ROI: click Draw ROI'
        else:
            text = 'Motion energy: not computed'
        self.status_label.setText(text)

    # --- player ---
    def frame_time(self, frame):
        """Time of a frame from the start of the video (s)"""
        if self.trace is not None and len(self.trace[0]):
            return float(self.trace[0][min(frame, len(self.trace[0]) - 1)])
        return frame / self.fps

    def time_frame(self, t):
        if self.trace is not None and len(self.trace[0]):
            return int(min(numpy.searchsorted(self.trace[0], t), len(self.trace[0]) - 1))
        return max(0, int(round(t * self.fps)))

    def last_frame(self):
        if self.trace is not None:
            return len(self.trace[0]) - 1
        return self.n_frames - 1 if self.n_frames else None

    def show_frame(self, frame):
        """Display a frame: decode on for a few frames ahead, seek otherwise"""
        last = self.last_frame()
        frame = max(0, frame if last is None else min(frame, last))
        ahead = frame - self.cap_next
        if not 0 <= ahead < 60:
            if not self.seekable:
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        else:
            for _ in range(ahead):
                self.cap.grab()
        ok, img = self.cap.read()
        if not ok:  # past the end of the file
            self.pause()
            self.cap_next = -1  # decoder position unknown: seek next time
            return
        self.cap_next = frame + 1
        self.frame = frame
        self.video_label.set_frame(img)
        t = self.frame_time(frame)
        clock = ''
        if self.t0 is not None:
            clock = f'  |  {(self.t0 + datetime.timedelta(seconds=t)).strftime("%H:%M:%S")}'
        self.time_label.setText(f'{format_s(t)}{clock}  |  frame {frame}')
        self.move_marker()

    def play_callback(self):
        if self.cap is None:
            return
        if self.play_timer.isActive():
            self.pause()
        else:
            self.play_timer.start(int(1000 / self.fps))
            self.play_button.setText('Pause [space]')

    def pause(self):
        self.play_timer.stop()
        self.play_button.setText('Play [space]')

    def play_tick(self):
        self.show_frame(self.frame + self.speed)

    def step(self, seconds):
        if self.cap is not None:
            self.show_frame(self.time_frame(self.frame_time(self.frame) + seconds))

    # --- plot ---
    def draw_plot(self):
        ax = self.ax
        ax.clear()
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Motion energy')
        if self.trace is not None:
            t, me = self.trace
            ax.plot(t / 60, me, color='#444444', linewidth=0.6)
            ax.set_xlim(0, t[-1] / 60 if len(t) else 1)
        else:
            ax.text(0.5, 0.5, 'No motion energy yet', transform=ax.transAxes, ha='center', va='center',
                    color='#888888')
        for e in self.events:
            color = STAGE_COLORS[str(e['Racine'])]
            if e['End(s)'] is not None:
                ax.axvspan(e['Start(s)'] / 60, e['End(s)'] / 60, color=color, alpha=0.3, linewidth=0)
            ax.axvline(e['Start(s)'] / 60, color=color, linewidth=1.5)
            ax.text(e['Start(s)'] / 60, 1, str(e['Racine']), transform=ax.get_xaxis_transform(), color=color,
                    va='bottom', ha='center', fontweight='bold')
        self.marker = ax.axvline(self.frame_time(self.frame) / 60, color='#c62828', linewidth=1, animated=True)
        self.canvas.draw_idle()

    def on_draw(self, event):
        # store the plot without the marker on every draw, then draw the marker on top
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.marker)

    def move_marker(self):
        if self.background is None or self.marker is None:
            return
        self.marker.set_xdata([self.frame_time(self.frame) / 60] * 2)
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.marker)
        self.canvas.blit(self.ax.bbox)

    def plot_click(self, event):
        if event.inaxes is not self.ax or self.plot_toolbar.mode or self.cap is None or event.button != 1:
            return
        if not self.seekable:
            return
        self.show_frame(self.time_frame(event.xdata * 60))

    # --- seizure marks ---
    def mark_callback(self, stage):
        if self.video is None:
            return
        t = self.frame_time(self.frame)
        clock = (self.t0 + datetime.timedelta(seconds=t)).isoformat(sep=' ', timespec='seconds') if self.t0 else None
        event = {'Start(s)': round(t, 3), 'End(s)': None, 'Racine': stage, 'Start time': clock}
        self.events.append(event)
        self.save_events()
        self.update_event_list(select=event)
        self.draw_plot()

    def end_callback(self):
        event = self.selected_event()
        if event is None:
            return
        t = self.frame_time(self.frame)
        if t <= event['Start(s)']:
            QtWidgets.QMessageBox.warning(self, 'Mark end', 'The end must be after the start of the seizure.')
            return
        event['End(s)'] = round(t, 3)
        self.save_events()
        self.update_event_list(select=event)
        self.draw_plot()

    def delete_callback(self):
        event = self.selected_event()
        if event is None:
            return
        self.events.remove(event)
        self.save_events()
        self.update_event_list()
        self.draw_plot()

    def selected_event(self):
        row = self.event_list.currentRow()
        return self.events[row] if 0 <= row < len(self.events) else None

    def update_event_list(self, select=None):
        self.events.sort(key=lambda e: e['Start(s)'])
        self.event_list.clear()
        for e in self.events:
            end = f' to {format_s(e["End(s)"])} ({e["End(s)"] - e["Start(s)"]:.0f} s)' if e['End(s)'] else ''
            clock = f'  [{str(e["Start time"])[11:19]}]' if e['Start time'] else ''
            self.event_list.addItem(f'Racine {e["Racine"]}: {format_s(e["Start(s)"])}{end}{clock}')
        if select is not None:
            self.event_list.setCurrentRow(self.events.index(select))

    def event_clicked(self, item):
        event = self.selected_event()
        if event is not None and self.seekable:
            self.pause()
            self.show_frame(self.time_frame(event['Start(s)']))

    def closeEvent(self, event):
        self.pause()
        self.status_timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()


def format_s(t):
    t = int(t)
    return f'{t // 3600:d}:{t // 60 % 60:02d}:{t % 60:02d}'


def launch_GUI(raw_dir=RAW_DIR):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet('QPushButton{font-size: 12pt;} QLabel{font-size: 12pt;} QListWidget{font-size: 12pt;}')
    window = MainWindow(raw_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    launch_GUI()
