import sys
import re

import pandas
from PySide6 import QtGui, QtWidgets, QtCore
import os.path
import cv2
import datetime
from subprocess import Popen
from pathlib import Path
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
from LFP.SzDet.AppSpikeSz.SzViewData import SzReviewData

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                             QGroupBox, QListWidget,QAbstractItemView, QLineEdit)

from Proc2P.utils import *
from LFP.Setup import Setup

'''
Manually review automatically detected seizures
'''


class GUI_main(QtWidgets.QMainWindow):
    def __init__(self, app, setupID=Setup.PINNACLE, defaults=None):
        super().__init__()

        self.setWindowTitle(f'Curate seizures')
        self.setGeometry(30, 60, 3200, 1600)  # Left, top, width, height.
        self.app = app
        self.user_defaults = defaults

        # peristent settings
        self.setup = Setup(setupID)
        self.set_defaults()

        # main groupbox (horizonal)
        self.szlist_groupbox = self.make_szlist_groupbox()
        self.display_traces_groupbox = self.make_traces_groupbox()
        self.video_groupbox = self.make_video_groupbox()  # shown when the recording has video

        # central widget
        centralwidget = QWidget(self)
        horizontal_layout = QHBoxLayout()

        # add main layouts
        horizontal_layout.addWidget(self.szlist_groupbox)
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.addWidget(self.display_traces_groupbox)
        splitter.addWidget(self.video_groupbox)
        splitter.setSizes([2000, 1000])
        horizontal_layout.addWidget(splitter)

        self.setCentralWidget(centralwidget)
        self.centralWidget().setLayout(horizontal_layout)
        self.show()

    def set_defaults(self):
        self.settings = {}
        self.prefix = None
        self.wdir = None
        self.separator = QLabel("<hr>")
        self.viewer_on = True
        self.unsaved_changes = 0
        self.settings['video_dir'] = None  # folder of the vEEG recordings, default: the EEG folder
        if self.setup == Setup.PINNACLE:
            self.settings['rec_suffix'] = '_seizure_times.xlsx'
            self.settings['pardir'] = 'D:/Shares/Data/_Processed/EEG/'
        elif self.setup == Setup.LNCM:
            self.settings['rec_suffix'] = '_seizure_times.xlsx'
            self.settings['pardir'] = 'D:/Shares/Data/_Processed/2P/'
        if self.user_defaults is not None:
            for key, value in self.user_defaults.items():
                self.settings[key] = value

    def get_prefix(self, ps, use_tag=''):
        if self.setup == Setup.PINNACLE:
            s1 = '.edf_Ch'
            if s1 in ps:
                # if saved with no tag
                f1 = ps.find(s1)
                return ps[:f1], ps[f1 + len(s1):][0], None
            else:
                # saved with tag, use regex to find tag and channel
                s1 = '.edf_'
                f1 = ps.find(s1)
                prefix = ps[:f1]
                pattern = r'.edf_(\w+)_Ch(.)_seizure_times.xlsx'
                match = re.match(pattern, ps[f1:])
                tag = match.group(1)
                ch = match.group(2)
                return prefix, ch, tag
        elif self.setup == Setup.LNCM:
            s1 = '_Ch'
            f1 = ps.find(s1)
            if use_tag in ('', None, 'None'):
                return ps[:f1], ps[f1 + len(s1):][0], None
            else:
                try:
                    f2 = ps.find(f'_{use_tag}_')
                except:
                    raise ValueError(f'Tag "{use_tag}" not in the filename')
                return ps[:f2], ps[f1 + len(s1):][0], use_tag



    def make_szlist_groupbox(self):
        groupbox = QGroupBox('File list')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        # select button
        select_path_button = QPushButton('Select recording', )
        vbox.addWidget(select_path_button)
        select_path_button.clicked.connect(self.select_rec_callback)
        # for when the matching video is not found automatically (vEEG recordings)
        self.locate_video_button = QPushButton('Locate video...')
        self.locate_video_button.clicked.connect(self.locate_video_callback)
        self.locate_video_button.setEnabled(False)
        self.locate_video_button.setVisible(self.setup == Setup.PINNACLE)
        vbox.addWidget(self.locate_video_button)
        vbox.addWidget(self.separator)

        # label
        self.path_label = QLabel('...', )
        self.path_label.setWordWrap(True)
        vbox.addWidget(self.path_label)

        # list
        self.sz_list = QListWidget(self)
        self.sz_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sz_list.currentRowChanged.connect(self.list_clicked)
        vbox.addWidget(self.sz_list)

        groupbox.setFixedWidth(180)
        return groupbox

    def make_traces_groupbox(self):
        groupbox = QGroupBox('Traces')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        # horizontal layout for buttons
        horizontal_layout = QHBoxLayout()

        # tag field
        horizontal_layout.addWidget(QtWidgets.QLabel('Tag:'))
        self.tag_field = QLineEdit()
        self.tag_field.setFixedWidth(80)
        self.tag_field.setText('None')
        horizontal_layout.addWidget(self.tag_field)


        # prev button
        left = "\u2190"
        button = QPushButton(f'Previous [{left}]', )
        horizontal_layout.addWidget(button)
        button.clicked.connect(self.previous_callback)

        # toggle button
        button = QPushButton(f'Toggle [t]', )
        horizontal_layout.addWidget(button)
        button.clicked.connect(self.toggle_callback)

        # interictal button
        button = QPushButton(f'Interictal [i]', )
        horizontal_layout.addWidget(button)
        button.clicked.connect(self.interictal_callback)

        # next button
        right = "\u2192"
        self.next_button = QPushButton(f'Next [{right}]', )
        horizontal_layout.addWidget(self.next_button)
        self.next_button.clicked.connect(self.next_callback)

        # save button
        self.save_button = QPushButton(f'Save', )
        horizontal_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_callback)

        # actions
        self.goLeftAction = QAction('Previous', self)
        self.goLeftAction.setShortcut('left')
        self.goLeftAction.triggered.connect(self.previous_callback)
        self.addAction(self.goLeftAction)

        self.goRightAction = QAction('Next', self)
        self.goRightAction.setShortcut('right')
        self.goRightAction.triggered.connect(self.next_callback)
        self.addAction(self.goRightAction)

        self.ToggleAction = QAction('Toggle', self)
        self.ToggleAction.setShortcut('t')
        self.ToggleAction.triggered.connect(self.toggle_callback)
        self.addAction(self.ToggleAction)

        self.InterictalAction = QAction('Interictal', self)
        self.InterictalAction.setShortcut('i')
        self.InterictalAction.triggered.connect(self.interictal_callback)
        self.addAction(self.InterictalAction)

        # add indicators
        self.DurationLabel = QLabel('SzDur', )
        self.FreqLabel = QLabel('SzFreq', )
        self.PISLabel = QLabel('Suppression', )
        for label in (self.DurationLabel, self.FreqLabel, self.PISLabel):
            label.setFixedWidth(60)
            label.setFixedHeight(20)
            label.setFont(QFont('Arial', 12))
            horizontal_layout.addWidget(label)

        vbox.addLayout(horizontal_layout)

        vbox.addWidget(self.separator)
        self.FigCanvas1 = SubplotsCanvas()
        self.toolbar = SelectorToolbar(self.FigCanvas1, self)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.FigCanvas1)

        return groupbox

    def make_video_groupbox(self):
        groupbox = QGroupBox('Video')
        vbox = QVBoxLayout()
        groupbox.setLayout(vbox)

        self.video_label = FrameLabel()
        vbox.addWidget(self.video_label, stretch=1)
        self.video_info = QLabel('')
        # text width must not resize the panel: that resizes the plot, and its full redraw stalls playback
        self.video_info.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        vbox.addWidget(self.video_info)

        controls = QHBoxLayout()
        self.play_button = QPushButton('Play [p]')
        self.play_button.clicked.connect(self.play_callback)
        controls.addWidget(self.play_button)
        button = QPushButton('Mark behavior [b]')
        button.clicked.connect(self.behavior_callback)
        controls.addWidget(button)
        button = QPushButton('Clear mark')
        button.clicked.connect(self.clear_behavior_callback)
        controls.addWidget(button)
        vbox.addLayout(controls)

        for name, key, callback in (('Play', 'p', self.play_callback), ('Mark behavior', 'b', self.behavior_callback)):
            action = QAction(name, self)
            action.setShortcut(key)
            action.triggered.connect(callback)
            self.addAction(action)

        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.timeout.connect(self.next_frame)
        self.cap, self.cap_path, self.cap_next = None, None, None  # open video file, frame that read() returns next
        self.video_t = None  # EEG time (s) of the displayed frame
        self.dragging = False
        self.marker_background = None
        self.FigCanvas1.mpl_connect('draw_event', self.on_draw)
        self.FigCanvas1.mpl_connect('button_press_event', self.marker_press)
        self.FigCanvas1.mpl_connect('motion_notify_event', self.marker_motion)
        self.FigCanvas1.mpl_connect('button_release_event', self.marker_release)
        groupbox.setVisible(False)
        return groupbox

    def reset_video(self):
        self.pause()
        if self.cap is not None:
            self.cap.release()
        self.cap, self.cap_path, self.cap_next = None, None, None
        self.video_t = None
        self.video_label.set_frame(None)
        self.video_info.clear()

    def show_video_time(self, t):
        """Display the frame at EEG time t (s) and move the marker there."""
        self.video_t = t
        loc = self.szdat.video_frame(t)
        sz_start = self.szdat.get_sz(self.active_sz, full=True)['Start']
        if loc is None:
            self.video_label.set_frame(None)
            self.video_info.setText(f'No video at {t - sz_start:+.2f} s from seizure start')
        else:
            path, frame = loc
            if path != self.cap_path:
                if self.cap is not None:
                    self.cap.release()
                self.cap, self.cap_path, self.cap_next = cv2.VideoCapture(path), path, None
            # the frame on display again needs no read (EEG time steps during playback are not exactly one frame)
            if self.cap_next is None or frame != self.cap_next - 1:
                if self.cap_next is not None and 0 <= frame - self.cap_next < 30:
                    for _ in range(frame - self.cap_next):  # a few frames ahead: decode on instead of seeking
                        self.cap.grab()
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                self.read_frame(frame)
            self.video_info.setText(f'{t - sz_start:+.2f} s from seizure start | {os.path.basename(path)} frame {frame}')
        self.move_marker()

    def read_frame(self, frame):
        ok, img = self.cap.read()
        self.cap_next = frame + 1 if ok else None
        self.video_label.set_frame(img[:, :, 0] if ok else None)  # mono camera: all channels are equal

    def play_callback(self):
        if self.video_t is None:
            return
        if self.frame_timer.isActive():
            self.pause()
            return
        if self.video_t >= self.szdat.plot_t1:
            self.show_video_time(self.szdat.get_sz(self.active_sz, full=True)['Start'])
        self.frame_timer.start(int(1000 / self.szdat.video_map['fps']))
        self.play_button.setText('Pause [p]')

    def pause(self):
        self.frame_timer.stop()
        self.play_button.setText('Play [p]')

    def next_frame(self):
        t = self.video_t + 1 / self.szdat.video_map['fps']
        if t > self.szdat.plot_t1:
            self.pause()
            return
        self.show_video_time(t)

    def on_draw(self, event):
        # the figure was redrawn: store it without the marker, then draw the marker on top
        if self.video_t is None:
            return
        ax = self.FigCanvas1.axd['top']
        self.marker_background = self.FigCanvas1.copy_from_bbox(ax.bbox)
        self.szdat.marker_line.set_xdata([self.szdat.time_to_x(self.video_t)] * 2)
        ax.draw_artist(self.szdat.marker_line)

    def move_marker(self):
        if self.marker_background is None:
            return
        canvas = self.FigCanvas1
        line = self.szdat.marker_line
        line.set_xdata([self.szdat.time_to_x(self.video_t)] * 2)
        canvas.restore_region(self.marker_background)
        line.axes.draw_artist(line)
        canvas.blit(line.axes.bbox)

    def marker_press(self, event):
        ax = self.FigCanvas1.axd['top']
        if self.video_t is None or event.inaxes is not ax or self.toolbar.mode or self.toolbar.cid is not None:
            return
        if self.frame_timer.isActive():  # while playing, a click anywhere jumps there and playback continues
            self.show_video_time(min(max(self.szdat.x_to_time(event.xdata), self.szdat.plot_t0), self.szdat.plot_t1))
            return
        marker_px = ax.transData.transform((self.szdat.time_to_x(self.video_t), 0))[0]
        if abs(event.x - marker_px) < 10:
            self.dragging = True
            self.pause()

    def marker_motion(self, event):
        if not self.dragging or event.inaxes is not self.FigCanvas1.axd['top']:
            return
        t = min(max(self.szdat.x_to_time(event.xdata), self.szdat.plot_t0), self.szdat.plot_t1)
        self.show_video_time(t)

    def marker_release(self, event):
        self.dragging = False

    def behavior_callback(self):
        if self.video_t is None:
            return
        self.szdat.set_sz(self.active_sz, self.video_t, 'BehaviorSz(s)')
        self.flag_unsaved()
        self.refresh_data()

    def clear_behavior_callback(self):
        if self.video_t is None:
            return
        self.szdat.set_sz(self.active_sz, numpy.nan, 'BehaviorSz(s)')
        self.flag_unsaved()
        self.refresh_data()

    def select_rec_callback(self):
        # open a recording: display filedioalog, call SzReviewData, update listbox.

        fn = QtWidgets.QFileDialog.getOpenFileName(self, caption='Select Recording', dir=self.settings['pardir'],
                                                   filter=f'*{self.settings["rec_suffix"]}')
        self.wdir = os.path.dirname(fn[0])
        if self.setup == Setup.LNCM:
            self.wdir = os.path.dirname(self.wdir) + '/'
        self.prefix, self.ch, self.tag = self.get_prefix(os.path.basename(fn[0]), use_tag=self.tag_field.text())

        self.path_label.setText(f'Loading data for {self.prefix} ... Please wait.')
        QApplication.processEvents()
        self.current_selected_i = None

        # load data
        print(f'Calling szdat with prefix {self.prefix} ch {self.ch}, tag {self.tag}')
        self.reset_video()
        self.szdat = SzReviewData(self.wdir, self.prefix, self.ch, tag=self.tag, setup=self.setup,
                                  video_dir=self.settings['video_dir'], pull_motion=True)
        self.video_groupbox.setVisible(self.szdat.video_map is not None)
        self.locate_video_button.setEnabled(True)

        # set list widget
        self.sz_list.clear()
        self.sz_list.addItems(self.szdat.szlist)
        self.active_sz = None

        # color if already completed:
        included_sz = self.szdat.output_sz['Included'].isin((1, 'TRUE', True)).array
        excluded_sz = self.szdat.output_sz['Included'].isin((0, 'FALSE', False)).array
        interictal_sz = self.szdat.output_sz['Interictal'].isin((1, 'TRUE', True)).array
        for i, sz in enumerate(self.szdat.szlist):
            if included_sz[i]:
                if interictal_sz[i]:
                    self.mark_complete(i, color='interictal')
                else:
                    self.mark_complete(i, color='true')
            elif excluded_sz[i]:
                self.mark_complete(i, color='false')
        # print(self.szdat.output_sz['Included'])
        # print(excluded_sz)

        if self.setup == Setup.PINNACLE and self.szdat.video_map is None:
            self.path_label.setText(f'{self.prefix}\nNo matching video found. Use Locate video to select it.')
        else:
            self.path_label.setText(self.prefix)

    def locate_video_callback(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption='Select a video of this recording',
                                                      dir=str(self.szdat.video_dir), filter='*.mkv')
        if not fn:
            return
        self.path_label.setText('Matching the video ... Please wait.')
        QApplication.processEvents()
        try:
            self.szdat.locate_video(fn)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Locate video', str(e))
            self.path_label.setText(self.prefix)
            return
        self.path_label.setText(self.prefix)
        self.video_groupbox.setVisible(True)
        if self.active_sz is not None:
            self.select_sz()  # show the video at the current seizure

    def mark_complete(self, i, color):
        color = self.color_lut(color)
        self.sz_list.item(i).setForeground(QtGui.QColor(color))
        self.sz_list.item(i).setSelected(False)

    def color_lut(self, color):
        lut = {
            'true': '#38fc59',
            'false': '#fc8638',
            'interictal': '#38cbfc'
        }
        return lut.get(color, 'grey')

    def next_callback(self):
        if pandas.isna(self.szdat.get_sz(self.active_sz)):  # if pressed next without marking it not sz, we set it as reviewed
            self.szdat.set_sz(self.active_sz, True)
            self.mark_complete(self.current_selected_i, 'true')
            self.flag_unsaved()
        if self.current_selected_i is None:
            self.current_selected_i = 0
        elif (self.current_selected_i + 1) < len(self.szdat.output_sz):
            self.current_selected_i += 1
        self.select_sz()

    def save_callback(self):
        self.save_button.setStyleSheet("background-color: grey")
        self.unsaved_changes = 0
        self.save_button.setText('Save')
        self.szdat.save()
        self.path_label.setText(self.prefix + f' saved {datetime.datetime.now():%H:%M}')

    def select_sz(self):
        current_item = self.sz_list.item(self.current_selected_i)
        if current_item is not None:
            self.active_sz = current_item.text()
            self.sz_list.setCurrentItem(current_item)
            self.new_plot = True
            self.reset_video()
            self.refresh_data()
            self.color_next_button()
            if self.szdat.video_map is not None:
                self.show_video_time(self.szdat.get_sz(self.active_sz, full=True)['Start'])

    def check_true(self, val, missing=False):
        if pandas.isna(val) or val == '':
            return missing
        return bool(val)

    def color_next_button(self):
        sz = self.szdat.get_sz(self.active_sz, full=True)
        is_sz = self.check_true(sz["Included"], True)  # value can be missing
        is_iis = self.check_true(sz["Interictal"])
        if is_sz and not is_iis:
            self.next_button.setStyleSheet(f"background-color: {self.color_lut('true')}")
        elif is_sz and is_iis:
            self.next_button.setStyleSheet(f"background-color: {self.color_lut('interictal')}")
        elif not is_sz:
            self.next_button.setStyleSheet(f"background-color: {self.color_lut('false')}")

    def update_indicators(self):
        self.curation_checks = {}
        sz = self.szdat.get_sz(self.active_sz, full=True)
        dur = sz["Duration(s)"]
        if dur > self.szdat.settings["Curation.MinDur"]:
            self.DurationLabel.setStyleSheet("background-color: green")
            self.curation_checks["Duration"] = True
        else:
            self.DurationLabel.setStyleSheet("background-color: red")
            self.curation_checks["Duration"] = False
        self.DurationLabel.setText(f'{dur:.1f} s')
        szfreq = sz["SpkFreq"]
        if szfreq > self.szdat.settings["Curation.MinFreq"]:
            self.FreqLabel.setStyleSheet("background-color: green")
            self.curation_checks["Frequency"] = True
        else:
            self.FreqLabel.setStyleSheet("background-color: red")
            self.curation_checks["Frequency"] = False
        self.FreqLabel.setText(f'{szfreq:.1f} Hz')
        PISdur = sz["PostIctalSuppression(s)"]
        if PISdur > self.szdat.settings["Curation.PISDur"]:
            self.PISLabel.setStyleSheet("background-color: green")
            self.curation_checks["PIS"] = True
        else:
            self.PISLabel.setStyleSheet("background-color: red")
            self.curation_checks["PIS"] = False
        self.PISLabel.setText(f'{PISdur:.1f} s')

    def list_clicked(self):
        self.current_selected_i = self.sz_list.currentRow()
        self.select_sz()

    def previous_callback(self):
        if self.current_selected_i is None:
            self.current_selected_i = 0
        elif self.current_selected_i > 0:
            self.current_selected_i -= 1
        self.select_sz()

    def flag_unsaved(self):
        if not self.unsaved_changes:
            self.save_button.setStyleSheet("background-color: red")
        self.unsaved_changes += 1
        self.save_button.setText(f'Save ({self.unsaved_changes})')

    def toggle_callback(self):
        if self.active_sz is None:
            self.next_callback()
        self.flag_unsaved()
        curr = self.szdat.get_sz(self.active_sz)
        if self.check_true(curr):
            self.szdat.set_sz(self.active_sz, False)
            self.mark_complete(self.current_selected_i, 'false')
        else:
            self.szdat.set_sz(self.active_sz, True)
            self.mark_complete(self.current_selected_i, 'true')
        self.color_next_button()

    def interictal_callback(self):
        if self.active_sz is None:
            self.next_callback()

        self.flag_unsaved()
        sz = self.szdat.get_sz(self.active_sz, full=True)
        is_sz = self.check_true(sz["Included"], False)  # value can be missing
        if is_sz and not (sz["Included"] is True):  # set to reviewed
            self.szdat.set_sz(self.active_sz, value=True)
        is_iis =  self.check_true(sz["Interictal"])
        if is_sz and not is_iis:  # toggle ii
            self.szdat.set_sz(self.active_sz, value=True, key='Interictal')
            self.mark_complete(self.current_selected_i, 'interictal')
        elif is_sz and is_iis:  # toggle ii
            self.szdat.set_sz(self.active_sz, value=False, key='Interictal')
            self.mark_complete(self.current_selected_i, 'true')
        elif not is_sz:  # set to sz and ii
            self.szdat.set_sz(self.active_sz, True)
            self.szdat.set_sz(self.active_sz, value=True, key='Interictal')
            self.mark_complete(self.current_selected_i, 'interictal')
        self.color_next_button()

    def edit_sz_callback(self, event_type, xcoord):
        self.szdat.edit_sz_from_gui(event_type, xcoord)
        self.flag_unsaved()
        self.refresh_data()

    def refresh_data(self):
        self.szdat.plot_sz(self.active_sz, self.FigCanvas1.axd)
        self.update_indicators()
        self.FigCanvas1.draw()
        if self.unsaved_changes > 5:
            self.save_callback()


class FrameLabel(QLabel):
    '''Shows a video frame scaled to the label, keeping the aspect ratio; rescales when the label is resized.'''

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        # Ignored: the label follows the layout instead of growing to the size of the frame
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image = None

    def set_frame(self, frame):
        if frame is None:
            self.image = None
            self.clear()
            return
        frame = numpy.ascontiguousarray(frame)
        h, w = frame.shape
        self.image = QtGui.QImage(frame.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        self.update_pixmap()

    def update_pixmap(self):
        if self.image is not None:
            self.setPixmap(QtGui.QPixmap.fromImage(self.image).scaled(self.size(), Qt.KeepAspectRatio,
                                                                      Qt.FastTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()


class SubplotsCanvas(FigureCanvasQTAgg):

    def __init__(self, *args, **kwargs):
        self.fig, self.axd = plt.subplot_mosaic([['top', 'top'],
                                                 ['motion', 'motion'],
                                                 ['lower left', 'lower right']],
                                                gridspec_kw={'width_ratios': [1, 0.3, ],
                                                             'height_ratios': [1, 0.35, 1]},
                                                figsize=(12, 6), layout="constrained")
        self.axd['motion'].sharex(self.axd['top'])
        super(SubplotsCanvas, self).__init__(self.fig)


class SelectorToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.main_gui = parent
        self.cid = None  # click handler waiting for the edit
        # Add a custom button to the toolbar
        self.addSeparator()
        self.start_button = QPushButton("Edit start")
        self.start_button.clicked.connect(self.start_action)
        self.addWidget(self.start_button)
        self.stop_button = QPushButton("Edit end")
        self.stop_button.clicked.connect(self.stop_action)
        self.addWidget(self.stop_button)

    def start_action(self, event):
        self.wait_for_click('Start')

    def stop_action(self):
        self.wait_for_click('Stop')

    def wait_for_click(self, event_type):
        self.event_type = event_type
        if self.cid is None:
            self.cid = self.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None
            self.main_gui.edit_sz_callback(self.event_type, event.xdata)


def main(*args, **kwargs):
    '''for launching in a subprocess'''
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QtWidgets.QApplication()
    # app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app.setStyleSheet("QPushButton{font-size: 12pt;};QLabel{font-size: 12pt;};QListWidget{font-size: 14pt;}")
    app.setStyle('Fusion')
    setup_ID,  = sys.argv[1:]
    gui_main = GUI_main(app, setupID=setup_ID)
    sys.exit(app.exec())

def launch_in_subprocess(*args, **kwargs):
    #can be called with args specifying the session to launch a standalone window
    Popen([sys.executable, Path(__file__), *args])

def launch_GUI(*args, **kwargs):
    '''For launcher script'''
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app.setStyleSheet("QPushButton{font-size: 12pt;};QLabel{font-size: 12pt;};QListWidget{font-size: 14pt;}")
    app.setStyle('Fusion')
    gui_main = GUI_main(app, *args, **kwargs)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
