import sys, os
import re
from PyQt5 import QtGui, QtWidgets, QtCore
import os.path
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
from LFP.SzDet.AppSpikeSz.SzViewData import SzReviewData

#GUI
# import sys
# from PyQt5 import Qt
# from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                                    QGroupBox, QListWidget, QAction, QAbstractItemView, QLineEdit, QCheckBox)

from Proc2P.utils import *

'''
Manually review automatically detected seizures
'''

class GUI_main(QtWidgets.QMainWindow):
    def __init__(self, app, setupID='Pinnacle', defaults=None):
        super().__init__()

        self.setWindowTitle(f'Curate seizures')
        self.setGeometry(30, 60, 3200, 1600) # Left, top, width, height.
        self.app = app
        self.user_defaults = defaults

        #peristent settings
        self.setup = setupID #'LNCM' or 'Pinnacle'
        self.set_defaults()

        #main groupbox (horizonal)
        self.szlist_groupbox = self.make_szlist_groupbox()
        self.display_traces_groupbox = self.make_traces_groupbox()
        self.video_groupbox = self.make_video_groupbox()

        #central widget
        centralwidget = QWidget(self)
        horizontal_layout = QHBoxLayout()

        # add main layouts
        horizontal_layout.addWidget(self.szlist_groupbox)
        horizontal_layout.addWidget(self.display_traces_groupbox)
        horizontal_layout.addWidget(self.video_groupbox)

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
        if self.setup == 'Pinnacle':
            self.settings['rec_suffix'] = '_seizure_times.xlsx'
            self.settings['pardir'] = 'D:/Shares/Data/_Processed/EEG/'
        elif self.setup == 'LNCM':
            self.settings['rec_suffix'] = '_seizure_times.xlsx'
            self.settings['pardir'] = 'D:/Shares/Data/_Processed/2P/'
        if self.user_defaults is not None:
            for key, value in self.user_defaults.items():
                self.settings[key] = value

    def get_prefix(self, ps):
        if self.setup == 'Pinnacle':
            s1 = '.edf_Ch'
            if s1 in ps:
                #if saved with no tag
                f1 = ps.find(s1)
                return ps[:f1], ps[f1 + len(s1):][0], None
            else:
                #saved with tag, use regex to find tag and channel
                s1 = '.edf_'
                f1 = ps.find(s1)
                prefix = ps[:f1]
                pattern = r'.edf_(\w+)_Ch(.)_seizure_times.xlsx'
                match = re.match(pattern, ps[f1:])
                tag = match.group(1)
                ch = match.group(2)
                return prefix, ch, tag
        elif self.setup == 'LNCM':
            s1 = '_Ch'
            f1 = ps.find(s1)
            return ps[:f1], ps[f1+len(s1):][0], None

    def make_szlist_groupbox(self):
        groupbox = QGroupBox('File list')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        #select button
        select_path_button = QPushButton('Select recording', )
        vbox.addWidget(select_path_button)
        select_path_button.clicked.connect(self.select_rec_callback)
        vbox.addWidget(self.separator)

        #label
        self.path_label = QLabel('...', )
        self.path_label.setWordWrap(True)
        vbox.addWidget(self.path_label)

        #list
        self.sz_list = QListWidget(self)
        self.sz_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sz_list.currentRowChanged.connect(self.list_clicked)
        vbox.addWidget(self.sz_list)

        return groupbox

    def make_traces_groupbox(self):
        groupbox = QGroupBox('Traces')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        #horizontal layout for buttons
        horizontal_layout = QHBoxLayout()

        #prev button
        left = "\u2190"
        button = QPushButton(f'Previous [{left}]', )
        horizontal_layout.addWidget(button)
        button.clicked.connect(self.previous_callback)

        #toggle button
        # sp = "\u2423"
        # up = "\u2191"
        # down = "\u2193"
        button = QPushButton(f'Toggle [T]', )
        horizontal_layout.addWidget(button)
        button.clicked.connect(self.toggle_callback)

        #next button
        right = "\u2192"
        button = QPushButton(f'Next [{right}]', )
        horizontal_layout.addWidget(button)
        button.clicked.connect(self.next_callback)

        #save button
        self.save_button = QPushButton(f'Save', )
        horizontal_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_callback)

        #actions
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

        vbox.addLayout(horizontal_layout)

        vbox.addWidget(self.separator)
        #later add a middle, thinner row for motion
        self.FigCanvas1 = SubplotsCanvas()
        toolbar = NavigationToolbar2QT(self.FigCanvas1, self)
        vbox.addWidget(toolbar)
        vbox.addWidget(self.FigCanvas1)

        return groupbox

    def make_video_groupbox(self):
        groupbox = QGroupBox('Video')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        #horizontal layout for buttons
        horizontal_layout = QHBoxLayout()

        #prev button
        button = QPushButton(f'Show [M]', )
        horizontal_layout.addWidget(button)

        vbox.addLayout(horizontal_layout)

        vbox.addWidget(self.separator)
        #later add motion triggered average and live movie canvases

        return groupbox

    def select_rec_callback(self):
        # open a recording: display filedioalog, call SzReviewData, update listbox.

        fn = QtWidgets.QFileDialog.getOpenFileName(self, caption='Select Recording', directory=self.settings['pardir'],
                                                   filter=f'*{self.settings["rec_suffix"]}')
        self.wdir = os.path.dirname(fn[0])
        if self.setup == 'LNCM':
            self.wdir = os.path.dirname(self.wdir)+'/'
        self.prefix, self.ch, self.tag = self.get_prefix(os.path.basename(fn[0]))

        self.path_label.setText(f'Loading data for {self.prefix} ... Please wait.')
        QApplication.processEvents()
        self.current_selected_i = None

        #load data
        self.szdat = SzReviewData(self.wdir, self.prefix, self.ch, tag=self.tag, setup=self.setup)

        #set list widget
        self.sz_list.clear()
        self.sz_list.addItems(self.szdat.szlist)
        self.active_sz = None

        #color if already completed:
        included_sz = self.szdat.output_sz['Included'].eq(True)
        excluded_sz = self.szdat.output_sz['Included'].eq(False)
        for i, sz in enumerate(self.szdat.szlist):
            if included_sz[i]:
                self.mark_complete(i, color='true')
            elif excluded_sz[1]:
                self.mark_complete(i, color='false')

        self.path_label.setText(self.prefix)

    def mark_complete(self, i, color):
        if color == 'true':
            color ='#50a3a4'
        elif color == 'false':
            color = '#fcaf38'
        self.sz_list.item(i).setForeground(QtGui.QColor(color))
        self.sz_list.item(i).setSelected(False)

    def next_callback(self):
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
        self.active_sz = current_item.text()
        self.sz_list.setCurrentItem(current_item)
        self.new_plot=True
        self.refresh_data()

    def list_clicked(self):
        self.current_selected_i = self.sz_list.currentRow()
        self.select_sz()

    def previous_callback(self):
        print('Prev cb')
        if self.current_selected_i is None:
            self.current_selected_i = 0
        elif self.current_selected_i > 0:
            self.current_selected_i -= 1
        self.select_sz()

    def toggle_callback(self):
        if self.active_sz is None:
            self.next_callback()

        if not self.unsaved_changes:
            self.save_button.setStyleSheet("background-color: red")
        self.unsaved_changes += 1
        self.save_button.setText(f'Save ({self.unsaved_changes})')

        curr = self.szdat.get_sz(self.active_sz)
        if curr == '' or curr:
            self.szdat.set_sz(self.active_sz, False)
            self.mark_complete(self.current_selected_i, 'false')
        elif curr == False:
            self.szdat.set_sz(self.active_sz, True)
            self.mark_complete(self.current_selected_i, 'true')

    def refresh_data(self):
        self.szdat.plot_sz(self.active_sz, self.FigCanvas1.axd)
        self.FigCanvas1.draw()
        if self.unsaved_changes > 9:
            self.save_callback()

class SubplotsCanvas(FigureCanvasQTAgg):

    def __init__(self, *args, **kwargs):
        self.fig, self.axd = plt.subplot_mosaic([['top', 'top'],
                                       ['lower left', 'lower right']],
                                      gridspec_kw={'width_ratios': [1, 0.3, ]},
                                      figsize=(12, 6), layout="constrained")
        super(SubplotsCanvas, self).__init__(self.fig)

def launch_GUI(*args, **kwargs):
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app.setStyleSheet("QPushButton{font-size: 12pt;};QLabel{font-size: 12pt;};QListWidget{font-size: 14pt;}")
    app.setStyle('Fusion')
    gui_main = GUI_main(app, *args, **kwargs)
    sys.exit(app.exec())

if __name__ == '__main__':
    launch_GUI()
