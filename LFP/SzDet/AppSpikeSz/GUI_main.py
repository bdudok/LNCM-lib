import os

#plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
import numpy

#GUI
import sys
from pyqtgraph import Qt
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.Qt.QtWidgets import (QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox, QListWidget,
QAbstractItemView)

#Workflow Specific functions
from LFP.SpikeDet import SpikesPower
from LFP.SzDet import InstRate
from Proc2P.Legacy.Loaders import load_ephys


class GUI_main(QtWidgets.QMainWindow):
    def __init__(self, app):
        super().__init__()

        self.setWindowTitle(f'Detect spikes and seizures')
        self.setGeometry(10, 30, 3200, 1600) # Left, top, width, height.
        self.app = app

        #peristent settings
        self.settings = QtCore.QSettings("pyqt_settings.ini", QtCore.QSettings.IniFormat)
        self.settings.value("LastFile")

        #variables
        self.set_defaults()

        #main groupbox (horizonal)
        self.filelist_groupbox = self.make_filelist_groupbox()
        self.display_traces_groupbox = self.make_traces_groupbox()

        #central widget
        centralwidget = QWidget(self)
        horizontal_layout = QHBoxLayout()
        # vertical_layout_files = QtWidgets.QVBoxLayout()

        # add main layouts
        horizontal_layout.addWidget(self.filelist_groupbox)
        horizontal_layout.addWidget(self.display_traces_groupbox)
        self.setCentralWidget(centralwidget)
        self.centralWidget().setLayout(horizontal_layout)

        self.show()

    def set_defaults(self):
        self.wdir = None
        self.active_prefix = None
        self.separator = QLabel("<hr>")

        self.param = {}
        self.param['tr1'] = 3
        self.param['tr2'] = 5
        self.param['fs'] = 10000

    def make_filelist_groupbox(self):
        groupbox = QGroupBox('File list')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        #select button
        select_path_button = QPushButton('Select folder', )
        vbox.addWidget(select_path_button)
        select_path_button.clicked.connect(self.select_path_callback)

        vbox.addWidget(self.separator)

        #label
        self.path_label = QLabel('...', )
        self.path_label.setWordWrap(True)
        vbox.addWidget(self.path_label)

        #list
        self.prefix_list = QListWidget(self)
        self.prefix_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        vbox.addWidget(self.prefix_list)

        return groupbox

    def make_traces_groupbox(self):
        groupbox = QGroupBox('Traces')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        #horizontal layout for buttons
        horizontal_layout = QHBoxLayout()
        #load button
        select_session_button = QPushButton('Open', )
        horizontal_layout.addWidget(select_session_button)
        select_session_button.clicked.connect(self.load_file_callback)

        #next button
        select_next_button = QPushButton('Next', )
        horizontal_layout.addWidget(select_next_button)
        select_next_button.clicked.connect(self.load_next_callback)

        #layout button
        reset_layout_button = QPushButton('Layout', )
        horizontal_layout.addWidget(reset_layout_button)
        reset_layout_button.clicked.connect(plt.tight_layout)

        vbox.addLayout(horizontal_layout)

        vbox.addWidget(self.separator)
        #raw
        self.FigCanvas1 = SubplotsCanvas(nrows=3, sharex=True, dpi=600) #figsize=(12, 9), dpi=450,
        self.format_plot(self.FigCanvas1)
        toolbar = NavigationToolbar2QT(self.FigCanvas1, self)
        vbox.addWidget(toolbar)
        vbox.addWidget(self.FigCanvas1)

        return groupbox

    def select_path_callback(self):
        #get a folder
        self.wdir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.settings.setValue("LastFile", self.wdir)
        self.path_label.setText(self.wdir)
        os.chdir(self.wdir)
        self.current_selected_i = None

        #get prefix list
        suffix = '.ephys'
        prefix_list = [fn[:-len(suffix)] for fn in os.listdir(self.wdir) if fn.endswith(suffix)]

        #set list widget
        self.prefix_list.clear()
        self.prefix_list.addItems(prefix_list)

    def format_plot(self, plot):
        for ca in plot.ax:
            ca.spines['right'].set_visible(False)
            ca.spines['top'].set_visible(False)


    def update_plot1(self):
        #clear plot
        ax = self.FigCanvas1.ax
        for ca in ax:
            ca.cla()

        #get data
        t = self.spikedet.get_spikes(tr1=self.param['tr1'], tr2=self.param['tr2'])
        fs = self.spikedet.fs
        framesize = 64
        sz_burden, sz_times = InstRate.SpikeTrace(t, framesize=framesize, cleanup=5, gap=3)
        tx = (t*fs).astype('int64')
        Xrate = framesize/1000
        X = numpy.arange(0, len(sz_burden) * Xrate, Xrate)
        time_xvals = numpy.arange(0, len(self.spikedet.trace)) / fs

        #plot data
        ax[0].plot(time_xvals, self.spikedet.trace, color='black', zorder=1, linewidth=1)
        ax[0].scatter(time_xvals[tx], self.spikedet.trace[tx], color='red', marker='x', s=10, zorder=2)
        ax[0].set_ylabel('LFP')

        ax[1].plot(time_xvals, self.spikedet.env, color='orange', zorder=1, linewidth=1)
        ax[1].scatter(time_xvals[tx], self.spikedet.env[tx], color='red', marker='x', s=10, zorder=2)
        for hv in ('tr1', 'tr2'):
            ax[1].axhline(self.spikedet.stdev_env * self.param[hv], color='black', linewidth=0.5)
        ax[1].set_ylabel('HF envelope')

        ax[2].plot(X, sz_burden, color='blue', linewidth=1)
        for sz in sz_times:
            for ca in (0, 2):
                ax[ca].axvspan(sz[0], sz[1], color='red', alpha=0.4, zorder=0)

        #set X axis
        # min_unit = fs
        # minutes = numpy.arange(0, len(self.spikedet.trace) / min_unit, min_unit)
        # minlabels = numpy.arange(0, len(minutes), 1)
        # ax[2].set_xticks(minutes)
        # ax[2].set_xticklabels(minlabels)
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Seizure')


        self.FigCanvas1.draw()
        plt.tight_layout()

    def load_file_callback(self):
        self.active_prefix = self.prefix_list.selectedItems()[0].text()
        self.refresh_data()

    def load_next_callback(self):
        self.current_selected_i += 1
        self.prefix_list.setCurrentRow(self.current_selected_i)
        self.active_prefix = self.prefix_list.selectedItems()[0].text()
        self.refresh_data()

    def refresh_data(self):
        print(self.active_prefix)
        r = load_ephys(self.active_prefix)

        #get example trace
        fs = self.param['fs']
        t_want = fs * 60 * 10 #10 minutes
        trace = r.trace
        if len(trace) > t_want:
            trace = trace[int(len(trace)/2 - t_want/2):int(len(trace)/2 + t_want/2)]
        self.spikedet = SpikesPower.Detect(trace, fs=fs)
        self.update_plot1()



class SubplotsCanvas(FigureCanvasQTAgg):

    def __init__(self, *args, **kwargs):
        self.fig, self.ax = plt.subplots(*args, **kwargs)
        super(SubplotsCanvas, self).__init__(self.fig)

def launch_GUI():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app)
    sys.exit(app.exec())

if __name__ == '__main__':
    # pass
    launch_GUI()
