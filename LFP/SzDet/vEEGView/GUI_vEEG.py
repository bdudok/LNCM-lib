import os
import json

# plotting
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
import numpy

# GUI
import sys
from pyqtgraph import Qt
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.Qt.QtWidgets import (QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox, QListWidget,
                                    QAbstractItemView, QLineEdit, QCheckBox)

# Workflow Specific functions
from LFP.SpikeDet import SpikesPower
from LFP.SzDet import InstRate

# Readers
from Proc2P.Legacy.Loaders import load_ephys
from Proc2P.Bruker import LoadEphys
from LFP.Pinnacle import ReadEDF
from Proc2P.Treadmill import rsync


class GUI_main(QtWidgets.QMainWindow):
    def __init__(self, app, path=None, savepath=None, prefix_list=None, setupID='LNCM',
                 defaults=None):
        super().__init__()

        self.setWindowTitle(f'Detect spikes and seizures')
        self.setGeometry(30, 60, 3200, 1600)  # Left, top, width, height.
        self.app = app
        self.user_defaults = defaults

        # peristent settings
        self.wdir = path

        self.savepath = savepath
        self.input_prefix = prefix_list
        self.setup = setupID  # 'LNCM' or "Soltesz'
        # self.settings = QtCore.QSettings("pyqt_settings.ini", QtCore.QSettings.IniFormat)
        # self.settings.value("LastFile")

        # variables
        self.set_defaults()

        # main groupbox (horizonal)
        self.filelist_groupbox = self.make_filelist_groupbox()
        self.display_traces_groupbox = self.make_traces_groupbox()
        self.options_groupbox = self.make_options_groupbox()
        self.controls_groupbox = self.make_controls_groupbox()
        self.results_groupbox = self.make_results_groupbox()

        # central widget
        centralwidget = QWidget(self)
        horizontal_layout = QHBoxLayout()
        # vertical_layout_files = QtWidgets.QVBoxLayout()

        # add main layouts
        horizontal_layout.addWidget(self.filelist_groupbox)

        horizontal_layout.addWidget(self.display_traces_groupbox)

        vertical_options_layout = QVBoxLayout()
        vertical_options_layout.addWidget(self.options_groupbox)
        vertical_options_layout.addWidget(self.separator)
        vertical_options_layout.addWidget(self.controls_groupbox)
        vertical_options_layout.addWidget(self.results_groupbox)
        horizontal_layout.addLayout(vertical_options_layout)

        self.setCentralWidget(centralwidget)
        self.centralWidget().setLayout(horizontal_layout)
        if path is not None:
            self.select_path_callback(path)
        self.show()

    def set_defaults(self):
        self.active_prefix = None
        self.separator = QLabel("<hr>")
        if self.savepath is None:
            self.savepath = './'
        self.suffix = '_SpikeSzDet'

        self.param = {}
        self.param_fields = {}
        self.result_fields = {}
        self.res_keys_sorted = ('Prefix', 'Spikes', 'Seizures', 'Sz.Duration')
        self.param_keys_sorted = ('LoCut', 'HiCut', 'Tr1', 'Tr2', 'TrDiff', 'Dur', 'Dist',
                                  'Sz.MinDur', 'Sz.Gap', 'SzDet.Framesize', 'fs', 'PlotDur')
        self.param['LoCut'] = 4
        self.param['HiCut'] = 35
        self.param['Tr1'] = 2
        self.param['Tr2'] = 4
        self.param['TrDiff'] = 3
        self.param['Dur'] = 20
        self.param['Dist'] = 50
        self.param['Sz.MinDur'] = 2
        self.param['Sz.Gap'] = 2
        self.param['SzDet.Framesize'] = 50
        self.param['fs'] = 2000
        self.param['PlotDur'] = 10
        if self.setup == 'Soltesz':
            self.param['fs'] = 10000
            self.param['SzDet.Framesize'] = 64
        elif self.setup == 'LNCM':
            self.param['SzDet.Framesize'] = 50
            self.param['fs'] = 2000
        if self.setup == 'Pinnacle':
            self.param['Channel'] = 1
            self.param_keys_sorted = (*self.param_keys_sorted, 'Channel',
                                      'rejection_value', 'rejection_step', 'rejection_tail', 'rejection_factor')
        if self.user_defaults is not None:
            for key, value in self.user_defaults.items():
                self.param[key] = value

    def make_filelist_groupbox(self):
        groupbox = QGroupBox('File list')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        # select button
        select_path_button = QPushButton('Select folder', )
        vbox.addWidget(select_path_button)
        select_path_button.clicked.connect(self.select_path_callback)

        vbox.addWidget(self.separator)

        # label
        self.path_label = QLabel('...', )
        self.path_label.setWordWrap(True)
        vbox.addWidget(self.path_label)

        # list
        twolists = QtWidgets.QHBoxLayout()
        self.prefix_list = QListWidget(self)
        self.prefix_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.prefix_list.itemSelectionChanged.connect(self.set_prefix)
        twolists.addWidget(self.prefix_list)
        self.video_list = QListWidget(self)
        self.video_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.video_list.itemSelectionChanged.connect(self.set_video)
        twolists.addWidget(self.video_list)
        vbox.addLayout(twolists)

        vbox.addWidget(self.separator)

        # align button
        align_button = QPushButton('Align', )
        vbox.addWidget(align_button)
        align_button.clicked.connect(self.align_callback)
        self.FigCanvas0 = SubplotsCanvas(figsize=(4, 2))
        self.format_plot(self.FigCanvas0)
        vbox.addWidget(self.FigCanvas0)

        return groupbox

    def make_traces_groupbox(self):
        groupbox = QGroupBox('Traces')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        # horizontal layout for buttons
        horizontal_layout = QHBoxLayout()
        # load button
        select_session_button = QPushButton('Open', )
        horizontal_layout.addWidget(select_session_button)
        select_session_button.clicked.connect(self.load_file_callback)

        # next button
        select_next_button = QPushButton('Next', )
        horizontal_layout.addWidget(select_next_button)
        select_next_button.clicked.connect(self.load_next_callback)

        # layout button
        reset_layout_button = QPushButton('Layout', )
        horizontal_layout.addWidget(reset_layout_button)
        reset_layout_button.clicked.connect(plt.tight_layout)

        vbox.addLayout(horizontal_layout)

        vbox.addWidget(self.separator)
        # raw
        self.FigCanvas1 = SubplotsCanvas(nrows=3, sharex=True, figsize=(12, 9))
        self.format_plot(self.FigCanvas1)
        toolbar = NavigationToolbar2QT(self.FigCanvas1, self)
        vbox.addWidget(toolbar)
        vbox.addWidget(self.FigCanvas1)

        return groupbox

    def make_field(self, label):
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QLabel(label))
        self.param_fields[label] = QLineEdit(self)
        if label in self.param:
            self.param_fields[label].setText(str(self.param[label]))
        hbox.addWidget(self.param_fields[label])
        return hbox

    def get_field(self, label):
        return self.param_fields[label].text()

    def set_field(self, label, value):
        self.result_fields[label].setText(str(value))

    def make_options_groupbox(self):
        groupbox = QGroupBox('Options')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        for label in self.param_keys_sorted:
            vbox.addLayout(self.make_field(label))

        if self.setup == 'Pinnacle':
            # vbox.addLayout(self.make_field('Channel'))
            self.channel_name_label = QLabel('Channel Name')
            vbox.addWidget(self.channel_name_label)

        return groupbox

    def make_controls_groupbox(self):
        groupbox = QGroupBox('Controls')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        reload_button = QPushButton('Reload', )
        vbox.addWidget(reload_button)
        reload_button.clicked.connect(self.update_plot1)
        vbox.addWidget(self.separator)

        return groupbox

    def make_result(self, label):
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QLabel(label))
        self.result_fields[label] = QLabel('')
        hbox.addWidget(self.result_fields[label])
        return hbox

    def make_results_groupbox(self):
        groupbox = QGroupBox('Results')
        vbox = QtWidgets.QVBoxLayout()
        groupbox.setLayout(vbox)

        for label in self.res_keys_sorted:
            vbox.addLayout(self.make_result(label))
        vbox.addWidget(self.separator)

        self.is_sz_checkbox = QCheckBox('Include')
        self.is_sz_checkbox.setChecked(True)
        vbox.addWidget(self.is_sz_checkbox)

        save_button = QPushButton('Save settings', )
        vbox.addWidget(save_button)
        save_button.clicked.connect(self.save_output_callback)
        return groupbox

    def select_path_callback(self, path=None):
        # get a folder
        if path is None:
            self.wdir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
            # self.settings.setValue("LastFile", self.wdir)
        self.path_label.setText(self.wdir)
        os.chdir(self.wdir)
        self.current_selected_i = None

        flist = os.listdir(self.wdir)
        # get prefix list
        if self.setup == 'Soltesz':
            suffix = '.ephys'
            if self.input_prefix is None:
                prefix_list = [fn[:-len(suffix)] for fn in flist if fn.endswith(suffix)]
            else:
                prefix_list = [prefix for prefix in self.input_prefix if prefix + suffix in flist]
        elif self.setup == 'LNCM':
            suffix = '_ephys.npy'
            self.dirpaths = {}
            prefix_list = []
            for fn in flist:
                dirpath = self.wdir + fn
                if os.path.isdir(dirpath):
                    dflist = os.listdir(dirpath)
                    for dfn in dflist:
                        if suffix in dfn:
                            self.dirpaths[fn] = dirpath
                            prefix_list.append(fn)
            if self.input_prefix is not None:
                prefix_list = [prefix for prefix in self.input_prefix if prefix in self.dirpaths]
        elif self.setup == 'Pinnacle':
            suffix = '.edf'
            prefix_list = [fn[:-len(suffix)] for fn in flist if fn.endswith(suffix)]
            if self.input_prefix is not None:
                prefix_list = [x for x in prefix_list if x in self.input_prefix]
            suffix = '.avi'
            video_list = [fn[:-len(suffix)] for fn in flist if fn.endswith(suffix)]
        # set list widget
        print(video_list)
        self.prefix_list.clear()
        self.prefix_list.addItems(prefix_list)
        self.video_list.clear()
        self.video_list.addItems(video_list)

        # check if output exist
        flist = os.listdir(self.savepath)
        if self.setup in ['Soltesz', 'Pinnacle']:
            for pi, prefix in enumerate(prefix_list):
                if prefix + self.suffix + '.json' in flist:
                    self.mark_complete(pi, color='#fcaf38')
        elif self.setup == 'LNCM':
            for pi, prefix in enumerate(prefix_list):
                if os.path.exists(os.path.join(self.dirpaths[prefix], prefix + '.json')):
                    self.mark_complete(pi, color='#fcaf38')

    def align_callback(self):
        #load video npy and perform align
        vttl = numpy.load(self.wdir + self.active_video + '.npy')[:, -1]
        self.refresh_data()
        ttl = self.edf.get_TTL()

        # thgis only for this test, recorder now updated. #todo remove this after testing
        if vttl[0] > numpy.diff(vttl).max():
            vttl = vttl[1:] - vttl[0]  # does not contain actual frame numbers but since live view started :S

        try:
            self.alignment = rsync.Rsync_aligner(vttl, ttl)
        except:
            self.alignment = None

        eegdur = len(self.edf.trace) / self.edf.fs
        xunit = 60
        ca = self.FigCanvas0.ax
        ca.cla()
        ca.plot((0, eegdur / xunit), (1, 1), color='black')
        ca.scatter((0, eegdur / xunit), (1, 1), marker='|', color='black')
        if self.alignment is not None:
            align_fps = self.alignment.units_B
            frametimes = self.alignment.B_to_A(numpy.arange(vttl[-1] + numpy.diff(vttl).max())) / align_fps
            ft_bounds = frametimes[numpy.logical_not(numpy.isnan(frametimes))]
            ca.scatter(ttl / xunit, numpy.ones(len(ttl)) * 0.5, marker='|', c=numpy.isnan(self.alignment.cor_times_A))
            ca.plot((ft_bounds[0] / xunit, ft_bounds[-1] / xunit), (1, 1), color='green', linewidth=2)
            ca.plot((vttl[0] / (align_fps * xunit), vttl[-1] / (align_fps * xunit)), (0, 0), color='grey')
        else:
            ca.plot((vttl[0] / (31 * xunit), vttl[-1] / (31 * xunit)), (0, 0), color='red')
        ca.set_ylim(-0.1, 1.1)
        ca.set_xlabel('Minutes')

        self.FigCanvas0.draw()
        plt.tight_layout()

    def format_plot(self, plot):
        ax = plot.ax
        if not hasattr(ax, '__len__'):
            ax = [ax]
        for ca in ax:
            ca.spines['right'].set_visible(False)
            ca.spines['top'].set_visible(False)

    def update_plot1(self):
        return 0 #todo for testing.
        print([self.get_field(field) for field in self.param_keys_sorted])
        # clear plot
        ax = self.FigCanvas1.ax
        if not self.new_plot:
            ylims = []
            xlims = []
            for ca in ax:
                ylims.append(ca.get_ylim())
                xlims.append(ca.get_xlim())
        for ca in ax:
            ca.cla()

        # get data
        t = self.spikedet.get_spikes(tr1=float(self.get_field('Tr1')), tr2=float(self.get_field('Tr2')),
                                     trdiff=float(self.get_field('TrDiff')), dur=float(self.get_field('Dur')),
                                     dist=float(self.get_field('Dist')))
        fs = self.spikedet.fs
        framesize = int(self.get_field('SzDet.Framesize'))
        sz_burden, sz_times_raw = InstRate.SpikeTrace(t, framesize,
                                                      # length=int(len(self.spikedet.trace/framesize)),
                                                      cleanup=float(self.get_field('Sz.MinDur')),
                                                      gap=float(self.get_field('Sz.Gap')))
        sz_times = sz_times_raw.astype('int32')
        tx = (t * fs).astype('int64')
        Xrate = framesize / 1000
        X = numpy.arange(0, len(sz_burden) * Xrate, Xrate)
        time_xvals = numpy.arange(0, len(self.spikedet.trace)) / fs

        # plot data
        if self.raw_trace is not None:
            wh = numpy.where(self.spikedet.trace == 0)
            scale = min(self.spikedet.trace.max(), -self.spikedet.trace.min()) / numpy.absolute(self.raw_trace).max()
            ax[0].plot(time_xvals[wh], self.raw_trace[wh] * scale, color='red', zorder=0, linewidth=1)
        ax[0].plot(time_xvals, self.spikedet.trace, color='black', zorder=1, linewidth=1)
        ax[0].scatter(time_xvals[tx], self.spikedet.trace[tx], color='lime',
                      marker='x', s=20, linewidth=0.7, zorder=2, alpha=0.8)
        ax[0].set_ylabel('LFP')

        ax[1].plot(time_xvals, self.spikedet.env, color='orange', zorder=1, linewidth=1)
        ax[1].scatter(time_xvals[tx], self.spikedet.env[tx], color='red',
                      marker='x', s=20, linewidth=0.7, zorder=2, alpha=0.8)
        for hv in ('Tr1', 'Tr2'):
            v = float(self.get_field(hv))
            ax[1].axhline(self.spikedet.stdev_env * v, color='black', linewidth=0.5)
        ax[1].set_ylabel('HF envelope')

        ax[2].plot(X, sz_burden, color='blue', linewidth=1)
        for sz in sz_times_raw:
            for ca in (0, 2):
                ax[ca].axvspan(sz[0], sz[1], color='red', alpha=0.4, zorder=0)

        if not self.new_plot:
            for ca, xlim, ylim in zip(ax, xlims, ylims):
                ca.set_xlim(xlim)
                ca.set_ylim(ylim)
        self.new_plot = False
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Seizure')

        self.FigCanvas1.draw()
        plt.tight_layout()

        # ('Spikes', 'Seizures', 'Sz.Duration')
        self.set_field('Spikes', len(t))
        self.set_field('Seizures', len(sz_times))
        self.set_field('Sz.Duration', f'{numpy.mean([sz[1] - sz[0] for sz in sz_times]):.2f}')

    def load_file_callback(self):
        # current_item = self.prefix_list.selectedItems()[0]
        # self.active_prefix = current_item.text()
        self.current_selected_i = self.prefix_list.currentRow()
        self.new_plot = True
        self.refresh_data()

    def set_prefix(self):
        self.active_prefix = self.prefix_list.selectedItems()[0].text()
    def set_video(self):
        self.active_video = self.video_list.selectedItems()[0].text()

    def save_output_callback(self):
        # save settings
        if self.setup == 'Soltesz':
            output_fn = self.savepath + self.active_prefix + self.suffix
        elif self.setup == 'Pinnacle':
            output_fn = self.savepath + self.active_prefix + f'_Ch{self.get_field("Channel")}' + self.suffix
        elif self.setup == 'LNCM':
            output_fn = os.path.join(self.savepath, self.active_prefix, self.active_prefix + self.suffix)
        op_dict = {}
        for key in self.param_keys_sorted:
            op_dict[key] = self.get_field(key)
        op_dict['Included'] = self.is_sz_checkbox.isChecked()
        with open(output_fn + '.json', 'w') as fp:
            json.dump(op_dict, fp)

        # save plot
        for ca in self.FigCanvas1.ax:
            ca.autoscale()
        self.FigCanvas1.fig.savefig(output_fn + '.png', dpi=300)

        self.mark_complete(self.current_selected_i)
        if not self.setup == 'Pinnacle':
            self.load_next_callback()

    def mark_complete(self, i, color='#50a3a4'):
        self.prefix_list.item(i).setBackground(QtGui.QColor(color))

    def load_next_callback(self):
        self.current_selected_i += 1
        current_item = self.prefix_list.item(self.current_selected_i)
        self.active_prefix = current_item.text()
        self.prefix_list.setCurrentItem(current_item)
        self.new_plot = True
        self.refresh_data()

    def refresh_data(self):
        print(self.active_prefix)
        if self.setup == 'Soltesz':
            r = load_ephys(self.active_prefix)
        elif self.setup == 'LNCM':
            r = LoadEphys.Ephys(self.savepath, self.active_prefix)
        elif self.setup == 'Pinnacle':
            r = ReadEDF.EDF(self.savepath, self.active_prefix, rejection_ops=self.param)
            self.edf=r
            self.param['fs'] = r.fs
            self.param['Channels'] = r.channels
            chi = int(self.get_field('Channel'))
            r.set_channel(chi - 1)
            chstr = f'{r.active_channel}; {chi}/{len(r.channels)}'
            self.channel_name_label.setText(chstr + ' (Use Open to change channels)')

        # get example trace
        fs = int(self.get_field('fs'))
        t_want = int(fs * 60 * float(self.get_field('PlotDur')))  # 10 minutes
        print(t_want)
        trace = r.trace
        raw_trace = r.raw_trace
        if len(trace) > t_want:
            want_slice = slice(int(len(trace) / 2 - t_want / 2), int(len(trace) / 2 + t_want / 2))
            trace = trace[want_slice]
            if raw_trace is not None:
                raw_trace = raw_trace[want_slice]
        self.raw_trace = raw_trace
        self.spikedet = SpikesPower.Detect(trace, fs=fs, lo=float(self.get_field('LoCut')),
                                           hi=float(self.get_field('HiCut')))
        self.set_field('Prefix', self.active_prefix)
        self.update_plot1()


class SubplotsCanvas(FigureCanvasQTAgg):

    def __init__(self, *args, **kwargs):
        self.fig, self.ax = plt.subplots(*args, **kwargs)
        super(SubplotsCanvas, self).__init__(self.fig)


def launch_GUI(*args, **kwargs):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app, *args, **kwargs)
    sys.exit(app.exec())


if __name__ == '__main__':
    path = 'D:\Shares\Data\_Processed\EEG\Tottering/'
    launch_GUI(path)