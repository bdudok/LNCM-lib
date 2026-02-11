import os
import json
from subprocess import Popen
from pathlib import Path
import argparse
#plotting
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
import numpy

#GUI
import sys
from pyqtgraph import Qt
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.Qt.QtWidgets import (QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox, QListWidget,
QAbstractItemView, QLineEdit, QCheckBox)

#Workflow Specific functions
from LFP.SpikeDet import SpikesPower
from LFP.SzDet import InstRate, ProcessSeizures

#Readers
from Proc2P.Legacy.Loaders import load_ephys
from Proc2P.Bruker import LoadEphys
from LFP.Pinnacle import ReadEDF

#for processing in background
# from multiprocessing import Process

class GUI_main(QtWidgets.QMainWindow):
    def __init__(self, app, path=None, savepath=None, prefix_list=None, setupID='LNCM',
                 defaults=None, set_prefix=None):
        super().__init__()

        self.setWindowTitle(f'Detect spikes and seizures')
        self.setGeometry(30, 60, 3200, 1600) # Left, top, width, height.
        self.app = app
        if defaults == 'hippocampus':
            defaults = gui_defaults
        self.user_defaults = defaults

        #peristent settings
        self.wdir = path
        self.set_prefix = set_prefix
        if savepath is None:
            savepath = path
        self.savepath = savepath
        self.input_prefix = prefix_list
        self.setup = setupID #'LNCM' or "Soltesz' or 'Pinnacle'
        # self.settings = QtCore.QSettings("pyqt_settings.ini", QtCore.QSettings.IniFormat)
        # self.settings.value("LastFile")

        #variables
        self.set_defaults()

        #main groupbox (horizonal)
        self.filelist_groupbox = self.make_filelist_groupbox()
        self.display_traces_groupbox = self.make_traces_groupbox()
        self.options_groupbox = self.make_options_groupbox()
        self.controls_groupbox = self.make_controls_groupbox()
        self.results_groupbox = self.make_results_groupbox()

        #central widget
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
        if set_prefix is not None:
            self.active_prefix = set_prefix
            self.new_plot = True
            self.refresh_data()
        self.settings_saved = False
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
                                  'Sz.MinDur', 'Sz.Gap', 'SzDet.Framesize', 'fs', 'PlotDur', 'Channel')
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
        self.param['fs'] = 5000
        self.param['PlotDur'] = 10
        self.param['Channel'] = 1
        if self.setup == 'Soltesz':
            self.param['fs'] = 10000
            self.param['SzDet.Framesize'] = 64
        elif self.setup == 'LNCM':
            self.param['SzDet.Framesize'] = 50
            self.param['fs'] = 5000
        if self.setup == 'Pinnacle':
            self.param_keys_sorted = (*self.param_keys_sorted,
                                      'rejection_value', 'rejection_step', 'rejection_tail', 'rejection_factor')
        if self.user_defaults is not None:
            for key, value in self.user_defaults.items():
                self.param[key] = value

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
        self.param_fields[label].setText(str(value))

    def set_result(self, label, value):
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

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QLabel(label))


        hline = QtWidgets.QHBoxLayout()
        hline.addWidget(QLabel('Tag'))
        self.default_savetag = 'No tag'
        self.savetag_field =  QLineEdit(self)
        self.savetag_field.setText(self.default_savetag)
        hline.addWidget(self.savetag_field)
        save_button = QPushButton('Save settings', )
        hline.addWidget(save_button)
        save_button.clicked.connect(self.save_output_callback)
        load_button = QPushButton('Load settings', )
        hline.addWidget(load_button)
        load_button.clicked.connect(self.load_output_callback)
        proc_button = QPushButton('Process', )
        hline.addWidget(proc_button)
        proc_button.clicked.connect(self.process_callback)
        vbox.addLayout(hline)

        return groupbox


    def select_path_callback(self, path=False):
        #get a folder
        if path in (None, False):
            self.wdir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder') + '/'
            self.savepath = self.wdir
            # self.settings.setValue("LastFile", self.wdir)
        else:
            self.wdir = path
        self.path_label.setText(self.wdir)
        os.chdir(self.wdir)
        self.current_selected_i = None

        flist = os.listdir(self.wdir)
        #get prefix list
        if self.setup == 'Soltesz':
            suffix = '.ephys'
            if self.input_prefix is None:
                prefix_list = [fn[:-len(suffix)] for fn in flist if fn.endswith(suffix)]
            else:
                prefix_list = [prefix for prefix in self.input_prefix if prefix+suffix in flist]
        elif self.setup == 'LNCM':
            suffix = '_ephys.npy'
            self.dirpaths = {}
            prefix_list = []
            available_pf = []
            for fn in flist:
                dirpath = self.wdir+fn
                if os.path.isdir(dirpath):
                    dflist = os.listdir(dirpath)
                    for dfn in dflist:
                        if suffix in dfn:
                            self.dirpaths[fn] = dirpath
                            prefix_list.append(fn)
            if self.input_prefix is not None:
                prefix_list = [prefix for prefix in self.input_prefix if prefix in prefix_list]
        elif self.setup == 'Pinnacle':
            suffix = '.edf'
            prefix_list = [fn[:-len(suffix)] for fn in flist if fn.endswith(suffix)]
            if self.input_prefix is not None:
                prefix_list = [x for x in prefix_list if x in self.input_prefix]
        #set list widget
        self.prefix_list.clear()
        self.prefix_list.addItems(prefix_list)

        #check if output exist
        flist = os.listdir(self.savepath)
        if self.setup in ['Soltesz', 'Pinnacle']:
            for pi, prefix in enumerate(prefix_list):
                if prefix+self.suffix+'.json' in flist:
                    self.mark_complete(pi, color='#fcaf38')
        elif self.setup == 'LNCM':
            for pi, prefix in enumerate(prefix_list):
                if os.path.exists(os.path.join(self.dirpaths[prefix], prefix+'.json')):
                    self.mark_complete(pi, color='#fcaf38')

    def format_plot(self, plot):
        for ca in plot.ax:
            ca.spines['right'].set_visible(False)
            ca.spines['top'].set_visible(False)


    def update_plot1(self):
        self.settings_saved = False
        print([self.get_field(field) for field in self.param_keys_sorted])
        #clear plot
        ax = self.FigCanvas1.ax
        if not self.new_plot:
            ylims = []
            xlims = []
            for ca in ax:
                ylims.append(ca.get_ylim())
                xlims.append(ca.get_xlim())
        for ca in ax:
            ca.cla()

        #get data
        self.fs = int(self.param['fs'])
        lo = int(float(self.get_field('LoCut')))
        hi = int(float(self.get_field('HiCut')))
        if (lo != self.previous_lo) or (hi != self.previous_hi):
            self.spikedet = SpikesPower.Detect(self.trace, fs=self.fs, lo=lo, hi=hi)
            self.previous_lo = lo
            self.previous_hi = hi
        t = self.spikedet.get_spikes(tr1=float(self.get_field('Tr1')), tr2=float(self.get_field('Tr2')),
                                     trdiff=float(self.get_field('TrDiff')), dur=float(self.get_field('Dur')),
                                     dist=float(self.get_field('Dist')))
        framesize = int(self.get_field('SzDet.Framesize'))
        sz_burden, sz_times_raw = InstRate.SpikeTrace(t, framesize,
                                                  # length=int(len(self.spikedet.trace/framesize)),
                                                  cleanup=float(self.get_field('Sz.MinDur')),
                                                  gap=float(self.get_field('Sz.Gap')))
        sz_times = sz_times_raw.astype('int32')
        tx = (t*self.fs).astype('int64')
        Xrate = framesize/1000
        X = numpy.arange(0, len(sz_burden)) * Xrate
        time_xvals = numpy.arange(0, len(self.spikedet.trace)) / self.fs

        #plot data
        if self.raw_trace is not None:
            wh = numpy.where(self.spikedet.trace == 0)
            scale = min(self.spikedet.trace.max(), -self.spikedet.trace.min()) / numpy.absolute(self.raw_trace).max()
            ax[0].plot(time_xvals[wh], self.raw_trace[wh]*scale, color='red', zorder=0, linewidth=1)
        ax[0].plot(time_xvals, self.spikedet.trace, color='black', zorder=1, linewidth=1)
        ax[0].scatter(time_xvals[tx], self.spikedet.trace[tx], color='lime',
                      marker='x', s=20, linewidth=0.7, zorder=2, alpha=0.8)
        ax[0].set_ylabel('LFP')

        ax[1].plot(time_xvals, self.spikedet.env, color='orange', zorder=1, linewidth=1)
        ax[1].scatter(time_xvals[tx], self.spikedet.env[tx], color='red',
                      marker='x', s=20, linewidth=0.7, zorder=2, alpha=0.8)
        for hv in ('Tr1', 'Tr2'):
            v = float(self.get_field(hv))
            ax[1].axhline(self.spikedet.mean_env + self.spikedet.stdev_env * v, color='black', linewidth=0.5)
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

        #('Spikes', 'Seizures', 'Sz.Duration')
        self.set_result('Spikes', len(t))
        self.set_result('Seizures', len(sz_times))
        self.set_result('Sz.Duration', f'{numpy.mean([sz[1] - sz[0] for sz in sz_times]):.2f}')

    def load_file_callback(self):
        current_item = self.prefix_list.selectedItems()[0]
        self.active_prefix = current_item.text()
        self.current_selected_i = self.prefix_list.currentRow()
        self.new_plot = True
        self.refresh_data()

    def get_output_fn(self):
        if self.setup == 'Soltesz':
            output_fn = self.savepath + self.active_prefix + self.suffix
        elif self.setup == 'Pinnacle':
            output_fn = self.savepath + self.active_prefix + f'_Ch{self.param["channel_index"]}' + self.suffix
        elif self.setup == 'LNCM':
            output_fn = os.path.join(self.savepath, self.active_prefix, self.active_prefix +
                                     f'_Ch{self.get_field("Channel")}' + self.suffix)
        #add tag
        savetag = self.savetag_field.text()
        if not savetag == self.default_savetag:
            output_fn += f'_{savetag}'
        return output_fn

    def save_output_callback(self):
        output_fn = self.get_output_fn()

        op_dict = {}
        for key in self.param_keys_sorted:
            op_dict[key] = self.get_field(key)
        op_dict['Included'] = self.is_sz_checkbox.isChecked()

        with open(output_fn + '.json', 'w') as fp:
            json.dump(op_dict, fp)
        self.settings_saved = True

        #save plot
        for ca in self.FigCanvas1.ax:
            ca.autoscale()
        self.FigCanvas1.fig.savefig(output_fn + '.png', dpi=300)

        self.mark_complete(self.current_selected_i)
        # if not self.setup == 'Pinnacle':
        #     self.load_next_callback()

    def load_file_callback(self):
        current_item = self.prefix_list.selectedItems()[0]
        self.active_prefix = current_item.text()
        self.current_selected_i = self.prefix_list.currentRow()
        self.new_plot = True
        self.refresh_data()

    def load_output_callback(self):
        output_fn = self.get_output_fn()
        with open(output_fn + '.json', 'r') as fp:
            op_dict = json.load(fp)

        for key in self.param_keys_sorted:
            if key in op_dict:
                self.set_field(key, op_dict[key])
        self.is_sz_checkbox.setChecked(bool(op_dict['Included']))
        self.update_plot1()

    def process_callback(self):
        savetag = self.savetag_field.text()
        if savetag == self.default_savetag:
            savetag = None
        #process the whole trace using current settings
        if not self.settings_saved:
            self.save_output_callback()
        opts = {}
        for key in self.param_keys_sorted:
            opts[key] = self.get_field(key)
        opts['channel_index'] = self.param['channel_index']
        if self.setup == 'Pinnacle':
            # tt = Process(target=ProcessSeizures, args=(self.edf, opts))
            # tt.start()
            ProcessSeizures(self.edf, opts, format='edf', savetag=savetag)
        elif self.setup == 'LNCM':
            ProcessSeizures(self.ephys, opts, format='ephys', save_envelope=True, savetag=savetag)

    def mark_complete(self, i, color='#50a3a4'):
        self.prefix_list.item(i).setBackground(QtGui.QColor(color))

    def load_next_callback(self):
        self.current_selected_i += 1
        current_item = self.prefix_list.item(self.current_selected_i)
        self.active_prefix = current_item.text()
        self.prefix_list.setCurrentItem(current_item)
        self.new_plot=True
        self.refresh_data()

    def refresh_data(self):
        print(self.active_prefix)
        ch = self.get_field('Channel')
        if self.setup == 'Soltesz':
            r = load_ephys(self.active_prefix)
        elif self.setup == 'LNCM':
            chi = int(ch)
            self.param['channel_index'] = chi
            r = LoadEphys.Ephys(self.savepath, self.active_prefix, channel=chi)
            self.ephys = r
            trace = self.ephys.trace
        elif self.setup == 'Pinnacle':
            if ch.isdigit():
                ch = int(ch)-1
            r = ReadEDF.EDF(self.savepath, self.active_prefix, rejection_ops=self.param, ch=ch)
            self.param['fs'] = r.fs
            self.param['Channels'] = r.channels
            self.param['channel_index'] = r.chi + 1
            chstr = f'{r.active_channel}; {r.chi+1}/{len(r.channels)}'
            self.channel_name_label.setText(chstr + ' (Use Open to change channels)')
            self.edf = r
            trace = r.trace
        #get example trace
        self.fs = int(self.param['fs'])
        plotdur = self.get_field('PlotDur')

        if hasattr(r, 'raw_trace'):
            raw_trace = r.raw_trace
        else:
            raw_trace = trace
        if plotdur != 'all':
            t_want = int(self.fs * 60 * float(plotdur)) #10 minutes
            print(t_want)
            if len(trace) > t_want:
                want_slice = slice(int(len(trace)/2 - t_want/2), int(len(trace)/2 + t_want/2))
                trace = trace[want_slice]
                raw_trace = raw_trace[want_slice]
        self.raw_trace = raw_trace
        self.trace = trace
        self.set_result('Prefix', self.active_prefix)
        self.previous_lo = 0
        self.previous_hi = 0
        self.update_plot1()

class SubplotsCanvas(FigureCanvasQTAgg):

    def __init__(self, *args, **kwargs):
        self.fig, self.ax = plt.subplots(*args, **kwargs)
        super(SubplotsCanvas, self).__init__(self.fig)

def main(*args, **kwargs):
    '''for launching in a subprocess'''
    app = QtWidgets.QApplication()
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)

    # SzProcGUI(self.wdir, 'LNCM', 'hippocampus', self.active_prefix[0])
    path, setup_ID, defaults, prefix = sys.argv[1:]
    gui_main = GUI_main(app, path=path, setupID=setup_ID, defaults=defaults, set_prefix=prefix)
    sys.exit(app.exec())

gui_defaults = { #aiming to detect ripples and HFOs
    'LoCut': 80, #band pass filter low cut (Hz)
    'HiCut': 500, #band pass filter high cut (Hz)
    'Tr1': 3, #spike treshold for spike width measurement (SD)
    'Tr2': 5, #spike amplitude treshold for spike detection (SD)
    'TrDiff': 5, #threshold for including broad spikes (based on abs diff, SD)
    'Dur': 5, #spike minimum duration (ms)
    'Dist': 50, #spike separation (ms)
    'Sz.MinDur': 3, #spike cluster duration to be considered seizure (s)
    'Sz.Gap': 3, #gap for merging neighboring spike clusters (s)
    'SzDet.Framesize': 50, #resolution of output instantaneous spike rate trace (ms)
    'fs': 5000, #sampling rate (read from input in case of Pinnacle EDF file)
    'Channel': 1, #channel number (indexed from 1)
    'PlotDur': 'all', #displayed trace length in minutes, or 'all'
}

def launch_in_subprocess(*args, **kwargs):
    #can be called with args specifying the session to launch a standalone window
    cli_args = list(args)
    #avoiding kwargs so I don't have to parse
    for k, v in kwargs.items():
        flag = f"--{k.replace('_', '-')}"
        cli_args.append(flag)
        cli_args.append(str(v))
    Popen([sys.executable, Path(__file__), *cli_args])

def launch_GUI(*args, **kwargs):
    '''For launcher script'''
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app, *args, **kwargs)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
