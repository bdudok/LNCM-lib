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
import datetime
from dataclasses import dataclass, fields, asdict

from Proc2P.utils import logger, lprint
from Proc2P.Analysis.BatchGUI.Config import *
from Proc2P.Analysis.AssetFinder import AssetFinder
# from Proc2P.Analysis.BatchGUI.RoiEditorQt import GUI_main as RoiEditorGUI
from Proc2P.Analysis.BatchGUI.RoiEditorQt import launch_in_subprocess as RoiEditorGUI
from Proc2P.Analysis.BatchGUI.QueueManager import Job, JobType
from Proc2P.Analysis.GEVIReg.Register import RegConfig
from Proc2P.Analysis.AnalysisClasses.NormalizeVm import PullVmConfig
from Proc2P.Analysis.RoiEditor import SIMAConfig
from Proc2P.Analysis.PullSignals import PullConfig
from Proc2P.Bruker.LoadRegistered import Source
from Proc2P.Analysis.CaTrace import ProcessConfig

'''
Gui for viewing and processing 2P data.
This file just defines widgets, all functions are imported from the analysis classes or the legacy (Tk) BatchGUI app.
'''


class Tabs(Enum):
    ROI = 'ROI Editor'
    GEVIReg = 'GEVIReg'
    PullROIs = 'Pull signals'
    ProcessCa = 'Process ROIs'
    PullVm = 'Pull Vm'
    SIMA = 'SIMA'


def apply_layout(widget):
    widget.setLayout(widget.layout)


class GUI_main(QtWidgets.QMainWindow):
    __name__ = 'BatchGUI'

    def __init__(self, app, title='BatchGUI', Q_manager=None):
        super().__init__()
        self.setWindowTitle(title)
        self.app = app
        self.config = GuiConfig()
        self.Q = Q_manager

        self.wdir = '.'
        self.saved_fields = ('wdir', 'tag_label')  # GUI values that are saved in a user-specific config file
        self.selector_fields = ()  # list those of saved_fields which are single-select widgets
        self.attr_fields = ('wdir',)  # list those of saved_fields where we store the attribute value
        # anything not in selector or attr fields is using the .text() of the corresponding widget
        self.settings_filename = self.config.settings_filename
        self.stripchars = "'+. *?~!@#$%^&*(){}:[]><,/" + '"' + '\\'
        self.separator = QtWidgets.QLabel("<hr>")
        self.assets = AssetFinder()
        self.load_settings_file()
        self.state = State.SETUP

        # session bar - on top, displays active path, session, ROI tag
        self.session_bar = QtWidgets.QWidget(self)
        self.make_session_bar()

        # tabs
        self.table_widget = QtWidgets.QWidget(self)
        self.table_widget.layout = QtWidgets.QHBoxLayout(self.table_widget)

        self.filelist_widget = QtWidgets.QWidget(self)
        self.make_filelist_widget()
        self.table_widget.layout.addWidget(self.filelist_widget)

        self.tabs = QtWidgets.QTabWidget()
        self.n_tabs = 0
        self.active_tab = Tabs.GEVIReg
        self.make_gevireg_tab()
        self.make_sima_tab()
        self.make_pull_tab()
        self.make_process_tab()
        self.make_editor_tab()
        self.make_PullVm_tab()
        self.tabs.resize(int(100 * self.n_tabs), 100)

        # add tabs to table
        self.table_widget.layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        apply_layout(self.table_widget)

        # "console" for printing responses
        self.console_widget = QtWidgets.QPlainTextEdit(self)
        self.make_console_widget()

        # add everything to main window layout
        self.main_widget = QtWidgets.QWidget(self)  # central widget
        self.main_widget.layout = QtWidgets.QVBoxLayout(self.table_widget)
        self.main_widget.layout.addWidget(self.session_bar)
        self.main_widget.layout.addWidget(self.separator)
        self.main_widget.layout.addWidget(self.table_widget)
        self.main_widget.layout.addWidget(self.separator)
        self.main_widget.layout.addWidget(self.console_widget)
        apply_layout(self.main_widget)

        self.setCentralWidget(self.main_widget)

        self.load_gui_sate()

        self.setGeometry(*self.config.MainWindowGeometry)  # Left, top, width, height.

        # timer to poll the queue
        self._queue_timer = QtCore.QTimer(self)
        self._queue_timer.setInterval(1000)
        self._queue_timer.timeout.connect(self.poll_result)
        self._queue_timer.start()

        self.show()

    def make_session_bar(self):
        self.session_bar.setFixedHeight(self.config.TextWidgetHeight)
        self.session_bar.layout = QtWidgets.QHBoxLayout(self.session_bar)
        select_path_button = QtWidgets.QPushButton('Select folder', )
        self.session_bar.layout.addWidget(select_path_button)
        select_path_button.clicked.connect(self.select_path_callback)
        self.session_bar.layout.addWidget(self.separator)
        self.path_label = QtWidgets.QLabel(str(self.settings_dict.get('wdir')))
        self.session_bar.layout.addWidget(self.path_label)
        self.session_bar.layout.addWidget(self.separator)

        self.session_bar.layout.addWidget(QtWidgets.QLabel('Prefix:'))
        self.prefix_label = QtWidgets.QLabel('')
        self.prefix_label.setFixedWidth(self.config.PrefixFieldWidth)
        self.session_bar.layout.addWidget(self.prefix_label)
        self.session_bar.layout.addWidget(self.separator)

        self.session_bar.layout.addWidget(QtWidgets.QLabel('ROI tag:'))
        self.tag_label = QtWidgets.QLineEdit()
        if 'tag_label' in self.settings_dict:
            self.tag_label.setText(str(self.settings_dict['tag_label']))
        self.tag_label.setFixedWidth(self.config.TagFieldWidth)
        self.session_bar.layout.addWidget(self.tag_label)

    def make_console_widget(self):
        self.console_widget.setFixedHeight(self.config.ConsoleWidgetHeight)
        self.cprint('~')
        self.redirector = StdRedirector(self.console_widget)
        sys.stdout = self.redirector

    def make_filelist_widget(self):
        self.filelist_widget.setFixedWidth(self.config.PrefixFieldWidth)
        self.filelist_widget.layout = QtWidgets.QVBoxLayout()
        self.filelist_widget.layout.addWidget(QtWidgets.QLabel('Prefix list'))
        self.prefix_list = QtWidgets.QListWidget(self)
        self.prefix_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.prefix_list.itemSelectionChanged.connect(self.set_prefix)
        self.filelist_widget.layout.addWidget(self.prefix_list)

        apply_layout(self.filelist_widget)

    def make_editor_tab(self):
        self.EditorTab = QtWidgets.QWidget()
        self.EditorTab.layout = QtWidgets.QVBoxLayout()
        self.EditorTab.layout.addWidget(QtWidgets.QLabel('Click on a session to open the editor in a new window.'))
        open_button = QtWidgets.QPushButton('Open')
        open_button.clicked.connect(self.update_editor_callback)
        self.EditorTab.layout.addWidget(open_button)

        apply_layout(self.EditorTab)
        self.tabs.addTab(self.EditorTab, Tabs.ROI.value)
        self.n_tabs += 1

    def update_editor_callback(self):
        RoiEditorGUI(self.wdir, self.active_prefix[0], self.tag_label.text())

    def make_gevireg_tab(self):
        self.GEVIRegTab = QtWidgets.QWidget()
        self.GEVIRegTab.layout = QtWidgets.QVBoxLayout()
        self.GEVIRegTab.layout.addWidget(QtWidgets.QLabel('Select sessions to add to processing queue'))
        reg_button = QtWidgets.QPushButton('Process')
        reg_button.clicked.connect(self.reg_button_callback)
        self.GEVIRegTab.layout.addWidget(reg_button)

        # add a form to edit config
        self.GEVI_config_editors = {}
        GEVI_Config_widget = QtWidgets.QWidget()
        self.GEVI_config_form = QtWidgets.QFormLayout()
        self.GEVI_config = RegConfig()
        self.create_config_form(self.GEVI_config_form, self.GEVI_config, self.GEVI_config_editors)
        GEVI_Config_widget.layout = self.GEVI_config_form
        apply_layout(GEVI_Config_widget)
        self.GEVIRegTab.layout.addWidget(GEVI_Config_widget)

        apply_layout(self.GEVIRegTab)
        self.tabs.addTab(self.GEVIRegTab, Tabs.GEVIReg.value)
        self.n_tabs += 1

    def reg_button_callback(self):
        self.get_config_from_form(self.GEVI_config, self.GEVI_config_editors)
        for prefix in self.active_prefix:
            self.cprint('GEVIReg:', prefix, 'queued.')
            self.Q.run_job(Job(JobType.GEVIReg, (self.wdir, prefix, self.GEVI_config)))

    def create_config_form(self, form: QtWidgets.QFormLayout, cfg: dataclass, editors: dict):
        '''Adds widgets to edit each field of a dataclass. Keeps them in a dict for the getter'''
        for field in fields(cfg):
            fname = field.name
            ftype = field.type
            fvalue = getattr(cfg, fname)
            if ftype is int:
                w = QtWidgets.QSpinBox()
                w.setMaximum(10 ** 9)
                w.setValue(fvalue)
            elif ftype is float:
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(2)
                w.setMaximum(1e9)
                w.setValue(fvalue)
            elif ftype is bool:
                w = QtWidgets.QCheckBox()
                w.setChecked(fvalue)
            else:
                w = QtWidgets.QLineEdit()
                w.setText(fvalue)

            editors[fname] = w
            form.addRow(fname, w)

    def get_config_from_form(self, cfg, editors):
        '''updates a dataclass form a from'''
        for field in fields(cfg):
            fname = field.name
            ftype = field.type
            w = editors[fname]
            if ftype in (int, float):
                setattr(cfg, fname, ftype(w.value()))
            elif ftype == bool:
                setattr(cfg, fname, w.isChecked())
            else:
                setattr(cfg, fname, w.text())

    def make_PullVm_tab(self):
        self.PullVmTab = QtWidgets.QWidget()
        self.PullVmTab.layout = QtWidgets.QVBoxLayout()
        self.PullVmTab.layout.addWidget(QtWidgets.QLabel('Select sessions to add to processing queue'))
        PullVm_button = QtWidgets.QPushButton('Process')
        PullVm_button.clicked.connect(self.PullVm_button_callback)
        self.PullVmTab.layout.addWidget(PullVm_button)

        # add a form to edit config
        self.PullVm_config_editors = {}
        PullVm_Config_widget = QtWidgets.QWidget()
        self.PullVm_config_form = QtWidgets.QFormLayout()
        self.PullVm_config = PullVmConfig()
        self.create_config_form(self.PullVm_config_form, self.PullVm_config, self.PullVm_config_editors)
        PullVm_Config_widget.layout = self.PullVm_config_form
        apply_layout(PullVm_Config_widget)
        self.PullVmTab.layout.addWidget(PullVm_Config_widget)

        apply_layout(self.PullVmTab)
        self.tabs.addTab(self.PullVmTab, Tabs.PullVm.value)
        self.n_tabs += 1

    def PullVm_button_callback(self):
        cells_tag = self.tag_label.text()
        self.get_config_from_form(self.PullVm_config, self.PullVm_config_editors)
        kwargs = asdict(self.PullVm_config)
        overwrite = kwargs.pop('overwrite')
        for prefix in self.active_prefix:
            self.cprint('PullVm:', prefix, 'queued.')
            self.Q.run_job(Job(JobType.PullVM, (self.wdir, prefix, cells_tag, overwrite, kwargs)))

    def make_sima_tab(self):
        self.SIMATab = QtWidgets.QWidget()
        self.SIMATab.layout = QtWidgets.QVBoxLayout()
        self.SIMATab.layout.addWidget(QtWidgets.QLabel('Select sessions to add to processing queue'))
        SIMA_button = QtWidgets.QPushButton('Process')
        SIMA_button.clicked.connect(self.SIMA_button_callback)
        self.SIMATab.layout.addWidget(SIMA_button)

        # add a form to edit config
        self.SIMA_config_editors = {}
        SIMA_Config_widget = QtWidgets.QWidget()
        self.SIMA_config_form = QtWidgets.QFormLayout()
        self.SIMA_config = SIMAConfig()
        self.create_config_form(self.SIMA_config_form, self.SIMA_config, self.SIMA_config_editors)
        SIMA_Config_widget.layout = self.SIMA_config_form
        apply_layout(SIMA_Config_widget)
        self.SIMATab.layout.addWidget(SIMA_Config_widget)

        apply_layout(self.SIMATab)
        self.tabs.addTab(self.SIMATab, Tabs.SIMA.value)
        self.n_tabs += 1

    def SIMA_button_callback(self):
        self.get_config_from_form(self.SIMA_config, self.SIMA_config_editors)
        kwargs = asdict(self.SIMA_config)
        # convert chekcboxes to list of SIMA approaches
        apps = []
        for field in fields(self.SIMA_config):
            if field.type is bool:
                incl = kwargs.pop(field.name)
                if incl:
                    apps.append(field.name.replace('_', '-'))
        for prefix in self.active_prefix:
            self.Q.run_job(Job(JobType.SIMA, (self.wdir, prefix, apps, kwargs)))
            self.cprint('SIMA:', prefix, 'queued.')

    def make_pull_tab(self):
        self.PullTab = QtWidgets.QWidget()
        self.PullTab.layout = QtWidgets.QVBoxLayout()
        self.PullTab.layout.addWidget(QtWidgets.QLabel('Select sessions to add to processing queue'))
        button = QtWidgets.QPushButton('Process')
        button.clicked.connect(self.pull_button_callback)
        self.PullTab.layout.addWidget(button)

        # add selector for channel
        channels = ('All', 'Green', 'Red')
        labels = ('All available channels', 'Green-Ch2', 'Red-Ch1')
        self.Pull_channel_selector = ModeSelector(channels, labels, self, self.PullTab, groupname='Channel')

        # add selector for source
        self.Pull_source_selector = ModeSelector(tuple(Source), tuple(m.name for m in Source), self, self.PullTab,
                                                 groupname='Source')

        # add a form to edit config
        self.PullTab.layout.addWidget(QtWidgets.QLabel('Options'))
        self.Pull_config_editors = {}
        config_widget = QtWidgets.QWidget()
        self.Pull_config_form = QtWidgets.QFormLayout()
        self.Pull_config = PullConfig()
        self.create_config_form(self.Pull_config_form, self.Pull_config, self.Pull_config_editors)
        config_widget.layout = self.Pull_config_form
        apply_layout(config_widget)
        self.PullTab.layout.addWidget(config_widget)

        apply_layout(self.PullTab)
        self.tabs.addTab(self.PullTab, Tabs.PullROIs.value)
        self.n_tabs += 1

    def pull_button_callback(self):
        self.get_config_from_form(self.Pull_config, self.Pull_config_editors)
        cells_tag = self.tag_label.text()
        for prefix in self.active_prefix:
            self.Q.run_job(Job(JobType.PullSignals, (self.wdir, prefix, cells_tag, self.Pull_channel_selector.mode, False,
                                              self.Pull_config.SNR_weighted, self.Pull_source_selector.mode)))
            self.cprint('Pull signals:', prefix, 'queued.')

    def make_process_tab(self):
        self.ProcessTab = QtWidgets.QWidget()
        self.ProcessTab.layout = QtWidgets.QVBoxLayout()
        self.ProcessTab.layout.addWidget(QtWidgets.QLabel('Select sessions to add to processing queue'))
        button = QtWidgets.QPushButton('Process')
        button.clicked.connect(self.process_button_callback)
        self.ProcessTab.layout.addWidget(button)
        # add a form to edit config
        self.Process_config_editors = {}
        config_widget = QtWidgets.QWidget()
        self.Process_config_form = QtWidgets.QFormLayout()
        self.Process_config = ProcessConfig()
        self.create_config_form(self.Process_config_form, self.Process_config, self.Process_config_editors)
        config_widget.layout = self.Process_config_form
        apply_layout(config_widget)
        self.ProcessTab.layout.addWidget(config_widget)

        apply_layout(self.ProcessTab)
        self.tabs.addTab(self.ProcessTab, Tabs.ProcessCa.value)
        self.n_tabs += 1

    def process_button_callback(self):
        self.get_config_from_form(self.Process_config, self.Process_config_editors)
        cells_tag = self.tag_label.text()
        for prefix in self.active_prefix:
            # path, prefix, bsltype, exclude, sz_mode, peakdet, last_bg, invert, tag
            self.Q.run_job(Job(JobType.ProcessROIs, (self.wdir, prefix, cells_tag, self.Process_config)))
            self.cprint('Pull signals:', prefix, 'queued.')

    def cprint(self, *args):
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        self.console_widget.appendPlainText(' '.join([str(x) for x in (ts, *args)]))

    def make_realtime_tab(self):
        self.realTimeTab = QtWidgets.QWidget()
        self.realTimeTab.layout = QtWidgets.QVBoxLayout()

        # control buttons
        horizontal_layout = QtWidgets.QHBoxLayout()
        self.record_baseline_button = QtWidgets.QPushButton('Record baseline', )
        horizontal_layout.addWidget(self.record_baseline_button)
        # self.record_baseline_button.clicked.connect(self.record_baseline_callback)

        self.realTimeTab.layout.addLayout(horizontal_layout)

        # real time plot area
        self.realTimeTab.layout.addWidget(self.separator)
        horizontal_layout = QtWidgets.QHBoxLayout()
        vertical_layout = QtWidgets.QVBoxLayout()
        self.FigCanvasRT = SubplotsCanvas(nrows=1, sharex=True, figsize=self.config.plot_canvas_size)
        self.setup_live_fig()
        toolbar = NavigationToolbar2QT(self.FigCanvasRT, self)
        vertical_layout.addWidget(toolbar)
        vertical_layout.addWidget(self.FigCanvasRT)
        horizontal_layout.addLayout(vertical_layout)

        self.realTimeTab.layout.addLayout(horizontal_layout)

        self.realTimeTab.setLayout(self.realTimeTab.layout)
        self.tabs.addTab(self.realTimeTab, "RealTime")

    def on_tab_changed(self, index):
        self.active_tab = Tabs(self.tabs.tabText(index))

    def save_gui_state(self):
        for fieldname in self.saved_fields:
            if fieldname in self.selector_fields:
                v = getattr(self, fieldname).currentText()  # dropdowns
            else:
                v = getattr(self, fieldname).text()  # simple entry
            self.settings_dict[fieldname] = v
        with open(self.settings_filename, 'w') as f:
            json.dump(self.settings_dict, f)

    def select_path_callback(self):
        start_dir = self.wdir
        if not os.path.exists(str(start_dir)):
            start_dir = None
        user_input = QtWidgets.QFileDialog.getExistingDirectory(self.filelist_widget, 'Select Folder', dir=start_dir)
        if user_input:
            self.wdir = user_input
            self.path_label.setText(self.wdir)
            os.chdir(self.wdir)
            self.assets.update(self.wdir, reverse=False)
            self.current_selected_i = None
            self.prefix_list.clear()
            self.prefix_list.addItems(self.assets.get_prefixes())

    def set_prefix(self):
        self.active_prefix = [x.text() for x in self.prefix_list.selectedItems()]
        self.prefix_label.setText(self.active_prefix[0])
        if self.active_tab == Tabs.ROI:
            self.update_editor_callback()

    def setup_live_fig(self):
        ca = self.FigCanvasRT.ax
        ca.spines['right'].set_visible(False)
        ca.spines['top'].set_visible(False)
        duration_seconds = self.config.LiveLineDuration
        data_len = int(duration_seconds * self.config.fs / self.config.LiveLineDecimate)
        xdata = numpy.linspace(-duration_seconds, 0, data_len)
        ydata = numpy.zeros(data_len)
        self.live_line = ca.plot(xdata, ydata, color='black')[0]
        ca.set_ylim(-self.config.PlotYScale, self.config.PlotYScale)

    def save_gui_state(self):
        for fieldname in self.saved_fields:
            if fieldname in self.selector_fields:
                v = getattr(self, fieldname).currentText()  # dropdowns
            elif fieldname in self.attr_fields:
                v = getattr(self, fieldname)  # attrs
            else:
                v = getattr(self, fieldname).text()  # simple entry
            self.settings_dict[fieldname] = v
        with open(self.settings_filename, 'w') as f:
            json.dump(self.settings_dict, f)

    def load_settings_file(self):
        if os.path.exists(self.settings_filename):
            with open(self.settings_filename) as f:
                self.settings_dict = json.load(f)
            print(self.settings_dict)
        else:
            self.settings_dict = {}

    def load_gui_sate(self):
        for fieldname in self.saved_fields:
            if fieldname in self.settings_dict:
                if fieldname in self.selector_fields:
                    getattr(self, fieldname).setCurrentText(self.settings_dict[fieldname])
                elif fieldname in self.attr_fields:
                    setattr(self, fieldname, self.settings_dict[fieldname])
                else:
                    getattr(self, fieldname).setText(self.settings_dict[fieldname])

    def closeEvent(self, event):
        self.save_gui_state()
        event.accept()

    def update_trace(self, data):
        decdat = decimate(data, self.config.LiveLineDecimate)
        _, ydata = self.live_line.get_data()
        ydata[:-len(decdat)] = ydata[len(decdat):]
        ydata[-len(decdat):] = decdat
        self.live_line.set_ydata(ydata)
        self.FigCanvasRT.draw()

    def poll_result(self):
        if self.Q is not None:
            # empties the q and calls the handler function with each item. called by a timer, non_blocking.
            self.Q.poll_result(self._handle_item_from_worker)

    def _handle_item_from_worker(self, item):
        worker_name, prefix = item
        self.cprint(worker_name + ':', prefix, 'done.')


class ModeSelector:

    def __init__(self, values: tuple[str], labels: tuple[str], parent, widget, default_index=0, groupname=None):
        '''
        Create a buttongroup, have a .mode attribute that reflects state
        :param values: the values self.mode will hold
        :param labels: the labels displayed on the buttons
        :param parent: call with the GUI's self
        :param widget: the widget where the buttons will be added (needs to have a layout)
        :param default_index: the option that will be active on creation
        :param groupname: a label displayed above the group
        '''
        modegroup = QtWidgets.QButtonGroup(parent)
        modegroup.setExclusive(True)
        self._button_to_mode = {}
        if groupname is not None:
            widget.layout.addWidget(QtWidgets.QLabel(groupname))
        for mode, label in zip(values, labels):
            btn = QtWidgets.QRadioButton(label, parent)
            widget.layout.addWidget(btn)
            modegroup.addButton(btn)
            self._button_to_mode[btn] = mode
        modegroup.buttonClicked.connect(self._on_pull_mode_changed)
        list(self._button_to_mode.keys())[default_index].setChecked(True)
        self.mode = values[0]

    def _on_pull_mode_changed(self, button):
        self.mode = self._button_to_mode[button]


class StdRedirector:
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.appendPlainText(string)

    def flush(self):
        pass


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
    launch_GUI()
