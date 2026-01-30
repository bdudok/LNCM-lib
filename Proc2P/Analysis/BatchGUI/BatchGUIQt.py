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
from Proc2P.Analysis.AssetFinder import AssetFinder, get_processed_tags
# from Proc2P.Analysis.BatchGUI.RoiEditorQt import GUI_main as RoiEditorGUI
from Proc2P.Analysis.BatchGUI.RoiEditorQt import launch_in_subprocess as RoiEditorGUI
from Proc2P.Analysis.BatchGUI.SessionGUIQt import launch_in_subprocess as SessionGUI
from LFP.SzDet.AppSpikeSz.GUI_SpikeSzDet import launch_in_subprocess as SzProcGUI
from LFP.SzDet.AppSpikeSz.SzViewGUI import launch_in_subprocess as SzViewGUI

from Proc2P.Analysis.BatchGUI.QueueManager import Job, JobType
from Proc2P.Analysis.GEVIReg.Register import RegConfig
from Proc2P.Analysis.AnalysisClasses.NormalizeVm import PullVmConfig
from Proc2P.Analysis.RoiEditor import SIMAConfig
from Proc2P.Analysis.PullSignals import PullConfig
from Proc2P.Bruker.LoadRegistered import Source
from Proc2P.Analysis.CaTrace import ProcessConfig
from Proc2P.Analysis.Ripples import Ripples, RippleConfig
import tifffile
import cv2

'''
Gui for viewing and processing 2P data.
This file just defines widgets, all functions are imported from the analysis classes or the legacy (Tk) BatchGUI app.
'''


class Tabs(Enum):
    Preview = 'Preview'
    SessionGUI = 'View Trace'
    ROI = 'ROI Editor'
    GEVIReg = 'GEVIReg'
    PullROIs = 'Pull signals'
    ProcessCa = 'Process ROIs'
    PullVm = 'Pull Vm'
    SIMA = 'SIMA'
    LFP = 'LFP'


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
        self.active_tab = Tabs.Preview
        self.make_preview_tab()
        self.make_gevireg_tab()
        self.make_sima_tab()
        self.make_pull_tab()
        self.make_process_tab()
        self.make_editor_tab()
        self.make_PullVm_tab()
        self.make_LFP_tab()

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

        open_button = QtWidgets.QPushButton('View')
        open_button.clicked.connect(self.open_viewer_callback)
        self.session_bar.layout.addWidget(open_button)

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

    def make_preview_tab(self):
        self.PreviewTab = QtWidgets.QWidget()
        self.PreviewTab.layout = QtWidgets.QVBoxLayout()
        self.preview_list = QtWidgets.QComboBox(self)
        self.preview_list.currentIndexChanged.connect(self.update_preview_file)
        self.PreviewTab.layout.addWidget(self.preview_list)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.preview_label.setMargin(0)
        self.preview_label.setContentsMargins(0, 0, 0, 0)
        self.preview_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.PreviewTab.layout.addWidget(self.preview_label)

        self.gamma = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma.setMinimum(1)
        self.gamma.setMaximum(30)
        self.gamma.setValue(15)
        self.gamma.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.PreviewTab.layout.addWidget(self.gamma)
        self.gamma.valueChanged.connect(self.update_preview_file)

        apply_layout(self.PreviewTab)
        self.tabs.addTab(self.PreviewTab, Tabs.Preview.value)
        self.n_tabs += 1

    def update_preview_session(self):
        wdir = self.wdir
        prefix = self.active_prefix[0]
        opPath = os.path.join(wdir, prefix)
        self.preview_list.clear()
        self.preview_w = self.preview_label.width()
        self.preview_h = self.preview_label.height()
        item_index = 0
        def_index = 0
        for fn in os.listdir(opPath):
            if prefix in fn and fn.endswith('.tif'):
                self.preview_list.addItem(fn[len(prefix) + 1:-4], fn)
                if fn.endswith('_preview.tif'):
                    def_index = item_index
                item_index += 1
        self.preview_list.setCurrentIndex(def_index)
        self.update_preview_file()

    def update_preview_file(self):
        items = (self.wdir, self.active_prefix[0], self.preview_list.currentData())
        if None in items:
            return 0
        preview_fn = os.path.join(*items)
        img = tifffile.imread(preview_fn)
        gamma = 15.0 / (self.gamma.value() + 1)
        lut = numpy.array([((i / 255.0) ** gamma) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
        img = cv2.LUT(img.squeeze(), lut)
        if len(img.shape) == 2:
            h, w = img.shape
            ch = 1
        else:
            h, w, ch = img.shape
        qimg = QtGui.QImage(img, w, h, ch * w, QtGui.QImage.Format_RGB888)

        # Resize while preserving aspect ratio to fit QLabel width
        aspect_ratio = h / w
        rescale_factor = min(self.preview_w / w, self.preview_h / h)
        target_width = min(self.preview_w - 5, int(w * rescale_factor))
        target_height = min(self.preview_h - 5, int(target_width * aspect_ratio))

        qimg_resized = qimg.scaled(target_width, target_height,
                                   QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(qimg_resized))

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

    def open_viewer_callback(self):
        prefix = self.active_prefix[0]
        tag = self.tag_label.text()
        traces = get_processed_tags(self.wdir, prefix)
        traces = ', '.join(set([x[0] for x in traces]))
        if tag not in traces:
            self.cprint(f'No processed data found for "{tag}". Available: {traces}')
        else:
            SessionGUI(self.wdir, self.active_prefix[0], tag)

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
            self.Q.run_job(
                Job(JobType.PullSignals, (self.wdir, prefix, cells_tag, self.Pull_channel_selector.mode, False,
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
            self.cprint('Process ROIs:', prefix, 'queued.')

    def make_LFP_tab(self):
        self.LFPTab = QtWidgets.QWidget()
        self.LFPTab.layout = QtWidgets.QVBoxLayout()
        self.LFPTab.layout.addWidget(QtWidgets.QLabel('Select sessions to process for ripple detection.'))
        # add a form to edit config
        self.Ripple_config_editors = {}
        config_widget = QtWidgets.QWidget()
        self.Ripple_config_form = QtWidgets.QFormLayout()
        self.Ripple_config = RippleConfig()
        self.create_config_form(self.Ripple_config_form, self.Ripple_config, self.Ripple_config_editors)
        config_widget.layout = self.Ripple_config_form
        apply_layout(config_widget)
        self.LFPTab.layout.addWidget(config_widget)

        button = QtWidgets.QPushButton('Process')
        button.clicked.connect(self.process_ripples_callback)
        self.LFPTab.layout.addWidget(button)

        self.ripples = None
        self.LFPTab.layout.addWidget(QtWidgets.QLabel('Show ripple selector for current session:'))
        buttons = {
            'Display Ripples': self.display_ripples_callback,
            'Detect recursively': self.ripples_rec_callback,
            'Save Ripples with tag': self.ripples_save_callback,
            'Save ripple times as .xlsx': self.ripples_export_callback}
        for button_name, callback in buttons.items():
            button = QtWidgets.QPushButton(button_name)
            button.clicked.connect(callback)
            self.LFPTab.layout.addWidget(button)

        self.LFPTab.layout.addWidget(QtWidgets.QLabel('Run seizure analysis tools:'))
        button = QtWidgets.QPushButton('Detect spikes')
        button.clicked.connect(self.run_spikeszdet_callback)
        self.LFPTab.layout.addWidget(button)
        button = QtWidgets.QPushButton('Curate seizures')
        button.clicked.connect(self.run_curate_sz_callback)
        self.LFPTab.layout.addWidget(button)

        apply_layout(self.LFPTab)
        self.tabs.addTab(self.LFPTab, Tabs.LFP.value)
        self.n_tabs += 1

    def process_ripples_callback(self):
        self.get_config_from_form(self.Ripple_config, self.Ripple_config_editors)
        kwargs = asdict(self.Ripple_config)
        for prefix in self.active_prefix:
            self.cprint('Ripples:', prefix, 'queued.')
            self.Q.run_job(Job(JobType.Ripples, (self.wdir, prefix, kwargs)))

    def load_ripple(self):
        self.get_config_from_form(self.Ripple_config, self.Ripple_config_editors)
        if (self.ripples is not None) and self.active_ripple_setting == json.dumps(asdict(self.Ripple_config)):
            return 0
        self.active_ripple_setting = json.dumps(asdict(self.Ripple_config))
        cfg = {}
        for key in ('tr1', 'tr2', 'y_scale'):
            cfg[key] = getattr(self.Ripple_config, key)
        self.ripples = Ripples(self.wdir, self.active_prefix[0], config=cfg,
                               force=self.Ripple_config.overwrite_existing,
                               ephys_channel=self.Ripple_config.channel, tag=self.Ripple_config.tag)

    def display_ripples_callback(self):
        self.load_ripple()
        self.ripples.enum_ripples(no_save=True)

    def ripples_rec_callback(self):
        self.load_ripple()
        self.ripples.rec_enum_ripples(exclude_spikes=self.Ripple_config.exclude_spikes)

    def ripples_save_callback(self):
        self.ripples.save_ripples(tag=self.Ripple_config.tag)

    def ripples_export_callback(self):
        self.ripples.export_ripple_times()

    def run_spikeszdet_callback(self):
        SzProcGUI(self.wdir, 'LNCM', 'hippocampus', self.active_prefix[0])

    def run_curate_sz_callback(self):
        SzViewGUI('LNCM')

    def cprint(self, *args):
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        self.console_widget.appendPlainText(' '.join([str(x) for x in (ts, *args)]))

    def on_tab_changed(self, index):
        self.active_tab = Tabs(self.tabs.tabText(index))
        if self.active_tab == Tabs.Preview:
            self.update_preview_session()
        elif self.active_tab == Tabs.ROI:
            self.update_editor_callback()

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
        self.ripples = None
        if self.active_tab == Tabs.ROI:
            self.update_editor_callback()
        elif self.active_tab == Tabs.Preview:
            self.update_preview_session()

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
        self._buffer = ""

    def write(self, string):
        # Accumulate text until we see a newline
        self._buffer += string
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            # appendPlainText adds a new block (line) in QPlainTextEdit
            self.text_space.appendPlainText(line)

    def flush(self):
        if self._buffer:
            self.text_space.appendPlainText(self._buffer)
            self._buffer = ""


def launch_GUI(*args, **kwargs):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app, *args, **kwargs)
    sys.exit(app.exec())
    print('GUI opened.')


if __name__ == '__main__':
    launch_GUI()
