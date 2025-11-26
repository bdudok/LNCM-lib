import matplotlib
from pathlib import Path
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
from enum import Enum
plt.rcParams['font.size'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
import numpy
import os
import json
import sys


from pyqtgraph import Qt
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from Proc2P.utils import logger, lprint

from Proc2P.Analysis.AssetFinder import AssetFinder

'''
Gui for viewing and processing 2P data.
This file just defines widgets, all functions are imported from the analysis classes or the legacy (Tk) BatchGUI app.
'''
class GuiConfig:
    settings_filename = os.path.join(Path.home(), 'BatchGUI-settings.json') #for gui setting permanence
    MainWindowGeometry = (30, 60, 1200, 800)
    TextWidgetHeight = 60
    ConsoleWidgetHeight = 300
    ConsoleWidgetWidth = 900
    plot_canvas_size = (9, 3) #inches,

class State(Enum):
    SETUP = 0
    EDITING_ROI = 1
    VIEWING_TRACE = 2

def apply_layout(widget):
    widget.setLayout(widget.layout)


class GUI_main(QtWidgets.QMainWindow):
    __name__= 'BatchGUI'
    def __init__(self, app, title='BatchGUI', ):
        super().__init__()
        self.setWindowTitle(title)
        self.app = app
        self.config = GuiConfig()

        self.wdir = '.'
        self.saved_fields = ('wdir',) #GUI values that are saved in a user-specific config file
        self.selector_fields = () #list those of saved_fields which are single-select widgets
        self.attr_fields = ('wdir', )#list those of saved_fields where we store the attribute value
        #anything not in selector or attr fields is using the .text() of the corresponding widget
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
        self.make_settings_tab()
        n_tabs = 1

        self.tabs.resize(int(100 * n_tabs), 100)

        # add tabs to table
        self.table_widget.layout.addWidget(self.tabs)
        apply_layout(self.table_widget)

        #"console" for printing responses
        self.console_widget = QtWidgets.QTextEdit(self)
        self.make_console_widget()

        #add everything to main window layout
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
        # self.connect_sockets() # not used
        self.show()

    def make_session_bar(self):
        self.session_bar.setFixedHeight(self.config.TextWidgetHeight)
        self.session_bar.layout = QtWidgets.QHBoxLayout(self.session_bar)
        select_path_button = QtWidgets.QPushButton('Select folder', )
        self.session_bar.layout.addWidget(select_path_button)
        select_path_button.clicked.connect(self.select_path_callback)
        self.session_bar.layout.addWidget(self.separator)
        self.path_label = QtWidgets.QLabel(str(self.settings_dict.get('wdir')))

    def make_console_widget(self):
        self.console_widget.setText('Console...')
        self.console_widget.setFixedHeight(self.config.ConsoleWidgetHeight)
        # self.console_widget.setFixedWidth(self.config.ConsoleWidgetWidth)


    def make_filelist_widget(self):
        self.filelist_widget.layout = QtWidgets.QVBoxLayout()
        self.filelist_widget.layout.addWidget(QtWidgets.QLabel('Prefix list'))
        self.prefix_list = QtWidgets.QListWidget(self)
        self.prefix_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.prefix_list.itemSelectionChanged.connect(self.set_prefix)
        self.filelist_widget.layout.addWidget(self.prefix_list)

        #TODO add list
        apply_layout(self.filelist_widget)

    def make_settings_tab(self):
        self.settingsTab = QtWidgets.QWidget()
        self.settingsTab.layout = QtWidgets.QVBoxLayout()
        self.tabs.addTab(self.settingsTab, "Settings")


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

    def make_detector_tab(self):
        self.detectorTab = QtWidgets.QWidget()
        self.detectorTab.layout = QtWidgets.QVBoxLayout()
        self.tabs.addTab(self.detectorTab, "Detectors")


    def list_clicked(self):
        pass
        # self.log_list

    def save_gui_state(self):
        for fieldname in self.saved_fields:
            if fieldname in self.selector_fields:
                v = getattr(self, fieldname).currentText()  # dropdowns
            else:
                v = getattr(self, fieldname).text()  # simple entry
            self.settings_dict[fieldname] = v
        with open(self.settings_filename, 'w') as f:
            json.dump(self.settings_dict, f)

    def set_state(self, state):
        pass

    def select_path_callback(self):
        start_dir = self.wdir
        if not os.path.exists(str(start_dir)):
            start_dir = None
        user_input = QtWidgets.QFileDialog.getExistingDirectory(self.filelist_widget, 'Select Folder', directory=start_dir)
        if user_input:
            self.wdir = user_input
            self.path_label.setText(self.wdir)
            os.chdir(self.wdir)
            self.assets.update(self.wdir)
            self.current_selected_i = None
            self.prefix_list.clear()
            self.prefix_list.addItems(self.assets.get_prefixes())

    def set_prefix(self):
        self.active_prefix = [x.text() for x in self.prefix_list.selectedItems()]

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
                v = getattr(self, fieldname) #attrs
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

    class StdRedirector:
        def __init__(self, text_widget):
            self.text_space = text_widget

        def write(self, string):
            #TODO update for qt
            self.text_space.config(state=NORMAL)
            self.text_space.insert("end", string)
            self.text_space.see("end")
            self.text_space.config(state=DISABLED)

    def update_trace(self, data):
        decdat = decimate(data, self.config.LiveLineDecimate)
        _, ydata = self.live_line.get_data()
        ydata[:-len(decdat)] = ydata[len(decdat):]
        ydata[-len(decdat):] = decdat
        self.live_line.set_ydata(ydata)
        self.FigCanvasRT.draw()


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
