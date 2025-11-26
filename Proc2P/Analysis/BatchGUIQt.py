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

class GUI_main(QtWidgets.QMainWindow):
    __name__= 'BatchGUI'
    def __init__(self, app, title='BatchGUI', ):
        super().__init__()
        self.setWindowTitle(title)
        self.app = app
        self.config = GuiConfig()

        self.saved_fields = ()
        self.selector_fields = ()
        self.settings_filename = self.config.settings_filename
        self.stripchars = "'+. *?~!@#$%^&*(){}:[]><,/" + '"' + '\\'
        if os.path.exists(self.settings_filename):
            with open(self.settings_filename) as f:
                self.settings_dict = json.load(f)
            print(self.settings_dict)
        else:
            self.settings_dict = {}
        self.separator = QtWidgets.QLabel("<hr>")

        self.state = State.SETUP

        # session bar - on top, displays active path, session, ROI tag
        self.session_bar = QtWidgets.QWidget(self)
        self.make_session_bar()

        # tabs
        self.table_widget = QtWidgets.QWidget(self)
        self.table_widget.layout = QtWidgets.QVBoxLayout(self.table_widget)
        self.tabs = QtWidgets.QTabWidget()

        self.make_settings_tab()
        n_tabs = 1

        self.tabs.resize(int(100 * n_tabs), 100)

        # add tabs to table
        self.table_widget.layout.addWidget(self.tabs)
        self.table_widget.setLayout(self.table_widget.layout)

        #"console" for printing responses
        self.console_widget = QtWidgets.QTextEdit(self)
        self.make_console_widget()

        #add everything to main window layout
        self.main_widget =QtWidgets.QWidget(self)  # central widget
        self.main_widget.layout = QtWidgets.QVBoxLayout(self.table_widget)
        self.main_widget.layout.addWidget(self.session_bar)
        self.main_widget.layout.addWidget(self.separator)
        self.main_widget.layout.addWidget(self.table_widget)
        self.main_widget.layout.addWidget(self.separator)
        self.main_widget.layout.addWidget(self.console_widget)

        self.setCentralWidget(self.main_widget)
        # update fields from file
        for fieldname in self.saved_fields:
            if fieldname in self.settings_dict:
                if fieldname in self.selector_fields:
                    getattr(self, fieldname).setCurrentText(self.settings_dict[fieldname])
                else:
                    getattr(self, fieldname).setText(self.settings_dict[fieldname])
        self.setGeometry(*self.config.MainWindowGeometry)  # Left, top, width, height.
        # self.connect_sockets() # not used
        self.show()

    def make_session_bar(self):
        self.session_bar.layout = QtWidgets.QHBoxLayout(self.session_bar)
        hbox = self.session_bar.layout
        select_path_button = QtWidgets.QPushButton('Select folder', )
        select_path_button.setFixedHeight(self.config.TextWidgetHeight)
        hbox.addWidget(select_path_button)
        select_path_button.clicked.connect(self.select_path_callback)
        hbox.addWidget(self.separator)

    def make_console_widget(self):
        self.console_widget.setText('Console...')
        self.console_widget.setFixedHeight(self.config.ConsoleWidgetHeight)
        self.console_widget.setFixedWidth(self.config.ConsoleWidgetWidth)


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
        pass

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
