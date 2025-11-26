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

from Proc2P.utils import logger, lprint
from Proc2P.Analysis.BatchGUI.Config import *
from Proc2P.Analysis.BatchGUI.utils import *


'''
Gui for manually drawing ROIs and editing ROI sets.
'''

class GUI_main(QtWidgets.QMainWindow):
    __name__= 'ROIEditor'
    def __init__(self, app, title='Editor', ):
        super().__init__()
        self.setWindowTitle(title)
        self.app = app
        self.config = GuiConfig()

        self.state = State.SETUP



def launch_GUI(*args, **kwargs):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app, *args, **kwargs)
    sys.exit(app.exec())


if __name__ == '__main__':
    launch_GUI()
