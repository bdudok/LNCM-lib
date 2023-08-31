#GUI
import sys
from pyqtgraph import Qt
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

'''
basic example, add widgets to this
'''
class GUI_main(QtWidgets.QMainWindow):
    def __init__(self, app, title='Main'):
        super().__init__()
        self.setWindowTitle(title)
        self.app = app

        #central widget
        centralwidget = QtWidgets.QWidget(self)
        horizontal_layout = QtWidgets.QHBoxLayout()

        #add widgets


        self.setCentralWidget(centralwidget)
        self.centralWidget().setLayout(horizontal_layout)
        self.show()

def launch_GUI(*args, **kwargs):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app, *args, **kwargs)
    sys.exit(app.exec())