#GUI
import sys, os
from PyQt5 import QtGui, QtWidgets, QtCore
from SzViewGUI import GUI_main

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
