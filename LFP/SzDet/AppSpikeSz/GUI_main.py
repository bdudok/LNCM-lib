import os
import sys
from pyqtgraph import Qt
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.Qt.QtWidgets import (QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox, QListWidget,
QAbstractItemView)

class GUI_main(QtWidgets.QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.setWindowTitle(f'Detect spikes and seizures')
        # self.setGeometry(10, 30, 1600, 900) # Left, top, width, height.
        self.app = app

        #variables
        self.wdir = None
        self.separator = QLabel("<hr>")

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
        select_path_button = QPushButton('Open', )
        horizontal_layout.addWidget(select_path_button)
        select_path_button.clicked.connect(self.load_file_callback)

        vbox.addLayout(horizontal_layout)

        vbox.addWidget(self.separator)
        #raw

        return groupbox

    def select_path_callback(self):
        #get a folder
        self.wdir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.path_label.setText(self.wdir)

        #get prefix list
        suffix = '.ephys'
        prefix_list = [fn[:-len(suffix)] for fn in os.listdir(self.wdir) if fn.endswith(suffix)]

        #set list widget
        self.prefix_list.clear()
        self.prefix_list.addItems(prefix_list)

    def load_file_callback(self):
        prefix = self.prefix_list.selectedItems()[0].text()
        print(prefix)


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
