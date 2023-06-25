import sys
from pyqtgraph import Qt
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

class GUI_main(QtWidgets.QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.setWindowTitle(f'Detect spikes and seizures')
        # self.setGeometry(10, 30, 1600, 900) # Left, top, width, height.
        self.app = app

        #variables
        self.wdir = None
        self.separator = QtWidgets.QLabel("<hr>")

        # self.filelist_groupbox = QtWidgets.QGroupBox('File list')
        # self.filelist_Vlayout = QtWidgets.QVBoxLayout(self.filelist_groupbox)
        # self.filelist_Hlayout_1 = QtWidgets.QHBoxLayout()

        #file selector dock
        # self.fileselector = self.FileSelectorDockWidget()


        #compose box
        # self.filelist_Vlayout.addLayout(self.filelist_Hlayout_1)
        # self.filelist_Vlayout.addWidget(self.separator)

        #add main layouts
        select_path_button = QtWidgets.QPushButton('Select folder',)
        # self.filelist_Hlayout_1.addWidget(self.select_path_button)
        select_path_button.clicked.connect(self.select_path)
        vertical_layout = QtWidgets.QVBoxLayout()
        # self.vertical_layout.addWidget(self.filelist_groupbox)

        vertical_layout.addWidget(select_path_button)
        self.setCentralWidget(QtWidgets.QWidget(self))
        self.centralWidget().setLayout(vertical_layout)
        # self.setLayout(vertical_layout)
        print(self.children())
        self.show()

    def select_path(self):
        self.wdir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        print(self.wdir)

    # def FileSelectorDockWidget(self):
    #     dockWidgetContents = QtWidgets.QWidget()
    #
    #     #select path
    #     self.select_path_button = QtWidgets.QPushButton('Select folder', )
    #     # self.filelist_Hlayout_1.addWidget(self.select_path_button)
    #     self.select_path_button.clicked.connect(self.select_path)
    #     dockWidgetContents.addWidget(self.select_path_button)
    #
    #     dockWidget = QtWidgets.QDockWidget(QtWidgets.tr("Dock Widget"), self)
    #     dockWidget.setAllowedAreas(Qt.LeftDockWidgetArea |
    #                                Qt.RightDockWidgetArea)
    #     dockWidget.setWidget(dockWidgetContents)
    #     QtWidgets.addDockWidget(Qt.LeftDockWidgetArea, dockWidget)
    #     return dockWidget


def launch_GUI():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QtGui.QFont()
    app.setFont(font)
    gui_main = GUI_main(app)
    sys.exit(app.exec())

if __name__ == '__main__':
    launch_GUI()
