from pyqtgraph.Qt import QtWidgets

# class Path_select_dialog(QtWidgets.QDialog):
#     def __init__(self, parent=None):
#         super(QtWidgets.QDialog, self).__init__(parent)
#         self.setWindowTitle('Select working directory')
#         folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
#

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