from pyqtgraph.Qt import QtWidgets

class Path_select_dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(QtWidgets.QDialog, self).__init__(parent)
        self.setWindowTitle('Select working directory')
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')