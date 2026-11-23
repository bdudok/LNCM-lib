'''Window and taskbar icon of the lab GUIs: a short label on a colored square.'''
import sys

from PySide6 import QtCore, QtGui

FONT_PX = {1: 170, 2: 125, 3: 95}  # label length: font size on the 256 px icon


def set_app_icon(app, label, color):
    '''
    Call right after creating the QApplication, before any window.
    :param label: 1-3 characters, e.g. 'MC'
    :param color: background color, e.g. '#4604D2'
    '''
    if sys.platform == 'win32':
        # own taskbar entry and icon, instead of grouping with python.exe
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f'LNCM.{label}')
    pixmap = QtGui.QPixmap(256, 256)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(color))
    painter.drawRoundedRect(8, 8, 240, 240, 44, 44)
    font = QtGui.QFont('Arial')
    font.setBold(True)
    font.setPixelSize(FONT_PX[len(label)])
    painter.setFont(font)
    painter.setPen(QtGui.QColor('white'))
    painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, label)
    painter.end()
    app.setWindowIcon(QtGui.QIcon(pixmap))
