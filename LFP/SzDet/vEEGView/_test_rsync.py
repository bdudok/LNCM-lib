import os
import json

#plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
import numpy

#GUI
import sys
# from pyqtgraph import Qt
# from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
# from pyqtgraph.Qt.QtWidgets import (QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox, QListWidget,
# QAbstractItemView, QLineEdit, QCheckBox)

#Workflow Specific functions
from LFP.SpikeDet import SpikesPower
from LFP.SzDet import InstRate

#Readers
from Proc2P.Legacy.Loaders import load_ephys
from Proc2P.Bruker import LoadEphys
from LFP.Pinnacle import ReadEDF
from Proc2P.Treadmill import rsync

path = 'D:\Shares\Data\_Processed\EEG/test/'
eeg_fn = 'VKPV1__0032_2024-01-11_11_03_58_TS_2024-01-11_11_03_58_export.edf'
vid_fn = 'VKPV1_s2-2024-01-11T12-28-29'

EEG = ReadEDF.EDF(path, eeg_fn,)
ttl = EEG.get_TTL()

vttl = numpy.load(path + vid_fn + '.npy')[:, -1]

ml = min(len(vttl), len(ttl))

plt.scatter(vttl[:ml], ttl[:ml])

align = rsync.Rsync_aligner(vttl, ttl)