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
eeg_fn = 'VKPV1_s2_0033_2024-01-11_12_28_30_TS_2024-01-11_12_28_30_export.edf'
vid_fn = 'VKPV1_s2-2024-01-11T12-28-29'

EEG = ReadEDF.EDF(path, eeg_fn,)
ttl = EEG.get_TTL()

vttl = numpy.load(path + vid_fn + '.npy')[:, -1]
vttl = vttl[1:] - vttl[0] #does not contain actual frame numbers but since live view started :S

ml = min(len(vttl), len(ttl))

plt.scatter(vttl[:ml], ttl[:ml])

align = rsync.Rsync_aligner(vttl, ttl)

eegdur = len(EEG.trace) / EEG.fs
xunit = 60
align_fps = align.units_B
frametimes = align.B_to_A(numpy.arange(vttl[-1]+numpy.diff(vttl).max())) / align_fps
ft_bounds = frametimes[numpy.logical_not(numpy.isnan(frametimes))]
fig, ca = plt.subplots()
ca.plot((0, eegdur/xunit), (1,1), color='black')
ca.scatter((0, eegdur/xunit), (1,1), marker='|', color='black')
ca.scatter(ttl/xunit, numpy.ones(len(ttl)) *0.5, marker='|', c=numpy.isnan(align.cor_times_A))
ca.plot((ft_bounds[0]/xunit, ft_bounds[-1]/xunit), (1,1), color='green', linewidth=2)
ca.plot((vttl[0]/(align_fps*xunit), vttl[-1]/(align_fps*xunit)), (0,0), color='grey')
ca.set_ylim(-0.1, 1.1)

