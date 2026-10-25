import os
import sys
import matplotlib
mpl_backend = 'TkAgg'
# TkAgg can't load without a display (e.g. in a container or CI); keep matplotlib's default there
has_display = sys.platform in ('win32', 'darwin') or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
# don't call matplotlib.get_backend() first: it selects a backend if none is set, which can import a Qt binding
if has_display:
    matplotlib.use(mpl_backend)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy
np = numpy
from matplotlib import pyplot as plt
import pandas
import datetime
pd = pandas
from scipy import stats


from BaserowAPI.BaserowRequests import GetSessions
from Proc2P.utils import read_excel, lprint, norm, ewma, completed_list, ts
from PlotTools.MyColorLib import get_color
from PlotTools import *


plt.rcParams['font.sans-serif'] = 'Arial'

micro = u"\N{GREEK SMALL LETTER MU}"
delta = u"\N{GREEK CAPITAL LETTER DELTA}"

