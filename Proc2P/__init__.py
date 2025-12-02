import matplotlib
mpl_backend = 'TkAgg'
if not matplotlib.get_backend() == mpl_backend:
    matplotlib.use(mpl_backend)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy
np = numpy
from matplotlib import pyplot as plt
import pandas
import datetime
pd = pandas
from scipy import stats

# from .Legacy.MyColorLib import get_color

from BaserowAPI.BaserowRequests import GetSessions
from Proc2P.utils import read_excel, lprint, norm, ewma, completed_list, ts
from PlotTools.MyColorLib import get_color
from PlotTools import *


plt.rcParams['font.sans-serif'] = 'Arial'

micro = u"\N{GREEK SMALL LETTER MU}"
delta = u"\N{GREEK CAPITAL LETTER DELTA}"

