import matplotlib
if not matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('TkAgg')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy
np = numpy
from matplotlib import pyplot as plt
import pandas
pd = pandas
from scipy import stats

from .Legacy.MyColorLib import get_color

from BaserowAPI.BaserowRequests import GetSessions
from Proc2P.utils import read_excel, lprint
