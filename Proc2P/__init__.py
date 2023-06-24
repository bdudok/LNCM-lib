import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats

from .Legacy.MyColorLib import get_color

def read_excel(*args, **kwargs):
    return pd.read_excel(*args, **kwargs, engine='openpyxl')

