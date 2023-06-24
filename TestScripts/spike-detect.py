import os

from Proc2P import *
from LFP.SpikeDet import SpikesPower
import pandas

#open example data
from Proc2P.Legacy.Loaders import load_ephys
data_path = 'D:\Shares\Data\old_2P\Sncg-IHK/'
fn = data_path + '_analysis/ImagingSessions-SNCG-IHK-20230615.csv'
session_info = pandas.read_csv(fn)
prefix = session_info.iloc[0]['SessionTag']
os.chdir(data_path)
r = load_ephys(prefix)
y = r.trace


d = SpikesPower.Detect(r.trace, fs=10000)