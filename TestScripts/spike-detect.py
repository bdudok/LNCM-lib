import os

from Proc2P import *
from LFP.SpikeDet import SpikesPower
from LFP.SzDet import InstRate
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


d = SpikesPower.Detect(r.trace, fs=10000, )
# HFO_duration, left_ips, right_ips, env = d.get_spikes(tr1=2.0, tr2=3.0)
t = d.get_spikes(tr1=2.0, tr2=3.0)
# print(len(t))

# sz_burden, sz_times = InstRate.SpikeTrace(t, framesize=64)
# X = numpy.arange(0, len(sz_burden)*64*2, 64*2)
# plt.plot(d.trace)
# plt.plot(X, sz_burden)