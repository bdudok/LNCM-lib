import os
import pandas

from matplotlib import pyplot as plt
fn = 'PVTot1_2023-08-28_opto_006-001_Cycle00001_VoltageRecording_001.csv'
os.chdir('D:\Shares\Data\_RawData\Bruker/testing/20230829_opto_intensity_series\PVTot1_2023-08-28_opto_006-001/')
d = pandas.read_csv(fn)
v = d[' Input 1']
plt.plot(v.values)
plt.show()