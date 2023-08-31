import os
import pandas

from matplotlib import pyplot as plt
fn = 'Animal_2023-08-25_movie_063-000_Cycle00001_VoltageRecording_001.csv'
os.chdir('D:\Shares\Data\_RawData\Bruker/testing/20230825_movie_for_pupil/Animal_2023-08-25_movie_063-000')
d = pandas.read_csv(fn)
v = d[' Input 1']
plt.plot(v.values)
plt.show()