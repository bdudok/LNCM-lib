import os
from matplotlib import pyplot as plt
from TreadmillRead import data_import, session_plot, Treadmill

drive = '/Users/u247640/Library/CloudStorage/OneDrive-BaylorCollegeofMedicine/_RawData/'

path = drive + 'Treadmill-test/'
prefix = 'Test_move_full_laps-2023-04-17-184123'
# prefix = 'Test_move_1m-2023-04-17-183859'

# flist = os.listdir(path)
# d = data_import.Session(path+prefix + '.txt')
#
# d.analog = {}
# for f in flist:
#     if prefix in f and f.endswith('.pca'):
#         tag = (f.split('_')[-1].split('.')[0])
#         d.analog[tag] = data_import.load_analog_data(path + f)
#
# pos = d.analog['pos']
#
# # plt.scatter(pos[:, 0], pos[:, 1])
# # plt.show()
#
# session_plot.play_session(path + prefix + '.txt')

t = Treadmill(path, prefix)