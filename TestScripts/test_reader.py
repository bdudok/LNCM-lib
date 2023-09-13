import os
from matplotlib import pyplot as plt
import numpy
import pandas as pd
from Proc2P.Treadmill.TreadmillRead import data_import, session_plot, Treadmill
from Proc2P.Treadmill import rsync
from Proc2P.Bruker.PreProc import PreProc

path = 'D:\Shares\Data\_RawData\Bruker\PVTot\PVTot5_2023-09-04_opto_023-000/'
prefix = 'PVTot5_2023-09-04_opto_023'
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

self = t = Treadmill(path, prefix)
d = t.d

dpath = 'D:\Shares\Data\_RawData\Bruker\PVTot/'
procpath = 'D:\Shares\Data\_Processed/2P\PVTot\Opto/'

md = PreProc(dpath, procpath, prefix, )
md.preprocess()

tm = t
spd = tm.smspd
pos = tm.pos
ptime = tm.pos_tX
tm_rsync = tm.get_Rsync_times()
sc_rsync = (md.ttl_times * 1000).astype('int')
l = min(len(tm_rsync), len(sc_rsync))
# plt.scatter(tm_rsync[:l], sc_rsync[:l])

align = rsync.Rsync_aligner(tm_rsync, sc_rsync, units_A='auto', units_B='auto',
                 chunk_size=5, plot=True, raise_exception=True)

frametime = align.B_to_A(md.frametimes * 1000)

tX = tm.pos_tX
indices = numpy.empty(len(frametime), dtype='int')
indices[:] = -1
ix = 0
for i, ft in enumerate(frametime):
    if not numpy.isnan(ft):
        ix += numpy.searchsorted(tX[ix:], ft/1000)
        indices[i] = ix