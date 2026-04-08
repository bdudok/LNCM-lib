import os
from Proc2P.Legacy.ImagingSession import ImagingSession
from matplotlib import pyplot as plt
path = 'D:\Shares\Data\old_2P\DLX-ECB'
prefix = 'DLX-ECB_431_560'
os.chdir(path)
session = ImagingSession(prefix, tag='skip', norip=True)
plt.plot(session.pos.pos)
