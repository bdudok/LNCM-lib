import numpy
from tifffile import imwrite
# import cv2
import os
from pathlib import Path

# from Proc2P.Bruker.PreProc import SessionInfo
from Proc2P.Analysis.LoadPolys import LoadImage


def TimeProfile(procpath, prefix, cfg, ret=False, channel=0):
    duration = (cfg['Start'], cfg['Stop'])
    orientation = 'horizontal'
    if 'Orientation' in cfg:
        orientation = cfg['Orientation']
    line = cfg['Line']
    kernel = cfg['Kernel']
    if 'Channel' in cfg:
        channel = cfg['Channel']
    dpath = Path(procpath) / prefix
    # si = SessionInfo(dpath, prefix)
    # si.load()
    image = LoadImage(procpath, prefix)
    sh = image.info['sz']
    height = sh[0]
    width = sh[1]
    if orientation == 'horizontal':
        im = numpy.empty((duration[1] - duration[0], width))
    elif orientation == 'vertical':
        im = numpy.empty((duration[1] - duration[0], height))
    if orientation == 'horizontal':
        c0 = 0
        c1 = width
        l0 = max(0, int(line - kernel / 2))
        l1 = min(height, int(line + kernel / 2))
    elif orientation == 'vertical':
        l0 = 0
        l1 = height
        c0 = max(0, int(line - kernel / 2))
        c1 = min(width, int(line + kernel / 2))
    if channel == 0:
        data = image.imdat.data
    elif channel == 1:
        data = image.imdat.data2
    for t in range(duration[1] - duration[0]):
        if orientation == 'horizontal':
            im[t] = data[t + duration[0], l0:l1, c0:c1].mean(axis=0)
        elif orientation == 'vertical':
            im[t] = data[t + duration[0], l0:l1, c0:c1].mean(axis=1)
    if ret:
        return im
    else:
        im -= im.min()
        im /= numpy.percentile(im, 95)
        im = numpy.minimum(1, im)
        fn = f'_linescan_x_{line}-{kernel}-ch{channel}_{duration[0]}-{duration[1]}.tif'
        imwrite(dpath / f'{prefix}{fn}', (im.transpose() * 255).astype('uint8'))


if __name__ == '__main__':

    procpath = r'D:\Shares\Data\_Processed\2P\JEDI-IPSP/'
    prefix = 'JEDI-Sncg80_2025-01-02_lfp_opto_230'
    duration = 50000, 100000
    line = 24
    kernel = 48
