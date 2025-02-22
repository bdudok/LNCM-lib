import numpy
from tifffile import imsave
# import cv2
import os

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
    dpath = os.path.join(procpath, prefix+'/')
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
        imsave(dpath + prefix + fn, (im.transpose() * 255).astype('uint8'))


# def OEProfile(prefix, ret=False): #not updated
#     '''
#     Pull the horizontal and verical line profile in all channels, to allow identifying sync events
#      due to sensor overexpression. Saves array and tif.
#     :param prefix:
#     :return: saved image
#     '''
#     a = LoadImage(prefix,)
#     line_margin = 45
#     col_margin = 80
#     t_margin = 100
#     sh = a.info['sz']
#     c0 = col_margin
#     c1 = sh[1] - col_margin
#     width = c1 - c0
#     l0 = line_margin
#     l1 = sh[0] - line_margin
#     height = l1 - l0
#     channels = a.channels
#     ch_lookup = (1, 0, 2)  # BGR
#     c_prof = numpy.empty((len(a.data), height, len(channels)))
#     l_prof = numpy.empty((len(a.data), width, len(channels)))
#     for y in c_prof, l_prof:
#         y[:t_margin] = numpy.nan
#         y[-t_margin:] = numpy.nan
#     # pull line profiles
#     for ci, c in enumerate(channels):
#         Y = a.data[t_margin:-t_margin, l0:l1, c0:c1, ci]
#         l_prof[t_margin:-t_margin, :, ci] = 65535 - numpy.mean(Y, axis=1)
#         c_prof[t_margin:-t_margin, :, ci] = 65535 - numpy.mean(Y, axis=2)
#     numpy.save(prefix + '_OE_prof_l.npy', l_prof)
#     numpy.save(prefix + '_OE_prof_c.npy', c_prof)
#     # assemble image
#     im = numpy.zeros((len(a.data), height + width, 3), dtype='uint8')
#     gamma = 1 / 1.8
#     table = numpy.array([((i / 255.0) ** gamma) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
#     for ci, c in enumerate(channels):
#         y = c_prof[t_margin:-t_margin, :, ci]
#         y = y - y.min()
#         y /= max(y.max(), 65536 / 4)  # norm if bright but keep dim if all dim.
#         im[t_margin:-t_margin, :height, ch_lookup[ci]] = cv2.LUT((y * 255).astype('uint8'), table)
#
#         y = l_prof[t_margin:-t_margin, :, ci]
#         y = y - y.min()
#         y /= max(y.max(), 65536 / 4 - 1)  # norm if bright but keep dim if all dim.
#         im[t_margin:-t_margin, height:, ch_lookup[ci]] = cv2.LUT((y * 255).astype('uint8'), table)
#
#     imr = cv2.resize(im, (width + height, 1920), interpolation=cv2.INTER_LANCZOS4)
#     imsave(prefix + f'_linescan_cl.tif', imr.transpose())
#     if ret:
#         return (l_prof, c_prof)
#     else:
#         return 0
#
#
# def GetNeuroPil(prefix, sig_only=True, intensity=False):
#     fn = prefix + '.neuropil.npy'
#     if os.path.exists(fn):
#         y = numpy.load(fn)
#     else:
#         print('Pulling neuropil trace: ', prefix)
#         y = PullNeuroPil(prefix)
#     if intensity:
#         return y[0]
#     if sig_only:
#         return y[3] * y[4]  # NB this returns synchrony signal, not intensity
#     else:
#         return y
#
#
# def GetNeuropilIntensity(prefix):
#     y = GetNeuroPil(prefix, sig_only=False)
#     return y[5]
#
#
# def PullNeuroPil(prefix, margin=80):
#     a = LoadImage(prefix)
#     nframes = a.nframes
#     nchannels = len(a.channels)
#     # find cropping
#     sh = a.info['sz']
#     width = int((sh[1] - 2 * margin) * 0.9)
#     kernel = int(sh[0] * 0.9)
#     line = int(sh[0] * 0.5)
#     c0 = int((sh[1] - width) / 2)
#     c1 = c0 + width
#     l0 = max(0, int(line - kernel / 2))
#     l1 = min(sh[0], int(line + kernel / 2))
#     # pull mean line profile
#     im = numpy.empty((nframes, width, nchannels))
#     for t in range(nframes):
#         im[t] = a.data[t, l0:l1, c0:c1].mean(axis=0)
#     m = im.mean(axis=1)
#     # fit baseline with a simple smoothing
#     smw = numpy.empty((nframes, nchannels))
#     t1 = 50
#     for t in range(nframes):
#         ti0 = max(0, int(t - t1 * 0.5))
#         ti1 = min(nframes, int(t + t1 * 0.5) + 1)
#         smw[t] = numpy.mean(m[ti0:ti1], axis=0)
#     # populate output array with: mean, median, IQR, diff
#     y = numpy.empty((6, nframes, nchannels))
#     y[0] = m - smw
#     # sub baseline from image
#     im -= smw[:, numpy.newaxis, :]
#     # store median and spread
#     s = numpy.sort(im, axis=1)
#     y[1] = s[:, int(width / 2), :]
#     y[2] = (s[:, int(width * 0.75), :] - s[:, int(width * 0.25), :])
#
#     # create normalized differential of increasing median and decreasing spread
#     def norm(d):
#         return numpy.maximum(numpy.minimum(d / numpy.percentile(d, 99, axis=0), 1), 0)
#
#     y[3:, 0, :] = 0
#     y[3, 1:, :] = norm(numpy.diff(y[1], axis=0))
#     y[4, 1:, :] = norm(-numpy.diff(y[2], axis=0))
#     y[5] = 65535 - m
#     numpy.save(prefix + '.neuropil', y)
#     return y


if __name__ == '__main__':

    procpath = r'D:\Shares\Data\_Processed\2P\JEDI-IPSP/'
    prefix = 'JEDI-Sncg80_2025-01-02_lfp_opto_230'
    duration = 50000, 100000
    line = 24
    kernel = 48
