import numpy
import os
import cv2
from Proc2P.Bruker.PreProc import SessionInfo
from Proc2P.Analysis.LoadPolys import LoadImage
from Proc2P.Bruker.SyncTools import Sync
from Proc2P.utils import startstop, gapless

def exportstop(procpath, prefix, mode='stop', channel='Green', stimlen=15, trim=75):
    dpath = os.path.join(procpath, prefix+'/')
    si = SessionInfo(dpath, prefix)
    si.load()
    image = LoadImage(procpath, prefix, explicit_need_data=True)
    sync = Sync(procpath, prefix)
    speed = sync.load('speed')
    mov = gapless(speed, threshold=0.05)
    if si.info['has_opto']:
        bad_frames = sync.load('opto')

    span = 100, len(speed) - 100
    duration = 50
    gap = 150
    # collect stops
    starts, stops = startstop(speed, duration=duration, gap=gap, span=span)
    im = image.data
    stoprun_image = numpy.zeros(image.info['sz'])
    frames = image.nframes
    # if channel == 'Green':
    #     ch = 0
    # elif channel == 'Red':
    #     ch = 1
    #     if ch not in image.channels:
    #         print('Channel not found:', channel)
    #         return -1
    # else:
    #     print('Channel unexpected:', channel)
    #     return -1
    if mode == 'stop':
        suf = '_StopActivity.tif'
        for start, stop in zip(starts, stops):
            #exclude bad frames
            incl_frames = numpy.arange(stop, min(frames, stop + 100))
            baseline_frames = numpy.arange(start, stop)
            if si.info['has_opto']:
                incl_frames = [t for t in incl_frames if t not in bad_frames]
                baseline_frames = [t for t in baseline_frames if t not in bad_frames]
            stoprun_image[:, :] += im[incl_frames, :, :].mean(axis=0) - im[baseline_frames, :, :].mean(axis=0)
    # elif mode == 'opto':
    #     suf = '_OptoActivity.tif'
    #     opto = numpy.load(prefix + '_opto.npy')
    #     starts = numpy.where(numpy.diff(opto.astype('byte')) > 0)[0]
    #     for start in starts:
    #         stoprun_image[:, :] += im[start - 100:start, :, :, ch].mean(axis=0) - im[start:start + stimlen, :, :,
    #                                                                               ch].mean(axis=0)
    # elif mode == 'run':
    #     suf = '_RunActivity.tif'
    #     for i in range(len(starts)):
    #         start = starts[i]
    #         stop = stops[i]
    #         if i == 0:
    #             t0 = 100
    #         else:
    #             t0 = stops[i - 1]
    #         l = min(start - t0, stop - start)
    #         stoprun_image[:, :] += im[start - l:start, :, :, ch].mean(axis=0) - im[start:stop, :, :, ch].mean(axis=0)
    # positive values as red, negs as green
    stoprun_rgb = numpy.zeros((*stoprun_image.shape, 3), dtype='uint8')
    neg = numpy.zeros(stoprun_image.shape)
    nw = numpy.where(stoprun_image < 0)
    neg[nw] -= stoprun_image[nw]
    pos = numpy.zeros(stoprun_image.shape)
    pw = numpy.where(stoprun_image > 0)
    pos[pw] += stoprun_image[pw]
    r = max(numpy.percentile(neg[trim:-trim, trim:-trim], 99), numpy.percentile(pos[trim:-trim, trim:-trim], 99))
    stoprun_rgb[:, :, 2] = numpy.minimum(neg / r, 1) * 255
    stoprun_rgb[:, :, 1] = numpy.minimum(pos / r, 1) * 255
    cv2.imwrite(dpath + prefix + suf, stoprun_rgb)
    print('Stop score export finished for', prefix)


if __name__ == '__main__':
    procpath = 'D:\Shares\Data\_Processed/2P\PVTot\Opto/'
    prefix = 'PVTot5_2023-09-04_opto_023'
    # exportstop(procpath, prefix)
