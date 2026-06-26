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
    image = LoadImage(procpath, prefix)
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
    im = image.imdat.get_channel(channel)
    stoprun_image = numpy.zeros(image.info['sz'])
    frames = image.nframes

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


def calc_response_image(procpath, prefix, events, window, savepath=None, suffix='_response.tif', channel='Green',
                        trim = 10):
    '''
    Calculate a response image using the difference between the averages of two windows relative event frames
    Args:
        procpath:
        prefix:
        events: a list of frames
        window: a 2-tuple of the baseline and response periods, e.g. ((-1, 0), (0, 5)) saves the response //
         between 0 and 5 s after the events minus 1 to 0 s before the events
        savepath: results will be saved here, if None, uses the session folder
        suffix: response image will be saved as prefix_suffix.tif
        channel: any channel that can be parsed by LoadImage ('Green'/'Red' or 'Ch1', 'Ch2' or 0, 1)
        trim: edges of image excluded for scaling the intensities (px)

    Returns: saves the image

    '''
    dpath = os.path.join(procpath, prefix+'/')
    if savepath is None:
        savepath = dpath
    if not suffix.endswith('.tif'):
        suffix += '.tif'
    si = SessionInfo(dpath, prefix)
    si.load()
    sync = Sync(procpath, prefix)
    fps = si.info['framerate']
    image = LoadImage(procpath, prefix)
    im = image.imdat.get_channel(channel)
    response_image = numpy.zeros(image.info['sz'])
    frames = image.nframes
    pre_indices = numpy.arange(int(window[0][0] * fps), int(window[0][1] * fps))
    post_indices = numpy.arange(int(window[1][0] * fps), int(window[1][1] * fps))
    incl_frames = numpy.arange(max(0, -min(pre_indices)), frames - max(post_indices))
    if si.info['has_opto']:
        bad_frames = sync.load('opto')
        incl_frames = [t for t in incl_frames if t not in bad_frames]
    for t in events:
        if (t < incl_frames[0]) or (t > incl_frames[-1]):
            continue
        baseline_frames = [x for x in  t + pre_indices if x in incl_frames]
        response_frames = [x for x in t + post_indices if x in incl_frames]
        response_image[:, :] += im[response_frames, :, :].mean(axis=0) - im[baseline_frames, :, :].mean(axis=0)
    resp_rgb = numpy.zeros((*response_image.shape, 3), dtype='uint8')
    neg = numpy.zeros(response_image.shape)
    nw = numpy.where(response_image < 0)
    neg[nw] -= response_image[nw]
    pos = numpy.zeros(response_image.shape)
    pw = numpy.where(response_image > 0)
    pos[pw] += response_image[pw]
    r = max(numpy.percentile(neg[trim:-trim, trim:-trim], 99), numpy.percentile(pos[trim:-trim, trim:-trim], 99))
    resp_rgb[:, :, 2] = numpy.minimum(neg / r, 1) * 255
    resp_rgb[:, :, 1] = numpy.minimum(pos / r, 1) * 255
    cv2.imwrite(os.path.join(savepath, prefix + suffix), resp_rgb)
    print('Response image saved for', prefix)



if __name__ == '__main__':
    procpath = 'D:\Shares\Data\_Processed/2P\PVTot\Opto/'
    prefix = 'PVTot5_2023-09-04_opto_023'
    # exportstop(procpath, prefix)
