import numpy
from Proc2P.Analysis.LoadPolys import LoadImage, load_roi_file
from Proc2P.Bruker.LoadMovie import LoadMovie, get_raw_movies
from Proc2P.Bruker.PreProc import SessionInfo
import matplotlib.path as mplpath
from statsmodels.stats.weightstats import DescrStatsW
import os
from multiprocessing import Process
from time import time
import datetime
from Proc2P.utils import logger, lprint


class Worker(Process):
    __name__ = 'PullSignals'
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue
        self.log = logger()

    def run(self):
        for job in iter(self.queue.get, None):
            path, prefix, roi, ch, szmode, snrw, useraw = job
            if roi == 'Auto':
                #locate last saved roi file
                exs = [1]
                for f in os.listdir():
                    if prefix in f and '_saved_roi_' in f:
                        try:
                            if f[-6:-4] != '-1':
                                exs.append(int(f[:-4].split('_')[-1]))
                        except:
                            pass
                tag = str(max(exs))
            else:
                tag = roi
            self.log.set_handle(path, prefix)
            retval = pull_signals(path, prefix, tag=tag, ch=ch, snr_weighted=snrw, use_raw_movie=useraw)
            if retval:
                lprint(self, retval, logger=self.log)


def pull_signals(path, prefix, tag=None, ch='All', snr_weighted=False, enable_alt_path=True,
                 overwrite=False, use_movie='S2P'):
    '''
    Compute average pixel intensity in each ROI and each frame of a movie
    :param path: the processed folder
    :param prefix: of the session
    :param tag: name of the ROI set (saved by the ROI editor)
    :param ch: which channel to pull ('Ch1', 'Ch2', 'All').
    If All, output order will be ['Ch2', 'Ch1] (green, red detector with Bruker)
    :param snr_weighted: if True, estimates the SNR of each pixel and uses it in a weighted average
    If False, simple average f all pixels in the ROI
    :param enable_alt_path: If the registered movie not found, looks in the backup folders specified in config.
    This is used because we delete the registered movies from the server after 6 months.
    :param overwrite: if False, checks if output exists and doesn't pull if yes
    :param use_movie:
        'S2P': default registered movie
        'Raw': uses the raw movie instead of the registered one. Looks up path in the session info file.
        'GEVIReg': uses the alternative corrected movie
    :return: saves the output traces to a file. Only returns report string (also saved in the log).
    '''
    #get binary mask
    opPath = os.path.join(path, prefix + '/')
    roi_name = opPath + f'{prefix}_saved_roi_{tag}.npy'
    if not os.path.exists(roi_name):
        print(roi_name)
        print(prefix, ': Roi file not found: ', tag)
        return -1
    if os.path.exists(opPath + f'{prefix}_trace_{tag}.npy') and not overwrite:
        print(prefix, f': Trace file for {tag} exists, skipping...')
        return -1

    #parse input
    if use_movie == 'S2P':
         #case: S2P registered movie
        im = LoadImage(path, prefix)
        if enable_alt_path: #check if the raw data exists. this can be moved if the session is archived
            if not len(im.imdat.input_files):
                im.imdat.find_alt_path()
        channelnames = im.channels
        nframes = im.nframes
        height, width = im.info['sz']
    elif use_movie == 'Raw':
         #case: raw bruker tiff
        si = SessionInfo(opPath, prefix)
        info = si.load()
        movies = get_raw_movies(info)
        channelnames = si.info["channelnames"]
        nframes, height, width = movies[channelnames[0]].shape
    elif use_movie == 'GEVIReg':
         #case: alternative registered movie
        #TODO now hard code path, later intergateh this as an option in LoadImage and just pass the alt tag to it
        im = numpy.load(os.path.join(opPath, 'GEVIReg', prefix+'_registered_Ch2.npy'), mmap_mode='r')
        print(im.shape)
        nframes, height, width = im.shape
        channelnames = ['Ch2',]
    else:
        raise ValueError(f'use_movie not implemented for {use_movie}')


    data = load_roi_file(roi_name)
    #calculate binary mask
    binmask = numpy.zeros((len(data), width, height), dtype='bool')
    print(f'Computing masks from {len(data)} rois...')
    for nroi, pr in enumerate(data):
        roi = numpy.empty((2, len(pr)), dtype='int')
        for j in range(len(pr)):
            roi[:, j] = pr[j]
        left, top = roi.min(axis=1)
        right, bottom = roi.max(axis=1)
        # load poly for pip function
        poly = mplpath.Path(pr)
        for x in range(left, right):
            for y in range(top, bottom):
                if poly.contains_point([x, y]):
                    binmask[nroi, x, y] = True
    #figure out channels to pull MODES = ['All', 'Green', 'Red']
    if ch == 'All':
        if 'Ch1' in channelnames and 'Ch2' in channelnames:
            channels = ['Ch2', 'Ch1'] #I want to keep green=0; red=1 throughout the pipeline, this step establishes that
            #and later steps inherit.
        else:
            channels = channelnames
    else:
        channels = [ch]

    #init empty array
    ncells = len(binmask)
    nchannels = len(channels)
    traces = numpy.empty((ncells, nframes, nchannels))
    # print('Reading data from disk to memory...')
    # im.force_read()  # miniscope compatibility and improves performance if low on mem. moved to default force chunk read
    message = f'Pulling {int(nframes*ncells*nchannels)} signals ({ncells} regions, {nchannels} channels) from {prefix} roi {tag}...'
    print(message)
    # create chunks so that the same array is combed for each cell before moving on - data is memory mapped
    #pick chunk size based on image size. on the server, using ~5 sec of 400MB/s read speed.
    chunk_len = int(min(nframes, (4e8*5*8)/(16*width*height)))
    #bring chunk to nearest 1000 and make sure we're not left with a tiny last chunk:
    minchunk = 1000
    if chunk_len < nframes/2:
        chunk_len = int(round(chunk_len/minchunk)+1)*minchunk
    if 0 < divmod(nframes, chunk_len)[1] < minchunk:
        chunk_len += minchunk
    #get indices of pixels in each polygon roi - indexing with this gives 1d array
    indices = []
    for c in range(ncells):
        x = binmask[c].nonzero()
        assert len(x), f'Mask for c {c} is empty, export new ROI file.'
        indices.append(x)
    rep_size = 0.20 * len(channels)
    for chi, ch in enumerate(channels):
        t0 = datetime.datetime.now()
        next_report = rep_size
        weights = None
        for t in range(int(nframes/chunk_len) + 1):
            start = int(t * chunk_len)
            if start / nframes > next_report:
                elapsed = datetime.datetime.now() - t0
                speed = (elapsed / start).microseconds
                print(f'Pulling {prefix}:{int(next_report*100/len(channels)+50*chi):2d}% ({speed/1000:.1f} ms/frame)')
                next_report += rep_size
            stop = int(min(nframes, start + chunk_len))
            if use_movie == 'Raw':
                inmem_data = movies[ch][start:stop]
            elif use_movie == 'S2P':
                inmem_data = numpy.array(im.imdat.get_channel(ch)[start:stop])
            elif use_movie == 'GEVIReg':
                inmem_data = numpy.array(im[start:stop])
            if snr_weighted:
                if weights is None:
                    #get the weights of each pixel within ROI for each cell based on snr in first chunk
                    weights = []
                    # pick frames that are close to median (to exclude synchronous activations and PMT gating)
                    frame_mean = numpy.nanmean(inmem_data, axis=(1,2))
                    frame_IQR = numpy.nanpercentile(frame_mean, [25,75])
                    snr_frames = numpy.where(numpy.logical_and(frame_mean>frame_IQR[0], frame_mean<frame_IQR[1]))
                    fovmean = numpy.nan_to_num(numpy.nanmean(inmem_data[snr_frames], axis=0))
                    fovsd = numpy.nan_to_num(numpy.nanstd(inmem_data[snr_frames], axis=0))
                    fovsnr = numpy.where(fovsd == 0, 0, fovmean / fovsd)
                    for c in range(ncells):
                        ind = indices[c]
                        weights.append(fovsnr[ind[1], ind[0]])
                for c in range(ncells):
                    #iterate frames and compute the weighted average of the polygon pixels
                    ind = indices[c]
                    wweight = weights[c]
                    for fi in range(len(inmem_data)):
                        weighted_stats = DescrStatsW(inmem_data[fi, ind[1], ind[0]], weights=wweight, ddof=0, )
                        traces[c, start+fi, chi] = weighted_stats.mean
            else:
                #compute simple average of the polygon in each frame
                for c in range(ncells):
                    ind = indices[c]
                    traces[c, start:stop, chi] = numpy.nanmean(inmem_data[:, ind[1], ind[0]], axis=1)

    numpy.save(opPath + f'{prefix}_trace_{tag}', traces)
    elapsed = datetime.datetime.now() - t0
    minutes = datetime.timedelta.total_seconds(elapsed) / 60
    speed = (elapsed / len(channels) / nframes).microseconds
    print(f'{prefix} finished in {minutes:.1f} minutes ({speed/1000:.1f} ms / frame)')
    # print(traces.shape)
    avgtype = ('simple', 'weighted')[snr_weighted]
    message = f'Pulled {avgtype} average from {int(nframes)} frames ({ncells} regions, {nchannels} channels) from {prefix} roi {tag}'
    message += '\r\n' + f'Channel order is: {channels}'
    message += '\r\n' + f'{use_movie} movie was used '
    return message

if __name__ == '__main__':
    path = r'D:\Shares\Data\_Processed\2P\eCB-GRAB/'
    prefix = 'BL6-31_2025-08-07_e-stimdrug_201'
    tag = 'rtest'

    path = r'D:\Shares\Data\_Processed/2P\JEDI-IPSP/'
    prefix = 'JEDI-Sncg73_2024-11-19_burst_033'
    tag = 'GRtest'

    # opPath = os.path.join(path, prefix + '/')
    # si = SessionInfo(opPath, prefix)
    # info = si.load()
    # movies = get_raw_movies(info)
    # channelnames = si.info["channelnames"]
    # chn = channelnames[0]


    retval = pull_signals(path, prefix, tag=tag, ch='All', snr_weighted=True, use_movie='GEVIReg')
    print(retval)

