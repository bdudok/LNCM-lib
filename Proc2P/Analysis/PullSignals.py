import numpy
from Proc2P.Analysis.LoadPolys import LoadImage, load_roi_file
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
            path, prefix, roi, ch, szmode, snrw = job
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
            retval = pull_signals(path, prefix, tag=tag, ch=ch, sz_mode=szmode, snr_weighted=snrw)
            if retval:
                lprint(self, retval, logger=self.log)


def pull_signals(path, prefix, tag=None, ch='All', sz_mode=False, snr_weighted=False, enable_alt_path=True):
    #get binary mask
    opPath = os.path.join(path, prefix + '/')
    roi_name = opPath + f'{prefix}_saved_roi_{tag}.npy'
    if not os.path.exists(roi_name):
        print(roi_name)
        print(prefix, ': Roi file not found: ', tag)
        return -1
    if os.path.exists(opPath + f'{prefix}_trace_{tag}.npy'):
        print(prefix, f': Trace file for {tag} exists, skipping...')
        return -1

    im = LoadImage(path, prefix)
    if enable_alt_path: #check if the raw data exists. this can be moved if the session is archived
        if not len(im.imdat.input_files):
            im.imdat.find_alt_path()

    data = load_roi_file(roi_name)
    #calculate binary mask
    height, width = im.info['sz']
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
        if 'Ch1' in im.channels and 'Ch2' in im.channels:
            channels = ['Ch2', 'Ch1'] #I want to keep green=0; red=1 throughout the pipeline, this step establishes that
            #and later steps inherit.
        else:
            channels = im.channels
    else:
        channels = [ch]

    #init empty array
    nframes = im.nframes
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
            inmem_data = numpy.array(im.imdat.get_channel(ch)[start:stop])
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
    return message

if __name__ == '__main__':
    path = r'D:\Shares\Data\_Processed\2P\JEDI-SWR/'
    prefix = 'JediPVCre50_2024-11-13_fast_487'
    tag = 'PVweighted'

    retval = pull_signals(path, prefix, tag=tag, ch=0, snr_weighted=True)
    print(retval)

