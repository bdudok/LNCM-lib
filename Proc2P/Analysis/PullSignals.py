import numpy
from Proc2P.Analysis.LoadPolys import LoadImage, load_roi_file
import matplotlib.path as mplpath
import os
from multiprocessing import Process
from time import time
import datetime

class Worker(Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue

    def run(self):
        for job in iter(self.queue.get, None):
            path, prefix, roi, ch = job
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
            pull_signals(path, prefix, tag=tag, ch=ch)


def pull_signals(path, prefix, tag=None, ch='All', raw=False):
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
    print(f'Pulling {int(nframes*ncells*nchannels)} signals ({ncells} regions, {nchannels} channels) from {prefix} roi {tag}...')
    # create chunks so that the same array is combed for each cell before moving on - data is memory mapped
    t0 = datetime.datetime.now()
    chunk_len = 500
    indices = []
    for c in range(ncells):
        indices.append(binmask[c].nonzero())
    rep_size = 0.20
    next_report = rep_size
    for chi, ch in enumerate(channels):
        for t in range(int(nframes/chunk_len) + 1):
            start = int(t * chunk_len)
            if start / nframes > next_report:
                elapsed = datetime.datetime.now() - t0
                speed = (elapsed / len(channels) / start).microseconds
                print(f'Pulling {prefix}:{int(next_report*100):2d}% ({speed/1000:.1f} ms / frame)')
                next_report += rep_size
            stop = int(min(nframes, start + chunk_len))
            inmem_data = numpy.array(im.imdat.get_channel(ch)[start:stop])
            for c in range(ncells):
                ind = indices[c]
                traces[c, start:stop, chi] = inmem_data[:, ind[1], ind[0]].mean(axis=1)

    numpy.save(opPath + f'{prefix}_trace_{tag}', traces)
    elapsed = datetime.datetime.now() - t0
    minutes = datetime.timedelta.total_seconds(elapsed) / 60
    speed = (elapsed / ncells / len(channels) / nframes).microseconds
    print(f'{prefix} finished in {minutes:.1f} minutes ({speed} microseconds / cell / frame)')
    # print(traces.shape)
    return traces

if __name__ == '__main__':
    path = 'D:/Shares/Data/_Processed/2P/testing/'
    prefix = 'SncgTot4_2023-10-23_movie_000'
    tag = '1'


