import os
import numpy
from sklearn import cluster
from pyedflib import highlevel

class EDF:
    __name__ = 'ReadEDF'
    def __init__(self, path, prefix, rejection_ops=None):
        self.path = path
        self.prefix = prefix
        if not prefix.endswith('.edf'):
            self.prefix = prefix + '.edf'
        d = highlevel.read_edf(os.path.join(self.path, self.prefix))
        self.data = d[0]
        self.fs = d[1][0]['sample_rate']
        self.unit = d[1][0]['dimension']
        self.channels = [x['label'] for x in d[1]]
        self.rejection_ops = rejection_ops
        self.set_channel(0)
        self.d = d

    def get_TTL(self, channel='GPIO0'):
        for chi, x in enumerate(self.channels):
            if channel in x:
                break
        assert channel in x
        y = self.data[chi]
        vmax = y.max()
        return numpy.where(numpy.convolve(y > vmax * 0.5, [1, -1]) == 1)[0] / self.fs


    def set_channel(self, ch):
        if type(ch) is int:
            chi = ch
        else:
            if ch in self.channels:
               chi = self.channels.index(ch)
            else:
                print(f'Channel {ch} not found. available: {self.channels}')
                assert False
        self.active_channel = self.channels[chi]
        if self.rejection_ops is not None and 'rejection_value' in self.rejection_ops:
            tr = numpy.copy(self.data[chi])
            rejection_value = self.rejection_ops['rejection_value']
            rejection_step = int(self.rejection_ops['rejection_step'] * self.fs)
            rejection_tail = int(self.rejection_ops['rejection_tail'] * self.fs)
            rejection_factor = self.rejection_ops['rejection_factor']
            min_n = int(rejection_step * rejection_factor)

            bad_index = numpy.where(numpy.absolute(tr) > rejection_value)[0]
            if len(bad_index) > min_n:
                clustering = cluster.DBSCAN(eps=rejection_step, min_samples=min_n).fit(bad_index.reshape(-1, 1))
                labels = clustering.labels_
                for cid in range(labels.max() + 1):
                    x = bad_index[numpy.where(labels == cid)[0]]  # this is expected to be in order
                    r_start = max(0, x.min() - rejection_step)
                    r_stop = min(len(tr), x.max() + rejection_tail)
                    tr[r_start:r_stop] = 0
            self.trace = tr
            self.raw_trace = self.data[chi]
        else:
            self.trace = self.data[chi]
            self.raw_trace = None

if __name__ == '__main__':
    path = 'D:\Shares\Data\_RawData\Pinnacle\Tottering\Tot6_Tottering++/'
    fn = 'Tot6_0014_2023-12-27_11_00_59_TS_2023-12-27_11_00_59_export.edf'
    a = EDF(path+fn)