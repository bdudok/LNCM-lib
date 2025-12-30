import os
import numpy
from sklearn import cluster
try:
    from pyedflib import highlevel
except:
    print('pyedflib not available, use: "conda install pyedflib" before processing EEG data')
from envs import CONFIG

class EDF:
    __name__ = 'ReadEDF'
    def __init__(self, path, prefix, rejection_ops=None, ch=0):
        self.path = path
        self.prefix = prefix
        if not prefix.endswith('.edf'):
            self.prefix = prefix + '.edf'
        d = highlevel.read_edf(os.path.join(self.path, self.prefix))
        self.data = d[0]
        self.fs = d[1][0][CONFIG.EDF_fs_key]
        self.unit = d[1][0]['dimension']
        self.channels = [x['label'] for x in d[1]]
        self.rejection_ops = rejection_ops
        self.set_channel(ch)
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
            assert ch < len(self.channels), f'Channel {ch} not available. Channels: {self.channels}'
            chi = ch
        else:
            chi = None
            for ni, chn in enumerate(self.channels):
                if ch in chn:
                    chi = ni
                    break
            assert chi is not None, f'Channel {ch} not found. available: {self.channels}'
        self.chi = chi
        self.active_channel = self.channels[chi]
        if self.rejection_ops is not None and 'rejection_value' in self.rejection_ops:
            tr = numpy.copy(self.data[chi])
            rejection_value = self.rejection_ops['rejection_value']
            rejection_step = int(self.rejection_ops['rejection_step'] * self.fs)
            rejection_tail = int(self.rejection_ops['rejection_tail'] * self.fs)
            rejection_factor = self.rejection_ops['rejection_factor']
            if rejection_factor > 1.1:
                min_n = int(rejection_factor) #use factor as n of samples
            else:
                min_n = int(rejection_step * rejection_factor) #use factor as ratio

            bad_index = numpy.where(numpy.absolute(tr) > rejection_value)[0]
            if len(bad_index) > min_n:
                clustering = cluster.DBSCAN(eps=rejection_step, min_samples=min_n).fit(bad_index.reshape(-1, 1))
                labels = clustering.labels_
                for cid in range(labels.max() + 1):
                    x = bad_index[numpy.where(labels == cid)[0]]  # this is expected to be in order
                    r_start = int(max(0, x.min() - rejection_step))
                    r_stop = int(min(len(tr), x.max() + rejection_tail))
                    tr[r_start:r_stop] = 0
            self.trace = tr
            self.raw_trace = self.data[chi]
        else:
            self.trace = self.data[chi]
            self.raw_trace = None

if __name__ == '__main__':
    path = 'D:\Shares\Data\_RawData\EEG\Kainate\VKPV6/2024-03-25/'
    fn = 'VKPV6_2024-03-25_2024-03-25_12_31_50_TS_2024-03-25_14_31_50_export.edf'
    a = EDF(path, fn)