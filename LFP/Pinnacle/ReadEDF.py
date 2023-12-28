import os
from pyedflib import highlevel

class EDF:
    __name__ = 'ReadEDF'
    def __init__(self, path, prefix):
        self.path = path
        self.prefix = prefix
        if not prefix.endswith('.edf'):
            self.prefix = prefix + '.edf'
        d = highlevel.read_edf(os.path.join(self.path, self.prefix))
        self.data = d[0]
        self.fs = d[1][0]['sample_rate']
        self.unit = d[1][0]['dimension']
        self.channels = [x['label'] for x in d[1]]
        self.set_channel(0)

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
        self.trace = self.data[chi]

if __name__ == '__main__':
    path = 'D:\Shares\Data\_RawData\Pinnacle\Tottering\Tot6_Tottering++/'
    fn = 'Tot6_0014_2023-12-27_11_00_59_TS_2023-12-27_11_00_59_export.edf'
    a = EDF(path+fn)