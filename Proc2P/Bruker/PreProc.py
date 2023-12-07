import json

import matplotlib.pyplot as plt
import numpy
import os
import pandas
import xml.etree.ElementTree as ET
from Proc2P.Treadmill import TreadmillRead, rsync
from Proc2P.utils import lprint

'''
PreProcess: load xml files, save all frame times, sync signals and metadata into the analysis folder
'''


class PreProc:
    __name__ = 'PreProc'

    def __init__(self, dpath, procpath, prefix, btag='000', rsync_channel=0, lfp_channels=None, led_channel=1, ):
        self.dpath = os.path.join(dpath, prefix + f'-{btag}/')  # raw data
        self.procpath = procpath + prefix + '/'  # output processed data
        self.prefix = prefix  # recorder prefix tag
        self.btag = btag  # tag appended by bruker (000 by default)

        # setup config
        self.led_channel = led_channel
        self.rsync_channel = rsync_channel
        self.lfp_channels = lfp_channels

        self.md_keys = ['led_channel', 'rsync_channel', 'lfp_channels', 'btag', 'dpath']

        if not os.path.exists(self.procpath):
            os.mkdir(self.procpath)

        # init sessioninfo, preprocess if empty
        self.si = SessionInfo(self.procpath, prefix)
        self.is_processed = self.si.load()
        if not self.is_processed:
            self.skip_analysis = False
            self.preprocess()

    def preprocess(self, tm_fig=True):
        '''
        Convert the XML and voltage outputs to frametimes, timestamped events
        Saves bad_frames, FrameTimes, StimFrames, 2pTTLtimes, and sessiondata
        '''
        lprint(self, 'Pre-processing ' + self.prefix)
        self.parse_frametimes()
        self.parse_TTLs()
        self.convert_ephys()
        self.parse_treadmill(tm_fig)
        self.save_metadata()

    def parse_frametimes(self):
        # find xml filename
        xfn = os.path.join(self.dpath, self.prefix + f'-{self.btag}.xml')
        tree = ET.parse(xfn)
        root = tree.getroot()
        sequence = root.find('Sequence')

        # get 2P info
        sdict = {}
        get_keys = ('framePeriod', 'scanLinePeriod')
        for child in list(sequence.find('Frame').find('PVStateShard')):
            key = child.attrib['key']
            if key in get_keys:
                sdict[key] = child.attrib['value']
        self.framerate = 1 / float(sdict['framePeriod'])
        if 'scanLinePeriod' in sdict:
            self.linetime = float(sdict['scanLinePeriod'])  # not in 2channel files.
        else:
            self.linetime = numpy.nan
            # self.skip_analysis = True
            print(self.prefix, 'XML parse error.')
        # later include number of channels

        # get voltage info
        voltage = sequence.find('VoltageRecording')
        self.voltage_config = voltage.attrib['configurationFile']
        self.voltage_data = voltage.attrib['dataFile']
        self.voltage_name = voltage.attrib['name']

        # get opto info
        opto = sequence.find('MarkPoints')
        self.has_opto = opto is not None
        if self.has_opto:
            self.opto_config = opto.attrib['filename']
            self.opto_name = opto.attrib['name']
        else:
            self.opto_name = None
            self.opto_config = None
        df = pandas.DataFrame()
        keys = ('relativeTime', 'absoluteTime')
        dfs = []
        for frame in sequence.findall('Frame'):
            dfs.append(
                pandas.DataFrame([[float(frame.get(key)) for key in keys]], columns=keys, index=[frame.get('index')]))
        ft_op = pandas.concat(dfs)
        ft_op.sort_values('absoluteTime', inplace=True)
        ft_op.to_excel(self.procpath + self.prefix + '_FrameTimes.xlsx')
        numpy.save(self.procpath + self.prefix + '_FrameTimes.npy', ft_op.values)
        self.frametimes = ft_op['relativeTime'].values
        self.n_frames = len(dfs)

        # append metadata attribs
        self.md_keys.extend(['n_frames', 'voltage_name', 'has_opto', 'opto_name', 'framerate', 'linetime'])

    def parse_TTLs(self):
        vdat = pandas.read_csv(self.dpath + self.voltage_data)
        self.vdat = vdat
        # get fs
        tree = ET.parse(self.dpath + self.voltage_config)
        root = tree.getroot()
        experiment = root.find('Experiment')
        rate = experiment.find('Rate')
        self.fs = int(rate.text)

        # LED pulses
        if self.led_channel is not None:
            trace = vdat[f' Input {self.led_channel}'].values
            vmax = 5.0  # 5V command is 100%LED
            pos = numpy.where(numpy.convolve(trace > vmax * 0.05, [1, -1]) == 1)[0]
            stimframes = numpy.searchsorted(self.frametimes * self.fs, pos) - 1  # convert to 0 indexing
            led_op = pandas.DataFrame({'Intensity': trace[pos] / vmax, 'ImgFrame': stimframes}, index=[pos])
            led_op.to_excel(self.procpath + self.prefix + '_StimFrames.xlsx')
            numpy.save(self.dpath + 'bad_frames.npy', stimframes)
            numpy.save(self.procpath + self.prefix + '_bad_frames.npy', stimframes)

        # rsync times
        if self.rsync_channel is not None:
            trace = vdat[f' Input {self.rsync_channel}'].values
            vmax = 3.3
            pos = numpy.where(numpy.convolve(trace > vmax * 0.5, [1, -1]) == 1)[0]
            self.ttl_times = pos / self.fs
            numpy.save(self.procpath + self.prefix + '_2pTTLtimes.npy', self.ttl_times)

        ##append metadata attribs
        self.md_keys.append('fs')

    def convert_ephys(self):
        if self.lfp_channels is not None:
            trace = [self.vdat[f' Input {ch}'].values for ch in self.lfp_channels]
            # get the frame for each sample
            # use int for this. data range is +-10 V, 1000x gain. data*1000 is in microvolt (raw). int16 range is 32k
            ephys = - numpy.ones((len(self.lfp_channels) + 1, len(trace[0])), dtype='int16')
            # get the frame number for each sample
            for ti, t in enumerate(self.frametimes):
                ephys[0, int(t * self.fs):int((t + 1) * self.fs)] = ti
            for ci in range(len(self.lfp_channels)):
                ephys[ci + 1] = trace[ci] * 1000
            self.ephys = ephys
            numpy.save(self.procpath + self.prefix + '_ephys.npy', ephys)

        ##append metadata attribs
        self.lfp_units = 'microvolts'
        self.md_keys.append('lfp_units')

    def parse_treadmill(self, save_fig=True):
        tm = TreadmillRead.Treadmill(self.dpath, self.prefix)
        self.treadmill_fn = tm.filename
        self.md_keys.append('treadmill_fn')
        #save figure
        if save_fig:
            fig, ax = tm.export_plot()
            fig.savefig(self.procpath + self.prefix + '_treadmill.png', dpi=300)
            plt.close()
        # read rSync
        tm_rsync = tm.get_Rsync_times()
        sc_rsync = (self.ttl_times * 1000).astype('int')
        try:
            align = rsync.Rsync_aligner(tm_rsync, sc_rsync)
            skip = False
        except:
            print('Treadmill pulse times:', tm_rsync)
            print('Scope pulse times:', sc_rsync)
            print('RSync align error')
            skip = True
        if not skip:
            self.frame_at_treadmill = align.B_to_A(self.frametimes * 1000)
            numpy.save(self.procpath + self.prefix + '_frame_tm_times.npy', self.frame_at_treadmill)
            # resample speed, pos to scope frames
            indices = self.get_frame_tm_x(self.frame_at_treadmill, tm.pos_tX * 1000)
            for Y, tag in zip((tm.smspd, tm.pos), ('smspd', 'pos')):
                op = numpy.empty(len(self.frame_at_treadmill))
                op[:] = numpy.nan
                mask = indices > -1
                op[mask] = Y[indices[mask]]
                numpy.save(self.procpath + self.prefix + f'_{tag}.npy', op)
            # add lap and reward number to output
            self.laps = len(tm.laptimes)
            self.rewards = 'not implemented'
            self.md_keys.extend(['laps', 'rewards'])

    def get_frame_tm_x(self, frametime, tX):
        '''for each frametime, find index in treadmill analog signal'''
        indices = numpy.empty(len(frametime), dtype='int')
        indices[:] = -1
        ix = 0
        for i, ft in enumerate(frametime):
            if not numpy.isnan(ft):
                ix += numpy.searchsorted(tX[ix:], ft)
                if not ix < len(tX):
                    break
                indices[i] = ix
        return indices

    def save_metadata(self):
        '''
        Create a SessionInfo, add all the relevant attrs, and save to json in the output path.
        '''
        for key in self.md_keys:
            self.si[key] = self.__getattribute__(key)
        self.si.save()


class SessionInfo:
    __name__ = 'SessionInfo'

    def __init__(self, procpath, prefix):
        self.procpath = procpath
        self.prefix = prefix
        self.filehandle = self.procpath + self.prefix + '_SessionInfo.json'
        self.info = {}

    def load(self):
        if os.path.exists(self.filehandle):
            with open(self.filehandle, 'r') as f:
                self.info = json.load(f)
            return self.info
        else:
            return None

    def save(self):
        with open(self.filehandle, 'w') as f:
            json.dump(self.info, f)

    def __getitem__(self, item):
        return self.info.get(item)

    def __setitem__(self, key, value):
        self.info[key] = value


if __name__ == '__main__':
    dpath = 'D:\Shares\Data\_RawData\Bruker\PVTot/'
    procpath = 'D:\Shares\Data\_Processed/2P\PVTot\LFP/'
    prefix = 'PVTot5_2023-09-15_LFP_026'
    btag = '000'

    s = PreProc(dpath, procpath, prefix, btag, lfp_channels=(2,))
    e = s.ephys
