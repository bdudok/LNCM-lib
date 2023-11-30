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

    def __init__(self, dpath, procpath, prefix, btag='000', rsync_channel=0, led_channel=1, ):
        self.dpath = os.path.join(dpath, prefix + f'-{btag}/')  # raw data
        self.procpath = procpath + prefix + '/'  # output processed data
        self.prefix = prefix  # recorder prefix tag
        self.btag = btag  # tag appended by bruker (000 by default)

        # setup config
        self.led_channel = led_channel
        self.rsync_channel = rsync_channel
        self.reserved_channels = 2 #ignore voltage from first x channels
        self.lfp_channels = []

        self.md_keys = ['led_channel', 'rsync_channel', 'lfp_channels', 'btag', 'dpath', 'channelnames']

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
        self.found_output = []
        self.parse_frametimes()
        self.parse_TTLs()
        self.convert_ephys()
        self.parse_treadmill(tm_fig)
        self.parse_cam()
        self.save_metadata()
        print('Found:', ','.join(self.found_output))

    def parse_frametimes(self):
        # find xml filename
        xfn = os.path.join(self.dpath, self.prefix + f'-{self.btag}.xml')
        tree = ET.parse(xfn)
        root = tree.getroot()
        sequence = root.find('Sequence')

        # get 2P info
        sdict = {}
        self.channelnames = []
        for child in list(sequence.find('Frame').findall('File')):
            chn = child.attrib['channelName']
            self.channelnames.append(chn)
            self.found_output.append('2P'+chn)
        get_keys = ('framePeriod', 'scanLinePeriod')
        for child in list(sequence.find('Frame').find('PVStateShard')):
            key = child.attrib['key']
            if key in get_keys:
                sdict[key] = child.attrib['value']
        self.framerate = 1 / float(sdict['framePeriod'])
        if 'scanLinePeriod' in sdict:
            self.linetime = float(sdict['scanLinePeriod'])  # not always there, not sure why.
        else:
            self.linetime = numpy.nan
            # self.skip_analysis = True
            # print(self.prefix, 'XML parse error.')

    #     '''
    #     In Frame, there's a File for each Ch
    #     in 1 ch:
    #         <Frame relativeTime="0" absoluteTime="2.95699999999988" index="1" parameterSet="CurrentSettings">
    #   <File channel="2" channelName="Ch2" page="1" filename="SncgTot1_2023-11-09_LFP_001-000_Cycle00001_Ch2_000001.ome.tif" />
    #   <ExtraParameters lastGoodFrame="0" />
    #   <PVStateShard>
    #     <PVStateValue key="framePeriod" value="0.049999992" />
    #     <PVStateValue key="scanLinePeriod" value="6.3177E-05" />
    #     <PVStateValue key="twophotonLaserPower">
    #       <IndexedValue index="0" value="1800.4" />
    #     </PVStateValue>
    #   </PVStateShard>
    # </Frame>
    #     in 2 ch:
    #         <Frame relativeTime="0" absoluteTime="2.99499999999898" index="1" parameterSet="CurrentSettings">
    #   <File channel="1" channelName="Ch1" page="1" filename="SncgTot4_2023-10-23_movie_000-000_Cycle00001_Ch1_000001.ome.tif" />
    #   <File channel="2" channelName="Ch2" page="1" filename="SncgTot4_2023-10-23_movie_000-000_Cycle00001_Ch2_000001.ome.tif" />
    #   <ExtraParameters lastGoodFrame="0" />
    #   <PVStateShard>
    #     <PVStateValue key="framePeriod" value="0.050000004" />
    #     <PVStateValue key="scanLinePeriod" value="6.3159E-05" />
    #   </PVStateShard>
    # </Frame>
    #     '''

        # get voltage info
        voltage = sequence.find('VoltageRecording')
        self.voltage_config = voltage.attrib['configurationFile']
        self.voltage_data = voltage.attrib['dataFile']
        self.voltage_name = voltage.attrib['name']
        self.found_output.append('ADC')

        # get opto info
        opto = sequence.find('MarkPoints')
        self.has_opto = opto is not None
        if self.has_opto:
            self.opto_config = opto.attrib['filename']
            self.opto_name = opto.attrib['name']
            self.found_output.append('Opto')
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
        self.md_keys.extend(['n_frames', 'voltage_name', 'has_opto', 'opto_name',
                             'framerate', 'linetime', 'channelnames'])


    def parse_cam(self):
        vfn = self.dpath+self.prefix+'.avi'
        if os.path.exists(vfn):
            self.cam_file = vfn
            self.md_keys.append('cam_file')
            self.found_output.append('Camera')

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
            self.found_output.append('StimPulses')

        # rsync times
        if self.rsync_channel is not None:
            trace = vdat[f' Input {self.rsync_channel}'].values
            vmax = 3.3
            pos = numpy.where(numpy.convolve(trace > vmax * 0.5, [1, -1]) == 1)[0]
            self.ttl_times = pos / self.fs
            numpy.save(self.procpath + self.prefix + '_2pTTLtimes.npy', self.ttl_times)
            self.found_output.append('Rsync')

        ##append metadata attribs
        self.md_keys.append('fs')

    def convert_ephys(self):
        #get list of available LFP channels
        all_channels = [str(x.split(' ')[-1]) for x in self.vdat.keys() if 'Time' not in x]
        for x in all_channels:
            try:
                ch_n = int(x)
                if not ch_n < self.reserved_channels:
                    self.lfp_channels.append(x)
            except:
                pass

        if len(self.lfp_channels):
            trace = [self.vdat[f' Input {ch}'].values for ch in self.lfp_channels]
            # get the frame for each sample
            # use int for this. data range is +-10 V, 1000x gain. data*1000 is in microvolt (raw). int16 range is 32k
            ephys = - numpy.ones((len(self.lfp_channels) + 1, len(trace[0])), dtype='int16')
            # get the frame number for each sample
            for ti, t in enumerate(self.frametimes):
                ephys[0, int(t * self.fs):int((t + 1) * self.fs)] = ti
            for ci in range(len(self.lfp_channels)):
                ephys[ci + 1] = trace[ci] * 1000
                self.found_output.append(f'LFP{ci+1}')
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
        align = rsync.Rsync_aligner(tm_rsync, sc_rsync)
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
        self.found_output.append('Treadmill')

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
