import json

import matplotlib.pyplot as plt
import numpy
import os
import pandas
from sklearn import cluster
import xml.etree.ElementTree as ET
from Proc2P.Treadmill import TreadmillRead, rsync
from Proc2P.utils import lprint

'''
PreProcess: load xml files, save all frame times, sync signals and metadata into the analysis folder
'''


class PreProc:
    __name__ = 'PreProc'

    def __init__(self, dpath, procpath, prefix, btag='000', rsync_channel=0, led_channel=1,
                 debug=False, overwrite=False):
        self.dpath = os.path.join(dpath, prefix + f'-{btag}/')  # raw data
        self.procpath = os.path.join(procpath, prefix) + '/'  # output processed data
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

        if not debug:
            # init sessioninfo, preprocess if empty
            self.si = SessionInfo(self.procpath, prefix)
            if not overwrite:
                self.is_processed = self.si.load()
            if overwrite or not self.is_processed:
                self.skip_analysis = False
                self.preprocess()
            else:
                self.load_metadata()

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

        #get scan settings
        sdict = {}
        simple_keys = ('activeMode', 'bitDepth', 'opticalZoom', 'objectiveLensMag', 'objectiveLensNA', 'samplesPerPixel')
        indexed_keys = {'laserWavelength': ('0',), 'twophotonLaserPower': ('0',),
                    'micronsPerPixel': ('XAxis', 'YAxis'), 'pmtGain': ('0', '1')}
        for child in list(root.find('PVStateShard')):
            key = child.attrib['key']
            if key in simple_keys:
                self.md_keys.append(key)
                self.__setattr__(key, child.attrib['value'])
            elif key in indexed_keys:
                for ielement in child:
                    if ielement.attrib['index'] in indexed_keys[key]:
                        opk = key+'_'+ielement.attrib['index']
                        self.md_keys.append(opk)
                        self.__setattr__(opk, ielement.attrib['value'])
        sequence = root.find('Sequence')

        # get 2P info
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
        # After introducing photostim device, opto can come from either mark points, or the device.
        # LED power is analog in MarkPoints, PWM by device. parse the voltage separately.
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
        tfn = self.dpath + self.prefix + '_CamTimers.npy'
        if os.path.exists(tfn):
            cam_frame_times = numpy.load(tfn)  # millis
            self.parse_frametimes()
            scope_frame_times = (self.frametimes * 1000).astype('int64')
            cam_frames = numpy.zeros(len(cam_frame_times), dtype='int64')
            scope_delay = cam_frame_times[0, 1]
            self.md_keys.append('cam_speed_actual')
            self.cam_speed_actual = (numpy.diff(cam_frame_times[:, 1]) / numpy.diff(cam_frame_times[:, 0])).mean()
            # find the scope frame closes to each camera frame, by time
            f0 = 0
            for fi, (frame, time) in enumerate(cam_frame_times):
                f1 = numpy.searchsorted(scope_frame_times[f0:], time - scope_delay)
                lt0 = max(0, f0 + f1 - 1)
                lt1 = min(len(scope_frame_times), f0 + f1 + 2)
                tdiff = numpy.abs(scope_frame_times[lt0:lt1] - (time - scope_delay))
                f = numpy.argmin(tdiff)
                if tdiff[f] < self.cam_speed_actual:
                    found_frame = lt0 + f
                    cam_frames[fi] = found_frame
                    f0 = found_frame
                else:
                    cam_frames[fi] = 0
            numpy.save(self.procpath + self.prefix + '_cam_sync_frames.npy', cam_frames)
            self.found_output.append('CameraTiming')

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
        # the way opto is parsed; parse_TTLs needs to be called after parse_frametimes so that has_opto is set
        if self.led_channel is not None:
            trace = vdat[f' Input {self.led_channel}'].values
            vmax = 5.0  # 5V command is 100%LED
            bintrace = trace > (vmax * 0.05)
            pos = numpy.where(numpy.convolve(bintrace, [1, -1]) == 1)[0] #rising edges
            if len(pos):
                if self.has_opto:
                    #case: using analog modulation with MarkPoints, find rising edges as stim starts
                    stimframes = numpy.searchsorted(self.frametimes * self.fs, pos) - 1  # convert to 0 indexing
                    led_op = pandas.DataFrame({'Intensity': trace[pos] / vmax, 'ImgFrame': stimframes}, index=[pos])
                else:
                    #case: using PWM modulation with external device. cluster within-frame signals and compute duty
                    # PWM is 500 Hz
                    clustering = cluster.DBSCAN(eps=self.fs/500*2, min_samples=2).fit(pos.reshape(-1, 1))
                    labels = clustering.labels_
                    nstims = labels.max() + 1
                    intensities = numpy.zeros(nstims)
                    durations = numpy.zeros(nstims)
                    stimframes = numpy.zeros(nstims, dtype='int')
                    posindex = numpy.zeros(nstims, dtype='int')
                    for cid in range(nstims):
                        x = pos[numpy.where(labels == cid)[0]]  # this is expected to be in order
                        pwmfreq = numpy.diff(x).mean()
                        posindex[cid] = x[0]
                        stimframes[cid] = numpy.searchsorted(self.frametimes, x[0]/self.fs) - 1  # convert to 0 indexing
                        intensities[cid] = bintrace[x[0]:int(x[-1]+pwmfreq)].mean()
                        durations[cid] = (int(x[-1]+pwmfreq) - x[0]) / self.fs * 1000
                    if nstims:
                        self.has_opto = True
                        self.opto_config = f'{round(durations.mean())} ms, {round(intensities.mean() * 100)} %'
                        self.opto_name = 'ExternalDevice'
                        self.found_output.append('Opto')

                        led_op = pandas.DataFrame({'Intensity': intensities,
                                                   'ImgFrame': stimframes, 'Duration': durations}, index=posindex)

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
        self.tm = tm = TreadmillRead.Treadmill(self.dpath, self.prefix)
        if tm.filename is None:
            return None
        self.treadmill_fn = tm.filename
        self.md_keys.append('treadmill_fn')
        #save figure
        if save_fig:
            if tm.pycontrol_version == 1:
                fig, ax = tm.export_plot()
            elif tm.pycontrol_version == 2:
                fig, ax = plt.subplots()
                ax.plot(tm.pos_tX, tm.abspos)
            fig.savefig(self.procpath + self.prefix + '_treadmill.png', dpi=300)
            plt.close()
        # read rSync
        tm_rsync = tm.get_Rsync_times().astype('int') #ms
        sc_rsync = (self.ttl_times * 1000).astype('int') #ms
        try:
            self.align = align = rsync.Rsync_aligner(tm_rsync, sc_rsync)
            skip = False
        except:
            print('Treadmill pulse times:', tm_rsync)
            print('Scope pulse times:', sc_rsync)
            print('RSync align error')
            skip = True
        if not skip:
            self.frame_at_treadmill = align.B_to_A(self.frametimes * 1000)
            numpy.save(self.procpath + self.prefix + '_frame_tm_times.npy', self.frame_at_treadmill)
            # resample speed, pos, laps to scope frames
            indices = self.get_frame_tm_x(self.frame_at_treadmill, tm.pos_tX * 1000)
            op = numpy.empty(len(self.frame_at_treadmill))
            mask = indices > -1
            for Y, tag in zip((tm.smspd, tm.pos, tm.speed, tm.laps), ('smspd', 'pos', 'spd', 'laps')):
                op[:] = numpy.nan
                op[mask] = Y[indices[mask]]
                numpy.save(self.procpath + self.prefix + f'_{tag}.npy', op)
            # add lap and reward number to output
            self.laps = len(tm.laptimes)
            licks = []
            rewards = []
            rzs = []
            rz_started = None
            for event in tm.d.events:
                if numpy.isnan(event.time):
                    continue
                if tm.pycontrol_version == 1:
                    et = event.time
                else:
                    et = int(event.time * 1000) #lookup uses ms
                event_scopetime = align.A_to_B(et) / 1000
                event_frame = int(numpy.searchsorted(self.frametimes, event_scopetime))
                if not 0 < event_frame < len(self.frametimes)-1:
                    continue
                if 'lick' in event.name:
                    licks.append(event_frame)
                elif event.name == 'reward':
                    rewards.append(event_frame)
                elif event.name == 'reward_zone_entry':
                    rz_started = event_frame
                elif event.name == 'reward_timer':
                    if rz_started is not None:
                        rzs.append((rz_started, event_frame))
                    rz_started = None
            if rz_started is not None:
                rzs.append((rz_started, len(self.frametimes)))
            bdat = {'licks': licks, 'rewards': rewards, 'RZ': rzs}
            with open(self.procpath + self.prefix + f'_bdat.json', 'w') as f:
                json.dump(bdat, f)

            self.rewards = len(rewards)
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

    def load_metadata(self):
        for key, value in self.si.info.items():
            self.__setattr__(key, value)


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
    dpath = 'D:\Shares\Data\_RawData\Bruker/testing/treadmill update test/'
    procpath = 'D:\Shares\Data\_Processed/testing/'
    prefix = 'PVTot7_2024-03-14_lfp_127'
    btag = '000'

    if not os.path.exists(procpath):
        os.mkdir(procpath)

    s = PreProc(dpath, procpath, prefix, btag, debug=False, overwrite=True)
    self = s




