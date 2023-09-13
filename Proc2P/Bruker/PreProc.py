import json
import numpy
import os
import pandas
import xml.etree.ElementTree as ET
from Proc2P.Treadmill import TreadmillRead, rsync

'''
PreProcess: load xml files, save all frame times, sync signals and metadata into the analysis folder
'''

class PreProc:
    def __init__(self, dpath, procpath, prefix, btag = '000'):
        self.dpath = os.path.join(dpath, prefix + f'-{btag}/')#raw data
        self.procpath = procpath + prefix + '/' #output processed data
        self.prefix = prefix #recorder prefix tag
        self.btag = btag #tag appended by bruker (000 by default)

        #setup config
        self.led_channel = 1
        self.rsync_channel = 0

        self.md_keys = ['led_channel', 'rsync_channel', 'btag', 'dpath']

        if not os.path.exists(self.procpath):
            os.mkdir(self.procpath)

        #init sessioninfo, preprocess if empty
        self.si = SessionInfo(self.procpath, prefix)
        if not self.si.load():
            self.skip_analysis = False
            self.preprocess()


    def preprocess(self):
        '''
        Convert the XML and voltage outputs to frametimes, timestamped events
        Saves bad_frames, FrameTimes, StimFrames, 2pTTLtimes, and sessiondata
        '''
        self.parse_frametimes()
        self.parse_TTLs()
        self.parse_treadmill()
        self.save_metadata()

    def parse_frametimes(self):
        #find xml filename
        xfn = os.path.join(self.dpath, self.prefix+f'-{self.btag}.xml')
        tree = ET.parse(xfn)
        root = tree.getroot()
        sequence = root.find('Sequence')

        #get 2P info
        sdict = {}
        get_keys = ('framePeriod', 'scanLinePeriod')
        for child in list(sequence.find('Frame').find('PVStateShard')):
            key = child.attrib['key']
            if key in get_keys:
                sdict[key] = child.attrib['value']
        self.framerate = 1/float(sdict['framePeriod'])
        if 'scanLinePeriod' in sdict:
            self.linetime = float(sdict['scanLinePeriod']) #not in 2channel files.
        else:
            self.linetime = numpy.nan
            self.skip_analysis = True
            print(self.prefix, 'XML parse error.')
        #later include number of channels


        #get voltage info
        voltage = sequence.find('VoltageRecording')
        self.voltage_config = voltage.attrib['configurationFile']
        self.voltage_data = voltage.attrib['dataFile']
        self.voltage_name = voltage.attrib['name']

        #get opto info
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
            dfs.append(pandas.DataFrame([[float(frame.get(key)) for key in keys]], columns=keys, index=[frame.get('index')]))
        ft_op = pandas.concat(dfs)
        ft_op.sort_values('absoluteTime', inplace=True)
        ft_op.to_excel(self.procpath + self.prefix + '_FrameTimes.xlsx')
        numpy.save(self.procpath + self.prefix + '_FrameTimes.npy', ft_op.values)
        self.frametimes = ft_op['relativeTime'].values
        self.n_frames = len(dfs)

        #append metadata attribs
        self.md_keys.extend(['n_frames', 'voltage_name', 'has_opto', 'opto_name', 'framerate', 'linetime'])

    def parse_TTLs(self):
        vdat = pandas.read_csv(self.dpath+self.voltage_data)

        #get fs
        tree = ET.parse(self.dpath + self.voltage_config)
        root = tree.getroot()
        experiment = root.find('Experiment')
        rate = experiment.find('Rate')
        self.fs = int(rate.text)

        #LED pulses
        trace = vdat[f' Input {self.led_channel}'].values
        vmax = 5.0 #5V command is 100%LED
        pos = numpy.where(numpy.convolve(trace > vmax*0.05, [1, -1]) == 1)[0]
        stimframes = numpy.searchsorted(self.frametimes*self.fs, pos)-1 #convert to 0 indexing
        led_op = pandas.DataFrame({'Intensity': trace[pos]/vmax, 'ImgFrame': stimframes}, index=[pos])
        led_op.to_excel(self.procpath + self.prefix + '_StimFrames.xlsx')
        numpy.save(self.dpath+'bad_frames.npy', stimframes)
        numpy.save(self.procpath + self.prefix + '_bad_frames.npy', stimframes)

        #rsync times
        trace = vdat[f' Input {self.rsync_channel}'].values
        vmax = 3.3
        pos = numpy.where(numpy.convolve(trace > vmax*0.5, [1, -1]) == 1)[0]
        self.ttl_times = pos / self.fs
        numpy.save(self.procpath+self.prefix+'_2pTTLtimes.npy', self.ttl_times)

        ##append metadata attribs
        self.md_keys.append('fs')

    def parse_treadmill(self):
        tm = TreadmillRead(self.dpath, self.prefix)
        # read rSync
        tm_rsync = tm.get_Rsync_times()
        sc_rsync = (self.ttl_times * 1000).astype('int')
        align = rsync.Rsync_aligner(tm_rsync, sc_rsync)
        self.frame_at_treadmill = align.B_to_A(self.frametimes*1000)
        numpy.save(self.procpath + self.prefix + '_frame_tm_times.npy', self.frame_at_treadmill)
        #resample speed, pos to scope frames
        indices = self.get_frame_tm_x(self.frame_at_treadmill*1000, tm.pos_tX)
        for Y, tag in zip((tm.smspd, tm.pos), ('smspd', 'pos')):
            op = numpy.empty(len(self.frame_at_treadmill))
            op[:] = numpy.nan
            mask = indices > -1
            op[mask] = Y[indices[mask]]
            numpy.save(self.procpath + self.prefix + f'_{tag}.npy', op)
        #add lap and reward number to output
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
    procpath = 'D:\Shares\Data\_Processed/2P\PVTot\Opto/'
    prefix = 'PVTot6_2023-08-31_opto_019'
    btag = '000'

    md = PreProc(dpath, procpath, prefix,)
    # md.preprocess()

    # si = SessionInfo(procpath+prefix+'/', prefix)
    # si.load()


