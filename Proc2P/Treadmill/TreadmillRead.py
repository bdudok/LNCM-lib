import os

import numpy
import numpy as np
import pandas as pd
from _Dependencies.PyControlv1.code.tools import data_import, session_plot
from _Dependencies.PyControlv2.code.tools import data_import as data_importv2
from _Dependencies.PyControlv2.code.tools import session_plot as session_plotv2
from Proc2P.Treadmill.ConfigVars import TR


class Treadmill:

    def __init__(self, path, prefix, calib=None):
        '''
        Checks if converted file exists. Reads it if yes. Converts raw PyControl data to it if not.
        Requires the PyControl repo installed under _Dependencies (see readme there)
        :param path: data folder
        :param prefix: session name (ending with timestamp)
        '''
        self.path = path
        self.pycontrol_version = 0
        self.prefix = prefix
        self.flist = os.listdir(path)
        # find pycontrol filename
        exts = ('.txt', '.tsv')
        longest = 0
        long_prefix = prefix
        for ext in exts:
            for fn in self.flist:
                if prefix in fn and fn.endswith(ext) and '.log.' not in fn:
                    if longest < len(fn):
                        long_prefix = fn[:-len(ext)]
                        longest = len(fn)
        v1fn = long_prefix + exts[0]
        v2fn = long_prefix + exts[1]
        if v1fn in self.flist:
            self.prefix = prefix
            self.filename = v1fn
            self.pycontrol_version = 1
            self.data_import_function = data_import
            self.session_export_function = session_plot
        elif v2fn in self.flist:
            self.filename = v2fn
            self.pycontrol_version = 2
            self.data_import_function = data_importv2
            self.session_export_function = session_plotv2
        else:
            print('Treadmill files not found')
            self.filename = None
            return None
        if calib is None:
            self.calib = TR.calib_cm
        else:
            self.calib = calib

        self.d = d = self.data_import_function.Session(self.path + self.filename)
        d.analog = {}
        if self.pycontrol_version == 1:
            for f in self.flist:
                if self.prefix in f and f.endswith('.pca'):
                    tag = (f.split('_')[-1].split('.')[0])
                    d.analog[tag] = data_import.load_analog_data(self.path + f)
            # store calibrated position
            self.abspos = d.analog['pos'][:, 1] / self.calib  # cms
            self.pos_tX = d.analog['pos'][:, 0] / 1000  # seconds
        elif self.pycontrol_version == 2:
            # we won't search for multiple analog races, implement later if necessary
            self.abspos = numpy.load(self.path+self.filename[:-4]+'_pos.data.npy') / self.calib # cms
            self.pos_tX = numpy.load(self.path+self.filename[:-4]+'_pos.time.npy')  # seconds

            # convert to speed
        spd = np.diff(self.abspos)
        # remove overflow effect
        wh_turn = numpy.where(abs(spd) > 100)[0]
        spd[wh_turn] = (spd[wh_turn - 1] + spd[wh_turn + 1]) / 2
        # make abspos continously increasing at overflows:
        for t_idx in wh_turn:
            self.abspos[t_idx+2:] = self.abspos[t_idx] + self.abspos[t_idx+2:]-self.abspos[t_idx+2]
            self.abspos[t_idx+1] = (self.abspos[t_idx] + self.abspos[t_idx+2]) / 2
        rate = np.diff(self.pos_tX)  # in ms
        self.speed = np.empty(len(self.abspos))
        self.speed[0] = 0
        self.speed[1:] = spd / rate
        self.smspd = np.asarray(pd.DataFrame(self.speed).ewm(span=1 / rate.mean()).mean())[:, 0]

        # parse laps
        self.lapends = []
        self.laptimes = []
        self.laps = numpy.zeros(len(self.abspos), dtype='uint8')
        any_lap = False
        if hasattr(d, 'print_lines'): #v1
            for event in d.print_lines:
                if 'lap_counter' in event:
                    e_time = int(event.split(' ')[0])
                    e_idx = np.searchsorted(d.analog['pos'][:, 0], e_time)
                    if e_idx < len(self.abspos): #summary print at session end should be ignored
                        any_lap = True
                        self.lapends.append(e_idx)
                        self.laptimes.append(e_time)
        elif hasattr(d, 'variables_df'): #v2
            any_lap = numpy.any(d.variables_df.loc[:, ('values', 'lap_counter')].gt(0))
            if any_lap:
                curr_lap = 0
                for _, event in d.variables_df.iterrows():
                    this_lap = event[('values', 'lap_counter')]
                    if not numpy.isnan(this_lap) and this_lap > curr_lap:
                        e_time = float(event['time'][0])
                        e_idx = np.searchsorted(self.pos_tX, e_time)
                        if e_idx < len(self.abspos): #summary print at session end should be ignored
                            self.lapends.append(e_idx)
                            self.laptimes.append(e_time)
                            curr_lap = this_lap
        if any_lap:
            # find reset position, make that 0
            self.laplen = np.median(np.diff(self.abspos[self.lapends]))
            self.pos = self.abspos + self.laplen - self.abspos[self.lapends[0]]
            for e_idx in self.lapends:
                self.pos[e_idx:] -= self.pos[e_idx]
                self.laps[e_idx - 1:] += 1
            self.relpos = np.minimum(1, self.pos / self.laplen)

        else:
            self.pos = self.abspos - self.abspos.min()
            self.relpos = self.pos / TR.beltlen

    def get_startstop(self):
        pass

    def get_Rsync_times(self):
        t = []
        for event in self.d.events:
            if event.name == 'rsync':
                t.append(event.time)
        if self.pycontrol_version == 2:
            return (numpy.array(t)*1000).astype('int')
        return numpy.array(t).astype('int')

    def export_plot(self):
        if self.pycontrol_version == 1:
            return self.session_export_function.session_plot(self.path + self.filename, fig_no=1, return_fig=True)
        else:
            return None


if __name__ == '__main__':
    dpath = 'D:\Shares\Data\_RawData\Bruker/testing/treadmill update test/JEDI-PV18_2024-03-13_Fast_043-000/'
    prefix = 'JEDI-PV18_2024-03-13_Fast_043'
    tm = Treadmill(dpath, prefix)

