import os

import numpy
import numpy as np
import pandas as pd
from _Dependencies.PyControl.code.tools import data_import, session_plot
from .ConfigVars import TR


class Treadmill:

    def __init__(self, path, prefix, calib=None):
        '''
        Checks if converted file exists. Reads it if yes. Converts raw PyControl data to it if not.
        Requires the PyControl repo installed under _Dependencies (see readme there)
        :param path: data folder
        :param prefix: session name (ending with timestamp)
        '''
        self.path = path
        # find pycontrol filename
        ext = '.txt'
        if not os.path.exists(path + prefix + ext):
            longest = 0
            for fn in os.listdir(path):
                if prefix in fn and fn.endswith(ext) and '.log.' not in fn:
                    if longest < len(fn):
                        prefix = fn[:-len(ext)]
                        longest = len(fn)
        self.prefix = prefix
        self.filename = prefix + '.txt'
        if not os.path.exists(self.path+self.filename):
            print('Treadmill files not found')
            self.filename = None
            return None
        self.flist = os.listdir(path)
        self.d = d = data_import.Session(self.path + self.filename)

        d.analog = {}
        for f in self.flist:
            if self.prefix in f and f.endswith('.pca'):
                tag = (f.split('_')[-1].split('.')[0])
                d.analog[tag] = data_import.load_analog_data(self.path + f)
        # store calibrated position
        if calib is None:
            self.calib = TR.calib_cm
        else:
            self.calib = calib
        self.abspos = d.analog['pos'][:, 1] / self.calib  # cms
        self.pos_tX = d.analog['pos'][:, 0] / 1000  # seconds

        # convert to speed
        spd = np.diff(self.abspos)
        # remove overflow effect
        wh_turn = numpy.where(abs(spd) > 100)
        spd[wh_turn] = (spd[wh_turn[0] - 1] + spd[wh_turn[0] + 1]) / 2
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
        for event in d.print_lines:
            if 'lap_counter' in event:
                any_lap = True
                e_time = int(event.split(' ')[0])
                e_idx = np.searchsorted(d.analog['pos'][:, 0], e_time)
                if e_idx < len(self.abspos): #summary print at session end should be ignored
                    self.lapends.append(e_idx)
                    self.laptimes.append(e_time)
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
        return numpy.array(t)

    def export_plot(self):
        return session_plot.session_plot(self.path + self.filename, fig_no=1, return_fig=True)
