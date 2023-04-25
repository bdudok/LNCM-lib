import os
from _Dependencies.PyControl.code.tools import data_import, session_plot
from ConfigVars import TR


class Treadmill:

    def __init__(self, path, prefix, calib = None):
        '''
        Checks if converted file exists. Reads it if yes. Converts raw PyControl data to it if not.
        Requires the PyControl repo installed under _Dependencies (see readme there)
        :param path: data folder
        :param prefix: session name (ending with timestamp)
        '''
        self.path = path
        self.prefix = prefix
        self.flist = os.listdir(path)
        d = data_import.Session(self.path + self.prefix + '.txt')
        d.analog = {}
        for f in self.flist:
            if self.prefix in f and f.endswith('.pca'):
                tag = (f.split('_')[-1].split('.')[0])
                d.analog[tag] = data_import.load_analog_data(self.path + f)
        #store calibrated position
        if calib is None:
            self.calib = TR.calib_cm
        else:
            self.calib = calib
        d.pos_cm = d.analog['pos'][:, 1] / self.calib

        #convert to speed

        #parse laps

    def get_startstop(self):
        pass


