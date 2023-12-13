from Proc2P.Analysis import ImagingSession, read_excel
import numpy
from sklearn.cluster import dbscan

'''
Load the list of stimulus frames and intensities, provide functions to get events (train start)
'''

class PhotoStim:

    def __init__(self, session:ImagingSession):
        self.session = session
        self.stimframes = read_excel(session.path + session.prefix + '_StimFrames.xlsx')

    def get_trains(self, isi=10, tolerance=1.5):
        '''
        Build a list of pulse trains by clustering events.
        :param isi: inter pulse interval within trains (frames)
        :param tolerance: isi is multiplied to allow for some jitter
        :return: train start frames, pulse intensity by train
        '''
        ft = self.stimframes['ImgFrame'].values
        fi = self.stimframes['Intensity'].values
        clustering = dbscan(ft.reshape(-1, 1), eps=isi*tolerance, min_samples=int(1))
        n_trains = clustering[1].max() + 1
        train_starts = numpy.ones(n_trains, dtype='int32') * -1  # start frame
        train_intensity = numpy.empty(n_trains)  # , intensity
        train_intensity[:] = numpy.nan
        for i in range(n_trains):
            train_starts[i] = ft[numpy.searchsorted(clustering[1], i)]
            train_intensity[i] = fi[clustering[1] == i].mean()
        return train_starts, train_intensity


