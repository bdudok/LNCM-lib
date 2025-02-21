import os, pandas
import numpy
try:
    from ellipse import LsqEllipse
except:
    print('Ellipse fitting not available, use: pip install lsq-ellipse')

class FitEye:
    __name__ = 'FitEye'
    '''For loading a coordinate set saved by DeepLabCut, fitting ellipses in each frame, and
     saving the ellipses and the pupil size time series'''

    def __init__(self, path, thr=0.2):
        '''
        :param path: where the coords are. get from ImagingSession.get_face_path()
        :param thr: likelihood threshold.
        '''
        self.path = path
        self.thr = thr
        self.ellipse_fn = os.path.join(self.path, f'_ellipse_fit_{int(thr*100)}.npy')
        self.eye_trace_fn = (os.path.join(self.path, f'_eye_trace_{int(thr*100)}.npy'))
        if not os.path.exists(self.eye_trace_fn):
            self.fit_ellipse()

    def get_trace(self):
        return numpy.load(self.eye_trace_fn)

    def load_coords(self):
        fn = None
        for x in os.listdir(self.path):
            if x.endswith('.csv'):
                fn = x
                break
        assert fn is not None
        df = pandas.read_csv(os.path.join(self.path, fn), header=[0,1,2]) #3 levels of labels
        #1st is scorer, 2 is bodypart, 3 is x/y/likelihood
        #reshape to x,y,likelihood of each coord.
        pupil_cols = [col for col in df.columns if 'pupil' in col[1]]
        df = df[pupil_cols]
        self.coords = df.values.reshape((df.shape[0], df.shape[1]//3, 3))

    def fit_ellipse(self):
        self.load_coords()
        ED = numpy.empty((len(self.coords), 5)) #center(x,y), width, height, phi
        ED[:] = numpy.nan
        for f in range(len(self.coords)):
            Z = self.coords[f, :, 2]
            XY = self.coords[f, Z>self.thr, :2]
            if len(XY)<5:
                continue
            reg = LsqEllipse().fit(XY)
            center, width, height, phi = reg.as_parameters()
            ED[f, :] = [*center, width, height, phi]
        numpy.save(self.ellipse_fn, ED)
        numpy.save(self.eye_trace_fn, (ED[:, 2]+ED[:, 3])/2) #pupil diameter in each frame
