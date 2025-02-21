import os, pandas
import numpy
try:
    from ellipse import LsqEllipse
except:
    print('Ellipse fitting not available, use: pip install lsq-ellipse')

from Video.LoadAvi import LoadAvi
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from PlotTools.Formatting import strip_ax

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
            good_markers = Z>self.thr
            if numpy.count_nonzero(good_markers) < 5: #ellipse fitting needs at least 5 markers, use 5 best
                good_markers = numpy.argsort(Z)[-5:]
            XY = self.coords[f, good_markers, :2]
            reg = LsqEllipse().fit(XY)
            center, width, height, phi = reg.as_parameters()
            ED[f, :] = [*center, width, height, phi]
        numpy.save(self.ellipse_fn, ED)
        numpy.save(self.eye_trace_fn, (ED[:, 2]+ED[:, 3])/2) #pupil diameter in each frame

    def export_pupil_fits(self, vid_fn, cropping, save_fn):
        '''
        Take deciles of the diameter distribution, get the corresponding frame, and draw the ellipse on it
        saves the plot in the face path
        '''
        self.load_coords()
        ED = numpy.load(self.ellipse_fn)
        eye_trace = self.get_trace()

        # get 9 frames
        pick_8 = [int(len(eye_trace) * (x / 10)) for x in range(8)]
        pick_8.append(-1)
        test_frames = numpy.argsort(eye_trace)[pick_8]

        # load movie
        vid = LoadAvi(vid_fn)
        fig, ax = plt.subplots(3, 3, figsize=(16, 9))
        fig.patch.set_facecolor('black')
        vmin = vid.frame.min()
        vmax = vid.frame.max()
        for ca, fr in zip(ax.flat, test_frames):
            strip_ax(ca)
            ca.axis('equal')
            ca.set_facecolor('black')
            frame = vid[fr]
            ca.imshow(frame[cropping[2]:cropping[3], cropping[0]:cropping[1]], vmin=vmin, vmax=vmax)

            # plot the coords
            XY = self.coords[fr]
            ca.plot(XY[:, 0], XY[:, 1], 'ro', zorder=1)
            center_x, center_y, width, height, phi = ED[fr]
            ellipse = Ellipse(
                xy=[center_x, center_y], width=2 * width, height=2 * height, angle=numpy.rad2deg(phi),
                edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
            )
            ca.add_patch(ellipse)
        fig.tight_layout()
        fig.savefig(save_fn, dpi=300)
        plt.close()
