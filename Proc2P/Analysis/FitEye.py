import os, pandas
import numpy
try:
    from ellipse import LsqEllipse
except:
    print('Ellipse fitting not available, use: pip install lsq-ellipse before processing pupil data')

from Video.LoadAvi import LoadAvi
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from PlotTools.Formatting import strip_ax
from statsmodels.tsa.arima.model import ARIMA
from Proc2P.utils import outlier_indices, path_to_list

class FitEye:
    __name__ = 'FitEye'
    '''For loading a coordinate set saved by DeepLabCut, fitting ellipses in each frame, and
     saving the ellipses and the pupil size time series. The markers and the ellipse parameters are filtered to suppress
     outliers.
     '''

    def __init__(self, path, thr=0.2, filter=True, overwrite=False, model_name=None, debug=False, input_pattern='.csv',
                 filt_coord_names='P'):
        '''
        :param path: where the coords are. get from ImagingSession.get_face_path()
        :param thr: likelihood threshold for dropping body part markers.
        :param filter: if True, markers are filtered with ARIMA.
        :param overwrite: if False, loads files from previous run
        :param model_name: look for this subfolder if specified
        :param filt_coord_names: substring of bodyparts that will be included
        '''
        if model_name is None:
            self.path = path
        else:
            self.path = os.path.join(path, model_name)
        self.thr = thr
        self.filter = filter
        filt_str = ('', '_filt')[filter]
        self.ellipse_fn = os.path.join(self.path, f'_ellipse_fit_{int(thr*100)}{filt_str}.npy')
        self.eye_trace_fn = os.path.join(self.path, f'_pupil_trace_{int(thr*100)}{filt_str}.npy')
        self.coords_fn = os.path.join(self.path, f'_pupil_coords_{int(thr*100)}{filt_str}.npy')
        self.filt_coord_names = filt_coord_names
        if not debug:
            if not os.path.exists(self.eye_trace_fn) or overwrite:
                self.load_coords(filter=filter, overwrite=overwrite, pattern=input_pattern)
                if filter:
                    self.fit_ellipse()
                else:
                    self.fit_ellipse_raw()

    def get_trace(self):
        return numpy.load(self.eye_trace_fn)

    def commit_eye_trace(self):
        '''save the current eye trace in the root _face folder.
        ImagingSession will default to read this without specifying model and filtering settings'''
        pardirs = path_to_list(self.path)
        if pardirs[-1] == '_face':
            path = self.path
        elif pardirs[-2] == '_face':
            path = os.path.join(*pardirs[:-1])
        pfn = os.path.join(path, '_pupil_trace_final.npy')
        numpy.save(pfn, self.get_trace())




    def load_coords(self, filter=True, overwrite=False, pattern='.csv'):
        '''
        Load a DLC marker coordinates file from the _face subfolder of the processed session.
        :param filter: if True, performs arima filtfilt on the coords
        :param overwrite: if False and coords already exist, load it from file.
        :param pattern: which DLC file to load if there are multiple. May include model, shuffle and snapshot info.
        :return: ndarray shape(frames, bodyparts, (x,y,z)).
        '''
        filt_coord_names = self.filt_coord_names
        assert filt_coord_names is not None
        if not overwrite and os.path.exists(self.coords_fn):
            self.coords = numpy.load(self.coords_fn)
            return self.coords
        fn = [x for x in os.listdir(self.path) if x.endswith(pattern)]
        if pattern.endswith('_filtered.csv'):
            fn = [x for x in fn if x.endswith(pattern)]
        else:
            fn = [x for x in fn if not x.endswith('_filtered.csv')]
        assert (len(fn) == 1)
        fn = fn[0]
        df = pandas.read_csv(os.path.join(self.path, fn), header=[0,1,2]) #3 levels of labels
        #1st is scorer, 2 is bodypart, 3 is x/y/likelihood
        #reshape to x,y,likelihood of each coord.
        pupil_cols = [col for col in df.columns if filt_coord_names in col[1]]
        df = df[pupil_cols]
        #filter the points
        raw_coords = df.values.reshape((df.shape[0], df.shape[1]//3, 3))
        if filter:
            filt_coords = numpy.empty(raw_coords.shape)
            filt_coords[:] = numpy.nan
            for i in range(raw_coords.shape[1]):
                for j in (0, 1):
                    filt_coords[:, i, j] = arima_filtfilt(raw_coords[:, i, j], thr=self.thr)
            filt_coords[:, :, 2] = raw_coords[:, :, 2]
            self.coords = filt_coords
        else:
            self.coords = raw_coords
        numpy.save(self.coords_fn, self.coords)
        return self.coords

    def fit_ellipse(self):
        '''Fit based on filtered bodyparts, then filter the ellipse parameters'''
        ED = numpy.empty((len(self.coords), 5)) #center(x,y), width, height, phi
        ED[:] = numpy.nan
        #fit the ellipse in each frame, using the filtered body part coordinates.
        nan_frames = []
        for f in range(len(self.coords)):
            XY = self.coords[f, :, :2]
            Z = self.coords[f, :, 2]
            good_markers = Z>self.thr
            if numpy.count_nonzero(good_markers) > 5: #if too many markers bad, skip fitting for this frame
                # if numpy.count_nonzero(good_markers) < 5: #ellipse fitting needs at least 5 markers, use 5 best
                #     good_markers = numpy.argsort(Z)[-5:]
                XY = self.coords[f, good_markers, :2]
                reg = LsqEllipse().fit(XY)
                center, width, height, phi = reg.as_parameters()
                ED[f, :] = [*center, width, height, phi]
            else:
                nan_frames.append(f)
        ED_filt = numpy.copy(ED)
        #filter each ellipse paramater
        # ED_filt = numpy.empty(ED.shape)
        # ED_filt[:] = numpy.nan
        # for i in range(ED.shape[1]):
        #     ED_filt[:, i] = arima_filtfilt(ED[:, i])
        numpy.save(self.ellipse_fn, ED_filt)
        ellipse_diameter = (ED_filt[:, 2]+ED_filt[:, 3])/2
        ellipse_diameter[nan_frames] = numpy.nan
        numpy.save(self.eye_trace_fn, ellipse_diameter) #pupil diameter in each frame

    def fit_ellipse_raw(self):
        '''This doesn't filter the points time series, but drops them based on goodness of fit'''
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
        self.load_coords(filter=self.filter)
        ED = numpy.load(self.ellipse_fn)
        eye_trace = self.get_trace()

        if numpy.count_nonzero(numpy.logical_not(numpy.isnan(eye_trace))) < 10:
            return -1

        # get 9 frames
        notna_indices = numpy.where(numpy.logical_not(numpy.isnan(eye_trace)))[0]
        pick_9 = [int(len(notna_indices) * ((x / 10)+0.05)) for x in range(9)]
        test_frames = notna_indices[numpy.argsort(eye_trace[notna_indices])[pick_9]]

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
            if cropping is None:
                fdat = frame
            else:
                fdat = frame[cropping[2]:cropping[3], cropping[0]:cropping[1]]
            ca.imshow(fdat, vmin=vmin, vmax=vmax)

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

def arima_filtfilt(trace, z=None, thr=0.2):
    '''
    run a forward and backward filter to remove outliers and nans
    :param trace: input time series (1D)
    :param z: the goodness of fit can be passed from DLC to exclude uncertain body parts
    :param thr: z < thr will be excluded
    :return: filtered time series, same shape as input
    '''
    Y = trace
    y = numpy.copy(Y)
    #mask outliers and the frames before and after them
    diff_indices = [x for x in outlier_indices(numpy.diff(Y)) if x < (len(Y)-2)]
    y[diff_indices] = numpy.nan
    y[[x + 1 for x in diff_indices]] = numpy.nan
    y[[x + 2 for x in diff_indices]] = numpy.nan
    if z is not None:
        y[z<thr] = numpy.nan
    #filter
    order = (1, 1, 1)  # dorpna not compatible with all models, this works fine but slows responses a bit.
    ma_model = ARIMA(y, order=order, missing='drop', )
    model_fit = ma_model.fit()
    result = model_fit.predict()
    result[0] = Y[0]
    #reverse filter
    ma_model = ARIMA(y[::-1], order=order, missing='drop', )
    model_fit = ma_model.fit()
    result_rev = model_fit.predict()
    result_rev[0] = Y[-1]
    #average
    return (result + result_rev[::-1]) / 2
