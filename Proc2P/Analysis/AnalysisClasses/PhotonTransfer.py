import os
import numpy
import pandas

from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Bruker.LoadMovie import LoadMovie

from Proc2P.utils import lprint, logger
from matplotlib import pyplot as plt

'''
Calculate the photon transfer function of a movie
using: https://github.com/multiphoton-tools/compress-multiphoton/blob/main/notebooks/EvaluatePhotonSensitivity.ipynb

    1.Lees, R. M. et al. Standardised Measurements for Monitoring and Comparing Multiphoton Microscope Systems.
     bioRxiv 2024.01.23.576417 (2024) doi:10.1101/2024.01.23.576417.
Written to interface with ImagingSession for fetching metadata (although only uses raw data) 
'''

# function from repo:
np = numpy
from sklearn.linear_model import HuberRegressor as Regressor


def _longest_run(bool_array):
    """Find the longest contiguous segment of True values inside bool_array.

    Args:
        bool_array: 1d boolean array.

    Returns:
        Slice with start and stop for the longest contiguous block of True values.
    """
    step = np.diff(np.int8(bool_array), prepend=0, append=0)
    on = np.where(step == 1)[0]
    off = np.where(step == -1)[0]
    i = np.argmax(off - on)
    return slice(on[i], off[i])


def compute_sensitivity(movie: np.array, count_weight_gamma=0.2) -> dict:
    """Calculate quantal sensitivity

    Args:
        movie (np.array):  A movie in the format (height, width, time).

    Returns:
        dict: A dictionary with the following keys:
            - 'model': The fitted TheilSenRegressor model.
            - 'min_intensity': Minimum intensity used.
            - 'max_intensity': Maximum intensity used.
            - 'variance': Variances at intensity levels.
            - 'sensitivity': Sensitivity.
            - 'zero_level': X-intercept.
    """
    assert (
            movie.ndim == 3
    ), f"A three dimensional (Height, Width, Time) grayscale movie is expected, got {movie.ndim}"

    movie = np.maximum(0, movie.astype(np.int32, copy=False))
    intensity = (movie[:, :, :-1] + movie[:, :, 1:] + 1) // 2
    difference = movie[:, :, :-1].astype(np.float32) - movie[:, :, 1:]

    select = intensity > 0
    intensity = intensity[select]
    difference = difference[select]

    counts = np.bincount(intensity.flatten())
    bins = _longest_run(counts > 0.01 * counts.mean())  # consider only bins with at least 1% of mean counts
    bins = slice(max(bins.stop * 3 // 100, bins.start), bins.stop)
    assert (
            bins.stop - bins.start > 100
    ), f"The image does not have a sufficient range of intensities to compute the noise transfer function."

    counts = counts[bins]
    idx = (intensity >= bins.start) & (intensity < bins.stop)
    variance = (
            np.bincount(
                intensity[idx] - bins.start,
                weights=(difference[idx] ** 2) / 2,
            )
            / counts
    )
    model = Regressor()
    model.fit(np.c_[bins], variance, counts ** count_weight_gamma)
    sensitivity = model.coef_[0]
    zero_level = - model.intercept_ / model.coef_[0]

    return dict(
        model=model,
        min_intensity=bins.start,
        max_intensity=bins.stop,
        variance=variance,
        sensitivity=sensitivity,
        zero_level=zero_level,
    )


class ExportPhotonTransfer:
    __name__ = 'ExportPhotonTransfer'

    def __init__(self, path, prefix, return_channel=None):
        '''
        :param path: processed path
        :param prefix: prefix(without btag)
        :param return_channel: 'Ch1' or 'Ch2', # Ch1 is red; Ch2 is green. If None, both are exported none is returned
        :return: flux #saves the figure and flux array, also returns it in photons/px/frame
        '''

        # path = r'D:\Shares\Data\_Processed\2P\JEDI/'
        # prefix = 'JEDI-PV21_2024-05-10_Fast_045' #prefix(without btag)
        # channel_name = 'Ch2' #'Ch1' or 'Ch2', # Ch1 is red; Ch2 is green
        # init an ImagingSession
        session = ImagingSession(path, prefix, tag='skip', norip=True)
        dpath = session.si.info['dpath']
        filelist = [fn for fn in os.listdir(dpath) if ((prefix in fn) and ('.ome.tif' in fn))]
        op_df = []
        for channel_name in session.si.info['channelnames']:
            if return_channel is not None:
                if not return_channel == channel_name:
                    continue

            # preload 500 frames of raw uncorrected movie
            input_tiff = [fn for fn in filelist if f'_{channel_name}_' in fn][0]
            raw_movie = LoadMovie(os.path.join(dpath, input_tiff))
            # writing custom loader because tifffile asarray stalls on voltage imaging tiffs with very large page numbers
            start_page = int(10 * session.fps)
            # make sure n is reasonable and within size of file
            mmap = raw_movie.f.pages
            n_pages = min(min(5000, max(500, start_page)),
                          len(mmap) - 2 * start_page)  # len(mmap) may or may not forces to read all
            frameshape = mmap[0].shape
            data = numpy.empty((n_pages, *frameshape[::-1]), dtype=mmap[0].dtype)
            for i in range(n_pages):
                data[i] = mmap[i + start_page].asarray().transpose()
            scan = data.transpose(0, 2, 1)
            # test a preview
            # plt.imshow(numpy.mean(scan, axis=0), aspect='auto')

            # def make_figure(scan, figure_filename, title=None):
            title = prefix
            figure_filename = session.get_file_with_suffix(f'_PhotonTransfer_{channel_name}.png')

            qs = compute_sensitivity(scan.transpose(1, 2, 0))
            print('Quantal size: {sensitivity}\nIntercept: {zero_level}\n'.format(**qs))

            fig, axx = plt.subplots(2, 2, figsize=(8, 12), tight_layout=True)
            q = qs['sensitivity']
            b = qs['zero_level']
            axx = iter(axx.flatten())

            ax = next(axx)
            m = scan.mean(axis=0)
            _ = ax.imshow(m, vmin=0, vmax=np.quantile(m, 0.999), cmap='gray')
            ax.axis(False)
            cbar = plt.colorbar(_, ax=ax, ticks=[0.05, .5, 0.95])
            cbar.remove()
            ax.set_title('average')
            label = "A"
            ax.text(-0.1, 1.15, label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')

            ax = next(axx)
            x = np.arange(qs["min_intensity"], qs["max_intensity"])
            fit = qs["model"].predict(x.reshape(-1, 1))
            ax.scatter(x, np.minimum(fit[-1] * 2, qs["variance"]), s=2, alpha=0.5)
            ax.plot(x, fit, 'r')
            ax.grid(True)
            ax.set_xlabel('intensity')
            ax.set_ylabel('variance')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_title('Photon Transfer Curve')
            label = "B"
            ax.text(-0.1, 1.15, label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')

            ax = next(axx)
            v = ((scan[1:, :, :].astype('float64') - scan[:-1, :, :]) ** 2 / 2).mean(axis=0)
            imx = np.stack(((m - b) / q, v / q / q, (m - b) / q), axis=-1)
            _ = ax.imshow(np.minimum(1, np.sqrt(0.01 + np.maximum(0, imx / np.quantile(imx, 0.9999))) - 0.1),
                          cmap='PiYG')
            cbar = plt.colorbar(_, ax=ax, ticks=[0.2, .5, 0.8])
            cbar.ax.set_yticklabels(['<< 1', '1', '>> 1'])
            ax.axis(False)
            ax.set_title('coefficient of variation')
            label = "C"
            ax.text(-0.1, 1.15, label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')

            ax = next(axx)
            im = (scan.mean(axis=0) - qs['zero_level']) / qs['sensitivity']
            numpy.save(session.get_file_with_suffix(f'_PhotonFlux_{channel_name}.npy'), im)
            mx = np.quantile(im, 0.99)
            _ = ax.imshow(im, vmin=0, vmax=mx, cmap='magma')  # cmap=cc.cm.CET_D13)
            plt.colorbar(_, ax=ax)
            ax.axis(False)
            ax.set_title('Quantum flux\nphotons / pixel / frame');
            label = "D"
            ax.text(-0.1, 1.15, label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')

            plt.suptitle(f'{title or figure_filename} {channel_name}\nPhoton sensitivity: {qs["sensitivity"]:4.1f}')
            fig.savefig(figure_filename, dpi=300)
            log = logger()
            log.set_handle(path, prefix)
            report_keys = ('min_intensity', 'max_intensity', 'sensitivity', 'zero_level')
            output_report = {}
            for key in report_keys:
                output_report[key] = round(qs[key])
            lprint(self, channel_name, output_report, logger=log)
            if return_channel is not None:
                return im, qs
            else:
                op_df.append(pandas.DataFrame(output_report, index=[channel_name]))
        pandas.concat(op_df).to_excel(session.get_file_with_suffix('_PhotonTransfer.xlsx'))


if __name__ == '__main__':
    path = r'D:\Shares\Data\_Processed\2P\JEDI/'
    prefix = 'JEDI-PV21_2024-05-10_Fast_045'
    ExportPhotonTransfer(path, prefix)
