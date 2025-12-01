import os
from matplotlib import pyplot as plt
import numpy
import json
from dataclasses import dataclass, asdict
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.utils import *
from Proc2P.Bruker.LoadMovie import LoadMovie, get_raw_movies
from Proc2P.Bruker.PreProc import SessionInfo
from skimage.registration import phase_cross_correlation
from skimage import transform
from scipy.signal import bessel, sosfiltfilt
from tifffile import imwrite
from numpy.lib.format import open_memmap

'''
Rigid motion correction for high frame rate (GEVI) movies
Each frame is rigid, and consecutive frames are not independent - displacements are smoothed 
'''

@dataclass
class RegConfig:
    ref_size = 1000  # frames
    min_ref_size = 500  # frames
    rolling_avg_size = 20  # ms
    ref_displacement_limit = 1 #pixel
    subpixel_precision = 10 # fraction of pixel
    ref_margin = 5  # sec
    displacement_lowpass = 60 #Hz
    disps_suffix = '_disps.npy'
    filt_disps_suffix = '_disps_filt.npy'
    setting_suffix = '_regconfig.json'
    registered_suffix = '_registered.npy'


# this one is constant running (hard example)
proc_path = r'D:\Shares\Data\_Processed/2P\JEDI-IPSP/'
prefix = 'JEDI-Sncg73_2024-11-19_burst_033'

# this one already looks good with S2P (easy example)
prefix = 'JEDI-Sncg122_2025-04-24_baseline_605'
scratch_path = 'E:\MCC\BD'
verbose = True

trace = numpy.load(os.path.join(proc_path, prefix, prefix + '_trace_GRtest.npy'))
plt.plot(trace[1, :, 0])
assert False

# load raw movie
if verbose:
    lprint(None, 'Opening input')
use_reference = 'S2P'
config = RegConfig()
session = ImagingSession(proc_path, prefix, tag='skip')
info = session.si
channelnames = info["channelnames"]
ref_channel = channelnames[0]  # for now
config.registered_suffix = f'_registered_{ref_channel}.npy'
fps = session.fps

memmap_fn = os.path.join(scratch_path, prefix + '_raw.npy')
oPath = touch_path(proc_path, prefix, 'GEVIReg')
disps_fn = os.path.join(oPath, prefix + config.disps_suffix)
if not os.path.exists(memmap_fn):
    if verbose:
        lprint(None, 'Reading and cacheing raw movie')
    movies = get_raw_movies(info)
    numpy.save(memmap_fn, movies[ref_channel])

data = numpy.load(memmap_fn, mmap_mode='r')
n_frames = info["n_frames"]
height, width = data.shape[1:]

# compute first reference.
if verbose:
    lprint(None, 'Computing reference')
# take frames when mouse is stationary (complete with random frames if less than min number)
exclude_move = gapless(session.pos.speed, int(fps), threshold=1, expand=int(fps))
bad_frames = session.ca.sync.load('opto', suppress=True)
exclude_move[bad_frames] = True
ref_margin = int(min(n_frames / 3, config.ref_margin * fps))
ref_frames = numpy.where(~exclude_move[ref_margin:-ref_margin])[0] + ref_margin
if len(ref_frames) > config.ref_size:
    ref_frames = numpy.random.choice(ref_frames, config.ref_size, replace=False)
elif len(ref_frames) < config.min_ref_size:
    more_frames = [x for x in numpy.arange(ref_margin, n_frames - ref_margin) if
                   ((x not in ref_frames) and (x not in bad_frames))]
    add_frames = numpy.random.choice(more_frames, min(config.ref_size - len(ref_frames), len(more_frames)),
                                     replace=False)
    ref_frames = numpy.concatenate((ref_frames, add_frames))

first_ref = numpy.average(data[ref_frames], axis=0)
norm_min = numpy.min(first_ref)
norm_min = numpy.percentile(data[ref_frames], 1)
norm_range = numpy.percentile(data[ref_frames], 99) - norm_min


def norm_img(img, imin, irange):
    return numpy.clip((img - imin) / irange, 0, 1)

d = norm_img(data, norm_min, norm_range)
r = norm_img(first_ref, norm_min, norm_range)
snr = numpy.nan_to_num(numpy.average(d, axis=(1, 2)) / numpy.std(d, axis=(1, 2)))
ref_frames = [x for x in ref_frames if snr[x] > 1]

avgs = numpy.zeros([len(ref_frames), *first_ref.shape])
n_roll = int(config.rolling_avg_size / 1000 * fps)
disps = numpy.zeros((len(ref_frames), 4))
stable_refs = []
for i, f in enumerate(ref_frames):
    i0 = max(0, f - n_roll)
    i1 = min(f + n_roll, n_frames)
    weighted_sample = d[i0:i1] * snr[i0:i1, None, None]
    avg_sample = numpy.sum(weighted_sample, axis=0) / numpy.sum(snr[i0:i1])
    displacements = phase_cross_correlation(avg_sample, r, upsample_factor=10)  # fit ref to target,
    disps[i, :2] = displacements[0]
    disps[i, 2:] = displacements[1:]
    if max([abs(x) for x in displacements[0]]) < 2:
        tforms = transform.SimilarityTransform(translation=displacements[0])
        avgs[i] = transform.warp(data[f], tforms, order=1, preserve_range=True).astype('float')
        stable_refs.append([i])
refined_reference = numpy.average(avgs[stable_refs], axis=0)[0]
avgmax = numpy.max(avgs[stable_refs], axis=0)
r = norm_img(refined_reference, norm_min, norm_range)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ca = ax[0, 0]
strip_ax(ca)
ca.imshow(first_ref)
ca.set_title('First reference')

ca = ax[0, 1]
strip_ax(ca)
ca.imshow(refined_reference)
ca.set_title('Refined reference')

# init the reg timeseries
disps = numpy.empty((n_frames, 2))
f_disp = numpy.empty(disps.shape)

# estimate raw displacement (single frames)
if not os.path.exists(disps_fn):
    if verbose:
        lprint(None, 'Estimating displacements')
    for i in range(len(d)):
        i0 = max(0, i - n_roll)
        i1 = min(i + n_roll, n_frames)
        weighted_sample = d[i0:i1] * snr[i0:i1, None, None]
        avg_sample = numpy.sum(weighted_sample, axis=0) / numpy.sum(snr[i0:i1])
        displacements = phase_cross_correlation(avg_sample, r, upsample_factor=config.subpixel_precision)  # fit ref to target,
        disps[i, :2] = displacements[0]
    numpy.save(disps_fn, disps)

else:
    disps = numpy.load(disps_fn)

# estimate smoothed displacement (rolling average)
nyq = 0.5 * fps
cutoff = config.displacement_lowpass / nyq
filter = bessel(3, cutoff, btype='lowpass', output='sos')
for i in range(2):
    f_disp[:, i] = sosfiltfilt(filter, disps[:, i])

# f3, a3 = plt.subplots(nrows=3, sharex=True)
# for i in range(2):
#     a3[i].plot(disps[:, i])
# for highcut in (20, 30, 40, 60):
#     nyq = 0.5 * fps
#     cutoff = highcut / nyq
#     filter = bessel(3, cutoff, btype='lowpass', output='sos')
#     f_disp = numpy.empty(disps.shape)
#     for i in range(2):
#         f_disp[:, i] = sosfiltfilt(filter, disps[:, i])
#         a3[i].plot(f_disp[:, i], label=f'{highcut} Hz')
# a3[1].legend()
# a3[2].plot(session.pos.speed)


# apply displacements and save output
if verbose:
    lprint(None, 'Applying displacements')
numpy.save(disps_fn.replace(config.disps_suffix, config.filt_disps_suffix), f_disp)
with open(os.path.join(oPath, prefix+config.setting_suffix), "w") as f:
    json.dump(asdict(config), f, indent=2)
# reg_movie = numpy.memmap(os.path.join(oPath, prefix+config.registered_suffix),
#                          shape=data.shape, dtype=numpy.uint16, mode='write')
# reg_movie = numpy.empty(shape=data.shape, dtype=numpy.uint16)
reg_movie = open_memmap(os.path.join(oPath, prefix+config.registered_suffix), mode='w+',
                        shape=data.shape, dtype=numpy.uint16,)
for i, frame in enumerate(data):
    tforms = transform.SimilarityTransform(translation=f_disp[i])
    reg_movie[i] = transform.warp(frame, tforms, order=1, mode='symmetric', preserve_range=True)
reg_movie.flush()
# numpy.save(os.path.join(oPath, prefix+config.registered_suffix), reg_movie)

if verbose:
    lprint(None, 'Saving previews')
#add ref frames and all frames previews to the image
ca = ax[1, 0]
strip_ax(ca)
ca.imshow(reg_movie[ref_frames].mean(axis=0))
ca.set_title('Output (ref frames)')

sum_disp = f_disp[:, 0] - f_disp[:, 0].mean()
sum_disp += f_disp[:, 1] - f_disp[:, 1].mean()
spaced_frames = numpy.linspace(config.ref_margin, n_frames-config.ref_margin, config.ref_size).astype('int64')
sort_disp = numpy.argsort(sum_disp)
exp_indices = sort_disp[spaced_frames]
ca = ax[1, 1]
strip_ax(ca)
ca.imshow(reg_movie[exp_indices].mean(axis=0))
ca.set_title('Output (sampled)')

fig.savefig(os.path.join(oPath, prefix + '_registration_preview.png'), dpi=300)

#save preview tif RGB
preview_channels = {'R': None, 'G': None, 'B': None}
pw_indices = sort_disp[int(n_frames*0.5-config.min_ref_size):int(n_frames*0.5+config.min_ref_size)]
mean1 = reg_movie[pw_indices].mean(axis=0)
for pw_img, suffix in zip((mean1, avgmax), ('_GEVIRegPreview.tif', '_GEVIRegAvgMax.tif')):
    previewname = os.path.join(session.path, prefix + suffix)
    preview = numpy.zeros([*pw_img.shape, 3], dtype='uint8')
    preview_channels['G'] = pw_img / pw_img.max() * 255
    for chi, chn in enumerate('RGB'):
        x = preview_channels[chn]
        if x is not None:
            preview[..., chi] = x
    imwrite(previewname, preview)

