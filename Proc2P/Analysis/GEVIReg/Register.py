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
    ref_size: int = 1000  # frames
    min_ref_size: int = 500  # frames
    rolling_avg_size: float = 10  # ms
    ref_displacement_limit: float = 1  # pixel
    subpixel_precision: float = 10  # fraction of pixel
    ref_margin: float = 5  # sec
    displacement_lowpass: float = 60  # Hz
    disps_suffix: str = '_disps.npy'
    filt_disps_suffix: str = '_disps_filt.npy'
    setting_suffix: str = '_regconfig.json'
    registered_suffix: str = '_registered.npy'
    scratch_path: str = 'E:\MCC\BD'


proc_path = r'D:\Shares\Data\_Processed/2P\JEDI-IPSP/'
prefix = 'JEDI-Sncg124_2025-04-24_burst_613'
config = RegConfig()

def register(proc_path, prefix, config):
# if True:
    verbose = True
    debug = True

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

    memmap_fn = os.path.join(config.scratch_path, prefix + '_raw.npy')
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
    if debug:
        print(f'After exclude move and bad frames, {len(ref_frames)} ref frames of {n_frames}')
    if len(ref_frames) > config.ref_size:
        ref_frames = numpy.random.choice(ref_frames, config.ref_size, replace=False)
    elif len(ref_frames) < config.min_ref_size:
        more_frames = [x for x in numpy.arange(ref_margin, n_frames - ref_margin) if
                       ((x not in ref_frames) and (x not in bad_frames))]
        add_frames = numpy.random.choice(more_frames, min(config.ref_size - len(ref_frames), len(more_frames)),
                                         replace=False)
        ref_frames = numpy.concatenate((ref_frames, add_frames))

    print(f'After checking for size, {len(ref_frames)} ref frames')
    first_ref = numpy.average(data[ref_frames], axis=0).squeeze()
    norm_min = numpy.percentile(data[ref_frames], 1)
    norm_range = numpy.percentile(data[ref_frames], 99) - norm_min

    def norm_img(img, imin, irange):
        return numpy.clip((img - imin) / irange, 0, 1)

    d = norm_img(data, norm_min, norm_range)
    r = norm_img(first_ref, norm_min, norm_range)
    reg_offset = [0, 0]
    if use_reference == 'S2P':
        # use previous registration result as reference, so segmentations can be reused
        lp = os.path.join(proc_path, prefix, 'suite2p/plane0/ops.npy')
        ops = numpy.load(lp, allow_pickle=True).item()
        s2pr = ops['meanImg']
        norm_s2pr = norm_img(s2pr, s2pr.min(), s2pr.max() - s2pr.min())
        ref_offset = phase_cross_correlation(norm_s2pr, r, upsample_factor=10)[0]
        lprint(None, f'Offset from S2P reference: {ref_offset}')
        r = norm_s2pr # this updates the reference for upcoming steps

    #refine reference
    snr = numpy.nan_to_num(numpy.average(d, axis=(1, 2)) / numpy.std(d, axis=(1, 2)))
    snr_lim = min(1, numpy.nanpercentile(snr[ref_frames], 25))
    ref_frames = [x for x in ref_frames if snr[x] > snr_lim]
    print(f'After exclude low snr, {len(ref_frames)} ref frames')
    max_offset = max([abs(x) for x in reg_offset]) # to add to limit for keeping frames as stable reference
    avgs = numpy.zeros([len(ref_frames), *first_ref.shape])
    n_roll = round(config.rolling_avg_size / 1000 * fps)
    disps = numpy.zeros((len(ref_frames), 2))
    stable_refs = []
    for i, f in enumerate(ref_frames):
        i0 = max(0, f - n_roll)
        i1 = min(f + n_roll, n_frames)
        weighted_sample = d[i0:i1] * snr[i0:i1, None, None]
        avg_sample = numpy.sum(weighted_sample, axis=0) / numpy.sum(snr[i0:i1])
        displacements = phase_cross_correlation(avg_sample, r, upsample_factor=10)  # fit ref to target,
        disps[i] = displacements[0]
        if max([abs(x) for x in displacements[0]]) < (config.ref_displacement_limit + max_offset):
            stable_refs.append(i)
    sr_size = int(config.min_ref_size / 2)
    print(f'{len(stable_refs)} stable ref frames, min is {sr_size}')
    if len(stable_refs) < sr_size:
        add_sr = [x for x in range(len(ref_frames)) if x not in stable_refs]
        print(len(add_sr))
        stable_refs.extend(numpy.random.choice(add_sr, sr_size-len(stable_refs), replace=False))
        print(f'After adding, {len(stable_refs)} "stable" ref frames')
    for i in stable_refs:
        f = ref_frames[i]
        tforms = transform.SimilarityTransform(translation=disps[i])
        avgs[i] = transform.warp(data[f], tforms, order=1, preserve_range=True).astype('float')

    #update reference one last time for using with the final registration
    refined_reference = numpy.average(avgs[stable_refs], axis=0).squeeze()
    avgmax = numpy.nanpercentile(avgs[stable_refs], 95, axis=0).squeeze() #this will be saved as a preview
    r = norm_img(refined_reference, norm_min, norm_range)

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
            displacements = phase_cross_correlation(avg_sample, r,
                                                    upsample_factor=config.subpixel_precision)  # fit ref to target,
            disps[i, :2] = displacements[0]
        numpy.save(disps_fn, disps)

    else:
        disps = numpy.load(disps_fn)

    # Low pass filter displacement
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
    with open(os.path.join(oPath, prefix + config.setting_suffix), "w") as f:
        json.dump(asdict(config), f, indent=2)

    reg_movie = open_memmap(os.path.join(oPath, prefix + config.registered_suffix), mode='w+',
                            shape=data.shape, dtype=numpy.uint16, )
    for i, frame in enumerate(data):
        tforms = transform.SimilarityTransform(translation=f_disp[i])
        reg_movie[i] = transform.warp(frame, tforms, order=1, mode='symmetric', preserve_range=True)
    reg_movie.flush()


    if verbose:
        lprint(None, 'Saving previews')

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ca = ax[0, 0]
    strip_ax(ca)
    ca.imshow(first_ref)
    ca.set_title('First reference')

    ca = ax[0, 1]
    strip_ax(ca)
    ca.imshow(refined_reference)
    ca.set_title('Refined reference')

    ca = ax[1, 0]
    strip_ax(ca)
    ca.imshow(reg_movie[ref_frames].mean(axis=0).squeeze())
    ca.set_title('Output (ref frames)')

    #add a sample to have a general preview of registration quality

    spaced_frames = numpy.linspace(config.ref_margin, n_frames - config.ref_margin, config.ref_size).astype('int64')
    # exp_indices = sort_disp[spaced_frames]
    ca = ax[1, 1]
    strip_ax(ca)
    ca.imshow(reg_movie[spaced_frames].mean(axis=0).squeeze())
    ca.set_title('Output (sampled)')

    fig.savefig(os.path.join(oPath, prefix + '_registration_preview.png'), dpi=300)

    # save preview tif RGB
    preview_channels = {'R': None, 'G': None, 'B': None}
    sum_disp = numpy.abs(f_disp[:, 0] - f_disp[:, 0].mean())
    sum_disp += numpy.abs(f_disp[:, 1] - f_disp[:, 1].mean())
    sort_disp = numpy.argsort(sum_disp)
    pw_indices = sort_disp[int((n_frames - config.ref_size) * 0.5):int((n_frames + config.ref_size) * 0.5)]
    mean1 = reg_movie[pw_indices].mean(axis=0).squeeze()
    for pw_img, suffix in zip((mean1, avgmax), ('_GEVIRegPreview.tif', '_GEVIRegAvgMax.tif')):
        previewname = os.path.join(session.path, prefix + suffix)
        preview = numpy.zeros([*pw_img.shape, 3], dtype='uint8')
        preview_channels['G'] = pw_img / pw_img.max() * 255
        for chi, chn in enumerate('RGB'):
            x = preview_channels[chn]
            if x is not None:
                preview[..., chi] = x
        imwrite(previewname, preview)
    plt.close()