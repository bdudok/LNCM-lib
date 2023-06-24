import os
import numpy

''''Don't import any Core files here (dependency loop). Use CommonFunc for that.'''

def split_ephys(path, n_channels):
    suffix = '.ephys'
    os.chdir(path)
    raw_shape = n_channels + 1
    for f in os.listdir(path):
        if not f.endswith(suffix):
            continue
        prefix = f[:f.find(suffix)]
        if not os.path.exists(prefix + suffix):
            continue
        if os.path.exists(f'{prefix}-ch1{suffix}'):
            continue
        print(f'Splitting {prefix} ...')
        ep_raw = numpy.fromfile(prefix + suffix, dtype='float32')
        n_samples = int(len(ep_raw) / raw_shape)
        ep_formatted = numpy.reshape(ep_raw, (n_samples, raw_shape))
        for ch in range(n_channels):
            new = numpy.empty((n_samples, 2), dtype=ep_raw.dtype)
            new[:, 0] = ep_formatted[:, 0]
            new[:, 1] = ep_formatted[:, ch + 1]
            new.tofile(f'{prefix}-ch{ch+1}{suffix}')
    print('All done.')


def outlier_indices(values, thresh=3.5):
    not_nan = numpy.logical_not(numpy.isnan(values))
    median = numpy.median(values[not_nan])
    diff = (values - median) ** 2
    diff = numpy.nan_to_num(numpy.sqrt(diff))
    med_abs_deviation = numpy.median(diff[not_nan])
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return numpy.where(modified_z_score > thresh)[0]

def export_speed(path, n_channels):
    '''export speed resampled to align with ephys samples'''
    from Quad import SplitQuad
    os.chdir(path)
    suffix = '_quadrature.mat'
    prefixes = [f[:f.find(suffix)] for f in os.listdir(path) if f.endswith(suffix)]
    # create speed and running files
    for prefix in prefixes:
        speed_fn = prefix + '.speed.npy'
        run_fn = prefix + '.running.npy'
        ep_fn = prefix + '.ephys'
        if os.path.exists(ep_fn) and not all((os.path.exists(speed_fn), os.path.exists(run_fn))):
            print('Exporting', prefix)
            ep_raw = numpy.fromfile(ep_fn, dtype='float32')
            raw_shape = n_channels + 1
            n_samples = int(len(ep_raw) / raw_shape)
            ep_formatted = numpy.reshape(ep_raw, (n_samples, raw_shape))
            ephys_trace = ep_formatted[:, 1:]
            ephys_frames = ep_formatted[:, 0]

            # load quadrature file
            pos = SplitQuad(prefix)
            # resample for ephys
            speed = numpy.zeros(n_samples)
            for t, s in enumerate(pos.speed):
                if s != 0:
                    speed[numpy.where(ephys_frames == t)] = s
            numpy.save(speed_fn, speed)
            running = numpy.zeros(n_samples, dtype='bool')
            for t, s in enumerate(pos.gapless):
                if s:
                    running[numpy.where(ephys_frames == t)] = True
            numpy.save(run_fn, running)

def export_speed_cms(path):
    '''export speed resampled to align with ephys samples'''
    from Quad import SplitQuad
    from ImportFirst import speed_calib, fps
    import pandas
    os.chdir(path)
    suffix = '_quadrature.mat'
    prefixes = [f[:f.find(suffix)] for f in os.listdir(path) if f.endswith(suffix)]
    # create speed and running files
    for prefix in prefixes:
        speed_fn = prefix + '.speed_cms.csv'
        if not os.path.exists(speed_fn):
            # load quadrature file
            pos = SplitQuad(prefix)
            speed = pos.speed / speed_calib * fps
            pandas.DataFrame({'Speed [cm/s]': speed}).to_csv(speed_fn)


def mad_based_outlier(points, thresh=3.5):
    median = numpy.median(points)
    diff = (points - median) ** 2
    diff = numpy.sqrt(diff)
    med_abs_deviation = numpy.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return numpy.where(modified_z_score < thresh)[0]


def strip_ax(ca):
    ca.spines['right'].set_visible(False)
    ca.spines['top'].set_visible(False)
    ca.spines['bottom'].set_visible(False)
    ca.spines['left'].set_visible(False)
    ca.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='off')
    ca.tick_params(axis='y', which='both', right='off', left='off', labelright='off')
    ca.xaxis.set_visible(False)
    ca.yaxis.set_visible(False)


def gapless(trace, gap=5):
    gapless = numpy.copy(trace)
    ready = False
    while not ready:
        ready = True
        for t, m in enumerate(gapless):
            if not m:
                if numpy.any(gapless[t - gap:t]) and numpy.any(gapless[t:t + gap]):
                    gapless[t] = 1
                    ready = False
    return gapless
