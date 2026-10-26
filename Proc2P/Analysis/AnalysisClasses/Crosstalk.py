import datetime
import json

import numpy
from scipy.signal import butter, sosfiltfilt

from Proc2P.Analysis.CaTrace import CaTrace
from Proc2P.Bruker.PreProc import SessionInfo


class Crosstalk:
    '''
    Removes the crosstalk (bleed-through) of one channel's fluorescence into another channel, for one ROI set.

    Model: F_target = F_target_true + k * F_source, with the same k in every ROI (an optical leak).
    k is fitted on the fast fluctuations of the raw fluorescence (high-pass above hp_cutoff), where the target sensor
    is assumed to have little signal of its own (e.g. a slow sensor in the target channel and a fast one in the source).
    For each ROI, the high-passed target trace is regressed on the high-passed source trace using the mean high-passed
    source trace of the other ROIs as the reference (instrumental variable), so that the shot noise of the ROI's own
    source trace does not bias k toward 0. k is pooled over ROIs:
        k = sum_i <target_i, ref_i> / sum_i <source_i, ref_i>
    The corrected target trace is F_target - k * F_source, with the CaTrace baseline corrected the same way:
        rel = (rel_target * bsl_target - k * rel_source * bsl_source) / (bsl_target - k * bsl_source)

    Outputs, in the target channel's trace folder: xtalk.json (parameters) and rel_xtalk.npy (corrected rel, cells x frames).
    '''
    __name__ = 'Crosstalk'
    key = 'rel_xtalk'
    param_fn = 'xtalk.json'
    version = '1'

    def __init__(self, procpath, prefix, tag, source_ch, target_ch, hp_cutoff=1.0):
        '''
        :param tag: ROI tag of the processed traces
        :param source_ch: channel that leaks (0 or 1)
        :param target_ch: channel that receives the leak and gets corrected (0 or 1)
        :param hp_cutoff: Hz, k is fitted on the fluctuations faster than this
        '''
        if source_ch == target_ch:
            raise ValueError('Source and target channel must be different')
        self.prefix = prefix
        self.tag = tag
        self.source_ch = source_ch
        self.target_ch = target_ch
        self.hp_cutoff = hp_cutoff
        self.source = CaTrace(procpath, prefix, tag=tag, ch=source_ch)
        self.target = CaTrace(procpath, prefix, tag=tag, ch=target_ch)
        for ca in (self.source, self.target):
            ca.load()
            if ca.trace.ndim != 2:
                raise ValueError(f'{ca.pf} holds more than one channel; Crosstalk needs single-channel trace folders')
        self.fps = SessionInfo(self.target.opPath, prefix).load(must_exist=True)['framerate']
        self.k = None

    def fit(self):
        '''Estimates k. Also stores the k of each ROI.'''
        self.k, self.roi_k, self.n_frames_used = self.estimate(self.target.trace, self.source.trace)
        return self.k

    def estimate(self, f_target, f_source):
        '''pooled k of f_target on f_source (cells x frames), and the k of each ROI (NaN for ROIs not used)'''
        f_target = numpy.asarray(f_target, dtype=float)
        f_source = numpy.asarray(f_source, dtype=float)
        rois = numpy.where(numpy.isfinite(f_target).any(axis=1) & numpy.isfinite(f_source).any(axis=1))[0]
        frames = numpy.isfinite(f_target[rois]).all(axis=0) & numpy.isfinite(f_source[rois]).all(axis=0)
        sos = butter(3, self.hp_cutoff, 'highpass', fs=self.fps, output='sos')
        h_target = sosfiltfilt(sos, f_target[rois][:, frames], axis=1)
        h_source = sosfiltfilt(sos, f_source[rois][:, frames], axis=1)
        # ROIs with a flat trace (e.g. outside the field) are not used
        keep = (h_target.std(axis=1) > 0) & (h_source.std(axis=1) > 0)
        rois, h_target, h_source = rois[keep], h_target[keep], h_source[keep]
        if len(rois) < 2:
            raise ValueError(f'{self.prefix}: at least 2 ROIs with valid traces are needed, found {len(rois)}')
        roi_k = numpy.full(len(f_target), numpy.nan)
        num, den = 0, 0
        for i, c in enumerate(rois):
            ref = numpy.delete(h_source, i, axis=0).mean(axis=0)
            roi_k[c] = (h_target[i] @ ref) / (h_source[i] @ ref)
            num += h_target[i] @ ref
            den += h_source[i] @ ref
        return num / den, roi_k, int(frames.sum())

    def correct(self):
        '''corrected rel of the target channel (cells x frames)'''
        if self.k is None:
            self.fit()
        k = self.k
        return ((self.target.rel * self.target.bsl - k * self.source.rel * self.source.bsl)
                / (self.target.bsl - k * self.source.bsl))

    def run(self):
        '''fit, correct, and save. Returns the corrected rel'''
        self.fit()
        self.rel = self.correct()
        self.save()
        return self.rel

    def save(self):
        self.target.save_custom(self.key, self.rel)
        params = {
            'version': self.version,
            'date': datetime.datetime.now().isoformat(timespec='seconds'),
            'prefix': self.prefix,
            'tag': self.tag,
            'source_ch': self.source_ch,
            'target_ch': self.target_ch,
            'hp_cutoff_Hz': self.hp_cutoff,
            'fps': self.fps,
            'k': float(self.k),
            'roi_k': [None if numpy.isnan(x) else float(x) for x in self.roi_k],
            'n_rois_used': int(numpy.isfinite(self.roi_k).sum()),
            'n_frames_used': self.n_frames_used,
        }
        with open(self.target.pf / self.param_fn, 'w') as f:
            json.dump(params, f, indent=1)

    def load(self):
        '''Reads the saved parameters and corrected rel. Returns the corrected rel'''
        with open(self.target.pf / self.param_fn, 'r') as f:
            self.params = json.load(f)
        if (self.params['source_ch'], self.params['hp_cutoff_Hz']) != (self.source_ch, self.hp_cutoff):
            raise ValueError(f'Saved crosstalk parameters differ: {self.params}')
        self.k = self.params['k']
        self.rel = self.target.get_custom(self.key)
        return self.rel
