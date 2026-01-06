from LFP.SzDet.AppSpikeSz.GUI_SpikeSzDet import launch_GUI
#specify the path to start at
path = 'D:\Shares\Data\_Processed\EEG\Kainate/'

# override the GUI's default settings (for convenience)
# you can comment out lines to use the hard coded defaults
settings = {
    'LoCut': 120, #band pass filter low cut (Hz)
    'HiCut': 500, #band pass filter high cut (Hz)
    'Tr1': 2, #spike treshold for spike width measurement (SD)
    'Tr2': 3, #spike amplitude treshold for spike detection (SD)
    'TrDiff': 7, #threshold for including broad spikes (based on abs diff, SD)
    'Dur': 3, #spike minimum duration (ms)
    'Dist': 50, #spike separation (ms)
    'Sz.MinDur': 5, #spike cluster duration to be considered seizure (s)
    'Sz.Gap': 5, #gap for merging neighboring spike clusters (s)
    'SzDet.Framesize': 50, #resolution of output instantaneous spike rate trace (ms)
    # 'fs': 2000, #sampling rate (read from input in case of Pinnacle EDF file)
    'Channel': 3, #channel number (indexed from 1)
    'rejection_value': 3000, #absolute voltage threshold for outlier samples. default:3000
    'rejection_step': 1, #clustering distance for outlier samples (s). dafault:1
    'rejection_tail': 3, #time to exclude after artefact (s). default:3
    'rejection_factor': 0.5, # minimum outlier cluster size for rejection. if < 1: fraction of samples
}

if __name__ == '__main__':
    launch_GUI(path=path, savepath=path, setupID='Pinnacle', defaults=settings)
