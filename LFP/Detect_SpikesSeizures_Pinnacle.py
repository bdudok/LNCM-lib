from LFP.SzDet.AppSpikeSz.GUI_SpikeSzDet import launch_GUI

#specify the path to start at
path = 'D:\Shares\Data\_Processed\EEG\Tottering/'

# override the GUI's default settings (for convenience)
# you can comment out lines to use the hard coded defaults
settings = {
    'LoCut': 12, #band pass filter low cut (Hz)
    'HiCut': 36, #band pass filter high cut (Hz)
    'Tr1': 3, #spike treshold for spike width measurement (SD)
    'Tr2': 4, #spike amplitude treshold for spike detection (SD)
    'TrDiff': 7, #threshold for including broad spikes (based on abs diff, SD)
    'Dur': 10, #spike minimum duration (ms)
    'Dist': 50, #spike separation (ms)
    'Sz.MinDur': 2, #spike cluster duration to be considered seizure (s)
    'Sz.Gap': 2, #gap for merging neighboring spike clusters (s)
    'SzDet.Framesize': 50, #resolution of output instantaneous spike rate trace (ms)
    # 'fs': 2000, #sampling rate (read from input in case of Pinnacle EDF file)
    'Channel': 1 #channel number (indexed from 1)
}

launch_GUI(path=path, savepath=path, setupID='Pinnacle', defaults=settings)
