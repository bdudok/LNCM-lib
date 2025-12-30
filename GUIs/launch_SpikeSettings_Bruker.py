from LFP.SzDet.AppSpikeSz.GUI_SpikeSzDet import launch_GUI

#specify the path to start at
path = 'D:\Shares\Data\_Processed/2P/'
use_deafults = ('hippocampus', 'cortex')[0] #select the relevant brain region

# override the GUI's default settings (for convenience)
# you can comment out lines to use the hard coded defaults
defs = {}

defs['cortex'] = { #aiming to detect SWDs
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
    'fs': 2000, #sampling rate (read from input in case of Pinnacle EDF file)
    'Channel': 2, #channel number (indexed from 1)
    'PlotDur': 'all', #displayed trace length in minutes, or 'all'
}
defs['hippocampus'] = { #aiming to detect ripples and HFOs
    'LoCut': 80, #band pass filter low cut (Hz)
    'HiCut': 500, #band pass filter high cut (Hz)
    'Tr1': 3, #spike treshold for spike width measurement (SD)
    'Tr2': 5, #spike amplitude treshold for spike detection (SD)
    'TrDiff': 5, #threshold for including broad spikes (based on abs diff, SD)
    'Dur': 5, #spike minimum duration (ms)
    'Dist': 50, #spike separation (ms)
    'Sz.MinDur': 3, #spike cluster duration to be considered seizure (s)
    'Sz.Gap': 3, #gap for merging neighboring spike clusters (s)
    'SzDet.Framesize': 50, #resolution of output instantaneous spike rate trace (ms)
    'fs': 5000, #sampling rate (read from input in case of Pinnacle EDF file)
    'Channel': 1, #channel number (indexed from 1)
    'PlotDur': 'all', #displayed trace length in minutes, or 'all'
}

launch_GUI(path=path, savepath=path, setupID='LNCM', defaults=defs[use_deafults])

