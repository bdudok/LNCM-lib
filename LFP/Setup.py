from enum import StrEnum


class Setup(StrEnum):
    '''Source of the EEG/LFP recording. The values are the strings used before the enum, so they still convert.'''
    PINNACLE = 'Pinnacle'  # Pinnacle EDF files, one file per recording, in the EEG folder
    LNCM = 'LNCM'  # Bruker 2P sessions, LFP in <prefix>/<prefix>_ephys.npy
