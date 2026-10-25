# LNCM-lib
Resources for data analysis in LNCM / Dudok lab

# Installation
See [installation.md](installation.md).
`envs/site_config.json` holds lab-specific settings (backup paths, scratch disk, SIMA env, uv env location). Edit it for your lab.

# The following resources are included:

## Ephys
### LFP.SpikeDet
detect epileptic spikes on LFP
### LFP.SzDet
detect seizures and compute seizure burden time series using spike times as input
### LFP.Pinnacle
read EDF files exported by the Pinnacle recorder

## 2-photon
### Proc2P.Treadmill
load treadmill data from PyControl file
### Proc2P.Bruker
classes for reading BCM Dudok lab raw data recorded on the Bruker setup
### Proc2P.Analysis
processing pipeline and GUI; ImagingSession and related classes

## Utilities
### BaserowAPI
find imaging sessions in self hosted database

# The following GUI tools are included: 

## Ephys
### LFP.SzDet.AppSpikeSz
SpikeSzDet - tune and save spike detection settings for a set of sessions, run seizure detection
SzViewGUI - review EEG with synced video, manually curate and edit automatically detected seizures

## 2-photon
### Proc2P.Analysis.BatchGUI
GUI for processing and inspecting ImagingSession 

# Developing LNCM-lib
Changing packages, code conventions and linting: see [envs/dev_readme.md](envs/dev_readme.md).

# See also from the lab:

## Recorder
Apps for controlling cameras and acquisition
https://github.com/bdudok/Recorder.git

## Treadmill control
Remote acquisition control, task configuration and treadmill build info
https://github.com/bdudok/PyControl_LNCM.git
