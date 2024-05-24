# LNCM-lib
Resources for data analysis in LNCM / Dudok lab

# The following libraries are included:

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
### Proc2P.Legacy
all scanbox reader classes made during Soltesz lab times
### Proc2p.Bruker
classes for reading BCM Dudok lab raw data recorded on the Bruker setup
### Proc2p.Analysis
processing pipeline and GUI; ImagingSession and related classes

## Utilities
### BaserowAPI
find imaging sessions in self hosted database

# The following GUI tools are included: 

## Ephys
### LFP.SzDet.AppSpikeSz
tune and save spike detection settings for a set of sessions

## 2-photon
### Proc2p.Analysis.BatchGUI
GUI for processing and inspecting ImagingSession 

# See also from the lab:

## Recorder
Scripts for controlling cameras and acquisition
https://github.com/bdudok/Recorder.git

## Treadmill control
Remote acquisition control, task configuration and treadmill build info
https://github.com/bdudok/PyControl_LNCM.git