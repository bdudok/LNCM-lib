import os
import datetime
import json
from PlotTools.MyColorLib import get_color
from PlotTools import *
from Proc2P.Bruker.ConfigVars import CF
from Proc2P.utils import *
from Proc2P.Analysis.ImagingSession import ImagingSession
from envs.site_config import CONFIG
from BaserowAPI.BaserowRequests import GetSessions

from datetime import datetime
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from dateutil.tz import tzlocal
import matplotlib.path as mplpath

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.image import ImageSeries
from pynwb.ophys import (
    CorrectedImageStack,
    Fluorescence,
    ImageSegmentation,
    MotionCorrection,
    OnePhotonSeries,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)

from ExportImagingSession import ExportImagingSession
from pathlib import Path
from NWBExporter import NWBExporter



# open a session
prefix = 'Dag47_2026-04-22_goal_128'
db = GetSessions()
db_entry = db.get_session(prefix).iloc[0]
procpath = db_entry['Processed.Path']
#
# ROI_tag = 'CellsRe'
# brainregion ='CA1'
# ROI_indicator = "jRGECO1a"
# ROI_description = "All GECO-expressing PCs segmented"
# session = ImagingSession(procpath, prefix, tag=ROI_tag)
#
# config = ''
this_package = Path(__file__).resolve().parent #the template config is in the repo;
# # but each dataset's config can be saved in the dataset folder
config_file = os.path.join(this_package, 'export_config_template_2ch-lean.yaml')

exporter = NWBExporter(prefix, config_file)
exporter.run()

