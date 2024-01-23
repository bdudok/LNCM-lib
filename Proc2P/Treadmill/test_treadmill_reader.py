import os

import numpy
import numpy as np
import pandas as pd
from _Dependencies.PyControl.code.tools import data_import as data_import_new
from _Dependencies.PyControlWorking.code.tools import data_import
from ConfigVars import TR

tm_path = 'D:\Shares\Data\_RawData\Bruker\PVTot\PVTot12_2024-01-18_expression_128-000/'
tm_fn = 'PVTot12_2024-01-18_expression_128-2024-01-18-103951.txt'

tm = data_import.Session(tm_path+tm_fn)
tm_new = data_import_new.Session(tm_path+tm_fn)