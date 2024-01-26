import os

import numpy
import numpy as np
import pandas as pd
from _Dependencies.PCN.code.tools import data_import as data_import_new
# from _Dependencies.PyControlWorking.code.tools import data_import
from _Dependencies.PyControl.code.tools import data_import
from ConfigVars import TR

tm_path = 'D:\Shares\Data\_RawData\Bruker\PVTot\PVTot12_2024-01-18_expression_128-000/'
tm_fn = 'PVTot12_2024-01-18_expression_128-2024-01-18-103951.txt'

tm = data_import.Session(tm_path+tm_fn)
tm_new = data_import_new.Session(tm_path+tm_fn)

pl = tm.print_lines
p_df = tm_new.variables_df

old_times = []
new_times = []

for event in tm.print_lines:
    if 'lap_counter' in event:
        e_time = int(event.split(' ')[0])
        old_times.append(e_time)

for _, event in tm_new.variables_df.iterrows():
    if not numpy.isnan(event[('values', 'lap_counter')]):
        e_time = event['time'].iloc[0]
        new_times.append(e_time)

