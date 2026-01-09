import os
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
import Proc2P.Analysis.BatchGUI.BatchGUIQt
from Proc2P.Analysis.BatchGUI.QueueManager import Job, JobType, BatchGUI_Q