from Proc2P.Analysis.BatchGUI import *
from multiprocessing import freeze_support, set_start_method

'''
Launches the BatchGUI app for 2P data processing and review
'''

if __name__ == '__main__':
    freeze_support()
    try:
        set_start_method('spawn')
    except:
        pass
    BatchGUIQt.launch_GUI(Q_manager=BatchGUI_Q())
