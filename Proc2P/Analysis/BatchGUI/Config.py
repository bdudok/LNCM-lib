import os
from enum import Enum
from pathlib import Path

class GuiConfig:
    settings_filename = os.path.join(Path.home(), 'BatchGUI-settings.json') #for gui setting permanence
    MainWindowGeometry = (30, 60, 1200, 800)
    TextWidgetHeight = 60
    ButtonLabelWidth = 80
    ConsoleWidgetHeight = 300
    ConsoleWidgetWidth = 900
    PrefixFieldWidth = 400
    TagFieldWidth = 100
    plot_canvas_size = (9, 3) #inches,
    ROI_preview_default = 'avgmax'
    zoomfactor = 4.0 #for ROI editor
    SessionControlsHeight = 250
    SessionPreviewWidth = 250
    RunColor = '#4604D2'
    GreenColor = '#04D246'
    RedColor = '#D24604'
    SzColor = '#FC9564'


class State(Enum):
    SETUP = 0
    EDITING_ROI = 1
    VIEWING_TRACE = 2
    LIVE = 3