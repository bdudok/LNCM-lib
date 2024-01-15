from LFP.SzDet.vEEGView.GUI_vEEG import launch_GUI

#specify the path to start at
path = 'D:\Shares\Data\_Processed\EEG/test/'

# override the GUI's default settings (for convenience)
# you can comment out lines to use the hard coded defaults
settings = {
}

launch_GUI(path=path, savepath=path, setupID='Pinnacle', defaults=settings)
