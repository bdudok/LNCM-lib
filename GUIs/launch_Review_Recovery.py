from Video.RecoveryReviewGUI import launch_GUI

'''
Review post-surgery recovery videos (MonitorCam) and mark seizures on the Racine scale.
'''

raw_dir = 'D:/Shares/Data/_RawData/MonitorCam'  # where Open video starts

if __name__ == '__main__':
    launch_GUI(raw_dir)
