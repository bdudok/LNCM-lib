from Proc2P.Bruker.MotionCorrectGUI import launch_GUI

'''
Select Bruker sessions from Baserow and queue their pre-processing and motion correction on the job queue.
When a session is motion corrected, the job queue manager sets its Baserow Task from MotionCorr to Segment.
'''

if __name__ == '__main__':
    launch_GUI()
