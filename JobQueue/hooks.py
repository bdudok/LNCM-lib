'''
Functions the manager runs after a job finished without error (JobTypeSpec.on_done), in the manager's account.
They get the job data as queued (JSON values; dataclasses and Enums are not rebuilt) and return a message that is
added to the job's result. Keep imports light: they are loaded in the manager.
'''


def motion_corrected(data):
    '''After MotionCorrect: set the session's Baserow Task from MotionCorr to Segment.'''
    from BaserowAPI.BaserowRequests import SetTask
    settings, entry, pre_only = data[:3]
    if pre_only:
        return None
    return SetTask().advance(entry['id'], 'MotionCorr', 'Segment')
