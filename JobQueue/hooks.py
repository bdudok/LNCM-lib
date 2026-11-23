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


def _advance_by_prefix(prefix, from_task, to_task):
    '''Set Task of the session named prefix (Image.ID without the -000 tag) if it is from_task (str or tuple).'''
    from BaserowAPI.BaserowRequests import GetSessions, SetTask
    setter = SetTask()  # raises first if the manager has no token
    rows = GetSessions(timeout=30).get_session(prefix)
    if len(rows):
        rows = rows[rows['Image.ID'].str[:-4] == prefix]
    if len(rows) != 1:
        return f'Baserow Task not changed: {len(rows)} sessions are named {prefix}'
    return setter.advance(rows.iloc[0]['id'], from_task, to_task)


def signals_pulled(data):
    '''After PullSignals (BatchGUI): set the session's Baserow Task from Segment or Pull to Process.'''
    return _advance_by_prefix(data[1], ('Segment', 'Pull'), 'Process')


def rois_processed(data):
    '''After ProcessROIs (BatchGUI): set the session's Baserow Task from Process to Done.'''
    return _advance_by_prefix(data[1], 'Process', 'Done')
