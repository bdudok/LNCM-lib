'''
Queue jobs on the lab job queue.
Scripts that only queue jobs:
    from JobQueue.client import submit
    job_id = submit('GEVIReg', (processed_path, prefix, RegConfig()), label=prefix)
GUIs that want the results back keep a QueueClient and call results() periodically.
Jobs run in the user's own account, in a runner process that this module starts when needed.
'''
import os
import sys
import time
import uuid
import subprocess
from urllib.parse import urlencode

from JobQueue.common import (JOB_TYPES, ManagerUnavailable, ManagerRefused, call, encode, current_user, user_token,
                             lib_root, user_state_dir)

RUNNER_START_WAIT_S = 20.0  # don't start another runner within this time after starting one


class QueueClient:

    def __init__(self, receive_results=True):
        '''
        :param receive_results: False for scripts that queue jobs and exit; their results are on the dashboard and
         can be checked with job_status
        '''
        self.client_id = uuid.uuid4().hex if receive_results else None
        self.user = current_user()
        self.token = user_token()
        self.lib_root = str(lib_root())
        self._runner_started = 0.0
        self.manager_up = False

    def submit(self, job_type: str, data, label: str = '', key: str | None = None) -> str:
        '''
        Queue a job. Raises ManagerUnavailable if the manager is not running. Returns the job id.
        :param label: shown on the dashboard and returned with the result
        :param key: identifies what the job computes (e.g. the output file). If a job of the same type and key is
         already waiting or running, the job is not queued again and the id of that job is returned. Default: label
        '''
        if job_type not in JOB_TYPES:
            raise ValueError(f'Unknown job type {job_type}. Known: {list(JOB_TYPES)}')
        label = str(label)
        resp = self._call('/submit', {'user': self.user, 'token': self.token, 'client_id': self.client_id,
                                      'job_type': job_type, 'label': label, 'key': key or label or None,
                                      'lib_root': self.lib_root, 'payload': encode(data)})
        if resp['duplicate']:
            print(f'{job_type} {label} is already queued')
        self.ensure_runner(force=True)
        return resp['job_id']

    def results(self) -> list[dict]:
        '''Jobs of this client that finished since the last call. Each has status 'done' with result, or 'failed'
        with error (the traceback).'''
        if self.client_id is None:
            return []
        return self._call('/results', {'user': self.user, 'token': self.token, 'client_id': self.client_id})['results']

    def job_status(self, job_ids: list[str]) -> list[dict]:
        return self._call('/jobs', {'user': self.user, 'token': self.token, 'job_ids': list(job_ids)})['jobs']

    def is_up(self) -> bool:
        try:
            self._call('/health', timeout=0.5)
        except ManagerUnavailable:
            pass
        return self.manager_up

    def ensure_runner(self, force=False):
        '''Start this user's runner for this checkout if the manager doesn't see one, and the user has jobs queued
        from this checkout (or force).'''
        if time.time() - self._runner_started < RUNNER_START_WAIT_S:
            return
        resp = self._call('/health?' + urlencode({'user': self.user, 'lib_root': self.lib_root}))
        if not resp['runner_alive'] and (force or resp['pending']):
            start_runner()
            self._runner_started = time.time()

    def _call(self, path, data=None, timeout=2.0):
        try:
            resp = call(path, data, timeout=timeout)
        except ManagerRefused:
            self.manager_up = True
            raise
        except ManagerUnavailable:
            self.manager_up = False
            raise
        self.manager_up = True
        return resp


def start_runner():
    '''Start a runner in the background, detached so it keeps running after this process exits.'''
    kwargs = {'cwd': str(lib_root()), 'stdin': subprocess.DEVNULL, 'stdout': subprocess.DEVNULL,
              'stderr': open(user_state_dir() / 'runner-start.log', 'a'), 'close_fds': True}
    cmd = [sys.executable, '-m', 'JobQueue.runner']
    if os.name == 'nt':
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        try:
            # leave the job object of the parent (e.g. PyCharm's), which would kill the runner when the parent exits
            subprocess.Popen(cmd, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
        except OSError:  # the parent's job object does not allow breaking away
            subprocess.Popen(cmd, creationflags=flags, **kwargs)
    else:
        subprocess.Popen(cmd, start_new_session=True, **kwargs)


_default_client = None


def submit(job_type: str, data, label: str = '', key: str | None = None) -> str:
    '''Queue a job from code that doesn't wait for the result (see QueueClient.submit). Returns the job id.'''
    global _default_client
    if _default_client is None:
        _default_client = QueueClient(receive_results=False)
    return _default_client.submit(job_type, data, label, key)
