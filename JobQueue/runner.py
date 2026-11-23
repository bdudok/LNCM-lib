'''
Per-user job runner. Clients start it automatically (JobQueue.client.ensure_runner), in the user's own account and
with the env and LNCM-lib checkout of the client, so job outputs are owned by the user who queued them.
- Reports to the manager every second and runs the jobs the manager assigns to it.
- Starts a worker process per job type when needed, reuses it for later jobs, and stops it after WORKER_IDLE_S.
- Exits after RUNNER_IDLE_S with no workers and no queued jobs of this user.
Log: <user profile>/.lncm_jobqueue/runner-<checkout>.log
Run: python -m JobQueue.runner (from the LNCM-lib folder)
'''
import os
import sys
import time
import uuid
import hashlib
import traceback
import multiprocessing as mp
from queue import Empty

import psutil

from JobQueue.common import (JOB_TYPES, RUNNER_SYNC_S, RUNNER_IDLE_S, WORKER_IDLE_S, ManagerUnavailable, call,
                             current_user, user_token, lib_root, user_state_dir, resolve, decode)


def log_path():
    tag = hashlib.sha1(str(lib_root()).lower().encode()).hexdigest()[:8]
    return user_state_dir() / f'runner-{tag}.log'


def worker_main(job_type: str, handler_path: str, in_q, out_q, log_file: str):
    '''Worker process: runs jobs of one type until it receives None.'''
    sys.stdout = sys.stderr = open(log_file, 'a', buffering=1)
    handler = resolve(handler_path)
    for job_id, payload in iter(in_q.get, None):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), f'{job_type} job {job_id} started')
        try:
            result = handler(decode(payload))
            out_q.put((job_id, True, None if result is None else str(result), None))
        except Exception:
            traceback.print_exc()
            out_q.put((job_id, False, None, traceback.format_exc()))


class WorkerProc:

    def __init__(self, ctx, job_type: str, name: str):
        self.job_type = job_type
        self.name = name
        self.in_q = ctx.Queue()
        self.out_q = ctx.Queue()
        self.proc = ctx.Process(target=worker_main, name=name,
                                args=(job_type, JOB_TYPES[job_type].handler, self.in_q, self.out_q,
                                      str(log_path())))
        self.proc.start()
        self.job_id = None
        self.idle_since = time.time()

    def start_job(self, job_id, payload):
        self.job_id = job_id
        self.in_q.put((job_id, payload))

    def poll(self):
        '''Returns a finished report, or None.'''
        if self.job_id is None:
            return None
        try:
            job_id, ok, result, error = self.out_q.get_nowait()
        except Empty:
            if self.proc.is_alive():
                return None
            job_id, ok, result, error = self.job_id, False, None, (f'Worker process exited (code '
                                                                    f'{self.proc.exitcode}) while running the job.')
        self.job_id = None
        self.idle_since = time.time()
        return {'job_id': job_id, 'ok': ok, 'result': result, 'error': error, 'worker': self.name}

    def stop(self):
        '''Stop the worker and any processes it started (e.g. the per-cell workers of ProcessROIs).'''
        try:
            parent = psutil.Process(self.proc.pid)
            procs = parent.children(recursive=True) + [parent]
        except psutil.NoSuchProcess:
            procs = []
        for p in procs:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        self.proc.join(timeout=5)


class Runner:

    def __init__(self):
        self.ctx = mp.get_context('spawn')
        self.runner_id = uuid.uuid4().hex
        self.user = current_user()
        self.token = user_token()
        self.lib_root = str(lib_root())
        self.workers = []
        self.n_started = {}  # job_type: count, for worker names
        self.started = []  # job ids started since the last report
        self.finished = []  # reports not yet delivered to the manager
        self.last_busy = time.time()

    def log(self, *args):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), *args, flush=True)

    def get_worker(self, job_type):
        for w in self.workers:
            if w.job_type == job_type and w.job_id is None and w.proc.is_alive():
                return w
        n = self.n_started.get(job_type, 0)
        self.n_started[job_type] = n + 1
        w = WorkerProc(self.ctx, job_type, f'{job_type}-{n}')
        self.workers.append(w)
        self.log('started worker', w.name)
        return w

    def step(self):
        '''One round: collect results, stop idle workers, report, start assigned jobs. Returns False to exit.'''
        now = time.time()
        for w in self.workers:
            report = w.poll()
            if report is not None:
                self.finished.append(report)
                self.log(w.name, 'finished', report['job_id'], 'ok' if report['ok'] else 'failed')
        for w in [w for w in self.workers if w.job_id is None and
                  (now - w.idle_since > WORKER_IDLE_S or not w.proc.is_alive())]:
            self.log('stopping idle worker', w.name)
            w.stop()
            self.workers.remove(w)
        try:
            resp = call('/sync', {
                'runner_id': self.runner_id, 'user': self.user, 'token': self.token, 'lib_root': self.lib_root,
                'pid': os.getpid(), 'started': self.started, 'finished': self.finished,
                'workers': [{'name': w.name, 'job_type': w.job_type, 'pid': w.proc.pid, 'job_id': w.job_id}
                            for w in self.workers]})
        except ManagerUnavailable as e:
            self.log(e)  # keep running jobs, report when the manager is back
            return True
        self.started, self.finished = [], []
        if resp.get('exit'):
            self.log('exiting:', resp.get('reason'))
            return False
        for job in resp['assign']:
            if job['job_type'] not in JOB_TYPES:
                self.finished.append({'job_id': job['job_id'], 'ok': False, 'worker': None, 'result': None,
                                      'error': f'This checkout has no job type {job["job_type"]}.'})
                continue
            w = self.get_worker(job['job_type'])
            w.start_job(job['job_id'], job['payload'])
            self.started.append(job['job_id'])
            self.log(w.name, 'started', job['job_id'])
        if self.workers or resp['pending'] or resp['assign']:
            self.last_busy = now
        return now - self.last_busy < RUNNER_IDLE_S

    def run(self):
        self.log('runner started for', self.user, 'in', self.lib_root)
        try:
            while self.step():
                time.sleep(RUNNER_SYNC_S)
        finally:
            for w in self.workers:
                w.stop()
            self.log('runner stopped')


def main():
    sys.stdout = sys.stderr = open(log_path(), 'a', buffering=1)
    Runner().run()


if __name__ == '__main__':
    main()
