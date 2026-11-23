'''
BatchGUI's connection to the lab job queue (JobQueue). Jobs run in the user's runner process, not in BatchGUI.
The handlers of the job types are in JobQueue/handlers.py.
'''
import json
from enum import Enum
from dataclasses import dataclass

from JobQueue.client import QueueClient
from JobQueue.common import QueueError, encode


class JobType(Enum):
    # names are the job types in JobQueue.common.JOB_TYPES
    TEST = 0
    GEVIReg = 1
    PullVM = 2
    SIMA = 3
    PullSignals = 4
    ProcessROIs = 5
    Ripples = 6


@dataclass
class Job:
    type: JobType
    data: tuple  # for all BatchGUI jobs, data[1] is the prefix


@dataclass
class WorkerError:
    '''Passed to the result handler instead of a result when a job failed'''
    traceback: str


class BatchGUIQueue:
    '''Queues BatchGUI jobs and hands back results as (worker name, result message or WorkerError).'''

    def __init__(self):
        self.client = QueueClient()

    @property
    def manager_up(self) -> bool:
        return self.client.manager_up

    def run_job(self, job: Job):
        try:
            # only a job with the same data counts as a duplicate
            self.client.submit(job.type.name, job.data, label=job.data[1], key=json.dumps(encode(job.data)))
        except QueueError as e:
            print(f'{job.type.name} {job.data[1]} not queued. {e}')

    def poll_result(self, handler_function):
        try:
            results = self.client.results()
        except QueueError:
            return
        for r in results:
            name = r['worker'] or r['job_type']
            if r['status'] == 'done':
                handler_function((name, r['result']))
            else:
                handler_function((name, WorkerError(f'{r["label"]}: {r["error"]}')))

    def check_manager(self) -> bool:
        '''Checks that the manager is running and restarts this user's runner if it stopped.'''
        try:
            self.client.ensure_runner()
        except QueueError:
            pass
        return self.client.manager_up


def BatchGUI_Q():
    return BatchGUIQueue()
