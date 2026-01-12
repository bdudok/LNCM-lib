from Proc2P.Analysis.BatchGUI.Config import *
from dataclasses import dataclass
from collections.abc import Callable
from multiprocessing import Queue, Process, freeze_support, set_start_method
from queue import Empty
#import worker functions
from Proc2P.Analysis.GEVIReg.Register import Worker as GEVIReg_Worker
from Proc2P.Analysis.AnalysisClasses.NormalizeVm import Worker as PullVM_Worker

class JobType(Enum):
    TEST = 0
    GEVIReg = 1
    PullVM = 2

@dataclass
class Job:
    type: JobType
    data: tuple

@dataclass
class Worker:
    task: Callable
    max_n: int

class testworker(Process):
    __name__ = 'Test-Worker'

    def __init__(self, queue, res_queue, n=0):
        super(testworker, self).__init__()
        self.queue = queue
        self.res_queue = res_queue
        self.n = n

    def run(self):
        for data in iter(self.queue.get, None):
            print(data)
            self.res_queue.put((self.__name__ + str(self.n), data))

class QManager:
    '''
    This class routes batch jobs to the correct worker type
    Usage:
        Set up freeze_support and spawn method, then init this from main
        call register_worker once with each job type to set the callable and the max worker count
        then pass Job objects to run_job
    '''

    def __init__(self):
       self.Q_job = {}
       self.Q_result = Queue()
       self.n_worker = {}
       self.worker_config = {}

    def register_worker(self, jobtype:JobType, task:Callable, max_n:int):
        '''
        Define a worker fucntion
        :param jobtype: name of the job type
        :param task: Worker function, init needs to take a job Q and a result Q, needs to have a run method
        :param max_n: max number of workers of this type to spawn
        :return:
        '''
        self.worker_config[jobtype] = Worker(task=task, max_n=max_n)

    def _lazy_register_worker(self, jobtype:JobType,):
        self.Q_job[jobtype] = Queue()
        self.n_worker[jobtype] = self.worker_config[jobtype].max_n

    def run_job(self, job:Job):
        '''
        Checks if workers are initialized and queues the job
        '''
        if job.type not in self.worker_config:
            raise ValueError(f'Use register_worker to define job type {job.type}')
        if job.type not in self.n_worker:
            self._lazy_register_worker(job.type)
        if self.n_worker[job.type]:
            self.worker_config[job.type].task(self.Q_job[job.type], self.Q_result,
                                              n=self.n_worker[job.type]-self.worker_config[job.type].max_n).start()
            self.n_worker[job.type] -= 1
        self.Q_job[job.type].put(job.data)

    def poll_result(self, handler_function):
        while True:
            try:
                item = self.Q_result.get_nowait()  # non-blocking [web:25]
            except Empty:
                break
            else:
                handler_function(item)

def BatchGUI_Q():
    Q = QManager()
    Q.register_worker(JobType.TEST, testworker, 3)
    Q.register_worker(JobType.GEVIReg, GEVIReg_Worker, 5)
    Q.register_worker(JobType.PullVM, PullVM_Worker, 24)
    return Q

if __name__ == '__main__':
    freeze_support()
    try:
        set_start_method('spawn')
    except:
        pass
    Q = BatchGUI_Q()
    for i in range(5):
        Q.run_job(Job(JobType.TEST, str(i)))
