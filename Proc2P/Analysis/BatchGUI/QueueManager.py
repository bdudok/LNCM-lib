from Proc2P.Analysis.BatchGUI.Config import *
from dataclasses import dataclass
from collections.abc import Callable
from multiprocessing import Queue, Process, freeze_support, set_start_method
from queue import Empty
#import worker functions
from Proc2P.Analysis.GEVIReg.Register import Worker as GEVIReg_Worker
from Proc2P.Analysis.AnalysisClasses.NormalizeVm import Worker as PullVM_Worker
from Proc2P.Analysis.RoiEditor import Worker as SIMA_Worker
from Proc2P.Analysis.PullSignals import Worker as Pull_Worker
from Proc2P.Analysis.CaTrace import CaTrace, ProcessConfig
from Proc2P.Analysis.CaTrace import Worker as Process_Cell_Worker

class JobType(Enum):
    TEST = 0
    GEVIReg = 1
    PullVM = 2
    SIMA = 3
    PullSignals = 4
    ProcessROIs = 5

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

class ProcessSessionWorker(Process):
    __name__ = 'Process-Session-Worker'

    def __init__(self, queue, res_queue, n=0, worker_number=24):
        super(ProcessSessionWorker, self).__init__()
        self.session_queue = queue
        self.session_res_queue = res_queue
        self.queue = Queue()
        self.result_queue = Queue()
        self.ncpu = worker_number
        self.nworker = 0
        self.n = n

    def run(self):
        '''This process spawns and starts workers for single-cell traces to process a session.
        Run is called with a config dataclass, and this worker parses it into CaTrace args'''
        for data in iter(self.session_queue.get, None):
            path, prefix, tag, config = data
            config = ProcessConfig
            #parse args
            if config.polynom_baseline:
                bsltype = 'poly'
            else:
                bsltype = 'original'
            peakdet = False
            last_bg =config.last_ROI_is_background
            invert = [config.invert_first_channel_G, config.invert_second_channel_R]
            excl = (config.exclude_from_frame, config.exclude_until_frame)
            sz_mode = config.seizure_mode

            #process cells
            for ch in (0, 1):
                session = CaTrace(path, prefix, bsltype=bsltype, exclude=excl, peakdet=peakdet, ch=ch, tag=tag,
                            invert=invert[ch], last_bg=last_bg, ignore_saturation=sz_mode)
                if session.open_raw() == -1:
                    continue
                if os.path.exists(session.pf):
                    print(f'{session.pf} folder exists, skipping')
                    continue
                print('Processing', session.pf)
                if sz_mode:
                    session.ol_index = []
                for c in range(session.cells):
                    if self.ncpu and c >= self.nworker:
                        Process_Cell_Worker(self.queue, self.result_queue).start()
                        self.nworker += 1
                        self.ncpu -= 1
                    self.queue.put(session.pack_data(c))
                for cell_data in iter(self.result_queue.get, None):
                    finished = session.unpack_data(cell_data)
                    if finished:
                        break

            self.session_res_queue.put((self.__name__, prefix))

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
                item = self.Q_result.get_nowait()  # non-blocking
            except Empty:
                break
            else:
                handler_function(item)

def BatchGUI_Q():
    Q = QManager()
    Q.register_worker(JobType.TEST, testworker, 3)
    Q.register_worker(JobType.GEVIReg, GEVIReg_Worker, 5)
    Q.register_worker(JobType.PullVM, PullVM_Worker, 24)
    Q.register_worker(JobType.SIMA, SIMA_Worker, 2)
    Q.register_worker(JobType.PullSignals, Pull_Worker, 1)
    Q.register_worker(JobType.ProcessROIs, ProcessSessionWorker, 1)
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
