'''
Shared by the job queue manager, the per-user runners and the clients: job types, settings, user identity,
the JSON codec for job data and the HTTP helper.
'''
import os
import json
import time
import getpass
import secrets
import importlib
import urllib.request
import urllib.error
from enum import Enum
from pathlib import Path, PurePath
from dataclasses import dataclass, is_dataclass, fields

from envs.site_config import CONFIG


@dataclass(frozen=True)
class JobTypeSpec:
    handler: str  # 'module:function'; the function takes the decoded job data and returns a result string
    max_n: int  # max number of jobs of this type running at the same time, for all users together
    on_done: str | None = None  # 'module:function' the manager runs after a job finished without error (hooks.py)


# Clients can only submit these types. To add one, write a handler function and add it here.
JOB_TYPES = {
    'TEST': JobTypeSpec('JobQueue.common:echo_job', 3),
    'GEVIReg': JobTypeSpec('JobQueue.handlers:gevireg', 5),
    'PullVM': JobTypeSpec('JobQueue.handlers:pull_vm', 24),
    'SIMA': JobTypeSpec('JobQueue.handlers:sima', 2),
    'PullSignals': JobTypeSpec('JobQueue.handlers:pull_signals', 1, on_done='JobQueue.hooks:signals_pulled'),
    'ProcessROIs': JobTypeSpec('JobQueue.handlers:process_rois', 1, on_done='JobQueue.hooks:rois_processed'),
    'Ripples': JobTypeSpec('JobQueue.handlers:ripples', 3),
    'MotionEnergy': JobTypeSpec('Video.MotionEnergy:run_job', 4),
    'RecoveryMotion': JobTypeSpec('Video.RecoveryMotion:run_job', 1),
    'MotionCorrect': JobTypeSpec('Proc2P.Bruker.MotionCorrect:run_job', 3, on_done='JobQueue.hooks:motion_corrected'),
}

RUNNER_SYNC_S = 1.0  # runners report to the manager this often
RUNNER_LOST_S = 30.0  # a runner that hasn't reported for this long is considered stopped
ASSIGN_TIMEOUT_S = 30.0  # an assigned job that the runner hasn't started by then goes back to pending
WORKER_IDLE_S = 600.0  # runners stop workers that had no job for this long
RUNNER_IDLE_S = 1800.0  # runners exit after this long with no workers and no jobs


def api_port() -> int:
    return int(os.environ.get('LNCM_JOBQUEUE_API_PORT', CONFIG.job_queue['api_port']))


def dashboard_port() -> int:
    return int(os.environ.get('LNCM_JOBQUEUE_DASHBOARD_PORT', CONFIG.job_queue['dashboard_port']))


def manager_data_dir() -> Path:
    return Path(os.environ.get('LNCM_JOBQUEUE_DATA_DIR', CONFIG.job_queue['data_dir']))


def user_state_dir() -> Path:
    '''Per-user folder for the token and the runner logs, in the user's profile.'''
    d = Path(os.environ.get('LNCM_JOBQUEUE_STATE_DIR', Path.home() / '.lncm_jobqueue'))
    d.mkdir(parents=True, exist_ok=True)
    return d


def lib_root() -> Path:
    '''The LNCM-lib checkout this code runs from. Runners only run jobs submitted from the same checkout.'''
    return Path(__file__).resolve().parent.parent


def current_user() -> str:
    return getpass.getuser().lower()


def _load_user_names() -> dict:
    '''envs/user_names.json: {"<windows user name>": "<name>"}, kept out of git (personal info)'''
    fn = lib_root() / 'envs' / 'user_names.json'
    return {k.lower(): v for k, v in json.loads(fn.read_text()).items()} if fn.exists() else {}


USER_NAMES = _load_user_names()


def display_name(user: str) -> str:
    return USER_NAMES.get(user, user)


def user_token() -> str:
    '''
    Random token kept in the user's profile, which other users can't read. The manager stores the first token it
    sees for each user and refuses requests with a different one, so nobody can queue jobs that run as someone else.
    '''
    fn = user_state_dir() / 'token'
    if not fn.exists():
        fn.write_text(secrets.token_hex(16))
    return fn.read_text().strip()


def resolve(path: str):
    '''Import 'module:attribute' '''
    module, name = path.split(':')
    obj = importlib.import_module(module)
    for part in name.split('.'):
        obj = getattr(obj, part)
    return obj


def _qualname(cls) -> str:
    return f'{cls.__module__}:{cls.__qualname__}'


def encode(obj):
    '''Convert job data to JSON-compatible values. Dataclasses and Enums are rebuilt by decode.'''
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {'__dataclass__': _qualname(type(obj)),
                'fields': {f.name: encode(getattr(obj, f.name)) for f in fields(obj)}}
    if isinstance(obj, Enum):
        return {'__enum__': _qualname(type(obj)), 'name': obj.name}
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError(f'Job data dict keys must be str: {list(obj)}')
        return {k: encode(v) for k, v in obj.items()}
    if hasattr(obj, 'item') and hasattr(obj, 'dtype') and getattr(obj, 'ndim', None) == 0:  # numpy scalar
        return obj.item()
    raise TypeError(f'Job data can not contain {type(obj)}')


def decode(obj):
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    if isinstance(obj, dict):
        if '__dataclass__' in obj:
            cls = resolve(obj['__dataclass__'])
            if not is_dataclass(cls):
                raise TypeError(f'{obj["__dataclass__"]} is not a dataclass')
            values = {k: decode(v) for k, v in obj['fields'].items()}
            init_names = {f.name for f in fields(cls) if f.init}
            instance = cls(**{k: v for k, v in values.items() if k in init_names})
            for k, v in values.items():
                if k not in init_names:
                    setattr(instance, k, v)
            return instance
        if '__enum__' in obj:
            cls = resolve(obj['__enum__'])
            if not issubclass(cls, Enum):
                raise TypeError(f'{obj["__enum__"]} is not an Enum')
            return cls[obj['name']]
        return {k: decode(v) for k, v in obj.items()}
    return obj


class QueueError(Exception):
    pass


class ManagerUnavailable(QueueError):
    '''The manager is not running.'''


class ManagerRefused(QueueError):
    '''The manager is running but rejected the request (unknown job type, token mismatch).'''


def call(path: str, data=None, timeout=2.0, port=None):
    '''GET (data is None) or POST JSON to the manager's localhost API. Raises ManagerUnavailable or ManagerRefused.'''
    url = f'http://127.0.0.1:{port or api_port()}{path}'
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get('error', str(e))
        except Exception:
            msg = str(e)
        raise ManagerRefused(f'Job queue manager refused the request: {msg}') from None
    except (urllib.error.URLError, OSError) as e:
        raise ManagerUnavailable(f'Job queue manager is not reachable: {e}') from None


def echo_job(data):
    '''Handler of TEST jobs. data: [message] or [message, seconds to sleep] or [message, seconds, 'fail']'''
    if len(data) > 1:
        time.sleep(data[1])
    if len(data) > 2 and data[2] == 'fail':
        raise RuntimeError(f'TEST job failed on request: {data[0]}')
    return str(data[0])
