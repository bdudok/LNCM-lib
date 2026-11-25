import sys
import warnings
from pathlib import Path

from envs.site_config import CONFIG

'''Warn when the lncm env is older than the repo's uv.lock.
envs/update_env.bat copies uv.lock into the env after a successful sync; the env is out of date if that copy is
missing or differs from the repo's uv.lock. Only envs under uv_env_dir (site_config.json) are checked, so envs built
another way (e.g. in a container) are left alone.'''

REPO_LOCK = Path(__file__).resolve().parents[1] / 'uv.lock'
SYNCED_LOCK = 'lncm_synced_uv.lock'  # name of the copy in the env folder


def env_is_current(env, env_dir, repo_lock=REPO_LOCK):
    '''False if env is under env_dir and was not synced from the current repo_lock; True otherwise.'''
    env, env_dir = Path(env), Path(env_dir)
    # not Path.is_relative_to (Python 3.9+): the SIMA worker imports this in Python 3.6
    if not (env == env_dir or env_dir in env.parents) or not repo_lock.is_file():
        return True
    synced = env / SYNCED_LOCK
    return synced.is_file() and synced.read_bytes() == repo_lock.read_bytes()


def check_env():
    if not env_is_current(sys.prefix, CONFIG.uv_env_dir):
        warnings.warn(f'The Python env ({sys.prefix}) is not up to date with LNCM-lib: run '
                      f'{REPO_LOCK.parent / "envs" / "update_env.bat"}', stacklevel=2)
