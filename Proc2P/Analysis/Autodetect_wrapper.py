'''
Because sima is not available to install on python > 3.6, future versions need to run ROI detection in a separate env.
This script does a system call to Autodetect_worker, which will run the detector in python 3.6.
'''
import os.path
import subprocess
import sys
import json
from pathlib import Path
from envs import CONFIG

PY36_PYTHON = CONFIG.python36_executable
PY36_SCRIPT = os.path.join(Path(__file__).parent,  'Autodetect_worker.py')

def _run_py36_job(path, prefix, apps, config):

    payoad = {"path": path, "prefix":prefix, "apps":apps, "config":config}

    try:
        result = subprocess.run(' '.join([os.path.realpath(x) for x in (PY36_PYTHON, PY36_SCRIPT)]),
            input=json.dumps(payoad),
            text=True,              # send/receive str instead of bytes
            capture_output=True,    # capture stdout/stderr
            check=True,             # raise CalledProcessError on non-zero return code
        )

        if result.returncode == 0:
            stdout = result.stdout.strip()
            return stdout
    except subprocess.CalledProcessError as e:
        print("COMMAND:", e.cmd, file=sys.stderr)
        print("RETURN CODE:", e.returncode, file=sys.stderr)
        print("STDOUT:", e.stdout, file=sys.stderr)
        print("STDERR:", e.stderr, file=sys.stderr)
        raise

roi_detector = _run_py36_job

if __name__ == "__main__":
    path = 'D:/Shares/Data/_Processed/2P/CCK/'
    prefix = 'Sncg146_2025-07-29_optostim_127'
    apps = ['STICA', 'iPC']
    config = {'Start': 0, 'Stop': 'end', 'Diameter': 20, 'MinSize':100, 'MaxSize':800}
    result = roi_detector(path=path, prefix=prefix, apps=apps, config=config)
    print(result)
