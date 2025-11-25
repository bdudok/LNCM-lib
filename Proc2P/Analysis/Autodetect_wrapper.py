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

def run_py36_job(path, prefix, apps, config):

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

if __name__ == "__main__":
    result = run_py36_job(path='test', prefix='test-prefix', apps=['a', 'b'], config={"good": True})
    print(result)
