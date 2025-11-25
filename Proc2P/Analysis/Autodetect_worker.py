'''
Because sima is not available to install on python > 3.6, future versions need to run ROI detection in a separate env.
This script should be called by a system call to the python 3.6, and will run the detector there.
That call is made by Autodetect_wrapper
'''

from Proc2P.utils import lprint, logger
from RoiEditor import RoiEditor
import sys
import json

def main():

    job = json.loads(sys.stdin.read())

    # Call your existing 3.6-only function here.
    # result = real_processing_function(args.input, args.output, args.option_flag)
    path = job["path"] #string to processed path
    prefix = job["prefix"]
    apps = job["apps"] # segmentation approaches to run. a list converted to string with json.dumps
    config = job["config"]


    # log = logger()
    # log.set_handle(path, prefix)
    # lprint(None, 'Calling autodetect with:', (path, prefix, apps, config), logger=log)
    # RoiEditor(path, prefix, ).autodetect(approach=apps, config=config, log=log)

    print(json.dumps({
        "status": "ok",
        # "env": sys.version_info,
        "processed_input": prefix,
    }))


if __name__ == "__main__":
    print('hello')
    # main()

