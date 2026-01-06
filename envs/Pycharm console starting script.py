'''Paste the block below to Settings/Python/Console/Starting script to circumvent error message from Qt backend loop
when scripts are run interactive.'''

import sys
print("Python %s on %s" % (sys.version, sys.platform))
sys.path.extend([WORKING_DIR_AND_PYTHON_PATHS])

import matplotlib
matplotlib.use("TkAgg")
