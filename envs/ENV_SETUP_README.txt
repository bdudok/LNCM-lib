# Description of installation process

# The main env (lncm, including suite2p for motion correction) is built with uv: see installation.md in the repo root.
# This file describes envs for other tools.

# To use SIMA for ROI detection in 2P movies, we need a Python 3.6 env ("lncm36"), 
# Note that calls to the functions that use this are made from the main env, and the path to the python 3.6 executable needs to be specified in envs/site_config.json
# It is enough to set this env up for one user on the server, and have the config point to that executable for all users.

conda create --name lncm36 python=3.6 -c conda-forge --strict-channel-priority
conda activate lncm36
conda install -c conda-forge -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scikit-image xlrd statsmodels numba requests pyqtgraph
pip install sima
# envs/2pcode36.yml is an export of a working env of this kind.

# For deeplabcut, we need a separate env (on GPU system with NVIDIA CUDA installed):
-clone DLC repo
-cd to repo root
conda env create -f DEEPLABCUT.yaml	
conda activate DEEPLABCUT
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
python -m deeplabcut

# For treadmill (PyControl_LNCM)
the lncm env does not work for this (gui written for an older pyqtgraph binding),
 use the default PyControl installation and add zmq if necessary