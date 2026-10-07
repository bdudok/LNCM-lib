# Description of installation process

# To create the main env used in most analysis GUIs and scripts ("lncm"), use Python 3.11 and stick to conda-forge channel.

conda create --name lncm python=3.11 -c conda-forge --strict-channel-priority
conda activate lncm
conda install -c conda-forge -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scikit-image xlrd statsmodels numba requests pyqtgraph pyedflib lsq-ellipse pandas-stubs pyarrow qt6-multimedia


# To use SIMA for ROI detection in 2P movies, we need a Python 3.6 env ("lncm36"), 
# Note that calls to the functions that use this are made from the main env, and the path to the python 3.6 executable needs to be specified in envs/site_config.json
# It is enough to set this env up for one user on the server, and have the config point to that executable for all users.

conda create --name lncm36 python=3.6 -c conda-forge --strict-channel-priority
conda activate lncm36
conda install -c conda-forge -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scikit-image xlrd statsmodels numba requests pyqtgraph
pip install sima

# To use Suite2P for motion correction, we need a Python 3.11 env ("suite2p") with suite2p 1.x (tested with 1.1.0):
if you have a previous suite2p env, delete it (conda env remove --name suite2p),
and delete the remaining empty folder in envs called suite2p
note: suite2p 0.x envs don't work with the current pipeline (MotionCorrect and the 20261007 template use the 1.x settings format).
in conda powershell prompt:
    conda create --name suite2p python=3.11 -c conda-forge --strict-channel-priority
    conda activate suite2p
    python -m pip install suite2p
    python -m pip install pandas requests openpyxl pyarrow
our analysis server has no GPU, suite2p runs on CPU with the default torch install (it logs that cuda is not working and uses cpu).

open the job script in your project (no need to have a separate S2P project)
for example, Proc2P/Analysis/S2P_template/20261007_BatchRun_MotionCorrect_Template.py (in LNCM_ClaudeCode)
In settings/Python/interpreter, add the suite2p env as system python (but don't select it for the project)
Select run/edit configurations for your job script
    select the suite2p interpreter to run this script with
run the script with this specific configuration.

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