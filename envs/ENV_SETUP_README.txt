# Description of installation process

# To create the main env used in most analysis GUIs and scripts ("lncm"), use Python 3.11 and stick to conda-forge channel.

conda create --name lncm python=3.11 -c conda-forge -c defaults --strict-channel-priority
conda activate lncm
conda install -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scikit-image xlrd statsmodels numba requests pyqtgraph pyedflib lsq-ellipse pandas-stubs pyarrow qt6-multimedia


# To use SIMA for ROI detection in 2P movies, we need a Python 3.6 env ("lncm36"), 
# Note that calls to the functions that use this are made from the main env, and the path to the python 3.6 executable needs to be specified in envs/site_config.json

conda create --name lncm36 python=3.6 -c conda-forge -c defaults --strict-channel-priority
conda activate lncm36
conda install -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scikit-image xlrd statsmodels numba requests pyqtgraph
pip install sima

# To use Suite2P for motion correction, we need a Python 3.9 env ("suite2p"):
if you have a previous suite2p env, delete it (conda env remove --name suite2p),
and delete the remaining empty folder in envs called suite2p
in conda powershell prompt:
    conda create --name suite2p python=3.9 -c defaults --strict-channel-priority --no-default-packages
    conda activate suite2p
    pip install suite2p
    pip install pandas, requests

open the job script in your project (no need to have a separate S2P project)
for example, Proc2P/Analysis/S2P_template/20260113_BatchRun_MotionCorrect_Template.py (in LNCM-lib)
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

